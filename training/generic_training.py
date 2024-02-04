import io
import os
from typing import Tuple

import lightning
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torch.optim.lr_scheduler import PolynomialLR
from torchmetrics.regression import MeanSquaredError

from utils.inverse_norm import inverse_norm


class GenericTraining(lightning.LightningModule):
    def __init__(
            self,
            batch_size: int,
            network: nn.Module,
            lr: float = 1e-3,
            weight_decay: float = 0.01,
            poly_lr_decay_power: float = 0.9,
    ):
        super().__init__()
        self.job_id = os.environ.get('SLURM_JOB_ID')
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.poly_lr_decay_power = poly_lr_decay_power
        self.val_ds_names = ["cifar10"]

        self.save_hyperparameters()

        self.network = network

        self.metrics = nn.ModuleList(
            [MeanSquaredError() for _ in range(len(self.val_ds_names))]
        )

    def training_step(
            self,
            batch: [torch.Tensor, torch.Tensor],
            batch_idx: int,
    ):
        in_img, _ = batch

        out_img = self.network(in_img)
        loss = F.mse_loss(out_img, in_img)

        self.log("loss", loss.detach(), prog_bar=True)
        return loss

    @torch.no_grad()
    def _log_pred(self, in_img, out_img, dataloader_idx, pred_idx, log_prefix):
        def handle_tensor_input(x):
            x = inverse_norm(x[0])
            return x.float().cpu().permute(1, 2, 0).numpy()

        cols = 2
        rows = 1
        fig, axes = plt.subplots(
            rows, cols, figsize=(int(in_img.shape[1] * cols), in_img.shape[0] * rows)
        )

        axes[0].imshow(handle_tensor_input(in_img))
        axes[0].axis("off")
        axes[1].imshow(handle_tensor_input(out_img))
        axes[1].axis("off")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(
            buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="black"
        )
        plt.close(fig)

        buf.seek(0)
        concatenated_image = Image.open(buf)
        ds_name = self.val_ds_names[dataloader_idx]
        self.trainer.logger.experiment.log(  # type: ignore
            {
                f"{log_prefix}_{ds_name}_pred_{pred_idx}": [
                    wandb.Image(concatenated_image)
                ]
            }
        )

    def eval_step(
            self,
            batch: [torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int,
            log_prefix: str,
    ):
        in_img, _ = batch

        with torch.no_grad():
            out_img = self.network(in_img)

        self.metrics[dataloader_idx].update(out_img, in_img)

        if batch_idx == 0:
            pred_idx = batch_idx
            self._log_pred(
                in_img,
                out_img,
                dataloader_idx,
                pred_idx,
                log_prefix,
            )

    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0,
    ):
        return self.eval_step(batch, batch_idx, dataloader_idx, "val")

    def test_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0,
    ):
        return self.eval_step(batch, batch_idx, dataloader_idx, "test")

    def _on_eval_epoch_end(self, log_prefix):
        eval_per_dataset = []
        for metric_idx, metric in enumerate(self.metrics):
            eval_per_dataset.append(metric.compute())
            metric.reset()
            ds_name = self.val_ds_names[metric_idx]

            self.log(
                f"{log_prefix}_{ds_name}_mse", eval_per_dataset[-1], sync_dist=True
            )

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_eval_epoch_end("test")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f"learning_rate/group_{i}", param_group["lr"], on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), weight_decay=self.weight_decay)

        lr_scheduler = {
            "scheduler": PolynomialLR(
                optimizer,
                int(self.trainer.estimated_stepping_batches),
                self.poly_lr_decay_power,
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
