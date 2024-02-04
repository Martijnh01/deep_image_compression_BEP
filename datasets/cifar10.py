from typing import Union

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

from datasets.custom_lightning_data_module import CustomLightningDataModule


class CIFAR10DataModule(CustomLightningDataModule):
    def __init__(
            self,
            root: str,
            img_size: int,
            devices,
            batch_size: int,
            num_workers: int,
    ) -> None:
        super().__init__(
            root=root,
            devices=devices,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.val_batch_size = 512

    def setup(self, stage: Union[str, None] = None) -> CustomLightningDataModule:
        DATASET_PATH = "./"
        self.cifar10_train = CIFAR10(root=DATASET_PATH, train=True, transform=self.transform, download=True)
        self.cifar10_val = CIFAR10(root=DATASET_PATH, train=False, transform=self.transform, download=True)
        return self

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar10_train,
            shuffle=True,
            drop_last=True,
            persistent_workers=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                self.cifar10_val,
                persistent_workers=False,
                num_workers=2,
                pin_memory=self.pin_memory,
                batch_size=self.val_batch_size,
            ),
        ]
