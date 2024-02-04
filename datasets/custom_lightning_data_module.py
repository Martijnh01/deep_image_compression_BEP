import lightning
from lightning.fabric.utilities.device_parser import _parse_gpu_ids


class CustomLightningDataModule(lightning.LightningDataModule):
    def __init__(
            self,
            root,
            devices,
            batch_size: int,
            num_workers: int,
            pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.root = root
        self.devices = devices
        if devices != "auto":
            self.devices = _parse_gpu_ids(devices, include_cuda=True)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
