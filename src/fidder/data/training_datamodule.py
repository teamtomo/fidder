from copy import deepcopy
from os import PathLike
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .training_dataset import FidderDataset


class FidderDataModule(pl.LightningDataModule):
    """Fiducial segmentation data module.

    https://zenodo.org/record/7104305
    """

    def __init__(self, dataset_directory: PathLike):
        super().__init__()
        self.dataset_directory = Path(dataset_directory)

    def prepare_data(self) -> None:
        FidderDataset(self.dataset_directory, train=True, download=True)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            full_dataset = FidderDataset(self.dataset_directory)
            n_total = len(full_dataset)
            n_train = int(0.7 * n_total)
            n_val = n_total - n_train
            self.train, self.val = random_split(full_dataset, [n_train, n_val])
            self.val = deepcopy(self.val)
            self.train.dataset.train()
            self.val.dataset.eval()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=1, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=4, num_workers=0)
