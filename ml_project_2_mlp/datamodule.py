import os
from typing import List, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from .utils import download_if_not_present, load


class TorchDataset(Dataset):
    """
    Simple PyTorch dataset that expected a feature and label tensor
    as input and implements __len__ and __getitem__.
    """

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        return self.embeddings.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


class WebsiteDataModule(LightningDataModule):
    """
    A `PyTorch Lightning DataModule` for the crowd-sourced data which includes
    functionality for downloading the data, splitting it into train, validation
    and test sets, and loading it into `DataLoader`s.
    """

    def __init__(
        self,
        data_dir: str,
        urls: dict[str, str],
        data_split: List[float] = [0.6, 0.2, 0.2],
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int | None = None,
    ):
        super().__init__()

        # Set the hyperparameters
        self.save_hyperparameters(logger=False)

        # Transforms (to float tensor)
        self.transforms = transforms.Compose([transforms.Lambda(lambda x: x.float())])

        # Data attributes
        self.embeddings = None
        self.labels = None

    def prepare_data(self):
        """
        Fetch the websites and embedded them.
        """
        # Load the embedded websites (download if necessary)
        download_if_not_present(
            dir_path=os.path.join(self.hparams.data_dir, "embedded"),
            gdrive_url=self.hparams.urls["embedded"],
            expected_files=["embeddings.pt", "labels.pt"],
        )

    def setup(self, stage: str | None = None):
        # Load the embeddings and labels
        self.embeddings, self.labels = load(
            dir_path=os.path.join(self.hparams.data_dir, "embedded"),
            expected_files=["embeddings.pt", "labels.pt"],
        )

        # Convert to float tensor
        self.embeddings = self.transforms(self.embeddings)
        self.labels = self.transforms(self.labels)

        # Define Custom Dataset
        dataset = TorchDataset(self.embeddings, self.labels)

        # Compute the sizes of the splits
        train_size = int(self.hparams.data_split[0] * len(dataset))
        val_size = int(self.hparams.data_split[1] * len(dataset))
        test_size = len(dataset) - train_size - val_size

        # Split the dataset
        generator = torch.Generator().manual_seed(self.hparams.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )
