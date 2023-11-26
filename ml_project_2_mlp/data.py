import os

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from .homepage2vec.model import WebsiteClassifier


class CrowdSourcedDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class CrowdSourcedDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir, data_split=[0.6, 0.2, 0.2], batch_size=32, num_workers=0
    ):
        super().__init__()

        # Set the hyperparameters
        self.save_hyperparameters(logger=False)

        # Data attributes
        self.embeddings = None
        self.labels = None

    def prepare_data(self):
        """
        Fetch the websites and embedded them.
        """

        # Load the csv file with the websites urls, id and one-hot encoded labels
        websites = pd.read_csv(os.path.join(self.hparams.data_path, "websites.csv"))

        # Make sure embeddings exist
        if not os.path.exists(os.path.join(self.hparams.data_path, "embeddings.pt")):
            # Load the model
            model = WebsiteClassifier()

            # Fetch the website and then embed them
            embeddings = []
            for url in tqdm(websites["Input.url"], desc="Embedding websites"):
                # Fetch the website
                website = model.fetch_website(url)

                # Obtain the features
                website.features = model.get_features(
                    website.url, website.html, website.screenshot_path
                )

                # Aggregate the features
                all_features = self.concatenate_features(website)

                # Turn into a tensor with gradient tracking off
                embedding = torch.FloatTensor(all_features, requires_grad=False)

                # Add to the list of embeddings
                embeddings.append(embedding)

            # Put the embeddings in torch tensor
            embeddings = torch.stack(embeddings)

            # Save the embeddings
            torch.save(
                embeddings, os.path.join(self.hparams.data_path, "embeddings.pt")
            )

            # Get the labels and save them
            labels = torch.LongTensor(websites.iloc[:, -14:].values)
            torch.save(labels, os.path.join(self.hparams.data_path, "labels.pt"))

    def setup(self):
        # Load the embeddings and labels
        self.embeddings = torch.load(
            os.path.join(self.hparams.data_path, "embeddings.pt")
        )
        self.labels = torch.load(os.path.join(self.hparams.data_path, "labels.pt"))

        # Define Custom Dataset
        dataset = CrowdSourcedDataset(self.embeddings, self.labels)

        # Compute the sizes of the splits
        train_size = int(self.hparams.data_split[0] * len(dataset))
        val_size = int(self.hparams.data_split[1] * len(dataset))
        test_size = len(dataset) - train_size - val_size

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
