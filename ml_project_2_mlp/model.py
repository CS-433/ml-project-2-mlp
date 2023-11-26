"""
Module containing the `LightningModule` for the Homepage2Vec model.
"""

import os
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MultilabelAccuracy as Accuracy

from .homepage2vec.model import SimpleClassifier, WebsiteClassifier


class Homepage2VecModule(LightningModule):
    """`LightningModule` for fine-tuning Homepage2Vec."""

    def __init__(
        self,
        model_path: str,
        device: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        """Initialize a `Homepage2VecLitModule`.

        Args:
            model_path: The path to the directory containing the pre-trained model
            device: The device to use for training
            optimizer: The optimiser to use for training
            scheduler: The learning rate scheduler to use for training
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Load website classifier (feature extractor)
        self.website_clf = WebsiteClassifier(model_path=model_path)  # Feature extractor
        self.input_dim = self.website_clf.input_dim
        self.output_dim = self.website_clf.output_dim

        # Load pre-trained model (classification head)
        weight_path = os.path.join(model_path, "model.pt")
        model_tensor = torch.load(weight_path, map_location=device)
        self.model = SimpleClassifier(
            input_dim=self.website_clf.input_dim, output_dim=self.website_clf.output_dim
        )
        self.model.load_state_dict(model_tensor)

        # loss function
        self.criterion = torch.nn.BCELoss()

        # metric objects for calculating and averaging accuracy across batches
        threshold = 0.5  # TODO: move this to hydra

        self.train_acc = Accuracy(
            threshold=threshold,
            num_labels=self.output_dim,
            average="micro",
            validate_args=True,
        )
        self.val_acc = Accuracy(
            threshold=threshold,
            num_labels=self.output_dim,
            average="micro",
            validate_args=True,
        )
        self.test_acc = Accuracy(
            threshold=threshold,
            num_labels=self.output_dim,
            average="micro",
            validate_args=True,
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the model `self.net`.

        Args:
            x: A batch of homepage embeddings

        Returns:
            A tensor of logits
        """
        return self.model(x)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of
            homepage embeddings and one-hot encoded target labels.

        Returns:
        A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits, _ = self.forward(x)  # Returns raw logits and embeddings
        loss = self.criterion(torch.sigmoid(logits), y)

        return loss, logits, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], _: int
    ) -> torch.Tensor:
        """
        Perform a single training step on a batch of data from the training set.

        Args:
            batch: A batch of data (tuple) containing the input tensor of images and target labels
            batch_idx: The index of the current batch

        Returns:
            A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # Update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        # Return loss or backpropagation will fail
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _: int) -> None:
        """
        Perform a single validation step on a batch of data from the validation set.

        Args:
            batch: A batch of data (tuple) containing the input tensor of images and target labels
            batch_idx: The index of the current batch

        Returns:
            None
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Lightning hook that is called when a validation epoch ends.
        """
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """
        Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.


        Returns:
            A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
