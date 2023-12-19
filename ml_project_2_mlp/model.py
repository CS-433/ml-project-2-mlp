"""
Module containing the `LightningModule` for the Homepage2Vec model.
"""

import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from lightning import LightningModule
from sklearn.metrics import classification_report
from torchmetrics import ConfusionMatrix, MaxMetric, MeanMetric, Metric
from torchmetrics.classification import MultilabelAccuracy as Accuracy
from torchmetrics.classification import MultilabelF1Score as F1
from torchmetrics.classification import MultilabelPrecision as Precision
from torchmetrics.classification import MultilabelRecall as Recall

from .homepage2vec.model import SimpleClassifier
from .metrics import LabelsPerPage


class Homepage2VecModule(LightningModule):
    """`LightningModule` for fine-tuning Homepage2Vec."""

    def __init__(
        self,
        model_dir: str,
        device: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        threshold: float,
        pos_ratio: list[float],
        calibrated: bool = True,
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

        self.input_dim = 4665
        self.output_dim = 14

        # Load pre-trained model (classification head)
        weight_path = os.path.join(self.hparams.model_dir, "model.pt")
        model_tensor = torch.load(weight_path, map_location=self.hparams.device)
        self.model = SimpleClassifier(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        self.model.load_state_dict(model_tensor)

        # Loss function
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.hparams.pos_ratio)

        # Setup metric kwargs
        clf_metrics_kwargs = {
            "num_labels": self.output_dim,
            "threshold": self.hparams.threshold,
            "average": "macro",
        }

        # Training metrics
        self.train_metrics = [
            ("f1", F1, clf_metrics_kwargs),
            ("acc", Accuracy, clf_metrics_kwargs),
            ("loss", MeanMetric, None),
        ]
        self._set_metrics(self.train_metrics, "train")

        # Validation metrics
        self.val_metrics = [
            ("f1", F1, clf_metrics_kwargs),
            ("acc", Accuracy, clf_metrics_kwargs),
            ("loss", MeanMetric, None),
        ]

        self._set_metrics(self.val_metrics, "val")
        # Tracks the first metric in `self.val_metrics`
        self.val_metric_best = MaxMetric()

        # Testing metrics
        self.test_metrics = [
            ("f1", F1, clf_metrics_kwargs),
            ("acc", Accuracy, clf_metrics_kwargs),
            ("precision", Precision, clf_metrics_kwargs),
            ("recall", Recall, clf_metrics_kwargs),
            ("lpp", LabelsPerPage, None),
            ("loss", MeanMetric, None),
        ]

        # Confusion matrix
        cm_kwargs = {
            "num_labels": self.output_dim,
            "task": "multilabel",
            "threshold": self.hparams.threshold,
        }
        self.test_cm = ConfusionMatrix(**cm_kwargs)
        self.test_probs = []
        self.test_preds = []
        self.test_targets = []

        self._set_metrics(self.test_metrics, "test")

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
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # Parse batch
        x, y = batch

        # Returns raw logits and embeddings
        logits, _ = self.forward(x)

        # logits = torch.sigmoid(logits)
        # if self.hparams.calibrated:
        #     pos_ratio = torch.tensor(self.hparams.pos_ratio)
        #     logits = logits / (logits + pos_ratio * (1 - logits))

        # Cast labels from floats to longs for computing metrics
        y = y.long()

        return logits, y

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

        # Forward pass
        logits, targets = self.model_step(batch)

        # Update and log metrics
        loss = self._update_log_metrics(self.train_metrics, logits, targets, "train")

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
        # Forward pass
        logits, targets = self.model_step(batch)

        # Update and log metrics
        self._update_log_metrics(self.val_metrics, logits, targets, "val")

    def on_validation_epoch_end(self) -> None:
        """
        Lightning hook that is called when a validation epoch ends.
        """
        metric_name = self.val_metrics[0][0]
        metric_value = getattr(self, f"val_{metric_name}").compute()
        self.val_metric_best(metric_value)
        self.log(
            f"val/{metric_name}_best",
            self.val_metric_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        # Forward pass
        logits, targets = self.model_step(batch)

        # Update and log metrics
        self._update_log_metrics(self.test_metrics, logits, targets, "test")

        # Accumulate predictions and targets for computing metrics at the end of the epoch
        probs = torch.sigmoid(logits)
        preds = (probs > self.hparams.threshold).int()
        self.test_probs.append(probs)
        self.test_preds.append(preds)
        self.test_targets.append(targets)

    def on_test_epoch_end(self) -> None:
        """ """
        probs = torch.cat(self.test_probs)
        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)

        test_report = classification_report(
            preds,
            targets,
            output_dict=True,
        )
        test_report_df = (
            pd.DataFrame(test_report)
            .T.reset_index()
            .rename({"index": "category"}, axis=1)
        )

        # Compute confusion matrix
        test_cms = self.test_cm(preds, targets)
        rows = []
        for i, test_cm in enumerate(test_cms):
            test_cm = test_cm.cpu().numpy()
            tn, fp, tp, fn = test_cm[0, 0], test_cm[0, 1], test_cm[1, 0], test_cm[1, 1]
            rows.append({"category": i, "tn": tn, "fp": fp, "tp": tp, "fn": fn})
        test_cm_df = pd.DataFrame(rows)

        for logger in self.loggers:
            logger.experiment.summary["test/report"] = json.dumps(
                test_report_df.to_dict()
            )
            logger.experiment.summary["test/cm"] = json.dumps(test_cm_df.to_dict())
            logger.experiment.summary["test/probs"] = json.dumps(probs.tolist())
            logger.experiment.summary["test/preds"] = json.dumps(preds.tolist())
            logger.experiment.summary["test/targets"] = json.dumps(targets.tolist())

        self.test_preds, self.test_targets = [], []

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

    def _set_metrics(self, metrics: List[Tuple[str, Metric, str]], mode: str) -> None:
        """
        Utility function to set metrics as attributes of the model.

        Args:
            metrics: A list of tuples containing the name of the metric, the metric class and
                the keyword arguments to pass to the metric class constructor.
            mode: The mode for which to set the metrics. Must be one of "train", "val" or "test".
        """
        for name, metric_class, kwargs in metrics:
            if kwargs is None:
                setattr(self, f"{mode}_{name}", metric_class())
            else:
                setattr(self, f"{mode}_{name}", metric_class(**kwargs))

    def _update_log_metrics(
        self,
        metrics: List[Tuple[str, Metric, str]],
        logits: torch.Tensor,
        targets: torch.Tensor,
        mode: str,
    ):
        """
        Utility function to update and log metrics for a given mode.
        """
        loss = None
        for name, _, _ in metrics:
            # Update
            if name != "loss":
                getattr(self, f"{mode}_{name}")(logits, targets)
            else:
                loss = self.criterion(logits, targets.float())
                getattr(self, f"{mode}_{name}")(loss)

            # Log
            self.log(
                f"{mode}/{name}",
                getattr(self, f"{mode}_{name}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss
