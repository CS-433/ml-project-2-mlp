"""
Module where we define any metrics we want to use in our project.
"""

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torchmetrics import Accuracy, F1Score, Metric, Precision, Recall


class LabelsPerPage(Metric):
    def __init__(self, **kwargs):
        super().__init__()
        self.add_state("total_labels", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_pages", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        total_labels = torch.sum(preds > 0.5)
        self.total_labels += total_labels
        self.total_pages += preds.size(0)

    def compute(self):
        return self.total_labels.float() / self.total_pages.float()


def validation_report(prediction, actual, categories):
    """
    Generate a validation report from the predictions and the actual labels

    Args:
        prediction: List of predictions
        actual: List of actual labels

    Returns:
        A classification report
    """
    # Get df from a list of dicts
    actual, categories_df = _allign_dataframes(prediction, actual, categories)
    report = classification_report(
        actual[categories], categories_df, target_names=categories
    )
    return report


def _allign_dataframes(predictions, ground_truth, categories):
    # From the list of dicts, get a df
    predictions_df = pd.DataFrame(predictions)
    predictions_df.set_index("wid", inplace=True)

    # Select only valid predictions
    predictions_df = predictions_df[predictions_df["is_valid"]]

    # From the valid predictions, create a df with the categories and as index website id
    categories_df = pd.DataFrame(
        predictions_df["output"].tolist(),
        index=predictions_df.index,
        columns=categories,
        dtype=int,
    )

    # Assert that the index of ground truth are in categories_df
    assert categories_df.index.isin(
        ground_truth.index
    ).all(), "The index of ground truth is not in categories_df!"

    # Select corresponding ground truth labels
    ground_truth = ground_truth.loc[categories_df.index]

    return ground_truth, categories_df


def compute_metrics(
    actual_labels: pd.DataFrame,
    predictions: list[dict],
    categories: list[str],
    avg="macro",
):
    actual, categories_df = _allign_dataframes(predictions, actual_labels, categories)
    y_true = torch.tensor(actual[categories].values)
    y_pred = torch.tensor(categories_df.values)

    accuracy = Accuracy(task="multilabel", num_labels=len(categories))
    precision = Precision(task="multilabel", num_labels=len(categories), average=avg)
    recall = Recall(task="multilabel", num_labels=len(categories), average=avg)
    f1 = F1Score(task="multilabel", num_labels=len(categories), average=avg)

    # Compute metrics
    metrics = {
        "accuracy": accuracy(y_pred, y_true).item(),
        "precision": precision(y_pred, y_true).item(),
        "recall": recall(y_pred, y_true).item(),
        "f1": f1(y_pred, y_true).item(),
    }
    return metrics
