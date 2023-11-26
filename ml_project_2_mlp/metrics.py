import torch
from torchmetrics import Metric


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
