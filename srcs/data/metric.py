from typing import Any
from torchmetrics import Metric


# https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
class sejong(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.preds = []
        self.target = []

    def update(self, preds, target):
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        pass

    def acc(self):
        pass

    def f1(self):
        pass


METRIC = {"sejong": sejong}
