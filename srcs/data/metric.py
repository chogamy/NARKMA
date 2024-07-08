from typing import Any
from torchmetrics import Metric


# https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
class sejong(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.preds = {}
        self.targets = {}

    def update(self, keys, preds, targets):
        for key, pred, target in zip(keys, preds, targets):
            if key in self.preds:
                self.preds[key].append(pred)
                self.targets[key].append(target)
            else:
                self.preds[key] = [pred]
                self.targets[key] = [target]

    def compute(self):
        f1s = []
        accs = []
        assert self.preds.keys() == self.targets.keys()
        for key in self.preds.keys():
            p = " ".join(self.preds[key])
            t = " ".join(self.targets[key])

            f1s.append(self.f1(p, t))
            accs.append(self.acc(p, t))

        f1s = sum(f1s) / len(f1s)
        accs = sum(accs) / len(accs)

        return {"f1": f1s, "acc": accs}

    def acc(self, p, t):
        target = set()
        eojs = t.split(" ")
        for eoj in eojs:
            i = 0
            while (eoj, i) in target:
                i += 1
            target.add((eoj, i))

        preds = set()
        eojs = p.split(" ")
        for eoj in eojs:
            i = 0
            while (eoj, i) in preds:
                i += 1
            preds.add((eoj, i))

        return len(preds & target) / len(target)

    def f1(self, p, t):
        target = set()
        t = t.replace("+", " ")
        eojs = t.split(" ")
        for eoj in eojs:
            i = 0
            while (eoj, i) in target:
                i += 1
            target.add((eoj, i))

        preds = set()
        p = p.replace("+", " ")
        eojs = p.split(" ")
        for eoj in eojs:
            i = 0
            while (eoj, i) in preds:
                i += 1
            preds.add((eoj, i))

        p = len(preds & target) / len(target)
        r = len(preds & target) / len(preds)

        if p + r == 0:
            return 0
        else:
            return 2 * p * r / (p + r)


METRIC = {"sejong": sejong}
