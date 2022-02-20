import numpy as np


class DSC:
    """
    Dice Similarity Coefficient
    """
    def __init__(self, name='DSC', alpha=1e-6) -> None:
        super().__init__()
        self.name = name
        self.alpha = alpha

    def __call__(self, pred, target):
        pred = (pred != 0).astype(np.float32)
        target = (target != 0).astype(np.float32)

        inter = np.sum((target + pred) == 2)
        return float(2 * inter) / (float(np.sum(pred) + np.sum(target)) + self.alpha)


class IOU:

    def __init__(self, name='IOU', alpha=1e-6) -> None:
        super().__init__()
        self.name = name
        self.alpha = alpha

    def __call__(self, pred, target):
        pred = (pred != 0).astype(np.float32)
        target = (target != 0).astype(np.float32)
        TP = (pred + target == 2).astype(np.float32)
        FP = (pred + (1 - target) == 2).astype(np.float32)
        FN = ((1 - pred) + target == 2).astype(np.float32)
        return float(np.sum(TP)) / (float(np.sum(TP + FP + FN)) + self.alpha)
