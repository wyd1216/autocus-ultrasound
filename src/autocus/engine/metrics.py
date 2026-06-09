from __future__ import annotations

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)
    return float((2 * np.logical_and(p, t).sum() + eps) / (p.sum() + t.sum() + eps))


def iou_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)
    return float((np.logical_and(p, t).sum() + eps) / (np.logical_or(p, t).sum() + eps))
