"""Quadratic weighted kappa with a fixed 1-6 label set."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score

LABELS = [1, 2, 3, 4, 5, 6]


def as_score(y_pred: np.ndarray) -> np.ndarray:
    """Round continuous predictions and clip them onto the 1-6 rubric."""
    pred = np.asarray(y_pred)
    if not np.issubdtype(pred.dtype, np.integer):
        pred = np.rint(pred)
    return np.clip(pred, 1, 6).astype(int)


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    """Agreement between human scores and predictions.

    Labels are fixed to 1-6 so a fold that happens to miss a rare score
    (especially 6) does not shrink the weight matrix.
    """
    true = np.asarray(y_true, dtype=int)
    pred = as_score(y_pred)
    return float(cohen_kappa_score(true, pred, labels=LABELS, weights="quadratic"))
