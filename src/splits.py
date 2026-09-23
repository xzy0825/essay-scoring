"""Outer stratified folds, with an inner slice used only for early stopping."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

HOLDOUT_FOLD = 0


def outer_folds(y, n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(y).astype(int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(labels)), labels))


def inner_fit_eval(indices: np.ndarray, y, seed: int = 42, eval_size: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Split one outer-training fold. The eval slice never contains the outer validation rows."""
    indices = np.asarray(indices)
    labels = np.asarray(y)[indices]
    fit_idx, eval_idx = train_test_split(
        indices,
        test_size=eval_size,
        stratify=labels,
        random_state=seed,
    )
    return fit_idx, eval_idx
