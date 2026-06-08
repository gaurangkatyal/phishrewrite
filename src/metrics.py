"""Shared classification metrics.

Used by both detectors.py (cross-validation) and evaluate.py (clean baseline +
degradation) so every metric is computed identically everywhere.

Under class imbalance (this project is ~1:4.44 phish:ham) PR-AUC / average
precision is the most informative single number, so it is reported alongside
ROC-AUC, F1, precision, and recall.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Authoritative metric order for tables.
METRIC_NAMES: tuple[str, ...] = ("roc_auc", "pr_auc", "f1", "precision", "recall")


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return the standard metric dict for binary labels and positive-class scores.

    y_score is P(label == 1). Threshold-based metrics use `threshold`.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
