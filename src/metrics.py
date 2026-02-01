# src/metrics.py
"""Evaluation metrics for binary, multiclass, and regression targets."""

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Metrics for binary risk classification."""
    valid = ~(np.isnan(y_true) | np.isnan(y_prob))
    yt, yp = y_true[valid].astype(int), y_prob[valid]
    yhat = (yp >= threshold).astype(int)

    m = {"n_samples": int(len(yt)), "positive_rate": float(yt.mean())}
    if len(np.unique(yt)) < 2:
        logger.warning("Only one class present; some metrics undefined")
        m.update({"auroc": np.nan, "auprc": np.nan})
    else:
        m["auroc"] = float(roc_auc_score(yt, yp))
        m["auprc"] = float(average_precision_score(yt, yp))

    m["brier"] = float(brier_score_loss(yt, yp))
    m["accuracy"] = float(accuracy_score(yt, yhat))
    m["precision"] = float(precision_score(yt, yhat, zero_division=0))
    m["recall"] = float(recall_score(yt, yhat, zero_division=0))
    m["f1"] = float(f1_score(yt, yhat, zero_division=0))
    return m


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Metrics for event type classification."""
    valid = ~np.isnan(y_true.astype(float))
    yt, yhat = y_true[valid].astype(int), y_pred[valid].astype(int)

    m = {"n_samples": int(len(yt))}
    m["accuracy"] = float(accuracy_score(yt, yhat))
    m["macro_f1"] = float(f1_score(yt, yhat, average="macro", zero_division=0))
    m["weighted_f1"] = float(f1_score(yt, yhat, average="weighted", zero_division=0))

    # Top-2 accuracy (if probabilities available)
    if y_prob is not None and y_prob.ndim == 2:
        vp = y_prob[valid]
        top2 = np.argsort(vp, axis=1)[:, -2:]
        m["top2_accuracy"] = float(np.mean([yt[i] in top2[i] for i in range(len(yt))]))

    m["confusion_matrix"] = confusion_matrix(yt, yhat).tolist()
    return m


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Metrics for social volatility regression."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[valid], y_pred[valid]

    m = {"n_samples": int(len(yt))}
    m["rmse"] = float(np.sqrt(mean_squared_error(yt, yp)))
    m["mae"] = float(mean_absolute_error(yt, yp))

    if len(yt) > 2:
        m["spearman"], _ = spearmanr(yt, yp)
        m["pearson"], _ = pearsonr(yt, yp)
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        m["r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        m["spearman"] = m["pearson"] = m["r2"] = np.nan

    return m
