"""Evaluation metrics per spec: AUROC, F1, Brier, Macro-F1, RMSE, Spearman."""
import numpy as np
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, brier_score_loss,
                             accuracy_score, mean_squared_error, r2_score)
from scipy.stats import spearmanr


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = 0.5
    return {
        "auroc": auroc,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
    }


def multiclass_metrics(y_true, y_prob, class_names=None):
    if y_prob.ndim > 1:
        y_pred = np.argmax(y_prob, axis=1)
    else:
        y_pred = y_prob.astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    try:
        spear = spearmanr(y_true, y_pred).correlation
        if np.isnan(spear):
            spear = 0.0
    except:
        spear = 0.0
    return {
        "rmse": rmse,
        "spearman": spear,
    }
