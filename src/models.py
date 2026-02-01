"""LightGBM models: single, ensemble, blend, with monotone constraint support.

Key design: ONE model trained on all horizons (7/14/21) stacked together.
The `horizon` column is a feature, and LightGBM's monotone_constraints
forces: longer horizon → higher risk probability.

This guarantees: risk_7d <= risk_14d <= risk_21d for the same date.
"""
import logging
import numpy as np
import lightgbm as lgb

logger = logging.getLogger(__name__)


class LGBModel:
    """Single LightGBM wrapper with optional monotone constraints."""

    def __init__(self, params, task="binary"):
        self.params = params
        self.task = task
        self.model = None

    def train(self, X_tr, y_tr, X_va=None, y_va=None):
        dtr = lgb.Dataset(X_tr, y_tr)
        cbs = [lgb.log_evaluation(period=0)]
        if X_va is not None and len(X_va) > 0:
            dva = lgb.Dataset(X_va, y_va, reference=dtr)
            cbs.append(lgb.early_stopping(50, verbose=False))
            self.model = lgb.train(self.params, dtr, valid_sets=[dva], callbacks=cbs)
        else:
            self.model = lgb.train(self.params, dtr, callbacks=cbs)

    def predict(self, X):
        if self.model is None:
            return np.zeros(len(X))
        return self.model.predict(X)

    def importance(self):
        if self.model is None:
            return np.zeros(1)
        return self.model.feature_importance(importance_type="gain")


class EnsembleStack:
    """3-model ensemble with different seeds, all sharing same constraints."""

    def __init__(self, params, task="binary", seeds=None):
        self.seeds = seeds or [42, 123, 777]
        self.models = []
        self.params = params
        self.task = task

    def train(self, X_tr, y_tr, X_va=None, y_va=None):
        self.models = []
        for seed in self.seeds:
            p = self.params.copy()
            p["seed"] = seed
            p["feature_fraction_seed"] = seed
            m = LGBModel(p, self.task)
            m.train(X_tr, y_tr, X_va, y_va)
            self.models.append(m)

    def predict(self, X):
        preds = [m.predict(X) for m in self.models]
        return np.mean(preds, axis=0)


def blend_predictions(ens_pred, single_pred, ens_w=0.60, single_w=0.40):
    """Blend ensemble + single predictions."""
    return ens_w * ens_pred + single_w * single_pred


def build_monotone_constraints(feature_cols):
    """Build monotone_constraints string for LightGBM.

    All 66 original features: 0 (no constraint)
    `horizon` column (last feature): +1 (positive = longer horizon → higher risk)

    Returns constraint string like "0,0,...,0,1"
    """
    # horizon is the last column (appended by stack_unified)
    constraints = [0] * len(feature_cols)

    # Find horizon column index
    for i, col in enumerate(feature_cols):
        if col == "horizon":
            constraints[i] = 1  # positive: higher horizon → higher prediction
            logger.info(f"Monotone +1 constraint on '{col}' (index {i})")

    return ",".join(str(c) for c in constraints)


def enforce_monotonic_post(risk_probs, horizons, n_dates):
    """Post-hoc enforcement: ensure risk_7d <= risk_14d <= risk_21d.

    After model prediction + calibration, clip so longer horizons
    are always >= shorter horizons for each date.

    Args:
        risk_probs: array of calibrated probabilities (stacked: n_dates * n_horizons)
        horizons: sorted list [7, 14, 21]
        n_dates: number of unique dates

    Returns:
        corrected risk_probs (same shape)
    """
    n_h = len(horizons)
    out = risk_probs.copy()
    reshaped = out.reshape(n_dates, n_h)  # rows=dates, cols=horizons

    for i in range(n_dates):
        for j in range(1, n_h):
            if reshaped[i, j] < reshaped[i, j-1]:
                reshaped[i, j] = reshaped[i, j-1]

    return reshaped.flatten()
