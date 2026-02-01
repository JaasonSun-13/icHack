# src/models.py
"""LightGBM models with optional ensemble stacking."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_HAS_LGB = False
_HAS_XGB = False
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    pass
if not _HAS_LGB:
    try:
        import xgboost as xgb
        _HAS_XGB = True
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(X_train, y_train, X_val, y_val, params, model_type="binary", num_class=8):
    if not (_HAS_LGB or _HAS_XGB):
        raise ImportError("pip install lightgbm")
    n_est = params.pop("n_estimators", 500)
    early = params.pop("early_stopping_rounds", 50)
    if model_type == "multiclass":
        params["num_class"] = num_class
    if _HAS_LGB:
        return _train_lgb(X_train, y_train, X_val, y_val, params, n_est, early)
    return _train_xgb(X_train, y_train, X_val, y_val, params, n_est, early, model_type)


def predict_model(model, X, model_type="binary"):
    if _HAS_LGB:
        return model.predict(X)
    import xgboost as xgb
    return model.predict(xgb.DMatrix(X))


def get_importance(model, feature_names):
    if _HAS_LGB:
        imp = model.feature_importance(importance_type="gain")
    else:
        imp = list(model.get_score(importance_type="gain").values())
    return dict(zip(feature_names, imp))


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE STACKING
# ═══════════════════════════════════════════════════════════════════════════════

class EnsembleStack:
    """3 diverse LightGBM base learners → logistic/ridge meta-learner.

    Typically adds 2-5% AUROC over single model.

    Usage:
        ens = EnsembleStack("binary")
        ens.train(X_tr, y_tr, X_va, y_va)
        probs = ens.predict(X_new)
    """

    def __init__(self, model_type="binary", num_class=8):
        self.model_type = model_type
        self.num_class = num_class
        self.base_models = []
        self.meta_model = None

        # Diverse base configs
        self.base_configs = [
            {"num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.8,
             "bagging_fraction": 0.8, "bagging_freq": 5},
            {"num_leaves": 63, "learning_rate": 0.03, "feature_fraction": 0.7,
             "bagging_fraction": 0.7, "bagging_freq": 3},
            {"num_leaves": 15, "learning_rate": 0.08, "feature_fraction": 0.9,
             "bagging_fraction": 0.9, "bagging_freq": 7},
        ]

    def train(self, X_train, y_train, X_val, y_val):
        from sklearn.linear_model import LogisticRegression, Ridge

        self.base_models = []
        val_preds = []

        for i, cfg in enumerate(self.base_configs):
            params = dict(cfg)
            params["verbose"] = -1
            if self.model_type == "binary":
                params.update(objective="binary", metric="auc", is_unbalance=True)
            elif self.model_type == "multiclass":
                params.update(objective="multiclass", metric="multi_logloss", num_class=self.num_class)
            else:
                params.update(objective="regression", metric="rmse")

            model = _train_lgb(X_train, y_train, X_val, y_val, params, 500, 50)
            self.base_models.append(model)
            if len(X_val) > 0:
                val_preds.append(model.predict(X_val))
            logger.info(f"    Base model {i+1}/{len(self.base_configs)} trained")

        # Skip meta-learner if no val data
        if len(X_val) == 0 or len(val_preds) == 0:
            logger.info(f"    No val data — skipping meta-learner, will average base models")
            self.meta_model = None
            return

        meta_X = np.column_stack(val_preds)

        if self.model_type == "binary":
            n_classes = len(np.unique(y_val[~np.isnan(y_val)]))
            if n_classes >= 2:
                self.meta_model = LogisticRegression(max_iter=1000)
                self.meta_model.fit(meta_X, y_val)
            else:
                logger.warning(f"    Only {n_classes} class in val data — "
                               f"falling back to averaging (no meta-learner)")
                self.meta_model = None
        elif self.model_type == "regression":
            self.meta_model = Ridge(alpha=1.0)
            self.meta_model.fit(meta_X, y_val)
        else:
            self.meta_model = None  # average for multiclass

        logger.info(f"    Ensemble trained ({self.model_type})")

    def predict(self, X):
        preds = [m.predict(X) for m in self.base_models]
        if self.meta_model is not None:
            meta_X = np.column_stack(preds)
            if self.model_type == "binary":
                return self.meta_model.predict_proba(meta_X)[:, 1]
            return self.meta_model.predict(meta_X)
        return np.mean(preds, axis=0)

    def feature_importance(self, importance_type="gain"):
        """Average importance across base models."""
        imps = [m.feature_importance(importance_type=importance_type) for m in self.base_models]
        return np.mean(imps, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL
# ═══════════════════════════════════════════════════════════════════════════════

def _train_lgb(X_tr, y_tr, X_val, y_val, params, n_est, early):
    import lightgbm as lgb
    dtrain = lgb.Dataset(X_tr, label=y_tr)

    # If val set is empty, train without early stopping
    if len(X_val) == 0:
        logger.info(f"    No val data — training {n_est} rounds without early stopping")
        return lgb.train(
            params, dtrain,
            num_boost_round=n_est,
            valid_sets=[dtrain],
            valid_names=["train"],
            callbacks=[lgb.log_evaluation(100)],
        )

    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    return lgb.train(
        params, dtrain,
        num_boost_round=n_est,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(early), lgb.log_evaluation(100)],
    )


def _train_xgb(X_tr, y_tr, X_val, y_val, params, n_est, early, mtype):
    import xgboost as xgb
    xp = {"max_depth": 6, "learning_rate": params.get("learning_rate", 0.05), "verbosity": 0}
    if mtype == "binary":
        xp.update(objective="binary:logistic", eval_metric="auc")
    elif mtype == "multiclass":
        xp.update(objective="multi:softprob", num_class=params.get("num_class", 8), eval_metric="mlogloss")
    else:
        xp.update(objective="reg:squarederror", eval_metric="rmse")
    dtrain = xgb.DMatrix(X_tr, label=y_tr)

    if len(X_val) == 0:
        return xgb.train(xp, dtrain, num_boost_round=n_est, verbose_eval=100)

    dval = xgb.DMatrix(X_val, label=y_val)
    return xgb.train(xp, dtrain, num_boost_round=n_est,
                     evals=[(dtrain, "train"), (dval, "val")],
                     early_stopping_rounds=early, verbose_eval=100)
