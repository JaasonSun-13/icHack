#!/usr/bin/env python3
"""
train.py – Training pipeline.

Usage:
    python train.py
    python train.py --config configs/default.yaml
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logging, load_config, save_model, save_json, get_timestamp
from src.data import load_master, load_events, validate, compute_severity
from src.features import build_all_features, get_feature_columns, EVENT_TYPES
from src.labels import create_all_labels
from src.splits import get_walk_forward_splits
from src.models import train_model, predict_model, get_importance, EnsembleStack
from src.calibration import CalibrationManager
from src.metrics import binary_metrics, multiclass_metrics, regression_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    logger.info("=" * 70)
    logger.info("Social Volatility Forecaster – Training")
    logger.info("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────
    master = load_master(cfg["data"]["master_csv"])
    events = load_events(cfg["data"]["events_csv"])
    ok, issues = validate(master, events)
    if not ok:
        return 1
    events = compute_severity(master, events)

    # ── 2. Build features ─────────────────────────────────────────────────
    feat_cfg = cfg["features"]
    features_df = build_all_features(
        master, events,
        rolling_windows=feat_cfg["rolling_windows"],
        event_windows=feat_cfg["event_windows"],
    )
    feature_cols = get_feature_columns(features_df)
    logger.info(f"Features: {len(feature_cols)} columns")

    # ── 3. Create labels ──────────────────────────────────────────────────
    lab_cfg = cfg["labels"]
    horizons = lab_cfg["horizons"]
    labels_df = create_all_labels(
        master, events,
        horizons=horizons,
        attention_weight=lab_cfg["social_vol_weights"]["attention"],
        market_weight=lab_cfg["social_vol_weights"]["market"],
    )

    # ── 4. Merge ──────────────────────────────────────────────────────────
    combined = features_df.merge(labels_df, on="date", how="inner")
    logger.info(f"Combined: {combined.shape}")

    # Event type encoding
    type_to_idx = {t: i for i, t in enumerate(sorted(EVENT_TYPES))}
    idx_to_type = {i: t for t, i in type_to_idx.items()}
    n_classes = len(type_to_idx)

    for k in horizons:
        combined[f"event_type_{k}d"] = combined[f"event_type_{k}d"].map(type_to_idx)

    # ── 5. Walk-forward splits ────────────────────────────────────────────
    val_cfg = cfg["validation"]
    splits = get_walk_forward_splits(
        combined["date"], val_cfg["val_years"], val_cfg["test_year"]
    )

    # ── 6. Train ──────────────────────────────────────────────────────────
    model_cfg = cfg["model"]["lgbm_params"]
    cal_manager = CalibrationManager(val_cfg["calibration_method"])
    all_predictions = []
    all_metrics = {}
    trained_models = {}

    for split in splits:
        fold = split["fold"]
        vy = split["val_year"]
        tr_idx = split["train_idx"]
        va_idx = split["val_idx"]
        fold_tag = f"fold_{fold}" if fold >= 0 else "final"

        logger.info(f"\n{'='*60}")
        logger.info(f"  {fold_tag.upper()} (val_year={vy}, train={len(tr_idx)}, val={len(va_idx)})")
        logger.info(f"{'='*60}")

        if len(va_idx) == 0 and fold >= 0:
            logger.warning(f"  Skipping {fold_tag} — no validation data for {vy}")
            continue

        X_tr = combined.iloc[tr_idx][feature_cols].values.astype(float)
        X_va = combined.iloc[va_idx][feature_cols].values.astype(float) if len(va_idx) > 0 else np.empty((0, len(feature_cols)))

        fold_pred = combined.iloc[va_idx][["date"]].copy()
        fold_pred["fold"] = fold
        fold_pred["val_year"] = vy
        fold_metrics = {}

        for k in horizons:
            logger.info(f"\n  ── Horizon {k}d ──")

            # ── Risk (binary) ──
            y_tr_r = combined.iloc[tr_idx][f"risk_{k}d"].values.astype(float)
            y_va_r = combined.iloc[va_idx][f"risk_{k}d"].values.astype(float)

            params = copy.deepcopy(model_cfg["binary"])
            risk_model = train_model(X_tr, y_tr_r, X_va, y_va_r, params, "binary")
            risk_prob_single = predict_model(risk_model, X_va, "binary")

            logger.info(f"    Training ensemble ...")
            ens = EnsembleStack("binary")
            ens.train(X_tr, y_tr_r, X_va, y_va_r)
            risk_prob_ens = ens.predict(X_va)

            risk_prob = 0.6 * risk_prob_ens + 0.4 * risk_prob_single
            risk_key = f"risk_{k}d"
            trained_models[risk_key] = risk_model
            trained_models[f"{risk_key}_ensemble"] = ens

            if fold >= 0 and len(y_va_r) > 0:
                cal_manager.fit(risk_key, risk_prob, y_va_r)
            cal_prob = cal_manager.transform(risk_key, risk_prob) if len(risk_prob) > 0 else risk_prob

            fold_pred[f"risk_{k}d_prob"] = cal_prob
            fold_pred[f"risk_{k}d"] = (cal_prob >= 0.5).astype(int) if len(cal_prob) > 0 else []
            fold_pred[f"actual_risk_{k}d"] = y_va_r

            if len(y_va_r) > 0:
                rm = binary_metrics(y_va_r, cal_prob)
                fold_metrics[risk_key] = rm
                logger.info(f"    Risk: AUROC={rm.get('auroc', 0):.3f}  Brier={rm['brier']:.3f}")
            else:
                logger.info(f"    Risk: No val data — skipping metrics")

            # ── Event type (multiclass) ──
            y_tr_t = combined.iloc[tr_idx][f"event_type_{k}d"].values.astype(float)
            y_va_t = combined.iloc[va_idx][f"event_type_{k}d"].values.astype(float)
            tr_mask_t = ~np.isnan(y_tr_t)
            va_mask_t = ~np.isnan(y_va_t)

            if tr_mask_t.sum() > 10 and va_mask_t.sum() > 5:
                params = copy.deepcopy(model_cfg["multiclass"])
                type_model = train_model(
                    X_tr[tr_mask_t], y_tr_t[tr_mask_t],
                    X_va[va_mask_t], y_va_t[va_mask_t],
                    params, "multiclass", num_class=n_classes,
                )
                type_prob = predict_model(type_model, X_va, "multiclass")
                type_pred = np.argmax(type_prob, axis=1)
                type_key = f"event_type_{k}d"
                trained_models[type_key] = type_model

                fold_pred[f"event_type_{k}d"] = type_pred
                fold_pred[f"actual_event_type_{k}d"] = y_va_t

                tm = multiclass_metrics(y_va_t[va_mask_t], type_pred[va_mask_t], type_prob[va_mask_t])
                fold_metrics[type_key] = tm
                logger.info(f"    Type: macro_f1={tm['macro_f1']:.3f}  acc={tm['accuracy']:.3f}")
            else:
                logger.warning(f"    Type: Skipped (insufficient samples)")

            # ── Social volatility (regression) ──
            y_tr_v = combined.iloc[tr_idx][f"social_vol_{k}d"].values.astype(float)
            y_va_v = combined.iloc[va_idx][f"social_vol_{k}d"].values.astype(float)
            tr_mask_v = ~np.isnan(y_tr_v)
            va_mask_v = ~np.isnan(y_va_v)

            if tr_mask_v.sum() > 10 and va_mask_v.sum() > 5:
                params = copy.deepcopy(model_cfg["regression"])
                vol_model = train_model(
                    X_tr[tr_mask_v], y_tr_v[tr_mask_v],
                    X_va[va_mask_v], y_va_v[va_mask_v],
                    params, "regression",
                )
                vol_pred = predict_model(vol_model, X_va, "regression")
                vol_key = f"social_vol_{k}d"
                trained_models[vol_key] = vol_model

                fold_pred[f"social_vol_{k}d"] = np.clip(vol_pred, 0, 1)
                fold_pred[f"actual_social_vol_{k}d"] = y_va_v

                vm = regression_metrics(y_va_v[va_mask_v], vol_pred[va_mask_v])
                fold_metrics[vol_key] = vm
                logger.info(f"    SocVol: RMSE={vm['rmse']:.3f}  Spearman={vm.get('spearman', 0):.3f}")
            else:
                logger.warning(f"    SocVol: Skipped (insufficient samples)")

        all_predictions.append(fold_pred)
        all_metrics[fold_tag] = fold_metrics

    # ── 7. Save ───────────────────────────────────────────────────────────
    timestamp = get_timestamp()
    art_dir = os.path.join(cfg["output"]["artifacts_dir"], timestamp)
    os.makedirs(art_dir, exist_ok=True)

    for key, model in trained_models.items():
        save_model(model, {"key": key}, os.path.join(art_dir, f"{key}_model.joblib"))

    import joblib
    joblib.dump(cal_manager, os.path.join(art_dir, "calibrators.joblib"))

    save_json({"type_to_idx": type_to_idx, "idx_to_type": idx_to_type},
              os.path.join(art_dir, "event_type_mapping.json"))
    save_json(feature_cols, os.path.join(art_dir, "feature_columns.json"))

    # Feature importance
    best_risk = f"risk_{horizons[0]}d"
    if best_risk in trained_models:
        imp = get_importance(trained_models[best_risk], feature_cols)
        imp_sorted = dict(sorted(imp.items(), key=lambda x: -x[1])[:30])
        save_json(imp_sorted, os.path.join(art_dir, "feature_importance.json"))

    # Symlink
    latest = os.path.join(cfg["output"]["artifacts_dir"], "latest")
    if os.path.islink(latest):
        os.remove(latest)
    elif os.path.isdir(latest):
        import shutil
        shutil.rmtree(latest)
    try:
        os.symlink(os.path.abspath(art_dir), latest)
    except OSError:
        # Windows or permission issue: copy instead
        import shutil
        shutil.copytree(art_dir, latest)

    # Predictions
    pred_df = pd.concat(all_predictions, ignore_index=True)
    out_dir = cfg["output"]["outputs_dir"]
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Predictions: {pred_path} ({len(pred_df)} rows)")

    # Metrics
    rep_dir = cfg["output"]["reports_dir"]
    os.makedirs(rep_dir, exist_ok=True)
    save_json(all_metrics, os.path.join(rep_dir, "metrics.json"))

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info(f"  Models:      {art_dir}")
    logger.info(f"  Predictions: {pred_path}")
    logger.info(f"  Evaluate:    python evaluate.py")
    logger.info(f"  Predict:     python predict.py --model_dir {art_dir} --latest")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
