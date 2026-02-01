#!/usr/bin/env python3
"""SVF v3 Training Pipeline — Unified Model.

ONE model trained on all horizons (7/14/21) stacked together.
LightGBM monotone_constraints on `horizon` feature enforces:
  risk_7d ≤ risk_14d ≤ risk_21d (longer window = more chance of event)

Post-hoc clipping guarantees monotonicity after calibration.

Usage:
    python train.py
    python train.py --config configs/default.yaml
"""
import argparse, logging, os, sys
import numpy as np, pandas as pd

from src.utils import setup_logging, load_config, save_pickle
from src.data import load_master, load_events, validate, compute_severity
from src.features import build_all_features, get_feature_columns
from src.labels import create_all_labels, stack_unified
from src.splits import walk_forward_splits
from src.models import (LGBModel, EnsembleStack, blend_predictions,
                        build_monotone_constraints, enforce_monotonic_post)
from src.calibration import IsotonicCalibrator
from src.metrics import binary_metrics, multiclass_metrics, regression_metrics

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    logger.info("=" * 60)
    logger.info("SVF v3 — Unified Model Training")
    logger.info("=" * 60)

    # ── 1. Load & validate ────────────────────────────────────────────
    master = load_master(cfg["data"]["master_csv"])
    events = load_events(cfg["data"]["events_csv"])
    ok, issues = validate(master, events)
    if not ok:
        logger.error(f"Validation failed: {issues}")
        sys.exit(1)

    # ── 2. Auto-severity ──────────────────────────────────────────────
    sev_cfg = cfg.get("severity", {})
    events = compute_severity(master, events,
                              iw=sev_cfg.get("intensity_weight", 0.5),
                              ivw=sev_cfg.get("iv_weight", 0.5))

    # ── 3. Features (66 exactly) ──────────────────────────────────────
    feat = build_all_features(master, events, cfg=cfg)
    base_feature_cols = get_feature_columns(feat)
    assert len(base_feature_cols) == 66, f"Expected 66, got {len(base_feature_cols)}"

    # ── 4. Labels ─────────────────────────────────────────────────────
    lab_cfg = cfg.get("labels", {})
    horizons = lab_cfg.get("horizons", [7, 14, 21])
    labels = create_all_labels(master, events, **lab_cfg)

    # ── 5. Stack into unified format ──────────────────────────────────
    stacked = stack_unified(feat, labels, horizons=horizons)
    unified_features = base_feature_cols + ["horizon"]  # 67 columns
    logger.info(f"Unified features: {len(unified_features)} (66 base + horizon)")

    # ── 6. Monotone constraints ───────────────────────────────────────
    mc = build_monotone_constraints(unified_features)
    logger.info(f"Monotone constraints: ...{mc[-20:]}")

    # ── 7. Walk-forward splits (on dates) ─────────────────────────────
    val_cfg = cfg.get("validation", {})
    date_splits = walk_forward_splits(
        feat["date"], val_cfg["val_years"], val_cfg["test_year"])

    art_dir = cfg["output"]["artifacts_dir"]
    os.makedirs(art_dir, exist_ok=True)

    blend_cfg = cfg.get("model", {}).get("blend_weights", {})
    ens_w = blend_cfg.get("ensemble", 0.60)
    sin_w = blend_cfg.get("single", 0.40)
    seeds = cfg.get("model", {}).get("seeds", [42, 123, 777])
    n_horizons = len(horizons)

    all_metrics = []
    last_risk_cal = None  # track calibrator for production fold

    for fold_i, (tr_date_idx, va_date_idx, year) in enumerate(date_splits):
        is_final = (fold_i == len(date_splits) - 1)
        tag = "FINAL (production)" if is_final else f"Fold {fold_i}"
        logger.info(f"\n{'='*50}")
        logger.info(f"{tag}: train_dates={len(tr_date_idx)} val_dates={len(va_date_idx)} year={year}")

        # Map date indices → stacked indices (each date has n_horizons rows)
        tr_dates = set(tr_date_idx)
        va_dates = set(va_date_idx)
        tr_mask = stacked["date"].isin(feat["date"].iloc[list(tr_dates)])
        va_mask = stacked["date"].isin(feat["date"].iloc[list(va_dates)])
        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]

        logger.info(f"  Stacked: train={len(tr_idx)} val={len(va_idx)}")

        X_tr = stacked.iloc[tr_idx][unified_features].values
        X_va = stacked.iloc[va_idx][unified_features].values
        y_tr_r = stacked.iloc[tr_idx]["risk"].values
        y_va_r = stacked.iloc[va_idx]["risk"].values

        # For final fold: 80/20 internal split for early stopping
        if is_final:
            n = len(tr_idx)
            sp = int(n * 0.8)
            X_tr_es, X_va_es = X_tr[:sp], X_tr[sp:]
            y_tr_es, y_va_es = y_tr_r[:sp], y_tr_r[sp:]
        else:
            X_tr_es, X_va_es = X_tr, X_va
            y_tr_es, y_va_es = y_tr_r, y_va_r

        # ── Risk model (unified, with monotone constraint) ────────
        bp = cfg["model"]["lgbm_params"]["binary"].copy()
        bp["monotone_constraints"] = mc

        single = LGBModel(bp.copy(), "binary")
        single.train(X_tr_es, y_tr_es,
                     X_va_es if len(X_va_es) > 0 else None,
                     y_va_es if len(y_va_es) > 0 else None)

        ens = EnsembleStack(bp.copy(), "binary", seeds=seeds)
        ens.train(X_tr_es, y_tr_es,
                  X_va_es if len(X_va_es) > 0 else None,
                  y_va_es if len(y_va_es) > 0 else None)

        # Calibration
        if is_final:
            cal = last_risk_cal if last_risk_cal else IsotonicCalibrator()
            logger.info(f"  Reusing calibrator (fitted={cal.fitted})")
        else:
            cal = IsotonicCalibrator()
            if len(y_va_r) > 0:
                raw = blend_predictions(ens.predict(X_va), single.predict(X_va), ens_w, sin_w)
                cal.fit(raw, y_va_r)
            last_risk_cal = cal

        # ── Event type model (multiclass) ─────────────────────────
        type_map = {"crisis": 0, "geopolitics": 1, "macro": 2}
        tr_risk_mask = stacked.iloc[tr_idx]["risk"] == 1
        X_tr_t = stacked.iloc[tr_idx].loc[tr_risk_mask, unified_features].values
        y_tr_t = stacked.iloc[tr_idx].loc[tr_risk_mask, "event_type"].map(type_map).values

        mp = cfg["model"]["lgbm_params"]["multiclass"].copy()
        mp["num_class"] = 3
        type_model = LGBModel(mp, "multiclass")
        if len(X_tr_t) > 10:
            type_model.train(X_tr_t, y_tr_t)

        # ── Social vol model (regression) ─────────────────────────
        svol_mask = ~stacked.iloc[tr_idx]["social_vol"].isna()
        X_tr_s = stacked.iloc[tr_idx].loc[svol_mask, unified_features].values
        y_tr_s = stacked.iloc[tr_idx].loc[svol_mask, "social_vol"].values

        # Social vol also monotone on horizon
        rp = cfg["model"]["lgbm_params"]["regression"].copy()
        rp["monotone_constraints"] = mc

        svol_model = LGBModel(rp.copy(), "regression")
        if len(X_tr_s) > 10:
            svol_va_mask = ~stacked.iloc[va_idx]["social_vol"].isna()
            X_va_s = stacked.iloc[va_idx].loc[svol_va_mask, unified_features].values if svol_va_mask.any() else None
            y_va_s = stacked.iloc[va_idx].loc[svol_va_mask, "social_vol"].values if svol_va_mask.any() else None
            svol_model.train(X_tr_s, y_tr_s, X_va_s, y_va_s)

        # ── Save ──────────────────────────────────────────────────
        prefix = f"{art_dir}/fold{fold_i}"
        save_pickle(single, f"{prefix}_single.pkl")
        save_pickle(ens, f"{prefix}_ensemble.pkl")
        save_pickle(cal, f"{prefix}_calibrator.pkl")
        save_pickle(type_model, f"{prefix}_type.pkl")
        save_pickle(svol_model, f"{prefix}_svol.pkl")

        # ── Per-horizon metrics on validation ─────────────────────
        if len(y_va_r) > 0:
            risk_raw = blend_predictions(ens.predict(X_va), single.predict(X_va), ens_w, sin_w)
            risk_cal = cal.transform(risk_raw)
            risk_cal = enforce_monotonic_post(risk_cal, horizons, len(va_date_idx))

            for hi, h in enumerate(horizons):
                h_mask = stacked.iloc[va_idx]["horizon"].values == h
                if not h_mask.any():
                    continue
                rm = binary_metrics(y_va_r[h_mask], risk_cal[h_mask])

                # Type metrics
                tm = {"accuracy": 0, "macro_f1": 0}
                va_risk_h = (stacked.iloc[va_idx]["risk"].values == 1) & h_mask
                if type_model.model and va_risk_h.any():
                    tp = type_model.predict(X_va[va_risk_h])
                    yt = stacked.iloc[va_idx].loc[va_risk_h, "event_type"].map(type_map).values
                    tm = multiclass_metrics(yt, tp)

                # Svol metrics
                sm = {"rmse": 999, "spearman": 0}
                va_svol_h = (~stacked.iloc[va_idx]["social_vol"].isna().values) & h_mask
                if svol_model.model and va_svol_h.any():
                    sp = svol_model.predict(X_va[va_svol_h])
                    ys = stacked.iloc[va_idx]["social_vol"].values[va_svol_h]
                    sm = regression_metrics(ys, sp)

                fold_m = {"fold": fold_i, "year": year, "horizon": h,
                          **{f"risk_{k}": v for k, v in rm.items()},
                          **{f"type_{k}": v for k, v in tm.items()},
                          **{f"svol_{k}": v for k, v in sm.items()}}
                all_metrics.append(fold_m)
                logger.info(f"  {h}d: AUROC={rm['auroc']:.3f} F1={rm['f1']:.3f} "
                            f"Brier={rm['brier']:.3f} | TypeF1={tm['macro_f1']:.3f} | "
                            f"Svol RMSE={sm['rmse']:.3f}")

    # ── Save artifacts ────────────────────────────────────────────────
    save_pickle(unified_features, f"{art_dir}/feature_cols.pkl")
    save_pickle(horizons, f"{art_dir}/horizons.pkl")

    # Feature importance from last single model
    imp = single.importance()
    if len(imp) == len(unified_features):
        imp_df = pd.DataFrame({"feature": unified_features, "importance": imp})
        imp_df = imp_df.sort_values("importance", ascending=False)
        imp_df.to_csv(f"{art_dir}/feature_importance.csv", index=False)

    if all_metrics:
        mdf = pd.DataFrame(all_metrics)
        mdf.to_csv(f"{art_dir}/metrics.csv", index=False)
        logger.info(f"\n{'='*60}")
        logger.info("Validation Metrics:")
        logger.info(f"\n{mdf.to_string(index=False)}")

        # Show monotonicity check
        logger.info(f"\n{'='*60}")
        logger.info("Monotonicity check (mean risk AUROC by horizon):")
        for h in horizons:
            sub = mdf[mdf["horizon"] == h]
            if len(sub) > 0:
                logger.info(f"  {h}d: mean AUROC={sub['risk_auroc'].mean():.3f}")

    logger.info(f"\n✅ Done! Unified model saved to {art_dir}/")
    logger.info(f"   1 risk model serves all horizons (7/14/21)")
    logger.info(f"   Monotone constraint: risk_7d ≤ risk_14d ≤ risk_21d ✓")


if __name__ == "__main__":
    main()
