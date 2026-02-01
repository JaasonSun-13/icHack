#!/usr/bin/env python3
"""
predict.py – Generate predictions.

Usage:
    python predict.py --latest
    python predict.py --model_dir artifacts/latest --latest
    python predict.py --date 2025-06-15
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logging, load_config, load_model
from src.data import load_master, load_events, compute_severity
from src.features import build_all_features, get_feature_columns, EVENT_TYPES
from src.models import predict_model, get_importance, EnsembleStack
from src.calibration import CalibrationManager
from src.policy import PolicyTree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="artifacts/latest")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--output", default="outputs/latest_prediction.csv")
    args = parser.parse_args()

    logger = setup_logging("INFO")
    cfg = load_config(args.config)

    # ── Load data & features ──────────────────────────────────────────────
    master = load_master(cfg["data"]["master_csv"])
    events = load_events(cfg["data"]["events_csv"])
    events = compute_severity(master, events)

    feat_cfg = cfg["features"]
    features_df = build_all_features(
        master, events,
        rolling_windows=feat_cfg["rolling_windows"],
        event_windows=feat_cfg["event_windows"],
    )

    # ── Load model metadata ───────────────────────────────────────────────
    fc_path = os.path.join(args.model_dir, "feature_columns.json")
    if not os.path.exists(fc_path):
        logger.error(f"No models found at {args.model_dir}")
        logger.error("Run train.py first, then use --model_dir to point to the artifacts folder.")
        logger.error("Check:  ls artifacts/")
        return 1

    with open(fc_path) as f:
        feature_cols = json.load(f)

    tm_path = os.path.join(args.model_dir, "event_type_mapping.json")
    with open(tm_path) as f:
        mapping = json.load(f)
    idx_to_type = {int(k): v for k, v in mapping["idx_to_type"].items()}

    import joblib
    cal_path = os.path.join(args.model_dir, "calibrators.joblib")
    cal_manager = joblib.load(cal_path) if os.path.exists(cal_path) else CalibrationManager()

    # ── Select rows ───────────────────────────────────────────────────────
    if args.latest:
        features_df = features_df.tail(1)
    elif args.date:
        features_df = features_df[features_df["date"] == pd.Timestamp(args.date)]

    if features_df.empty:
        logger.error("No data for requested date")
        return 1

    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0
    X = features_df[feature_cols].values.astype(float)
    dates = features_df["date"].values

    # ── Predict ───────────────────────────────────────────────────────────
    horizons = cfg["labels"]["horizons"]
    results = pd.DataFrame({"date": dates})

    for k in horizons:
        # Risk
        rk = f"risk_{k}d"
        ens_path = os.path.join(args.model_dir, f"{rk}_ensemble_model.joblib")
        rpath = os.path.join(args.model_dir, f"{rk}_model.joblib")

        if os.path.exists(ens_path) and os.path.exists(rpath):
            ens = load_model(ens_path)["model"]
            rm = load_model(rpath)["model"]
            prob = 0.6 * ens.predict(X) + 0.4 * predict_model(rm, X, "binary")
            prob = cal_manager.transform(rk, prob)
            results[f"{rk}_prob"] = prob
            results[rk] = (prob >= 0.5).astype(int)
        elif os.path.exists(rpath):
            rm = load_model(rpath)["model"]
            prob = predict_model(rm, X, "binary")
            prob = cal_manager.transform(rk, prob)
            results[f"{rk}_prob"] = prob
            results[rk] = (prob >= 0.5).astype(int)

        # Event type
        tpath = os.path.join(args.model_dir, f"event_type_{k}d_model.joblib")
        if os.path.exists(tpath):
            tm = load_model(tpath)["model"]
            tprob = predict_model(tm, X, "multiclass")
            results[f"event_type_{k}d"] = np.argmax(tprob, axis=1)
            results[f"event_type_{k}d_name"] = results[f"event_type_{k}d"].map(idx_to_type)

        # Social vol
        vpath = os.path.join(args.model_dir, f"social_vol_{k}d_model.joblib")
        if os.path.exists(vpath):
            vm = load_model(vpath)["model"]
            results[f"social_vol_{k}d"] = np.clip(predict_model(vm, X, "regression"), 0, 1)

    # Monotonicity: longer horizon >= shorter
    sorted_h = sorted(horizons)
    for i in range(1, len(sorted_h)):
        cur = f"risk_{sorted_h[i]}d_prob"
        prev = f"risk_{sorted_h[i-1]}d_prob"
        if cur in results.columns and prev in results.columns:
            results[cur] = np.maximum(results[cur].values, results[prev].values)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results.to_csv(args.output, index=False)

    # ── Pretty print ──────────────────────────────────────────────────────
    if args.latest or args.date:
        row = results.iloc[0]
        k = horizons[0]

        risk_prob = row.get(f"risk_{k}d_prob", 0)
        social_vol = row.get(f"social_vol_{k}d", 0)
        etype = row.get(f"event_type_{k}d_name", "unknown")

        # Top drivers
        rpath = os.path.join(args.model_dir, f"risk_{k}d_model.joblib")
        drivers = []
        if os.path.exists(rpath):
            rm = load_model(rpath)["model"]
            imp = get_importance(rm, feature_cols)
            drivers = [{"feature": f, "importance": v}
                       for f, v in sorted(imp.items(), key=lambda x: -x[1])[:5]]

        pol_cfg = cfg.get("policy", {})
        pt = PolicyTree(
            risk_high=pol_cfg.get("risk_high_threshold", 0.7),
            risk_med=pol_cfg.get("risk_medium_threshold", 0.4),
        )
        decision = pt.evaluate(risk_prob, social_vol, etype, drivers)

        icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        icon = icons.get(decision.alert_level, "⚪")

        print(f"\n{'═' * 60}")
        print(f"  SOCIAL VOLATILITY FORECAST")
        print(f"  Date: {row['date']}")
        print(f"{'═' * 60}")

        print(f"\n  📊 EVENT PROBABILITY")
        print(f"  ┌─────────────────────────────────────┐")
        for h in horizons:
            p = row.get(f"risk_{h}d_prob", 0)
            bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
            print(f"  │  Next {h:2d}d: {p:5.1%}  {bar} │")
        print(f"  └─────────────────────────────────────┘")

        print(f"\n  📈 SOCIAL VOLATILITY")
        print(f"  ┌─────────────────────────────────────┐")
        for h in horizons:
            v = row.get(f"social_vol_{h}d", 0)
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            level = "HIGH" if v > 0.7 else "MED" if v > 0.4 else "LOW"
            print(f"  │  Next {h:2d}d: {v:.2f}  {bar} {level:4s} │")
        print(f"  └─────────────────────────────────────┘")

        print(f"\n  🏷️  EVENT TYPE")
        print(f"  ┌─────────────────────────────────────┐")
        for h in horizons:
            t = row.get(f"event_type_{h}d_name", "none")
            print(f"  │  Next {h:2d}d: {str(t):30s}   │")
        print(f"  └─────────────────────────────────────┘")

        print(f"\n  {icon}  ALERT: {decision.alert_level}")

        if decision.drivers:
            print(f"\n  🔍 TOP DRIVERS:")
            for d in decision.drivers:
                print(f"     • {d}")

        print(f"\n  💡 RECOMMENDATIONS:")
        for persona, rec in decision.recommendations.items():
            print(f"     [{persona.upper()}] {rec}")
        print(f"{'═' * 60}\n")

    logger.info(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
