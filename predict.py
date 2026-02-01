#!/usr/bin/env python3
"""SVF v3 Prediction — Unified Model.

One model, queried 3 times (horizon=7/14/21) for each date.
Post-hoc monotonicity enforced: risk_7d ≤ risk_14d ≤ risk_21d.

Usage:
    python predict.py --spx 5800 --iv 28 --intensity 8.5 --spike 25
    python predict.py --new_csv today.csv
    python predict.py --last 10
    python predict.py --date 2025-03-01
"""
import argparse, glob, logging, os, sys
import numpy as np, pandas as pd

from src.utils import setup_logging, load_config, load_pickle
from src.data import load_master, load_events, compute_severity
from src.features import build_all_features, get_feature_columns
from src.models import blend_predictions, enforce_monotonic_post
from src.policy import apply_policy, format_alert

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=-1, help="-1=production (last)")
    parser.add_argument("--spx", type=float, default=None)
    parser.add_argument("--iv", type=float, default=None)
    parser.add_argument("--rv", type=float, default=None)
    parser.add_argument("--rv3", type=float, default=None)
    parser.add_argument("--intensity", type=float, default=None)
    parser.add_argument("--spike", type=float, default=None)
    parser.add_argument("--new_csv", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--last", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config()
    setup_logging("INFO")

    master = load_master(cfg["data"]["master_csv"])
    events = load_events(cfg["data"]["events_csv"])
    events = compute_severity(master, events)

    has_new = args.spx is not None or args.new_csv is not None

    if args.new_csv:
        new = pd.read_csv(args.new_csv)
        new.columns = new.columns.str.strip().str.lower()
        new["date"] = pd.to_datetime(new["date"])
        master = pd.concat([master, new], ignore_index=True).sort_values("date").reset_index(drop=True)
    elif args.spx is not None:
        last_date = master["date"].max()
        new_date = last_date + pd.tseries.offsets.BDay(1)
        new_row = pd.DataFrame([{
            "date": new_date,
            "spx_close": args.spx,
            "1_day_vol": args.rv or master["1_day_vol"].iloc[-1],
            "3_day_avg_vol": args.rv3 or master["3_day_avg_vol"].iloc[-1],
            "30_day_implied_vol": args.iv or master["30_day_implied_vol"].iloc[-1],
            "event_intensity": args.intensity or 0,
            "trend_spike": args.spike or 0,
        }])
        master = pd.concat([master, new_row], ignore_index=True)
        logger.info(f"Added: {new_date.strftime('%Y-%m-%d')} SPX={args.spx} IV={args.iv}")

    feat = build_all_features(master, events, cfg=cfg)
    art = cfg["output"]["artifacts_dir"]

    feature_cols = load_pickle(f"{art}/feature_cols.pkl")  # 67 cols (66 + horizon)
    horizons = load_pickle(f"{art}/horizons.pkl")
    base_cols = [c for c in feature_cols if c != "horizon"]

    # Find fold
    matches = sorted(glob.glob(f"{art}/fold*_single.pkl"))
    if not matches:
        logger.error("No models found."); sys.exit(1)
    folds = sorted(set(int(os.path.basename(m).split("fold")[1].split("_")[0]) for m in matches))
    fi = folds[-1] if args.fold == -1 else args.fold

    prefix = f"{art}/fold{fi}"
    single = load_pickle(f"{prefix}_single.pkl")
    ens = load_pickle(f"{prefix}_ensemble.pkl")
    cal = load_pickle(f"{prefix}_calibrator.pkl")
    type_model = load_pickle(f"{prefix}_type.pkl")
    svol_model = load_pickle(f"{prefix}_svol.pkl")

    blend_cfg = cfg.get("model", {}).get("blend_weights", {})
    ens_w = blend_cfg.get("ensemble", 0.60)
    sin_w = blend_cfg.get("single", 0.40)

    # Select dates to predict
    if args.date:
        target = pd.Timestamp(args.date)
        idx = (feat["date"] - target).abs().idxmin()
        date_indices = [idx]
    elif has_new:
        date_indices = [len(feat) - 1]
    else:
        date_indices = list(range(max(0, len(feat) - args.last), len(feat)))

    type_map = {0: "crisis", 1: "geopolitics", 2: "macro"}
    pcfg = cfg["policy"]

    for di in date_indices:
        row_base = feat.iloc[di:di+1][base_cols].values  # (1, 66)
        date = feat.iloc[di]["date"]

        risk_probs = {}
        svols = {}
        etypes = {}

        # Query model once per horizon
        rows_stacked = []
        for h in horizons:
            row = np.hstack([row_base[0], [h]])  # append horizon
            rows_stacked.append(row)
        X = np.array(rows_stacked)  # (3, 67)

        # Risk
        risk_raw = blend_predictions(ens.predict(X), single.predict(X), ens_w, sin_w)
        risk_cal = cal.transform(risk_raw)

        # Enforce monotonicity: risk_7d <= risk_14d <= risk_21d
        for j in range(1, len(horizons)):
            if risk_cal[j] < risk_cal[j-1]:
                risk_cal[j] = risk_cal[j-1]

        # Social vol
        if svol_model.model:
            sv = svol_model.predict(X)
            for j in range(1, len(horizons)):
                if sv[j] < sv[j-1]:
                    sv[j] = sv[j-1]
        else:
            sv = np.full(len(horizons), 0.5)

        # Event type
        if type_model.model:
            tp = type_model.predict(X)
            if tp.ndim > 1:
                et = [type_map[np.argmax(tp[j])] for j in range(len(horizons))]
            else:
                et = [type_map[int(tp[j])] for j in range(len(horizons))]
        else:
            et = ["crisis"] * len(horizons)

        # Apply policy per horizon
        print(f"\n{'='*65}")
        print(f"  SVF v3 — {pd.Timestamp(date).strftime('%Y-%m-%d')}")
        print(f"{'='*65}")
        print(f"  {'Horizon':>8s}  {'Alert':15s}  {'Risk':>8s}  {'SVol':>8s}  Type")
        print(f"  {'─'*8}  {'─'*15}  {'─'*8}  {'─'*8}  {'─'*12}")

        for j, h in enumerate(horizons):
            alerts, recs = apply_policy(
                np.array([risk_cal[j]]), np.array([sv[j]]), [et[j]],
                pcfg["risk_high"], pcfg["risk_medium"],
                pcfg["svol_high"], pcfg["svol_medium"],
                pcfg.get("boost_types", ["geopolitics", "macro"]))
            print(f"  {h:>5d}d    {format_alert(alerts[0]):15s}  {risk_cal[j]:7.1%}  {sv[j]:7.1%}  {et[j]}")

        # Show the longest horizon recommendation
        alerts_21, recs_21 = apply_policy(
            np.array([risk_cal[-1]]), np.array([sv[-1]]), [et[-1]],
            pcfg["risk_high"], pcfg["risk_medium"],
            pcfg["svol_high"], pcfg["svol_medium"],
            pcfg.get("boost_types"))
        print(f"\n  📋 {recs_21[0]}")
        print(f"  ✅ Monotonic: {risk_cal[0]:.1%} ≤ {risk_cal[1]:.1%} ≤ {risk_cal[2]:.1%}")
        print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
