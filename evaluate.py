#!/usr/bin/env python3
"""
evaluate.py – Evaluate predictions against actuals.

Usage:
    python evaluate.py --pred outputs/predictions.csv
    python evaluate.py --pred outputs/predictions.csv --by_fold
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logging, save_json
from src.metrics import binary_metrics, multiclass_metrics, regression_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", default="outputs/predictions.csv")
    p.add_argument("--output", default="reports/evaluation.json")
    p.add_argument("--horizons", nargs="+", type=int, default=[7, 14, 30])
    p.add_argument("--by_fold", action="store_true")
    return p.parse_args()


def evaluate_subset(df, horizons):
    m = {}
    for k in horizons:
        hm = {}
        # Risk
        prob_col, act_col = f"risk_{k}d_prob", f"actual_risk_{k}d"
        if prob_col in df.columns and act_col in df.columns:
            hm[f"risk_{k}d"] = binary_metrics(
                df[act_col].values, df[prob_col].values
            )
        # Type
        pred_col, act_col = f"event_type_{k}d", f"actual_event_type_{k}d"
        if pred_col in df.columns and act_col in df.columns:
            hm[f"event_type_{k}d"] = multiclass_metrics(
                df[act_col].values, df[pred_col].values
            )
        # Vol
        pred_col, act_col = f"social_vol_{k}d", f"actual_social_vol_{k}d"
        if pred_col in df.columns and act_col in df.columns:
            hm[f"social_vol_{k}d"] = regression_metrics(
                df[act_col].values, df[pred_col].values
            )
        if hm:
            m[f"horizon_{k}d"] = hm
    return m


def main():
    args = parse_args()
    logger = setup_logging("INFO")

    df = pd.read_csv(args.pred)
    logger.info(f"Loaded {len(df)} predictions")

    results = {}

    if args.by_fold and "fold" in df.columns:
        for fold in sorted(df["fold"].unique()):
            fdf = df[df["fold"] == fold]
            tag = f"fold_{fold}" if fold >= 0 else "final"
            results[tag] = evaluate_subset(fdf, args.horizons)
    else:
        results["overall"] = evaluate_subset(df, args.horizons)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_json(results, args.output)

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    for section, data in results.items():
        print(f"\n  {section.upper()}")
        for hk, hv in data.items():
            print(f"    {hk}:")
            for mk, mv in hv.items():
                if isinstance(mv, dict):
                    for k2, v2 in mv.items():
                        if k2 not in ("confusion_matrix", "n_samples"):
                            print(f"        {mk}.{k2}: {v2}")
                elif isinstance(mv, (int, float)) and not isinstance(mv, bool):
                    if mk not in ("confusion_matrix", "n_samples"):
                        print(f"      {mk}: {mv:.4f}" if isinstance(mv, float) else f"      {mk}: {mv}")
    print(f"\n{'='*60}")
    logger.info(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
