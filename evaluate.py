#!/usr/bin/env python3
"""Evaluate SVF v3 models across all folds."""
import glob, os
import pandas as pd
from src.utils import load_config

def main():
    cfg = load_config()
    art = cfg["output"]["artifacts_dir"]
    try:
        mdf = pd.read_csv(f"{art}/metrics.csv")
        print("\n" + "="*80)
        print("SVF v3 Validation Metrics")
        print("="*80)
        print(mdf.to_string(index=False))
        print("\nMean by horizon:")
        for h in [7, 14, 30]:
            sub = mdf[mdf["horizon"] == h]
            if len(sub) > 0:
                print(f"  {h}d: AUROC={sub['risk_auroc'].mean():.3f} "
                      f"F1={sub['risk_f1'].mean():.3f} "
                      f"Brier={sub['risk_brier'].mean():.3f}")
    except FileNotFoundError:
        print("No metrics.csv found. Run train.py first.")

if __name__ == "__main__":
    main()
