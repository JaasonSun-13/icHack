# src/data.py
"""Data loading, validation, and auto-severity computation.

Severity is computed automatically from your data:
  - peak GDELT event_intensity during the event
  - peak VIX during the event
  - both converted to percentiles, blended, bucketed 1-5

So events.csv only needs: event_start, event_end, event_type, headline
"""

import logging
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_master(csv_path: str) -> pd.DataFrame:
    """Load master.csv with columns:
    date, vix, vix_average_prev3d, vix_average_prev7d,
    vix_slope_20d, event_intensity, trend_spike
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Master data not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"Master: {df.shape[0]} rows, {df.shape[1]} cols, "
                f"{df['date'].min().date()} to {df['date'].max().date()}")
    return df


def load_events(path: str) -> pd.DataFrame:
    """Load events.csv with columns:
    event_start, event_end, event_type, headline
    (severity column is optional — will be auto-computed later)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Events file not found: {path}")
    df = pd.read_csv(path)
    df["event_start"] = pd.to_datetime(df["event_start"])
    if "event_end" in df.columns:
        df["event_end"] = pd.to_datetime(df["event_end"])
    else:
        df["event_end"] = df["event_start"]
    df = df.sort_values("event_start").reset_index(drop=True)
    logger.info(f"Events: {df.shape[0]} rows, types={df['event_type'].unique().tolist()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# AUTO-SEVERITY
# ═══════════════════════════════════════════════════════════════════════════

def compute_severity(master: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Auto-compute severity (1-5) for each event from master.csv data.

    For each event:
      1. Find all trading days in [event_start, event_end]
      2. peak_intensity = max event_intensity during those days
      3. peak_vix       = max VIX during those days
      4. Convert both to percentile ranks vs ALL daily values
      5. raw_score = 0.5 * intensity_pct + 0.5 * vix_pct
      6. severity  = ceil(raw_score * 5), clipped to [1, 5]

    Overwrites any existing severity column.
    """
    ev = events.copy()
    dates = master["date"].values
    intensity = master["event_intensity"].values if "event_intensity" in master.columns else None
    vix = master["vix"].values if "vix" in master.columns else None

    if intensity is None and vix is None:
        logger.warning("No intensity or VIX data — defaulting severity to 3")
        ev["severity"] = 3
        return ev

    # Pre-sort for fast percentile lookup
    all_intensity_sorted = np.sort(intensity[~np.isnan(intensity)]) if intensity is not None else np.array([])
    all_vix_sorted = np.sort(vix[~np.isnan(vix)]) if vix is not None else np.array([])

    severities = np.full(len(ev), 3.0)

    for idx in range(len(ev)):
        start = ev.iloc[idx]["event_start"]
        end = ev.iloc[idx]["event_end"]

        # Find trading days within [start, end]
        mask = (dates >= np.datetime64(start)) & (dates <= np.datetime64(end))
        day_indices = np.where(mask)[0]

        if len(day_indices) == 0:
            # Event on weekend/holiday — expand ±3 days
            mask = (dates >= np.datetime64(start) - np.timedelta64(3, "D")) & \
                   (dates <= np.datetime64(end) + np.timedelta64(3, "D"))
            day_indices = np.where(mask)[0]

        if len(day_indices) == 0:
            continue

        # Peak during event → percentile vs all history
        int_pct = 0.5
        vix_pct = 0.5

        if intensity is not None and len(all_intensity_sorted) > 0:
            peak_int = np.nanmax(intensity[day_indices])
            int_pct = np.searchsorted(all_intensity_sorted, peak_int) / len(all_intensity_sorted)

        if vix is not None and len(all_vix_sorted) > 0:
            peak_vix = np.nanmax(vix[day_indices])
            vix_pct = np.searchsorted(all_vix_sorted, peak_vix) / len(all_vix_sorted)

        raw = 0.5 * int_pct + 0.5 * vix_pct
        severities[idx] = max(1, min(5, math.ceil(raw * 5)))

    ev["severity"] = severities.astype(int)

    dist = ev["severity"].value_counts().sort_index()
    logger.info(f"Auto-severity: {dict(dist)}")

    return ev


def validate(master: pd.DataFrame, events: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate data integrity."""
    issues = []

    required_master = ["date", "vix", "event_intensity", "trend_spike"]
    for col in required_master:
        if col not in master.columns:
            issues.append(f"Missing '{col}' in master.csv")

    required_events = ["event_start", "event_type"]
    for col in required_events:
        if col not in events.columns:
            issues.append(f"Missing '{col}' in events.csv")

    if issues:
        for i in issues:
            logger.error(f"  ✗ {i}")
    else:
        logger.info("Data validation passed ✓")

    return len(issues) == 0, issues
