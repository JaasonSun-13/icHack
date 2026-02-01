# src/features.py
"""Feature engineering from your 7 raw inputs.

Input columns: vix, vix_average_prev3d, vix_average_prev7d,
               vix_slope_20d, event_intensity, trend_spike

Creates ~60 features using rolling windows, percentiles, slopes,
event history. All features use PAST data only (no leakage).
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Event type mapping (raw → 3 merged classes) ──────────────────────────
EVENT_TYPES = ["crisis", "geopolitics", "macro"]

TYPE_MERGE = {
    "geopolitics": "geopolitics",
    "political":   "geopolitics",
    "government":  "geopolitics",
    "finance":     "macro",
    "macro":       "macro",
    "economic":    "macro",
    "tech":        "crisis",
    "tech_ai":     "crisis",
    "big_tech":    "crisis",
    "corporate":   "crisis",
    "business":    "crisis",
    "health":      "crisis",
    "security":    "crisis",
    "other":       "crisis",
}


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def build_all_features(
    master: pd.DataFrame,
    events: pd.DataFrame,
    rolling_windows: List[int] = [5, 20, 60],
    event_windows: List[int] = [30, 90],
    **kwargs,
) -> pd.DataFrame:
    """Build complete feature matrix.

    Input: master with 7 columns + events with start/end/type/severity
    Output: DataFrame with ~60 features + date column
    """
    logger.info("Building feature matrix ...")
    df = master[["date"]].copy()

    # 1. VIX features
    vix_feats = _build_vix_features(master, rolling_windows)
    df = df.join(vix_feats)

    # 2. Event intensity features (GDELT)
    intensity_feats = _build_intensity_features(master, rolling_windows)
    df = df.join(intensity_feats)

    # 3. Trend spike features (Google Trends)
    trend_feats = _build_trend_features(master, rolling_windows)
    df = df.join(trend_feats)

    # 4. Event history features (from events.csv)
    event_feats = _build_event_features(master["date"], events, event_windows)
    df = df.join(event_feats)

    # 5. Calendar features
    df["day_of_week"] = master["date"].dt.dayofweek
    df["month"] = master["date"].dt.month

    # Fill NaN from early rows
    df = df.fillna(0)

    n_feat = len(df.columns) - 1
    logger.info(f"Feature matrix: {len(df)} rows × {n_feat} features")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """All feature columns (everything except date)."""
    return [c for c in df.columns if c != "date"]


# ═══════════════════════════════════════════════════════════════════════════
# VIX FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def _build_vix_features(master: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Features from vix + your 3 pre-computed VIX columns."""
    logger.info(f"  VIX features over windows {windows}")
    out = pd.DataFrame(index=master.index)
    vix = master["vix"].astype(float)

    # Raw inputs (you already computed these)
    out["vix"] = vix.values
    for col in ["vix_average_prev3d", "vix_average_prev7d", "vix_slope_20d"]:
        if col in master.columns:
            out[col] = master[col].astype(float).values

    # Rolling features on VIX
    for w in windows:
        tag = f"vix_{w}d"
        out[f"{tag}_avg"] = vix.rolling(w, min_periods=1).mean().values
        out[f"{tag}_max"] = vix.rolling(w, min_periods=1).max().values
        out[f"{tag}_min"] = vix.rolling(w, min_periods=1).min().values
        out[f"{tag}_std"] = vix.rolling(w, min_periods=2).std().fillna(0).values

    # VIX percentile vs 60d (is current VIX high or low historically?)
    out["vix_60d_pctile"] = _rolling_percentile(vix, 60)

    # VIX percentile vs 252d (1 year)
    out["vix_252d_pctile"] = _rolling_percentile(vix, 252)

    # VIX z-score vs 60d
    avg60 = vix.rolling(60, min_periods=10).mean()
    std60 = vix.rolling(60, min_periods=10).std()
    out["vix_60d_zscore"] = ((vix - avg60) / (std60 + 1e-8)).values

    # Slope over 20d (you provide this, but we also compute over other windows)
    out["vix_5d_slope"] = _rolling_slope(vix, 5)

    # VIX acceleration: 5d avg vs 20d avg
    avg5 = vix.rolling(5, min_periods=1).mean()
    avg20 = vix.rolling(20, min_periods=1).mean()
    out["vix_5d_vs_20d"] = (avg5 / (avg20 + 1e-8)).values

    # VIX spike flag: today > 1.5 * 20d avg
    out["vix_spike"] = (vix > 1.5 * avg20).astype(float).values

    return out


# ═══════════════════════════════════════════════════════════════════════════
# EVENT INTENSITY FEATURES (GDELT)
# ═══════════════════════════════════════════════════════════════════════════

def _build_intensity_features(master: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Features from daily GDELT event_intensity."""
    logger.info(f"  Intensity features over windows {windows}")
    out = pd.DataFrame(index=master.index)

    # Use yesterday's intensity (shift 1 to avoid leakage)
    intensity = master["event_intensity"].astype(float).shift(1).fillna(0)

    out["intensity"] = intensity.values

    for w in windows:
        tag = f"intensity_{w}d"
        out[f"{tag}_avg"] = intensity.rolling(w, min_periods=1).mean().values
        out[f"{tag}_max"] = intensity.rolling(w, min_periods=1).max().values
        out[f"{tag}_std"] = intensity.rolling(w, min_periods=2).std().fillna(0).values

    # Intensity percentile vs 60d
    out["intensity_60d_pctile"] = _rolling_percentile(intensity, 60)

    # Intensity surge: today vs 20d avg
    avg20 = intensity.rolling(20, min_periods=1).mean()
    out["intensity_surge"] = (intensity / (avg20 + 1e-8)).values

    # Intensity slope over 20d
    out["intensity_20d_slope"] = _rolling_slope(intensity, 20)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# TREND FEATURES (Google Trends)
# ═══════════════════════════════════════════════════════════════════════════

def _build_trend_features(master: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Features from weekly Google Trends trend_spike."""
    logger.info(f"  Trend features over windows {windows}")
    out = pd.DataFrame(index=master.index)
    trend = master["trend_spike"].astype(float)

    out["trend_spike"] = trend.values

    for w in windows:
        tag = f"trend_{w}d"
        out[f"{tag}_avg"] = trend.rolling(w, min_periods=1).mean().values
        out[f"{tag}_max"] = trend.rolling(w, min_periods=1).max().values

    # Trend acceleration: 5d avg vs 20d avg
    avg5 = trend.rolling(5, min_periods=1).mean()
    avg20 = trend.rolling(20, min_periods=1).mean()
    out["trend_5d_vs_20d"] = (avg5 / (avg20 + 1e-8)).values

    # Trend percentile vs 60d
    out["trend_60d_pctile"] = _rolling_percentile(trend, 60)

    # Trend slope
    out["trend_20d_slope"] = _rolling_slope(trend, 20)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# EVENT HISTORY FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def _build_event_features(
    dates: pd.Series,
    events: pd.DataFrame,
    windows: List[int],
) -> pd.DataFrame:
    """Features from events.csv (event_start/end/type/severity). Past only."""
    logger.info(f"  Event history features over windows {windows}")
    out = pd.DataFrame(index=dates.index)
    dates_arr = dates.values

    # Remap types
    ev = events.copy()
    ev["event_type"] = ev["event_type"].map(TYPE_MERGE).fillna("crisis")

    ev_starts = ev["event_start"].values
    ev_ends = ev["event_end"].values
    ev_types = ev["event_type"].values
    ev_sevs = ev["severity"].values

    # Sort by start
    sort_idx = np.argsort(ev_starts)
    ev_starts = ev_starts[sort_idx]
    ev_ends = ev_ends[sort_idx]
    ev_types = ev_types[sort_idx]
    ev_sevs = ev_sevs[sort_idx]

    # ── Days since last event (any + by type) ──
    dsl_any = np.full(len(dates_arr), 999.0)
    dsl_type = {et: np.full(len(dates_arr), 999.0) for et in EVENT_TYPES}

    for i, d in enumerate(dates_arr):
        # Events that started before today
        past = ev_starts[ev_starts < d]
        if len(past) > 0:
            dsl_any[i] = (d - past[-1]) / np.timedelta64(1, "D")
        for et in EVENT_TYPES:
            mask = (ev_starts < d) & (ev_types == et)
            tp = ev_starts[mask]
            if len(tp) > 0:
                dsl_type[et][i] = (d - tp[-1]) / np.timedelta64(1, "D")

    out["days_since_event"] = dsl_any
    for et in EVENT_TYPES:
        out[f"days_since_{et}"] = dsl_type[et]

    # ── Are we currently inside an active event? ──
    in_event = np.zeros(len(dates_arr))
    in_event_sev = np.zeros(len(dates_arr))
    for i, d in enumerate(dates_arr):
        active = (ev_starts <= d) & (ev_ends >= d)
        if active.any():
            in_event[i] = 1
            in_event_sev[i] = ev_sevs[active].max()
    out["in_active_event"] = in_event
    out["active_event_severity"] = in_event_sev

    # ── Window-based counts and severity ──
    for w in windows:
        total_cnt = np.zeros(len(dates_arr))
        sev_sum = np.zeros(len(dates_arr))
        sev_max = np.zeros(len(dates_arr))
        type_cnt = {et: np.zeros(len(dates_arr)) for et in EVENT_TYPES}

        for i, d in enumerate(dates_arr):
            cutoff = d - np.timedelta64(w, "D")
            mask = (ev_starts > cutoff) & (ev_starts < d)
            if mask.any():
                total_cnt[i] = mask.sum()
                sev_sum[i] = ev_sevs[mask].sum()
                sev_max[i] = ev_sevs[mask].max()
                for et in EVENT_TYPES:
                    type_cnt[et][i] = (mask & (ev_types == et)).sum()

        out[f"events_{w}d"] = total_cnt
        out[f"severity_sum_{w}d"] = sev_sum
        out[f"severity_max_{w}d"] = sev_max
        for et in EVENT_TYPES:
            out[f"events_{et}_{w}d"] = type_cnt[et]

    # ── Severity EWM ──
    out["severity_ewm"] = _severity_ewm(dates_arr, ev_starts, ev_sevs)

    logger.info(f"  Created {len(out.columns)} event features")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _rolling_slope(series: pd.Series, window: int) -> np.ndarray:
    """Rolling linear regression slope."""
    result = np.zeros(len(series))
    values = series.values
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        if len(chunk) < 3:
            continue
        x = np.arange(len(chunk), dtype=float)
        xm, ym = x.mean(), np.nanmean(chunk)
        denom = ((x - xm) ** 2).sum()
        if denom > 0:
            result[i] = ((x - xm) * (chunk - ym)).sum() / denom
    return result


def _rolling_percentile(series: pd.Series, window: int) -> np.ndarray:
    """Percentile of current value within rolling window (0-1)."""
    result = np.zeros(len(series))
    values = series.values
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) < 2:
            result[i] = 0.5
        else:
            result[i] = (valid < values[i]).sum() / len(valid)
    return result


def _severity_ewm(dates, ev_dates, ev_sevs, halflife=14):
    """Exponentially weighted severity (past only)."""
    if len(ev_dates) == 0:
        return np.zeros(len(dates))
    all_d = pd.date_range(dates.min(), dates.max(), freq="D")
    daily = pd.Series(0.0, index=all_d)
    for d, s in zip(ev_dates, ev_sevs):
        ts = pd.Timestamp(d)
        if ts in daily.index:
            daily.loc[ts] += s
    ewm = daily.ewm(halflife=halflife).mean()

    result = np.zeros(len(dates))
    for i, d in enumerate(dates):
        prev = pd.Timestamp(d) - pd.Timedelta(days=1)
        if prev in ewm.index:
            result[i] = ewm.loc[prev]
    return result
