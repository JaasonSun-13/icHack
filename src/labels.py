# src/labels.py
"""Label construction for risk, event type, and social volatility.

Labels use FUTURE data by design (they are what we predict).

Uses event_start/event_end from events.csv and event_intensity/trend_spike
from master.csv.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def create_all_labels(
    master: pd.DataFrame,
    events: pd.DataFrame,
    horizons: List[int] = [7, 14, 30],
    attention_weight: float = 0.70,
    market_weight: float = 0.30,
    **kwargs,
) -> pd.DataFrame:
    """Create labels for every horizon.

    For each day t and horizon k:
      risk_kd        : 1 if any event active in (t+1, t+k], else 0
      event_type_kd  : dominant type (highest severity) in (t+1, t+k], or NaN
      social_vol_kd  : 0.70 * attention_pct + 0.30 * vix_stress_pct
    """
    logger.info("Creating labels ...")
    result = pd.DataFrame({"date": master["date"]})

    # Remap event types to 3 classes
    from src.features import TYPE_MERGE
    ev = events.copy()
    ev["event_type"] = ev["event_type"].map(TYPE_MERGE).fillna("crisis")

    dates = master["date"].values
    intensity = master["event_intensity"].values if "event_intensity" in master.columns else np.zeros(len(master))
    vix = master["vix"].values if "vix" in master.columns else np.zeros(len(master))

    ev_starts = ev["event_start"].values
    ev_ends = ev["event_end"].values
    ev_types = ev["event_type"].values
    ev_sevs = ev["severity"].values

    for k in horizons:
        logger.info(f"  Horizon {k}d ...")

        risk, etype = _risk_and_type(dates, ev_starts, ev_ends, ev_types, ev_sevs, k)
        svol = _social_vol(dates, intensity, vix, k, attention_weight, market_weight)

        result[f"risk_{k}d"] = risk
        result[f"event_type_{k}d"] = etype
        result[f"social_vol_{k}d"] = svol

        pos = np.nansum(risk)
        logger.info(f"    risk: {pos:.0f}/{len(risk)} positive ({pos/len(risk):.1%})")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# INTERNALS
# ═══════════════════════════════════════════════════════════════════════════

def _risk_and_type(dates, ev_starts, ev_ends, ev_types, ev_sevs, k):
    """Binary risk + dominant event type for horizon k.

    An event is 'in window' if any day in (t+1, t+k] falls within
    [event_start, event_end].
    """
    n = len(dates)
    risk = np.zeros(n, dtype=int)
    etype = [np.nan] * n

    for i in range(n):
        d = dates[i]
        window_start = d + np.timedelta64(1, "D")
        window_end = d + np.timedelta64(k, "D")

        # Events that overlap with our forward window
        # overlap = event_start <= window_end AND event_end >= window_start
        overlap = (ev_starts <= window_end) & (ev_ends >= window_start)

        if overlap.any():
            risk[i] = 1
            # Pick highest severity event
            idx = np.where(overlap)[0]
            best = idx[np.argmax(ev_sevs[idx])]
            etype[i] = ev_types[best]

    return risk, etype


def _social_vol(dates, intensity, vix, k, aw, mw):
    """Social volatility label.

    attention = max event_intensity in forward window
    market    = max VIX change in forward window
    Both converted to percentiles, then blended.
    """
    n = len(dates)
    att_raw = np.full(n, np.nan)
    mkt_raw = np.full(n, np.nan)

    for i in range(n):
        fut = list(range(i + 1, min(i + k + 1, n)))
        if not fut:
            continue
        # Attention: max intensity in forward window
        fut_int = [intensity[j] for j in fut if not np.isnan(intensity[j])]
        if fut_int:
            att_raw[i] = max(fut_int)
        # Market stress: max VIX in forward window minus current
        fut_vix = [vix[j] for j in fut if not np.isnan(vix[j])]
        if fut_vix:
            mkt_raw[i] = max(fut_vix) - vix[i]

    # Convert to percentiles (global, will be re-done per fold if needed)
    valid_att = att_raw[~np.isnan(att_raw)]
    valid_mkt = mkt_raw[~np.isnan(mkt_raw)]

    out = np.full(n, np.nan)
    for i in range(n):
        a, m = att_raw[i], mkt_raw[i]
        if np.isnan(a) or np.isnan(m):
            continue
        a_pct = stats.percentileofscore(valid_att, a) / 100.0 if len(valid_att) else 0
        m_pct = stats.percentileofscore(valid_mkt, m) / 100.0 if len(valid_mkt) else 0
        out[i] = np.clip(aw * a_pct + mw * m_pct, 0, 1)

    return out
