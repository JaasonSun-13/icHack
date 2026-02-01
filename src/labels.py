"""Labels: risk (binary), event_type (3-class), social_vol (0-1).

Social vol = 0.40 * pct(attention) + 0.35 * pct(market_move) + 0.25 * pct(iv_jump)

Supports both:
  - Per-horizon columns (risk_7d, risk_14d, risk_21d, ...)
  - Unified long format (one row per date×horizon, horizon as feature)
"""
import logging
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)


def create_all_labels(master, events, min_severity=4, horizons=[7, 14, 21],
                      social_vol_weights=None, **kw):
    """Create 3 labels per horizon: risk, event_type, social_vol."""
    if social_vol_weights is None:
        social_vol_weights = {"attention": 0.40, "market_move": 0.35, "iv_jump": 0.25}
    w_att = social_vol_weights.get("attention", 0.40)
    w_mkt = social_vol_weights.get("market_move", 0.35)
    w_ivj = social_vol_weights.get("iv_jump", 0.25)

    ev = events[events["severity"] >= min_severity].copy()
    from src.features import TYPE_MERGE
    ev["event_type"] = ev["event_type"].map(TYPE_MERGE).fillna("crisis")
    ev = ev[ev["event_type"] != "other"].reset_index(drop=True)

    es = ev["event_start"].values
    ee = ev["event_end"].values
    et = ev["event_type"].values
    sv = ev["severity"].values

    dates = master["date"].values
    intensity = master["event_intensity"].astype(float).values
    spx = master["spx_close"].astype(float).values
    iv = master["30_day_implied_vol"].astype(float).values

    log_ret = np.zeros(len(spx))
    log_ret[1:] = np.abs(np.log(spx[1:] / spx[:-1]))

    result = pd.DataFrame({"date": master["date"]})

    for h in horizons:
        risk = np.zeros(len(dates))
        etype = [np.nan] * len(dates)
        att_raw = np.zeros(len(dates))
        mkt_raw = np.zeros(len(dates))
        ivj_raw = np.zeros(len(dates))

        for i in range(len(dates)):
            d = dates[i]
            win_start = d + np.timedelta64(1, "D")
            win_end = d + np.timedelta64(h, "D")

            overlap = (es <= win_end) & (ee >= win_start)
            if overlap.any():
                risk[i] = 1
                best = np.argmax(sv[overlap])
                etype[i] = et[overlap][best]

            fwd = (dates > d) & (dates <= win_end)
            fwd_idx = np.where(fwd)[0]
            if len(fwd_idx) > 0:
                att_raw[i] = intensity[fwd_idx].max()
                mkt_raw[i] = log_ret[fwd_idx].max()
                ivj_raw[i] = iv[fwd_idx].max() - iv[i]
            else:
                att_raw[i] = np.nan
                mkt_raw[i] = np.nan
                ivj_raw[i] = np.nan

        att_pct = _percentile_transform(att_raw)
        mkt_pct = _percentile_transform(mkt_raw)
        ivj_pct = _percentile_transform(ivj_raw)
        svol = np.clip(w_att * att_pct + w_mkt * mkt_pct + w_ivj * ivj_pct, 0, 1)

        result[f"risk_{h}d"] = risk
        result[f"event_type_{h}d"] = etype
        result[f"social_vol_{h}d"] = svol
        pos = int(risk.sum())
        logger.info(f"  {h}d: risk=1 on {pos}/{len(risk)} ({100*pos/len(risk):.1f}%)")

    return result


def stack_unified(features_df, labels_df, horizons=[7, 14, 21]):
    """Stack into unified long format: each date appears len(horizons) times.

    Adds `horizon` column as a feature. LightGBM monotone_constraints
    on this column enforces risk_7d <= risk_14d <= risk_21d.

    Returns stacked DataFrame with columns:
      [features..., horizon, risk, event_type, social_vol, date]
    """
    feature_cols = [c for c in features_df.columns if c != "date"]
    frames = []
    for h in horizons:
        chunk = features_df.copy()
        chunk["horizon"] = h
        chunk["risk"] = labels_df[f"risk_{h}d"].values
        chunk["event_type"] = labels_df[f"event_type_{h}d"].values
        chunk["social_vol"] = labels_df[f"social_vol_{h}d"].values
        frames.append(chunk)
    stacked = pd.concat(frames, ignore_index=True)
    stacked = stacked.sort_values(["date", "horizon"]).reset_index(drop=True)
    logger.info(f"Unified stacked: {stacked.shape[0]} rows "
                f"({len(features_df)} days × {len(horizons)} horizons)")
    return stacked


def _percentile_transform(arr):
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return np.full_like(arr, 0.5)
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            out[i] = 0.5
        else:
            out[i] = (valid < arr[i]).sum() / len(valid)
    return out
