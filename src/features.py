"""Exactly 66 backward-looking features from 7 raw columns.

Groups (66 total):
  Market & Volatility (30): IV=15, RV=6, SPX=6, Cross=3
  Event Intensity    (13): lag-1 + rolling + percentile + surge + slope
  Trend / Attention  (10): raw + rolling + accel + percentile + slope
  Event History      (11): days-since + active + counts + severity + ewm
  Calendar            (2): day_of_week + month
"""
import logging
from typing import List
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)

EVENT_TYPES = ["crisis", "geopolitics", "macro"]


def _build_type_merge(cfg=None):
    """Build event type mapping from config or defaults."""
    if cfg and "event_type_mapping" in cfg:
        m = {}
        for target, sources in cfg["event_type_mapping"].items():
            for s in sources:
                m[s] = target
        return m
    return {
        "geopolitics": "geopolitics", "political": "geopolitics",
        "government": "geopolitics", "war": "geopolitics",
        "election": "geopolitics", "sanctions": "geopolitics",
        "Politics & Government": "geopolitics",
        "War, Geopolitics & Terrorism": "geopolitics",
        "macro": "macro", "economic": "macro", "finance": "macro",
        "rates": "macro", "inflation": "macro", "recession": "macro",
        "Economy & Finance": "macro", "Crypto & Digital Assets": "macro",
        "tech": "crisis", "corporate": "crisis", "security": "crisis",
        "health": "crisis", "disaster": "crisis", "other": "crisis",
        "crisis": "crisis",
        "General / Breaking News": "crisis", "Public Health & Disease": "crisis",
        "Natural Disasters & Climate": "crisis", "Technology & AI": "crisis",
    }

TYPE_MERGE = _build_type_merge()


def build_all_features(master, events, cfg=None, **kw):
    """Build exactly 66 features. Asserts count."""
    global TYPE_MERGE
    if cfg:
        TYPE_MERGE = _build_type_merge(cfg)

    logger.info("Building features ...")
    df = master[["date"]].copy()
    df = df.join(_implied_vol(master))       # 15
    df = df.join(_realized_vol(master))      # 6
    df = df.join(_spx(master))               # 6
    df = df.join(_cross(master))             # 3
    df = df.join(_intensity(master))         # 13
    df = df.join(_trend(master))             # 10
    df = df.join(_event_hist(master["date"], events))  # 11
    df["day_of_week"] = master["date"].dt.dayofweek    # 1
    df["month"] = master["date"].dt.month              # 1
    df = df.fillna(0)

    fcols = get_feature_columns(df)
    n = len(fcols)
    assert n == 66, f"Expected 66 features, got {n}: {fcols}"
    logger.info(f"Features: {n} ✓")
    return df


def get_feature_columns(df):
    return [c for c in df.columns if c != "date"]


# ── Implied Vol (15) ─────────────────────────────────────────────────────
def _implied_vol(m):
    out = pd.DataFrame(index=m.index)
    iv = m["30_day_implied_vol"].astype(float)
    out["iv"] = iv.values
    for w in [5, 20, 60]:
        out[f"iv_{w}d_avg"] = iv.rolling(w, min_periods=1).mean().values
        out[f"iv_{w}d_max"] = iv.rolling(w, min_periods=1).max().values
        out[f"iv_{w}d_std"] = iv.rolling(w, min_periods=2).std().fillna(0).values
    out["iv_60d_pctile"] = _pctile(iv, 60)
    a60 = iv.rolling(60, min_periods=10).mean()
    s60 = iv.rolling(60, min_periods=10).std()
    out["iv_60d_zscore"] = ((iv - a60) / (s60 + 1e-8)).values
    out["iv_5d_slope"] = _slope(iv, 5)
    a5 = iv.rolling(5, min_periods=1).mean()
    a20 = iv.rolling(20, min_periods=1).mean()
    out["iv_accel"] = (a5 / (a20 + 1e-8)).values
    out["iv_spike"] = (iv > 1.5 * a20).astype(float).values
    return out  # 1 + 9 + 5 = 15


# ── Realized Vol (6) ────────────────────────────────────────────────────
def _realized_vol(m):
    out = pd.DataFrame(index=m.index)
    rv = m["1_day_vol"].astype(float)
    out["rv_1d"] = rv.values
    for w in [5, 20, 60]:
        out[f"rv_{w}d_avg"] = rv.rolling(w, min_periods=1).mean().values
    out["rv_60d_pctile"] = _pctile(rv, 60)
    a20 = rv.rolling(20, min_periods=1).mean()
    out["rv_surge"] = (rv / (a20 + 1e-8)).values
    return out  # 1 + 3 + 1 + 1 = 6


# ── SPX (6) ─────────────────────────────────────────────────────────────
def _spx(m):
    out = pd.DataFrame(index=m.index)
    px = m["spx_close"].astype(float)
    ret = px.pct_change().fillna(0)
    out["spx_ret"] = ret.values
    for w in [5, 20]:
        out[f"spx_ret_{w}d"] = ret.rolling(w, min_periods=1).sum().values
    rm60 = px.rolling(60, min_periods=1).max()
    out["spx_dd_60d"] = ((px - rm60) / (rm60 + 1e-8)).values
    sma20 = px.rolling(20, min_periods=1).mean()
    out["spx_vs_sma20"] = ((px - sma20) / (sma20 + 1e-8)).values
    out["spx_big_down"] = (ret < -0.02).astype(float).values
    return out  # 1 + 2 + 3 = 6


# ── Cross (3) ────────────────────────────────────────────────────────────
def _cross(m):
    out = pd.DataFrame(index=m.index)
    iv = m["30_day_implied_vol"].astype(float)
    rv_ann = m["1_day_vol"].astype(float) * np.sqrt(252) * 100
    out["vol_premium"] = (iv - rv_ann).values
    out["iv_rv_ratio"] = (iv / (rv_ann + 1e-8)).values
    out["iv_x_intensity"] = (iv * m["event_intensity"].astype(float)).values
    return out  # 3


# ── Event Intensity (13) ────────────────────────────────────────────────
def _intensity(m):
    out = pd.DataFrame(index=m.index)
    ei = m["event_intensity"].astype(float).shift(1).fillna(0)  # lag 1d
    out["intensity"] = ei.values
    for w in [5, 20, 60]:
        out[f"int_{w}d_avg"] = ei.rolling(w, min_periods=1).mean().values
        out[f"int_{w}d_max"] = ei.rolling(w, min_periods=1).max().values
        out[f"int_{w}d_std"] = ei.rolling(w, min_periods=2).std().fillna(0).values
    out["int_60d_pctile"] = _pctile(ei, 60)
    a20 = ei.rolling(20, min_periods=1).mean()
    out["int_surge"] = (ei / (a20 + 1e-8)).values
    out["int_20d_slope"] = _slope(ei, 20)
    return out  # 1 + 9 + 3 = 13


# ── Trend / Attention (10) ──────────────────────────────────────────────
def _trend(m):
    out = pd.DataFrame(index=m.index)
    ts = m["trend_spike"].astype(float)
    out["trend"] = ts.values
    for w in [5, 20, 60]:
        out[f"trend_{w}d_avg"] = ts.rolling(w, min_periods=1).mean().values
        out[f"trend_{w}d_max"] = ts.rolling(w, min_periods=1).max().values
    a5 = ts.rolling(5, min_periods=1).mean()
    a20 = ts.rolling(20, min_periods=1).mean()
    out["trend_accel"] = (a5 / (a20 + 1e-8)).values
    out["trend_60d_pctile"] = _pctile(ts, 60)
    out["trend_20d_slope"] = _slope(ts, 20)
    return out  # 1 + 6 + 3 = 10


# ── Event History (11) ──────────────────────────────────────────────────
def _event_hist(dates, events):
    out = pd.DataFrame(index=dates.index)
    da = dates.values
    ev = events.copy()
    ev["event_type"] = ev["event_type"].map(TYPE_MERGE).fillna("crisis")
    ev = ev[ev["event_type"] != "other"].reset_index(drop=True)

    if len(ev) == 0:
        for c in ["days_since_event"] + [f"days_since_{t}" for t in EVENT_TYPES] + \
                 ["in_active_event", "active_event_sev", "events_30d", "events_90d",
                  "sev_sum_90d", "sev_max_90d", "sev_ewm"]:
            out[c] = 0.0
        return out

    es = ev["event_start"].values
    ee = ev["event_end"].values
    et = ev["event_type"].values
    sv = ev["severity"].values
    si = np.argsort(es)
    es, ee, et, sv = es[si], ee[si], et[si], sv[si]

    dsl = np.full(len(da), 90.0)  # cap at 90
    dsl_t = {t: np.full(len(da), 90.0) for t in EVENT_TYPES}
    in_ev = np.zeros(len(da))
    in_sev = np.zeros(len(da))

    for i, d in enumerate(da):
        past = es[es < d]
        if len(past):
            dsl[i] = min(90, (d - past[-1]) / np.timedelta64(1, "D"))
        for t in EVENT_TYPES:
            tp = es[(es < d) & (et == t)]
            if len(tp):
                dsl_t[t][i] = min(90, (d - tp[-1]) / np.timedelta64(1, "D"))
        act = (es <= d) & (ee >= d)
        if act.any():
            in_ev[i] = 1
            in_sev[i] = sv[act].max()

    out["days_since_event"] = dsl
    for t in EVENT_TYPES:
        out[f"days_since_{t}"] = dsl_t[t]
    out["in_active_event"] = in_ev
    out["active_event_sev"] = in_sev

    # Window counts
    for w in [30, 90]:
        cnt = np.zeros(len(da))
        for i, d in enumerate(da):
            cutoff = d - np.timedelta64(w, "D")
            cnt[i] = ((es > cutoff) & (es < d)).sum()
        out[f"events_{w}d"] = cnt

    # Severity stats (90d only)
    ssum = np.zeros(len(da))
    smax = np.zeros(len(da))
    for i, d in enumerate(da):
        cutoff = d - np.timedelta64(90, "D")
        mask = (es > cutoff) & (es < d)
        if mask.any():
            ssum[i] = sv[mask].sum()
            smax[i] = sv[mask].max()
    out["sev_sum_90d"] = ssum
    out["sev_max_90d"] = smax

    out["sev_ewm"] = _sev_ewm(da, es, sv)
    logger.info(f"  Event history: {len(out.columns)} features")
    return out  # 4 + 2 + 2 + 2 + 1 = 11


# ── Helpers ──────────────────────────────────────────────────────────────
def _slope(s, w):
    r = np.zeros(len(s))
    v = s.values
    for i in range(len(v)):
        st = max(0, i - w + 1)
        ch = v[st:i + 1]
        if len(ch) < 3:
            continue
        x = np.arange(len(ch), dtype=float)
        xm, ym = x.mean(), np.nanmean(ch)
        d = ((x - xm) ** 2).sum()
        if d > 0:
            r[i] = ((x - xm) * (ch - ym)).sum() / d
    return r


def _pctile(s, w):
    r = np.zeros(len(s))
    v = s.values
    for i in range(len(v)):
        st = max(0, i - w + 1)
        ch = v[st:i + 1]
        vl = ch[~np.isnan(ch)]
        r[i] = (vl < v[i]).sum() / len(vl) if len(vl) >= 2 else 0.5
    return r


def _sev_ewm(dates, ev_dates, ev_sevs, hl=14):
    if len(ev_dates) == 0:
        return np.zeros(len(dates))
    rng = pd.date_range(dates.min(), dates.max(), freq="D")
    ds = pd.Series(0.0, index=rng)
    for d, s in zip(ev_dates, ev_sevs):
        t = pd.Timestamp(d)
        if t in ds.index:
            ds.loc[t] += s
    ew = ds.ewm(halflife=hl).mean()
    r = np.zeros(len(dates))
    for i, d in enumerate(dates):
        p = pd.Timestamp(d) - pd.Timedelta(days=1)
        if p in ew.index:
            r[i] = ew.loc[p]
    return r
