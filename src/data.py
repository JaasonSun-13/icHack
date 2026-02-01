"""Load master.csv + events.csv, validate schema, auto-compute severity."""
import logging
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)

MASTER_COLS = ["date", "spx_close", "1_day_vol", "3_day_avg_vol",
               "30_day_implied_vol", "event_intensity", "trend_spike"]
EVENT_COLS = ["event_start", "event_end", "event_type", "headline"]


def load_master(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"Master: {df.shape[0]} rows, {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def load_events(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["event_start"] = pd.to_datetime(df["event_start"])
    df["event_end"] = pd.to_datetime(df["event_end"])
    logger.info(f"Events: {df.shape[0]} rows")
    return df


def validate(master, events):
    """Check required columns exist."""
    issues = []
    for c in MASTER_COLS:
        if c not in master.columns:
            issues.append(f"master missing: {c}")
    for c in EVENT_COLS:
        if c not in events.columns:
            issues.append(f"events missing: {c}")
    ok = len(issues) == 0
    if not ok:
        logger.error(f"Validation: {issues}")
    return ok, issues


def compute_severity(master, events, iw=0.5, ivw=0.5):
    """Auto-severity 1-5 from peak intensity + peak implied vol during event window."""
    all_iv = master["30_day_implied_vol"].dropna().values
    all_int = master["event_intensity"].dropna().values
    sevs = []
    for _, ev in events.iterrows():
        mask = (master["date"] >= ev["event_start"]) & (master["date"] <= ev["event_end"])
        window = master.loc[mask]
        if len(window) == 0:
            sevs.append(1)
            continue
        peak_iv = window["30_day_implied_vol"].max()
        peak_int = window["event_intensity"].max()
        iv_pct = (all_iv < peak_iv).sum() / len(all_iv) if len(all_iv) > 0 else 0.5
        int_pct = (all_int < peak_int).sum() / len(all_int) if len(all_int) > 0 else 0.5
        raw = iw * int_pct + ivw * iv_pct
        sev = int(np.ceil(raw * 5))
        sevs.append(max(1, min(5, sev)))
    events = events.copy()
    events["severity"] = sevs
    logger.info(f"Severity distribution: {pd.Series(sevs).value_counts().sort_index().to_dict()}")
    return events
