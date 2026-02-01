#!/usr/bin/env python3
"""SVF v3 Streamlit Dashboard — Unified Model."""
import glob, os, sys
import numpy as np, pandas as pd

try:
    import streamlit as st
except ImportError:
    print("pip install streamlit"); sys.exit(1)

from src.utils import load_config, load_pickle
from src.data import load_master, load_events, compute_severity
from src.features import build_all_features, get_feature_columns
from src.models import blend_predictions
from src.policy import apply_policy, format_alert

st.set_page_config(page_title="SVF v3", layout="wide")
st.title("🌐 Social Volatility Forecaster v3")

cfg = load_config()
art = cfg["output"]["artifacts_dir"]

@st.cache_data
def load_base():
    m = load_master(cfg["data"]["master_csv"])
    e = load_events(cfg["data"]["events_csv"])
    e = compute_severity(m, e)
    return m, e

master, events = load_base()

# Sidebar: new data point
st.sidebar.header("📊 New Data")
new_spx = st.sidebar.number_input("SPX", value=float(master["spx_close"].iloc[-1]), step=10.0)
new_iv = st.sidebar.number_input("IV", value=float(master["30_day_implied_vol"].iloc[-1]), step=0.5)
new_rv = st.sidebar.number_input("RV 1d", value=float(master["1_day_vol"].iloc[-1]), step=0.001, format="%.4f")
new_int = st.sidebar.number_input("Intensity", value=0.0, step=0.5)
new_spike = st.sidebar.number_input("Trend Spike", value=0.0, step=1.0)
add_new = st.sidebar.button("🔮 Predict")

if add_new:
    nd = master["date"].max() + pd.tseries.offsets.BDay(1)
    master = pd.concat([master, pd.DataFrame([{
        "date": nd, "spx_close": new_spx, "1_day_vol": new_rv,
        "3_day_avg_vol": new_rv, "30_day_implied_vol": new_iv,
        "event_intensity": new_int, "trend_spike": new_spike,
    }])], ignore_index=True)

feat = build_all_features(master, events, cfg=cfg)
fc = load_pickle(f"{art}/feature_cols.pkl")
horizons = load_pickle(f"{art}/horizons.pkl")
base_cols = [c for c in fc if c != "horizon"]

matches = sorted(glob.glob(f"{art}/fold*_single.pkl"))
if not matches:
    st.error("No models found."); st.stop()
folds = sorted(set(int(os.path.basename(m).split("fold")[1].split("_")[0]) for m in matches))
fold = st.sidebar.selectbox("Model", folds, index=len(folds)-1,
    format_func=lambda x: f"Fold {x} (Prod)" if x == folds[-1] else f"Fold {x}")

prefix = f"{art}/fold{fold}"
single = load_pickle(f"{prefix}_single.pkl")
ens = load_pickle(f"{prefix}_ensemble.pkl")
cal = load_pickle(f"{prefix}_calibrator.pkl")
tm = load_pickle(f"{prefix}_type.pkl")
sm = load_pickle(f"{prefix}_svol.pkl")

bcfg = cfg.get("model", {}).get("blend_weights", {})
pcfg = cfg["policy"]
type_map = {0: "crisis", 1: "geopolitics", 2: "macro"}

# Run predictions for all dates × all horizons
results = []
X_base = feat[base_cols].values

for h in horizons:
    X_h = np.hstack([X_base, np.full((len(X_base), 1), h)])
    risk_raw = blend_predictions(ens.predict(X_h), single.predict(X_h),
                                  bcfg.get("ensemble", 0.6), bcfg.get("single", 0.4))
    risk_cal = cal.transform(risk_raw)
    sv = sm.predict(X_h) if sm.model else np.full(len(X_h), 0.5)
    if tm.model:
        tp = tm.predict(X_h)
        et = [type_map[np.argmax(tp[i])] for i in range(len(tp))] if tp.ndim > 1 else [type_map[int(t)] for t in tp]
    else:
        et = ["crisis"] * len(X_h)
    for i in range(len(X_base)):
        results.append({"date": feat["date"].iloc[i], "horizon": h,
                        "risk": risk_cal[i], "svol": sv[i], "type": et[i]})

rdf = pd.DataFrame(results)

# Enforce monotonicity per date
for d in rdf["date"].unique():
    mask = rdf["date"] == d
    vals = rdf.loc[mask].sort_values("horizon")["risk"].values
    for j in range(1, len(vals)):
        if vals[j] < vals[j-1]:
            vals[j] = vals[j-1]
    rdf.loc[mask, "risk"] = np.sort(rdf.loc[mask, "risk"].values)

# Latest date display
latest_date = rdf["date"].max()
latest = rdf[rdf["date"] == latest_date].sort_values("horizon")

if add_new:
    st.success(f"🆕 New prediction: {latest_date.strftime('%Y-%m-%d')}")

st.divider()
st.subheader(f"📅 {latest_date.strftime('%Y-%m-%d')}")

cols = st.columns(len(horizons))
for i, (_, row) in enumerate(latest.iterrows()):
    h = int(row["horizon"])
    alerts, recs = apply_policy(
        np.array([row["risk"]]), np.array([row["svol"]]), [row["type"]],
        pcfg["risk_high"], pcfg["risk_medium"],
        pcfg["svol_high"], pcfg["svol_medium"],
        pcfg.get("boost_types"))
    with cols[i]:
        st.metric(f"{h}d Alert", format_alert(alerts[0]))
        st.metric(f"{h}d Risk", f"{row['risk']:.1%}")
        st.metric(f"{h}d SVol", f"{row['svol']:.1%}")
        st.caption(f"Type: {row['type']}")

mono = latest["risk"].values
st.success(f"✅ Monotonic: {mono[0]:.1%} ≤ {mono[1]:.1%} ≤ {mono[2]:.1%}")

# Charts
st.divider()
h_chart = st.selectbox("Chart horizon", horizons, index=0)
chart_df = rdf[rdf["horizon"] == h_chart].set_index("date").tail(252)
st.subheader(f"Risk & SVol ({h_chart}d)")
st.line_chart(chart_df[["risk", "svol"]])

st.divider()
st.subheader("Recent (all horizons)")
recent = rdf[rdf["date"] >= rdf["date"].max() - pd.Timedelta(days=30)].copy()
recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
recent["risk"] = recent["risk"].apply(lambda x: f"{x:.1%}")
recent["svol"] = recent["svol"].apply(lambda x: f"{x:.1%}")
st.dataframe(recent.sort_values(["date", "horizon"], ascending=[False, True]),
             use_container_width=True, hide_index=True)

st.divider()
try:
    st.subheader("Feature Importance")
    imp = pd.read_csv(f"{art}/feature_importance.csv")
    st.bar_chart(imp.head(20).set_index("feature")["importance"])
except: pass

try:
    st.subheader("Metrics")
    st.dataframe(pd.read_csv(f"{art}/metrics.csv"), use_container_width=True, hide_index=True)
except: pass
