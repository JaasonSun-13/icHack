#!/usr/bin/env python3
"""Streamlit dashboard: streamlit run app.py"""

import json, os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
from src.policy import PolicyTree

st.set_page_config(page_title="Social Volatility Forecaster", page_icon="📊", layout="wide")

# ── Helpers ──────────────────────────────────────────────────────────────────

def gauge(value, title, max_val=1.0, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": title}, number={"suffix": suffix},
        gauge={"axis": {"range": [0, max_val]},
               "bar": {"color": "darkblue"},
               "steps": [
                   {"range": [0, max_val * 0.4], "color": "#90EE90"},
                   {"range": [max_val * 0.4, max_val * 0.7], "color": "#FFD700"},
                   {"range": [max_val * 0.7, max_val], "color": "#FF6B6B"}]}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

ALERT_COLOURS = {"HIGH": "#FF4B4B", "MEDIUM": "#FFA500", "LOW": "#00CC00"}

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.title("📊 Social Volatility Forecaster")

    # Sidebar
    model_dir = st.sidebar.text_input("Model directory", "artifacts/latest")
    pred_path = st.sidebar.text_input("Predictions file", "outputs/predictions.csv")
    horizon = st.sidebar.selectbox("Horizon", [7, 14, 30], format_func=lambda x: f"{x} days")

    # Load predictions
    if not os.path.exists(pred_path):
        st.warning("No predictions found. Run `python train.py` first.")
        return

    df = pd.read_csv(pred_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Load type mapping
    idx_to_type = {}
    tm_path = os.path.join(model_dir, "event_type_mapping.json")
    if os.path.exists(tm_path):
        with open(tm_path) as f:
            m = json.load(f)
            idx_to_type = {int(k): v for k, v in m.get("idx_to_type", {}).items()}

    # Latest row
    latest = df.iloc[-1]
    risk_prob = latest.get(f"risk_{horizon}d_prob", 0)
    social_vol = latest.get(f"social_vol_{horizon}d", 0)
    etype_idx = latest.get(f"event_type_{horizon}d", -1)
    etype_name = idx_to_type.get(int(etype_idx), "unknown") if not pd.isna(etype_idx) else "unknown"

    # Policy
    pt = PolicyTree()
    dec = pt.evaluate(risk_prob, social_vol, etype_name)
    colour = ALERT_COLOURS.get(dec.alert_level, "#808080")

    # Header
    st.markdown(f"### Forecast date: {latest.get('date', 'N/A')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div style="background:{colour};padding:12px;border-radius:6px;'
                f'text-align:center;color:#fff;font-weight:bold;font-size:20px">'
                f'{dec.alert_level}</div>', unsafe_allow_html=True)
    c2.metric(f"Risk ({horizon}d)", f"{risk_prob:.1%}")
    c3.metric(f"Social Vol ({horizon}d)", f"{social_vol:.2f}")
    c4.metric("Event Type", etype_name)

    st.divider()

    # Gauges
    g1, g2 = st.columns(2)
    g1.plotly_chart(gauge(risk_prob * 100, "Risk Probability", 100, "%"), use_container_width=True)
    g2.plotly_chart(gauge(social_vol, "Social Volatility"), use_container_width=True)

    # Time series
    st.subheader("📈 Historical Trends")
    rp_col = f"risk_{horizon}d_prob"
    sv_col = f"social_vol_{horizon}d"
    if rp_col in df.columns and sv_col in df.columns and "date" in df.columns:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Risk Probability", "Social Volatility"))
        fig.add_trace(go.Scatter(x=df["date"], y=df[rp_col], name="Risk"), row=1, col=1)
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df[sv_col], name="SocVol",
                                 line=dict(color="green")), row=2, col=1)
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", row=2, col=1)
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.subheader("💡 Recommendations")
    tabs = st.tabs(["🏦 Investor", "🏢 Firm", "🚀 Entrepreneur"])
    for tab, persona in zip(tabs, ["investor", "firm", "entrepreneur"]):
        with tab:
            st.write(dec.recommendations.get(persona, "No recommendation"))

    # Feature importance
    imp_path = os.path.join(model_dir, "feature_importance.json")
    if os.path.exists(imp_path):
        with open(imp_path) as f:
            imp = json.load(f)
        st.subheader("🔍 Top Feature Drivers")
        fig = go.Figure(go.Bar(
            x=list(imp.values()), y=list(imp.keys()),
            orientation="h", marker_color="#1f77b4"))
        fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"},
                          margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True)

    # Raw data
    with st.expander("📋 Raw Predictions"):
        st.dataframe(df.tail(30))


if __name__ == "__main__":
    main()
