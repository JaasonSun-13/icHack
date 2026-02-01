# Social Volatility Forecaster v3

An early warning system that predicts social/market disruption using VIX, GDELT news intensity, Google Trends, and historical events.

## What It Does

Every day it answers three questions:

| Output | Question | Range |
|--------|----------|-------|
| **Risk Probability** | Will a significant event occur in the next k days? | 0–100% |
| **Event Type** | If so, what kind? | geopolitics, macro, crisis |
| **Social Volatility** | How chaotic will it be? | 0 (calm) to 1 (crisis) |

Predictions are made at three horizons: **7 days**, **14 days**, and **30 days**.

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Add your data
#    data/master.csv   (daily market + GDELT + Google Trends)
#    data/events.csv   (historical events)

# 3. Train
python train.py

# 4. Predict (latest day)
python predict.py --latest

# 5. Evaluate
python evaluate.py --by_fold

# 6. Dashboard
streamlit run app.py

# 7. Predict a next day
python predict.py --spx 5200 --iv 35 --intensity 10 --spike 30
```

---

## Your Input Files

### `data/master.csv` — 7 columns, daily

| Column | Source | Description |
|--------|--------|-------------|
| date | — | Trading date (YYYY-MM-DD) |
| 1_day_vol | CBOE | VIX closing value |
| 3_day_avg_vol | You compute | Mean VIX over prior 3 trading days |
| 30_day_implied_vol | You compute | Mean VIX over prior 7 trading days |
| spx_close | You compute | Linear regression slope of VIX over 20 days |
| event_intensity = average daily proportion (e.g. 3.9% of all news)
| trend_spike = proportion × duration (e.g. 3.9% × 30 days = 117) 



```csv
date,vix,vix_average_prev3d,vix_average_prev7d,vix_slope_20d,event_intensity,trend_spike
2015-01-02,14.2,14.5,15.1,-0.12,5.3,18
```

### `data/events.csv` — 4 columns

| Column | Description |
|--------|-------------|
| event_start | First date of the event |
| event_end | Last date of the event |
| event_type | Category (geopolitics, macro, political, corporate, security, etc.) |
| headline | Short description |

```csv
event_start,event_end,event_type,headline
2020-02-24,2020-04-30,macro,COVID pandemic crash
2022-02-24,2022-04-15,geopolitics,Russia invades Ukraine
```

**Severity is auto-computed** from your data — you don't assign it. See Formulas below.

---

## How It Works

1. **Load**: Reads your 7-column master.csv + events.csv
2. **Auto-severity**: Computes event severity (1–5) from peak VIX and GDELT intensity during each event
3. **Features**: Creates 66 features from rolling windows, percentiles, slopes, and event history (all past data only)
4. **Labels**: Creates 3 targets per horizon from future data (risk, event type, social volatility)
5. **Walk-forward validation**: Trains on past years, validates on the next year (no leakage)
6. **LightGBM ensemble**: 3-model stack + single model, blended 60/40, with isotonic calibration
7. **Policy tree**: Converts predictions to 🟢 LOW / 🟡 MEDIUM / 🔴 HIGH alerts with recommendations

---

## Formulas

### Auto-Severity (per event)

```
peak_intensity  = max(event_intensity) during [event_start, event_end]
peak_vix        = max(vix) during [event_start, event_end]
intensity_pct   = percentile_rank(peak_intensity vs ALL daily values)
vix_pct         = percentile_rank(peak_vix vs ALL daily values)
raw_score       = 0.5 × intensity_pct + 0.5 × vix_pct
severity        = ceil(raw_score × 5), clipped to [1, 5]
```

COVID crash → peak VIX ~65 (99th pct), peak GDELT ~55 (99th pct) → severity **5**
Routine Fed meeting → VIX ~16 (25th pct), GDELT ~8 (30th pct) → severity **2**

### Event Type Mapping

| Raw types in your events.csv | → Merged class |
|------------------------------|---------------|
| geopolitics, political, government | **geopolitics** |
| macro, economic, finance | **macro** |
| tech, corporate, security, health, other | **crisis** |

### Features (66 total)

**VIX features (22)**: Raw VIX + 3 pre-computed columns + rolling avg/max/min/std over 5d/20d/60d + 60d & 252d percentiles + 60d z-score + 5d slope + 5d-vs-20d acceleration ratio + spike flag (VIX > 1.5 × 20d avg)

**Event intensity features (13)**: Yesterday's intensity (lagged 1d) + rolling avg/max/std over 5d/20d/60d + 60d percentile + surge ratio (today / 20d avg) + 20d slope

**Trend features (10)**: Raw trend_spike + rolling avg/max over 5d/20d/60d + 5d-vs-20d acceleration + 60d percentile + 20d slope

**Event history features (19)**: Days since last event (any + per type) + currently inside active event flag + active event max severity + event counts in 30d/90d + severity sum/max in 30d/90d + per-type counts in 30d/90d + exponentially weighted severity (halflife 14d)

**Calendar features (2)**: Day of week + month

### Labels

**Risk** (binary):
```
risk = 1 if ANY event overlaps with (t+1, t+k], else 0
overlap = event_start ≤ window_end AND event_end ≥ window_start
```

**Event type** (3-class):
```
If risk = 1: type of the highest-severity event overlapping the window
If risk = 0: NaN (no label)
```

**Social volatility** (continuous 0–1):
```
attention  = max(event_intensity) in forward window [t+1, t+k]
market     = max(VIX in [t+1, t+k]) − VIX at t
social_vol = 0.40 × percentile(attention) + 0.60 × percentile(market)
```

### Policy Tree

```
risk_prob ≥ 0.7 OR social_vol ≥ 0.7         → 🔴 HIGH
risk_prob ≥ 0.4 OR social_vol ≥ 0.4         → 🟡 MEDIUM
otherwise                                    → 🟢 LOW

event_type ∈ {geopolitics, macro} AND MEDIUM → boost to 🔴 HIGH
```

---

## Walk-Forward Validation

```
Fold 0:  Train [2015–2021]  →  Validate 2022
Fold 1:  Train [2015–2022]  →  Validate 2023
Fold 2:  Train [2015–2023]  →  Validate 2024
Fold 3:  Train [2015–2024]  →  Validate 2025
Final:   Train [2015–2025]  →  Production model
```

## Project Structure

```
├── configs/default.yaml     # All hyperparameters and paths
├── src/
│   ├── data.py              # Load CSV + auto-compute severity
│   ├── features.py          # 66 features from 7 raw inputs
│   ├── labels.py            # Risk, event type, social volatility labels
│   ├── splits.py            # Walk-forward expanding window splits
│   ├── models.py            # LightGBM + 3-model ensemble stack
│   ├── calibration.py       # Isotonic probability calibration
│   ├── metrics.py           # AUROC, F1, RMSE, Spearman
│   ├── policy.py            # Alert decision tree + recommendations
│   └── utils.py             # Logging, I/O helpers
├── train.py                 # Full training pipeline
├── predict.py               # Generate predictions + pretty-print
├── evaluate.py              # Evaluate predictions per fold
├── app.py                   # Streamlit dashboard
└── data/
    ├── master.csv           # Your daily data (7 columns)
    └── events.csv           # Your events (4 columns)
```

