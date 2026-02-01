#!/usr/bin/env python3
"""
generate_example_data.py

Generates realistic example data matching YOUR actual data sources:
  - VIX (daily) + 3 derived VIX cols
  - Google Trends (weekly emerging keywords → grouped into events)
  - GDELT (daily article counts per event/keyword)
  - Events with start/end/intensity/type
"""

import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit

np.random.seed(42)

dates = pd.bdate_range("2015-01-02", "2025-12-31")
n = len(dates)

# ══════════════════════════════════════════════════════════════════════════
# VIX (daily)
# ══════════════════════════════════════════════════════════════════════════

vix = np.full(n, 15.0)
for i in range(1, n):
    vix[i] = vix[i-1] + 0.02 * (15 - vix[i-1]) + np.random.normal(0, 0.8)

crises = [
    ("2015-08-20", "2015-09-10", 28),
    ("2018-02-02", "2018-02-15", 37),
    ("2018-12-01", "2018-12-26", 36),
    ("2020-02-20", "2020-04-15", 65),
    ("2020-10-25", "2020-11-05", 40),
    ("2022-01-20", "2022-03-15", 35),
    ("2022-06-01", "2022-06-20", 33),
    ("2022-09-20", "2022-10-15", 32),
    ("2023-03-08", "2023-03-20", 30),
    ("2024-08-01", "2024-08-10", 38),
    ("2025-04-02", "2025-04-20", 42),
]
for start, end, peak in crises:
    mask = (dates >= start) & (dates <= end)
    idx = np.where(mask)[0]
    if len(idx) > 0:
        mid = len(idx) // 2
        for j, ix in enumerate(idx):
            dist = abs(j - mid) / max(mid, 1)
            vix[ix] = peak * (1 - 0.4 * dist) + np.random.normal(0, 1.5)
vix = np.clip(vix, 9, 82)

# Derived VIX columns
vix_avg_3d = pd.Series(vix).rolling(3).mean().values
vix_avg_7d = pd.Series(vix).rolling(7).mean().values

vix_slope_20d = np.full(n, 0.0)
for i in range(20, n):
    y = vix[i-20:i]
    x = np.arange(20)
    coef = polyfit(x, y, 1)
    vix_slope_20d[i] = coef[1]

# ══════════════════════════════════════════════════════════════════════════
# EVENT_INTENSITY (daily, from GDELT)
# ══════════════════════════════════════════════════════════════════════════

event_intensity = np.random.uniform(2, 8, n)

intensity_spikes = [
    ("2015-08-24", "2015-09-05", 25),
    ("2016-06-23", "2016-07-05", 30),
    ("2016-11-08", "2016-11-20", 28),
    ("2018-02-02", "2018-02-12", 22),
    ("2020-02-24", "2020-04-30", 45),
    ("2020-03-09", "2020-03-20", 55),
    ("2020-11-03", "2020-11-10", 30),
    ("2021-01-06", "2021-01-15", 28),
    ("2021-01-27", "2021-02-05", 20),
    ("2022-02-24", "2022-04-15", 40),
    ("2022-06-10", "2022-06-20", 22),
    ("2022-11-08", "2022-11-20", 30),
    ("2023-03-08", "2023-03-25", 32),
    ("2023-10-07", "2023-11-01", 35),
    ("2024-04-13", "2024-04-20", 25),
    ("2024-08-02", "2024-08-10", 28),
    ("2024-11-05", "2024-11-15", 26),
    ("2025-04-02", "2025-04-25", 38),
]
for start, end, peak in intensity_spikes:
    mask = (dates >= start) & (dates <= end)
    idx = np.where(mask)[0]
    if len(idx) > 0:
        mid = len(idx) // 2
        for j, ix in enumerate(idx):
            dist = abs(j - mid) / max(mid, 1)
            event_intensity[ix] = peak * (1 - 0.3 * dist) + np.random.normal(0, 3)
event_intensity = np.clip(event_intensity, 0, 60)

# ══════════════════════════════════════════════════════════════════════════
# TREND_SPIKE (weekly Google Trends → assigned to Friday → forward-fill)
# ══════════════════════════════════════════════════════════════════════════

fridays = dates[dates.weekday == 4]
trend_weekly = np.random.uniform(5, 25, len(fridays))

trend_spikes = [
    ("2016-06-24", 85),
    ("2020-01-24", 45),
    ("2020-03-06", 95),
    ("2020-03-13", 100),
    ("2021-01-29", 70),
    ("2022-02-25", 90),
    ("2022-06-10", 65),
    ("2022-11-11", 75),
    ("2023-03-10", 80),
    ("2023-10-07", 72),
    ("2024-08-02", 60),
    ("2025-04-04", 88),
]
for spike_date, level in trend_spikes:
    sd = pd.Timestamp(spike_date)
    for fi, fri in enumerate(fridays):
        if abs((fri - sd).days) <= 3:
            trend_weekly[fi] = level
            if fi + 1 < len(fridays):
                trend_weekly[fi + 1] = level * 0.6 + np.random.normal(0, 5)
            if fi + 2 < len(fridays):
                trend_weekly[fi + 2] = level * 0.3 + np.random.normal(0, 5)
            break

trend_weekly = np.clip(trend_weekly, 0, 100)

trend_spike = np.full(n, np.nan)
for i, fri in enumerate(fridays):
    idx = np.where(dates == fri)[0]
    if len(idx) > 0:
        trend_spike[idx[0]] = trend_weekly[i]
trend_spike = pd.Series(trend_spike).ffill().bfill().values

# ══════════════════════════════════════════════════════════════════════════
# MASTER.CSV
# ══════════════════════════════════════════════════════════════════════════

master = pd.DataFrame({
    "date": dates,
    "vix": np.round(vix, 2),
    "vix_average_prev3d": np.round(vix_avg_3d, 2),
    "vix_average_prev7d": np.round(vix_avg_7d, 2),
    "vix_slope_20d": np.round(vix_slope_20d, 4),
    "event_intensity": np.round(event_intensity, 2),
    "trend_spike": np.round(trend_spike, 1),
})

master.to_csv("data/master.csv", index=False)
print(f"master.csv: {len(master)} rows, {len(master.columns)} columns")
print(f"  Columns: {master.columns.tolist()}")
print(f"  VIX: {vix.min():.1f} – {vix.max():.1f}")
print(f"  event_intensity: {event_intensity.min():.1f} – {event_intensity.max():.1f}")
print(f"  trend_spike: {trend_spike.min():.1f} – {trend_spike.max():.1f}")
print(f"\n  First 5 rows:")
print(master.head().to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# EVENTS.CSV
# ══════════════════════════════════════════════════════════════════════════

events_raw = [
    # (event_start, event_end, event_type, severity, headline)
    # 2015
    ("2015-06-12", "2015-07-15", "macro",       3, "Greece debt crisis"),
    ("2015-08-11", "2015-09-10", "macro",       4, "China devaluation + global selloff"),
    ("2015-11-13", "2015-11-20", "security",    4, "Paris terror attacks"),
    ("2015-12-16", "2015-12-18", "macro",       3, "Fed first rate hike since 2008"),
    # 2016
    ("2016-01-04", "2016-01-20", "macro",       3, "China circuit breaker + oil crash"),
    ("2016-06-23", "2016-07-10", "geopolitics", 4, "Brexit referendum"),
    ("2016-11-08", "2016-11-18", "political",   4, "Trump wins election"),
    # 2017
    ("2017-04-07", "2017-04-12", "geopolitics", 3, "US strikes Syria"),
    ("2017-08-09", "2017-08-20", "geopolitics", 3, "North Korea missile crisis"),
    # 2018
    ("2018-02-02", "2018-02-15", "macro",       5, "Volmageddon VIX blowup"),
    ("2018-03-22", "2018-07-06", "geopolitics", 3, "US-China trade war begins"),
    ("2018-10-03", "2018-10-30", "macro",       3, "Fed hawkish selloff"),
    ("2018-12-01", "2018-12-26", "macro",       4, "Christmas Eve crash"),
    # 2019
    ("2019-05-05", "2019-06-01", "geopolitics", 3, "US-China trade war escalates"),
    ("2019-08-05", "2019-08-15", "macro",       3, "Yuan breaks 7 + yield curve inverts"),
    ("2019-09-14", "2019-09-20", "geopolitics", 3, "Saudi Aramco drone attack"),
    # 2020
    ("2020-01-03", "2020-01-10", "geopolitics", 4, "US kills Soleimani"),
    ("2020-01-20", "2020-02-20", "security",    3, "COVID initial reports"),
    ("2020-02-24", "2020-03-23", "security",    5, "COVID pandemic crash"),
    ("2020-03-16", "2020-04-15", "macro",       5, "Fed emergency cut + unlimited QE"),
    ("2020-06-01", "2020-06-15", "political",   3, "George Floyd protests"),
    ("2020-11-03", "2020-11-10", "political",   4, "US election contested"),
    # 2021
    ("2021-01-06", "2021-01-15", "political",   4, "Capitol insurrection"),
    ("2021-01-27", "2021-02-05", "corporate",   3, "GameStop short squeeze"),
    ("2021-02-25", "2021-03-05", "macro",       3, "Bond tantrum"),
    ("2021-05-12", "2021-05-15", "macro",       3, "CPI 4.2% inflation fears"),
    ("2021-09-20", "2021-10-05", "corporate",   3, "Evergrande crisis"),
    ("2021-11-26", "2021-12-10", "security",    3, "Omicron variant"),
    # 2022
    ("2022-02-24", "2022-06-01", "geopolitics", 5, "Russia invades Ukraine"),
    ("2022-03-16", "2022-03-20", "macro",       4, "Fed begins rate hike cycle"),
    ("2022-06-10", "2022-06-20", "macro",       4, "CPI 9.1% 40-year high"),
    ("2022-09-23", "2022-10-15", "macro",       4, "UK gilt crisis"),
    ("2022-11-08", "2022-11-25", "corporate",   5, "FTX collapse"),
    # 2023
    ("2023-03-08", "2023-03-25", "corporate",   5, "SVB + banking crisis"),
    ("2023-05-01", "2023-05-05", "corporate",   3, "First Republic seized"),
    ("2023-10-07", "2023-11-15", "geopolitics", 5, "Hamas attacks Israel"),
    # 2024
    ("2024-01-11", "2024-02-01", "geopolitics", 3, "Houthi Red Sea attacks"),
    ("2024-04-13", "2024-04-17", "geopolitics", 4, "Iran drones at Israel"),
    ("2024-07-13", "2024-07-18", "political",   4, "Trump assassination attempt"),
    ("2024-08-02", "2024-08-10", "macro",       5, "Yen carry trade unwind"),
    ("2024-11-05", "2024-11-15", "political",   4, "Trump wins 2024"),
    # 2025
    ("2025-02-01", "2025-02-15", "geopolitics", 3, "Tariff threats China/Mexico"),
    ("2025-04-02", "2025-04-25", "geopolitics", 5, "Liberation Day tariffs"),
    ("2025-04-09", "2025-04-15", "geopolitics", 4, "Reciprocal tariffs take effect"),
    ("2025-06-01", "2025-06-10", "macro",       3, "Recession fears resurface"),
]

events = pd.DataFrame(events_raw,
    columns=["event_start", "event_end", "event_type", "severity", "headline"])
events.to_csv("data/events.csv", index=False)

print(f"\nevents.csv: {len(events)} events")
print(f"  Types:")
for t, c in events["event_type"].value_counts().items():
    print(f"    {t:<15} {c}")
print(f"\n  First 5:")
print(events.head().to_string(index=False))
