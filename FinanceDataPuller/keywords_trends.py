import pandas as pd
import numpy as np
import yfinance as yf
import time
from pytrends.request import TrendReq
from gdeltdoc import GdeltDoc, Filters
from gdeltdoc.errors import RateLimitError

def safe_timeline_search(gdelt, mode, filters, sleep_time=1):
    while True:
        try:
            return gdelt.timeline_search(mode, filters)
        except RateLimitError:
            print("Rate limit hit — sleeping 1 second...")
            time.sleep(sleep_time)
        except ValueError as e:
            print("Invalid query, skipping:", e)
            return pd.DataFrame()


def timeline_trend(keywords, year):
    print(f"Fetching GDELT timeline for year {year}")

    start = f"{year}0101"   # GDELT format
    end   = f"{year}1231"

    gd = GdeltDoc()

    all_series = []

    for tick in keywords:
        tick = str(tick).strip()

        if len(tick) < 3:
            print(f"Skipping short keyword: {tick}")
            continue

        print(f"  → querying keyword: {tick}")

        f = Filters(
            keyword=tick,
            start_date=start,
            end_date=end,
            country="US",
        )

        df = safe_timeline_search(gd, "timelinevol", f)
        print(df)

        if df.empty:
            continue

        # --- FIXES ---
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.date
        df = df.set_index("datetime")
        df[f'{tick}_intensity'] = df['Volume Intensity']
        df = df.drop(columns="Volume Intensity")

        all_series.append(df)

    if len(all_series) == 0:
        print("No valid keywords for this year.")
        return pd.DataFrame()

    # merge all keyword series by date
    df_year = pd.concat(all_series, axis=1).fillna(0)

    # save yearly file
    out_path = f"FinanceDataPuller/Day_gdbltdoc/timeline_keywords_{year}.csv"
    df_year.to_csv(out_path)

    print(f"Saved: {out_path}")

    return df_year




years = np.arange(2017, 2026)

for i in years:
    i = int(i)

    df_trend = pd.read_csv(
        f"FinanceDataPuller/TrendCsv/searched_with_rising-queries_US_{i}0101-0000_20260131-17.csv"
    )

    keywords = df_trend["query"].dropna().tolist()

    print(f"\nRunning year {i} with {len(keywords)} keywords")

    timeline_trend(keywords, i)

