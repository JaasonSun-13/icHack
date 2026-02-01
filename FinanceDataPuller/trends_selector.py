import pandas as pd
import numpy as np

years = np.arange(2017, 2026)

# load trading days correctly
trading_days_df = pd.read_csv('FinanceDataPuller/finalvol.csv', index_col=0)
trading_days_df.index = pd.to_datetime(trading_days_df.index)
trading_days = trading_days_df.index

major_event_frame = pd.DataFrame(index=trading_days)

for year in years:
    df = pd.read_csv(f"FinanceDataPuller/Day_gdbltdoc/timeline_keywords_{year}.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index("datetime", inplace=True)

    # keep only trading days
    df = df.loc[df.index.isin(trading_days)]

    # drop datetime column if still present
    df = df.drop(columns=["datetime"], errors="ignore")

    for col in df.columns:
        series = df[col].astype(float).dropna()

        if series.empty:
            continue

        Q1 = np.percentile(series, 25)
        Q3 = np.percentile(series, 75)
        IQR = Q3 - Q1

        upper_bound = Q3 + 1.5 * IQR

        outliers = series[series > upper_bound]

        if outliers.empty:
            outliers = series.nlargest(15)

        event_col_name = f"{col}"

        if event_col_name not in major_event_frame.columns:
            major_event_frame[event_col_name] = 0

        major_event_frame.loc[outliers.index, event_col_name] = 1

print(major_event_frame)

major_event_frame.to_csv('major_event_frame.csv')
