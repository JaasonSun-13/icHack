import pandas as pd
import numpy as np
import yfinance as yf

tickers = ["^GSPC", "^VIX"]
start_date = "2017-01-01"
end_date = "2025-12-31"


# # Download intraday data (last ~60 days max for 5m)
# data = yf.download(tickers, start=start_date, end=end_date)
# data.to_csv('spxnol_prices.csv')


vol_30 = pd.read_csv('FinanceDataPuller/1-day-vol.csv')
df = pd.read_csv("FinanceDataPuller/spxnvol_prices.csv")
df.index = df['Date']
vol_30['date'] = pd.to_datetime(vol_30['date'])
vol_30 = vol_30.set_index("date")

prices_spx = pd.DataFrame()
prices_spx['spx_close'] = df['Close^GSPC']
prices_spx['1_day_vol'] = prices_spx['spx_close'].pct_change().abs()

prices_spx['3_day_avg_vol'] = prices_spx['1_day_vol'].rolling(3).mean()
print(vol_30['vol'])
print(prices_spx['3_day_avg_vol'])
prices_spx['30_day_implied_vol'] = vol_30['vol'].reindex(prices_spx.index)

prices_spx.to_csv('FinanceDataPuller/finalvol.csv')