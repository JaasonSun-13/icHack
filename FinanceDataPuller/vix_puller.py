import pandas as pd
import numpy as np
import yfinance as yf

# Volatility index tickers
tickers = ["^VIX", "^VIX9D", "^VVIX", "^MOVE", "^VXN", "^RVX"]

start_date = "2015-01-01"
end_date = "2025-12-31"

# Download data
data = yf.download(tickers, start=start_date, end=end_date)

# Extract Close prices only
close_prices = data["Close"]

# Rename columns for clarity
close_prices.columns = ["VIX", "VIX9D", "VVIX", "MOVE", "VXN", "RVX"]

# Save to CSV
close_prices.to_csv("volatility_indices_2015_2025.csv")

# Preview
print(close_prices.head())
print(close_prices.tail())
