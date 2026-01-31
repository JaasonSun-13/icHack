import pandas as pd
import numpy as np
import yfinance as yf
import time
from pytrends.request import TrendReq
# import anthropic
# key = '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'
# client = anthropic.Anthropic()

# message = client.messages.create(
#     model="claude-sonnet-4-5",
#     max_tokens=1000,
#     messages=[
#         {
#             "role": "user",
#             "content": "What should I search for to find the latest developments in renewable energy?"
#         }
#     ]
# )
# print(message.content)

years = np.arange(2023, 2026)

for i in years:
    i = int(i)

    df_trend = pd.read_csv(
        f'FinanceDataPuller/TrendCsv/searched_with_rising-queries_US_{i}0101-0000_20260131-17.csv'
    )

    keywords = df_trend['query'].dropna().tolist()

    pytrends = TrendReq(hl='en-US', tz=360)

    all_data = []

    # batch into groups of 5
    for j in range(0, len(keywords), 5):
        kw_batch = keywords[j:j+5]

        pytrends.build_payload(
            kw_batch,
            timeframe=f'{i}-01-01 {i}-12-31'
        )

        df = pytrends.interest_over_time()

        if not df.empty:
            all_data.append(df)
        

        time.sleep(5)  # avoid 429 / 400

    if all_data:
        final_df = pd.concat(all_data, axis=1)
        final_df.to_csv(f'FinanceDataPuller/pytrendsCSV/pytrends_{i}.csv')
