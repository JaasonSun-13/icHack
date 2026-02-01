#!/usr/bin/env python3
"""
Quick test: fetch GDELT timeline for Ukraine/Russia war keywords.

Usage:
    pip install gdeltdoc pandas
    python test_gdelt.py
"""

import time
import pandas as pd
from gdeltdoc import GdeltDoc, Filters

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

# Just one keyword to test
keee = "stephen hawking" 
KEYWORD = keee
version = keee

# Recent 3 months (GDELT works best with ~3 month windows)
START = "2023-01-01"
END   = "2025-03-31"

# ═══════════════════════════════════════════════════════════════════════════════
# FETCH
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Fetching GDELT timeline for '{KEYWORD}'")
    print(f"Date range: {START} to {END}\n")

    gd = GdeltDoc()

    f = Filters(
        keyword=KEYWORD,
        start_date=START,
        end_date=END,
        country="US",
    )

    df = gd.timeline_search("timelinevol", f)

    print(f"Raw response shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 10 rows:")
    print(df.head(10))
    print(f"\nLast 10 rows:")
    print(df.tail(10))

    # Save raw
    # df.to_csv(f"test_gdelt_raw{version}2.csv", index=False)
    print(f"\nSaved: test_gdelt_raw.csv")


if __name__ == "__main__":
    main()