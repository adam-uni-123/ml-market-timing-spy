"""
download_data.py - how I got the raw data

I ran this once on my own laptop on 22 Oct 2026. It saves one CSV per series
into data/raw/ plus metadata.json (source, dates, row count and a SHA-256 hash
of each file). You don't need to run it, because the analysis only reads the
saved files. Running it again overwrites data/raw/ with newer data (and Yahoo
revises its adjusted prices over time), so main.py would stop at the hash
check and the numbers wouldn't match the report anyway.

Needs:  python -m pip install yfinance pandas requests
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

START_DATE = "2005-01-01"
OUT_DIR = Path(__file__).resolve().parent / "data" / "raw"

YAHOO_TICKERS = {
    "SPY": "SPDR S&P 500 ETF Trust (traded asset)",
    "XLB": "Materials Select Sector SPDR",
    "XLC": "Communication Services Select Sector SPDR (starts June 2018)",
    "XLE": "Energy Select Sector SPDR",
    "XLF": "Financial Select Sector SPDR",
    "XLI": "Industrial Select Sector SPDR",
    "XLK": "Technology Select Sector SPDR",
    "XLP": "Consumer Staples Select Sector SPDR",
    "XLRE": "Real Estate Select Sector SPDR (starts October 2015)",
    "XLU": "Utilities Select Sector SPDR",
    "XLV": "Health Care Select Sector SPDR",
    "XLY": "Consumer Discretionary Select Sector SPDR",
    "^VIX": "Cboe Volatility Index (an index: volume is always 0)",
}
FRED_SERIES = {
    "DGS3MO": "3-month Treasury, constant maturity, % p.a. (risk-free proxy)",
    "DGS2": "2-year Treasury, constant maturity, % p.a.",
    "DGS10": "10-year Treasury, constant maturity, % p.a.",
    "T10Y2Y": "10-year minus 2-year Treasury spread, percentage points",
    "T10Y3M": "10-year minus 3-month Treasury spread, percentage points",
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def record(name, path, df, source, description):
    # one entry in metadata.json: what the file is, plus its hash
    return {"name": name, "file": path.name, "description": description, "source": source,
            "rows": len(df), "first_date": f"{df.index.min():%Y-%m-%d}", "last_date": f"{df.index.max():%Y-%m-%d}",
            "missing_values": {c: int(n) for c, n in df.isna().sum().items()},
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


OUT_DIR.mkdir(parents=True, exist_ok=True)
started = datetime.now(timezone.utc)
files, failures = [], []

for ticker, description in YAHOO_TICKERS.items():
    print("Yahoo Finance:", ticker)
    try:
        # auto_adjust=False so we get both Close and Adj Close (which includes dividends)
        df = yf.download(ticker, start=START_DATE, auto_adjust=False, actions=False,
                         progress=False, multi_level_index=False)
        df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        df.index.name = "Date"
        path = OUT_DIR / f"{ticker.replace('^', '')}.csv"          # ^VIX is saved as VIX.csv
        df.to_csv(path, date_format="%Y-%m-%d")
        files.append(record(ticker, path, df, "Yahoo Finance via yfinance", description))
    except Exception as error:
        failures.append({"name": ticker, "error": str(error)})

for series, description in FRED_SERIES.items():
    print("FRED:", series)
    try:
        response = requests.get(FRED_URL, params={"id": series, "cosd": START_DATE}, timeout=60)
        response.raise_for_status()
        # FRED uses "." for bond-market holidays, so read those as blanks
        df = pd.read_csv(StringIO(response.text), na_values=["."], index_col=0, parse_dates=True)
        df.index.name = "Date"
        path = OUT_DIR / f"FRED_{series}.csv"
        df.to_csv(path, date_format="%Y-%m-%d")
        files.append(record(series, path, df, f"FRED, {FRED_URL}?id={series}", description))
    except Exception as error:
        failures.append({"name": series, "error": str(error)})

metadata = {"download_started_utc": started.isoformat(timespec="seconds"),
            "requested_start": START_DATE,
            "environment": {"python": sys.version.split()[0], "pandas": pd.__version__,
                            "yfinance": yf.__version__},
            "files": files, "failures": failures}
(OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Saved {len(files)} files to {OUT_DIR}; failed: {[f['name'] for f in failures] or 'none'}")
