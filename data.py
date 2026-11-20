"""
data.py - settings, loading and cleaning

Reads the Yahoo Finance and FRED files saved in data/raw/, checks they are
exactly what I downloaded, drops the last (unfinished) day and lines everything
up on SPY's trading days. The result is one clean daily table. This file also
works out the descriptive stats for Section 2 of the report.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday, GoodFriday,
                                    USMartinLutherKingJr, USPresidentsDay,
                                    USMemorialDay, USLaborDay, USThanksgivingDay,
                                    nearest_workday)

# settings shared by all the other files
SEED = 42
ROOT = Path(__file__).resolve().parent
RAW, PROCESSED = ROOT / "data" / "raw", ROOT / "data" / "processed"
FIG_DIR, TAB_DIR = ROOT / "figures", ROOT / "tables"
for folder in (PROCESSED, FIG_DIR, TAB_DIR):
    folder.mkdir(parents=True, exist_ok=True)

TRADED = "SPY"                                    # SPY is the only thing actually traded
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
SECTORS_EDA_ONLY = ["XLC", "XLRE"]                # too short (start 2018 / 2015), EDA only
FRED_SERIES = ["DGS3MO", "DGS2", "DGS10", "T10Y2Y", "T10Y3M"]
TRAIN_END = pd.Timestamp("2016-12-31")            # train 2006-2016
VALID_END = pd.Timestamp("2019-12-31")            # validation 2017-2019, test 2020 onwards
SAMPLE_END = pd.Timestamp("2026-10-21")           # last full trading day before I downloaded
TRADING_DAYS = 252


def save_table(df, name, **kwargs):
    df.to_csv(TAB_DIR / f"{name}.csv", **kwargs)


def load_raw():
    # First make sure every raw file is exactly what I downloaded (SHA-256 hash in metadata.json)
    meta = json.loads((RAW / "metadata.json").read_text(encoding="utf-8"))
    for rec in meta["files"]:
        digest = hashlib.sha256((RAW / rec["file"]).read_bytes()).hexdigest()
        assert digest == rec["sha256"], f"{rec['file']} differs from the recorded download"
    assert not meta["failures"], meta["failures"]
    print(f"All {len(meta['files'])} raw files verified (downloaded {meta['download_started_utc']})")

    rows = []                                      # T01: one row per series
    for r in meta["files"]:
        if r["source"].startswith("Yahoo"):
            source = "Yahoo Finance"
        else:
            source = "FRED (St. Louis Fed)"
        rows.append({"Series": r["name"].replace("^", ""), "Description": r["description"], "Source": source,
                     "First date": r["first_date"], "Last date": r["last_date"], "Rows": r["rows"]})
    sources = pd.DataFrame(rows).set_index("Series")
    save_table(sources, "T01_data_sources")

    tickers = [TRADED] + SECTORS + SECTORS_EDA_ONLY + ["VIX"]
    raw = {t: pd.read_csv(RAW / f"{t}.csv", index_col="Date", parse_dates=True) for t in tickers}
    fred_files = {s: pd.read_csv(RAW / f"FRED_{s}.csv", index_col="Date", parse_dates=True)[s]
                  for s in FRED_SERIES}
    return meta, raw, fred_files


def quality_checks(raw, fred_files):
    # basic sanity checks: gaps, duplicate dates, zero volume and prices that can't be right
    def quality(df):
        px = df[["Open", "High", "Low", "Close", "Adj Close"]]
        return pd.Series({
            "Rows": len(df), "First": df.index.min().date(), "Last": df.index.max().date(),
            "Missing cells": int(df.isna().sum().sum()),
            "Duplicate dates": int(df.index.duplicated().sum()),
            "Zero volume": int((df["Volume"] == 0).sum()),
            "Non-positive price": int((px <= 0).any(axis=1).sum()),
            "High < Low": int((df["High"] < df["Low"]).sum()),
            "Close outside range": int(((df["Close"] > df["High"] * 1.0001) |
                                        (df["Close"] < df["Low"] * 0.9999)).sum()),
        })
    table = pd.DataFrame({t: quality(df) for t, df in raw.items()}).T
    save_table(table, "T02a_quality_prices")
    missing = {s: int(x.isna().sum()) for s, x in fred_files.items()}
    print("FRED blanks (bond-market holidays):", missing)
    return table


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    # normal NYSE holidays. Any other weekday with no trading must be an unscheduled closure
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def clean_data(raw, fred_files):
    # 1. I downloaded at 12:43 New York time on 22 Oct 2026, before the close, so the 22 Oct row
    #    is only part of a day. The sample stops at 21 Oct, the last full day
    raw = {t: df.loc[:SAMPLE_END] for t, df in raw.items()}
    fred_raw = pd.concat(fred_files, axis=1).loc[:SAMPLE_END]
    print("Sample end date:", SAMPLE_END.date())

    # 2. SPY's dates are the trading calendar. Print any odd closures (e.g. Hurricane Sandy in 2012)
    calendar = raw[TRADED].index
    non_trading = pd.bdate_range(calendar.min(), calendar.max()).difference(calendar)
    unscheduled = non_trading.difference(NYSEHolidayCalendar().holidays(calendar.min(), calendar.max()))
    print(f"{len(calendar)} trading days; unscheduled closures:", [d.date() for d in unscheduled])

    # 3. Line everything up. Sector ETFs should have exactly SPY's dates, VIX has a couple of extra
    #    holiday rows that get dropped, and FRED gaps are only ever filled forwards
    for t in SECTORS:
        assert raw[t].index.equals(calendar), f"{t} calendar differs from SPY"
    for t in SECTORS_EDA_ONLY:
        assert raw[t].index.equals(calendar[calendar >= raw[t].index.min()]), t
    print("VIX rows on non-trading days (dropped):", [d.date() for d in raw["VIX"].index.difference(calendar)])
    vix = raw["VIX"].reindex(calendar)
    assert vix["Close"].notna().all()
    # The bond market shuts on some days the stock market is open, so use the last published yield.
    # Forward fill only (max 5 days). A backfill would mean using a yield from the future
    fred = fred_raw.reindex(fred_raw.index.union(calendar))     # add SPY trading days FRED doesn't have
    fred = fred.ffill(limit=5)                                  # fill blanks with the last known yield
    fred = fred.reindex(calendar)                               # keep SPY's trading days only
    assert fred.isna().sum().sum() == 0

    # 4. Returns use Adj Close so dividends are included
    spy = raw[TRADED]
    adj = pd.DataFrame({t: raw[t]["Adj Close"] for t in [TRADED] + SECTORS + SECTORS_EDA_ONLY})
    log_ret = np.log(adj).diff()
    simple_ret = adj.pct_change(fill_method=None)
    years = (calendar[-1] - calendar[0]).days / 365.25
    tr = (adj[TRADED].iloc[-1] / adj[TRADED].iloc[0]) ** (1 / years) - 1
    pr = (spy["Close"].iloc[-1] / spy["Close"].iloc[0]) ** (1 / years) - 1
    print(f"SPY total return {tr:.2%} a year, price return {pr:.2%}, dividends {tr - pr:.2%}")

    # 5. The big outliers are real days (2008, Covid etc.), so I flag them but don't remove them
    r = log_ret[TRADED].dropna()
    z = (r - r.mean()) / r.std()
    print(f"|z| > 4: observed {int((z.abs() > 4).sum())} days, "
          f"normal distribution would give {len(z) * 2 * stats.norm.sf(4):.2f}")
    moves = pd.DataFrame({"Log return": r, "z-score": z, "VIX close": vix["Close"]})
    top10 = r.abs().sort_values(ascending=False).index[:10]       # the 10 biggest daily moves, up or down
    largest = moves.loc[top10]
    largest["Log return"] = largest["Log return"].map("{:.2%}".format)
    largest = largest.round(1)
    largest.index = largest.index.date
    save_table(largest, "T03_largest_moves")

    # 6. Save one clean daily table. features.py only ever works from this
    clean = pd.DataFrame(index=calendar)
    for col in spy.columns:                                  # SPY_Open, SPY_High, ..., SPY_AdjClose
        clean[f"{TRADED}_{col.replace(' ', '')}"] = spy[col]
    for t in SECTORS + SECTORS_EDA_ONLY:
        clean[f"{t}_AdjClose"] = raw[t]["Adj Close"]
    for t in SECTORS:
        clean[f"{t}_Volume"] = raw[t]["Volume"]
    clean["VIX_Close"] = vix["Close"]
    for s in FRED_SERIES:
        clean[s] = fred[s]
    clean.index.name = "Date"
    clean.to_csv(PROCESSED / "daily_clean.csv", date_format="%Y-%m-%d")
    return dict(calendar=calendar, sample_end=SAMPLE_END, spy=spy, vix=vix, fred=fred, adj=adj,
                log_ret=log_ret, simple_ret=simple_ret, r=r, z=z, clean=clean)


def max_drawdown(simple):
    # worst peak-to-trough fall (the first peak is the first day's close)
    wealth = (1 + simple).cumprod()
    return (wealth / wealth.cummax() - 1).min()


def describe_market(market):
    # descriptive stats, autocorrelation and how the market behaved (whole sample)
    log_ret, r, vix = market["log_ret"], market["r"], market["vix"]

    def describe_returns(x):
        x = x.dropna()
        return pd.Series({
            "Start": x.index.min().date(), "Days": len(x),
            "Ann. mean (%)": x.mean() * TRADING_DAYS * 100,
            "Ann. vol (%)": x.std() * np.sqrt(TRADING_DAYS) * 100,
            "Mean/vol": x.mean() / x.std() * np.sqrt(TRADING_DAYS),
            "Skew": stats.skew(x), "Excess kurtosis": stats.kurtosis(x),
            "Min (%)": x.min() * 100, "Max (%)": x.max() * 100,
            "Up days (%)": (x > 0).mean() * 100,
            "AC(1) return": x.autocorr(1), "AC(1) |return|": x.abs().autocorr(1),
        })
    desc = pd.DataFrame({t: describe_returns(log_ret[t]) for t in log_ret.columns}).T
    num_cols = desc.columns.drop(["Start", "Days"])
    desc[num_cols] = desc[num_cols].astype(float).round(2)
    save_table(desc, "T04_descriptive_returns")

    # autocorrelation of returns (direction) and of absolute returns (size, i.e. volatility clustering)
    lags = range(1, 21)
    acf_ret = [r.autocorr(k) for k in lags]
    acf_abs = [r.abs().autocorr(k) for k in lags]
    band = 1.96 / np.sqrt(len(r))
    acf_table = pd.DataFrame({"Returns": acf_ret, "Absolute returns": acf_abs}, index=lags).round(3)
    acf_table.index.name = "Lag"
    save_table(acf_table, "T05_autocorrelation")
    r_tr = r.loc[:TRAIN_END]                       # same thing on the training years only
    acf_abs_tr = np.array([r_tr.abs().autocorr(k) for k in lags])
    print(f"Training period only: absolute-return AC outside band at "
          f"{(np.abs(acf_abs_tr) > 1.96 / np.sqrt(len(r_tr))).sum()} of 20 lags "
          f"(smallest {acf_abs_tr.min():.2f}); lag-1 return AC {r_tr.autocorr(1):.3f}")

    # Does the lag-1 reversal hold up? Split days into calm / turbulent using the volatility
    # known the evening before (that's why there's a shift(1))
    prior_vol = r.rolling(21).std().shift(1)
    pairs = pd.DataFrame({"r": r, "r_prev": r.shift(1), "turbulent": prior_vol > prior_vol.median()}).dropna()

    def lag1(df):                                  # correlation of today's return with yesterday's
        return df["r"].corr(df["r_prev"])

    regime_ac = pd.Series({
        "Calm days (prior vol below median)": lag1(pairs[~pairs["turbulent"]]),
        "Turbulent days (prior vol above median)": lag1(pairs[pairs["turbulent"]]),
        "Train period (to 2016)": lag1(pairs.loc[:TRAIN_END]),
        "Validation period (2017-19)": lag1(pairs.loc[TRAIN_END:VALID_END]),
        "Test period (2020-)": lag1(pairs.loc[VALID_END:]),
    }, name="Lag-1 autocorrelation").round(3).to_frame()
    save_table(regime_ac, "T05b_reversal_by_regime")

    # how the market did each year, and in train / validation / test
    s = market["simple_ret"][TRADED].dropna()
    rows = {}
    for year, x in s.groupby(s.index.year):        # x = that year's daily returns
        rows[year] = {"Return (%)": ((1 + x).prod() - 1) * 100,
                      "Volatility (%)": x.std() * np.sqrt(TRADING_DAYS) * 100,
                      "Max drawdown (%)": max_drawdown(x) * 100,
                      "Up days (%)": (x > 0).mean() * 100,
                      "Avg VIX": vix["Close"][vix.index.year == year].mean()}
    yearly = pd.DataFrame.from_dict(rows, orient="index").round(1)
    yearly.index.name = "Year"
    save_table(yearly, "T06_yearly_conditions")
    periods = {"Train (to 2016)": s[s.index <= TRAIN_END],
               "Validation (2017-19)": s[(s.index > TRAIN_END) & (s.index <= VALID_END)],
               "Test (2020-)": s[s.index > VALID_END]}
    rows = {}
    for name, x in periods.items():
        rows[name] = {"Days": len(x),
                      "Up days (%)": (x > 0).mean() * 100,
                      "Ann. return (%)": ((1 + x).prod() ** (TRADING_DAYS / len(x)) - 1) * 100,
                      "Ann. vol (%)": x.std() * np.sqrt(TRADING_DAYS) * 100,
                      "Max drawdown (%)": max_drawdown(x) * 100}
    by_period = pd.DataFrame.from_dict(rows, orient="index").round(1)
    save_table(by_period, "T07_period_conditions")
    print(by_period)
    return dict(desc=desc, lags=lags, acf_ret=acf_ret, acf_abs=acf_abs, band=band, regime_ac=regime_ac)
