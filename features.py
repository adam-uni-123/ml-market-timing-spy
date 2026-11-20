"""
features.py - the target and the predictors

Target: does SPY go up from the close on day t to the close on day t+1?
Predictors: 37 things we already know at the close on day t (past returns,
momentum, moving averages, RSI, volatility, volume, sectors, VIX and rates).

The thing that matters most here is timing. shift(k) with k > 0 moves past
values forward to day t, which is safe. shift(-k) pulls future values back,
so it only appears in build_targets (the answers) and in the alignment test
further down. VIX and Treasury yields are lagged an extra day because they
come out after SPY closes at 16:00.
"""

import numpy as np
import pandas as pd

from data import (TRADED, SECTORS, FRED_SERIES, TRAIN_END, VALID_END, TRADING_DAYS,
                  SEED, PROCESSED, save_table)

CYCLICAL = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLY"]
DEFENSIVE = ["XLP", "XLU", "XLV"]


def build_targets(panel):
    # These look forward on purpose. They're what we're predicting and are never used as inputs
    p = panel[f"{TRADED}_AdjClose"]
    t = pd.DataFrame(index=panel.index)
    t["ret_next"] = p.shift(-1) / p - 1                 # return from close t to close t+1
    t["logret_next"] = np.log(p.shift(-1) / p)          # same in logs (regression target)
    t["y_1d"] = (t["ret_next"] > 0).astype(float).where(t["ret_next"].notna())
    t["ret_next5"] = p.shift(-5) / p - 1                # 5-day horizon (robustness check)
    t["y_5d"] = (t["ret_next5"] > 0).astype(float).where(t["ret_next5"].notna())
    return t


def rsi_wilder(price, n=14):
    # RSI with Wilder's smoothing (the standard way it's calculated)
    delta = price.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss)


def build_features(panel):
    # One row per day t. Everything in here has to be known by the close on day t
    p = panel[f"{TRADED}_AdjClose"]
    lr = np.log(p).diff()
    f = pd.DataFrame(index=panel.index)

    # SPY's own history
    for k in range(5):
        f[f"ret_lag{k}"] = lr.shift(k)                  # returns on days t, t-1, ..., t-4
    for k in (20, 60, 120, 252):
        f[f"mom_{k}"] = np.log(p / p.shift(k))          # momentum over the past k days
    for k in (20, 50, 200):
        f[f"sma_gap_{k}"] = p / p.rolling(k).mean() - 1  # gap to trailing average
    macd = p.ewm(span=12, adjust=False).mean() - p.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / p
    f["rsi_14"] = rsi_wilder(p, 14)
    vol20 = lr.rolling(20).std() * np.sqrt(TRADING_DAYS)
    f["vol_20"] = vol20
    f["vol_ratio_20_60"] = vol20 / (lr.rolling(60).std() * np.sqrt(TRADING_DAYS))
    f["range_5"] = np.log(panel[f"{TRADED}_High"] / panel[f"{TRADED}_Low"]).rolling(5).mean()
    v = panel[f"{TRADED}_Volume"]
    f["volume_abn"] = np.log(v / v.rolling(20).mean())
    f["volume_trend"] = np.log(v.rolling(5).mean() / v.rolling(60).mean())

    # Sector ETFs close at 16:00 like SPY, so their day-t prices are fine to use
    sec = pd.DataFrame({t: panel[f"{t}_AdjClose"] for t in SECTORS})   # one column per sector ETF
    spy5 = np.log(p / p.shift(5))
    for t in SECTORS:
        f[f"rel5_{t}"] = np.log(sec[t] / sec[t].shift(5)) - spy5
    sma50 = sec.rolling(50).mean()
    above = (sec > sma50).mean(axis=1)                  # share of the 9 sectors above their 50-day average
    f["breadth_50"] = above.where(sma50.notna().all(axis=1))   # blank until all 9 averages exist (warm-up)
    r20 = np.log(sec / sec.shift(20))
    f["cyc_def_20"] = r20[CYCLICAL].mean(axis=1) - r20[DEFENSIVE].mean(axis=1)

    # VIX: lagged one day (settles at 16:15, after the SPY close)
    vix_known = panel["VIX_Close"].shift(1)
    f["vix_log"] = np.log(vix_known)
    f["vix_chg_5"] = np.log(vix_known / vix_known.shift(5))
    f["vrp"] = (vix_known / 100) ** 2 - lr.rolling(21).var() * TRADING_DAYS

    # Interest rates: lagged one day (H.15 yields are published the next business day)
    rates = panel[FRED_SERIES].shift(1)
    f["term_10y3m"] = rates["T10Y3M"]
    f["term_10y2y"] = rates["T10Y2Y"]
    f["rate_3m"] = rates["DGS3MO"]
    f["d10y_20"] = rates["DGS10"] - rates["DGS10"].shift(20)
    return f


# T08: what each predictor is, why it might work, and where the idea comes from
FEATURE_INFO = [
    ("ret_lag0 … ret_lag4", "Lagged returns", "SPY log return on days t, t−1, …, t−4",
     "Short-horizon autocorrelation / reversal of index returns", "Lo and MacKinlay (1988)"),
    ("mom_20, mom_60, mom_120, mom_252", "Momentum", "log(P_t / P_{t−k}) for k = 20, 60, 120, 252 days",
     "Time-series momentum: past trends tend to persist", "Moskowitz, Ooi and Pedersen (2012); Jegadeesh and Titman (1993)"),
    ("sma_gap_20, sma_gap_50, sma_gap_200", "Moving averages", "P_t / SMA_k(t) − 1, trailing k-day average",
     "Trend-following: price relative to its moving average had predictive content for the index", "Brock, Lakonishok and LeBaron (1992); Lecture 6"),
    ("macd_hist", "Moving averages", "(MACD − signal line) / P_t, EMAs 12/26/9",
     "Trend acceleration: momentum of the trend itself", "Lecture 6"),
    ("rsi_14", "RSI", "14-day RSI, Wilder smoothing",
     "Overbought / oversold conditions, short-term reversal", "Wilder (1978); Lecture 6"),
    ("vol_20, vol_ratio_20_60", "Volatility", "20-day realised volatility; 20-day / 60-day volatility",
     "Volatility clusters; high volatility predicts weaker risk-adjusted returns", "Moreira and Muir (2017)"),
    ("range_5", "Volatility", "5-day mean of ln(High / Low)",
     "Intraday range is an efficient volatility estimator", "Parkinson (1980)"),
    ("volume_abn, volume_trend", "Volume", "ln(V_t / 20-day mean V); ln(5-day / 60-day mean V)",
     "Volume conditions the sign of return autocorrelation; confirms trend strength", "Campbell, Grossman and Wang (1993); Lecture 6"),
    ("rel5_XLB … rel5_XLY (9)", "Sectors", "5-day sector log return minus SPY 5-day log return",
     "Industry returns lead the aggregate market", "Hong, Torous and Valkanov (2007)"),
    ("breadth_50", "Sectors", "Share of the 9 sectors above their own 50-day SMA",
     "Breadth of participation in the market trend", "Hong, Torous and Valkanov (2007); Lecture 6"),
    ("cyc_def_20", "Sectors", "20-day return, cyclical minus defensive sectors",
     "Risk appetite and growth expectations", "Hong, Torous and Valkanov (2007)"),
    ("vix_log, vix_chg_5", "Risk sentiment", "ln VIX_{t−1}; 5-day change in ln VIX (lagged one day)",
     "Implied volatility as a gauge of fear and expected risk", "Whaley (2000)"),
    ("vrp", "Risk sentiment", "(VIX_{t−1}/100)² − 21-day realised variance",
     "Variance risk premium predicts index returns", "Bollerslev, Tauchen and Zhou (2009)"),
    ("term_10y3m, term_10y2y", "Interest rates", "Treasury term spreads (lagged one day)",
     "Term spread tracks the business cycle and expected returns; predicts recessions", "Campbell (1987); Fama and French (1989); Estrella and Mishkin (1998)"),
    ("rate_3m, d10y_20", "Interest rates", "3-month yield; 20-day change in 10-year yield (lagged one day)",
     "Short rate and rate shocks predict equity returns", "Ang and Bekaert (2007); Campbell (1987)"),
]


def make_model_table(clean):
    features, targets = build_features(clean), build_targets(clean)
    feature_names = list(features.columns)
    save_table(pd.DataFrame(FEATURE_INFO, columns=["Feature(s)", "Group", "Definition (known at close of t)",
                                                   "Financial rationale", "Source"]),
               "T08_feature_list", index=False)

    # Drop the first 252 days, because the longest rolling window needs a year to fill up (I don't
    # backfill them), and the last day, because we don't know its next-day return yet
    first_valid = features.dropna().index.min()
    data = features.join(targets).loc[first_valid:]
    assert data[feature_names].isna().sum().sum() == 0, "missing predictor values after warm-up"
    data = data.dropna(subset=["y_1d"])
    print(f"Modelling table: {data.shape[0]} days x {len(feature_names)} predictors, "
          f"{data.index.min().date()} to {data.index.max().date()}")

    periods = {"Train": data.loc[data.index <= TRAIN_END, "y_1d"],
               "Validation": data.loc[(data.index > TRAIN_END) & (data.index <= VALID_END), "y_1d"],
               "Test": data.loc[data.index > VALID_END, "y_1d"]}
    rows = {}
    for name, y in periods.items():
        rows[name] = {"Start": y.index.min().date(), "End": y.index.max().date(),
                      "Days": len(y), "Up days (%)": round(y.mean() * 100, 1)}
    split_summary = pd.DataFrame.from_dict(rows, orient="index")
    split_summary.index.name = "Split"
    save_table(split_summary, "T09_split_summary")
    data.to_csv(PROCESSED / "model_data.csv", date_format="%Y-%m-%d")
    return data, features, feature_names


def leakage_tests(clean, features, data):
    # Truncation test: rebuild the predictors using only data up to date d. If any value changes,
    # that predictor must have been using data from after d
    rng = np.random.default_rng(SEED)
    random_dates = rng.choice(data.index[:-1], size=12, replace=False)    # 12 random days
    check_dates = sorted(pd.DatetimeIndex(random_dates))
    check_dates.append(data.index[data.index <= TRAIN_END][-1])           # last training day
    check_dates.append(data.index[data.index <= VALID_END][-1])           # last validation day
    for d in check_dates:
        pd.testing.assert_series_equal(build_features(clean.loc[:d]).loc[d], features.loc[d],
                                       rtol=1e-9, check_names=False)
    print(f"Truncation test passed on {len(check_dates)} dates: no predictor uses future data")

    # Alignment test: the target on day t should be the same number as ret_lag0 on day t+1
    next_day_lag0 = features["ret_lag0"].shift(-1).loc[data.index]
    assert np.allclose(data["logret_next"], next_day_lag0, equal_nan=False)
    print("Target alignment test passed: target(t) equals the realised return of day t+1")


def predictor_diagnostics(data, feature_names):
    # Training years only: how well each predictor on its own lines up with tomorrow's return
    # (rank correlation), and which pairs are basically measuring the same thing
    train = data.loc[data.index <= TRAIN_END]
    ic = train[feature_names].corrwith(train["ret_next"], method="spearman").sort_values()
    print(f"Largest absolute correlation with next-day return: {ic.abs().max():.3f} ({ic.abs().idxmax()})")
    fcorr = train[feature_names].corr()
    rows = []                                           # every pair of predictors, each pair once
    for i, a in enumerate(feature_names):
        for b in feature_names[i + 1:]:
            rows.append({"Feature A": a, "Feature B": b, "Correlation": fcorr.loc[a, b]})
    pairs = pd.DataFrame(rows)
    high = pairs[pairs["Correlation"].abs() > 0.8].sort_values("Correlation", key=abs, ascending=False)
    save_table(high.round(3), "T10_collinear_pairs", index=False)
    print(f"{len(high)} of {len(pairs)} predictor pairs have |correlation| > 0.8")
    return ic, 1.96 / np.sqrt(len(train))
