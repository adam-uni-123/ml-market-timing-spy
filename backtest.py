"""
backtest.py - turning the predictions into a trading strategy

The rule: at the close on day t, if the model's P(up) > 0.5, hold SPY until the
next close, otherwise hold T-bills. So each day's return is

    R(t+1) = w(t) * SPY return(t+1) + (1 - w(t)) * T-bill(t) - cost * |w(t) - w(t-1)|

with 10 bps charged per unit traded (the very first buy pays it too).
The benchmarks are buy-and-hold, a 50/200-day moving average crossover and
volatility-managed SPY (holds less SPY when recent volatility is high).
"""

import numpy as np
import pandas as pd

from data import TRADED, TRAIN_END, TRADING_DAYS, save_table

COST_BPS = 10


class Backtester:
    # keeps what every backtest needs in one place (next-day returns, T-bill rate, benchmarks)
    def __init__(self, data, clean, fred):
        self.ret_next = data["ret_next"]
        self.rf = (fred["DGS3MO"].shift(1) / 100 / TRADING_DAYS).reindex(data.index)   # yields come out a day late
        assert self.rf.notna().all()

        px = clean[f"{TRADED}_AdjClose"]
        sma_signal = (px.rolling(50).mean() > px.rolling(200).mean()).astype(float).reindex(data.index)
        # vol target = average 20-day vol over the training years only. Weights move in 10% steps
        # and are capped at 1, so there's no leverage
        self.vol_target = data.loc[data.index <= TRAIN_END, "vol_20"].mean()
        vol_weight = (self.vol_target / data["vol_20"]).clip(upper=1.0)
        self.vol_weight = (vol_weight * 10).round() / 10
        self.benchmarks = {"Buy-and-hold": pd.Series(1.0, index=data.index),
                           "SMA 50/200 crossover": sma_signal,
                           "Vol-managed SPY": self.vol_weight}

        # quick check the maths is right on a tiny example: long, cash, long
        toy = self.run(pd.Series([1.0, 0.0, 1.0], index=data.index[:3]))
        expected = [self.ret_next.iloc[0] - 0.001, self.rf.iloc[1] - 0.001, self.ret_next.iloc[2] - 0.001]
        assert np.allclose(toy["ret"], expected), "backtest arithmetic"

    def run(self, position, cost_bps=COST_BPS):
        # position: weight in SPY decided at the close of t, held from t to t+1
        w = position.astype(float)
        r, rf = self.ret_next.loc[w.index], self.rf.loc[w.index]
        trade = w.diff().abs()
        trade.iloc[0] = abs(w.iloc[0])                  # first entry from cash is costed
        net = w * r + (1 - w) * rf - trade * cost_bps / 1e4
        return pd.DataFrame({"position": w, "trade": trade, "ret": net, "rf": rf})

    def strategy_table(self, proba_df, idx, cost_bps=COST_BPS, threshold=0.5, overlay=None):
        # results for every model and benchmark. overlay = the model whose signal also gets scaled
        # by the vol weight
        rows, curves = {}, {}
        positions = {name: (proba_df[name] > threshold).astype(float) for name in proba_df}
        positions.update({name: pos.loc[idx] for name, pos in self.benchmarks.items()})
        if overlay is not None:
            positions[f"{overlay} + vol-managed"] = (proba_df[overlay] > threshold).astype(float) * self.vol_weight.loc[idx]
        for name, pos in positions.items():
            bt = self.run(pos, cost_bps)
            rows[name], curves[name] = performance(bt), bt["ret"]
        return pd.DataFrame(rows).T, pd.DataFrame(curves)

    def variant(self, positions):
        # results for each column of positions (one per model), after 10 bps costs
        return pd.DataFrame({name: performance(self.run(positions[name])) for name in positions}).T


def max_drawdown_of(ret):
    # worst fall from a peak. Starting wealth of 1 counts as a peak, so a loss on day one still counts
    wealth = (1 + ret).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return (wealth / peak - 1).min()


def sharpe(r, rf):
    ex = r - rf
    return ex.mean() / ex.std() * np.sqrt(TRADING_DAYS)


def performance(bt):
    # the usual performance stats for one backtest (bt comes from Backtester.run)
    r, n = bt["ret"], len(bt)
    growth = (1 + r).prod()
    return pd.Series({
        "Cumulative return (%)": (growth - 1) * 100,
        "Annualised return (%)": (growth ** (TRADING_DAYS / n) - 1) * 100,
        "Annualised volatility (%)": r.std() * np.sqrt(TRADING_DAYS) * 100,
        "Sharpe ratio": sharpe(r, bt["rf"]),
        "Max drawdown (%)": max_drawdown_of(r) * 100,
        "Turnover (x per year)": bt["trade"].sum() / n * TRADING_DAYS,
        "Trades": int((bt["trade"] > 0).sum()),
        "Time in SPY (%)": (bt["position"] > 0).mean() * 100,
        "Average SPY weight (%)": bt["position"].mean() * 100,
    })


def robustness_tables(bt, test_proba, proba_5d, test_perf, test_curves, candidate):
    names, idx = list(test_proba.columns), test_proba.index

    # T22: Sharpe ratio at different cost levels
    costs = pd.DataFrame()
    for bps in (0, 5, 10, 20):
        perf, _ = bt.strategy_table(test_proba, idx, cost_bps=bps, overlay=candidate)
        costs[f"{bps} bps"] = perf["Sharpe ratio"]
    costs["Turnover (x per year)"] = test_perf["Turnover (x per year)"]
    save_table(costs.round(2), "T22_cost_sensitivity")

    # T23: long/short, acting a day late, and the 5-day horizon
    long_cash = (test_proba > 0.5).astype(float)       # 1 = hold SPY, 0 = T-bills
    long_short = long_cash * 2 - 1                      # 1 = long SPY, -1 = short SPY
    delayed = long_cash.shift(1).fillna(0.0)            # act one day late (in cash on the first day)
    five_day = (proba_5d > 0.5).astype(float)           # signal from the 5-day models
    cols = ["Annualised return (%)", "Sharpe ratio", "Max drawdown (%)"]
    variants = pd.concat({
        "Long/cash (main)": test_perf.loc[names, cols],
        "Long/short": bt.variant(long_short)[cols],
        "One-day delay": bt.variant(delayed)[cols],
        "5-day horizon": bt.variant(five_day)[cols],
    }, axis=1)
    save_table(variants.round(2), "T23_variants")

    # T24: year by year, plus how risky vol-managed SPY was compared with buy-and-hold
    yearly = {}
    for name in [candidate, f"{candidate} + vol-managed", "Vol-managed SPY", "Buy-and-hold", "SMA 50/200 crossover"]:
        r_ = test_curves[name]
        yearly[name] = r_.groupby(r_.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    yearly = pd.DataFrame(yearly)
    yearly["Candidate time in SPY (%)"] = (test_proba[candidate] > 0.5).groupby(idx.year).mean() * 100
    yearly.index.name = "Year"
    for name, short in [("Vol-managed SPY", "Vol-managed"), ("Buy-and-hold", "B&H")]:
        r_ = test_curves[name]
        yearly[f"{short} volatility (%)"] = r_.groupby(r_.index.year).std() * np.sqrt(TRADING_DAYS) * 100
        yearly[f"{short} max drawdown (%)"] = r_.groupby(r_.index.year).apply(max_drawdown_of) * 100
    save_table(yearly.round(1), "T24_test_by_year")
    print(costs.round(2), "\n", variants.round(2))
