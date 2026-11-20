"""
evaluation.py - how good are the predictions, and is any of it just luck?

The classification metrics (accuracy, precision, recall, F1, ROC-AUC), the
Pesaran-Timmermann direction test, MAE/RMSE for the regression versions, the
McNemar test between models, a block bootstrap of the Sharpe ratios and the
deflated Sharpe ratio. The placebo and positive control checks live here too,
because they need the fitted models.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, log_loss, mean_absolute_error, mean_squared_error)

from data import SEED, TRADING_DAYS, save_table
from models import tscv, walk_forward
from backtest import sharpe


def predictive_metrics(y_true, proba, threshold=0.5):
    pred = (proba > threshold).astype(int)
    return pd.Series({
        "Accuracy": accuracy_score(y_true, pred),
        "Always-up accuracy": y_true.mean(),
        "Precision (up)": precision_score(y_true, pred, zero_division=0),
        "Recall (up)": recall_score(y_true, pred, zero_division=0),
        "F1 (up)": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, proba),
        "Log loss": log_loss(y_true, np.column_stack([1 - proba, proba])),
        "Predicted up (%)": pred.mean() * 100,
    })


def pesaran_timmermann(y_true, y_pred):
    # Are the predicted directions right more often than chance would give? (one-sided p-value)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    p_hat = (y_true == y_pred).mean()
    py, px = y_true.mean(), y_pred.mean()
    p_star = py * px + (1 - py) * (1 - px)
    var_p = p_star * (1 - p_star) / n
    var_star = ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n
                + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    if var_p - var_star <= 0:
        return np.nan, np.nan
    stat = (p_hat - p_star) / np.sqrt(var_p - var_star)
    return stat, stats.norm.sf(stat)


def metrics_table(y, proba_df, name, with_pt=False):
    table = pd.DataFrame({m: predictive_metrics(y, proba_df[m]) for m in proba_df}).T
    if with_pt:
        for m in proba_df:
            table.loc[m, ["PT statistic", "PT p-value"]] = pesaran_timmermann(y, (proba_df[m] > 0.5).astype(int))
    # benchmark: a "model" that always says 51% chance of up, i.e. always predicts up
    table.loc["Always up (benchmark)"] = predictive_metrics(y, pd.Series(0.51, index=y.index))
    table.loc["Always up (benchmark)", "ROC-AUC"] = 0.5      # a constant forecast can't rank days, so 0.5
    table.loc["Always up (benchmark)", "Log loss"] = np.nan   # doesn't mean much for a constant
    if with_pt:
        table.loc["Always up (benchmark)", ["PT statistic", "PT p-value"]] = np.nan
    save_table(table.round(3), name)
    return table


def placebo_and_positive_control(searches, X_train, y_train, data):
    # Placebo: shuffle the labels and AUC should drop to about 0.5. If it didn't, the validation
    # setup itself would be finding "skill" in pure noise
    rng = np.random.default_rng(SEED)
    rows = []
    for name, search in searches.items():
        scores = []
        for _ in range(5):
            y_shuffled = pd.Series(rng.permutation(y_train.values), index=y_train.index)
            scores.extend(cross_val_score(clone(search.best_estimator_), X_train, y_shuffled,
                                          cv=tscv, scoring="roc_auc", n_jobs=-1))
        rows.append({"Model": name, "Real labels: CV ROC-AUC": search.best_score_,
                     "Shuffled labels: mean ROC-AUC": np.mean(scores), "Shuffled labels: std": np.std(scores)})
    placebo = pd.DataFrame(rows).set_index("Model").round(3)
    save_table(placebo, "T13_placebo_test")
    assert (placebo["Shuffled labels: mean ROC-AUC"].sub(0.5).abs() < 0.03).all(), "placebo AUC far from 0.5"

    # Positive control: add a leaked predictor on purpose (tomorrow's return). It should give a
    # near-perfect AUC, which shows the checks really would spot leakage if there was any
    X_leak = X_train.assign(LEAK_ret_next=data.loc[X_train.index, "ret_next"])
    leak_est = clone(searches["Logistic regression"].best_estimator_)
    leak_real = cross_val_score(leak_est, X_leak, y_train, cv=tscv, scoring="roc_auc", n_jobs=-1).mean()
    shuffled_scores = []
    for _ in range(5):
        y_shuffled = pd.Series(rng.permutation(y_train.values), index=y_train.index)
        shuffled_scores.append(cross_val_score(leak_est, X_leak, y_shuffled, cv=tscv,
                                               scoring="roc_auc", n_jobs=-1).mean())
    leak_shuf = np.mean(shuffled_scores)
    control = pd.DataFrame(
        {"Real labels: CV ROC-AUC": [placebo.loc["Logistic regression", "Real labels: CV ROC-AUC"], leak_real],
         "Shuffled labels: mean ROC-AUC": [placebo.loc["Logistic regression", "Shuffled labels: mean ROC-AUC"], leak_shuf]},
        index=["Logistic regression, 37 genuine predictors", "Same + leaked tomorrow's return"]).round(3)
    save_table(control, "T13b_positive_control")
    assert leak_real > 0.9, "positive control failed: the leak was not detected"
    print(placebo.to_string(), "\n", control.to_string())


def permutation_table(searches, X_train, y_train, X_val, y_val, feature_names):
    # how much does validation AUC drop when you shuffle one predictor?
    perm = {}
    for name, s in searches.items():
        model = clone(s.best_estimator_).fit(X_train, y_train)
        res = permutation_importance(model, X_val, y_val, scoring="roc_auc", n_repeats=20,
                                     random_state=SEED, n_jobs=-1)
        perm[name] = pd.Series(res.importances_mean, index=feature_names)
    perm = pd.DataFrame(perm)
    save_table(perm.round(4), "T15_permutation_importance")
    return perm


def regression_table(preds, y_ret_test):
    sse_mean = ((y_ret_test - preds["Historical mean"]) ** 2).sum()
    rows = {}
    for name, p in preds.items():
        rows[name] = {"MAE (%)": mean_absolute_error(y_ret_test, p) * 100,
                      "RMSE (%)": np.sqrt(mean_squared_error(y_ret_test, p)) * 100,
                      "Out-of-sample R² vs historical mean (%)": (1 - ((y_ret_test - p) ** 2).sum() / sse_mean) * 100,
                      "Direction accuracy": ((p > 0) == (y_ret_test > 0)).mean()}
    table = pd.DataFrame(rows).T
    save_table(table.round(4), "T18_test_regression")
    return table


def accuracy_vs_profit(test_proba, y_test, test_metrics, perf, perf_gross, ret_next):
    names = list(test_proba.columns)
    acc = pd.DataFrame({
        "Accuracy": test_metrics.loc[names, "Accuracy"],
        "ROC-AUC": test_metrics.loc[names, "ROC-AUC"],
        "Sharpe before costs": perf_gross.loc[names, "Sharpe ratio"],
        "Sharpe after costs": perf.loc[names, "Sharpe ratio"],
        "Turnover (x per year)": perf.loc[names, "Turnover (x per year)"],
        "Time in SPY (%)": perf.loc[names, "Time in SPY (%)"],
    })
    acc["Rank by accuracy"] = acc["Accuracy"].rank(ascending=False).astype(int)
    acc["Rank by Sharpe after costs"] = acc["Sharpe after costs"].rank(ascending=False).astype(int)
    save_table(acc.round(3), "T20_accuracy_vs_sharpe")

    # McNemar test: is the accuracy gap between two models real or just noise? (exact test on the
    # days where one model was right and the other wasn't)
    hit = {m: ((test_proba[m] > 0.5).astype(int) == y_test) for m in names}
    rows = []
    for a, b in combinations(names, 2):
        only_a, only_b = int((hit[a] & ~hit[b]).sum()), int((~hit[a] & hit[b]).sum())
        rows.append({"Model A": a, "Model B": b,
                     "Accuracy A (%)": hit[a].mean() * 100, "Accuracy B (%)": hit[b].mean() * 100,
                     "Right only A": only_a, "Right only B": only_b,
                     "McNemar exact p-value": stats.binomtest(only_a, only_a + only_b, 0.5).pvalue})
    save_table(pd.DataFrame(rows).set_index(["Model A", "Model B"]).round(3), "T20b_mcnemar_accuracy")

    # Where do the gains and losses come from? SPY's return on days each model was in vs out
    r_next = ret_next.loc[test_proba.index]                   # SPY return earned the day after the decision
    r_today = ret_next.shift(1).loc[test_proba.index]         # SPY return ON the decision day
    spring = slice("2020-03", "2020-06")                      # COVID crash and rebound
    rows = {}
    for m, p in test_proba.items():
        invested, cash = p > 0.5, p <= 0.5
        rows[m] = {
            "Days invested": int(invested.sum()),
            "Avg SPY return on invested days (bps)": r_next[invested].mean() * 1e4,
            "Days in cash": int(cash.sum()),
            "Avg SPY return on cash days (bps)": r_next[cash].mean() * 1e4,
            "Std. error, cash-day average (bps)": r_next[cash].sem() * 1e4,
            "Avg SPY return ON the decision day, cash days (bps)": r_today[cash].mean() * 1e4,
            "Avg SPY return ON the decision day, invested days (bps)": r_today[invested].mean() * 1e4,
            "Cash days that followed an up day (%)": (r_today[cash] > 0).mean() * 100,
            "Cash days in 2020": int(cash.loc["2020"].sum()),
            "Cash days in Mar-Jun 2020": int(cash.loc[spring].sum()),
            "SPY gain missed on Mar-Jun 2020 cash days (%)": ((1 + r_next.loc[spring][cash.loc[spring]]).prod() - 1) * 100,
            "Best 10 SPY days missed": int(cash.loc[r_next.nlargest(10).index].sum()),
            "Worst 10 SPY days avoided": int(cash.loc[r_next.nsmallest(10).index].sum()),
        }
    days = pd.DataFrame(rows).T
    save_table(days.round(2), "T21_invested_vs_cash_days")
    print(acc.round(3))
    return acc


def bootstrap_table(curves, rf, names, block=20, reps=2000):
    # Circular block bootstrap: resample 20-day blocks (keeps the volatility clustering) and
    # compare each strategy's Sharpe with buy-and-hold
    rows = {}
    b, f = curves["Buy-and-hold"].values, rf.values
    for name in names:
        rng = np.random.default_rng(SEED)
        a, n = curves[name].values, len(curves[name])
        diffs = np.empty(reps)
        for i in range(reps):
            starts = rng.integers(0, n, size=n // block + 1)          # random start day of each 20-day block
            # join the blocks up; "% n" wraps a block that runs off the end back round to the start
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n] % n
            # pd.Series so it goes through the same sharpe() as everywhere else
            strategy_sr = sharpe(pd.Series(a[idx]), pd.Series(f[idx]))
            buy_hold_sr = sharpe(pd.Series(b[idx]), pd.Series(f[idx]))
            diffs[i] = strategy_sr - buy_hold_sr
        rows[name] = {"Sharpe difference vs buy-and-hold": sharpe(curves[name], rf) - sharpe(curves["Buy-and-hold"], rf),
                      "95% CI lower": np.percentile(diffs, 2.5), "95% CI upper": np.percentile(diffs, 97.5),
                      "Share of resamples above 0 (%)": (diffs > 0).mean() * 100}
    table = pd.DataFrame(rows).T
    save_table(table.round(3), "T25_bootstrap_sharpe")
    return table


def deflated_sharpe(r, rf, n_trials, var_trials):
    # Bailey and Lopez de Prado (2014): the chance the Sharpe beats what the best of n trials with
    # no skill at all would get by luck
    ex = (r - rf).values
    sr = ex.mean() / ex.std()                          # daily units
    T = len(ex)
    g3, g4 = stats.skew(ex), stats.kurtosis(ex, fisher=False)
    emc = 0.5772156649
    sr0 = np.sqrt(var_trials) * ((1 - emc) * stats.norm.ppf(1 - 1 / n_trials)
                                 + emc * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return stats.norm.cdf(z), sr0 * np.sqrt(TRADING_DAYS)


def deflated_sharpe_table(searches, split, bt, valid_years, val_curves, val_idx, test_curves, test_idx, candidate):
    # The "trials" = validation Sharpe of every setting in the grid searches, plus the two vol
    # versions. Realistically this undercounts everything I tried, so treat it as a lower bound
    trials = []
    for search in searches.values():
        for params in search.cv_results_["params"]:
            p = walk_forward(clone(search.estimator).set_params(**params), valid_years, split["X_all"], split["y_all"])
            res = bt.run((p > 0.5).astype(float))
            excess = res["ret"] - res["rf"]
            trials.append(excess.mean() / excess.std())       # daily (not annualised) Sharpe ratio
    for extra in ("Vol-managed SPY", f"{candidate} + vol-managed"):
        ex = val_curves[extra] - bt.rf.loc[val_idx]
        trials.append(ex.mean() / ex.std())
    trials = np.array(trials)
    print(f"Trials: {len(trials)} configurations")

    rf_test, rows = bt.rf.loc[test_idx], {}
    for name in [candidate, f"{candidate} + vol-managed", "Vol-managed SPY", "Buy-and-hold"]:
        dsr, sr0_ann = deflated_sharpe(test_curves[name], rf_test, len(trials), trials.var())
        psr0, _ = deflated_sharpe(test_curves[name], rf_test, 2, 0.0)
        rows[name] = {"Test Sharpe after costs": sharpe(test_curves[name], rf_test),
                      "Luck benchmark (expected max Sharpe of skill-less trials)": sr0_ann,
                      "Deflated Sharpe ratio (%)": dsr * 100, "Probability Sharpe > 0 (%)": psr0 * 100}
    # The textbook way: apply it to the candidate's validation Sharpe, since that's where it was picked
    dsr_val, _ = deflated_sharpe(val_curves[candidate], bt.rf.loc[val_idx], len(trials), trials.var())
    rows[f"{candidate} (validation period, where it was selected)"] = {
        "Test Sharpe after costs": np.nan, "Luck benchmark (expected max Sharpe of skill-less trials)": sr0_ann,
        "Deflated Sharpe ratio (%)": dsr_val * 100, "Probability Sharpe > 0 (%)": np.nan}
    table = pd.DataFrame(rows).T
    save_table(table.round(3), "T26_deflated_sharpe")
    return table
