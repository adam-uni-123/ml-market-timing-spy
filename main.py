"""
FI4008 Assessment 1 - ML trading strategy on SPY

This runs the whole analysis from start to finish. The question is whether a
model can decide each evening to hold SPY or cash for the next day, and
whether that beats just buying and holding once trading costs are included.

Run it from this folder (it needs data/raw/):  python main.py
Takes about 7 minutes and rewrites everything in tables/ and figures/.

Student ID: 52317064
"""

import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib

import data
import features
import models
import backtest
import evaluation
import plots
from data import SEED, TRAIN_END, VALID_END, save_table


def before_after_costs(gross, net, gross_cols, name):
    # puts a few before-cost columns next to the full after-cost (10 bps) table so they're easy to compare
    table = pd.concat({"Before costs": gross[gross_cols], "After 10 bps costs": net}, axis=1).round(2)
    save_table(table, name)


np.random.seed(SEED)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
print(f"Python {platform.python_version()} | numpy {np.__version__} | pandas {pd.__version__} | "
      f"scipy {scipy.__version__} | scikit-learn {sklearn.__version__} | matplotlib {matplotlib.__version__}")

# 1. Load the data, check nothing has changed since download, clean it and describe it
print("\n--- 1. Data ---")
meta, raw, fred_files = data.load_raw()
data.quality_checks(raw, fred_files)
market = data.clean_data(raw, fred_files)
eda = data.describe_market(market)
plots.market_figures(market, eda)

# 2. Build the target and the 37 predictors, then test them for look-ahead
print("\n--- 2. Features ---")
model_data, feature_values, feature_names = features.make_model_table(market["clean"])
features.leakage_tests(market["clean"], feature_values, model_data)
ic, ic_band = features.predictor_diagnostics(model_data, feature_names)
plots.univariate_signal(ic, ic_band)

# 3. Tune the models on 2006-2016 only, then run the placebo and positive control
print("\n--- 3. Models ---")
split = models.split_data(model_data, feature_names)
searches, fold_df = models.tune(split["X_train"], split["y_train"])
plots.cv_folds(fold_df)
evaluation.placebo_and_positive_control(searches, split["X_train"], split["y_train"], model_data)

# 4. Predict 2017-2019 one year at a time (walk-forward)
print("\n--- 4. Validation ---")
VALID_YEARS = range(TRAIN_END.year + 1, VALID_END.year + 1)
val_proba = models.predict_all(searches, VALID_YEARS, split)
y_val, X_val = split["y_all"][split["val_mask"]], split["X_all"][split["val_mask"]]
print(evaluation.metrics_table(y_val, val_proba, "T14_validation_predictive").round(3))
plots.confusion_matrices(y_val, val_proba)
perm = evaluation.permutation_table(searches, split["X_train"], split["y_train"], X_val, y_val, feature_names)
plots.permutation_importance_plot(perm)

# 5. Backtest the validation years. The design is fixed after this point
print("\n--- 5. Backtest ---")
bt = backtest.Backtester(model_data, market["clean"], market["fred"])
val_idx = model_data.index[split["val_mask"]]
val_perf_gross, _ = bt.strategy_table(val_proba, val_idx, cost_bps=0)
val_perf, val_curves = bt.strategy_table(val_proba, val_idx)
before_after_costs(val_perf_gross, val_perf, ["Annualised return (%)", "Sharpe ratio"], "T16_validation_backtest")
plots.validation_equity(val_curves, list(val_proba.columns))

# I set this rule before looking: the model with the best validation Sharpe after costs becomes
# the candidate, and it's the one that also gets the volatility overlay
CANDIDATE = val_perf.loc[list(val_proba.columns), "Sharpe ratio"].idxmax()
COMBO = f"{CANDIDATE} + vol-managed"
val_perf_all, val_curves_all = bt.strategy_table(val_proba, val_idx, overlay=CANDIDATE)
val_perf_all_gross, _ = bt.strategy_table(val_proba, val_idx, cost_bps=0, overlay=CANDIDATE)
before_after_costs(val_perf_all_gross, val_perf_all, ["Annualised return (%)", "Sharpe ratio"],
                   "T16b_validation_backtest_with_vol")
print(f"Candidate: {CANDIDATE}; volatility target {bt.vol_target:.1%}")
print(val_perf_all[["Annualised return (%)", "Sharpe ratio", "Max drawdown (%)"]].round(2))

# 6. Test period, 2020 onwards. Run once at the end, nothing changed after seeing it
print("\n--- 6. Test ---")
TEST_YEARS = range(VALID_END.year + 1, model_data.index.max().year + 1)
test_proba = models.predict_all(searches, TEST_YEARS, split)
test_idx = model_data.index[split["test_mask"]]
assert test_proba.index.equals(test_idx)
y_test = split["y_all"][split["test_mask"]]
test_metrics = evaluation.metrics_table(y_test, test_proba, "T17_test_predictive", with_pt=True)
print(test_metrics.round(3))

reg_preds = models.regression_forecasts(model_data, split, TEST_YEARS)
print(evaluation.regression_table(reg_preds, model_data["logret_next"][split["test_mask"]]).round(3))

test_perf_gross, _ = bt.strategy_table(test_proba, test_idx, cost_bps=0, overlay=CANDIDATE)
test_perf, test_curves = bt.strategy_table(test_proba, test_idx, overlay=CANDIDATE)
before_after_costs(test_perf_gross, test_perf, ["Cumulative return (%)", "Annualised return (%)", "Sharpe ratio"],
                   "T19_test_backtest")
print(test_perf[["Cumulative return (%)", "Annualised return (%)", "Sharpe ratio", "Max drawdown (%)"]].round(2))
plots.test_equity(test_curves, list(test_proba.columns), CANDIDATE)

# Does the most accurate model make the most money?
acc = evaluation.accuracy_vs_profit(test_proba, y_test, test_metrics, test_perf, test_perf_gross, bt.ret_next)
plots.accuracy_vs_sharpe(acc, test_metrics.loc["Always up (benchmark)", "Accuracy"],
                         test_perf.loc["Buy-and-hold", "Sharpe ratio"])

# 7. Robustness checks, and could the results just be luck?
print("\n--- 7. Robustness ---")
proba_5d = models.five_day_proba(searches, model_data, split["X_all"], TEST_YEARS)
backtest.robustness_tables(bt, test_proba, proba_5d, test_perf, test_curves, CANDIDATE)
names = list(test_proba.columns) + ["SMA 50/200 crossover", "Vol-managed SPY", COMBO]
print(evaluation.bootstrap_table(test_curves, bt.rf.loc[test_idx], names).round(3))
print(evaluation.deflated_sharpe_table(searches, split, bt, VALID_YEARS, val_curves_all, val_idx,
                                       test_curves, test_idx, CANDIDATE).round(3))

# 8. Summary of the leakage checks. Each one is an assert earlier in the run, so getting
#    this far means they all passed
print("\n--- 8. Checks ---")
for item in ["Raw files match their recorded SHA-256 fingerprints",
             "Calendars aligned; no missing predictor values after warm-up",
             "Truncation test: predictors identical when computed with data up to each date only",
             "Target alignment: target(t) equals the realised return of day t+1",
             "Chronology and purging: train < validation < test, no label overlap",
             "Placebo (sanity check of the validation procedure): shuffled labels give ROC-AUC ≈ 0.5",
             "Positive control: a deliberately leaked predictor is detected (ROC-AUC > 0.9)",
             "Walk-forward: every model is fitted only on data before the year it predicts",
             "Backtest arithmetic verified on a hand-computed example"]:
    print("  ✓", item)
print(f"Random seed {SEED}; sample {market['calendar'][0].date()} to {market['sample_end'].date()}; "
      f"data downloaded {meta['download_started_utc']}")
