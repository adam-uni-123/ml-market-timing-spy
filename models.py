"""
models.py - the three models and how they're validated

Logistic regression, random forest and gradient boosting. Each one sits inside
a scikit-learn Pipeline, so the scaling and winsorising are only ever fitted on
the training fold. Tuning uses 2006-2016 only, with TimeSeriesSplit (5 folds,
5-day gap). After that each model is refitted every January on all the data
before it and predicts the coming year (walk-forward).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              RandomForestRegressor, HistGradientBoostingRegressor)
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from data import SEED, TRAIN_END, VALID_END, save_table

H = 1                                         # forecast horizon in days
tscv = TimeSeriesSplit(n_splits=5, gap=5)     # 5-day gap so the fit and score blocks never touch


def purged_fit_mask(index, cutoff, horizon=H):
    # Training days = everything up to the cutoff minus the last `horizon` days. Their target is a
    # return that finishes after the cutoff, so we wouldn't know it yet. This is called "purging"
    fit_dates = index[index <= cutoff][:-horizon]
    return index.isin(fit_dates)


class Winsorizer(BaseEstimator, TransformerMixin):
    # Caps each predictor at its 1st and 99th percentile so a handful of crash days don't dominate
    # the logistic regression. Written as a scikit-learn step so it can go inside the Pipeline:
    # fit() learns the limits from the training data only, transform() just applies them.
    def __init__(self, q=0.01):
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.q, axis=0)
        self.upper_ = np.nanquantile(X, 1 - self.q, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


MODELS = {
    "Logistic regression": (
        Pipeline([("winsor", Winsorizer(0.01)), ("scale", StandardScaler()),
                  ("pca", "passthrough"), ("model", LogisticRegression(max_iter=5000))]),
        {"model__C": [0.001, 0.01, 0.1, 1.0],
         "pca": ["passthrough", PCA(n_components=0.90, random_state=SEED)]},
    ),
    "Random forest": (
        Pipeline([("model", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1))]),
        {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [50, 200],
         "model__max_features": ["sqrt", 0.5]},
    ),
    "Gradient boosting": (
        Pipeline([("model", HistGradientBoostingClassifier(random_state=SEED, early_stopping=False,
                                                           min_samples_leaf=50))]),
        {"model__learning_rate": [0.01, 0.05], "model__max_depth": [2, 3],
         "model__max_iter": [100, 300], "model__l2_regularization": [0.0, 1.0]},
    ),
}

# Regression versions. Only here so I can report MAE and RMSE as well
REG_MODELS = {
    "Ridge regression": (Pipeline([("winsor", Winsorizer(0.01)), ("scale", StandardScaler()),
                                   ("model", Ridge())]),
                         {"model__alpha": [1.0, 100.0, 1000.0, 10000.0]}),
    "Random forest": (Pipeline([("model", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1))]),
                      {"model__max_depth": [3, 5], "model__min_samples_leaf": [50, 200]}),
    "Gradient boosting": (Pipeline([("model", HistGradientBoostingRegressor(random_state=SEED, early_stopping=False,
                                                                            min_samples_leaf=50, max_iter=100))]),
                          {"model__learning_rate": [0.01, 0.05], "model__max_depth": [2, 3]}),
}


def split_data(data, feature_names):
    X_all, y_all = data[feature_names], data["y_1d"].astype(int)
    fit_train = purged_fit_mask(data.index, TRAIN_END)
    val_mask = (data.index > TRAIN_END) & (data.index <= VALID_END)
    test_mask = data.index > VALID_END
    X_train, y_train = X_all[fit_train], y_all[fit_train]

    # check the order really is train < validation < test, and no training target uses a 2017 return
    assert X_train.index.max() < data.index[val_mask].min() < data.index[val_mask].max() < data.index[test_mask].min()
    assert data.index.to_series().shift(-H).loc[X_train.index].max() <= TRAIN_END
    print(f"Train {len(X_train)} days (purged), validation {val_mask.sum()}, test {test_mask.sum()}")
    return dict(X_all=X_all, y_all=y_all, fit_train=fit_train, X_train=X_train, y_train=y_train,
                val_mask=val_mask, test_mask=test_mask)


def tune(X_train, y_train):
    # Grid search on the training years, scored on ROC-AUC. Accuracy would reward just saying
    # "up" every day, since the market rises on about 55% of days
    searches, rows, folds = {}, [], {}
    for name, (pipe, grid) in MODELS.items():
        search = GridSearchCV(pipe, grid, cv=tscv, scoring={"auc": "roc_auc", "acc": "accuracy"},
                              refit="auc", n_jobs=-1)
        search.fit(X_train, y_train)
        searches[name] = search
        best = search.best_index_
        folds[name] = [search.cv_results_[f"split{k}_test_auc"][best] for k in range(tscv.n_splits)]
        best_params = {}                                  # readable names, e.g. "C" not "model__C"
        for key, value in search.best_params_.items():
            if isinstance(value, PCA):
                value = "PCA(90%)"
            best_params[key.replace("model__", "")] = value
        rows.append({
            "Model": name,
            "Combinations tried": len(search.cv_results_["params"]),
            "Best hyperparameters": best_params,
            "CV ROC-AUC (mean)": search.cv_results_["mean_test_auc"][best],
            "CV ROC-AUC (std)": search.cv_results_["std_test_auc"][best],
            "CV accuracy (mean)": search.cv_results_["mean_test_acc"][best],
        })
    cv_table = pd.DataFrame(rows).set_index("Model")
    cols = ["CV ROC-AUC (mean)", "CV ROC-AUC (std)", "CV accuracy (mean)"]
    cv_table[cols] = cv_table[cols].round(3)
    save_table(cv_table, "T11_cv_tuning")
    print(cv_table)

    # AUC in each fold separately: does the chosen model work in every period or only some?
    fold_df = pd.DataFrame(folds, index=[f"Fold {k}" for k in range(1, tscv.n_splits + 1)])
    scored_periods = []
    for _, score_rows in tscv.split(X_train):         # the years each fold was scored on
        first, last = X_train.index[score_rows[0]], X_train.index[score_rows[-1]]
        scored_periods.append(f"{first.year}-{last.year}")
    fold_df.insert(0, "Scored period", scored_periods)
    save_table(fold_df.round(3), "T12_cv_folds")
    return searches, fold_df


def walk_forward(estimator, years, X, y, horizon=H, regression=False):
    # Each January: refit on data up to 31 Dec of the previous year, predict that year
    out = []
    for year in years:
        fit = purged_fit_mask(X.index, pd.Timestamp(f"{year - 1}-12-31"), horizon)
        pred = X.index.year == year
        model = clone(estimator).fit(X[fit], y[fit])
        # check: the model never trains on the year it predicts
        assert X.index[fit].max() < X.index[pred].min()
        if regression:
            forecast = model.predict(X[pred])                 # predicted next-day log return
        else:
            forecast = model.predict_proba(X[pred])[:, 1]     # predicted probability of an up day
        out.append(pd.Series(forecast, index=X.index[pred]))
    return pd.concat(out)


def predict_all(searches, years, split):
    # walk-forward P(up) for all three tuned models
    return pd.DataFrame({name: walk_forward(search.best_estimator_, years, split["X_all"], split["y_all"])
                         for name, search in searches.items()})


def regression_forecasts(data, split, test_years):
    # Same walk-forward, but forecasting the return itself (only for MAE / RMSE), plus two simple
    # benchmarks to compare against
    y_ret, preds = data["logret_next"], {}
    for name, (pipe, grid) in REG_MODELS.items():
        search = GridSearchCV(pipe, grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
        search.fit(split["X_train"], y_ret[split["fit_train"]])
        preds[name] = walk_forward(search.best_estimator_, test_years, split["X_all"], y_ret, regression=True)
    preds["Zero forecast"] = pd.Series(0.0, index=data.index[split["test_mask"]])
    # historical mean = average return of all the days we'd have known about at each January refit
    hist_mean = []
    for year in test_years:
        known = purged_fit_mask(data.index, pd.Timestamp(f"{year - 1}-12-31"))
        hist_mean.append(pd.Series(y_ret[known].mean(), index=data.index[data.index.year == year]))
    preds["Historical mean"] = pd.concat(hist_mean)
    return preds


def five_day_proba(searches, data, X_all, test_years):
    # Robustness check: same models, but predicting whether SPY is up over the next 5 days
    # (so the purge is 5 days as well)
    return pd.DataFrame({name: walk_forward(search.best_estimator_, test_years, X_all, data["y_5d"], horizon=5)
                         for name, search in searches.items()})
