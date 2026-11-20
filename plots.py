"""
plots.py - all the figures in the report

All 13 figures (F01-F13) share the same style. Each function takes the results
it needs and saves a PNG into figures/.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                         # just save the files, don't open windows
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker
from scipy import stats
from sklearn.metrics import confusion_matrix

from data import FIG_DIR, TRADED, SECTORS, TRAIN_END, VALID_END

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SPLIT_SHADE = {"Validation": "#ecebe7", "Test": "#d9d8d2"}
STYLES = {"Logistic regression": (C1, "-"), "Random forest": (C2, "-"), "Gradient boosting": (C3, "-"),
          "Buy-and-hold": (INK, "-"), "SMA 50/200 crossover": (INK_2, "--"), "Vol-managed SPY": (INK_2, ":")}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.0, "legend.frameon": False, "axes.axisbelow": True,
})


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)


def year_axis(ax, step=1):
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def shade_splits(ax, start, end, label=True):
    # grey shading for the validation and test periods
    for name, lo, hi in [("Validation", TRAIN_END, VALID_END), ("Test", VALID_END, end)]:
        ax.axvspan(lo, hi, color=SPLIT_SHADE[name], lw=0, zorder=0)
        if label:
            ax.text(lo + (hi - lo) / 2, 1.01, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=INK_2)
    if label:
        ax.text(start + (TRAIN_END - start) / 2, 1.01, "Train", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=INK_2)


def market_figures(market, eda):
    # F01-F06, the data section: prices and VIX, returns, volume, rates, autocorrelation
    # and sector correlations
    adj, vix, spy, fred, r, z = market["adj"], market["vix"], market["spy"], market["fred"], market["r"], market["z"]
    start, end = market["calendar"][0], market["calendar"][-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    ax1.plot(adj[TRADED] / adj[TRADED].iloc[0], color=C1)
    ax1.set_yscale("log")
    ax1.set_yticks([0.5, 1, 2, 4, 8])
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax1.set_ylabel("Growth of $1 (log scale)")
    shade_splits(ax1, start, end)
    ax2.plot(vix["Close"], color=C2)
    ax2.set_ylabel("VIX level")
    shade_splits(ax2, start, end, label=False)
    year_axis(ax2, 2)
    ax1.set_title("SPY total return (top) and VIX (bottom), with train / validation / test periods",
                  loc="left", pad=16)
    save(fig, "F01_spy_vix")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax1.plot(r * 100, color=C1, lw=0.5)
    ax1.set_ylabel("Daily log return (%)")
    year_axis(ax1, 4)
    ax1.set_title("(a) SPY daily returns", loc="left")
    grid = np.linspace(-8, 8, 400)
    ax2.hist(z, bins=120, density=True, color=C1, alpha=0.85)
    ax2.plot(grid, stats.norm.pdf(grid), color=C2, lw=1.2)
    ax2.set_yscale("log")
    ax2.set_ylim(3e-4, 1)
    ax2.text(2.6, 0.03, "Normal", color=C2, fontsize=8)
    ax2.text(-9.5, 0.3, "SPY", color=C1, fontsize=8)
    ax2.set_xlabel("Standard deviations from mean")
    ax2.set_ylabel("Density (log scale)")
    ax2.set_title("(b) Standardised returns vs normal", loc="left")
    fig.tight_layout()
    save(fig, "F02_returns_distribution")

    dollar_vol = spy["Close"] * spy["Volume"] / 1e9
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 3.8), sharex=True, gridspec_kw={"hspace": 0.25})
    for ax, series, unit, title in [(ax1, spy["Volume"] / 1e6, "Shares (millions)", "(a) SPY share volume"),
                                    (ax2, dollar_vol, "US$ billions", "(b) SPY dollar volume")]:
        ax.plot(series, color=INK_2, lw=0.3, alpha=0.5, label="Daily")
        ax.plot(series.rolling(63).mean(), color=C1, lw=1.2, label="63-day average")
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left")
    ax1.legend(loc="upper right", fontsize=7)
    year_axis(ax2, 2)
    save(fig, "F03_trading_activity")
    print(f"Median daily dollar volume: ${dollar_vol.median():.1f}bn")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.0), sharex=True, gridspec_kw={"hspace": 0.32})
    for s, c, lab in [("DGS3MO", C1, "3-month"), ("DGS2", C2, "2-year"), ("DGS10", C3, "10-year")]:
        ax1.plot(fred[s], color=c, label=lab)
    ax1.set_ylabel("Yield (% p.a.)")
    ax1.set_title("(a) US Treasury yields", loc="left")
    ax1.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, fontsize=7)
    ax2.plot(fred["T10Y3M"], color=C1, label="10y − 3m")
    ax2.plot(fred["T10Y2Y"], color=C2, label="10y − 2y")
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.fill_between(fred.index, fred["T10Y3M"], 0, where=fred["T10Y3M"] < 0,
                     color=C1, alpha=0.15, lw=0, label="Inverted (10y − 3m < 0)")
    ax2.set_ylabel("Percentage points")
    ax2.set_title("(b) Term spreads", loc="left")
    ax2.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, fontsize=7)
    year_axis(ax2, 2)
    save(fig, "F04_rates")

    lags, band = eda["lags"], eda["band"]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    w = 0.38
    ax.bar(np.array(lags) - w / 2, eda["acf_ret"], width=w, color=C1, label="Returns (direction)")
    ax.bar(np.array(lags) + w / 2, eda["acf_abs"], width=w, color=C2, label="Absolute returns (size / risk)")
    ax.axhspan(-band, band, color=GRID, lw=0, zorder=0, label="Approx. 95% band if no memory")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(list(lags))
    ax.set_xlabel("Lag (trading days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation of SPY daily returns and of absolute returns", loc="left")
    ax.legend(loc="upper right", fontsize=7)
    save(fig, "F05_autocorrelation")

    corr = market["log_ret"][[TRADED] + SECTORS].dropna().corr()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(corr, cmap="Blues", vmin=0.4, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45)
    ax.set_yticks(range(len(corr)), corr.index)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if corr.iloc[i, j] > 0.72 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation of daily returns")
    ax.set_title("Correlation of daily returns, SPY and nine sector ETFs", loc="left")
    save(fig, "F06_sector_correlation")


def univariate_signal(ic, ic_band):
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    ax.barh(ic.index, ic.values, color=[C1 if abs(v) > ic_band else "#a9c7ee" for v in ic], height=0.7)
    ax.axvspan(-ic_band, ic_band, color=GRID, lw=0, zorder=0)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Spearman correlation with next-day SPY return (training period)")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", visible=False)
    ax.set_title("Univariate predictive signal of each predictor, 2006-2016\n"
                 "(dark bars lie outside the approx. 95% no-signal band)", loc="left")
    save(fig, "F07_univariate_signal")


def cv_folds(fold_df):
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    xs = np.arange(len(fold_df))
    for k, (name, c) in enumerate(zip(fold_df.columns[1:], (C1, C2, C3))):
        ax.bar(xs + (k - 1) * 0.26, fold_df[name], width=0.26, color=c, label=name)
    ax.axhline(0.5, color=INK, lw=0.8)
    ax.set_xticks(xs, fold_df["Scored period"])
    ax.set_ylim(0.4, max(0.6, fold_df.iloc[:, 1:].values.max() + 0.02))
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("Period scored in each time-series CV fold")
    ax.set_title("Out-of-fold ROC-AUC of the tuned models, training period (0.5 = no skill)", loc="left")
    ax.legend(loc="upper left", ncol=3, fontsize=7)
    ax.grid(axis="x", visible=False)
    save(fig, "F08_cv_folds")


def confusion_matrices(y_val, val_proba):
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3))
    for ax, name in zip(axes, val_proba):
        cm = confusion_matrix(y_val, (val_proba[name] > 0.5).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > cm.max() * 0.6 else INK)
        ax.set_xticks([0, 1], ["Down", "Up"])
        ax.set_yticks([0, 1], ["Down", "Up"])
        ax.set_xlabel("Predicted")
        ax.grid(False)
        ax.set_title(name, fontsize=8.5, loc="left")
    axes[0].set_ylabel("Actual")
    fig.suptitle("Confusion matrices, validation period 2017-2019", x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "F09_confusion_validation")


def permutation_importance_plot(perm):
    top = perm.mean(axis=1).sort_values(ascending=False).index[:12]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ys = np.arange(len(top))
    for k, (name, c) in enumerate(zip(perm.columns, (C1, C2, C3))):
        ax.barh(ys + (k - 1) * 0.27, perm.loc[top, name], height=0.27, color=c, label=name)
    ax.set_yticks(ys, top)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Fall in validation ROC-AUC when the predictor is shuffled")
    ax.set_title("Permutation importance, validation period (12 most important predictors)", loc="left", pad=20)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7)
    save(fig, "F10_permutation_importance")


def validation_equity(val_curves, models):
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for name, (c, ls) in STYLES.items():
        ax.plot((1 + val_curves[name]).cumprod(), color=c, ls=ls, lw=1.2 if name in models else 1.0, label=name)
    ax.set_ylabel("Growth of $1 (after costs)")
    ax.set_title("Validation period 2017-2019: strategies vs benchmarks, 10 bps costs", loc="left")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    year_axis(ax)
    save(fig, "F11_validation_equity")


def test_equity(test_curves, models, candidate):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    for name, (c, ls) in {**STYLES, f"{candidate} + vol-managed": (C4, "-")}.items():
        wealth = (1 + test_curves[name]).cumprod()
        ax1.plot(wealth, color=c, ls=ls, lw=1.2 if name in models else 1.0, label=name)
        dd = (wealth / wealth.cummax() - 1) * 100
        if name == "Buy-and-hold":                 # shaded so it stands out from the lines
            ax2.fill_between(dd.index, dd, 0, color="#c9c8c4", lw=0, label=name)
        elif name in (candidate, "Vol-managed SPY"):
            ax2.plot(dd, color=c, ls="-", lw=1.4 if name == "Vol-managed SPY" else 1.0, label=name)
    ax1.set_ylabel("Growth of $1 (after costs)")
    ax1.legend(loc="upper left", fontsize=7, ncol=2)
    ax1.set_title("Test period: strategies vs benchmarks after 10 bps costs (top), drawdowns (bottom)", loc="left")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(loc="lower right", fontsize=7)
    year_axis(ax2)
    save(fig, "F12_test_equity")


def accuracy_vs_sharpe(acc, always_up_acc, bh_sharpe):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for name, c in zip(acc.index, (C1, C2, C3)):
        x = acc.loc[name, "Accuracy"] * 100
        y0, y1 = acc.loc[name, ["Sharpe before costs", "Sharpe after costs"]]
        ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
        ax.scatter(x, y0, s=55, facecolor="white", edgecolor=c, lw=1.5, zorder=3)
        ax.scatter(x, y1, s=55, color=c, edgecolor="white", lw=1.5, zorder=3)
        ax.text(x + 0.05, y0, name, color=INK, fontsize=7.5, va="center")
    ax.scatter(always_up_acc * 100, bh_sharpe, s=55, color=INK, edgecolor="white", lw=1.5, zorder=3)
    ax.text(always_up_acc * 100 - 0.05, bh_sharpe, "Buy-and-hold\n(= always up)",
            ha="right", va="center", fontsize=7.5, color=INK)
    ax.set_xlabel("Directional accuracy, test period (%)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Accuracy versus investment performance, test period\n"
                 "(hollow = before costs, filled = after 10 bps costs)", loc="left")
    save(fig, "F13_accuracy_vs_sharpe")
