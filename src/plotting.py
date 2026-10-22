"""Publication-quality figures.  Every function saves a PNG (200 dpi) and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # headless backend: runs on any machine without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from . import config

PALETTE = {"data": "#4C566A", "blue": "#2E6F9E", "orange": "#D9822B", "green": "#3C8D5A", "red": "#B5403A",
           "purple": "#7A5C9E", "teal": "#2A9D9D", "gold": "#C9A227", "grey": "#8A8F98"}
MODEL_COLORS = {"Normal": "#8A8F98", "Student-t": "#2E6F9E", "Log-normal": "#7A5C9E", "Gamma": "#D9822B",
                "Weibull": "#3C8D5A", "Beta (scaled)": "#2A9D9D", "2-component normal mixture": "#B5403A"}


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.labelsize": 10.5, "legend.fontsize": 9, "legend.frameon": True,
                         "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def _save(fig, name: str) -> Path:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Exploratory data analysis
# --------------------------------------------------------------------------- #
def plot_overview(hourly: pd.Series) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), gridspec_kw={"height_ratios": [1.3, 1, 1]})
    ax = axes[0]
    ax.plot(hourly.index, hourly.values, lw=0.35, color=PALETTE["blue"], label="Hourly mean output")
    monthly = hourly.resample("MS").mean()
    ax.step(list(monthly.index) + [hourly.index[-1]], list(monthly.values) + [monthly.values[-1]], where="post",
            color=PALETTE["orange"], lw=2, label="Monthly mean")
    ax.set(title="Fleet-total solar PV output, Alabama, 2006 (137 plants, hourly means)",
           ylabel="Power (MW)", xlabel="Date (local standard time)")
    ax.legend(loc="upper right")
    for ax, (start, label, color) in zip(axes[1:], [("2006-06-19", "Summer week (19-25 June)", PALETTE["red"]),
                                                    ("2006-12-18", "Winter week (18-24 December)", PALETTE["purple"])]):
        week = hourly[start:pd.Timestamp(start) + pd.Timedelta(days=7)]
        ax.plot(week.index, week.values, color=color, lw=1.5, marker="o", ms=2.5)
        ax.set(title=label, ylabel="Power (MW)", xlabel="Date / hour (local standard time)")
    fig.tight_layout()
    return _save(fig, "fig01_hourly_series_overview.png")


def plot_diurnal_seasonal(frame: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    grid = frame["hourly_total_mw"].to_numpy().reshape(-1, 24).T
    im = axes[0].imshow(grid, aspect="auto", origin="lower", cmap="magma", extent=[1, grid.shape[1], -0.5, 23.5])
    axes[0].set(title="Output by hour of day and day of year", xlabel="Day of year", ylabel="Hour of day (local standard time)")
    fig.colorbar(im, ax=axes[0], label="Power (MW)")
    seasons = {"Winter (Dec-Feb)": [12, 1, 2], "Spring (Mar-May)": [3, 4, 5], "Summer (Jun-Aug)": [6, 7, 8],
               "Autumn (Sep-Nov)": [9, 10, 11]}
    for (label, months), color in zip(seasons.items(), [PALETTE["purple"], PALETTE["green"], PALETTE["red"], PALETTE["orange"]]):
        sub = frame[frame["month"].isin(months)]
        axes[1].plot(range(24), sub.groupby("hour")["hourly_total_mw"].mean().values, label=label, color=color, lw=2)
    axes[1].set(title="Mean diurnal profile by season", xlabel="Hour of day", ylabel="Mean power (MW)")
    axes[1].legend()
    fig.tight_layout()
    return _save(fig, "fig02_diurnal_and_seasonal_pattern.png")


def plot_raw_distribution(hourly: pd.Series) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(hourly.values, bins=60, color=PALETTE["blue"], edgecolor="white")
    zero_share = (hourly == 0).mean()
    axes[0].annotate(f"{100 * zero_share:.1f}% of hours have\nexactly zero output (night)", xy=(0, axes[0].get_ylim()[1] * 0.9),
                     xytext=(1300, axes[0].get_ylim()[1] * 0.85), arrowprops={"arrowstyle": "->"}, fontsize=10)
    axes[0].set(title="Distribution of hourly output (all hours)", xlabel="Power (MW)", ylabel="Number of hours")
    pos = hourly[hourly > 0]
    axes[1].hist(pos.values, bins=60, color=PALETTE["orange"], edgecolor="white")
    axes[1].set(title="Distribution of strictly positive hourly output", xlabel="Power (MW)", ylabel="Number of hours")
    fig.tight_layout()
    return _save(fig, "fig03_distribution_of_raw_output.png")


def plot_envelope(frame: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    start, end = "2006-04-10", "2006-04-17"
    week = frame[start:end]
    axes[0].plot(week.index, week["hourly_total_mw"], color=PALETTE["blue"], lw=1.6, marker="o", ms=3, label="Observed hourly output")
    axes[0].plot(week.index, week["envelope_mw"], color=PALETTE["orange"], lw=1.6, ls="--", label="Empirical clear-sky envelope (95th pct)")
    axes[0].set(title="Output and empirical clear-sky envelope (example week)", ylabel="Power (MW)")
    axes[0].legend()
    axes[1].plot(week.index, week["clearsky_index"], color=PALETTE["green"], lw=1.6, marker="o", ms=3)
    axes[1].axhline(1.0, color=PALETTE["grey"], ls=":")
    axes[1].set(title="Clear-sky index = output / envelope (daylight hours only)", ylabel="Index (dimensionless)", xlabel="Date / hour (local standard time)")
    fig.tight_layout()
    return _save(fig, "fig04_clearsky_envelope_and_index.png")


# --------------------------------------------------------------------------- #
# Stationarity
# --------------------------------------------------------------------------- #
def plot_rolling_and_groups(frame: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    raw = frame["hourly_total_mw"]
    r = raw.rolling(24 * 7, min_periods=24 * 3)
    axes[0, 0].plot(raw.index, r.mean(), color=PALETTE["blue"], label="7-day rolling mean")
    axes[0, 0].plot(raw.index, r.std(), color=PALETTE["orange"], label="7-day rolling std. dev.")
    axes[0, 0].set(title="Raw output: rolling mean and standard deviation", ylabel="Power (MW)", xlabel="Date")
    axes[0, 0].legend()
    hour_means = frame.groupby("hour")["hourly_total_mw"].agg(["mean", "std"])
    axes[0, 1].errorbar(hour_means.index, hour_means["mean"], yerr=hour_means["std"], fmt="o-", color=PALETTE["red"], capsize=3)
    axes[0, 1].set(title="Raw output: mean +/- std. dev. by hour of day", xlabel="Hour of day", ylabel="Power (MW)")
    daily = frame.groupby("day_index")["clearsky_index"].mean()
    dates = frame.index[::24][: len(daily)]
    axes[1, 0].plot(dates, daily.values, color=PALETTE["green"], lw=0.8, alpha=0.7, label="Daily mean index")
    axes[1, 0].plot(dates, daily.rolling(15, center=True, min_periods=5).mean(), color=PALETTE["purple"], lw=2.2, label="15-day rolling mean")
    axes[1, 0].set(title="Clear-sky index: daily mean and 15-day rolling mean", ylabel="Index (dimensionless)", xlabel="Date")
    axes[1, 0].legend()
    sub = frame[frame["in_sample"]]
    sns.boxplot(data=sub, x="month", y="clearsky_index", color="#9BC1BC", ax=axes[1, 1], fliersize=1.5)
    axes[1, 1].set(title="Clear-sky index by month (mean and spread both vary)", xlabel="Month", ylabel="Index (dimensionless)")
    fig.tight_layout()
    return _save(fig, "fig05_rolling_statistics_and_group_effects.png")


def plot_acf_pacf(acf_pacf: dict, path_name: str, title_prefix: str) -> Path:
    """acf_pacf: {label: (acf_array, pacf_array, n)} -> one row of (ACF, PACF) per entry."""
    rows = len(acf_pacf)
    fig, axes = plt.subplots(rows, 2, figsize=(12.5, 3.6 * rows), squeeze=False)
    for r, (label, (acf, pacf, n)) in enumerate(acf_pacf.items()):
        band = 1.96 / np.sqrt(n)
        for c, (values, kind) in enumerate([(acf, "Autocorrelation (ACF)"), (pacf, "Partial autocorrelation (PACF)")]):
            ax = axes[r, c]
            lags = np.arange(len(values))
            ax.vlines(lags[1:], 0, values[1:], color=PALETTE["blue"], lw=1.6)
            ax.plot(lags[1:], values[1:], "o", ms=3, color=PALETTE["blue"])
            ax.axhline(0, color="black", lw=0.8)
            ax.axhspan(-band, band, color=PALETTE["grey"], alpha=0.25, label="95% band (white noise)")
            ax.set(xlabel="Lag (hours)", ylabel="Correlation", ylim=(-1.05, 1.05))
            ax.set_title(f"{title_prefix}: {kind}\n{label}", fontsize=10)
            if r == 0 and c == 0:
                ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, path_name)


# --------------------------------------------------------------------------- #
# Distributions and MLE
# --------------------------------------------------------------------------- #
def plot_fitted_densities(x: np.ndarray, fitted: dict, name: str = "fig08_fitted_densities.png") -> Path:
    grid = np.linspace(max(1e-3, x.min() * 0.9), x.max() * 1.05, 600)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ylog in zip(axes, (False, True)):
        ax.hist(x, bins=50, density=True, color="#D8DEE9", edgecolor="white", label="Observed index")
        for model, (dist, fit) in fitted.items():
            p = dist.params_array(fit)
            ax.plot(grid, np.exp(dist.logpdf(grid, p)), lw=2, color=MODEL_COLORS.get(model), label=f"{model} (lnL={fit.loglik:,.0f})")
        ax.set(xlabel="Clear-sky index (dimensionless)", ylabel="Density" + (" (log scale)" if ylog else ""),
               title="MLE-fitted densities" + (" - log scale" if ylog else ""))
        if ylog:
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 20)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


def plot_qq(x: np.ndarray, fitted: dict, name: str = "fig09_qq_plots.png") -> Path:
    n = len(x)
    probs = (np.arange(1, n + 1) - 0.5) / n
    xs = np.sort(x)
    cols = 4
    rows = int(np.ceil(len(fitted) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.9 * cols, 3.8 * rows), squeeze=False)
    for ax, (model, (dist, fit)) in zip(axes.ravel(), fitted.items()):
        theo = dist.ppf(probs, dist.params_array(fit))
        ax.plot(theo, xs, ".", ms=2.5, color=MODEL_COLORS.get(model), alpha=0.7)
        lo, hi = min(theo.min(), xs.min()), max(theo.max(), xs.max())
        ax.plot([lo, hi], [lo, hi], color="black", lw=1)
        ax.set(title=model, xlabel="Theoretical quantile", ylabel="Empirical quantile")
    for ax in axes.ravel()[len(fitted):]:
        ax.axis("off")
    fig.suptitle("Q-Q plots of the clear-sky index against each MLE-fitted model (45-degree line = perfect fit)", fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save(fig, name)


def plot_likelihood_surface(x: np.ndarray, dist, fit, name: str = "fig07_likelihood_surface_weibull.png") -> Path:
    """Contour plot of the log-likelihood of a 2-parameter model, with the MLE and a 95 % likelihood region."""
    p_hat = dist.params_array(fit)
    a = np.linspace(p_hat[0] * 0.85, p_hat[0] * 1.15, 120)
    b = np.linspace(p_hat[1] * 0.97, p_hat[1] * 1.03, 120)
    ll = np.array([[dist.loglik(x, np.array([ai, bi])) for ai in a] for bi in b])
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    levels = fit.loglik - np.array([0.5, 2, 5, 10, 20, 40, 80, 160])[::-1]
    cs = ax.contour(a, b, ll, levels=sorted(levels), cmap="viridis")
    ax.clabel(cs, fmt=lambda v: f"{v - fit.loglik:.0f}", fontsize=7)
    ci_level = fit.loglik - stats.chi2.ppf(0.95, 2) / 2
    ax.contour(a, b, ll, levels=[ci_level], colors=PALETTE["red"], linewidths=2)
    ax.plot(p_hat[0], p_hat[1], "*", ms=15, color=PALETTE["red"], label="MLE")
    ax.plot([], [], color=PALETTE["red"], lw=2, label="95% likelihood region")
    ax.set(xlabel=f"{dist.param_names[0]} parameter", ylabel=f"{dist.param_names[1]} parameter",
           title=f"{dist.name}: log-likelihood surface (labels = lnL - lnL_max)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, name)


def plot_model_selection(table: pd.DataFrame, name: str = "fig10_distribution_model_selection.png") -> Path:
    order = table.sort_values("AIC")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = [MODEL_COLORS.get(m, PALETTE["grey"]) for m in order["model"]]
    axes[0].barh(order["model"], order["loglik"], color=colors)
    axes[0].set(title="Maximised log-likelihood (higher is better)", xlabel="ln L")
    axes[0].invert_yaxis()
    width = 0.38
    y = np.arange(len(order))
    axes[1].barh(y - width / 2, order["delta_AIC"], height=width, color=PALETTE["blue"], label="Delta AIC")
    axes[1].barh(y + width / 2, order["delta_BIC"], height=width, color=PALETTE["orange"], label="Delta BIC")
    axes[1].set_yticks(y, order["model"])
    axes[1].set_xscale("symlog", linthresh=10)
    axes[1].invert_yaxis()
    axes[1].set(title="Difference to the best model (lower is better, symlog axis)", xlabel="Delta criterion")
    axes[1].legend()
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------- #
# Time-series diagnostics and forecasts
# --------------------------------------------------------------------------- #
def plot_cross_family(dist_table: pd.DataFrame, ts_table: pd.DataFrame, name: str = "fig11_iid_vs_dynamic_models.png") -> Path:
    combined = pd.concat([
        dist_table[["model", "loglik", "BIC"]].assign(family="i.i.d. distribution"),
        ts_table[["model", "loglik", "BIC"]].assign(family="ARMA-type"),
    ]).sort_values("BIC")
    fig, ax = plt.subplots(figsize=(10.5, 6))
    colors = [PALETTE["orange"] if f.startswith("i.i.d") else PALETTE["blue"] for f in combined["family"]]
    ax.barh(combined["model"], combined["loglik"], color=colors)
    ax.invert_yaxis()
    ax.set(title="Same 3,846 observations: maximised log-likelihood of i.i.d. distributions vs ARMA models",
           xlabel="Maximised log-likelihood (higher is better)")
    ax.plot([], [], "s", color=PALETTE["orange"], label="i.i.d. distribution (ignores time order)")
    ax.plot([], [], "s", color=PALETTE["blue"], label="ARMA-type model (uses time order)")
    ax.legend(loc="center right")
    fig.tight_layout()
    return _save(fig, name)


def plot_arma_diagnostics(series_z: pd.Series, lb_p: pd.DataFrame, label: str, acf: np.ndarray,
                          name: str = "fig12_arma_residual_diagnostics.png") -> Path:
    z = series_z.dropna()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    axes[0, 0].plot(series_z.index, series_z.values, ".", ms=2.2, color=PALETTE["blue"])
    axes[0, 0].axhline(0, color="black", lw=0.8)
    axes[0, 0].set(title=f"Standardised residuals - {label}", xlabel="Date", ylabel="Standardised one-step error")
    stats.probplot(z.values, dist="norm", plot=axes[0, 1])
    axes[0, 1].get_lines()[0].set(markersize=2.5, color=PALETTE["blue"])
    axes[0, 1].get_lines()[1].set(color="black")
    axes[0, 1].set(title="Normal Q-Q plot of standardised residuals", xlabel="Theoretical quantile", ylabel="Sample quantile")
    lags = np.arange(1, len(acf))
    axes[1, 0].vlines(lags, 0, acf[1:], color=PALETTE["blue"], lw=1.6)
    axes[1, 0].plot(lags, acf[1:], "o", ms=3, color=PALETTE["blue"])
    band = 1.96 / np.sqrt(len(z))
    axes[1, 0].axhspan(-band, band, color=PALETTE["grey"], alpha=0.25)
    axes[1, 0].set(title="Residual ACF (pairs of observed hours only; no pair exists at lag 12)", xlabel="Lag (hours)", ylabel="Autocorrelation")
    for kind, color in [("Ljung-Box", PALETTE["blue"]), ("Ljung-Box (squared residuals)", PALETTE["red"])]:
        sub = lb_p[lb_p["test"] == kind]
        axes[1, 1].semilogy(sub["lag"], sub["p_value"], "o-", color=color, label=kind)
    axes[1, 1].axhline(config.ALPHA, color="black", ls="--", lw=1, label="5% level")
    axes[1, 1].set(title="Ljung-Box p-values", xlabel="Lag (hours)", ylabel="p-value (log scale)")
    axes[1, 1].legend()
    fig.tight_layout()
    return _save(fig, name)


def plot_garch(innovations: pd.Series, cond_vol: np.ndarray, name: str = "fig13_garch_conditional_volatility.png") -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    t = np.arange(len(innovations))
    ax.plot(t, innovations.values * 100, ".", ms=2, color=PALETTE["grey"], alpha=0.6, label="ARMA innovation x 100")
    ax.plot(t, 2 * cond_vol, color=PALETTE["red"], lw=1.2, label="+/- 2 conditional std. dev. (GARCH(1,1)-t)")
    ax.plot(t, -2 * cond_vol, color=PALETTE["red"], lw=1.2)
    ax.set(title="Volatility clustering in the ARMA innovations", xlabel="Daylight observation number (night hours removed)",
           ylabel="Index points (x100)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, name)


def plot_forecast(test_frame: pd.DataFrame, scores: pd.DataFrame, model_label: str,
                  name: str = "fig14_forecast_evaluation.png") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    sub = test_frame.iloc[:60]
    ax = axes[0]
    ax.plot(sub.index, sub["actual"] * sub["envelope"], "o-", ms=3, color="black", lw=1, label="Observed power")
    ax.plot(sub.index, sub["arma"] * sub["envelope"], "s--", ms=3, color=PALETTE["blue"], lw=1, label="Selected ARMA, 1 h ahead")
    ax.plot(sub.index, sub["persist"] * sub["envelope"], "^:", ms=3, color=PALETTE["orange"], lw=1, label="Persistence of index")
    ax.set(title=f"One-hour-ahead forecasts, first test days ({model_label})", xlabel="Date / hour", ylabel="Power (MW)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=25)
    bars = scores[["method", "RMSE_MW", "MAE_MW"]].set_index("method")
    bars.plot.barh(ax=axes[1], color=[PALETTE["blue"], PALETTE["orange"]])
    axes[1].invert_yaxis()
    axes[1].set(title="Out-of-sample error (lower is better)", xlabel="MW", ylabel="")
    fig.tight_layout()
    return _save(fig, name)
