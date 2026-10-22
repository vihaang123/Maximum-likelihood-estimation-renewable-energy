"""Stationarity analysis: ADF, KPSS, rolling statistics, periodic mean/variance tests, ACF/PACF.

Terminology used throughout
---------------------------
* Stationarity in the MEAN     : E[X_t] does not depend on t (no trend, no deterministic cycle).
* Stationarity in the VARIANCE : Var[X_t] does not depend on t (no volatility regimes, no time-of-day
                                 or seasonal change in spread).
* Weak (second-order) stationarity requires both, plus Cov(X_t, X_{t+h}) depending on h only.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss

from . import config


# --------------------------------------------------------------------------- #
# Unit-root / stationarity tests
# --------------------------------------------------------------------------- #
def adf_test(series: pd.Series, name: str, regression: str = "c",
             alpha: float = config.ALPHA) -> dict:
    """Augmented Dickey-Fuller test.  H0: unit root (non-stationary).  H1: no unit root."""
    values = pd.Series(series).dropna().to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)      # statsmodels announces a future change of the return type
        stat, pvalue, used_lag, nobs, crit, _ = adfuller(values, regression=regression, autolag="AIC")
    reject = pvalue < alpha
    return {
        "series": name,
        "test": "ADF",
        "specification": f"regression='{regression}', lags by AIC",
        "H0": "unit root present (non-stationary)",
        "H1": "no unit root (stationary around deterministic terms)",
        "alpha": alpha,
        "n_obs": int(nobs),
        "n_lags": int(used_lag),
        "statistic": float(stat),
        "p_value": float(pvalue),
        "p_value_truncated": False,
        "crit_1pct": float(crit["1%"]), "crit_5pct": float(crit["5%"]), "crit_10pct": float(crit["10%"]),
        "reject_H0": bool(reject),
        "stationary_evidence": bool(reject),
        "interpretation": ("Reject H0: evidence against a unit root." if reject
                           else "Fail to reject H0: a unit root cannot be ruled out."),
    }


def kpss_test(series: pd.Series, name: str, regression: str = "c", nlags="auto",
              alpha: float = config.ALPHA) -> dict:
    """KPSS test.  H0: stationary (around a constant 'c' or a trend 'ct').  H1: unit root.

    statsmodels interpolates p-values from a table, so values are clipped to [0.01, 0.10]; the
    `p_value_truncated` column records when that happened.
    """
    values = pd.Series(series).dropna().to_numpy(dtype=float)
    n = len(values)
    if nlags == "long":                                   # Schwert-type long bandwidth
        lags = int(np.ceil(12 * (n / 100.0) ** 0.25))
        lag_label = f"{lags} (long rule)"
    else:
        lags, lag_label = nlags, "auto (data-driven)"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stat, pvalue, used_lag, crit = kpss(values, regression=regression, nlags=lags)
    truncated = any("interpolat" in str(w.message).lower() for w in caught)
    reject = pvalue < alpha
    level = "level" if regression == "c" else "trend"
    return {
        "series": name,
        "test": "KPSS",
        "specification": f"regression='{regression}', bandwidth={lag_label}",
        "H0": f"{level}-stationary",
        "H1": "unit root present (non-stationary)",
        "alpha": alpha,
        "n_obs": n,
        "n_lags": int(used_lag),
        "statistic": float(stat),
        "p_value": float(pvalue),
        "p_value_truncated": bool(truncated),
        "crit_1pct": float(crit["1%"]), "crit_5pct": float(crit["5%"]), "crit_10pct": float(crit["10%"]),
        "reject_H0": bool(reject),
        "stationary_evidence": bool(not reject),
        "interpretation": (f"Reject H0: evidence against {level} stationarity." if reject
                           else f"Fail to reject H0: consistent with {level} stationarity."),
    }


def run_stationarity_battery(named_series: dict, alpha: float = config.ALPHA) -> pd.DataFrame:
    """ADF (constant) + KPSS (level, auto bandwidth; level, long bandwidth; trend) on each series."""
    rows = []
    for name, series in named_series.items():
        rows.append(adf_test(series, name, "c", alpha))
        rows.append(kpss_test(series, name, "c", "auto", alpha))
        rows.append(kpss_test(series, name, "c", "long", alpha))
        rows.append(kpss_test(series, name, "ct", "auto", alpha))
    frame = pd.DataFrame(rows)
    # Combined reading of ADF + KPSS(level, auto) for each series
    return frame


def joint_verdict(table: pd.DataFrame) -> pd.DataFrame:
    """Combine ADF and KPSS(level, auto) into the standard four-way reading."""
    rows = []
    for name, grp in table.groupby("series", sort=False):
        adf = grp[grp["test"] == "ADF"].iloc[0]
        kp = grp[(grp["test"] == "KPSS") & grp["specification"].str.contains("regression='c', bandwidth=auto")].iloc[0]
        if adf["reject_H0"] and not kp["reject_H0"]:
            verdict = "Stationary (ADF rejects unit root, KPSS does not reject stationarity)"
        elif (not adf["reject_H0"]) and kp["reject_H0"]:
            verdict = "Non-stationary (unit root not rejected, KPSS rejects stationarity)"
        elif adf["reject_H0"] and kp["reject_H0"]:
            verdict = "Conflicting: no unit root but not level-stationary (deterministic cycle / slowly varying mean)"
        else:
            verdict = "Inconclusive (neither test rejects)"
        rows.append({"series": name, "adf_p": adf["p_value"], "kpss_stat": kp["statistic"],
                     "kpss_p": kp["p_value"], "verdict": verdict})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Rolling statistics and group tests
# --------------------------------------------------------------------------- #
def rolling_statistics(series: pd.Series, window: int) -> pd.DataFrame:
    return pd.DataFrame({"rolling_mean": series.rolling(window, min_periods=window // 2).mean(),
                         "rolling_std": series.rolling(window, min_periods=window // 2).std()})


def group_tests(groups: dict, label: str, series_name: str) -> list:
    """Tests of equal means (one-way ANOVA, Kruskal-Wallis) and equal spreads (Levene/Brown-Forsythe)
    across groups, e.g. hour-of-day or month.  These directly test stationarity in the mean and in
    the variance with respect to a periodic factor."""
    samples = [np.asarray(v, dtype=float)[~np.isnan(np.asarray(v, dtype=float))] for v in groups.values()]
    samples = [s for s in samples if len(s) > 2]
    f_stat, f_p = stats.f_oneway(*samples)
    h_stat, h_p = stats.kruskal(*samples)
    l_stat, l_p = stats.levene(*samples, center="median")
    common = {"series": series_name, "factor": label, "n_groups": len(samples)}
    return [
        {**common, "test": "One-way ANOVA", "H0": "equal group means", "statistic": float(f_stat), "p_value": float(f_p),
         "reject_H0": bool(f_p < config.ALPHA)},
        {**common, "test": "Kruskal-Wallis", "H0": "equal group distributions (location)", "statistic": float(h_stat),
         "p_value": float(h_p), "reject_H0": bool(h_p < config.ALPHA)},
        {**common, "test": "Levene (Brown-Forsythe)", "H0": "equal group variances", "statistic": float(l_stat),
         "p_value": float(l_p), "reject_H0": bool(l_p < config.ALPHA)},
    ]


# --------------------------------------------------------------------------- #
# ACF / PACF (with support for series that contain gaps)
# --------------------------------------------------------------------------- #
def pairwise_acf(values: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelation using only pairs (x_t, x_{t+h}) where BOTH are observed.

    Needed for the clear-sky index, which is undefined at night.  Lags for which no pair exists
    (e.g. exactly one day-length) are returned as NaN.
    """
    x = np.asarray(values, dtype=float)
    mean = np.nanmean(x)
    variance = np.nanmean((x - mean) ** 2)
    acf = np.full(nlags + 1, np.nan)
    acf[0] = 1.0
    for h in range(1, nlags + 1):
        a, b = x[:-h] - mean, x[h:] - mean
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() > 30:
            acf[h] = np.mean(a[ok] * b[ok]) / variance
    return acf


def pacf_from_acf(acf: np.ndarray, nlags: int) -> np.ndarray:
    """Partial autocorrelations by the Durbin-Levinson recursion applied to a given ACF.
    NaN ACF values are skipped by stopping at the first NaN."""
    pacf = np.full(nlags + 1, np.nan)
    pacf[0] = 1.0
    phi_prev = np.array([])
    for k in range(1, nlags + 1):
        if np.isnan(acf[k]):
            break
        if k == 1:
            phi_kk = acf[1]
            phi = np.array([phi_kk])
        else:
            num = acf[k] - np.dot(phi_prev, acf[k - 1:0:-1])
            den = 1.0 - np.dot(phi_prev, acf[1:k])
            phi_kk = num / den
            phi = np.concatenate([phi_prev - phi_kk * phi_prev[::-1], [phi_kk]])
        pacf[k] = phi_kk
        phi_prev = phi
    return pacf


def remove_annual_harmonics(values: np.ndarray, design: np.ndarray) -> tuple:
    """OLS removal of a deterministic annual cycle; NaNs are ignored in the fit and kept in the output."""
    values = np.asarray(values, dtype=float)
    ok = ~np.isnan(values)
    beta, *_ = np.linalg.lstsq(design[ok], values[ok], rcond=None)
    fitted = design @ beta
    residual = values - fitted
    ss_res = np.nansum((values - fitted) ** 2)
    ss_tot = np.nansum((values - np.nanmean(values)) ** 2)
    return residual, fitted, beta, float(1 - ss_res / ss_tot)
