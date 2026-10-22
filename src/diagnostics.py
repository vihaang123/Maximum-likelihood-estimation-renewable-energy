"""Residual diagnostics: autocorrelation, heteroskedasticity, normality.

For a correctly specified time-series model the standardised one-step prediction errors should behave like
white noise:  no autocorrelation (Ljung-Box), constant variance (ARCH-LM on squared residuals) and, if the
model is Gaussian, approximately normal (Jarque-Bera).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from . import config


def ljung_box(residuals: np.ndarray, label: str, lags=config.LJUNG_BOX_LAGS, model_df: int = 0,
              squared: bool = False) -> list:
    """Ljung-Box Q test.  H0: residuals are independently distributed (no autocorrelation up to lag h).

    Q(h) = n(n+2) sum_{j=1..h} r_j^2 / (n-j)  ~  chi2(h - model_df)  under H0,
    where model_df = number of estimated ARMA coefficients (p + q) -- not applicable to raw data (0).
    With squared=True the test is applied to the squared residuals (a test for ARCH-type dependence).
    """
    data = np.asarray(residuals, dtype=float)
    data = data[np.isfinite(data)]
    series = data ** 2 if squared else data
    rows = []
    for h in lags:
        out = acorr_ljungbox(series, lags=[h], model_df=model_df if h > model_df else 0)
        stat, p = float(out["lb_stat"].iloc[0]), float(out["lb_pvalue"].iloc[0])
        rows.append({"model": label, "test": "Ljung-Box" + (" (squared residuals)" if squared else ""),
                     "lag": int(h), "H0": "no autocorrelation up to this lag", "statistic": stat,
                     "p_value": p, "reject_H0": bool(p < config.ALPHA), "n": int(len(series))})
    return rows


def arch_lm(residuals: np.ndarray, label: str, lags=(5, 12, 24)) -> list:
    """Engle's ARCH-LM test.  H0: no ARCH effects (squared residuals not predictable from their own past)."""
    data = np.asarray(residuals, dtype=float)
    data = data[np.isfinite(data)]
    rows = []
    for h in lags:
        lm, lm_p, f_stat, f_p = het_arch(data, nlags=h)
        rows.append({"model": label, "test": "ARCH-LM", "lag": int(h), "H0": "no conditional heteroskedasticity",
                     "statistic": float(lm), "p_value": float(lm_p), "reject_H0": bool(lm_p < config.ALPHA),
                     "n": int(len(data))})
    return rows


def normality_tests(residuals: np.ndarray, label: str) -> list:
    data = np.asarray(residuals, dtype=float)
    data = data[np.isfinite(data)]
    jb, jb_p = stats.jarque_bera(data)
    return [{"model": label, "test": "Jarque-Bera", "lag": np.nan, "H0": "normality (skewness 0, excess kurtosis 0)",
             "statistic": float(jb), "p_value": float(jb_p), "reject_H0": bool(jb_p < config.ALPHA), "n": int(len(data))},
            {"model": label, "test": "Residual skewness", "lag": np.nan, "H0": "-", "statistic": float(stats.skew(data)),
             "p_value": np.nan, "reject_H0": np.nan, "n": int(len(data))},
            {"model": label, "test": "Residual excess kurtosis", "lag": np.nan, "H0": "-",
             "statistic": float(stats.kurtosis(data)), "p_value": np.nan, "reject_H0": np.nan, "n": int(len(data))}]


def residual_diagnostics(std_residuals: np.ndarray, label: str, model_df: int = 0) -> pd.DataFrame:
    """Full diagnostic battery for one model's standardised residuals."""
    rows = []
    rows += ljung_box(std_residuals, label, model_df=model_df)
    rows += ljung_box(std_residuals, label, model_df=0, squared=True)
    rows += arch_lm(std_residuals, label)
    rows += normality_tests(std_residuals, label)
    return pd.DataFrame(rows)


def diagnostics_pass_summary(table: pd.DataFrame, label: str) -> dict:
    """Quick yes/no flags used in the report: which diagnostics reject at the 5 % level."""
    sub = table[table["model"] == label]
    lb = sub[sub["test"] == "Ljung-Box"]
    lb2 = sub[sub["test"] == "Ljung-Box (squared residuals)"]
    arch = sub[sub["test"] == "ARCH-LM"]
    return {
        "model": label,
        "ljung_box_rejections": int(lb["reject_H0"].sum()), "ljung_box_tests": int(len(lb)),
        "ljung_box_sq_rejections": int(lb2["reject_H0"].sum()), "ljung_box_sq_tests": int(len(lb2)),
        "arch_lm_rejections": int(arch["reject_H0"].sum()), "arch_lm_tests": int(len(arch)),
    }
