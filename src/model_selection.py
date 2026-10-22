"""Likelihood-based model selection and goodness-of-fit testing.

AIC  = 2k - 2 ln(L_hat)
BIC  = k ln(n) - 2 ln(L_hat)
AICc = AIC + 2k(k+1)/(n-k-1)          (small-sample correction; reported for completeness)

k = number of estimated parameters, n = number of observations, L_hat = maximised likelihood.
Smaller is better.  Differences are what matter:  Delta_i = IC_i - min_j IC_j, and the Akaike weight
w_i = exp(-Delta_i / 2) / sum_j exp(-Delta_j / 2) can be read as the relative support for model i
among the models compared (it is NOT the probability that the model is 'true').

All models in one comparison MUST be evaluated on exactly the same observations and with log-likelihoods
expressed as densities of the same variable; `compare_models` checks n.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Information criteria
# --------------------------------------------------------------------------- #
def aic(loglik: float, k: int) -> float:
    return 2.0 * k - 2.0 * loglik


def bic(loglik: float, k: int, n: int) -> float:
    return k * np.log(n) - 2.0 * loglik


def aicc(loglik: float, k: int, n: int) -> float:
    if n - k - 1 <= 0:
        return float("nan")
    return aic(loglik, k) + 2.0 * k * (k + 1) / (n - k - 1)


def information_table(rows: list) -> pd.DataFrame:
    """rows: dicts with keys model, loglik, k, n (+ any extra columns).  Adds AIC/BIC/AICc, deltas, weights, ranks."""
    frame = pd.DataFrame(rows).copy()
    if frame["n"].nunique() != 1:
        raise ValueError(
            "Models were evaluated on different numbers of observations "
            f"({sorted(frame['n'].unique())}); likelihood-based criteria are not comparable."
        )
    frame["AIC"] = [aic(l, k) for l, k in zip(frame["loglik"], frame["k"])]
    frame["BIC"] = [bic(l, k, n) for l, k, n in zip(frame["loglik"], frame["k"], frame["n"])]
    frame["AICc"] = [aicc(l, k, n) for l, k, n in zip(frame["loglik"], frame["k"], frame["n"])]
    for crit in ("AIC", "BIC"):
        frame[f"delta_{crit}"] = frame[crit] - frame[crit].min()
        weights = np.exp(-0.5 * frame[f"delta_{crit}"])
        frame[f"weight_{crit}"] = weights / weights.sum()
        frame[f"rank_{crit}"] = frame[crit].rank(method="min").astype(int)
    return frame.sort_values("AIC").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Goodness of fit (estimated parameters => bootstrap critical values)
# --------------------------------------------------------------------------- #
def ks_statistic(x_sorted: np.ndarray, cdf_values: np.ndarray) -> float:
    n = len(x_sorted)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - cdf_values), np.max(cdf_values - (i - 1) / n)))


def anderson_darling_statistic(cdf_values: np.ndarray) -> float:
    """A^2 = -n - (1/n) sum (2i-1) [ln F(x_i) + ln(1 - F(x_{n+1-i}))]   (x sorted ascending)."""
    n = len(cdf_values)
    f = np.clip(cdf_values, 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    return float(-n - np.mean((2 * i - 1) * (np.log(f) + np.log(1 - f[::-1]))))


def gof_statistics(x: np.ndarray, dist, p: np.ndarray) -> tuple:
    xs = np.sort(x)
    f = dist.cdf(xs, p)
    return ks_statistic(xs, f), anderson_darling_statistic(f)


def parametric_bootstrap_gof(dist, p_hat: np.ndarray, x: np.ndarray, n_boot: int = config.N_BOOTSTRAP,
                             seed: int = config.SEED) -> dict:
    """Parametric bootstrap p-values for the KS and Anderson-Darling statistics.

    Because the parameters are ESTIMATED from the same data, the textbook KS/AD null distributions do not apply
    (they are too conservative).  Following the Lilliefors idea we (1) simulate samples from the fitted model,
    (2) RE-ESTIMATE the parameters on every simulated sample, (3) recompute the statistics, and (4) compare.
    The bootstrap assumes independent observations, which the clear-sky index violates (strong serial
    correlation); p-values are therefore best read as a *relative* measure of fit.
    """
    rng = np.random.default_rng(seed)
    ks_obs, ad_obs = gof_statistics(x, dist, p_hat)
    ks_boot, ad_boot, failures = [], [], 0
    for _ in range(n_boot):
        sample = dist.rvs(len(x), p_hat, rng)
        try:
            fit_b = dist.fit(sample, start=p_hat)
            p_b = dist.params_array(fit_b)
            ks_b, ad_b = gof_statistics(sample, dist, p_b)
        except (ValueError, RuntimeError, FloatingPointError):
            failures += 1
            continue
        ks_boot.append(ks_b)
        ad_boot.append(ad_b)
    ks_boot, ad_boot = np.array(ks_boot), np.array(ad_boot)
    return {
        "KS_statistic": ks_obs,
        "KS_p_bootstrap": float((1 + np.sum(ks_boot >= ks_obs)) / (len(ks_boot) + 1)),
        "AD_statistic": ad_obs,
        "AD_p_bootstrap": float((1 + np.sum(ad_boot >= ad_obs)) / (len(ad_boot) + 1)),
        "n_bootstrap_ok": int(len(ks_boot)),
        "n_bootstrap_failed": int(failures),
    }


def quantile_rmse(x: np.ndarray, dist, p: np.ndarray, n_points: int = 99) -> float:
    """Root-mean-square gap between empirical and model quantiles (percentiles 1..99) in the units of x.
    A readable summary of how far the Q-Q plot departs from the 45-degree line."""
    probs = np.linspace(0.01, 0.99, n_points)
    return float(np.sqrt(np.mean((np.quantile(x, probs) - dist.ppf(probs, p)) ** 2)))


# --------------------------------------------------------------------------- #
# Comparison helper
# --------------------------------------------------------------------------- #
def compare_distributions(x: np.ndarray, fitted: dict, n_boot: int = config.N_BOOTSTRAP,
                          do_gof: bool = True) -> tuple:
    """Build the ranked comparison table and the long-format estimates table.

    fitted: {name: (Distribution, MLEFit)} from mle_models.fit_all.
    Returns (comparison_df, estimates_df).
    """
    rows, estimates = [], []
    for name, (dist, fit) in fitted.items():
        p = dist.params_array(fit)
        mean_m, sd_m = dist.implied_mean_sd(p)
        row = fit.as_row()
        row.update({"implied_mean": mean_m, "implied_sd": sd_m, "quantile_rmse": quantile_rmse(x, dist, p)})
        if do_gof:
            row.update(parametric_bootstrap_gof(dist, p, x, n_boot=n_boot))
        rows.append(row)
        for pname in fit.param_names:
            se = fit.std_errors.get(pname, np.nan)
            value = fit.params[pname]
            estimates.append({"model": name, "parameter": pname, "estimate": value, "std_error": se,
                              "ci95_low": value - 1.96 * se if np.isfinite(se) else np.nan,
                              "ci95_high": value + 1.96 * se if np.isfinite(se) else np.nan,
                              "loglik": fit.loglik, "n": fit.n})
    return information_table(rows), pd.DataFrame(estimates)


def select_best(table: pd.DataFrame, criterion: str = "BIC") -> str:
    return str(table.sort_values(criterion).iloc[0]["model"])
