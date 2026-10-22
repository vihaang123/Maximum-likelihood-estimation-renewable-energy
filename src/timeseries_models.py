"""Time-series models for the clear-sky index: regression with ARMA errors, GARCH, and one-step forecasting.

Model (regression with seasonal-ARMA errors, estimated by exact Gaussian maximum likelihood)
--------------------------------------------------------------------------------------------
    y_t = x_t' beta + u_t ,                          x_t = [1, sin(2 pi d_t/365), cos(2 pi d_t/365)]
    Phi(L^24) phi(L) u_t = Theta(L^24) theta(L) e_t , e_t ~ N(0, sigma^2)

y_t is the clear-sky index, defined only in daylight hours.  Night-time hours are treated as MISSING values:
the Kalman filter used by statsmodels simply skips them (it propagates the latent ARMA state through the gap),
so the likelihood is the exact Gaussian likelihood of the observed daylight values.  This makes the log-likelihood
directly comparable with the i.i.d. distribution fits of the same observations.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import config, preprocessing
from .model_selection import aic, bic, aicc


@dataclass(frozen=True)
class ARMASpec:
    name: str
    order: tuple                    # (p, 0, q)
    seasonal_order: tuple = (0, 0, 0, 0)
    harmonics: bool = True

    @property
    def n_arma_coefficients(self) -> int:
        p, _, q = self.order
        sp, _, sq, _ = self.seasonal_order
        return p + q + sp + sq


DEFAULT_SPECS = [
    ARMASpec("iid Gaussian + annual cycle", (0, 0, 0)),
    ARMASpec("AR(1)", (1, 0, 0)),
    ARMASpec("AR(2)", (2, 0, 0)),
    ARMASpec("AR(3)", (3, 0, 0)),
    ARMASpec("ARMA(1,1)", (1, 0, 1)),
    ARMASpec("ARMA(2,1)", (2, 0, 1)),
    ARMASpec("ARMA(2,2)", (2, 0, 2)),
    ARMASpec("ARMA(1,1)x(1,0,0)24", (1, 0, 1), (1, 0, 0, 24)),
    ARMASpec("ARMA(2,1)x(1,0,0)24", (2, 0, 1), (1, 0, 0, 24)),
    ARMASpec("ARMA(2,1)x(1,0,0)24, no annual cycle", (2, 0, 1), (1, 0, 0, 24), harmonics=False),
]


def build_endog_exog(frame: pd.DataFrame, harmonics: bool = True,
                     n_harmonics: int = config.ANNUAL_HARMONICS) -> tuple:
    """Series with NaN at night / excluded hours, and the regressor matrix (constant [+ annual harmonics])."""
    y = frame["clearsky_index"].where(frame["in_sample"])
    y.index = pd.DatetimeIndex(y.index, freq="h")
    design = preprocessing.annual_harmonic_design(frame["day_of_year"].to_numpy(),
                                                   n_harmonics if harmonics else 0)
    columns = ["const"] + [f"{f}{k}" for k in range(1, n_harmonics + 1) for f in ("sin", "cos")] if harmonics else ["const"]
    exog = pd.DataFrame(design, index=y.index, columns=columns)
    return y, exog


def fit_arma(endog: pd.Series, exog: pd.DataFrame, spec: ARMASpec, maxiter: int = 400):
    """Exact Gaussian ML fit of one specification.  Returns (results, row_dict)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(endog, exog=exog, order=spec.order, seasonal_order=spec.seasonal_order, trend=None,
                        enforce_stationarity=True, enforce_invertibility=True)
        res = model.fit(disp=False, maxiter=maxiter)
    n = int(endog.notna().sum())
    k = int(len(res.params))
    ll = float(res.llf)
    row = {"model": spec.name, "order": str(spec.order), "seasonal_order": str(spec.seasonal_order),
           "harmonics": spec.harmonics, "n": n, "k": k, "loglik": ll, "AIC": aic(ll, k), "BIC": bic(ll, k, n),
           "AICc": aicc(ll, k, n), "converged": bool(res.mle_retvals.get("converged", False)),
           "arma_coefficients": spec.n_arma_coefficients}
    return res, row


def fit_candidates(endog: pd.Series, exog: pd.DataFrame, specs=DEFAULT_SPECS, verbose: bool = True) -> tuple:
    """Fit every specification on the SAME observations.  Returns (table sorted by BIC, {name: results})."""
    rows, results = [], {}
    for spec in specs:
        use_exog = exog if spec.harmonics else exog[["const"]]
        try:
            res, row = fit_arma(endog, use_exog, spec)
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"  [warning] {spec.name} failed: {exc}")
            continue
        results[spec.name] = res
        rows.append(row)
        if verbose:
            print(f"    {spec.name:24s} loglik={row['loglik']:10.2f}  AIC={row['AIC']:11.2f}  BIC={row['BIC']:11.2f}"
                  f"  converged={row['converged']}")
    table = pd.DataFrame(rows)
    for crit in ("AIC", "BIC"):
        table[f"delta_{crit}"] = table[crit] - table[crit].min()
        w = np.exp(-0.5 * table[f"delta_{crit}"])
        table[f"weight_{crit}"] = w / w.sum()
        table[f"rank_{crit}"] = table[crit].rank(method="min").astype(int)
    return table.sort_values("BIC").reset_index(drop=True), results


def parameter_table(res, model_name: str) -> pd.DataFrame:
    """Estimates, standard errors (inverse observed information) and z-tests for one fitted model."""
    frame = pd.DataFrame({"model": model_name, "parameter": res.params.index, "estimate": res.params.to_numpy(),
                          "std_error": res.bse.to_numpy(), "z": res.tvalues.to_numpy(), "p_value": res.pvalues.to_numpy()})
    frame["ci95_low"] = frame["estimate"] - 1.96 * frame["std_error"]
    frame["ci95_high"] = frame["estimate"] + 1.96 * frame["std_error"]
    return frame


def standardized_residuals(res) -> np.ndarray:
    """Standardised one-step prediction errors at the OBSERVED (daylight) times."""
    z = np.asarray(res.standardized_forecasts_error).ravel()
    observed = ~np.isnan(np.asarray(res.model.endog).ravel())
    z = z[observed]
    return z[np.isfinite(z)]


def raw_residuals(res) -> np.ndarray:
    e = np.asarray(res.resid).ravel()
    observed = ~np.isnan(np.asarray(res.model.endog).ravel())
    e = e[observed]
    return e[np.isfinite(e)]


# --------------------------------------------------------------------------- #
# GARCH on the ARMA innovations
# --------------------------------------------------------------------------- #
def fit_garch_family(innovations: np.ndarray, scale: float = 100.0) -> tuple:
    """Fit constant-variance and GARCH(1,1) models (normal and Student-t errors) to ARMA innovations.

    The innovations of consecutive DAYLIGHT hours are concatenated (night gaps removed); this is an
    approximation, stated in the report.  Innovations are multiplied by `scale` (100) purely for numerical
    conditioning of the optimiser; the log-likelihoods below are reported on the ORIGINAL scale by subtracting
    n ln(scale).
    """
    from arch import arch_model

    e = np.asarray(innovations, dtype=float) * scale
    n = len(e)
    # if e_scaled = scale * e then f_e(e) = scale * f_scaled(scale * e), so ln L_original = ln L_scaled + n ln(scale)
    log_jacobian = n * np.log(scale)
    rows, fits = [], {}

    # constant variance (i.i.d. normal, zero mean): sigma^2_hat = mean(e^2)
    sigma2 = float(np.mean(e ** 2))
    ll_const = float(-0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0))
    rows.append({"model": "Constant variance (normal)", "k": 1, "n": n, "loglik": ll_const + log_jacobian,
                 "converged": True, "params": f"sigma2={sigma2 / scale ** 2:.6g}"})
    for label, kwargs in [("ARCH(1) (normal)", dict(vol="ARCH", p=1, dist="normal")),
                          ("GARCH(1,1) (normal)", dict(vol="GARCH", p=1, q=1, dist="normal")),
                          ("GARCH(1,1) (Student-t)", dict(vol="GARCH", p=1, q=1, dist="t"))]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(e, mean="Zero", rescale=False, **kwargs).fit(disp="off", options={"maxiter": 2000})
        k = len(res.params)
        rows.append({"model": label, "k": k, "n": n, "loglik": float(res.loglikelihood) + log_jacobian,
                     "converged": bool(res.convergence_flag == 0),
                     "params": ", ".join(f"{name}={val:.4g}" for name, val in res.params.items())})
        fits[label] = res
    table = pd.DataFrame(rows)
    table["AIC"] = [aic(l, k) for l, k in zip(table["loglik"], table["k"])]
    table["BIC"] = [bic(l, k, n) for l, k in zip(table["loglik"], table["k"])]
    for crit in ("AIC", "BIC"):
        table[f"delta_{crit}"] = table[crit] - table[crit].min()
    return table.sort_values("BIC").reset_index(drop=True), fits


# --------------------------------------------------------------------------- #
# Out-of-sample one-step-ahead forecasting experiment (chronological split, no leakage)
# --------------------------------------------------------------------------- #
FORECAST_SPECS = [
    ARMASpec("AR(1)", (1, 0, 0), harmonics=False),
    ARMASpec("AR(2)", (2, 0, 0), harmonics=False),
    ARMASpec("ARMA(1,1)", (1, 0, 1), harmonics=False),
    ARMASpec("ARMA(2,1)", (2, 0, 1), harmonics=False),
    ARMASpec("ARMA(1,1)x(1,0,0)24", (1, 0, 1), (1, 0, 0, 24), harmonics=False),
]


def forecast_experiment(hourly: pd.Series, train_fraction: float = config.TRAIN_FRACTION) -> dict:
    """One-hour-ahead forecasting of the clear-sky index and of power (MW) on a held-out final period.

    Leakage control
    ---------------
    * The envelope is the CAUSAL one (trailing window, today excluded) so no future data enter the index.
    * The model order is chosen by BIC on the TRAINING days only; parameters are estimated on training only.
    * The fitted parameters are then applied (not re-estimated) to the whole series: the Kalman filter's
      one-step-ahead prediction for time t uses data up to t-1 only.
    * No annual-harmonic regressors: with <1 year of training data the annual cycle cannot be extrapolated.

    Baselines: persistence of the index (y_{t-1}, i.e. 'smart persistence'), the training-sample mean index,
    and naive power persistence (P_{t-1}).  All methods are scored on exactly the same hours.
    """
    frame = preprocessing.build_analysis_frame(hourly, causal=True)
    n_days = len(frame) // 24
    split_day = int(round(train_fraction * n_days))
    split_pos = split_day * 24
    y, exog = build_endog_exog(frame, harmonics=False)

    train_y, train_x = y.iloc[:split_pos], exog.iloc[:split_pos]
    table, results = fit_candidates(train_y, train_x, FORECAST_SPECS, verbose=False)
    best_name = str(table.sort_values("BIC").iloc[0]["model"])
    spec = next(s for s in FORECAST_SPECS if s.name == best_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        applied = results[best_name].apply(y, exog=exog)
    pred_arma = pd.Series(np.asarray(applied.fittedvalues).ravel(), index=y.index)

    prev = y.shift(1)
    test = pd.DataFrame({"actual": y, "arma": pred_arma, "persist": prev, "envelope": frame["envelope_mw"],
                         "power": frame["hourly_total_mw"], "power_prev": frame["hourly_total_mw"].shift(1)})
    test["train_mean"] = float(np.nanmean(train_y))
    test = test.iloc[split_pos:]
    test = test[test["actual"].notna() & test["persist"].notna() & test["envelope"].notna()].copy()

    rows = []
    methods = {"Selected ARMA (" + best_name + ")": test["arma"], "Persistence of index": test["persist"],
               "Training-mean index": test["train_mean"]}
    for label, pred in methods.items():
        err = test["actual"] - pred
        err_mw = (test["actual"] - pred) * test["envelope"]
        rows.append({"method": label, "n_test": int(len(test)), "MAE_index": float(err.abs().mean()),
                     "RMSE_index": float(np.sqrt((err ** 2).mean())), "MAE_MW": float(err_mw.abs().mean()),
                     "RMSE_MW": float(np.sqrt((err_mw ** 2).mean()))})
    err_raw = test["power"] - test["power_prev"]
    rows.append({"method": "Naive power persistence (P_{t-1})", "n_test": int(len(test)), "MAE_index": np.nan,
                 "RMSE_index": np.nan, "MAE_MW": float(err_raw.abs().mean()),
                 "RMSE_MW": float(np.sqrt((err_raw ** 2).mean()))})
    scores = pd.DataFrame(rows)
    base = scores.loc[scores["method"] == "Persistence of index", "RMSE_MW"].iloc[0]
    scores["RMSE_skill_vs_index_persistence"] = 1.0 - scores["RMSE_MW"] / base
    return {"scores": scores, "selected_model": best_name, "selection_table": table,
            "train_days": split_day, "test_days": n_days - split_day,
            "train_end": str(frame.index[split_pos - 1]), "test_start": str(frame.index[split_pos]),
            "test_frame": test, "train_nobs": int(train_y.notna().sum()), "arma_params": applied.params.to_dict()}


# --------------------------------------------------------------------------- #
# Sensitivity of the ARMA likelihood to the length of the night gap ("time deformation" check)
# --------------------------------------------------------------------------- #
def night_gap_sensitivity(frame: pd.DataFrame, night_slots=(0, 4, 12, 24),
                          order=(2, 0, 1)) -> pd.DataFrame:
    """Re-fit ARMA(order)x(1,0,0)_S after changing how many missing 'night' steps separate two days.

    The ordinary hourly grid has 12 night slots (hours 18-05) between the last and the first daylight value.
    Residual diagnostics of the main model show negative correlation across that gap, so we ask whether the model
    would fit better if the night were treated as shorter/longer than 12 hours.  Each day is laid out as the
    12 daylight-window slots (06:00-17:00) followed by `N` missing slots, giving a seasonal period S = 12 + N.
    N = 12 is the real hourly grid.  A likelihood that keeps rising with N would signal that the AR dynamics
    over-bridge the night; one that peaks near N = 12 supports the natural time scale.
    """
    from . import diagnostics

    y = frame["clearsky_index"].where(frame["in_sample"]).to_numpy().reshape(-1, 24)[:, 6:18]
    day_of_year = frame["day_of_year"].to_numpy().reshape(-1, 24)[:, 6:18]
    rows = []
    for n_night in night_slots:
        period = 12 + n_night
        grid = np.full((y.shape[0], period), np.nan)
        grid[:, :12] = y
        doy = np.zeros((y.shape[0], period))
        doy[:, :12] = day_of_year
        doy[:, 12:] = day_of_year[:, [-1]]
        endog = pd.Series(grid.ravel())
        exog = pd.DataFrame(preprocessing.annual_harmonic_design(doy.ravel(), config.ANNUAL_HARMONICS),
                            columns=["const", "sin1", "cos1"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(endog, exog=exog, order=order, seasonal_order=(1, 0, 0, period), trend=None).fit(disp=False, maxiter=300)
        n_obs = int(np.isfinite(grid).sum())
        k = len(res.params)
        z = np.asarray(res.standardized_forecasts_error).ravel()
        z = np.where(np.isfinite(grid.ravel()), z, np.nan)
        lb = diagnostics.ljung_box(z[np.isfinite(z)], "x", lags=(24,), model_df=sum(order) - order[1] + 1)[0]
        rows.append({"night_slots": n_night, "seasonal_period": period, "n": n_obs, "k": k, "loglik": float(res.llf),
                     "AIC": aic(res.llf, k), "BIC": bic(res.llf, k, n_obs), "ljung_box_24_p": lb["p_value"],
                     "is_real_hourly_grid": bool(n_night == 12)})
    return pd.DataFrame(rows)
