#!/usr/bin/env python
"""Maximum Likelihood Estimation and Model Selection for Stationary Renewable Energy Output Data.

Runs the complete analysis and writes every table/figure used in the report:

    python main.py            # full analysis (about 6-10 minutes)
    python main.py --fast     # quick smoke run (fewer bootstrap replicates, skips two slow checks)

Outputs: results/*.csv, results/key_results.json, results/figures/*.png, data/cleaned_dataset.csv,
data/dataset_metadata.md
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf

from src import config, diagnostics as dg, mle_models as mm, model_selection as ms, plotting as pl
from src import preprocessing as pp, stationarity as st, timeseries_models as tsm

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def native(obj):
    """Make numpy / pandas objects JSON serialisable."""
    if isinstance(obj, dict):
        return {str(k): native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_csv(frame: pd.DataFrame, name: str) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.RESULTS_DIR / name, index=False, float_format="%.6g")


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


# --------------------------------------------------------------------------- #
# dataset metadata (generated from the actual validation numbers)
# --------------------------------------------------------------------------- #
def write_metadata(checks: dict, summary: dict, info: dict, evidence: dict) -> None:
    text = f"""# Dataset metadata

## Identification
| Item | Value |
|---|---|
| Dataset | Solar PV power output of 137 plants in Alabama, calendar year 2006 ("solar-energy" dataset) |
| Origin | U.S. National Renewable Energy Laboratory (NREL), *Solar Power Data for Integration Studies*; project page: https://www.nrel.gov/grid/solar-power-data.html |
| Copy used | Unmodified file `solar_AL.txt.gz` from the public LSTNet repository: {info.get('url', config.DATA_URL)} |
| Associated paper (source of the redistributed copy) | Lai, G., Chang, W.-C., Yang, Y., Liu, H. (2018). Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks. SIGIR '18, 95-104. |
| Download date (UTC) | {info.get('download_date_utc', 'n/a')} |
| File size / SHA-256 | {info.get('bytes', 'n/a')} bytes / `{info.get('sha256', 'n/a')}` |
| Local copy | `data/solar_AL.txt.gz` (preserved byte-for-byte; the analysis reads it directly) |
| Cleaned/derived data | `data/cleaned_dataset.csv` (hourly fleet total, envelope, clear-sky index, flags) |

## Variable, units, frequency, scope
* **Variable:** PV plant power output. One column per plant (137 columns), no header, comma separated, gzip-compressed.
* **Units:** MW (NREL convention; the redistributed file carries no unit header).
* **Native frequency:** 10 minutes, {checks['n_rows']:,} rows = 365 days x 144 steps (a non-leap year).
* **Geographic scope:** 137 PV plants in the U.S. state of Alabama.
* **Period:** 2006 (first row = 2006-01-01 00:00, last row = {checks['last_timestamp']}).
* **Analysis variable:** fleet total (sum over the 137 plants), averaged to hourly means (8,760 values).

## IMPORTANT: nature of the data
NREL's *Solar Power Data for Integration Studies* are, to the best of our knowledge, **modelled (simulated) PV plant
outputs** produced from satellite-derived solar-radiation and weather data for hypothetical plants, **not metered
generation from operating plants**. They are genuine, publicly released NREL data, but the series is model output, so the
statistical conclusions describe the behaviour of this simulated PV output. The NREL web pages were not reachable from the
environment used to build this project (the network allow-list blocked them), so this description is taken from NREL's
public project description and should be re-checked on the NREL page before submission.
Open Power System Data, NREL developer APIs, EIA, Zenodo, etc. were also unreachable, which is why this publicly mirrored copy was used.

## Timestamps are reconstructed (documented assumption)
The file has no time column. Row 0 is assumed to be 2006-01-01 00:00 local standard time and rows follow a regular
10-minute grid (365 x 144 = {checks['expected_rows']:,} rows, exactly what the file contains). Evidence that this assumption is
physically consistent (computed by the code from the data):
* Peak of the mean diurnal profile at **{evidence['peak_time']}** local time (solar noon in Alabama is about 12:00-12:15 CST).
* Mean number of hours per day with positive output: **{evidence['daylength_dec']:.1f} h in December** and
  **{evidence['daylength_jun']:.1f} h in June** (astronomical day length at ~32-33 deg N is about 10 h and 14 h).
* Monthly mean output peaks in late spring/summer and is lowest in winter.

## Integrity checks (from `results/data_validation.json`)
| Check | Result |
|---|---|
| Rows x plants | {checks['n_rows']:,} x {checks['n_plants']} (expected {checks['expected_rows']:,} x {config.EXPECTED_PLANTS}) |
| Missing / non-finite values | {checks['n_missing']} / {checks['n_non_finite']} |
| Negative values | {checks['n_negative']} |
| Duplicate timestamps | {checks['n_duplicate_timestamps']} |
| Distinct time steps | {checks['n_distinct_time_steps']} ({checks['time_step_minutes']:.0f} minutes) |
| Plants that are always zero | {checks['n_plants_always_zero']} |
| Largest single-plant value | {checks['plant_max_mw']} MW |
| Hours with exactly zero fleet output | {summary['n_hours_exact_zero']:,} of {summary['n_hours']:,} ({100 * summary['share_hours_exact_zero']:.1f} %) |

## Licence / usage conditions
No explicit licence file accompanies the redistributed copy. The data originate from NREL, a U.S. Department of Energy
laboratory that publishes this dataset for public download; the original NREL terms should be consulted before any
redistribution beyond this academic submission. The file is included here, unmodified, for reproducibility.

## Preprocessing decisions (all recorded in `results/preprocessing_summary.json`)
1. Plants are summed to a fleet total; no value is removed or altered.
2. 10-minute values are averaged to hourly means (every hour contains exactly six values; no missing data to impute).
3. Night-time zeros are structural (sun below the horizon), not missing data or errors: they are kept in the raw
   hourly series and excluded from the clear-sky-index analysis sample (see report, Sections 14 and 18).
4. A data-derived clear-sky envelope (95th percentile of same-hour output within +/-7 days) is used to compute the
   clear-sky index. Hours whose envelope is below {100 * config.MIN_ENVELOPE_FRACTION:.0f} % of its annual maximum are outside the daylight window.
5. Unusual values (index > 1.20 or < 0.05) are flagged in `cleaned_dataset.csv` but never deleted.
"""
    config.METADATA_FILE.write_text(text)


# --------------------------------------------------------------------------- #
# main pipeline
# --------------------------------------------------------------------------- #
def main(fast: bool = False) -> None:
    started = time.time()
    n_boot = 40 if fast else config.N_BOOTSTRAP
    np.random.seed(config.SEED)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.FIG_DIR.mkdir(exist_ok=True)
    pl.setup_style()
    key: dict = {"settings": {"seed": config.SEED, "alpha": config.ALPHA, "n_bootstrap": n_boot,
                              "envelope_quantile": config.ENVELOPE_QUANTILE, "envelope_half_window_days": config.ENVELOPE_HALF_WINDOW,
                              "min_envelope_fraction": config.MIN_ENVELOPE_FRACTION, "annual_harmonics": config.ANNUAL_HARMONICS,
                              "train_fraction": config.TRAIN_FRACTION, "fast_mode": fast}}

    # ------------------------------------------------------------------ 1. data
    section("1. Load, validate and aggregate the data")
    matrix = pp.load_raw_matrix()
    index10 = pp.build_timestamps(len(matrix))
    checks = pp.validate_raw(matrix, index10)
    (config.RESULTS_DIR / "data_validation.json").write_text(json.dumps(native(checks), indent=2))
    print(json.dumps(native(checks), indent=2))
    fleet10 = pp.aggregate_fleet(matrix, index10)
    hourly = pp.to_hourly(fleet10)
    frame = pp.build_analysis_frame(hourly)
    summary = pp.summarize_missing_and_unusual(frame)
    summary.update({"sha256_raw_file": pp.file_sha256(config.RAW_FILE),
                    "hourly_mean_mw": float(hourly.mean()), "hourly_max_mw": float(hourly.max()),
                    "hourly_median_mw": float(hourly.median()),
                    "hourly_sd_mw": float(hourly.std()),
                    "plant_mean_mw_min": float(matrix.mean(axis=0).min()), "plant_mean_mw_max": float(matrix.mean(axis=0).max())})
    pp.save_clean_dataset(frame)
    (config.RESULTS_DIR / "preprocessing_summary.json").write_text(json.dumps(native(summary), indent=2))
    # evidence for the reconstructed time axis
    mean10 = fleet10.groupby(fleet10.index.hour + fleet10.index.minute / 60).mean()
    peak = float(mean10.idxmax())
    daylength = (fleet10 > 0).groupby(fleet10.index.date).sum() / 6.0
    daylength.index = pd.to_datetime(daylength.index)
    dl_month = daylength.groupby(daylength.index.month).mean()
    evidence = {"peak_time": f"{int(peak):02d}:{int(round((peak % 1) * 60)):02d}", "daylength_dec": float(dl_month[12]),
                "daylength_jun": float(dl_month[6])}
    write_metadata(checks, summary, pp.read_download_info(), evidence)
    key.update({"data_checks": checks, "preprocessing": summary, "time_axis_evidence": evidence})
    print(f"  hourly observations: {len(hourly):,}; daylight-window hours: {summary['n_daylight_hours']:,}; "
          f"index sample n = {summary['n_in_sample']:,}")

    # ------------------------------------------------------------------ 2. EDA
    section("2. Exploratory data analysis and figures")
    pl.plot_overview(hourly)
    pl.plot_diurnal_seasonal(frame)
    pl.plot_raw_distribution(hourly)
    pl.plot_envelope(frame)
    desc = hourly.describe()
    key["eda"] = {"skewness_raw": float(stats.skew(hourly)), "kurtosis_raw": float(stats.kurtosis(hourly)),
                  "monthly_mean_mw": hourly.groupby(hourly.index.month).mean().round(1).to_dict(),
                  "describe": desc.to_dict()}
    p0 = float((hourly == 0).mean())
    key["zero_mass"] = {"p0_mle": p0, "se": float(np.sqrt(p0 * (1 - p0) / len(hourly))), "n": int(len(hourly))}
    # per-plant summary
    plant_stats = pd.DataFrame({"plant": np.arange(matrix.shape[1]), "mean_mw": matrix.mean(axis=0),
                                "max_mw": matrix.max(axis=0), "share_zero_10min": (matrix == 0).mean(axis=0)})
    save_csv(plant_stats, "plant_summary.csv")
    daylight = frame[frame["in_sample"]]
    x = daylight["clearsky_index"].to_numpy()
    key["index_summary"] = {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std()), "min": float(x.min()),
                            "max": float(x.max()), "skewness": float(stats.skew(x)), "excess_kurtosis": float(stats.kurtosis(x)),
                            "quantiles": {str(q): float(np.quantile(x, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
                            "monthly_mean": daylight.groupby("month")["clearsky_index"].mean().round(3).to_dict(),
                            "monthly_sd": daylight.groupby("month")["clearsky_index"].std().round(3).to_dict(),
                            "hourly_mean": daylight.groupby("hour")["clearsky_index"].mean().round(3).to_dict()}

    # ------------------------------------------------------------------ 3. stationarity
    section("3. Stationarity analysis")
    idx_concat = daylight["clearsky_index"].reset_index(drop=True)
    harmonic_design = pp.annual_harmonic_design(daylight["day_of_year"].to_numpy(), config.ANNUAL_HARMONICS)
    idx_resid, _, harmonic_beta, harmonic_r2 = st.remove_annual_harmonics(idx_concat.to_numpy(), harmonic_design)
    noon = frame[(frame["hour"] == 12) & frame["in_sample"]]["clearsky_index"].reset_index(drop=True)
    series = {
        "Raw hourly output (MW)": hourly.reset_index(drop=True),
        "Raw daily mean output (MW)": hourly.groupby(frame["day_index"].to_numpy()).mean().reset_index(drop=True),
        "Raw hourly, seasonal difference lag 24": hourly.diff(24).dropna().reset_index(drop=True),
        "Clear-sky index, daylight hours (concatenated)": idx_concat,
        "Clear-sky index at 12:00 (daily)": noon,
        "Daily mean clear-sky index": daylight.groupby("day_index")["clearsky_index"].mean().reset_index(drop=True),
        "Clear-sky index minus annual harmonic": pd.Series(idx_resid),
    }
    stat_table = st.run_stationarity_battery(series)
    verdict = st.joint_verdict(stat_table)
    save_csv(stat_table, "stationarity_tests.csv")
    save_csv(verdict, "stationarity_verdicts.csv")
    print(verdict.to_string(index=False))

    hour_groups = {h: g["hourly_total_mw"].to_numpy() for h, g in frame.groupby("hour")}
    month_groups_raw = {m: g["hourly_total_mw"].to_numpy() for m, g in frame.groupby("month")}
    hour_groups_idx = {h: g["clearsky_index"].to_numpy() for h, g in daylight.groupby("hour")}
    month_groups_idx = {m: g["clearsky_index"].to_numpy() for m, g in daylight.groupby("month")}
    periodic = (st.group_tests(hour_groups, "hour of day", "Raw hourly output")
                + st.group_tests(month_groups_raw, "month", "Raw hourly output")
                + st.group_tests(hour_groups_idx, "hour of day", "Clear-sky index")
                + st.group_tests(month_groups_idx, "month", "Clear-sky index"))
    periodic = pd.DataFrame(periodic)
    save_csv(periodic, "periodicity_tests.csv")
    key["stationarity"] = {"harmonic_r2": harmonic_r2, "harmonic_beta": harmonic_beta.tolist()}

    pl.plot_rolling_and_groups(frame)
    nl = 72
    raw_acf = sm_acf(hourly.to_numpy(), nlags=nl, fft=True)
    raw_pacf = sm_pacf(hourly.to_numpy(), nlags=nl, method="ywm")
    idx_full = frame["clearsky_index"].where(frame["in_sample"]).to_numpy()
    idx_acf = st.pairwise_acf(idx_full, 48)
    # pairwise (gap-aware) ACF is not guaranteed positive semi-definite, so the derived PACF is only trusted for small lags
    idx_pacf = st.pacf_from_acf(idx_acf, 6)
    pl.plot_acf_pacf({"raw hourly output": (raw_acf, raw_pacf, len(hourly))}, "fig06a_acf_pacf_raw_output.png", "Raw")
    pl.plot_acf_pacf({"clear-sky index (PACF to lag 6 only)": (idx_acf, idx_pacf, len(x))}, "fig06b_acf_pacf_clearsky_index.png", "Index")
    key["acf"] = {"raw_lag1": float(raw_acf[1]), "raw_lag24": float(raw_acf[24]), "raw_lag12": float(raw_acf[12]),
                  "index_lag1": float(idx_acf[1]), "index_lag2": float(idx_acf[2]), "index_lag6": float(idx_acf[6]),
                  "index_lag24": float(idx_acf[24]), "index_lag48": float(idx_acf[48]),
                  "index_lag13": float(idx_acf[13])}
    daily_idx = daylight.groupby("day_index")["clearsky_index"].mean()
    rho_d = float(daily_idx.autocorr(1))
    key["effective_sample"] = {"rho_daily_lag1": rho_d, "n_days": int(len(daily_idx)),
                               "n_eff_days_ar1_approx": float(len(daily_idx) * (1 - rho_d) / (1 + rho_d))}

    # ------------------------------------------------------------------ 4. MLE of distributions
    section("4. Maximum likelihood estimation of candidate distributions (clear-sky index)")
    candidates = mm.build_candidates(x)
    fitted = mm.fit_all(x, candidates)
    validation = pd.DataFrame([mm.validate_against_scipy(x, d, f) for d, f in fitted.values()])
    save_csv(validation, "mle_validation.csv")
    comparison, estimates = ms.compare_distributions(x, fitted, n_boot=n_boot)
    save_csv(comparison, "model_comparison.csv")
    save_csv(estimates, "mle_estimates.csv")
    cols = ["model", "k", "loglik", "AIC", "BIC", "delta_AIC", "delta_BIC", "KS_statistic", "KS_p_bootstrap", "AD_p_bootstrap", "converged"]
    print(comparison[cols].to_string(index=False))
    best_dist_aic, best_dist_bic = ms.select_best(comparison, "AIC"), ms.select_best(comparison, "BIC")
    standard_only = comparison[comparison["model"] != "2-component normal mixture"]
    key["distributions"] = {"best_AIC": best_dist_aic, "best_BIC": best_dist_bic,
                            "best_single_family_BIC": ms.select_best(standard_only, "BIC"),
                            "best_single_family_AIC": ms.select_best(standard_only, "AIC")}
    pl.plot_fitted_densities(x, fitted)
    pl.plot_qq(x, fitted)
    pl.plot_likelihood_surface(x, fitted["Weibull"][0], fitted["Weibull"][1])
    pl.plot_model_selection(comparison)

    # PIT / serial dependence of the i.i.d. best model (time-ordered concatenated daylight values)
    best_d, best_f = fitted[best_dist_bic]
    pit = best_d.cdf(x, best_d.params_array(best_f))
    pit_scores = stats.norm.ppf(np.clip(pit, 1e-6, 1 - 1e-6))
    iid_diag = pd.DataFrame(dg.ljung_box(pit_scores, f"{best_dist_bic} (i.i.d.), normal scores of PIT", lags=config.LJUNG_BOX_LAGS))

    # supplementary: strictly positive raw hourly MW
    pos = hourly[hourly > 0].to_numpy()
    fitted_raw = mm.fit_all(pos, mm.build_candidates(pos))
    comp_raw, _ = ms.compare_distributions(pos, fitted_raw, do_gof=False)
    save_csv(comp_raw, "supplementary_raw_positive_mw_comparison.csv")
    key["raw_positive"] = {"n": int(len(pos)), "best_BIC": ms.select_best(comp_raw, "BIC"), "best_AIC": ms.select_best(comp_raw, "AIC")}

    # ------------------------------------------------------------------ 5. sensitivity of the ranking
    section("5. Sensitivity of the distribution ranking")
    variants = [("Baseline (q=0.95, +/-7 d, threshold 10%)", {}),
                ("Envelope quantile 0.90", {"quantile": 0.90}), ("Envelope quantile 0.98", {"quantile": 0.98}),
                ("Daylight threshold 5%", {"min_fraction": 0.05}), ("Daylight threshold 20%", {"min_fraction": 0.20}),
                ("Envelope window +/-15 days", {"half_window": 15})]
    sens_rows = []
    samples = [(label, pp.build_analysis_frame(hourly, **kw)) for label, kw in variants]
    noon_frame = samples[0][1]
    samples.append(("Noon (12:00) values only, n~365", noon_frame[noon_frame["hour"] == 12]))
    samples.append(("Every 3rd day, all daylight hours", noon_frame[noon_frame["day_index"] % 3 == 0]))
    for label, fr in samples:
        xs = fr.loc[fr["in_sample"], "clearsky_index"].to_numpy()
        fits = mm.fit_all(xs, mm.build_candidates(xs))
        table, _ = ms.compare_distributions(xs, fits, do_gof=False)
        single = table[table["model"] != "2-component normal mixture"]
        sens_rows.append({"variant": label, "n": len(xs), "best_by_AIC": ms.select_best(table, "AIC"), "best_by_BIC": ms.select_best(table, "BIC"),
                          "best_single_family_BIC": ms.select_best(single, "BIC"),
                          "delta_BIC_runner_up": float(table.sort_values("BIC")["delta_BIC"].iloc[1]),
                          "mixture_weight1": float(fits["2-component normal mixture"][1].params["weight1"]),
                          "mixture_mu1": float(fits["2-component normal mixture"][1].params["mu1"]),
                          "mixture_mu2": float(fits["2-component normal mixture"][1].params["mu2"])})
    sens = pd.DataFrame(sens_rows)
    save_csv(sens, "sensitivity_distribution_ranking.csv")
    print(sens.to_string(index=False))

    # ------------------------------------------------------------------ 6. time-series models
    section("6. Time-series models with exact Gaussian maximum likelihood")
    endog, exog = tsm.build_endog_exog(frame)
    ts_table, ts_results = tsm.fit_candidates(endog, exog)
    save_csv(ts_table, "timeseries_model_comparison.csv")
    best_ts_bic, best_ts_aic = str(ts_table.iloc[0]["model"]), str(ts_table.sort_values("AIC").iloc[0]["model"])
    spec_lookup = {s.name: s for s in tsm.DEFAULT_SPECS}
    best_res = ts_results[best_ts_bic]
    params = tsm.parameter_table(best_res, best_ts_bic)
    save_csv(params, "timeseries_best_model_parameters.csv")
    print(params.round(4).to_string(index=False))

    # likelihood-ratio test for the annual cycle
    no_cycle_name = "ARMA(2,1)x(1,0,0)24, no annual cycle"
    lr_stat = 2 * (ts_results["ARMA(2,1)x(1,0,0)24"].llf - ts_results[no_cycle_name].llf)
    lr = {"statistic": float(lr_stat), "df": 2, "p_value": float(stats.chi2.sf(lr_stat, 2))}
    # time-series vs i.i.d. on the SAME observations
    n_ts = int(ts_table["n"].iloc[0])
    assert n_ts == int(comparison["n"].iloc[0]), "i.i.d. and time-series comparisons must use the same observations"
    cross = pd.concat([comparison[["model", "k", "n", "loglik", "AIC", "BIC"]].assign(family="i.i.d. distribution"),
                       ts_table[["model", "k", "n", "loglik", "AIC", "BIC"]].assign(family="ARMA-type (time order used)")])
    cross["delta_BIC"] = cross["BIC"] - cross["BIC"].min()
    cross = cross.sort_values("BIC").reset_index(drop=True)
    save_csv(cross, "model_comparison_all_families.csv")
    pl.plot_cross_family(comparison, ts_table)

    # residual diagnostics for every ARMA candidate
    diag_frames = [iid_diag.assign(family="i.i.d. distribution")]
    for name, res in ts_results.items():
        z = tsm.standardized_residuals(res)
        diag_frames.append(dg.residual_diagnostics(z, name, model_df=spec_lookup[name].n_arma_coefficients).assign(family="ARMA"))

    # structure of the residual misfit for the selected model
    z_full = pd.Series(np.where(endog.notna(), np.asarray(best_res.standardized_forecasts_error).ravel(), np.nan), index=endog.index)
    fitted_vals = pd.Series(np.asarray(best_res.fittedvalues).ravel(), index=endog.index)
    z_obs = z_full.dropna()
    quint = pd.qcut(fitted_vals[z_obs.index], 5, labels=[f"Q{i}" for i in range(1, 6)])
    var_level = z_obs.groupby(quint).agg(["std", "count"]).rename(columns={"std": "std_standardised_residual"}).reset_index().rename(columns={"index": "fitted_value_quintile"})
    var_level.columns = ["fitted_value_quintile", "std_standardised_residual", "count"]
    var_level["mean_fitted_index"] = fitted_vals[z_obs.index].groupby(quint).mean().to_numpy()
    var_hour = z_obs.groupby(z_obs.index.hour).agg(["std", "count"]).reset_index()
    var_hour.columns = ["hour", "std_standardised_residual", "count"]
    save_csv(var_level, "residual_variance_by_fitted_level.csv")
    save_csv(var_hour, "residual_variance_by_hour.csv")
    resid_acf = st.pairwise_acf(z_full.to_numpy(), 48)
    key["residual_structure"] = {"acf_lag1": float(resid_acf[1]), "acf_lag13": float(resid_acf[13]), "acf_lag24": float(resid_acf[24]),
                                 "std_lowest_level_quintile": float(var_level["std_standardised_residual"].iloc[0]),
                                 "std_highest_level_quintile": float(var_level["std_standardised_residual"].iloc[-1])}

    # GARCH on the innovations of the selected ARMA model
    innovations = tsm.raw_residuals(best_res)
    garch_table, garch_fits = tsm.fit_garch_family(innovations)
    save_csv(garch_table, "garch_comparison.csv")
    print(garch_table[["model", "k", "loglik", "AIC", "BIC", "converged"]].to_string(index=False))
    best_garch = str(garch_table.iloc[0]["model"])
    gfit = garch_fits[best_garch]
    garch_params = pd.DataFrame({"model": best_garch, "parameter": gfit.params.index, "estimate": gfit.params.to_numpy(),
                                 "std_error": gfit.std_err.to_numpy(), "p_value": gfit.pvalues.to_numpy()})
    save_csv(garch_params, "garch_best_model_parameters.csv")
    z_garch = np.asarray(gfit.std_resid)
    diag_frames.append(dg.residual_diagnostics(z_garch, f"{best_garch} on {best_ts_bic} innovations (standardised)",
                                               model_df=spec_lookup[best_ts_bic].n_arma_coefficients).assign(family="ARMA-GARCH"))
    diagnostics_table = pd.concat(diag_frames, ignore_index=True)
    save_csv(diagnostics_table, "diagnostic_results.csv")
    best_diag_lb = diagnostics_table[(diagnostics_table["model"] == best_ts_bic) & diagnostics_table["test"].str.startswith("Ljung-Box")]
    pl.plot_arma_diagnostics(z_full, best_diag_lb, best_ts_bic, resid_acf)
    pl.plot_garch(pd.Series(innovations), np.asarray(gfit.conditional_volatility))
    garch_vol = np.asarray(gfit.conditional_volatility)

    # annual-cycle and sensitivity checks
    if not fast:
        gap = tsm.night_gap_sensitivity(frame)
        save_csv(gap, "night_gap_sensitivity.csv")
        print(gap.to_string(index=False))
        key["night_gap"] = gap.to_dict(orient="records")

    # ------------------------------------------------------------------ 7. forecasting
    section("7. Out-of-sample one-hour-ahead forecasting (chronological split)")
    fe = tsm.forecast_experiment(hourly)
    save_csv(fe["scores"], "forecast_scores.csv")
    save_csv(fe["selection_table"][["model", "n", "k", "loglik", "AIC", "BIC", "converged"]], "forecast_training_model_selection.csv")
    print(fe["scores"].to_string(index=False))
    pl.plot_forecast(fe["test_frame"], fe["scores"][fe["scores"]["method"].isin([m for m in fe["scores"]["method"] if m.startswith("Selected") or m.startswith("Persistence")])],
                     fe["selected_model"])

    # ------------------------------------------------------------------ 8. key results
    arma_params = params.set_index("parameter")["estimate"].to_dict()
    key["timeseries"] = {"best_BIC": best_ts_bic, "best_AIC": best_ts_aic, "n": n_ts,
                         "best_loglik": float(ts_table.iloc[0]["loglik"]), "best_params": arma_params,
                         "lr_test_annual_cycle": lr, "best_garch": best_garch, "garch_params": gfit.params.to_dict(),
                         "garch_persistence": float(gfit.params.get("alpha[1]", np.nan) + gfit.params.get("beta[1]", 0.0)),
                         "diag_summary": dg.diagnostics_pass_summary(diagnostics_table, best_ts_bic),
                         "diag_summary_garch": dg.diagnostics_pass_summary(diagnostics_table, f"{best_garch} on {best_ts_bic} innovations (standardised)")}
    key["forecast"] = {"selected_model": fe["selected_model"], "train_days": fe["train_days"], "test_days": fe["test_days"],
                       "train_end": fe["train_end"], "test_start": fe["test_start"], "train_nobs": fe["train_nobs"],
                       "n_test": int(fe["scores"]["n_test"].iloc[0]), "scores": fe["scores"].to_dict(orient="records")}
    key["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                          "runtime_seconds": round(time.time() - started, 1)}
    for lib in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "seaborn", "arch"):
        module = __import__(lib)
        key["environment"][lib] = module.__version__
    (config.RESULTS_DIR / "key_results.json").write_text(json.dumps(native(key), indent=2))
    section(f"Done in {time.time() - started:.0f} s. Results in {config.RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true", help="smoke-test mode: fewer bootstrap replicates, skip the night-gap check")
    arguments = parser.parse_args()
    try:
        main(fast=arguments.fast)
    except FileNotFoundError as error:
        sys.exit(f"Error: {error}")
