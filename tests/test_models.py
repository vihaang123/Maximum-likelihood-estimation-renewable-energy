"""Unit tests.  Run from the project root with:  python -m pytest -q

The tests use simulated data with KNOWN parameters, so they do not depend on the project dataset (except the
final integration test, which is skipped if the dataset is missing).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, mle_models as mm, model_selection as ms, preprocessing as pp, stationarity as st  # noqa: E402

RNG = np.random.default_rng(12345)


# ------------------------------------------------------------------ closed-form checks
def test_normal_mle_matches_closed_form():
    x = RNG.normal(0.7, 0.2, 5000)
    fit = mm.Normal().fit(x)
    assert fit.params["mu"] == pytest.approx(x.mean(), abs=1e-6)
    assert fit.params["sigma"] == pytest.approx(x.std(), abs=1e-6)         # MLE uses 1/n, i.e. numpy default
    assert fit.converged


def test_lognormal_mle_matches_closed_form():
    x = RNG.lognormal(-0.3, 0.5, 5000)
    fit = mm.LogNormal().fit(x)
    assert fit.params["mu_log"] == pytest.approx(np.log(x).mean(), abs=1e-6)
    assert fit.params["sigma_log"] == pytest.approx(np.log(x).std(), abs=1e-6)


def test_gamma_profile_likelihood_equation():
    x = RNG.gamma(5.0, 0.15, 5000)
    fit = mm.Gamma().fit(x)
    a, scale = fit.params["shape"], fit.params["scale"]
    from scipy.special import digamma
    assert scale == pytest.approx(x.mean() / a, rel=1e-6)
    assert np.log(a) - digamma(a) == pytest.approx(np.log(x.mean()) - np.log(x).mean(), abs=1e-6)


# ------------------------------------------------------------------ recovery and validation vs scipy
@pytest.mark.parametrize("dist, sampler, truth", [
    (mm.Weibull(), lambda n: 0.8 * RNG.weibull(3.5, n), {"shape": 3.5, "scale": 0.8}),
    (mm.StudentT(), lambda n: 0.8 + 0.2 * RNG.standard_t(5, n), {"mu": 0.8, "sigma": 0.2, "nu": 5.0}),
])
def test_parameter_recovery_and_scipy_agreement(dist, sampler, truth):
    x = sampler(20000)
    fit = dist.fit(x)
    for name, value in truth.items():
        assert fit.params[name] == pytest.approx(value, rel=0.08)
    report = mm.validate_against_scipy(x, dist, fit)
    assert report["validated"]
    assert abs(report["loglik_difference_ours_minus_scipy"]) < 1e-2


def test_beta_loglik_includes_jacobian():
    y = RNG.beta(3.0, 2.0, 4000)
    c = 1.3
    x = c * y
    fit = mm.Beta(upper_bound=c).fit(x)
    assert fit.params["alpha"] == pytest.approx(3.0, rel=0.1)
    ll_y = np.sum(stats.beta.logpdf(y, fit.params["alpha"], fit.params["beta"]))
    assert fit.loglik == pytest.approx(ll_y - len(x) * np.log(c), abs=1e-6)


def test_mixture_recovers_two_components_and_orders_them():
    x = np.concatenate([RNG.normal(0.55, 0.2, 3000), RNG.normal(0.93, 0.06, 3000)])
    fit = mm.GaussianMixture2().fit(x)
    p = fit.params
    assert p["mu1"] < p["mu2"]
    assert p["mu1"] == pytest.approx(0.55, abs=0.03) and p["mu2"] == pytest.approx(0.93, abs=0.02)
    assert p["weight1"] == pytest.approx(0.5, abs=0.05)
    assert fit.converged


def test_gamma_rejects_non_positive_data():
    with pytest.raises(ValueError):
        mm.Gamma().fit(np.array([-0.1, 0.5, 0.7]))


def test_standard_errors_shrink_with_sample_size():
    small = mm.Normal().fit(RNG.normal(0, 1, 200)).std_errors["mu"]
    large = mm.Normal().fit(RNG.normal(0, 1, 20000)).std_errors["mu"]
    assert large < small / 5


# ------------------------------------------------------------------ model selection
def test_information_criteria_formulas():
    ll, k, n = -123.4, 3, 500
    assert ms.aic(ll, k) == pytest.approx(2 * 3 + 246.8)
    assert ms.bic(ll, k, n) == pytest.approx(3 * np.log(500) + 246.8)
    assert ms.aicc(ll, k, n) == pytest.approx(ms.aic(ll, k) + 2 * 3 * 4 / (500 - 3 - 1))


def test_information_table_rejects_different_n_and_ranks():
    with pytest.raises(ValueError):
        ms.information_table([{"model": "a", "loglik": 1.0, "k": 1, "n": 10}, {"model": "b", "loglik": 2.0, "k": 1, "n": 11}])
    table = ms.information_table([{"model": "a", "loglik": 10.0, "k": 2, "n": 100},
                                  {"model": "b", "loglik": 12.0, "k": 5, "n": 100}])
    assert table.iloc[0]["rank_AIC"] == 1
    assert table["weight_AIC"].sum() == pytest.approx(1.0)


def test_bootstrap_gof_does_not_reject_the_true_model():
    x = RNG.normal(0.0, 1.0, 400)
    fit = mm.Normal().fit(x)
    out = ms.parametric_bootstrap_gof(mm.Normal(), mm.Normal().params_array(fit), x, n_boot=60, seed=1)
    assert out["KS_p_bootstrap"] > 0.05 and out["AD_p_bootstrap"] > 0.05


def test_correct_model_beats_wrong_model_by_aic():
    x = RNG.gamma(5.0, 0.15, 8000)
    table, _ = ms.compare_distributions(x, mm.fit_all(x, [mm.Normal(), mm.Gamma(), mm.Weibull()]), do_gof=False)
    assert table.iloc[0]["model"] == "Gamma"


# ------------------------------------------------------------------ stationarity helpers
def test_adf_and_kpss_distinguish_noise_from_random_walk():
    noise = RNG.normal(size=1500)
    walk = np.cumsum(RNG.normal(size=1500))
    assert st.adf_test(noise, "noise")["reject_H0"] and not st.adf_test(walk, "walk")["reject_H0"]
    assert not st.kpss_test(noise, "noise")["reject_H0"] and st.kpss_test(walk, "walk")["reject_H0"]


def test_pairwise_acf_and_pacf_of_ar1():
    phi, n = 0.7, 20000
    e = RNG.normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    acf = st.pairwise_acf(x, 5)
    pacf = st.pacf_from_acf(acf, 5)
    assert acf[1] == pytest.approx(phi, abs=0.03) and acf[2] == pytest.approx(phi ** 2, abs=0.03)
    assert pacf[1] == pytest.approx(phi, abs=0.03) and abs(pacf[2]) < 0.03


# ------------------------------------------------------------------ preprocessing
def test_envelope_and_index_shape_on_synthetic_days():
    days = 40
    profile = np.maximum(0, np.sin(np.linspace(-0.5, np.pi + 0.5, 24)))
    hourly = np.tile(profile * 1000, days)
    import pandas as pd
    series = pd.Series(hourly, index=pd.date_range("2006-01-01", periods=days * 24, freq="h"))
    frame = pp.build_analysis_frame(series)
    assert frame.loc[frame["in_sample"], "clearsky_index"].between(0.99, 1.01).all()
    assert (frame["hourly_total_mw"] == series).all()               # raw output is never altered


def test_causal_envelope_never_uses_future_values():
    matrix = np.ones((30, 24))
    matrix[20] = 100.0                                              # a spike on day 20
    env = pp.compute_envelope(matrix, quantile=1.0, causal=True, trailing_days=14)
    assert np.all(env[:20][~np.isnan(env[:20])] == 1.0)             # days before the spike never see it
    assert env[21, 0] == 100.0


@pytest.mark.skipif(not config.RAW_FILE.exists(), reason="dataset file not present")
def test_dataset_integrity():
    matrix = pp.load_raw_matrix()
    checks = pp.validate_raw(matrix, pp.build_timestamps(len(matrix)))
    assert checks["rows_match_expected"] and checks["plants_match_expected"]
    assert checks["n_missing"] == 0 and checks["n_negative"] == 0 and checks["n_duplicate_timestamps"] == 0
