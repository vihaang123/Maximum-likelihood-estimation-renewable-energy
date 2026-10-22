"""Builds notebooks/analysis.ipynb (a guided walk-through that re-uses the src modules).

Run from the project root:  python notebooks/make_notebook.py
Then execute it (optional, needs `pip install jupyter`):
    jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md("# Maximum Likelihood Estimation and Model Selection for Stationary Renewable Energy Output Data\n"
       "Guided walk-through. The heavy computations live in `src/` and are run by `main.py`; this notebook re-runs the key steps "
       "interactively and displays the saved results. Run `python main.py` first so that `results/` exists."),
    code("import sys, json, warnings\nfrom pathlib import Path\nROOT = Path.cwd()\nif not (ROOT / 'src').exists():\n    ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT))\n"
         "warnings.filterwarnings('ignore')\nimport numpy as np, pandas as pd\nfrom IPython.display import Image, display\n"
         "from src import config, preprocessing as pp, stationarity as st, mle_models as mm, model_selection as ms\n"
         "pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)"),
    md("## 1. Data: load, validate, aggregate"),
    code("matrix = pp.load_raw_matrix()\nindex10 = pp.build_timestamps(len(matrix))\nprint(json.dumps(pp.validate_raw(matrix, index10), indent=1))\n"
         "hourly = pp.to_hourly(pp.aggregate_fleet(matrix, index10))\nframe = pp.build_analysis_frame(hourly)\nprint(pp.summarize_missing_and_unusual(frame))\nframe.head(12)"),
    code("display(Image(str(config.FIG_DIR / 'fig01_hourly_series_overview.png')))"),
    md("## 2. Zeros and the clear-sky index\nNight-time zeros are structural, so the analysis series is the clear-sky index "
       "(output divided by an empirical clear-sky envelope) on daylight hours."),
    code("display(Image(str(config.FIG_DIR / 'fig04_clearsky_envelope_and_index.png')))"),
    md("## 3. Stationarity"),
    code("tests = pd.read_csv(config.RESULTS_DIR / 'stationarity_tests.csv')\ntests[['series','test','specification','statistic','p_value','reject_H0','interpretation']]"),
    code("pd.read_csv(config.RESULTS_DIR / 'stationarity_verdicts.csv')"),
    code("display(Image(str(config.FIG_DIR / 'fig05_rolling_statistics_and_group_effects.png')))\ndisplay(Image(str(config.FIG_DIR / 'fig06b_acf_pacf_clearsky_index.png')))"),
    md("## 4. Maximum likelihood estimation\nA minimal demonstration with the Weibull model: maximise the log-likelihood numerically and compare with SciPy."),
    code("x = frame.loc[frame['in_sample'], 'clearsky_index'].to_numpy()\nweibull = mm.Weibull()\nfit = weibull.fit(x)\nprint(fit.params, fit.loglik, fit.converged)\nprint(mm.validate_against_scipy(x, weibull, fit))"),
    code("pd.read_csv(config.RESULTS_DIR / 'mle_estimates.csv')"),
    code("display(Image(str(config.FIG_DIR / 'fig07_likelihood_surface_weibull.png')))"),
    md("## 5. Model selection (AIC / BIC / goodness of fit)"),
    code("pd.read_csv(config.RESULTS_DIR / 'model_comparison.csv')[['model','k','loglik','AIC','BIC','delta_AIC','delta_BIC','weight_BIC','KS_statistic','KS_p_bootstrap','AD_p_bootstrap']]"),
    code("display(Image(str(config.FIG_DIR / 'fig08_fitted_densities.png')))\ndisplay(Image(str(config.FIG_DIR / 'fig09_qq_plots.png')))"),
    md("## 6. Time-series models and diagnostics"),
    code("pd.read_csv(config.RESULTS_DIR / 'timeseries_model_comparison.csv')[['model','k','loglik','AIC','BIC','delta_BIC','converged']]"),
    code("pd.read_csv(config.RESULTS_DIR / 'model_comparison_all_families.csv')"),
    code("display(Image(str(config.FIG_DIR / 'fig12_arma_residual_diagnostics.png')))"),
    code("d = pd.read_csv(config.RESULTS_DIR / 'diagnostic_results.csv')\nd[d['family'] != 'i.i.d. distribution'].query(\"test in ['Ljung-Box','ARCH-LM']\").head(30)"),
    md("## 7. Out-of-sample forecasting"),
    code("pd.read_csv(config.RESULTS_DIR / 'forecast_scores.csv')"),
    code("display(Image(str(config.FIG_DIR / 'fig14_forecast_evaluation.png')))"),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = Path(__file__).resolve().parent / "analysis.ipynb"
nbf.write(nb, out)
print("wrote", out)
