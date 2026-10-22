# Maximum Likelihood Estimation and Model Selection for Stationary Renewable Energy Output Data

Academic project: statistical analysis of solar PV output (NREL "Solar Power Data for Integration Studies",
Alabama 2006, 137 plants), stationarity testing, maximum likelihood estimation of candidate distributions and
ARMA/GARCH models, and AIC/BIC model selection.

> **Data caveat (read first).** The series is NREL-published *modelled* PV plant output, not metered generation.
> Timestamps are reconstructed from the documented regular 10-minute layout. Details and evidence are in
> `data/dataset_metadata.md` and in Section 10 of the report.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                       # full analysis, about 6-10 minutes, regenerates results/ and figures
python -m pytest -q                  # unit tests (about 30 seconds)
python report/build_report.py        # rebuilds report/project_report.docx from results/ (needs python-docx)
```

`python main.py --fast` is a quick smoke test (fewer bootstrap replicates, skips one slow check); use the full run for
the numbers reported. No API key, account or paid service is needed: the dataset is included in `data/`.
The PDF version of the report is produced from the DOCX with LibreOffice
(`soffice --headless --convert-to pdf --outdir report report/project_report.docx`); a copy is included.

## Folder structure

```
main.py                   runs the whole analysis
requirements.txt
data/
  solar_AL.txt.gz         ORIGINAL dataset, byte-for-byte (52,560 x 137, 10-minute, MW)
  cleaned_dataset.csv     hourly fleet total + clear-sky envelope + clear-sky index + flags
  dataset_metadata.md     source, units, licence note, integrity checks, assumptions
  download_info.json      URL, download date, SHA-256
src/
  config.py               all settings (seed, alpha, window sizes, split ...)
  preprocessing.py        loading, validation, aggregation, envelope, clear-sky index
  stationarity.py         ADF, KPSS, rolling statistics, group tests, ACF/PACF
  mle_models.py           MLE of 7 distributions (own log-likelihoods, BFGS, SEs, validation vs SciPy)
  model_selection.py      AIC/BIC/AICc, weights, bootstrap KS/AD goodness of fit
  diagnostics.py          Ljung-Box, ARCH-LM, Jarque-Bera
  timeseries_models.py    ARMA via state-space ML, GARCH, night-gap check, forecasting experiment
  plotting.py             all figures
  download_data.py        optional re-download with checksum verification
notebooks/analysis.ipynb  guided walk-through that re-uses the src modules and displays the saved results
results/                  CSV tables, key_results.json, figures/
report/                   project_report.docx, project_report.pdf (course format), project_report_detailed.* (longer 27-part version), build_report.py, build_report_detailed.py
tests/test_models.py      unit tests with known-parameter simulations
```

## What the analysis does

1. **Data**: validate (no missing/negative/duplicate values), sum plants, average to hourly means.
2. **Zeros**: night-time zeros (46.5 % of hours) are structural. Raw output is kept untouched; distribution fitting and
   time-series modelling use the *clear-sky index* (output / empirical clear-sky envelope) on daylight hours.
3. **Stationarity**: ADF and KPSS (several specifications), rolling statistics, hour-of-day and month ANOVA/Levene tests,
   ACF/PACF, seasonal differencing, annual-harmonic removal.
4. **MLE**: Normal, Student-t, Log-normal, Gamma, Weibull, scaled Beta and a 2-component normal mixture, fitted by
   numerical maximisation of the log-likelihood with convergence/Hessian checks and independent validation.
5. **Selection**: AIC, BIC, AICc, Akaike weights, bootstrap KS/Anderson-Darling tests (parameters re-estimated), Q-Q plots.
6. **Time series**: regression with (seasonal) ARMA errors by exact Gaussian ML with night hours as missing values;
   residual diagnostics; GARCH(1,1); i.i.d. versus dynamic models compared on the same observations.
7. **Forecasting**: one-hour-ahead, chronological split, causal envelope, order chosen on training data only.

## Reproducibility notes

* Random seed and all settings are in `src/config.py`; the bootstrap uses a fixed seed.
* The project was run with Python 3.13, numpy 2.5, pandas 3.0, scipy 1.18, statsmodels 0.15, matplotlib 3.11,
  seaborn 0.13, arch 8.0 (exact versions are stored in `results/key_results.json`).
* All paths are relative to the project root.
