"""Central configuration for the project.

Every analysis setting lives here so that results are reproducible and so that
the report can state exactly which settings were used.
All paths are relative to the project root (no machine-specific absolute paths).
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
REPORT_DIR = PROJECT_ROOT / "report"

RAW_FILE = DATA_DIR / "solar_AL.txt.gz"          # original file, byte-for-byte as downloaded
DOWNLOAD_INFO_FILE = DATA_DIR / "download_info.json"
CLEAN_FILE = DATA_DIR / "cleaned_dataset.csv"
METADATA_FILE = DATA_DIR / "dataset_metadata.md"

DATA_URL = (
    "https://raw.githubusercontent.com/laiguokun/"
    "multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz"
)

# --------------------------------------------------------------------------- #
# Dataset layout assumptions (see data/dataset_metadata.md for the evidence)
# --------------------------------------------------------------------------- #
START_TIMESTAMP = "2006-01-01 00:00"   # local standard time, first row of the file
SAMPLE_MINUTES = 10                    # sampling interval of the raw file
EXPECTED_ROWS = 365 * 24 * 60 // SAMPLE_MINUTES   # 52 560 rows for a 365-day year
EXPECTED_PLANTS = 137

# --------------------------------------------------------------------------- #
# Statistical settings
# --------------------------------------------------------------------------- #
SEED = 20061019                 # master random seed (bootstrap, simulations)
ALPHA = 0.05                    # significance level for all hypothesis tests

# Empirical clear-sky envelope (a location-free proxy for clear-sky output)
ENVELOPE_QUANTILE = 0.95        # quantile of same-hour outputs within the window
ENVELOPE_HALF_WINDOW = 7        # +/- 7 days  (15-day window)
MIN_ENVELOPE_FRACTION = 0.10    # daylight window: envelope >= 10 % of its annual maximum
CAUSAL_TRAILING_DAYS = 14       # trailing window length used for the leak-free forecasting envelope

# Annual harmonic regressors for the time-series models (justified in stationarity step)
ANNUAL_HARMONICS = 1

# Goodness-of-fit bootstrap
N_BOOTSTRAP = 200

# Time-series candidate search
MAX_LJUNG_LAGS = 48
LJUNG_BOX_LAGS = (10, 24, 48)

# Forecast experiment: chronological split (first 75 % of days = training)
TRAIN_FRACTION = 0.75
