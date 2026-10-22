"""Data loading, validation, aggregation and the clear-sky-index transformation.

Pipeline
--------
raw 10-minute matrix (52 560 x 137 plants, MW)
  -> fleet total per 10-minute step
  -> hourly mean power (8 760 values)
  -> empirical clear-sky envelope E(day, hour)           (high quantile of same-hour output)
  -> clear-sky index  I = P / E  on daylight hours       (removes the deterministic diurnal/annual shape)
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def load_raw_matrix(path: Path = config.RAW_FILE) -> np.ndarray:
    """Read the gzip-compressed, comma-separated plant matrix (rows = time, cols = plants)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Re-download it with `python -m src.download_data`."
        )
    try:
        with gzip.open(path, "rt") as handle:
            frame = pd.read_csv(handle, header=None)
    except (OSError, pd.errors.ParserError) as exc:  # corrupted gzip / malformed csv
        raise ValueError(f"Could not parse {path}: {exc}") from exc
    return frame.to_numpy(dtype=float)


def build_timestamps(n_rows: int,
                     start: str = config.START_TIMESTAMP,
                     minutes: int = config.SAMPLE_MINUTES) -> pd.DatetimeIndex:
    """The file has no time column; timestamps follow the documented regular 10-minute layout."""
    return pd.date_range(start=start, periods=n_rows, freq=f"{minutes}min")


def validate_raw(matrix: np.ndarray, index: pd.DatetimeIndex) -> dict:
    """Run integrity checks and return them as a dictionary (also written to the metadata)."""
    steps = pd.Series(index).diff().dropna().unique()
    checks = {
        "n_rows": int(matrix.shape[0]),
        "n_plants": int(matrix.shape[1]),
        "expected_rows": int(config.EXPECTED_ROWS),
        "rows_match_expected": bool(matrix.shape[0] == config.EXPECTED_ROWS),
        "plants_match_expected": bool(matrix.shape[1] == config.EXPECTED_PLANTS),
        "n_missing": int(np.isnan(matrix).sum()),
        "n_non_finite": int((~np.isfinite(matrix)).sum()),
        "n_negative": int((matrix < 0).sum()),
        "n_duplicate_timestamps": int(index.duplicated().sum()),
        "n_distinct_time_steps": int(len(steps)),
        "time_step_minutes": float(steps[0] / np.timedelta64(1, "m")) if len(steps) else float("nan"),
        "n_plants_always_zero": int((matrix.sum(axis=0) == 0).sum()),
        "plant_max_mw": float(np.nanmax(matrix)),
        "first_timestamp": str(index[0]),
        "last_timestamp": str(index[-1]),
    }
    return checks


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_download_info() -> dict:
    if config.DOWNLOAD_INFO_FILE.exists():
        return json.loads(config.DOWNLOAD_INFO_FILE.read_text())
    return {}


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate_fleet(matrix: np.ndarray, index: pd.DatetimeIndex) -> pd.Series:
    """Sum of all plants at each 10-minute step (MW). No values are altered or removed."""
    return pd.Series(matrix.sum(axis=1), index=index, name="fleet_total_mw")


def to_hourly(series_10min: pd.Series) -> pd.Series:
    """Hourly mean power. Each hour averages exactly six 10-minute values (checked)."""
    counts = series_10min.resample("1h").count()
    if not (counts == 6).all():
        raise ValueError("Some hours do not contain six 10-minute observations.")
    hourly = series_10min.resample("1h").mean()
    hourly.name = "hourly_total_mw"
    return hourly


# --------------------------------------------------------------------------- #
# Empirical clear-sky envelope and clear-sky index
# --------------------------------------------------------------------------- #
def compute_envelope(hourly_matrix: np.ndarray,
                     quantile: float = config.ENVELOPE_QUANTILE,
                     half_window: int = config.ENVELOPE_HALF_WINDOW,
                     causal: bool = False,
                     trailing_days: int = config.CAUSAL_TRAILING_DAYS) -> np.ndarray:
    """High-quantile envelope of same-hour output, shape (n_days, 24).

    centered (causal=False): quantile over days d-half_window ... d+half_window (circular wrap at the
        year ends).  Uses the whole year, so it is appropriate for in-sample description/estimation only.
    causal (causal=True): quantile over the previous `trailing_days` days only (today excluded).  It never
        uses future information, so it is the version used in the out-of-sample forecasting experiment.
        The first `trailing_days` days have no envelope (NaN).
    """
    n_days = hourly_matrix.shape[0]
    envelope = np.full(hourly_matrix.shape, np.nan)
    for day in range(n_days):
        if causal:
            if day < trailing_days:
                continue
            window = np.arange(day - trailing_days, day)
        else:
            window = np.arange(day - half_window, day + half_window + 1) % n_days
        envelope[day] = np.quantile(hourly_matrix[window], quantile, axis=0)
    return envelope


def build_analysis_frame(hourly: pd.Series,
                         quantile: float = config.ENVELOPE_QUANTILE,
                         half_window: int = config.ENVELOPE_HALF_WINDOW,
                         min_fraction: float = config.MIN_ENVELOPE_FRACTION,
                         causal: bool = False) -> pd.DataFrame:
    """Hourly data frame with envelope, daylight flag and clear-sky index.

    Columns
    -------
    hourly_total_mw   fleet output (never modified)
    envelope_mw       empirical clear-sky envelope
    in_daylight       True when envelope >= min_fraction * max(envelope)
    clearsky_index    hourly_total_mw / envelope_mw on daylight hours, NaN otherwise
    in_sample         daylight AND index > 0 (support requirement of positive-support distributions)
    unusual           flag only (nothing is deleted): index > 1.20 or index < 0.05 on daylight hours
    """
    if len(hourly) % 24 != 0:
        raise ValueError("Hourly series must cover whole days.")
    n_days = len(hourly) // 24
    matrix = hourly.to_numpy().reshape(n_days, 24)
    envelope = compute_envelope(matrix, quantile, half_window, causal=causal)
    envelope_flat = envelope.ravel()

    frame = pd.DataFrame({"hourly_total_mw": hourly.to_numpy(), "envelope_mw": envelope_flat},
                         index=hourly.index)
    threshold = min_fraction * np.nanmax(envelope_flat)
    frame["in_daylight"] = frame["envelope_mw"] >= threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        index_values = np.where(frame["in_daylight"], frame["hourly_total_mw"] / frame["envelope_mw"], np.nan)
    frame["clearsky_index"] = index_values
    frame["in_sample"] = frame["in_daylight"] & (frame["clearsky_index"] > 0)
    frame["unusual"] = frame["in_daylight"] & ((frame["clearsky_index"] > 1.20) | (frame["clearsky_index"] < 0.05))
    frame["hour"] = frame.index.hour
    frame["day_of_year"] = frame.index.dayofyear
    frame["month"] = frame.index.month
    frame["day_index"] = np.arange(len(frame)) // 24
    return frame


def annual_harmonic_design(day_of_year: np.ndarray, n_harmonics: int = config.ANNUAL_HARMONICS,
                           period: float = 365.0, include_constant: bool = True) -> np.ndarray:
    """Design matrix [1, sin(2*pi*k*d/P), cos(2*pi*k*d/P)] for k = 1..n_harmonics."""
    day_of_year = np.asarray(day_of_year, dtype=float)
    columns = [np.ones_like(day_of_year)] if include_constant else []
    for k in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * k * day_of_year / period
        columns += [np.sin(angle), np.cos(angle)]
    return np.column_stack(columns)


def summarize_missing_and_unusual(frame: pd.DataFrame) -> dict:
    """Counts used in the report (no observation is removed because of these flags)."""
    daylight = frame["in_daylight"]
    return {
        "n_hours": int(len(frame)),
        "n_hours_exact_zero": int((frame["hourly_total_mw"] == 0).sum()),
        "share_hours_exact_zero": float((frame["hourly_total_mw"] == 0).mean()),
        "n_daylight_hours": int(daylight.sum()),
        "n_daylight_zero_output": int((daylight & (frame["hourly_total_mw"] == 0)).sum()),
        "n_in_sample": int(frame["in_sample"].sum()),
        "n_unusual_flagged": int(frame["unusual"].sum()),
        "n_index_above_1_20": int((frame["clearsky_index"] > 1.20).sum()),
        "n_index_below_0_05": int((frame["clearsky_index"] < 0.05).sum()),
        "n_index_above_1": int((frame["clearsky_index"] > 1.0).sum()),
        "max_index": float(np.nanmax(frame["clearsky_index"])),
        "envelope_threshold_mw": float(config.MIN_ENVELOPE_FRACTION * np.nanmax(frame["envelope_mw"])),
    }


def save_clean_dataset(frame: pd.DataFrame, path: Path = config.CLEAN_FILE) -> None:
    out = frame[["hourly_total_mw", "envelope_mw", "in_daylight", "clearsky_index", "in_sample", "unusual"]].copy()
    out.index.name = "timestamp_local_standard_time"
    out.to_csv(path, float_format="%.6f")
