"""Train/test split used in the forecasting experiment: summary table and figure.

Run from the project root:  python -m src.split_summary
The split is chronological (first 75% of days = training, last 25% = test); it is used ONLY for the
one-hour-ahead forecasting experiment.  Distribution fitting and ARMA/GARCH model comparison use all
n = 3,846 daylight observations (in-sample), as described in the report.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config

from . import preprocessing as pp

matrix = pp.load_raw_matrix()
hourly = pp.to_hourly(pp.aggregate_fleet(matrix, pp.build_timestamps(len(matrix))))
frame = pp.build_analysis_frame(hourly, causal=True)          # the causal-envelope index used by the forecast experiment
fc = json.loads((config.RESULTS_DIR / "key_results.json").read_text())["forecast"]
cut = pd.Timestamp(fc["test_start"])
day = frame[frame.clearsky_index.notna()]
tr, te = day[day.index < cut], day[day.index >= cut]
n_scored = fc["n_test"]
summary = pd.DataFrame([
    ("Training", frame.index[0].date(), (cut - pd.Timedelta(days=1)).date(), fc["train_days"], len(tr), "order selection (BIC) and parameter estimation"),
    ("Test (hold-out)", cut.date(), frame.index[-1].date(), fc["test_days"], len(te), f"one-hour-ahead scoring ({n_scored} hours with an observed previous hour)"),
], columns=["set", "start", "end", "days", "daylight_observations", "used_for"])
summary.to_csv(config.RESULTS_DIR / "train_test_split.csv", index=False)
print(summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(tr.index, tr.clearsky_index, ".", ms=2, color="#1f3864", label=f"Training: {fc['train_days']} days, {len(tr):,} daylight hours")
ax.plot(te.index, te.clearsky_index, ".", ms=2, color="#c0392b", label=f"Test: {fc['test_days']} days, {len(te):,} daylight hours ({n_scored} scored)")
ax.axvline(cut, color="k", ls="--", lw=1)
ax.text(cut, ax.get_ylim()[1] * 0.97, "  split: 2006-10-02 00:00", va="top", fontsize=8)
ax.set_ylabel("Clear-sky index"); ax.set_title("Chronological train/test split (75% / 25% of days)")
ax.legend(loc="lower left", fontsize=8, frameon=False); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(config.FIG_DIR / "fig15_train_test_split.png", dpi=160)

# ---- export the two sets (hourly rows; causal-envelope index used by the forecast experiment) ----
out = frame.copy(); out.index.name = "timestamp_local_standard_time"
out.loc[out.index < cut].to_csv(config.DATA_DIR / "train_dataset.csv", float_format="%.6f")
out.loc[out.index >= cut].to_csv(config.DATA_DIR / "test_dataset.csv", float_format="%.6f")
print("wrote data/train_dataset.csv and data/test_dataset.csv")
