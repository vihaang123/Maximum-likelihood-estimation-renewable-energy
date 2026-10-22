# Dataset metadata

## Identification
| Item | Value |
|---|---|
| Dataset | Solar PV power output of 137 plants in Alabama, calendar year 2006 ("solar-energy" dataset) |
| Origin | U.S. National Renewable Energy Laboratory (NREL), *Solar Power Data for Integration Studies*; project page: https://www.nrel.gov/grid/solar-power-data.html |
| Copy used | Unmodified file `solar_AL.txt.gz` from the public LSTNet repository: https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz |
| Associated paper (source of the redistributed copy) | Lai, G., Chang, W.-C., Yang, Y., Liu, H. (2018). Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks. SIGIR '18, 95-104. |
| Download date (UTC) | 2026-10-19 |
| File size / SHA-256 | 8748993 bytes / `1ae9329c3e1032acbb8dd080bfc397c44507006812a715d2788f172820489a1c` |
| Local copy | `data/solar_AL.txt.gz` (preserved byte-for-byte; the analysis reads it directly) |
| Cleaned/derived data | `data/cleaned_dataset.csv` (hourly fleet total, envelope, clear-sky index, flags) |

## Variable, units, frequency, scope
* **Variable:** PV plant power output. One column per plant (137 columns), no header, comma separated, gzip-compressed.
* **Units:** MW (NREL convention; the redistributed file carries no unit header).
* **Native frequency:** 10 minutes, 52,560 rows = 365 days x 144 steps (a non-leap year).
* **Geographic scope:** 137 PV plants in the U.S. state of Alabama.
* **Period:** 2006 (first row = 2006-01-01 00:00, last row = 2006-12-31 23:50:00).
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
10-minute grid (365 x 144 = 52,560 rows, exactly what the file contains). Evidence that this assumption is
physically consistent (computed by the code from the data):
* Peak of the mean diurnal profile at **11:40** local time (solar noon in Alabama is about 12:00-12:15 CST).
* Mean number of hours per day with positive output: **10.1 h in December** and
  **13.9 h in June** (astronomical day length at ~32-33 deg N is about 10 h and 14 h).
* Monthly mean output peaks in late spring/summer and is lowest in winter.

## Integrity checks (from `results/data_validation.json`)
| Check | Result |
|---|---|
| Rows x plants | 52,560 x 137 (expected 52,560 x 137) |
| Missing / non-finite values | 0 / 0 |
| Negative values | 0 |
| Duplicate timestamps | 0 |
| Distinct time steps | 1 (10 minutes) |
| Plants that are always zero | 0 |
| Largest single-plant value | 88.9 MW |
| Hours with exactly zero fleet output | 4,076 of 8,760 (46.5 %) |

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
   clear-sky index. Hours whose envelope is below 10 % of its annual maximum are outside the daylight window.
5. Unusual values (index > 1.20 or < 0.05) are flagged in `cleaned_dataset.csv` but never deleted.
