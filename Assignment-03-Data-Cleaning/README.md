# Assignment 3 - Data Cleaning & Preprocessing

**Tata Technologies TechPulse FY-26 · Applied AI/ML Course**

> Official objective: *"Handle missing values, outliers, and scale features
> using Pandas and Scikit-learn."*

A complete, reproducible cleaning-and-preprocessing demonstration on a real
Indian used-car dataset (CarDekho). Every number below is produced by
`python -m src.main` and stored in `artifacts/reports/` — nothing is
fabricated. This is a preprocessing assignment: no predictive model is built;
the target column is carried along only for context.

---

## Official Objective

"Handle missing values, outliers, and scale features using Pandas and
Scikit-learn."

## Problem Statement

Used-car listings are a classic messy ML source: values arrive as strings
with units ("23.4 kmpl", "1248 CC", "74 bhp"), listings are duplicated across
scrapes, optional fields are missing for older cars, and prices/odometer
readings span orders of magnitude. Feeding such data straight into a model
corrupts both training (numeric columns become categorical) and inference
(unscaled features dominate distances and gradients). This assignment walks
the full hygiene path — profile → clean types → domain checks → missing
values → outliers → scaling/encoding — inside leakage-free sklearn
pipelines, and quantifies every step before/after.

## Dataset

| Property | Value (actual) |
|---|---|
| Name | Car details v3 ("Vehicle dataset from CarDekho") |
| Source | Kaggle: [nehalbirla/vehicle-dataset-from-cardekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho); byte-identical public mirror: [imanishshahu/Car-Price-Prediction](https://raw.githubusercontent.com/imanishshahu/Car-Price-Prediction/main/Car%20details%20v3.csv) |
| Rows | **8,128** |
| Columns | **13** (`name, year, selling_price, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, torque, seats`) |
| Size on disk | ~1.0 MB (bundled at `data/car_data.csv`) |
| Feature types | 4 native numerics, 4 categoricals, 4 string-encoded numerics, 1 identifier |
| Target (context only) | `selling_price` (INR) — **not modeled** in this assignment |

`src/load_data.py` reads the local copy and falls back to the mirror URL if
the file is missing.

## Data Quality Issues

All issues below were measured on the raw file (see
`artifacts/reports/data_profile.json`):

1. **String-encoded numerics.** `mileage` ("23.4 kmpl", "17.3 km/kg" — two
   unit families), `engine` ("1248 CC"), `max_power` ("74 bhp") are stored as
   text; `torque` mixes **three** formats (`190Nm@ 2000rpm`,
   `22.4 kgm at 1750-2750rpm`, `12.7@ 2,700(kgm@ rpm)`).
2. **Missing values.** 1,100 raw cells: `mileage` 221, `engine` 221,
   `max_power` 215, `torque` 222, `seats` 221 (~2.7% each — the same old
   cars lack all specs).
3. **Exact duplicate rows.** After type parsing, **1,221 rows** (15.0% of the
   raw data) are exact duplicates — repeated listings.
4. **Domain anomalies.** A used car listed with `km_driven = 1`;
   **15 cars with 0 kmpl** fuel economy and **3 cars with 0 bhp** power
   (physically impossible; the "bhp 0" rows parse from malformed strings).
5. **Heavy-tailed magnitudes.** `selling_price` spans ₹29,999–₹1,00,00,000;
   `km_driven` spans 1–2,360,457 km; engine sizes 624–3,604 CC.
6. **Category cardinality.** `name` has 1,811 distinct values (an identifier,
   not a feature) but yields a manageable 32-value `brand` feature.

## Missing Value Handling

**Strategy selection (and why):**

* Numeric features → **median** imputation: robust to the extreme skew in
  `selling_price`/`km_driven` (a mean imputer would invent implausible cars).
* Categorical features → **most-frequent** imputation: no natural "Unknown"
  bucket exists for `fuel`/`transmission`, and the missing rate is ~3%.
* Rows were **not** dropped: 208–223 missing per column out of 6,907 rows is
  recoverable information, and dropping would bias against older cars.
* No column exceeded a drop-worthiness threshold (~40%), so none were
  dropped for missingness.
* Domain-impossible values (below) are converted to NaN **before**
  imputation, so they are repaired rather than propagated.

**Before/after table** (actual, from `artifacts/reports/missing_values_before_after.csv`;
"before" = after parsing/dedup/domain-flagging, "after" = after imputation):

| Column | Missing before | % before | Missing after | % after |
|---|---:|---:|---:|---:|
| km_driven | 1 | 0.014 | 0 | 0.0 |
| seats | 208 | 3.011 | 0 | 0.0 |
| mileage_kmpl | 223 | 3.229 | 0 | 0.0 |
| engine_cc | 208 | 3.011 | 0 | 0.0 |
| max_power_bhp | 209 | 3.026 | 0 | 0.0 |
| torque_nm | 209 | 3.026 | 0 | 0.0 |
| **Total (all columns)** | **1,058** | — | **0** | — |

## Domain Sanity Checks

Checks applied (each bound justified, not arbitrary):

* `km_driven < 100` on a *used*-car listing is a data-entry error (1 row,
  a 2011 Maruti Eeco listed with 1 km) → set to NaN and imputed.
* `mileage_kmpl == 0` (15 rows) and `max_power_bhp == 0` (3 rows) are
  physically impossible for a functioning car → NaN + imputed.
* Extreme-but-plausible values are **kept**: engine 3,604 CC (large SUVs),
  ₹1 cr for a 2019 Audi A6 test-drive unit, 236,0457 km trims — these are
  real listings, and capping/deleting them would fabricate a tidier but
  false market.
* No negative values exist anywhere in the numeric columns (verified in the
  profile: `negative_count = 0` for all numerics).

## Outlier Detection

**Method:** 1.5×IQR fences on each numeric feature —
`lower = Q1 − 1.5·IQR`, `upper = Q3 + 1.5·IQR`. Actual counts
(`artifacts/reports/outliers_report.csv`):

| Feature | Q1 | Q3 | Lower fence | Upper fence | Outliers | % affected |
|---|---:|---:|---:|---:|---:|---:|
| km_driven | 35,000 | 95,000 | −65,000 | 195,000 | 166 | 2.40% |
| selling_price | 245,000 | 630,000 | −310,000 | 1,185,000 | 327 | 4.73% |
| mileage_kmpl | 16.70 | 22.30 | 8.40 | 30.60 | 6 | 0.09% |
| max_power_bhp | 68.32 | 100.00 | 20.87 | 147.45 | 305 | 4.55% |
| engine_cc | 1197.0 | 1498.0 | 745.5 | 1949.5 | 1,203 | 17.96% |
| torque_nm | 145.15 | 200.0 | 62.7 | 282.4 | 306 | 4.57% |

**Strategy — capping, removal, or retention:**

* **Winsorized (capped to the IQR fences):** `km_driven`,
  `selling_price`, `mileage_kmpl`, `max_power_bhp` —
  **876 values changed in total** (km_driven 166, selling_price 327,
  mileage 12, max_power 371). These four have extreme *magnitudes* that
  would dominate any distance/gradient-based model; capping preserves the
  rows and their rank order while bounding influence. (Capped counts are
  computed after imputation, so they can slightly exceed the pre-imputation
  detection counts — e.g. mileage 12 vs 6.)
* **Retained (deliberately uncapped):** `engine_cc` (1,203 flags = 18%!) and
  `torque_nm` (306), plus `year` and `seats`. When nearly a fifth of values
  are "outliers", the IQR assumption (roughly symmetric bulk) is failing, not
  the cars — India's fleet genuinely clusters at 1.2–1.5 L with a long tail
  of larger engines. Capping would erase real, discriminative information.
* **Removed:** no rows were removed for outliers — only the 1,221 exact
  duplicate listings were dropped (duplicates are a data-integrity issue,
  not an outlier issue).

The capping thresholds are **learned from training data only** inside the
pipeline via the `IQRCapper` transformer (see below).

## Feature Scaling

Two scalers are compared on the cleaned data (statistics from
`artifacts/reports/scaling_comparison.csv`):

* **StandardScaler** — subtracts the mean, divides by the standard
  deviation. Appropriate when the feature is roughly symmetric and has no
  dominant extremes, because both statistics are sensitive to outliers.
* **RobustScaler** — subtracts the median, divides by the IQR. Both
  statistics are outlier-resistant, so extreme values cannot inflate the
  denominator; appropriate precisely when outliers **remain** in the data
  (e.g. the retained engine/torque tails) or for skewed features.

Actual before/after summary for the winsorized features:

| Feature | Scaling | mean | std | median | IQR |
|---|---|---:|---:|---:|---:|
| km_driven | raw | 72,214.32 | 43,417.71 | 70,000 | 60,000 |
| | Standard | 0.0000 | 1.0001 | −0.0510 | 1.3820 |
| | Robust | 0.0369 | 0.7236 | 0.0000 | 1.0000 |
| selling_price | raw | 470,609.02 | 294,894.26 | 400,000 | 384,500 |
| | Standard | 0.0000 | 1.0001 | −0.2395 | 1.3040 |
| | Robust | 0.1836 | 0.7670 | 0.0000 | 1.0000 |
| mileage_kmpl | raw | 19.5038 | 3.8708 | 19.49 | 5.32 |
| | Standard | 0.0000 | 1.0001 | −0.0036 | 1.3745 |
| | Robust | 0.0026 | 0.7276 | 0.0000 | 1.0000 |
| max_power_bhp | raw | 85.9244 | 26.0281 | 81.83 | 31.00 |
| | Standard | 0.0000 | 1.0001 | −0.1573 | 1.1911 |
| | Robust | 0.1321 | 0.8396 | 0.0000 | 1.0000 |

Reading the table: StandardScaler guarantees mean 0 / std 1 by construction;
RobustScaler guarantees median 0 / IQR 1. On this skewed data the Robust
versions keep a non-zero mean (e.g. 0.18 for price) — that residual skew is
real signal, not noise. The `scaling_comparison.png` figure shows the three
distributions side by side per feature.

## Categorical Encoding

* Nominal features (`fuel`, `seller_type`, `transmission`, `owner`,
  `brand`) are encoded with
  **`OneHotEncoder(handle_unknown="ignore")`** — no arbitrary integer
  ranking is imposed on unordered categories, and unseen categories at
  inference time map to an all-zero row instead of crashing.
* `owner` is nominally ordered ("First" < "Second" < …), but the small
  cardinality (5, including `Test Drive Car`, which is not ordered) makes
  one-hot the safer, simpler choice; the assignment brief prefers one-hot
  unless a clear reason exists for integer labels.
* The encoder is fit on training data only; `handle_unknown="ignore"` is
  verified by a dedicated unit test and by the pipeline demo.

## Preprocessing Pipeline

`src/preprocessing.py` builds a sklearn `ColumnTransformer` with two
branches:

```
numeric    (year, km_driven, mileage_kmpl, engine_cc, max_power_bhp,
            torque_nm, seats):
    SimpleImputer(median) -> IQRCapper(k=1.5) -> StandardScaler | RobustScaler

categorical (fuel, seller_type, transmission, owner, brand):
    SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown="ignore")
```

**Leakage prevention is structural, not procedural.** Every learned statistic
— imputer medians/modes, IQR fences, scaler mean/std, one-hot vocabulary — is
computed inside `fit()` only. The runner fits the transformer **on the
training split** and merely calls `transform()` on the test split. The custom
`IQRCapper` follows the same contract (`fit` learns fences; `transform` only
applies them).

**Leakage verification** (`artifacts/reports/leakage_verification.json`,
25% holdout, seed 42 — all checks pass):

| Check | Result |
|---|---|
| Imputer medians equal **train-only** medians (e.g. km_driven 70,000; max_power 81.83) | ✅ True |
| Scaler means equal the mean of **capped train** data (e.g. km_driven 72,043.12) | ✅ True |
| Scaler statistics **unchanged after transforming** the test set | ✅ True |

The learned medians match the train-only medians exactly; if the test rows
had been included, the pooled medians/means would differ (they are stored in
the report for comparison).

## Results

Actual end-to-end numbers from `python -m src.main`
(`artifacts/reports/cleaning_report.json`):

| Metric | Before | After |
|---|---:|---:|
| Rows | 8,128 | 6,907 (−1,221 exact duplicates) |
| Columns (features context) | 13 raw | 12 cleaned (−`name`, −`torque` raw; +`brand`) |
| Total missing cells | 1,058 | **0** |
| Values winsorized | — | 876 (of 2,308 flagged across 6 features) |
| Processed feature matrix | — | 6,907 × 54 (7 scaled numerics + 47 one-hot columns) |

The processed matrix is saved to `artifacts/processed_data.csv` (with the raw
`selling_price` kept as a reference column) and is bit-for-bit reproducible —
the end-to-end test re-runs the whole pipeline and asserts an identical CSV.

## Visualizations

All figures live in `artifacts/plots/`:

| Figure | File |
|---|---|
| Missing values BEFORE cleaning | `missing_values_before.png` |
| Missing values AFTER cleaning (empty chart = zero) | `missing_values_after.png` |
| Boxplots before/after IQR winsorization (4 features) | `outliers_before_after.png` |
| Raw vs StandardScaler vs RobustScaler distributions (4 features × 3 panels) | `scaling_comparison.png` |

Each figure has titled, labeled axes; the boxplot pairs share an x-scale for
an honest before/after comparison.

## Testing

`python -m pytest tests -v` — **48 tests, all passing** (~9 s), covering:

* **Loading/schema (4):** file loads, 13 expected columns, row count > 5,000.
* **Cleaning (20):** string→numeric parsing (all unit suffixes), torque Nm/kgm
  formats, brand extraction, duplicate removal, domain checks
  (impossible km/zero mileage/zero power → NaN), IQR bounds math, outlier
  detection on synthetic + real data, capper clipping + train-only fitting,
  median/mode imputation, before/after report table.
* **Preprocessing (12):** numeric pipeline imputation, StandardScaler zero-mean/unit-std,
  RobustScaler zero-median, invalid scaler rejection, capper-inside-pipeline
  bounding, most-frequent categorical imputation, one-hot shape,
  **unknown-category handling**, full ColumnTransformer fit/transform,
  feature-name counts, **no NaN after preprocessing even with injected
  NaNs**, leakage checks (fit-on-train-only; transform does not mutate
  fitted state), scaler comparison structure.
* **End-to-end (9):** `main()` runs on the real data, every report/plot is
  written, processed CSV has zero NaNs, 1,221 duplicates removed, missing
  totals 1,058 → 0, leakage checks pass, profile JSON contents, and a
  full **reproducibility** assertion (second run → identical CSV).

## How to Run

```bash
cd Assignment-03-Data-Cleaning

python -m pip install -r requirements.txt

# full pipeline: profile -> clean -> impute -> outliers -> scalers -> reports
python -m src.main

# run the test suite
python -m pytest tests -v
```

No Jupyter is required; everything runs headless and writes to `artifacts/`.
An internet connection is needed only if `data/car_data.csv` is missing.

## Project Structure

```
Assignment-03-Data-Cleaning/
├── README.md
├── requirements.txt
├── conftest.py                        # pytest import bootstrap
├── pytest.ini                         # test configuration
├── data/
│   ├── README.md                      # dataset provenance
│   └── car_data.csv                   # CarDekho v3 (8,128 x 13, ~1 MB)
├── src/
│   ├── __init__.py
│   ├── config.py                      # paths, schema, domain rules, knobs
│   ├── load_data.py                   # local CSV + URL fallback
│   ├── profiling.py                   # data_profile.json + summaries + plot
│   ├── cleaning.py                    # parsers, domain checks, IQR, IQRCapper
│   ├── preprocessing.py               # ColumnTransformer, scalers, leakage check
│   └── main.py                        # orchestration (python -m src.main)
├── tests/
│   ├── __init__.py
│   ├── test_cleaning.py
│   ├── test_preprocessing.py
│   └── test_pipeline.py
└── artifacts/
    ├── processed_data.csv             # final matrix (6,907 x 54)
    ├── reports/
    │   ├── data_profile.json
    │   ├── cleaning_report.json
    │   ├── leakage_verification.json
    │   ├── missing_values_before_after.csv
    │   ├── outliers_report.csv
    │   ├── scaling_comparison.csv
    │   ├── profile_numeric_summary.csv
    │   └── profile_categorical_summary.csv
    └── plots/
        ├── missing_values_before.png
        ├── missing_values_after.png
        ├── outliers_before_after.png
        └── scaling_comparison.png
```

## Conclusion

Cleaning this real CarDekho extract changed the dataset far more than a
casual glance suggests: 15% of rows were exact duplicates, five columns were
numeric values wearing string costumes, ~3% of specification cells were
missing, and 2,308 values sat outside the 1.5×IQR fences. The main lessons:

1. **Profile first, transform second** — the profiling pass discovered the
   kgm/Nm dual-unit torque problem and the zero-bhp rows that no generic
   pipeline would have caught.
2. **Domain knowledge beats blanket rules** — a used car with 1 km is wrong;
   an engine with 3,604 CC is just a big SUV. Treating both the same would
   either keep garbage or delete truth, so the pipeline *repairs* the former
   and *retains* the latter.
3. **Outlier handling is a decision, not a default** — capping the four
   magnitude features (876 values) while keeping the 1,203 flagged engine
   sizes is defensible precisely because the reasoning is written down and
   reproducible via the fitted `IQRCapper`.
4. **RobustScaler earns its keep on skewed data** — after scaling, the
   Standard versions force std = 1 while the Robust versions keep IQR = 1
   and leave the residual skew visible instead of amplifying tail influence.
5. **Leakage prevention belongs in the structure** — because every statistic
   is learned in `fit()` on training data and the verification report proves
   medians/means match train-only values, the same transformer is safe to
   reuse at inference time.

## References

* Dataset: Kaggle — *Vehicle dataset from CarDekho* (nehalbirla), "Car
  details v3.csv".
* scikit-learn documentation: `ColumnTransformer`, `Pipeline`,
  `SimpleImputer`, `OneHotEncoder`, `StandardScaler`, `RobustScaler`.
