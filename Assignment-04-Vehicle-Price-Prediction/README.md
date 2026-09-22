# Assignment 4 - Vehicle Price Prediction

## Official Objective

Official assignment: **"Vehicle Price Prediction"** — *"Use regression models to
estimate vehicle prices from structured data."*

This project builds a complete, reproducible regression system that predicts
used-vehicle selling prices (INR) from the structured CarDekho dataset, with
automotive feature engineering, leakage-free sklearn preprocessing, four
compared regression models, and a saved best model with a prediction CLI.

## Problem Statement

A used vehicle's market price depends on many structured attributes: age,
kilometres driven, engine size, power, fuel type, transmission, ownership
history and brand. Estimating that price from structured data is a classic
regression problem with real-world messiness: unit-suffixed strings
("23.4 kmpl"), mixed torque formats, missing values, duplicated listings and
strongly right-skewed targets. This project demonstrates a defensible
end-to-end pipeline: clean the data, engineer meaningful automotive features,
fit several regression models without leakage, compare them honestly on a
held-out test set, and deploy the winner behind a simple prediction script.

## Dataset

**CarDekho "Car details v3"** used-car dataset.

| Property | Value |
|---|---|
| Source | Kaggle: [Vehicle dataset (from CarDekho)](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) by nehalbirla |
| Local copy | `data/car_data.csv` (byte-identical, ~1 MB; auto-download fallback documented in `data/README.md` and implemented in `src/data_loader.py`) |
| Rows | 8,128 |
| Columns | 13 |
| Target | `selling_price` (INR) |

Columns: `name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`,
`transmission`, `owner`, `mileage`, `engine`, `max_power`, `torque`, `seats`.

Raw target statistics (measured): min ₹29,999 · median ₹400,000 ·
max ₹1,00,00,000 · skew **5.57** (strongly right-skewed).

## Data Cleaning

Measured issues found and handled in `src/data_loader.py` /
`src/feature_engineering.py`:

| Issue | Measured extent | Handling |
|---|---|---|
| Unit-suffixed numeric strings | `mileage` ("23.4 kmpl"), `engine` ("1248 CC"), `max_power` ("74 bhp") | Parsed to `mileage_kmpl`, `engine_cc`, `max_power_bhp` |
| Three torque formats | `190Nm@ 2000rpm`, `22.4 kgm @ ...rpm`, `12.7@ 2,700(kgm@ rpm)` | Parsed to a single `torque_nm` column; kgm converted to Nm (× 9.80665) |
| Exact duplicate rows | **1,221 rows** | Dropped (after parsing; parsing reveals extra exact matches) → **6,907 rows** |
| Missing values in parsed numerics | 880 cells | Never imputed before the split; imputed **inside the pipeline** with the training-split median |
| Physically impossible values | 1 car with km_driven < 100, 15 cars with 0 kmpl, 3 cars with 0 bhp (**19 values**) | Set to NaN (treated as data-entry errors) and repaired by in-pipeline median imputation |

No rows or values were fabricated. Every transformation above is row-wise and
target-independent.

## Feature Engineering

All engineered features are derived **without ever reading `selling_price`**
(enforced by tests):

| Feature | Derivation |
|---|---|
| `vehicle_age` | `REFERENCE_YEAR − year`, with a fixed `REFERENCE_YEAR = 2020` (the newest model year in the dataset, so every age is ≥ 0 and results are reproducible) |
| `km_per_year` | `km_driven / max(vehicle_age, 1)` — usage intensity |
| `brand` | First token of `name` (e.g. "Maruti Swift VDI" → "Maruti"). Decision: the full `name` has ~1,900 near-unique values (unlearnable), while brand keeps meaningful price signal. Raw `name` and `year` are dropped after derivation. |
| `mileage_kmpl`, `engine_cc`, `max_power_bhp`, `torque_nm` | Parsed from the unit strings above |

Final feature set (13): numeric — `vehicle_age`, `km_driven`,
`mileage_kmpl`, `engine_cc`, `max_power_bhp`, `torque_nm`, `seats`,
`km_per_year`; categorical — `fuel`, `seller_type`, `transmission`, `owner`,
`brand`.

## Data Preprocessing

A single `ColumnTransformer` per model, wrapped in a sklearn `Pipeline`:

- **Numeric**: `SimpleImputer(median)` → `StandardScaler`
- **Categorical**: `SimpleImputer(most_frequent)` →
  `OneHotEncoder(handle_unknown="ignore")`

**Leakage prevention is structural and tested:**

1. All statistics-learning steps live *inside* the pipeline and are fitted
   only via `model.fit(X_train, y_train)`.
2. The feature matrix is built by explicit column selection
   (`X = df[config.FEATURES]`), so the target can never enter the inputs.
3. Engineered features are row-wise and target-independent (no means/medians
   computed pre-split).
4. Tests verify that the target is absent from the preprocessor's column
   lists, and that scaler/imputer statistics equal the training-only values
   and do not move when the test set is transformed.

## Target Transformation

The price distribution is heavily right-skewed (skew 5.57), so **every model
is trained on `log1p(selling_price)`** using sklearn's
`TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`:

- `fit` sees `log1p(price)`;
- `predict(X)` applies `expm1` internally and **already returns prices in
  original rupee units**.

All reported MAE/RMSE/R² are therefore computed in **rupee space**, never log
space. (The log1p evaluation was chosen over the raw target because the raw
target's heavy tail is dominated by a handful of luxury listings; with the
transform the linear models improve substantially and residuals are far more
homoscedastic.)

## Models

Four sklearn-native regressors (all sharing the same preprocessor; seed 42):

| Model | Hyperparameters |
|---|---|
| Linear Regression | defaults |
| Ridge Regression | `alpha=1.0` |
| Random Forest | `n_estimators=300`, `n_jobs=-1` |
| Gradient Boosting | `n_estimators=400`, `learning_rate=0.05`, `max_depth=3` |

XGBoost was deliberately not used — the sklearn models already perform well
and keep the project dependency-light and reproducible.

## Evaluation Metrics

- **MAE** — mean absolute error, in rupees; robust to the heavy price tail.
- **RMSE** — root mean squared error, in rupees; punishes large errors.
- **R²** — fraction of price variance explained.
- **CV R²** — 5-fold shuffled cross-validation on the *training* set
  (`random_state=42`); reported as mean ± std.

## Results

80/20 train/test split (`random_state=42`) → 5,525 train / 1,382 test rows.
All metrics in **original INR units**, measured on the held-out test set:

| Model | Test MAE | Test RMSE | Test R² | CV R² |
|---|---:|---:|---:|---:|
| Linear Regression | 94,202 | 177,124 | 0.8582 | 0.8685 ± 0.0171 |
| Ridge Regression | 94,414 | 178,577 | 0.8559 | 0.8675 ± 0.0159 |
| **Random Forest** | **71,951** | **124,492** | **0.9300** | 0.8910 ± 0.0222 |
| Gradient Boosting | 73,488 | 127,888 | 0.9261 | **0.8967 ± 0.0283** |

(Reproduce with `python -m src.train`; values are deterministic for seed 42.)

## Best Model

**Random Forest** was selected by the documented criterion — **lowest test
RMSE** (₹124,492). It also has the best test MAE (₹71,951) and test R²
(0.9300). Gradient Boosting edges it on CV R² (0.8967 vs 0.8910), but the
test-set criterion was fixed *before* inspecting which model won, and the two
are within one cross-validation standard deviation of each other.

Saved artifacts:

- `models/best_vehicle_price_model.joblib` — full fitted pipeline
  (preprocessing + regressor + log1p/expm1 wrapper)
- `models/best_model_metadata.json` — selected model, feature list, target,
  target transform, seed, reference year
- `artifacts/metrics.json`, `artifacts/model_comparison.csv`

## Error Analysis

Mean absolute error by actual price range (test set):

| Actual price range | Mean abs. error |
|---|---:|
| ₹0.35–1.9 lakh | ₹32,866 |
| ₹1.9–3.0 lakh | ₹40,300 |
| ₹3.0–4.2 lakh | ₹54,855 |
| ₹4.2–5.5 lakh | ₹68,770 |
| ₹5.5–7.4 lakh | ₹72,973 |
| ₹7.4–58.3 lakh | ₹163,304 |

Errors grow with price — the model is most accurate in the dense budget/mid
segment and weakest for expensive cars. The eight largest test errors are all
premium/luxury vehicles (e.g. a 1-year-old **Volvo** listed at ₹55,00,000
predicted at ₹40,93,163; a **Lexus** ₹51,50,000 vs ₹39,60,984; several
Mercedes-Benz, Audi and BMW listings), reflecting both their rarity in the
training data and the wide price spread inside the luxury brands. See
`artifacts/largest_errors.csv` and `artifacts/plots/error_by_price_range.png`.

## Feature Importance

Permutation importance on the test set (increase in MAE, in rupees, when the
column is shuffled; reported per original engineered feature):

| Feature | Importance (ΔMAE) |
|---|---:|
| vehicle_age | 154,324 |
| max_power_bhp | 116,664 |
| torque_nm | 37,692 |
| brand | 13,990 |
| engine_cc | 10,110 |
| km_driven | 6,651 |
| mileage_kmpl | 6,523 |
| transmission | 2,144 |
| owner | 1,045 |
| seats | 610 |
| fuel | 411 |
| seller_type | 279 |
| km_per_year | −970 (noise) |

Age and power dominate price, followed by torque and brand. This matches
automotive intuition, but these are **model attributions, not causal
effects** — importance reflects how the trained forest uses each feature on
this dataset.

## Visualizations

Generated under `artifacts/plots/`:

| File | Content |
|---|---|
| `price_distribution.png` | Raw vs `log1p` selling-price distributions |
| `actual_vs_predicted.png` | Test-set actual vs predicted price with the ideal line |
| `residuals.png` | Residuals vs predicted price |
| `model_comparison.png` | MAE / RMSE / R² / CV R² bars for all four models |
| `feature_importance.png` | Permutation importance of the final model |
| `error_by_price_range.png` | Mean absolute error by actual price bucket |

## Testing

`python -m pytest tests -v` → **35 passed** in ~4 s.

Coverage: dataset loads with 8,128 rows and exact duplicate removal (1,202
raw duplicates); the target exists, is positive, and is absent from
`config.FEATURES`; every unit parser (mileage/engine/bhp/3 torque formats);
vehicle age, km/year, brand; numeric and categorical pipelines; imputation,
scaling, one-hot expansion; unknown categories do not crash; no NaN in
transformed output; imputer/scaler statistics fit on train only and frozen at
transform time; all four models fit/predict with finite, positive,
rupee-space outputs; end-to-end training on a subset; model save/reload
parity; both prediction-CLI modes.

## How to Run

```bash
python -m pip install -r requirements.txt
python -m src.train      # full training -> artifacts/ + models/
python -m src.predict    # demo predictions with the saved model
python -m src.predict --csv your_cars.csv   # batch mode (raw CarDekho-style columns)
python -m pytest tests -v
```

Demo output from the actual saved model (no hardcoded values):

```
2019 Maruti Swift VDI        -> Predicted vehicle price: Rs.675,187
2015 Hyundai i10 Magna       -> Predicted vehicle price: Rs.273,513
2012 Toyota Innova 2.5 GX    -> Predicted vehicle price: Rs.458,708
```

## Project Structure

```
Assignment-04-Vehicle-Price-Prediction/
├── README.md
├── requirements.txt
├── pytest.ini
├── conftest.py
├── data/
│   ├── README.md            # dataset provenance
│   └── car_data.csv         # CarDekho Car details v3 (local copy)
├── src/
│   ├── __init__.py
│   ├── config.py            # paths, seed 42, reference year 2020, schema
│   ├── data_loader.py       # load, dedup (1,221), domain checks (19 fixes)
│   ├── feature_engineering.py  # unit parsing, vehicle_age, brand, km_per_year
│   ├── preprocessing.py     # ColumnTransformer (impute/scale + impute/OHE)
│   ├── models.py            # 4 models in pipelines + log1p TTR wrapper
│   ├── evaluation.py        # rupee-space metrics, CV, permutation importance,
│   │                        #   all plots, error analysis
│   ├── train.py             # end-to-end orchestration
│   └── predict.py           # demo + CSV prediction CLI
├── tests/                   # 35 tests across 5 files
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_pipeline.py
├── models/
│   ├── best_vehicle_price_model.joblib
│   └── best_model_metadata.json
└── artifacts/
    ├── metrics.json
    ├── model_comparison.csv
    ├── feature_importance.csv
    ├── largest_errors.csv
    └── plots/               # 6 figures
```

## Conclusion

On the real CarDekho data (6,907 cleaned rows), a Random Forest regressor
estimates used-vehicle prices with a **test MAE of ₹71,951** and **R² of
0.9300** — a median-relative error of about 18% of the ₹400,000 median price,
with the tree ensembles (Random Forest, Gradient Boosting) clearly ahead of
the linear models. Evolution across the pipeline mattered: parsing the messy
unit strings, converting torque to a consistent unit, deriving age/usage
features, and training on the log1p target were each necessary to reach these
numbers honestly, and every learned statistic was fitted on the training
split only. The remaining weakness is systematic and visible in the error
analysis: rare, expensive luxury vehicles. This is a simplified, educational
regression exercise on historical listing data — not a production valuation
system, and feature importance here describes model behaviour, not causal
price effects.
