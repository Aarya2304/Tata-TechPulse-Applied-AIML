# Assignment 01 — ML Model for Car Mileage Estimation

**Tata Technologies TechPulse FY-26 · Applied AI/ML Course**

> Predict car mileage (MPG) using regression and Python libraries, on the classic
> [UCI Auto MPG dataset](https://archive.ics.uci.edu/dataset/9/auto+mpg).

The trained pipeline predicts a car's fuel economy in miles per gallon from seven
specifications (cylinders, displacement, horsepower, weight, acceleration, model
year, origin). Three regression models are trained and compared; the best one
(**Random Forest**) is saved with `joblib` and served by a prediction CLI.

---

## 1. Results Summary (real, reproduced runs)

All numbers below are actual outputs of `python -m src.train` in this environment
(Python 3.11.9, scikit-learn 1.3.0, seed = 42 everywhere). Rerunning the pipeline
reproduces them exactly.

### Hold-out test set — 80 cars never touched during training

| Model              |   MAE  |  RMSE  |   R²   |
|--------------------|--------|--------|--------|
| Linear Regression  | 2.424  | 3.217  | 0.826  |
| Ridge Regression   | 2.429  | 3.224  | 0.825  |
| **Random Forest**  | **1.941** | **2.842** | **0.864** |

### 5-fold cross-validation (on the 318-row training set, mean ± std)

| Model              | CV MAE        | CV RMSE       | CV R²           |
|--------------------|---------------|---------------|-----------------|
| Linear Regression  | 2.631 ± 0.181 | 3.410 ± 0.361 | 0.803 ± 0.033   |
| Ridge Regression   | 2.621 ± 0.174 | 3.408 ± 0.362 | 0.803 ± 0.033   |
| **Random Forest**  | **1.945 ± 0.173** | **2.717 ± 0.509** | **0.875 ± 0.036** |

**Winner: Random Forest Regressor** (300 trees). It is the best model on every
metric, in both CV and on the held-out test set. Saved to
`artifacts/best_model.joblib` (refit on the full 398-row dataset for deployment).

Interpretation: on unseen cars the model is typically within **~1.9 mpg** of the
true value (MAE), explains **~86%** of mileage variance (R²), and clearly beats
the naive baseline of always predicting the mean (R² = 0).

### Example predictions (from `python predict.py`)

| Car | Predicted MPG |
|---|---|
| 1970 Chevrolet Chevelle Malibu (V8, 307 ci, 3504 lb) — actual: 18.0 | **17.34** |
| Typical 4-cyl compact, 1978, European | **30.76** |
| Fuel-sipping 4-cyl, 1982, Japanese | **36.52** |

---

## 2. Dataset

* **Source:** UCI Machine Learning Repository — *Auto MPG* (classic 398-row version).
* **Rows:** 398 cars (model years 1970–1982, origins: 1 = USA, 2 = Europe, 3 = Japan).
* **Target:** `mpg` (continuous) — mean 23.51, median 23.0, range 9.0–46.6.
* **Features (7):** `cylinders`, `displacement`, `horsepower`, `weight`,
  `acceleration`, `model_year`, `origin`. `car_name` is kept only for reporting
  and is never fed to the models.
* The raw file is bundled at `data/auto-mpg.data`, so everything runs offline.

### EDA highlights (real numbers)

Pearson correlation of each feature with `mpg`:

| weight | displacement | horsepower | cylinders | acceleration | origin | model_year |
|--------|--------------|------------|-----------|--------------|--------|------------|
| −0.832 | −0.804 | −0.778 | −0.775 | +0.420 | +0.563 | +0.579 |

* **Weight is the strongest single predictor** (r = −0.832): heavier cars burn
  more fuel. Engine size (displacement, horsepower, cylinders) follows closely.
* **Model year correlates positively** (+0.579) — the fleet became steadily more
  efficient through the 1970s (post oil-crisis emissions/CAFE era).
* **Origin matters** (+0.563): Japanese and European cars average far better
  mileage than the heavy US models of that era.
* `acceleration` is only weakly related (+0.420) — it mostly proxies for engine
  power in this dataset.
* Strong multicollinearity among the engine-size features (pairwise |r| ≈ 0.87–0.95)
  — which is why Ridge (regularised) is included alongside plain Linear Regression.

EDA plots are written to `artifacts/plots/`:

| Plot | File |
|---|---|
| Feature & target distributions | `eda_distributions.png` |
| Correlation heatmap | `eda_correlation_heatmap.png` |
| mpg vs key features (coloured by origin) | `eda_scatter_vs_mpg.png` |
| mpg by cylinders / origin | `eda_mpg_by_category.png` |

---

## 3. Data Cleaning & Missing-Value Handling

* File parsed as whitespace-separated columns; the UCI missing marker `"?"`
  is converted to `NaN` on load.
* **Exactly 6 rows** have missing `horsepower`: *ford pinto (1971), ford maverick
  (1974), renault lecar deluxe (1980), ford mustang cobra (1980), renault 18i
  (1981), amc concord dl (1982)*.
* The UCI documentation suggests dropping these rows, but imputing is more
  informative and keeps 398 rows. Missing horsepower is filled with the
  **median horsepower of cars with the same cylinder count**
  (e.g. 4-cyl → 78 hp, 6-cyl → 100 hp, 8-cyl → 150 hp), with the overall
  training median as fallback for unseen cylinder counts.
* Exact duplicates are dropped; rows with a missing target would also be
  dropped (none exist here); ranges sanity-checked (mpg ∈ (0, 100], etc.).

### Why the imputation cannot leak data

The imputer is implemented as a scikit-learn transformer (`HorsepowerImputer`)
that **lives inside every model `Pipeline`**. It therefore:

* learns group medians **only from the training split**, never from test data;
* is re-fit on the training folds of every cross-validation split automatically.

A dedicated unit test (`test_no_leakage_train_stats_only`) poisons the test split
and asserts the learned statistics do not change. The same logic covers the
Ridge pipeline's `StandardScaler` (fit on training data only).

### Train/test split without data leakage

* 80/20 split with `StratifiedShuffleSplit` on quantile-binned MPG
  (5 buckets, `random_state=42`), so both splits span the full 9–47 mpg range
  even though MPG is skewed.
* **318 train / 80 test rows.** The test set is used exactly once — for final
  evaluation — and never for imputation, scaling or model selection decisions.

---

## 4. Models

| Model | Pipeline |
|---|---|
| Linear Regression | `HorsepowerImputer → LinearRegression` |
| Ridge Regression | `HorsepowerImputer → StandardScaler → Ridge(alpha=1.0)` |
| Random Forest | `HorsepowerImputer → RandomForestRegressor(n_estimators=300)` |

* **Linear Regression** — interpretable baseline.
* **Ridge** — L2-regularised linear model; the dataset's engine features are
  highly collinear, so regularisation is expected to help (here it roughly ties
  plain OLS).
* **Random Forest** — nonlinear, captures interactions (e.g. weight × year);
  clear winner on every metric.

Each model is a full `sklearn.pipeline.Pipeline`, so preprocessing is bundled
with the estimator and exported as a single picklable object.

### Metrics

* **MAE** (mean absolute error, mpg) — average miss, robust to outliers.
* **RMSE** (root mean squared error, mpg) — penalises large misses.
* **R²** — fraction of MPG variance explained.

Cross-validation: 5-fold, shuffled, `random_state=42`, on the training set only.
Model selection is based on cross-validated performance; the test set provides
the final unbiased check.

---

## 5. Project Structure

```
Assignment-01-Car-Mileage/
├── data/
│   └── auto-mpg.data              # bundled UCI dataset (398 rows)
├── src/
│   ├── config.py                  # paths, constants, reproducibility
│   ├── data_pipeline.py           # load, clean, report + HorsepowerImputer
│   ├── eda.py                     # summaries + 4 EDA plots
│   ├── metrics_utils.py           # MAE/RMSE/R² + CV helpers
│   └── train.py                   # training pipeline & artifact export
├── tests/
│   ├── test_pipeline.py           # data, imputer, split, models, metrics, E2E
│   └── test_predict.py            # prediction CLI unit tests
├── artifacts/                     # generated by training
│   ├── best_model.joblib          # best pipeline (Random Forest), full refit
│   ├── metrics.json               # all CV + test metrics
│   ├── model_comparison.csv       # tidy comparison table
│   └── plots/                     # EDA + evaluation plots (7 PNGs)
├── predict.py                     # prediction CLI
├── conftest.py                    # pytest path bootstrap
├── pytest.ini                     # test config
├── requirements.txt
└── README.md
```

---

## 6. How to Run

**Environment:** Python 3.11 (tested on 3.11.9, Windows).

```bash
# 1) create & activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2) install dependencies
pip install -r requirements.txt

# 3) run the full pipeline: clean -> EDA -> split -> CV -> test eval -> save
python -m src.train

# 4) predict mileage
python predict.py                                  # built-in demo cars
python predict.py --specs cylinders=4 displacement=120 horsepower=88 \
    weight=2130 acceleration=14.5 model_year=82 origin=3

# 5) run the automated test suite (28 tests)
python -m pytest tests -v
```

### Using the saved model from Python

```python
import joblib
import pandas as pd

model = joblib.load("artifacts/best_model.joblib")
car = pd.DataFrame([{
    "cylinders": 4, "displacement": 120.0, "horsepower": 88.0,
    "weight": 2130.0, "acceleration": 14.5, "model_year": 82, "origin": 3,
}])
print(model.predict(car))   # -> [36.52] mpg
```

---

## 7. Automated Tests (28 passing)

`python -m pytest tests -v` — 28 passed, covering:

* **Data:** raw shape (398 × 9), the 6 missing-horsepower rows, dtypes,
  ranges, duplicate handling, missing-value report.
* **Imputer:** group medians are learned, all gaps filled, fallback medians,
  works inside a sklearn pipeline, and a **no-leakage test** that poisons the
  test split and asserts learned stats are unchanged.
* **Split:** 318/80 sizes, zero train/test overlap, exact reproducibility
  under a fixed seed, matched MPG ranges.
* **Models:** all three pipelines fit & predict finite values, metric
  sanity bands (e.g. R² > 0.70), CV returns MAE/RMSE/R² with std.
* **End-to-end:** every model beats the mean-predictor; best R² > 0.80.
* **Predict CLI:** spec parsing, error handling, prediction range checks.

> Note for this machine: the global Python environment ships a broken
> `pytest-flask` plugin (incompatible with the installed Flask). `pytest.ini`
> disables that one plugin (`-p no:flask`) for this project only — it is
> unrelated to the assignment code. In a clean virtualenv the flag is harmless.

---

## 8. Key Takeaways

1. **Vehicle weight is the dominant driver of fuel economy** in this era's fleet
   (r = −0.832); engine size features add little beyond it due to collinearity.
2. **Nonlinearity pays off:** Random Forest beats both linear models by
   ~0.5 mpg MAE (1.941 vs 2.424) and +0.04 R², thanks to interactions such as
   weight × model year and origin-specific efficiency patterns.
3. **Leakage discipline:** every preprocessing statistic (group medians,
   scaling) is fit strictly on training data — enforced structurally by putting
   the imputer inside the sklearn `Pipeline`, not by convention.
4. **Simple and honest model selection:** candidates are ranked by 5-fold CV on
   the training set; the test set is touched exactly once, for the final report.

---

## 9. References

* UCI Auto MPG dataset — <https://archive.ics.uci.edu/dataset/9/auto+mpg>
* Quinlan, R. (1993). *Combining Instance-Based and Model-Based Learning*,
  Proceedings of the Tenth International Conference on Machine Learning, 236–243.
* scikit-learn documentation — <https://scikit-learn.org/>
