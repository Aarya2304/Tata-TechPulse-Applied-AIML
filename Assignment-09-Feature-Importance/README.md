# Assignment 9 — Feature Importance Visualization

## 1. Objective

Build a clean, reproducible feature-importance workflow for a supervised
automotive ML task: train a tree-based model on real vehicle data, compute
**both** Random-Forest impurity importance and permutation importance, map
them correctly through the one-hot-expanded feature space, visualize and
compare the results, and interpret them without claiming causation.

## 2. Official Requirement

> "Feature Importance Visualization."

The TechPulse course covers feature selection and importance, including
importance from tree-based models; this assignment demonstrates that workflow
end-to-end with real, measured values.

## 3. Dataset

| Property | Value |
|---|---|
| Dataset | CarDekho **"Car details v3"** — 8,128 rows × 13 columns of real Indian used-car listings |
| Origin | Kaggle: `nehalbirla/vehicle-dataset-from-cardekho` |
| Access | **Reused read-only** from the repository's existing copy (`Assignment-04-Vehicle-Price-Prediction/data/car_data.csv`, byte-identical to Assignment 3's copy, md5 `08c5f2987e4e9366b11011e7f12aee92`). Nothing in those folders is copied or modified. If both are missing (fresh clone), the file downloads from the public mirror into this assignment's own `data/` (gitignored). |
| Rows used | 8,128 (0 rows lacked the target) |
| Target | `selling_price` (INR) — **never** an input feature |
| Inputs | 14 features: 9 numeric, 5 categorical |

## 4. Problem Definition

Supervised regression: predict a used car's `selling_price` from its
characteristics. Price prediction is the recommended task because its feature
importances are meaningful (a feature matters if it changes prediction error).
The model only needs to be good enough for importances to be informative —
metrics below establish that.

## 5. Data Preparation

- **Unit parsing** (safe regex extraction, documented fallbacks):
  `"23.4 kmpl"` → `mileage_kmpl` · `"1248 CC"` → `engine_cc` ·
  `"74 bhp"` → `max_power_bhp` · torque strings in all three dataset formats
  (`190Nm@ 2000rpm`, `250Nm@ 1500-2500rpm`, `12.7 kgm@ 2500 rpm`) →
  `torque_nm`, converting kgm → Nm (× 9.80665).
- **Engineered features:** `vehicle_age = 2020 − year` (fixed documented
  reference), `km_per_year = km_driven / vehicle_age` (NaN where age = 0,
  imputed in-pipeline), `brand` = first token of the vehicle name (the full
  `name` string is dropped as an identifier).
- **Preprocessing (sklearn `ColumnTransformer` inside a `Pipeline`):**
  numeric → median imputation (+ scaling); categorical → most-frequent
  imputation + `OneHotEncoder(handle_unknown="ignore")`. All learned
  statistics are fitted on the **training split only** (asserted by a test
  comparing imputer statistics to train-only medians).
- **Split:** 80/20, `random_state=42` → 6,502 train / 1,626 test.

## 6. Model

```
RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
```

Test-set metrics (supporting evidence that the model is meaningful):

| Metric | Value |
|---|---:|
| MAE | **67,110 INR** |
| RMSE | 134,939 INR |
| R² | **0.9722** |

The transformed feature space has **55 columns** (9 numeric + 46 one-hot
dummies across `brand` (32), `fuel` (4), `seller_type` (3), `transmission`
(2), `owner` (5)).

## 7. Feature Importance Methods

1. **Impurity importance** (`feature_importances_`): each feature's share of
   the total variance reduction achieved by all splits that use it, averaged
   over 300 trees; normalised to sum to 1. Computed on training data; fast
   and deterministic for a fixed seed, but biased towards high-cardinality /
   continuous features and unreliable when features are correlated.
2. **Permutation importance** (`sklearn.inspection.permutation_importance`):
   computed **on the untouched test set**, scoring
   `neg_mean_absolute_error`, 15 shuffling repeats (± std reported). Each
   value is the **MAE increase in rupees** when one feature's column is
   randomly shuffled — directly interpretable as "how much does prediction
   error worsen without this feature's information?" Values ≈ 0 mean the
   model barely relies on the feature.

**Correct name mapping:** the test data is transformed through the fitted
`ColumnTransformer` first, so permutation runs on exactly the 55 transformed
columns and both tables share the same names (`get_feature_names_out`).
**Grouped view:** one-hot dummies are aggregated back to their logical
features by prefix matching for a second, more interpretable table
(`grouped_feature_importance.csv`).

*Caveat (documented in code and report):* permuting one dummy of a
multi-level categorical creates unseen value combinations; the grouped view
mitigates this by summing all dummies of a feature.

## 8. Results

### Top 10 — Random-Forest impurity importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | max_power_bhp | 0.7021 |
| 2 | year | 0.0929 |
| 3 | vehicle_age | 0.0927 |
| 4 | torque_nm | 0.0363 |
| 5 | km_driven | 0.0221 |
| 6 | mileage_kmpl | 0.0101 |
| 7 | engine_cc | 0.0073 |
| 8 | km_per_year | 0.0062 |
| 9 | brand_Volvo | 0.0047 |
| 10 | seller_type_Individual | 0.0027 |

### Top 10 — permutation importance (test set, MAE increase in INR)

| Rank | Feature | MAE increase | ± std |
|---:|---|---:|---:|
| 1 | max_power_bhp | 261,917 | 8,900 |
| 2 | vehicle_age | 82,627 | 2,643 |
| 3 | year | 75,897 | 2,657 |
| 4 | torque_nm | 55,310 | 2,094 |
| 5 | km_driven | 19,271 | 1,482 |
| 6 | mileage_kmpl | 15,646 | 974 |
| 7 | engine_cc | 12,040 | 472 |
| 8 | brand_Tata | 6,382 | 545 |
| 9 | brand_Toyota | 6,201 | 434 |
| 10 | km_per_year | 5,993 | 536 |

### Grouped (original-feature) view — where the aggregation matters

Summing all one-hot dummies back to their logical features makes `brand` the
**5th-ranked original feature** (21,496 INR total across 32 dummies) even
though no single brand dummy reaches the individual top 10 — exactly the
blind spot the grouped view is meant to reveal:

| Original feature | Impurity share | Permutation (INR) | Dummies |
|---|---:|---:|---:|
| max_power_bhp | 0.702 | 261,917 | 1 |
| year | 0.093 | 75,897 | 1 |
| vehicle_age | 0.093 | 82,627 | 1 |
| torque_nm | 0.036 | 55,310 | 1 |
| brand | 0.019 | 21,496 | 32 |
| km_driven | 0.022 | 19,271 | 1 |
| mileage_kmpl | 0.010 | 15,646 | 1 |

### Reproducibility check

Permutation importance recomputed with seeds 42–46: top-1 feature is
`max_power_bhp` in **all 5 runs** (stable). Impurity importance is
deterministic given the seed. Full run takes ≈ 1.8 min on a laptop CPU.

## 9. Visualizations

All under `artifacts/` (generated from the same run as the numbers above):

| File | Content |
|---|---|
| `top_feature_importance.png` | Top-20 impurity importances, horizontal bars, sorted |
| `top_permutation_importance.png` | Top-20 permutation importances with ±std error bars |
| `importance_comparison.png` | Grouped bars comparing *normalised* impurity vs permutation importance for the top common features |

## 10. Interpretation

- **What importance means here:** how much the *model* uses a feature —
  either to reduce variance during tree construction (impurity) or how much
  test error rises when the feature's information is destroyed (permutation).
- **Why the methods disagree:** impurity importance is computed on training
  data and is biased towards continuous/high-cardinality features; when two
  features are correlated, the forest splits credit between them
  arbitrarily. Permutation importance is measured on unseen data and
  reflects the model's *actual reliance*: if a correlated partner can
  compensate, shuffling one feature costs little. Measured example:
  `vehicle_age` and `year` are near-perfectly correlated (`age = 2020 −
  year`) and tie in impurity (0.093 each), but permutation ranks
  `vehicle_age` clearly above `year` (82.6k vs 75.9k INR) while `torque_nm`
  gains much more weight by permutation (0.052 normalised impurity vs 0.211
  normalised permutation) — the forest apparently leans on `max_power` and
  can partly cover torque's information.
- **What the rankings mean for this dataset/model:** engine power
  (`max_power_bhp`) dominates both methods by a wide margin; age/year is the
  second signal; torque, usage intensity (`km_driven`, `km_per_year`),
  fuel economy and engine size follow; brand matters as a *group* (premium
  vs mass-market marques) rather than through any single dummy; `owner`,
  `fuel`, `seller_type`, `transmission` and `seats` contribute little once
  power and age are known.
- **Importance is not causation:** these values describe the model's
  reliance patterns within this dataset's distribution — they do not prove
  that changing a car's power *causes* its price to change, and they must
  not be read as a causal price model.

## 11. Limitations

- **Correlated features** (`year`/`vehicle_age`/`km_per_year`;
  `engine_cc`/`max_power`/`torque_nm`) split and hide credit; importance is
  a property of the *model+data*, not of the feature alone.
- **Categorical encoding:** impurity importance is spread across dummies,
  and single-dummy permutation tests create unrealistic category
  combinations (mitigated by the grouped view, not eliminated).
- **Dataset bias:** 2016-era Indian used-car listings from CarDekho —
  rankings need not transfer to other markets or periods.
- **Model dependence:** a gradient-boosting or linear model could rank
  features differently; these results describe this Random Forest only.
- **Impurity bias** towards continuous features inflates numeric importance
  relative to low-cardinality categoricals.

## 12. Project Structure

```
Assignment-09-Feature-Importance/
├── README.md, requirements.txt, pytest.ini, conftest.py, .gitignore
├── data/README.md                  # provenance (read-only reuse + fallback)
├── src/
│   ├── config.py                   # paths, seed, model/importance settings
│   ├── data.py                     # locate (read-only), parse, engineer, split
│   ├── model.py                    # ColumnTransformer pipeline + RF + metrics
│   ├── importance.py               # both methods, grouping, plots, CSVs
│   └── main.py                     # orchestration + auto-generated report
├── tests/                          # test_data.py, test_model.py, test_importance.py
├── artifacts/                      # 3 CSVs, metrics.json, 3 PNGs
└── reports/feature_importance_report.md   # generated with actual values
```

No model file is committed: the pipeline is refit from data in ~2 minutes,
so `artifacts/` contains only small CSV/PNG/JSON outputs.

## 13. How to Run

```bash
cd Assignment-09-Feature-Importance
python -m pip install -r requirements.txt
python -m src.main              # full workflow ≈ 2 min on CPU
python -m src.main --skip-repro # skip the 5-seed reproducibility check
```

Works both from inside the folder and from the repository root
(`python -m src.main` with the assignment folder as working directory);
all paths are resolved relative to `src/config.py`, never to the user's
machine.

## 14. How to Test

```bash
python -m pytest Assignment-09-Feature-Importance/tests -q   # from repo root
# or, from inside the folder:
python -m pytest tests -q
```

**21 tests: 19 pass, 2 skip** (artifact checks that auto-enable once
`python -m src.main` has been run — after the run, **21/21 pass**).
Covered: dataset loading and provenance, unit parsers (incl. kgm→Nm),
feature engineering, target presence/numeric type, **no target leakage in
the feature matrix**, split reproducibility, pipeline structure with
`handle_unknown="ignore"`, **fit-on-train-only imputer statistics**, finite
metrics, deterministic predictions, unseen-category robustness, impurity
finiteness/non-negativity/sum-to-1, permutation finiteness/sorting,
same-seed permutation reproducibility, grouped-view mass conservation,
comparison bounds, sorted saved tables, and artifact generation.

## 15. Reproducibility

- `random_state=42` in the split, the forest, and permutation importance;
- deterministic unit parsing and feature engineering;
- re-running `python -m src.main` regenerates every artifact with identical
  values (verified: the multi-seed check keeps the top feature stable, and
  same-seed permutation tables match to 10 decimal places in tests);
- the report `reports/feature_importance_report.md` is regenerated from the
  same run — README and report can never drift from the code's output.
