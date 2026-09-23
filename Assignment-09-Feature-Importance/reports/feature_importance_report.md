# Feature Importance Report — CarDekho "Car details v3"

Generated automatically by `python -m src.main` (random_state=42).

## Dataset summary

- Source: reused read-only from Assignment 3/4 local copy
- Origin: Kaggle: nehalbirla/vehicle-dataset-from-cardekho (Car details v3)
- Rows used: 8128 (dropped 0 rows with missing target)
- Features: 14 (9 numeric, 5 categorical)
- Numeric: year, km_driven, seats, mileage_kmpl, engine_cc, max_power_bhp, torque_nm, vehicle_age, km_per_year
- Categorical: fuel, seller_type, transmission, owner, brand

## Model metrics (test set, 20% split)

| Metric | Value |
|---|---:|
| MAE | 67,110 INR |
| RMSE | 134,939 INR |
| R² | 0.9722 |

## Top 15 features — Random-Forest impurity importance

| Rank | Transformed feature | Importance |
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
| 11 | brand_Toyota | 0.0025 |
| 12 | brand_Mercedes-Benz | 0.0023 |
| 13 | brand_BMW | 0.0022 |
| 14 | brand_Land | 0.0020 |
| 15 | seller_type_Dealer | 0.0018 |

## Top 15 features — permutation importance (test set, 15 repeats)

Values are the MAE increase (INR) when the feature is shuffled; ± std across repeats.

| Rank | Transformed feature | MAE increase (INR) | ± std |
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
| 11 | brand_BMW | 3,070 | 767 |
| 12 | seller_type_Individual | 2,716 | 160 |
| 13 | seats | 2,108 | 252 |
| 14 | seller_type_Dealer | 1,759 | 140 |
| 15 | brand_Volvo | 1,491 | 176 |

## Grouped (original-feature) view

One-hot columns are aggregated back to their logical features.

| Original feature | Impurity share | Permutation (MAE, INR) | Transformed columns |
|---|---:|---:|---:|
| max_power_bhp | 0.7021 | 261,917 | 1 |
| year | 0.0929 | 75,897 | 1 |
| vehicle_age | 0.0927 | 82,627 | 1 |
| torque_nm | 0.0363 | 55,310 | 1 |
| km_driven | 0.0221 | 19,271 | 1 |
| brand | 0.0194 | 21,496 | 32 |
| mileage_kmpl | 0.0101 | 15,646 | 1 |
| engine_cc | 0.0073 | 12,040 | 1 |
| km_per_year | 0.0062 | 5,993 | 1 |
| seller_type | 0.0045 | 4,724 | 3 |
| transmission | 0.0018 | 2,286 | 2 |
| owner | 0.0017 | 1,764 | 5 |
| fuel | 0.0016 | 1,295 | 4 |
| seats | 0.0011 | 2,108 | 1 |

## Impurity vs permutation — where they disagree

Normalised (0-1 within each method) importances of the top features, with
the absolute gap between the two methods. Larger gaps mark features whose
model usage (impurity) and measured usefulness (permutation) diverge —
typically features correlated with others (impurity splits credit across
them; permutation hides the redundancy because the model can compensate):

| Feature | Impurity (norm) | Permutation (norm) | Gap |
|---|---:|---:|---:|
| max_power_bhp | 1.000 | 1.000 | 0.000 |
| year | 0.132 | 0.290 | 0.157 |
| vehicle_age | 0.132 | 0.316 | 0.184 |
| torque_nm | 0.052 | 0.211 | 0.160 |
| km_driven | 0.032 | 0.074 | 0.042 |
| mileage_kmpl | 0.014 | 0.060 | 0.045 |
| engine_cc | 0.010 | 0.046 | 0.036 |
| km_per_year | 0.009 | 0.023 | 0.014 |
| brand_Volvo | 0.007 | 0.006 | 0.001 |
| seller_type_Individual | 0.004 | 0.010 | 0.007 |


## Reproducibility check

Impurity importance is deterministic given the seed. Permutation importance
was recomputed with seeds 42–46 (top-1 feature each time): top-1 feature = 'max_power_bhp' in all 5 runs (stable: True)
