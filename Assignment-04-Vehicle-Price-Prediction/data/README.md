# Data

`car_data.csv` is a byte-identical copy of **"Car details v3.csv"** from the
Kaggle dataset [*Vehicle dataset (from CarDekho)*](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
(nehalbirla), obtained from the public GitHub mirror
<https://raw.githubusercontent.com/imanishshahu/Car-Price-Prediction/main/Car%20details%20v3.csv>.

- **8,128 rows x 13 columns**, ~1.0 MB.
- The same raw file used by Assignment 3 (copied here read-only; identical
  MD5 `08c5f2987e4e9366b11011e7f12aee92` — Assignment 3 was not modified).
- `src/data_loader.py` falls back to the mirror URL if the local file is
  missing (internet only needed on first download).
