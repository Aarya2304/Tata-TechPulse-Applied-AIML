# Data

This project does **not** bundle or duplicate the dataset.

## Source

- Dataset: CarDekho **"Car details v3"** — 8,128 rows × 13 columns of real
  Indian used-car listings (Kaggle: `nehalbirla/vehicle-dataset-from-cardekho`).
- **Primary access (read-only reuse):** the identical CSV already exists in
  this repository as `Assignment-04-Vehicle-Price-Prediction/data/car_data.csv`
  and `Assignment-03-Data-Cleaning/data/car_data.csv` (both byte-identical,
  md5 `08c5f2987e4e9366b11011e7f12aee92`). Assignment 9 opens the file
  **read-only** — nothing there is copied, modified or deleted.
- **Fallback:** if neither local copy exists (e.g. a fresh clone),
  `src.data.locate_raw_csv()` downloads the same file from the public GitHub
  mirror
  <https://raw.githubusercontent.com/imanishshahu/Car-Price-Prediction/main/Car%20details%20v3.csv>
  into this assignment's own `data/car_data.csv` (gitignored).

Either way, the raw data is never modified by this project.
