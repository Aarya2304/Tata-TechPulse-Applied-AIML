# Data

`car_data.csv` in this folder is a byte-for-byte copy of the **"Car details
v3.csv"** file from the Kaggle dataset
[*Vehicle dataset (from CarDekho)*](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
by nehalbirla, obtained from the public mirror at
<https://raw.githubusercontent.com/imanishshahu/Car-Price-Prediction/main/Car%20details%20v3.csv>
(checked 2026-09-22).

- **8,128 rows x 13 columns**, ~1.0 MB -- small enough to keep in the repo.
- Original Kaggle page for citation/licensing: see the link above.

If the file is missing, `src/load_data.py` falls back to this GitHub URL
automatically (an internet connection is only needed on first download).
