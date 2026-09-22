# Data

`ai4i2020.csv` is the **AI4I 2020 Predictive Maintenance Dataset** from the
UC Irvine Machine Learning Repository (dataset id **601**, CC BY 4.0):

- Landing page: <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
- Direct download used: <https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip>
  (zip contains `ai4i2020.csv`; the copy here is byte-identical, ~0.5 MB)

The dataset is the synthetic ground truth produced by the AI4I 2020
challenge organisers (Matzka, 2020, *Third International Conference on
Applied Machine Learning and Intelligence Informatics*) and simulates 10,000
operating hours of a CNC milling machine.

If `data/ai4i2020.csv` is missing, `python -m src.train` re-downloads the
same official zip automatically and extracts it (see `src/data_loader.py`),
so the pipeline works from a clean clone with internet access.
