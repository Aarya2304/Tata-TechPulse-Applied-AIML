# Data

This project uses the **Amazon Automotive product reviews** dataset
(`reviews_Automotive_5.json.gz`) — real customer reviews of vehicle parts and
accessories (car batteries, wiper blades, jumper cables, floor mats, …), i.e.
genuine vehicle/customer feedback.

## Source / provenance

- SNAP (Stanford Network Analysis Project) / Julian McAuley's Amazon
  product-data release, "Automotive" category, 5-core version (2016):
  <https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Automotive_5.json.gz>
- Downloaded automatically by `python -m src.train` (or
  `src.data_loader.download_dataset()`) when missing; **4.7 MB** gzipped and
  gitignored under `data/downloads/` — nothing large is committed.

## Contents (measured from the actual file)

- **20,473 reviews**, JSON-lines with 9 fields; used here:
  `reviewText` (text), `summary` (title), `overall` (1-5 star rating).
- Ratings: 1★ 542 · 2★ 606 · 3★ 1,430 · 4★ 3,967 · 5★ 13,928.
- Binary sentiment labels derived from the rating: **1-2★ → Negative (0)**,
  **4-5★ → Positive (1)**, 3★ treated as neutral and dropped (documented).
  After cleaning: **19,038 usable rows** — 1,148 negative / 17,890 positive.
- Review length: median 52 words, p95 ≈ 262, max 2,239 (motivates the
  120-token padded sequence length).

No reviews or labels are fabricated; the loader only parses, labels from
ratings, and removes 5 duplicate texts and rows with empty text.
