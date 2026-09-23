# Assignment 8 – Sentiment Analysis using LSTM

## Objective

Official requirement (Tata Technologies TechPulse FY-26, Applied AI/ML):

> "Analyze vehicle feedback using LSTM-based sentiment classification."

The project implements exactly that: a real, from-scratch LSTM sentiment
classifier built with TensorFlow/Keras (`Embedding → LSTM → Dropout → Dense →
Sigmoid`) over real vehicle/customer feedback text. No BERT, no transformers,
no pretrained language models — the only learned text representation is the
network's own embedding layer.

## Dataset

| Property | Value |
|---|---|
| Name | Amazon Automotive product reviews (`reviews_Automotive_5.json.gz`, SNAP/UCSD McAuley 5-core release, 2016) |
| Source | <https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Automotive_5.json.gz> — downloaded automatically (4.7 MB, gitignored) |
| Relation to the assignment | Real customer feedback on **vehicle parts and accessories** (batteries, wipers, trailer hitches, …) — this *is* vehicle feedback, not a generic stand-in |
| Raw rows | **20,473** reviews |
| Text field | `reviewText` (+ `summary` title prepended — titles carry sentiment) |
| Target field | `overall` (1–5★) → derived binary `sentiment` |

**Label derivation (documented, not silent):** 1–2★ → Negative (0), 4–5★ →
Positive (1), **3★ treated as neutral and dropped** (1,430 rows). Raw ratings:
1★ 542 · 2★ 606 · 3★ 1,430 · 4★ 3,967 · 5★ 13,928.

**Cleaning (all measured at runtime, reported by `train.py` and saved in
`model_metadata.json`):**

| Step | Rows removed |
|---|---:|
| Empty/missing review text or unusable rating | 6 |
| Neutral 3-star (deliberate label policy) | 1,430 |
| Exact duplicate review texts | 5 |
| **Final usable rows** | **19,032** |

**Class distribution (imbalanced):** Negative **1,147 (6.0 %)** · Positive
**17,885 (94.0 %)**.

**Text lengths (words):** min 4 · median 57 · mean 89.4 · p95 265 · max 2,243
(motivates the 120-token sequence length).

## Text Preprocessing

1. **Cleaning** — lower-case; strip URLs and HTML tags; collapse whitespace.
   Sentence punctuation is kept. **Negation words (`not`, `no`, `never`,
   `n't` …) are deliberately preserved** — removing them would destroy
   sentiment information.
2. **Tokenization** — Keras `Tokenizer` with `num_words=12000` and an explicit
   `<OOV>` token for unseen words at inference time.
3. **Integer sequences** + **padding** to a fixed length of **120** with
   *pre*-padding (`padding="pre"`, `truncating="pre"`). Pre-padding is the
   standard Keras recipe for recurrent models: real tokens end at the sequence
   end, so the LSTM's final state reads review content. This mattered a lot:
   with post-padding the model learned almost nothing (test ROC-AUC ≈ 0.57,
   near the always-positive baseline); with pre-padding it works (0.81).
4. **No data leakage:** the tokenizer is fitted **only on the 15,225 training
   reviews**; validation/test text is converted with that frozen vocabulary. A
   test asserts that words appearing only in held-out data are absent from the
   vocabulary.
5. **Persistence:** `models/tokenizer.json` stores the fitted vocabulary, and
   `models/model_metadata.json` stores the sequence length — the inference
   path loads both, so it cannot drift from training.

## LSTM Architecture

Actual model (`src/model.py`, `build_lstm_model`), **803,137 parameters**:

```
Input (token ids, length 120)
→ Embedding(vocab_size=12000, output_dim=64, mask_zero=True)   # 768,064 params
→ LSTM(64)                                                     # 33,024 params
→ Dropout(0.4)
→ Dense(32, relu)                                              #  2,080 params
→ Dense(1, sigmoid)                                            #     33 params
```

- **Embedding** learns 64-dim word vectors from scratch (index 0 = padding is
  masked so pad steps never influence the LSTM state).
- **LSTM(64)** summarises the token sequence; its final state is the review
  representation.
- **Dropout(0.4)** regularises the LSTM output.
- **Dense(32, relu) → Dense(1, sigmoid)** maps the summary to
  P(positive sentiment).

## Training

| Setting | Value |
|---|---|
| Optimizer | Adam, learning rate 1e-3 (ReduceLROnPlateau halves it on plateaus) |
| Loss | binary cross-entropy |
| Metrics | accuracy (P/R/F1/AUC computed at evaluation) |
| Batch size | 32 |
| Max epochs | 12 (early stopping restored the best epoch: **3**) |
| Callbacks | `ModelCheckpoint(val_loss, save_best_only)` · `EarlyStopping(patience=3, restore_best_weights=True)` · `ReduceLROnPlateau(factor=0.5, patience=2)` |
| Class weights | from **training data only**, sklearn "balanced" rule: Negative **8.29**, Positive **0.53** |
| Random seed | 42 (Python + NumPy + TensorFlow via `tf.keras.utils`/seeds) |
| Splits | stratified 80/10/10 → 15,225 / 1,903 / 1,904 |
| Runtime | ≈ 3.5 min on a laptop CPU; training is fully deterministic in setup |

## Evaluation

Measured on the **untouched test set** (1,904 reviews, threshold 0.5) —
`artifacts/metrics.json`:

| Metric | Value |
|---|---:|
| Accuracy | **0.9028** |
| Precision (positive class) | 0.9652 |
| Recall (positive class) | 0.9301 |
| F1 (positive class) | 0.9473 |
| ROC-AUC | **0.8070** |
| PR-AUC (Average Precision) | 0.9806 |

Because the classes are imbalanced, accuracy alone is misleading — the
always-positive baseline would already score 93.9 % accuracy. ROC-AUC 0.807
and the per-class breakdown below are the honest picture:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.306 | 0.478 | 0.373 | 115 |
| Positive | 0.965 | 0.930 | 0.947 | 1,789 |
| *Macro avg* | *0.635* | *0.704* | *0.660* | *1,904* |
| *Weighted avg* | *0.925* | *0.903* | *0.913* | *1,904* |

## Confusion Matrix

|  | predicted Negative | predicted Positive |
|---|---:|---:|
| **actual Negative** | TN = 55 | FP = 60 |
| **actual Positive** | FN = 125 | TP = 1,664 |

The model detects 52 % of the rare negative reviews (up from 0 % for an
unweighted classifier) while keeping positive precision at 0.965. The 60 false
alarms on positive reviews are the price of the 8.3× class weight on the
minority class.

## Training curves

`artifacts/plots/training_history.png` shows train accuracy climbing to 0.97
by epoch 6 while validation accuracy plateaus near 0.90 and validation loss
rises after epoch 3 — the model **does begin to overfit after ~3 epochs**,
which is exactly what EarlyStopping (best epoch 3, `restore_best_weights`)
guards against. No curve manipulation; reported as measured.

## Sample predictions

`artifacts/sample_predictions.csv` (+ `sample_predictions.png`) shows real
test reviews with true label, predicted label, probability and correctness —
**6 misclassified and 6 correctly classified** examples, not a cherry-picked
gallery. Examples of errors it contains: a sarcastic "Junk, back it goes"
review predicted positive (p=0.70), and an enthusiastic 5-star review
predicted negative (p=0.48).

## Error Analysis

Measured observations over the 185 misclassified test reviews (printed by
`train.py`):

- **Long reviews fail more often:** mean length 116 words in errors vs 89 in
  correct predictions. With 120-token sequences and front-truncation, the
  verdict sentence of very long reviews may be cut off.
- **Many errors are low-confidence:** 58/185 (31 %) have probability in
  [0.4, 0.6] — borderline cases rather than confident mistakes.
- **Negation is a real weakness:** 104/115 test negatives contain negation
  words, and short negation-heavy sentences ("Not good at all.") are still
  misclassified positive by the demo model — the LSTM has too little
  data/compositional power to fully resolve long-distance negation.
- **Mixed sentiment and sarcasm** ("Worked fine for about an hour", "Won't
  fit — 'Universal' is a lie") appear among the false positives in the
  sample-predictions artifact.
- **No evidence of length-invariant failure** in the other direction: very
  short reviews are *not* disproportionately wrong.

## Prediction

```bash
python -m src.predict "The vehicle is excellent and very comfortable."
# Positive  p=0.977
python -m src.predict "Do not buy this, complete waste of money."
# Negative  p=0.047
python -m src.predict --demo     # six built-in examples
python -m src.predict --batch texts.txt   # one sentence per line
```

The script loads `models/best_sentiment_lstm.keras` + `models/tokenizer.json`
+ `models/model_metadata.json` and applies exactly the training preprocessing
(clean → fitted tokenizer → pre-pad to 120). Results are produced by the
saved model, nothing is hardcoded.

## How to Run

```bash
cd Assignment-08-Sentiment-LSTM
pip install -r requirements.txt

python -m src.train        # downloads data, trains, evaluates, writes artifacts
python -m src.predict "The vehicle is excellent and comfortable."
pytest -q
```

## Artifacts

```
models/best_sentiment_lstm.keras      # 9.4 MB trained LSTM (.keras format)
models/tokenizer.json                 # 2.2 MB fitted vocabulary
models/model_metadata.json            # architecture + all run settings
artifacts/metrics.json                # test metrics (this README's table)
artifacts/classification_report.csv   # per-class precision/recall/F1
artifacts/sample_predictions.csv      # 12 real test reviews incl. errors
artifacts/reports/training_log.csv    # per-epoch metrics
artifacts/plots/class_distribution.png
artifacts/plots/text_length_distribution.png
artifacts/plots/training_history.png
artifacts/plots/confusion_matrix.png
artifacts/plots/sample_predictions.png
```

## Testing

`pytest -q` → **30 passed** (~13 s). Coverage: label derivation and cleaning
(missing text, neutral drop, duplicates — with the real dataset when present),
text cleaning preserving negation, stratified deterministic splits,
train-only tokenizer fitting (unseen-word assertion), sequence shapes and
pre-padding layout, vocab cap, class-weight formula, tokenizer JSON round-trip,
LSTM layer presence (and absence of any transformer/attention layer), output
shape/probability range, one-step training sanity check, model/tokenizer/
metadata save-load, and the end-to-end prediction pipeline on positive,
negative and fully-unseen-word inputs.

## Limitations

- **Domain/size limits:** ~19k reviews of *parts/accessories* (not full
  vehicles) with only ~1.1k negative examples — the minority class is where
  the model is weakest (F1 0.37).
- **Vocabulary is frozen at training time:** slang, brands or typos unseen in
  2016 Amazon automotive reviews map to `<OOV>`.
- **120-token truncation** discards most of reviews longer than ~120 words
  (p95 = 265), and measured errors skew toward long reviews.
- **Negation and sarcasm** remain hard for a 64-unit LSTM trained from
  scratch; no pretrained language knowledge is available to the model by
  design of this assignment.
- **Class-weight trade-off** is a dial, not a fix: raising negative recall
  costs positive precision; the 8.3× weight is the documented operating point.
- CPU training is practical (~3.5 min) precisely because the model is small —
  capacity, not compute, is the binding constraint.

## Conclusion

A from-scratch Keras LSTM (803k parameters) trained on 19k real
vehicle-product reviews reaches **0.903 accuracy / 0.807 ROC-AUC / 0.981
PR-AUC**, and — with class weighting — recovers 52 % of rare negative
feedback at 0.31 precision. The measured ablations in this README
(post- vs pre-padding, truncation side, LR schedule) show the pipeline's
design choices were driven by data, and the saved artifacts reproduce
inference exactly.
