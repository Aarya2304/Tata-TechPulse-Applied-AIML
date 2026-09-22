"""Classification model zoo (CPU-friendly, imbalance-aware).

All three models receive ``class_weight="balanced"`` (weights inversely
proportional to class frequencies) so the rare failure class is upweighted;
no resampling (e.g. SMOTE) is used anywhere. Each model is combined with
the shared leakage-safe preprocessor in a sklearn Pipeline by train.py /
build_pipeline().
"""

from __future__ import annotations

from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing import build_preprocessor

LOGISTIC_REGRESSION = "Logistic Regression"
RANDOM_FOREST = "Random Forest"
HIST_GRADIENT_BOOSTING = "Hist Gradient Boosting"


def build_models() -> dict:
    """Return name -> unfitted classifier for all candidates."""
    return {
        LOGISTIC_REGRESSION: LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000,
            random_state=config.RANDOM_STATE),
        RANDOM_FOREST: RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1),
        HIST_GRADIENT_BOOSTING: HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
            class_weight="balanced", early_stopping=False,
            random_state=config.RANDOM_STATE),
    }


def build_pipeline(estimator) -> Pipeline:
    """Wrap the shared preprocessor and a classifier in one Pipeline."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", estimator),
    ])
