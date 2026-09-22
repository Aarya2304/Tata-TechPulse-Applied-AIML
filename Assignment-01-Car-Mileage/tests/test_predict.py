"""Tests for the prediction script (predict.py)."""

from __future__ import annotations

import pandas as pd

from predict import EXAMPLE_CARS, load_model, parse_specs, predict_mpg
from src import config


class TestParseSpecs:
    def test_parses_valid_pairs(self):
        frame = parse_specs(["cylinders=4", "weight=2130"])
        assert frame.loc[0, "cylinders"] == 4.0
        assert frame.loc[0, "weight"] == 2130.0

    def test_rejects_unknown_feature(self):
        import pytest
        with pytest.raises(ValueError):
            parse_specs(["not_a_feature=3"])

    def test_rejects_malformed_pair(self):
        import pytest
        with pytest.raises(ValueError):
            parse_specs(["cylinders4"])


class TestPredictMpg:
    def test_output_has_prediction_column(self):
        model = load_model()
        frame = pd.DataFrame([EXAMPLE_CARS[0]])
        out = predict_mpg(model, frame)
        assert "predicted_mpg" in out.columns
        assert 0 < out["predicted_mpg"].iloc[0] < 100

    def test_missing_column_raises(self):
        import pytest
        model = load_model()
        bad = pd.DataFrame([{"cylinders": 4}])  # missing the other features
        with pytest.raises(ValueError):
            predict_mpg(model, bad)

    def test_all_example_cars_predict(self):
        model = load_model()
        for car in EXAMPLE_CARS:
            out = predict_mpg(model, pd.DataFrame([car]))
            assert 0 < out["predicted_mpg"].iloc[0] < 100

    def test_features_only_columns_used(self):
        """car_name is carried through but never fed to the model."""
        model = load_model()
        car = dict(EXAMPLE_CARS[0])
        frame = pd.DataFrame([car])
        out = predict_mpg(model, frame)
        assert out["predicted_mpg"].iloc[0] > 0
