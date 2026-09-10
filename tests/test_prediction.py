"""
Tests for the ML model and prediction pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config


class TestModelLoading:
    """Verify the trained model artifact loads and works correctly."""

    def test_model_file_exists(self):
        assert Path(Config.MODEL_PATH).exists(), (
            f"Model file not found at {Config.MODEL_PATH}. Run 'python -m ml.train' first."
        )

    def test_metadata_file_exists(self):
        assert Path(Config.MODEL_METADATA_PATH).exists(), (
            f"Metadata file not found at {Config.MODEL_METADATA_PATH}."
        )

    def test_model_loads_successfully(self):
        import joblib
        pipeline = joblib.load(Config.MODEL_PATH)
        assert pipeline is not None

    def test_metadata_contains_required_fields(self):
        import json
        with open(Config.MODEL_METADATA_PATH) as f:
            meta = json.load(f)

        required_keys = [
            "model_name", "model_version", "training_date",
            "test_metrics", "valid_brands", "valid_models",
        ]
        for key in required_keys:
            assert key in meta, f"Metadata missing key: {key}"

    def test_metadata_has_valid_metrics(self):
        import json
        with open(Config.MODEL_METADATA_PATH) as f:
            meta = json.load(f)

        metrics = meta["test_metrics"]
        assert metrics["R2"] > 0.5, f"R² too low: {metrics['R2']}"
        assert metrics["MAE"] > 0, f"MAE should be positive: {metrics['MAE']}"
        assert metrics["RMSE"] > 0, f"RMSE should be positive: {metrics['RMSE']}"


class TestPrediction:
    """Verify predictions produce valid numeric outputs."""

    @pytest.fixture
    def pipeline(self):
        import joblib
        return joblib.load(Config.MODEL_PATH)

    def test_prediction_returns_numeric(self, pipeline):
        from ml.feature_engineering import prepare_features_for_prediction

        input_data = {
            "brand": "Maruti",
            "model": "Alto",
            "year": 2020,
            "kilometers_driven": 30000,
            "fuel_type": "Petrol",
            "transmission": "Manual",
            "seller_type": "Individual",
            "engine": 796,
            "mileage": 22.0,
            "power": 47.3,
            "seats": 5,
        }

        features_df = prepare_features_for_prediction(input_data)
        log_pred = pipeline.predict(features_df)
        price = float(np.expm1(log_pred[0]))

        assert isinstance(price, float)
        assert price > 0, f"Predicted price should be positive, got {price}"
        assert price < 50_000_000, f"Predicted price unreasonably high: {price}"

    def test_prediction_for_luxury_car(self, pipeline):
        from ml.feature_engineering import prepare_features_for_prediction

        input_data = {
            "brand": "BMW",
            "model": "5 Series",
            "year": 2022,
            "kilometers_driven": 15000,
            "fuel_type": "Diesel",
            "transmission": "Automatic",
            "seller_type": "Dealer",
            "engine": 2993,
            "mileage": 15.0,
            "power": 265.0,
            "seats": 5,
        }

        features_df = prepare_features_for_prediction(input_data)
        log_pred = pipeline.predict(features_df)
        price = float(np.expm1(log_pred[0]))

        assert price > 500_000, "Luxury car price should be substantial"

    def test_required_features_are_present(self, pipeline):
        """Verify the pipeline knows about all expected features."""
        preprocessor = pipeline.named_steps["preprocessor"]
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert "cat" in transformer_names
        assert "num" in transformer_names


class TestFeatureEngineering:
    """Verify feature engineering produces expected columns."""

    def test_prepare_features_creates_vehicle_age(self):
        from ml.feature_engineering import prepare_features_for_prediction

        result = prepare_features_for_prediction({
            "brand": "Maruti", "model": "Swift", "year": 2020,
            "kilometers_driven": 20000, "fuel_type": "Petrol",
            "transmission": "Manual", "seller_type": "Individual",
            "engine": 1197, "mileage": 23.0, "power": 88.0, "seats": 5,
        })

        assert "vehicle_age" in result.columns
        assert "year" not in result.columns
        assert result["vehicle_age"].iloc[0] == 6  # 2026 - 2020

    def test_engineered_features_present(self):
        from ml.feature_engineering import prepare_features_for_prediction

        result = prepare_features_for_prediction({
            "brand": "Hyundai", "model": "i20", "year": 2021,
            "kilometers_driven": 25000, "fuel_type": "Petrol",
            "transmission": "Manual", "seller_type": "Individual",
            "engine": 1197, "mileage": 20.0, "power": 82.0, "seats": 5,
        })

        assert "power_per_cc" in result.columns
        assert "km_per_year" in result.columns
        assert "brand_segment" in result.columns
