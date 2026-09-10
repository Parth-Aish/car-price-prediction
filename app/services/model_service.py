"""
Model service — singleton that loads and holds the ML model in memory.

The model is loaded ONCE at app startup and reused for all requests,
avoiding the overhead of loading a ~50MB pickle on every prediction.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class ModelService:
    """Loads and serves the trained ML pipeline."""

    _instance = None

    def __init__(self):
        self.pipeline = None
        self.metadata = None
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, model_path: str, metadata_path: str) -> None:
        """Load model pipeline and metadata from disk."""
        try:
            logger.info("Loading model from %s", model_path)
            self.pipeline = joblib.load(model_path)
            logger.info("Model loaded successfully")

            logger.info("Loading metadata from %s", metadata_path)
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            logger.info("Metadata loaded: %s v%s", self.metadata["model_name"], self.metadata["model_version"])

            specs_path = Path(model_path).parent.parent / "data" / "car_specs.json"
            if specs_path.exists():
                with open(specs_path, "r") as f:
                    self.metadata["car_specs"] = json.load(f)
                logger.info("Car specs loaded")
            else:
                self.metadata["car_specs"] = {}
                logger.warning("car_specs.json not found")

            self._loaded = True

        except FileNotFoundError as e:
            logger.error("Model file not found: %s. Run 'python -m ml.train' first.", e)
            raise
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, features_df) -> float:
        """
        Run prediction on a prepared feature DataFrame.
        The pipeline handles preprocessing internally.
        Returns the predicted price in original scale (INR).
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Pipeline predicts in log-space since we trained on log1p(price)
        log_prediction = self.pipeline.predict(features_df)
        price = float(np.expm1(log_prediction[0]))

        # Clamp to a sensible minimum
        price = max(price, 10000)

        return round(price, -2)  # Round to nearest 100
