"""
Feature engineering module.

Creates derived features and handles the target variable transformation.
All transformations are deterministic and reproducible for inference.
"""

import logging
import pandas as pd
import numpy as np

from ml.preprocess import BRAND_SEGMENTS

logger = logging.getLogger(__name__)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that capture domain-relevant information:
    - power_per_cc: engine efficiency / performance density
    - km_per_year: usage intensity (high-use vehicles depreciate differently)
    - brand_segment: market tier (Budget → Luxury) — strong price signal
    """
    df = df.copy()

    # Power density — sports/performance cars have higher ratios
    df["power_per_cc"] = df["max_power"] / df["engine"].replace(0, np.nan)
    df["power_per_cc"] = df["power_per_cc"].fillna(df["power_per_cc"].median())

    # Usage intensity — a 3-year-old car with 100k km is different from one with 20k km
    df["km_per_year"] = df["km_driven"] / df["vehicle_age"].replace(0, 1)

    # Brand market segment — captures the price tier better than individual brand dummies
    df["brand_segment"] = df["brand"].map(BRAND_SEGMENTS).fillna("Mid")

    logger.info("Added engineered features: power_per_cc, km_per_year, brand_segment")
    return df


def prepare_features_for_prediction(input_data: dict, current_year: int = 2026) -> pd.DataFrame:
    """
    Transform a single prediction request dict into the feature DataFrame
    expected by the trained pipeline.

    The API accepts 'year' (manufacturing year) but the model expects 'vehicle_age'.
    """
    df = pd.DataFrame([input_data])

    # Convert year → vehicle_age if year was provided
    if "year" in df.columns:
        df["vehicle_age"] = current_year - df["year"]
        df = df.drop(columns=["year"])

    # Map API field names to dataset column names
    column_mapping = {
        "kilometers_driven": "km_driven",
        "transmission": "transmission_type",
        "power": "max_power",
    }
    df = df.rename(columns=column_mapping)

    # Add engineered features
    df = add_engineered_features(df)

    return df
