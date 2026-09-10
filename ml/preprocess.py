"""
Data preprocessing module.

Handles loading, cleaning, and building the sklearn preprocessing pipeline.
The same pipeline is used for both training and inference to prevent skew.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Brand name normalization map — fixes inconsistencies found in EDA
BRAND_NORMALIZATION = {
    "ISUZU": "Isuzu",
    "Mercedes-AMG": "Mercedes-Benz",
}

# Brand → market segment mapping for feature engineering
BRAND_SEGMENTS = {
    "Maruti": "Budget", "Datsun": "Budget", "Renault": "Budget", "Tata": "Budget",
    "Hyundai": "Mid", "Honda": "Mid", "Ford": "Mid", "Volkswagen": "Mid",
    "Nissan": "Mid", "Kia": "Mid", "MG": "Mid", "Skoda": "Mid",
    "Toyota": "Mid-Premium", "Mahindra": "Mid-Premium", "Jeep": "Mid-Premium",
    "BMW": "Premium", "Mercedes-Benz": "Premium", "Audi": "Premium",
    "Volvo": "Premium", "Mini": "Premium", "Jaguar": "Premium",
    "Land Rover": "Luxury", "Porsche": "Luxury", "Lexus": "Luxury",
    "Bentley": "Luxury", "Maserati": "Luxury", "Ferrari": "Luxury",
    "Rolls-Royce": "Luxury", "Isuzu": "Mid", "Force": "Mid",
}


def load_dataset(path: str) -> pd.DataFrame:
    """Load raw CSV and drop the useless index column."""
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    logger.info("Dataset loaded: %d rows, %d columns", len(df), len(df.columns))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
    1. Normalize brand names
    2. Fix invalid seat counts
    3. Drop car_name (redundant with brand+model)
    4. Cap extreme outliers in km_driven and selling_price
    """
    df = df.copy()
    logger.info("Starting data cleaning on %d rows", len(df))

    # 1. Normalize brand names
    df["brand"] = df["brand"].replace(BRAND_NORMALIZATION)
    logger.info("Normalized brand names")

    # 2. Fix seats=0 (only 2 records) — impute with mode
    zero_seats = df["seats"] == 0
    if zero_seats.any():
        seat_mode = df.loc[~zero_seats, "seats"].mode()[0]
        df.loc[zero_seats, "seats"] = seat_mode
        logger.info("Fixed %d records with seats=0 → %d", zero_seats.sum(), seat_mode)

    # 3. Drop car_name — redundant with brand + model
    if "car_name" in df.columns:
        df = df.drop(columns=["car_name"])

    # 4. Cap outliers using IQR method on km_driven
    q1_km = df["km_driven"].quantile(0.25)
    q3_km = df["km_driven"].quantile(0.75)
    iqr_km = q3_km - q1_km
    upper_km = q3_km + 3 * iqr_km  # Using 3×IQR to be conservative
    capped_km = (df["km_driven"] > upper_km).sum()
    df["km_driven"] = df["km_driven"].clip(upper=upper_km)
    logger.info("Capped %d km_driven outliers at %.0f", capped_km, upper_km)

    # 5. Cap extreme selling_price outliers (top 1%)
    price_cap = df["selling_price"].quantile(0.99)
    capped_price = (df["selling_price"] > price_cap).sum()
    df["selling_price"] = df["selling_price"].clip(upper=price_cap)
    logger.info("Capped %d selling_price outliers at %.0f", capped_price, price_cap)

    logger.info("Data cleaning complete: %d rows remain", len(df))
    return df


def build_preprocessor(categorical_features: list, numerical_features: list) -> ColumnTransformer:
    """
    Build the sklearn ColumnTransformer that encodes categoricals and scales numericals.
    This object gets fitted on training data and reused identically at inference.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            (
                "num",
                StandardScaler(),
                numerical_features,
            ),
        ],
        remainder="drop",
    )
    return preprocessor
