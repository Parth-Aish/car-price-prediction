"""
Central application configuration.
All paths and secrets are driven by environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Ensure the instance directory exists for SQLite
INSTANCE_DIR = BASE_DIR / "instance"
INSTANCE_DIR.mkdir(exist_ok=True)


class Config:
    """Base configuration."""

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{BASE_DIR / 'instance' / 'predictions.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ML paths
    MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "car_price_model.pkl"))
    MODEL_METADATA_PATH = os.getenv(
        "MODEL_METADATA_PATH",
        str(BASE_DIR / "models" / "model_metadata.json")
    )
    DATASET_PATH = os.getenv("DATASET_PATH", str(BASE_DIR / "data" / "cardekho_dataset.csv"))

    # Feature configuration — single source of truth shared by training and inference
    CATEGORICAL_FEATURES = ["brand", "model", "seller_type", "fuel_type", "transmission_type"]
    NUMERICAL_FEATURES = [
        "vehicle_age", "km_driven", "mileage", "engine", "max_power",
        "seats", "power_per_cc", "km_per_year"
    ]
    TARGET = "selling_price"

    # Valid categorical values (populated from training data at training time,
    # loaded from metadata at inference time)
    VALID_FUEL_TYPES = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
    VALID_TRANSMISSION_TYPES = ["Manual", "Automatic"]
    VALID_SELLER_TYPES = ["Individual", "Dealer", "Trustmark Dealer"]

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}
