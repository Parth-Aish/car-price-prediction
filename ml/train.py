"""
Model training pipeline.

Usage:
    python -m ml.train

Loads data → cleans → engineers features → trains multiple models →
evaluates → selects best → serializes pipeline + metadata.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Add project root to path so config is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from ml.preprocess import load_dataset, clean_data, build_preprocessor
from ml.feature_engineering import add_engineered_features
from ml.evaluate import compute_metrics, evaluate_model_cv, print_comparison_table, select_best_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_candidate_models() -> list[tuple[str, object]]:
    """Return a list of (name, model) pairs to evaluate."""
    return [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression", Ridge(alpha=1.0)),
        ("Lasso Regression", Lasso(alpha=1.0, max_iter=10000)),
        ("Random Forest", RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, n_jobs=1, random_state=42
        )),
        ("Gradient Boosting", GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            min_samples_split=5, random_state=42
        )),
        ("Extra Trees", ExtraTreesRegressor(
            n_estimators=200, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, n_jobs=1, random_state=42
        )),
        ("HistGradient Boosting", HistGradientBoostingRegressor(
            max_iter=200, max_depth=8, learning_rate=0.1,
            min_samples_leaf=10, random_state=42
        )),
    ]


def train_pipeline():
    """Execute the full training pipeline."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── 1. Load and clean data ─────────────────────────────────────
    df = load_dataset(Config.DATASET_PATH)
    df = clean_data(df)
    df = add_engineered_features(df)

    logger.info("Final feature set: %s", df.columns.tolist())
    logger.info("Dataset shape after processing: %s", df.shape)

    # ── 2. Separate features and target ────────────────────────────
    # Use log-transform on target to handle right-skew
    X = df[Config.CATEGORICAL_FEATURES + Config.NUMERICAL_FEATURES]
    y = np.log1p(df[Config.TARGET])  # log(1 + price) for numerical stability

    logger.info("Feature matrix shape: %s", X.shape)
    logger.info("Target range (log): %.2f – %.2f", y.min(), y.max())

    # ── 3. Train/test split ────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    # ── 4. Build preprocessor and transform ────────────────────────
    preprocessor = build_preprocessor(Config.CATEGORICAL_FEATURES, Config.NUMERICAL_FEATURES)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    logger.info("Processed feature dimensions: %d", X_train_processed.shape[1])

    # ── 5. Train and evaluate all candidate models ─────────────────
    candidates = get_candidate_models()
    results = []

    for name, model in candidates:
        logger.info("Training: %s", name)
        model_start = time.time()

        model.fit(X_train_processed, y_train)

        # Predictions in log-space, convert back for metrics
        y_pred_log = model.predict(X_test_processed)
        y_pred = np.expm1(y_pred_log)
        y_actual = np.expm1(y_test)

        metrics = compute_metrics(y_actual.values, y_pred)
        cv_result = evaluate_model_cv(model, X_train_processed, y_train, cv=5)

        # Measure single-sample inference latency
        single_sample = X_test_processed[0:1]
        latency_runs = []
        for _ in range(100):
            t0 = time.perf_counter()
            model.predict(single_sample)
            latency_runs.append((time.perf_counter() - t0) * 1000)
        cv_result["inference_latency_ms"] = np.median(latency_runs)

        train_time = time.time() - model_start
        logger.info(
            "%s — R²: %.4f, MAE: ₹%.0f, RMSE: ₹%.0f, MAPE: %.2f%%, Time: %.1fs",
            name, metrics["R2"], metrics["MAE"], metrics["RMSE"],
            metrics["MAPE"], train_time
        )

        results.append({
            "name": name,
            "model": model,
            "metrics": metrics,
            "cv": cv_result,
            "train_time_seconds": train_time,
        })

    # ── 6. Compare and select best model ───────────────────────────
    print_comparison_table(results)
    best = select_best_model(results)

    logger.info("Selected: %s", best["name"])
    logger.info("  R²: %.4f", best["metrics"]["R2"])
    logger.info("  MAE: ₹%s", f"{best['metrics']['MAE']:,.0f}")
    logger.info("  RMSE: ₹%s", f"{best['metrics']['RMSE']:,.0f}")
    logger.info("  MAPE: %.2f%%", best["metrics"]["MAPE"])
    logger.info("  CV R² mean: %.4f ± %.4f", best["cv"]["cv_r2_mean"], best["cv"]["cv_r2_std"])

    # ── 7. Build final pipeline and refit on full training data ────
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", best["model"]),
    ])
    # Refit the full pipeline (preprocessor is already fitted, regressor refits)
    final_pipeline.fit(X_train, y_train)

    # ── 8. Serialize model and metadata ────────────────────────────
    models_dir = Path(Config.MODEL_PATH).parent
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_pipeline, Config.MODEL_PATH)
    logger.info("Model saved to %s", Config.MODEL_PATH)

    # Collect valid values from training data for inference validation
    valid_brands = sorted(df["brand"].unique().tolist())
    valid_models = sorted(df["model"].unique().tolist())
    brand_model_map = (
        df.groupby("brand")["model"]
        .apply(lambda x: sorted(x.unique().tolist()))
        .to_dict()
    )

    metadata = {
        "model_name": best["name"],
        "model_version": "1.0",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset_path": Config.DATASET_PATH,
        "dataset_rows": len(df),
        "n_features": len(Config.CATEGORICAL_FEATURES) + len(Config.NUMERICAL_FEATURES),
        "categorical_features": Config.CATEGORICAL_FEATURES,
        "numerical_features": Config.NUMERICAL_FEATURES,
        "target": Config.TARGET,
        "log_transform_target": True,
        "test_metrics": {k: round(v, 4) for k, v in best["metrics"].items()},
        "cv_r2_mean": round(best["cv"]["cv_r2_mean"], 4),
        "cv_r2_std": round(best["cv"]["cv_r2_std"], 4),
        "inference_latency_ms": round(best["cv"]["inference_latency_ms"], 3),
        "composite_score": round(best["composite_score"], 4),
        "train_time_seconds": round(best["train_time_seconds"], 2),
        "all_model_results": [
            {
                "name": r["name"],
                "metrics": {k: round(v, 4) for k, v in r["metrics"].items()},
                "cv_r2_mean": round(r["cv"]["cv_r2_mean"], 4),
                "cv_r2_std": round(r["cv"]["cv_r2_std"], 4),
                "composite_score": round(r.get("composite_score", 0), 4),
            }
            for r in results
        ],
        "valid_brands": valid_brands,
        "valid_models": valid_models,
        "brand_model_map": brand_model_map,
        "valid_fuel_types": Config.VALID_FUEL_TYPES,
        "valid_transmission_types": Config.VALID_TRANSMISSION_TYPES,
        "valid_seller_types": Config.VALID_SELLER_TYPES,
    }

    with open(Config.MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", Config.MODEL_METADATA_PATH)

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE in %.1f seconds", total_time)
    logger.info("=" * 60)

    return final_pipeline, metadata


if __name__ == "__main__":
    train_pipeline()
