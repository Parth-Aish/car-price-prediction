"""
Prediction service — orchestrates input transformation, model inference, and DB storage.

Keeps route handlers thin by encapsulating business logic here.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from app.models.database import db
from app.models.prediction import Prediction
from app.services.model_service import ModelService
from ml.feature_engineering import prepare_features_for_prediction

logger = logging.getLogger(__name__)


def make_prediction(input_data: dict) -> dict:
    """
    Full prediction flow:
    1. Transform API input to model-ready DataFrame
    2. Run inference
    3. Save to database
    4. Return formatted result
    """
    model_service = ModelService.get_instance()

    # Prepare features (year → vehicle_age, add engineered features)
    features_df = prepare_features_for_prediction(input_data.copy())
    logger.info("Features prepared for prediction: %s", features_df.columns.tolist())

    # Run prediction
    predicted_price = model_service.predict(features_df)
    logger.info("Predicted price: ₹%s", format(predicted_price, ",.0f"))

    # Save to database
    prediction_record = Prediction(
        brand=input_data["brand"],
        model=input_data["model"],
        year=int(input_data["year"]),
        kilometers_driven=int(input_data["kilometers_driven"]),
        fuel_type=input_data["fuel_type"],
        transmission=input_data["transmission"],
        seller_type=input_data.get("seller_type", "Individual"),
        engine=int(input_data["engine"]),
        mileage=float(input_data["mileage"]),
        power=float(input_data["power"]),
        seats=int(input_data["seats"]),
        predicted_price=predicted_price,
    )
    db.session.add(prediction_record)
    db.session.commit()
    logger.info("Prediction saved to database with id=%d", prediction_record.id)

    return {
        "predicted_price": predicted_price,
        "currency": "INR",
        "model_version": model_service.metadata.get("model_version", "unknown"),
        "prediction_id": prediction_record.id,
        "timestamp": prediction_record.timestamp.isoformat(),
    }


def get_prediction_history(page: int = 1, per_page: int = 20) -> dict:
    """Get paginated prediction history, newest first."""
    pagination = (
        Prediction.query
        .order_by(Prediction.timestamp.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )

    return {
        "predictions": [p.to_dict() for p in pagination.items],
        "total": pagination.total,
        "page": pagination.page,
        "per_page": pagination.per_page,
        "pages": pagination.pages,
    }


def get_prediction_by_id(prediction_id: int) -> dict | None:
    """Get a single prediction by ID."""
    prediction = db.session.get(Prediction, prediction_id)
    if prediction is None:
        return None
    return prediction.to_dict()


def get_analytics() -> dict:
    """Compute summary analytics from prediction history."""
    from sqlalchemy import func

    total = Prediction.query.count()

    if total == 0:
        return {
            "total_predictions": 0,
            "average_price": 0,
            "most_predicted_brand": None,
            "recent_prediction": None,
            "price_range": {"min": 0, "max": 0},
        }

    avg_price = db.session.query(func.avg(Prediction.predicted_price)).scalar()
    min_price = db.session.query(func.min(Prediction.predicted_price)).scalar()
    max_price = db.session.query(func.max(Prediction.predicted_price)).scalar()

    # Most predicted brand
    brand_counts = (
        db.session.query(Prediction.brand, func.count(Prediction.id))
        .group_by(Prediction.brand)
        .order_by(func.count(Prediction.id).desc())
        .first()
    )

    # Most recent prediction
    recent = Prediction.query.order_by(Prediction.timestamp.desc()).first()

    return {
        "total_predictions": total,
        "average_price": round(avg_price, 2) if avg_price else 0,
        "most_predicted_brand": brand_counts[0] if brand_counts else None,
        "most_predicted_brand_count": brand_counts[1] if brand_counts else 0,
        "recent_prediction": recent.to_dict() if recent else None,
        "price_range": {
            "min": round(min_price, 2) if min_price else 0,
            "max": round(max_price, 2) if max_price else 0,
        },
    }
