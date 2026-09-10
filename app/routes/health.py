"""
Health and model information routes.

GET /api/health     — App health check
GET /api/model-info — Model metadata and performance metrics
GET /api/analytics  — Prediction analytics dashboard data
"""

import logging
from flask import Blueprint, jsonify

from app.services.model_service import ModelService
from app.services.prediction_service import get_analytics

logger = logging.getLogger(__name__)

health_bp = Blueprint("health", __name__)


@health_bp.route("/api/health", methods=["GET"])
def health_check():
    """Return application health status."""
    model_service = ModelService.get_instance()
    return jsonify({
        "status": "healthy",
        "model_loaded": model_service.is_loaded,
    }), 200


@health_bp.route("/api/model-info", methods=["GET"])
def model_info():
    """Return model metadata including metrics, features, and training info."""
    model_service = ModelService.get_instance()

    if not model_service.is_loaded or not model_service.metadata:
        return jsonify({"error": "Model not loaded"}), 503

    meta = model_service.metadata
    return jsonify({
        "model_name": meta.get("model_name"),
        "model_version": meta.get("model_version"),
        "training_date": meta.get("training_date"),
        "dataset_rows": meta.get("dataset_rows"),
        "n_features": meta.get("n_features"),
        "test_metrics": meta.get("test_metrics"),
        "cv_r2_mean": meta.get("cv_r2_mean"),
        "cv_r2_std": meta.get("cv_r2_std"),
        "inference_latency_ms": meta.get("inference_latency_ms"),
        "all_model_results": meta.get("all_model_results"),
        "valid_brands": meta.get("valid_brands"),
        "valid_models": meta.get("valid_models"),
        "brand_model_map": meta.get("brand_model_map"),
        "valid_fuel_types": meta.get("valid_fuel_types"),
        "valid_transmission_types": meta.get("valid_transmission_types"),
        "valid_seller_types": meta.get("valid_seller_types"),
        "car_specs": meta.get("car_specs", {}),
    }), 200


@health_bp.route("/api/analytics", methods=["GET"])
def analytics():
    """Return prediction analytics for the dashboard."""
    result = get_analytics()
    return jsonify(result), 200
