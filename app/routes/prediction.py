"""
Prediction API routes.

POST /api/predict         — Make a price prediction
GET  /api/predictions     — List prediction history (paginated)
GET  /api/predictions/<id> — Get a single prediction
"""

import logging
from flask import Blueprint, request, jsonify

from app.utils.validation import validate_prediction_input
from app.utils.errors import APIError
from app.services.prediction_service import (
    make_prediction,
    get_prediction_history,
    get_prediction_by_id,
)

logger = logging.getLogger(__name__)

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.route("/api/predict", methods=["POST"])
def predict():
    """Handle a car price prediction request."""
    data = request.get_json(silent=True)

    if data is None:
        raise APIError("Request body must be valid JSON", 400)

    # Validate input
    errors = validate_prediction_input(data)
    if errors:
        return jsonify({"errors": errors}), 400

    try:
        result = make_prediction(data)
        logger.info("Prediction completed: ₹%,.0f", result["predicted_price"])
        return jsonify(result), 200

    except Exception as e:
        import traceback
        err_msg = str(e) + " | " + traceback.format_exc()
        logger.exception("Prediction failed: %s", e)
        raise APIError(f"Prediction failed: {err_msg}", 500)


@prediction_bp.route("/api/predictions", methods=["GET"])
def list_predictions():
    """Return paginated prediction history."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    per_page = min(per_page, 100)  # Cap to prevent abuse

    result = get_prediction_history(page=page, per_page=per_page)
    return jsonify(result), 200


@prediction_bp.route("/api/predictions/<int:prediction_id>", methods=["GET"])
def get_prediction(prediction_id):
    """Return a single prediction by ID."""
    result = get_prediction_by_id(prediction_id)
    if result is None:
        raise APIError(f"Prediction {prediction_id} not found", 404)
    return jsonify(result), 200
