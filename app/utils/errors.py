"""
Centralized error handlers.

Registers consistent JSON error responses for common HTTP errors.
Internal details are logged but never exposed to the client.
"""

import logging
from flask import jsonify

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Custom API error with status code and message."""

    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


def register_error_handlers(app):
    """Register global error handlers on the Flask app."""

    @app.errorhandler(APIError)
    def handle_api_error(error):
        return jsonify({"error": error.message}), error.status_code

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"error": "Bad request"}), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Resource not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(500)
    def internal_error(error):
        logger.exception("Internal server error: %s", error)
        return jsonify({"error": "Internal server error"}), 500
