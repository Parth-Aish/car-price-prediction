"""
Flask application factory.

Creates and configures the Flask app:
- Loads configuration
- Initializes database
- Loads ML model once at startup
- Registers blueprints and error handlers
- Sets up structured logging
"""

import logging
import os
from pathlib import Path

from flask import Flask, render_template
from flask_cors import CORS

from config import Config, config_by_name
from app.models.database import init_db
from app.services.model_service import ModelService
from app.utils.errors import register_error_handlers
from app.utils.validation import init_valid_values


def create_app(config_name: str = None) -> Flask:
    """Create and configure the Flask application."""

    if config_name is None:
        config_name = os.getenv("FLASK_ENV", "development")

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent.parent / "templates"),
        static_folder=str(Path(__file__).resolve().parent.parent / "static"),
    )

    app_config = config_by_name.get(config_name, Config)
    app.config.from_object(app_config)

    # ── Logging ──────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, app.config.get("LOG_LEVEL", "INFO")),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Creating Flask app with config: %s", config_name)

    # ── Extensions ───────────────────────────────────────────────
    CORS(app)
    init_db(app)
    register_error_handlers(app)

    # ── Load ML model (once at startup) ──────────────────────────
    model_service = ModelService.get_instance()
    try:
        model_path = app.config.get("MODEL_PATH", Config.MODEL_PATH)
        metadata_path = app.config.get("MODEL_METADATA_PATH", Config.MODEL_METADATA_PATH)
        model_service.load(model_path, metadata_path)
        init_valid_values(model_service.metadata)
        logger.info("ML model loaded successfully at startup")
    except FileNotFoundError:
        logger.warning(
            "Model files not found. Run 'python -m ml.train' to train the model. "
            "The app will start but predictions will fail."
        )
    except Exception as e:
        logger.error("Failed to load ML model: %s", e)

    # ── Register blueprints ──────────────────────────────────────
    from app.routes.prediction import prediction_bp
    from app.routes.health import health_bp

    app.register_blueprint(prediction_bp)
    app.register_blueprint(health_bp)

    # ── Frontend route ───────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html")

    logger.info("Flask app created successfully")
    return app
