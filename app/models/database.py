"""
SQLAlchemy database instance.

Separated from the app factory to avoid circular imports.
Designed for easy migration from SQLite to PostgreSQL — just change DATABASE_URL.
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_db(app):
    """Initialize the database and create all tables."""
    db.init_app(app)
    with app.app_context():
        import os
        os.makedirs(app.instance_path, exist_ok=True)
        # Import models here so SQLAlchemy knows about them before create_all
        from app.models.prediction import Prediction
        db.create_all()
