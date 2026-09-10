"""
Prediction SQLAlchemy model.

Stores every prediction request and result for history and analytics.
"""

from datetime import datetime, timezone
from app.models.database import db


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    timestamp = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    # Input features
    brand = db.Column(db.String(50), nullable=False, index=True)
    model = db.Column(db.String(100), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    kilometers_driven = db.Column(db.Integer, nullable=False)
    fuel_type = db.Column(db.String(20), nullable=False)
    transmission = db.Column(db.String(20), nullable=False)
    seller_type = db.Column(db.String(30), nullable=True)
    engine = db.Column(db.Integer, nullable=False)
    mileage = db.Column(db.Float, nullable=False)
    power = db.Column(db.Float, nullable=False)
    seats = db.Column(db.Integer, nullable=False)

    # Output
    predicted_price = db.Column(db.Float, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "brand": self.brand,
            "model": self.model,
            "year": self.year,
            "kilometers_driven": self.kilometers_driven,
            "fuel_type": self.fuel_type,
            "transmission": self.transmission,
            "seller_type": self.seller_type,
            "engine": self.engine,
            "mileage": self.mileage,
            "power": self.power,
            "seats": self.seats,
            "predicted_price": self.predicted_price,
        }
