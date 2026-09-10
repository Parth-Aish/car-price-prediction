"""
Tests for the Flask REST API endpoints.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app


@pytest.fixture
def app():
    """Create a test Flask application."""
    app = create_app("testing")
    yield app


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"

    def test_health_reports_model_status(self, client):
        response = client.get("/api/health")
        data = response.get_json()
        assert "model_loaded" in data


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        response = client.get("/api/model-info")
        assert response.status_code == 200

    def test_model_info_contains_metrics(self, client):
        response = client.get("/api/model-info")
        data = response.get_json()
        if "error" not in data:
            assert "model_name" in data
            assert "test_metrics" in data
            assert "valid_brands" in data


class TestPredictionEndpoint:
    VALID_INPUT = {
        "brand": "Maruti",
        "model": "Swift",
        "year": 2021,
        "kilometers_driven": 25000,
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "seller_type": "Individual",
        "engine": 1197,
        "mileage": 23.2,
        "power": 88.5,
        "seats": 5,
    }

    def test_valid_prediction_returns_200(self, client):
        response = client.post("/api/predict", json=self.VALID_INPUT)
        assert response.status_code == 200
        data = response.get_json()
        assert "predicted_price" in data
        assert data["predicted_price"] > 0
        assert data["currency"] == "INR"

    def test_prediction_returns_reasonable_price(self, client):
        response = client.post("/api/predict", json=self.VALID_INPUT)
        data = response.get_json()
        price = data["predicted_price"]
        assert 50_000 <= price <= 10_000_000, f"Price ₹{price} seems unreasonable for a Maruti Swift"

    def test_missing_fields_returns_400(self, client):
        incomplete = {"brand": "Maruti"}
        response = client.post("/api/predict", json=incomplete)
        assert response.status_code == 400

    def test_invalid_year_returns_400(self, client):
        bad_input = {**self.VALID_INPUT, "year": 1950}
        response = client.post("/api/predict", json=bad_input)
        assert response.status_code == 400

    def test_negative_km_returns_400(self, client):
        bad_input = {**self.VALID_INPUT, "kilometers_driven": -100}
        response = client.post("/api/predict", json=bad_input)
        assert response.status_code == 400

    def test_invalid_fuel_type_returns_400(self, client):
        bad_input = {**self.VALID_INPUT, "fuel_type": "Nuclear"}
        response = client.post("/api/predict", json=bad_input)
        assert response.status_code == 400

    def test_empty_body_returns_400(self, client):
        response = client.post("/api/predict", data="", content_type="application/json")
        assert response.status_code == 400

    def test_non_json_body_returns_400(self, client):
        response = client.post("/api/predict", data="not json")
        assert response.status_code == 400


class TestPredictionHistory:
    VALID_INPUT = {
        "brand": "Honda",
        "model": "City",
        "year": 2020,
        "kilometers_driven": 30000,
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "seller_type": "Individual",
        "engine": 1497,
        "mileage": 17.4,
        "power": 119.0,
        "seats": 5,
    }

    def test_history_returns_200(self, client):
        response = client.get("/api/predictions")
        assert response.status_code == 200
        data = response.get_json()
        assert "predictions" in data
        assert "total" in data

    def test_prediction_is_saved_and_retrievable(self, client):
        # Make a prediction
        client.post("/api/predict", json=self.VALID_INPUT)

        # Verify it appears in history
        response = client.get("/api/predictions")
        data = response.get_json()
        assert data["total"] >= 1
        assert len(data["predictions"]) >= 1

        # Verify it's retrievable by ID
        pred_id = data["predictions"][0]["id"]
        detail_response = client.get(f"/api/predictions/{pred_id}")
        assert detail_response.status_code == 200

    def test_nonexistent_prediction_returns_404(self, client):
        response = client.get("/api/predictions/99999")
        assert response.status_code in (404, 400)


class TestAnalyticsEndpoint:
    def test_analytics_returns_200(self, client):
        response = client.get("/api/analytics")
        assert response.status_code == 200
        data = response.get_json()
        assert "total_predictions" in data
        assert "average_price" in data


class TestFrontend:
    def test_index_page_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"CarPrice AI" in response.data
