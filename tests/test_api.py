import os
import sys
import pytest
from fastapi.testclient import TestClient
from src.app import app

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction(client):
    """Test the prediction endpoint with valid data."""
    payload = {
        "pickup_datetime": "2024-01-15T09:30:00",
        "pickup_longitude": -73.9854,
        "pickup_latitude": 40.7488,
        "dropoff_longitude": -73.9780,
        "dropoff_latitude": 40.7648,
        "passenger_count": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_fare" in response.json()
    assert isinstance(response.json()["predicted_fare"], float)


def test_prediction_invalid_input(client):
    """Test the prediction endpoint with invalid data."""
    payload = {
        "pickup_datetime": "2024-01-15T09:30:00",
        "pickup_longitude": -73.9854,
        "pickup_latitude": 40.7488,
        "dropoff_longitude": -73.9780,
        "dropoff_latitude": 40.7648,
        "passenger_count": 99 
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
