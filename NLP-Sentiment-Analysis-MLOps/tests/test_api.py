"""
Integration tests for FastAPI endpoints.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Mock the inference engine for testing
class MockInferenceEngine:
    """Mock inference engine for testing."""

    def __init__(self, *args, **kwargs):
        pass

    def predict_single(self, text, return_probabilities=True):
        return {
            "text": text,
            "label": "positive" if len(text) > 10 else "negative",
            "confidence": 0.95,
            "probabilities": {"positive": 0.95, "negative": 0.05},
        }

    def predict(self, texts, return_probabilities=True):
        return [
            {
                "text": t,
                "label": "positive" if len(t) > 10 else "negative",
                "confidence": 0.95,
                "probabilities": {"positive": 0.95, "negative": 0.05},
            }
            for t in texts
        ]

    def explain_prediction(self, text):
        return {
            "text": text,
            "prediction": "positive",
            "confidence": 0.95,
            "tokens": text.split(),
            "probabilities": {"positive": 0.95, "negative": 0.05},
        }


@pytest.fixture
def client():
    """Create test client."""
    # Monkey patch for testing
    import serving.inference

    serving.inference.SentimentAnalysisInference = MockInferenceEngine

    from app.main import app

    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint(client):
    """Test single prediction endpoint."""
    response = client.post("/predict", json={"text": "This is a great movie!"})

    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert "text" in data


def test_batch_predict_endpoint(client):
    """Test batch prediction endpoint."""
    response = client.post(
        "/predict_batch",
        json={"texts": ["Great movie!", "Terrible experience", "Just okay"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 3


def test_batch_size_limit(client):
    """Test batch size limit."""
    texts = ["text"] * 101
    response = client.post("/predict_batch", json={"texts": texts})

    assert response.status_code == 400


def test_model_info_endpoint(client):
    """Test model info endpoint."""
    response = client.get("/model-info")

    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "labels" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
