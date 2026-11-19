"""Integration test that loads and calls the original model without mocks."""

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from src.backend.api import app


def test_real_model_predicts_for_uploaded_image():
    stream = BytesIO()
    Image.new("RGB", (256, 256), (180, 90, 40)).save(stream, format="JPEG")
    with TestClient(app) as client:
        assert client.get("/health").json()["available"] is True
        response = client.post("/predict", files={"image": ("sample.jpg", stream.getvalue(), "image/jpeg")})
        categories = client.get("/model/info").json()["categories"]
    assert response.status_code == 200
    result = response.json()
    assert result["category"] in categories
    assert result["state"] in {"Fresh", "Rotten"}
    assert isinstance(result["confidence"], float)
