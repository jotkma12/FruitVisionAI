from io import BytesIO
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.backend.api import app
from src.backend.config import get_settings


class FakeModel:
    def predict(self, batch, verbose=0):
        return np.asarray([[0.9] + [0.1 / 19] * 19], dtype=np.float32)


def fake_load(self):
    self.model = FakeModel()
    self.class_names = ["FreshApple"] + [f"RottenItem{i}" for i in range(19)]


def image_bytes():
    stream = BytesIO()
    Image.new("RGB", (20, 20), "green").save(stream, format="PNG")
    return stream.getvalue()


def test_health_and_info_report_loaded_model():
    with patch("src.backend.api.ModelService.load", autospec=True, side_effect=fake_load) as load_model, TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ready", "available": True}
        info = client.get("/model/info").json()
        assert info["model"] == "EfficientNetB0"
        assert len(info["categories"]) == 20
        assert load_model.call_count == 1


def test_predict_returns_model_response_for_upload():
    with patch("src.backend.api.ModelService.load", fake_load), TestClient(app) as client:
        response = client.post("/predict", files={"image": ("sample.png", image_bytes(), "image/png")})
        assert response.status_code == 200
        assert response.json() == {"category": "FreshApple", "state": "Fresh", "confidence": 90.0}


def test_predict_rejects_unsupported_and_empty_uploads():
    with patch("src.backend.api.ModelService.load", fake_load), TestClient(app) as client:
        unsupported = client.post("/predict", files={"image": ("sample.bmp", b"bmp", "image/bmp")})
        empty = client.post("/predict", files={"image": ("empty.png", b"", "image/png")})
        assert unsupported.status_code == 400
        assert empty.status_code == 400


def test_predict_rejects_upload_over_configured_limit():
    settings = get_settings()
    settings = settings.__class__(settings.model_path, settings.class_names_path, max_upload_bytes=10)
    with patch("src.backend.api.get_settings", return_value=settings), patch(
        "src.backend.api.ModelService.load", fake_load
    ), TestClient(app) as client:
        response = client.post("/predict", files={"image": ("sample.png", image_bytes(), "image/png")})
        assert response.status_code == 413


def test_failed_model_load_never_reports_ready_or_predicts():
    with patch("src.backend.api.ModelService.load", side_effect=RuntimeError("model file unavailable")), TestClient(app) as client:
        assert client.get("/health").json() == {"status": "unavailable", "available": False}
        assert client.get("/model/info").status_code == 503
        response = client.post("/predict", files={"image": ("sample.png", image_bytes(), "image/png")})
        assert response.status_code == 503
