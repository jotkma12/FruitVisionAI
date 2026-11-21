"""Integration test that loads and calls the original model without mocks."""

from io import BytesIO
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from src.backend.api import app
from src.backend.config import get_settings
from src.frontend import app as frontend_app
from src.frontend import client as frontend_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def image_bytes():
    stream = BytesIO()
    Image.new("RGB", (256, 256), (180, 90, 40)).save(stream, format="JPEG")
    return stream.getvalue()


def test_real_model_predicts_through_frontend_client_and_api(monkeypatch):
    settings = get_settings()
    expected_model = PROJECT_ROOT / "models" / "fruitvision_ai_model.keras"
    expected_classes = PROJECT_ROOT / "models" / "class_names.json"
    assert settings.model_path.resolve() == expected_model.resolve()
    assert settings.class_names_path.resolve() == expected_classes.resolve()
    with expected_classes.open("r", encoding="utf-8") as class_file:
        configured_categories = json.load(class_file)

    with TestClient(app) as client:
        assert client.get("/health").json()["available"] is True
        model_info = client.get("/model/info")
        assert model_info.status_code == 200
        assert model_info.json()["model"] == "EfficientNetB0"
        assert model_info.json()["categories"] == configured_categories
        assert len(configured_categories) == 20

        def local_get(url, timeout):
            assert url.endswith("/model/info")
            return client.get("/model/info")

        def local_post(url, files, timeout):
            assert url.endswith("/predict")
            return client.post("/predict", files=files)

        monkeypatch.setattr("src.frontend.client.requests.get", local_get)
        monkeypatch.setattr("src.frontend.client.requests.post", local_post)
        monkeypatch.setattr(frontend_app, "get_model_info", frontend_client.get_model_info)
        monkeypatch.setattr(frontend_app, "predict_image", frontend_client.predict_image)

        result, verification = frontend_app.process_prediction(image_bytes(), "sample.jpg")

        assert verification["completed"] is True
        assert verification["http_codes"]["GET /model/info"] == 200
        assert verification["http_codes"]["POST /predict"] == 200
        assert result is not None
        category = result["category"]
        assert category in configured_categories
        expected_state = "Fresh" if category.startswith("Fresh") else "Rotten"
        assert result["state"] == expected_state
        confidence = result["confidence"]
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 100

        invalid_response = client.post(
            "/predict", files={"image": ("invalid.jpg", b"not an image", "image/jpeg")}
        )
        assert invalid_response.status_code == 400
