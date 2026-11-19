"""HTTP client for the inference API."""

import os

import requests


def predict_image(image_bytes: bytes, filename: str) -> requests.Response:
    base_url = os.getenv("FRUITVISION_API_URL", "http://127.0.0.1:8000").rstrip("/")
    return requests.post(
        f"{base_url}/predict",
        files={"image": (filename, image_bytes)},
        timeout=120,
    )


def get_model_info() -> requests.Response:
    base_url = os.getenv("FRUITVISION_API_URL", "http://127.0.0.1:8000").rstrip("/")
    return requests.get(f"{base_url}/model/info", timeout=15)
