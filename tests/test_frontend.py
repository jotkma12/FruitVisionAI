from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
import requests
from streamlit.testing.v1 import AppTest

from src.frontend.client import predict_image


def png_bytes(color="orange"):
    stream = BytesIO()
    Image.new("RGB", (24, 24), color).save(stream, format="PNG")
    return stream.getvalue()


def api_response(status_code, payload):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def model_info_response(status_code=200):
    return api_response(
        status_code,
        {"model": "EfficientNetB0", "categories": ["FreshOrange", "RottenOrange"]},
    )


def app_script():
    return Path(__file__).resolve().parents[1] / "src" / "frontend" / "app.py"


def app_text(app):
    elements = list(app.markdown) + list(app.caption)
    return "\n".join(str(element.value) for element in elements)


def test_client_sends_uploaded_bytes_and_returns_http_response(monkeypatch):
    response = api_response(
        200,
        {"category": "FreshOrange", "state": "Fresh", "confidence": 98.5},
    )
    post = Mock(return_value=response)
    monkeypatch.setattr("src.frontend.client.requests.post", post)

    result = predict_image(b"selected image bytes", "orange.png")

    assert result is response
    post.assert_called_once_with(
        "http://127.0.0.1:8000/predict",
        files={"image": ("orange.png", b"selected image bytes")},
        timeout=120,
    )


def test_client_fetches_model_identity_from_existing_endpoint(monkeypatch):
    response = model_info_response()
    get = Mock(return_value=response)
    monkeypatch.setattr("src.frontend.client.requests.get", get)
    monkeypatch.setenv("FRUITVISION_API_URL", "http://api.local/")

    from src.frontend.client import get_model_info

    assert get_model_info() is response
    get.assert_called_once_with("http://api.local/model/info", timeout=15)


def test_ui_displays_successful_verification_and_api_response():
    image = png_bytes()
    prediction = api_response(
        200,
        {"category": "FreshOrange", "state": "Fresh", "confidence": 98.5},
    )
    with patch("src.frontend.client.get_model_info", return_value=model_info_response()) as info, patch(
        "src.frontend.client.predict_image", return_value=prediction
    ) as predict:
        app = AppTest.from_file(app_script()).run()
        app.file_uploader[0].set_value(("orange.png", image, "image/png")).run()
        app.button[0].click().run()

    info.assert_called_once_with()
    predict.assert_called_once_with(image, "orange.png")
    assert app.subheader[0].value == "FreshOrange"
    assert app.metric[0].value == "98.50%"
    assert any("Procesamiento completado" in item.value for item in app.success)
    text = app_text(app)
    assert "Solicitud enviada al backend FastAPI" in text
    assert "Respuesta HTTP recibida correctamente" in text
    assert "Predicción procesada con EfficientNetB0" in text
    assert "Resultado recibido y presentado en Streamlit" in text
    assert "POST /predict" in text and "200" in text
    assert "EfficientNetB0" in text
    assert "FreshOrange" in text and "Fresh" in text


def test_ui_rejects_invalid_image_without_calling_api():
    with patch("src.frontend.client.get_model_info") as info, patch(
        "src.frontend.client.predict_image"
    ) as predict:
        app = AppTest.from_file(app_script()).run()
        app.file_uploader[0].set_value(("notes.png", b"not an image", "image/png")).run()

    info.assert_not_called()
    predict.assert_not_called()
    assert any("No se pudo leer la imagen" in item.value for item in app.error)
    assert any("Error en el procesamiento" in item.value for item in app.error)
    assert "No se envió ninguna solicitud al backend" in app_text(app)


def test_connection_failure_clears_old_result_and_reports_unconfirmed_steps():
    first_image = png_bytes("orange")
    second_image = png_bytes("red")
    first_prediction = api_response(
        200,
        {"category": "FreshOrange", "state": "Fresh", "confidence": 98.5},
    )
    with patch(
        "src.frontend.client.get_model_info",
        side_effect=[model_info_response(), model_info_response()],
    ), patch(
        "src.frontend.client.predict_image",
        side_effect=[first_prediction, requests.ConnectionError("offline")],
    ) as predict:
        app = AppTest.from_file(app_script()).run()
        app.file_uploader[0].set_value(("first.png", first_image, "image/png")).run()
        app.button[0].click().run()
        assert app.subheader[0].value == "FreshOrange"

        app.file_uploader[0].set_value(("second.png", second_image, "image/png")).run()
        assert not app.subheader
        assert not app.metric
        app.button[0].click().run()

    assert predict.call_count == 2
    assert not app.metric
    assert any("Error en el procesamiento" in item.value for item in app.error)
    text = app_text(app)
    assert "No se recibió un código HTTP para /predict" in text
    assert "Predicción procesada con EfficientNetB0" in text
    assert "No se recibió respuesta" in text
    assert "FreshOrange" not in text


def test_selecting_different_file_with_same_bytes_clears_previous_result():
    image = png_bytes()
    prediction = api_response(
        200,
        {"category": "FreshOrange", "state": "Fresh", "confidence": 98.5},
    )
    with patch("src.frontend.client.get_model_info", return_value=model_info_response()), patch(
        "src.frontend.client.predict_image", return_value=prediction
    ):
        app = AppTest.from_file(app_script()).run()
        app.file_uploader[0].set_value(("first.png", image, "image/png")).run()
        app.button[0].click().run()
        assert app.subheader[0].value == "FreshOrange"

        app.file_uploader[0].set_value(("second.png", image, "image/png")).run()

    assert not app.subheader
    assert not app.metric
    assert not app.success
    assert not app.error


def test_api_category_and_state_are_displayed_from_the_same_response():
    prediction = api_response(
        200,
        {"category": "RottenOrange", "state": "Rotten", "confidence": 73.25},
    )
    with patch("src.frontend.client.get_model_info", return_value=model_info_response()), patch(
        "src.frontend.client.predict_image", return_value=prediction
    ):
        app = AppTest.from_file(app_script()).run()
        app.file_uploader[0].set_value(("orange.png", png_bytes(), "image/png")).run()
        app.button[0].click().run()

    assert app.subheader[0].value == "RottenOrange"
    assert app.metric[0].value == "73.25%"
    text = app_text(app)
    assert "RottenOrange" in text
    assert "(Rotten)" in text
    assert "FreshOrange" not in text


def test_http_error_shows_received_code_without_claiming_inference():
    error_response = api_response(503, {"detail": "Inference model is not available."})
    with patch("src.frontend.client.get_model_info", return_value=model_info_response()), patch(
        "src.frontend.client.predict_image", return_value=error_response
    ):
        app = AppTest.from_file(app_script()).run()
        app.file_uploader[0].set_value(("orange.png", png_bytes(), "image/png")).run()
        app.button[0].click().run()

    text = app_text(app)
    assert "POST /predict" in text and "503" in text
    assert "Predicción procesada con EfficientNetB0" in text
    assert "FreshOrange" not in text
    assert not app.metric
    assert any("Error en el procesamiento" in item.value for item in app.error)


def test_ui_reports_response_parsing_failure_without_a_result():
    prediction = api_response(200, ["invalid", "payload"])
    with patch("src.frontend.client.get_model_info", return_value=model_info_response()), patch(
        "src.frontend.client.predict_image", return_value=prediction
    ):
        app = AppTest.from_file(app_script()).run()
        app.file_uploader[0].set_value(("orange.png", png_bytes(), "image/png")).run()
        app.button[0].click().run()

    text = app_text(app)
    assert "POST /predict" in text and "200" in text
    assert "Respuesta HTTP recibida correctamente" in text
    assert "La respuesta no se pudo validar como una predicción del modelo" in text
    assert "Procesamiento completado" not in text
    assert not app.metric
