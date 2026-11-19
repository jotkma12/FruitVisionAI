"""Interfaz Streamlit de FruitVision AI con trazabilidad de la inferencia."""

import hashlib
import math
from io import BytesIO
from typing import Any

from PIL import Image, UnidentifiedImageError
import requests
import streamlit as st

from src.frontend.client import get_model_info, predict_image


ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
EXPECTED_MODEL = "EfficientNetB0"
STATUS_ICON = {
    "complete": "✅",
    "failed": "❌",
    "unconfirmed": "⚠️",
    "pending": "⏳",
    "not_run": "—",
}


def image_validation_error(filename: str, content: bytes) -> str | None:
    if not content:
        return "El archivo está vacío. Selecciona una imagen JPEG, PNG o WEBP."
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        return "Formato no admitido. Selecciona una imagen JPEG, PNG o WEBP."
    try:
        with Image.open(BytesIO(content)) as image:
            if image.format not in ALLOWED_FORMATS:
                return "El contenido no corresponde a una imagen JPEG, PNG o WEBP."
            image.verify()
    except (UnidentifiedImageError, OSError):
        return "No se pudo leer la imagen. Selecciona un archivo válido."
    return None


def new_verification() -> dict[str, Any]:
    return {
        "completed": False,
        "error": None,
        "http_codes": {},
        "model_name": None,
        "category": None,
        "state": None,
        "steps": [
            {
                "label": "Imagen recibida por Streamlit",
                "status": "complete",
                "detail": "El archivo seleccionado llegó a la interfaz.",
            },
            {
                "label": "Solicitud enviada al backend FastAPI",
                "status": "pending",
                "detail": "Aún no se ha enviado la solicitud de predicción.",
            },
            {
                "label": "Respuesta HTTP recibida correctamente",
                "status": "pending",
                "detail": "Aún no se ha recibido la respuesta de /predict.",
            },
            {
                "label": "Predicción procesada con EfficientNetB0",
                "status": "pending",
                "detail": "La inferencia todavía no está confirmada.",
            },
            {
                "label": "Resultado recibido y presentado en Streamlit",
                "status": "pending",
                "detail": "Todavía no hay un resultado para presentar.",
            },
        ],
    }


def set_step(
    verification: dict[str, Any], index: int, status: str, detail: str
) -> None:
    verification["steps"][index]["status"] = status
    verification["steps"][index]["detail"] = detail


def render_verification(verification: dict[str, Any]) -> None:
    with st.container(border=True):
        st.subheader("Verificación del procesamiento")
        if verification["completed"]:
            st.success("Procesamiento completado")
        else:
            st.error("Error en el procesamiento")

        for step in verification["steps"]:
            icon = STATUS_ICON[step["status"]]
            st.write(f"{icon} **{step['label']}** — {step['detail']}")

        if verification["http_codes"]:
            st.write("**Códigos HTTP recibidos:**")
            for endpoint, status_code in verification["http_codes"].items():
                st.write(f"- `{endpoint}`: `{status_code}`")

        if verification["model_name"]:
            st.write(f"**Modelo reportado por la API:** {verification['model_name']}")
        if verification["category"] and verification["state"]:
            st.write(
                f"**Respuesta de la API:** {verification['category']} "
                f"({verification['state']})"
            )
        if verification["error"]:
            st.caption(f"Detalle: {verification['error']}")


def parse_model_info(response: requests.Response) -> tuple[str, list[str]]:
    if not 200 <= response.status_code < 300:
        raise requests.HTTPError(
            f"GET /model/info devolvió HTTP {response.status_code}.", response=response
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("GET /model/info devolvió una respuesta JSON inválida.")
    model_name = payload.get("model")
    categories = payload.get("categories")
    if not isinstance(model_name, str) or not isinstance(categories, list):
        raise ValueError("La respuesta de /model/info no contiene modelo y categorías válidos.")
    if not all(isinstance(category, str) for category in categories):
        raise ValueError("La lista de categorías devuelta por la API no es válida.")
    if model_name != EXPECTED_MODEL:
        raise ValueError(f"La API reportó un modelo inesperado: {model_name}.")
    return model_name, categories


def parse_prediction(payload: Any, categories: list[str]) -> dict[str, str | float]:
    if not isinstance(payload, dict):
        raise ValueError("POST /predict devolvió una respuesta JSON inválida.")
    category = payload.get("category")
    state = payload.get("state")
    confidence = payload.get("confidence")
    if not isinstance(category, str) or category not in categories:
        raise ValueError("La categoría de la respuesta no coincide con /model/info.")
    expected_state = "Fresh" if category.startswith("Fresh") else "Rotten" if category.startswith("Rotten") else None
    if expected_state is None or state != expected_state:
        raise ValueError("El estado de la respuesta no coincide con la categoría.")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 100
    ):
        raise ValueError("La puntuación de confianza de la respuesta no es válida.")
    return {"category": category, "state": state, "confidence": float(confidence)}


def process_prediction(image_bytes: bytes, filename: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    verification = new_verification()
    active_endpoint = "GET /model/info"

    try:
        info_response = get_model_info()
        verification["http_codes"][active_endpoint] = info_response.status_code
        model_name, categories = parse_model_info(info_response)
        verification["model_name"] = model_name

        active_endpoint = "POST /predict"
        prediction_response = predict_image(image_bytes, filename)
        verification["http_codes"][active_endpoint] = prediction_response.status_code
        set_step(
            verification,
            1,
            "complete",
            "FastAPI respondió a la solicitud de predicción.",
        )
        if not 200 <= prediction_response.status_code < 300:
            set_step(
                verification,
                2,
                "failed",
                f"Se recibió una respuesta HTTP de error ({prediction_response.status_code}).",
            )
            raise requests.HTTPError(
                f"POST /predict devolvió HTTP {prediction_response.status_code}.",
                response=prediction_response,
            )

        set_step(
            verification,
            2,
            "complete",
            f"FastAPI devolvió HTTP {prediction_response.status_code}.",
        )
        prediction = parse_prediction(prediction_response.json(), categories)
        verification["category"] = prediction["category"]
        verification["state"] = prediction["state"]
        set_step(
            verification,
            3,
            "complete",
            "La respuesta válida de /predict corresponde al modelo reportado por la API.",
        )

        result = prediction
        verification["completed"] = True
        set_step(
            verification,
            4,
            "complete",
            "La respuesta fue recibida y se presentará en esta interfaz.",
        )
        return result, verification

    except (requests.ConnectionError, requests.Timeout) as exc:
        if active_endpoint == "POST /predict":
            set_step(
                verification,
                1,
                "unconfirmed",
                "No se recibió respuesta; la recepción por FastAPI no puede confirmarse.",
            )
            set_step(
                verification,
                2,
                "unconfirmed",
                "No se recibió un código HTTP para /predict.",
            )
            set_step(
                verification,
                3,
                "unconfirmed",
                "La inferencia no puede confirmarse sin respuesta de /predict.",
            )
        else:
            set_step(
                verification,
                1,
                "not_run",
                "No se envió /predict porque no se pudo consultar /model/info.",
            )
            set_step(
                verification,
                2,
                "not_run",
                "No se solicitó una predicción al backend.",
            )
            set_step(
                verification,
                3,
                "not_run",
                "No se confirmó ninguna inferencia.",
            )
        set_step(verification, 4, "not_run", "No se recibió ningún resultado.")
        verification["error"] = (
            f"No se pudo conectar con {active_endpoint}. Comprueba que la API esté iniciada. "
            f"Detalle: {exc}"
        )
        return None, verification

    except requests.HTTPError as exc:
        response = exc.response
        status_code = response.status_code if response is not None else None
        if status_code is not None:
            verification["http_codes"][active_endpoint] = status_code
        if active_endpoint == "POST /predict":
            set_step(
                verification,
                1,
                "complete",
                "La respuesta confirma que FastAPI recibió /predict.",
            )
            if verification["steps"][2]["status"] == "pending":
                set_step(
                    verification,
                    2,
                    "failed",
                    f"Se recibió HTTP {status_code}; la predicción no fue aceptada.",
                )
            set_step(
                verification,
                3,
                "failed",
                "FastAPI no devolvió una predicción satisfactoria.",
            )
        else:
            set_step(
                verification,
                1,
                "not_run",
                "No se envió /predict porque falló la consulta de /model/info.",
            )
            set_step(
                verification,
                2,
                "not_run",
                "No se solicitó una predicción al backend.",
            )
            set_step(
                verification,
                3,
                "not_run",
                "La inferencia no fue confirmada.",
            )
        set_step(verification, 4, "not_run", "No se recibió ningún resultado.")
        verification["error"] = str(exc)
        return None, verification

    except Exception as exc:
        if active_endpoint == "POST /predict":
            status_code = verification["http_codes"].get(active_endpoint)
            if status_code is not None:
                set_step(
                    verification,
                    1,
                    "complete",
                    "FastAPI respondió a la solicitud de predicción.",
                )
            if status_code is not None and 200 <= status_code < 300:
                set_step(
                    verification,
                    2,
                    "complete",
                    f"Se recibió HTTP {status_code}; la respuesta se validó como HTTP satisfactoria.",
                )
                set_step(
                    verification,
                    3,
                    "failed",
                    "La respuesta no se pudo validar como una predicción del modelo.",
                )
            elif status_code is not None:
                set_step(
                    verification,
                    2,
                    "failed",
                    f"Se recibió una respuesta HTTP de error ({status_code}).",
                )
                set_step(verification, 3, "failed", "La inferencia no se confirmó.")
            else:
                set_step(
                    verification,
                    1,
                    "unconfirmed",
                    "No se recibió respuesta; la recepción por FastAPI no puede confirmarse.",
                )
                set_step(
                    verification,
                    2,
                    "unconfirmed",
                    "No se recibió un código HTTP para /predict.",
                )
                set_step(
                    verification,
                    3,
                    "unconfirmed",
                    "La inferencia no puede confirmarse sin respuesta de /predict.",
                )
        else:
            set_step(
                verification,
                1,
                "not_run",
                "No se envió /predict porque no se pudo validar /model/info.",
            )
            set_step(
                verification,
                2,
                "not_run",
                "No se solicitó una predicción al backend.",
            )
            set_step(
                verification,
                3,
            "not_run",
            "No se envió /predict; no se confirmó una inferencia con EfficientNetB0.",
            )
        set_step(verification, 4, "not_run", "No se recibió ningún resultado.")
        verification["error"] = f"No se pudo verificar el procesamiento. Detalle: {exc}"
        return None, verification


st.set_page_config(page_title="FruitVision AI", page_icon="🍎")
st.title("FruitVision AI")
st.write("Carga una fotografía para analizar su categoría y estado con el modelo entrenado.")

uploaded = st.file_uploader("Selecciona una fotografía", type=["jpg", "jpeg", "png", "webp"])
if "result_key" not in st.session_state:
    st.session_state.result_key = None
if "result" not in st.session_state:
    st.session_state.result = None
if "verification" not in st.session_state:
    st.session_state.verification = None

if uploaded is None:
    st.session_state.result_key = None
    st.session_state.result = None
    st.session_state.verification = None
else:
    image_bytes = uploaded.getvalue()
    # Include the selected filename so replacing the upload with another file
    # clears the previous result even when both files contain identical bytes.
    image_key = (uploaded.name, hashlib.sha256(image_bytes).hexdigest())
    if st.session_state.result_key != image_key:
        st.session_state.result_key = image_key
        st.session_state.result = None
        st.session_state.verification = None

    validation_error = image_validation_error(uploaded.name, image_bytes)
    if validation_error:
        st.session_state.result = None
        verification = new_verification()
        verification["error"] = validation_error
        set_step(
            verification,
            0,
            "complete",
            f"El archivo llegó a Streamlit, pero no se aceptó: {validation_error}",
        )
        set_step(
            verification,
            1,
            "not_run",
            "No se envió ninguna solicitud al backend.",
        )
        set_step(
            verification,
            2,
            "not_run",
            "No se recibió respuesta HTTP porque no se llamó a la API.",
        )
        set_step(
            verification,
            3,
            "not_run",
            "La inferencia no se ejecutó.",
        )
        set_step(
            verification,
            4,
            "not_run",
            "No se recibió ningún resultado.",
        )
        st.session_state.verification = verification
        st.error(validation_error)
        render_verification(verification)
    else:
        st.image(image_bytes, caption=uploaded.name, use_container_width=True)

        if st.button("Analizar imagen", type="primary"):
            st.session_state.result = None
            result, verification = process_prediction(image_bytes, uploaded.name)
            st.session_state.result = result
            st.session_state.verification = verification

        result = st.session_state.result
        verification = st.session_state.verification
        if result and verification and verification["completed"]:
            st.subheader(result["category"])
            st.write(f"Estado: {result['state']}")
            st.metric("Confianza del modelo", f"{result['confidence']:.2f}%")
            st.caption(
                "Porcentaje de salida del modelo para esta imagen; "
                "no representa certeza absoluta ni rendimiento global."
            )
        if verification:
            render_verification(verification)
