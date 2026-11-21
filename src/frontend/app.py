"""Interfaz Streamlit de FruitVision AI con trazabilidad de la inferencia."""

import hashlib
import html
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
import requests
import streamlit as st

from src.frontend.client import get_health, get_model_info, predict_image


ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
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
    if not isinstance(model_name, str) or not model_name or not isinstance(categories, list) or not categories:
        raise ValueError("La respuesta de /model/info no contiene modelo y categorías válidos.")
    if not all(isinstance(category, str) for category in categories):
        raise ValueError("La lista de categorías devuelta por la API no es válida.")
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


def process_prediction(
    image_bytes: bytes,
    filename: str,
    model_info_data: tuple[str, list[str]] | None = None,
    model_info_status_code: int | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    verification = new_verification()
    active_endpoint = "GET /model/info"

    try:
        if model_info_data is None:
            info_response = get_model_info()
            verification["http_codes"][active_endpoint] = info_response.status_code
            model_name, categories = parse_model_info(info_response)
        else:
            model_name, categories = model_info_data
            verification["http_codes"][active_endpoint] = model_info_status_code or 200
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


FRUIT_CATALOG = [
    ("Manzana", "Apple"),
    ("Banano", "Banana"),
    ("Pimentón", "Bellpepper"),
    ("Zanahoria", "Carrot"),
    ("Pepino", "Cucumber"),
    ("Mango", "Mango"),
    ("Naranja", "Orange"),
    ("Papa", "Potato"),
    ("Fresa", "Strawberry"),
    ("Tomate", "Tomato"),
]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"


def load_catalog_categories() -> list[str]:
    try:
        with CLASS_NAMES_PATH.open("r", encoding="utf-8") as class_file:
            categories = json.load(class_file)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(categories, list) or not all(isinstance(item, str) for item in categories):
        return []
    return categories


@st.cache_data(ttl=5, show_spinner=False)
def fetch_system_info() -> dict[str, Any]:
    system_info: dict[str, Any] = {
        "connected": False,
        "available": False,
        "health_status": None,
        "health_status_code": None,
        "model_name": None,
        "categories": [],
        "model_info_status_code": None,
        "error": None,
    }
    try:
        health_response = get_health()
        system_info["health_status_code"] = health_response.status_code
        system_info["connected"] = True
        if not 200 <= health_response.status_code < 300:
            raise requests.HTTPError(
                f"GET /health devolvió HTTP {health_response.status_code}.",
                response=health_response,
            )
        health_payload = health_response.json()
        if (
            not isinstance(health_payload, dict)
            or not isinstance(health_payload.get("available"), bool)
            or not isinstance(health_payload.get("status"), str)
        ):
            raise ValueError("La respuesta de /health no tiene el formato esperado.")

        system_info["available"] = health_payload["available"]
        system_info["health_status"] = health_payload["status"]
        if health_payload["available"]:
            info_response = get_model_info()
            system_info["model_info_status_code"] = info_response.status_code
            model_name, categories = parse_model_info(info_response)
            system_info["model_name"] = model_name
            system_info["categories"] = categories
    except (requests.ConnectionError, requests.Timeout) as exc:
        system_info["error"] = f"No se pudo conectar con el backend: {exc}"
    except Exception as exc:
        system_info["error"] = f"No se pudo consultar el estado del backend: {exc}"
    return system_info


def category_pill(category: str) -> str:
    prefix = "Fresh" if category.startswith("Fresh") else "Rotten"
    label = html.escape(category[len(prefix) :])
    color = "fresh" if prefix == "Fresh" else "rotten"
    return f'<span class="class-pill {color}"><span>{prefix}</span><wbr>{label}</span>'


def render_catalog(categories: list[str]) -> None:
    st.markdown("### Catálogo de frutas y verduras")
    st.caption("Clases que reconoce el modelo. Las etiquetas provienen del archivo de categorías.")
    fruit_columns = st.columns(2, gap="small")
    for index, (display_name, class_suffix) in enumerate(FRUIT_CATALOG):
        fresh_category = f"Fresh{class_suffix}"
        rotten_category = f"Rotten{class_suffix}"
        labels = []
        if fresh_category in categories:
            labels.append(category_pill(fresh_category))
        if rotten_category in categories:
            labels.append(category_pill(rotten_category))
        label_content = "".join(labels) or (
            '<span class="catalog-unavailable">Categorías no disponibles</span>'
        )
        fruit_columns[index % 2].markdown(
            f'<div class="fruit-card"><strong>{html.escape(display_name)}</strong>'
            f'<div class="class-pills">{label_content}</div></div>',
            unsafe_allow_html=True,
        )


def render_system_panel(system_info: dict[str, Any]) -> None:
    st.markdown("### Estado del sistema y API")
    st.caption("Información consultada en vivo desde FastAPI.")

    connected = system_info["connected"]
    available = system_info["available"]
    connection_label = "Conectado" if connected else "Desconectado"
    availability_label = (
        system_info["health_status"]
        if connected and system_info["health_status"]
        else "No disponible"
    )
    architecture = system_info["model_name"] or "No disponible"
    category_count = (
        len(system_info["categories"]) if system_info["categories"] else "No disponible"
    )
    connection_class = "online" if connected else "offline"
    availability_class = "online" if available else "offline"

    status_columns = st.columns(4)
    status_items = [
        ("Conexión", connection_label, connection_class),
        ("Disponibilidad", availability_label, availability_class),
        ("Arquitectura", architecture, "neutral"),
        ("Categorías", str(category_count), "neutral"),
    ]
    for column, (label, value, state_class) in zip(status_columns, status_items):
        with column:
            st.markdown(
                f'<div class="system-card"><span>{html.escape(label)}</span>'
                f'<strong class="{state_class}">{html.escape(value)}</strong></div>',
                unsafe_allow_html=True,
            )

    st.markdown("#### Flujo de inferencia")
    flow_columns = st.columns([1, 0.18, 1, 0.18, 1])
    flow_items = [
        (0, "Streamlit", "Interfaz activa", "online"),
        (2, "FastAPI", connection_label, connection_class),
        (
            4,
            system_info["model_name"] or "Modelo Keras",
            "Disponible" if available else "No disponible",
            availability_class,
        ),
    ]
    for column_index, title, subtitle, state_class in flow_items:
        with flow_columns[column_index]:
            st.markdown(
                f'<div class="flow-node"><strong>{html.escape(title)}</strong>'
                f'<span class="{state_class}">{html.escape(subtitle)}</span></div>',
                unsafe_allow_html=True,
            )
    for column_index in (1, 3):
        with flow_columns[column_index]:
            st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)

    endpoint_columns = st.columns(3)
    for column, endpoint in zip(
        endpoint_columns, ("GET /health", "GET /model/info", "POST /predict")
    ):
        with column:
            st.caption(f"`{endpoint}`")
    if system_info["error"]:
        st.warning(system_info["error"])


st.set_page_config(page_title="FruitVision AI", page_icon="🍎", layout="wide")
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
        background: #101B17; color: #D4E3D9;
    }
    [data-testid="stHeader"] { background: rgba(16, 27, 23, 0.94); }
    [data-testid="stToolbar"] { background: transparent; }
    .block-container {
        max-width: 100%; padding: 0.35rem clamp(1.2rem, 2vw, 2.5rem) 0.65rem;
    }
    h1, h2, h3, h4 { color: #FFFFFF; letter-spacing: -0.025em; }
    h1 {
        display: block !important; visibility: visible !important; overflow: visible !important;
        color: #FFFFFF !important; font-size: clamp(2.3rem, 2.8vw, 2.8rem);
        line-height: 1.18; margin: 0 0 0.1rem; padding: 0.08rem 0;
    }
    h2, h3 { font-size: 1.28rem; margin-bottom: 0.25rem; }
    [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p, label, small, .st-emotion-cache-ue6h4q {
        color: #D4E3D9;
    }
    [data-testid="stCaptionContainer"] { font-size: 0.91rem; }
    [data-testid="stVerticalBlock"] { gap: 0.48rem; }
    [data-testid="stMarkdownContainer"] p { margin-bottom: 0.22rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #1C2B24; border: 1px solid #34483D; border-radius: 16px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.18); padding: 0.22rem;
    }
    .fruit-card {
        background: #263B30; border: 1px solid #3A5145; border-radius: 13px;
        padding: 0.62rem 0.68rem; margin: 0.16rem 0; color: #FFFFFF;
        min-height: 76px; display: flex; flex-direction: column; justify-content: center;
    }
    .fruit-card strong { font-size: 0.98rem; letter-spacing: 0.005em; }
    .class-pills { display: flex; gap: 0.3rem; flex-wrap: wrap; margin-top: 0.35rem; }
    .class-pill {
        border-radius: 999px; padding: 0.22rem 0.48rem; font-size: 0.71rem;
        font-weight: 700; line-height: 1.15;
    }
    .class-pill.fresh { color: #A9DDB9; background: #1C2B24; }
    .class-pill.rotten { color: #F2D5A2; background: #3C3425; }
    .catalog-unavailable { color: #D4E3D9; font-size: 0.75rem; }
    .system-card, .flow-node {
        min-height: 62px; border: 1px solid #3A5145; border-radius: 13px;
        background: #263B30; padding: 0.52rem 0.7rem; display: flex;
        flex-direction: column; justify-content: center; gap: 0.28rem;
    }
    .system-card span { color: #D4E3D9; font-size: 0.8rem; }
    .system-card strong, .flow-node strong { color: #FFFFFF; font-size: 0.96rem; overflow-wrap: anywhere; }
    .flow-node span { font-size: 0.8rem; }
    .online { color: #A9DDB9 !important; }
    .offline { color: #FF9B91 !important; }
    .neutral { color: #D4E3D9 !important; }
    .flow-arrow { text-align: center; color: #A9DDB9; font-size: 1.5rem; padding-top: 0.8rem; }
    .stButton > button {
        border: 0; border-radius: 11px; background: #2F7958; color: #FFFFFF;
        font-size: 0.96rem; font-weight: 700; min-height: 2.8rem; padding: 0.45rem 1.25rem;
        box-shadow: 0 4px 12px rgba(36, 107, 69, 0.18); transition: background 0.15s ease;
    }
    .stButton > button:hover { background: #3B916A; color: #FFFFFF; border: 0; }
    .stButton > button:focus { outline: 2px solid #A9DDB9; outline-offset: 2px; }
    [data-testid="stFileUploader"] section {
        min-height: 176px; border: 2px dashed #6E9B7E; border-radius: 16px;
        background: linear-gradient(145deg, #263B30, #1C2B24); padding: 0.72rem 1rem;
    }
    [data-testid="stFileUploader"] section > div {
        min-height: 142px; display: flex; align-items: center;
    }
    [data-testid="stFileUploader"] section, [data-testid="stFileUploader"] section * {
        color: #D4E3D9;
    }
    [data-testid="stFileUploader"] button, [data-testid="stFileUploader"] button:hover {
        color: #FFFFFF; background: #2F7958; border-color: #2F7958;
    }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div {
        background: #263B30; border-color: #52695A; color: #FFFFFF;
    }
    [data-baseweb="select"] *, [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea { color: #FFFFFF; }
    [role="listbox"], [role="option"] { background: #263B30; color: #FFFFFF; }
    [data-testid="stAlert"] {
        background: #263B30; border: 1px solid #52695A; color: #D4E3D9;
    }
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] { color: #D4E3D9; }
    [data-testid="stAlert"]:has(svg[aria-label="Error"]),
    [data-testid="stAlert"]:has(svg[aria-label="Warning"]) {
        border-color: #A85D55;
    }
    [data-testid="stImage"] img {
        display: block; width: auto !important; max-width: 100%;
        max-height: min(34vh, 340px); height: auto; object-fit: contain;
        border-radius: 13px; margin-left: auto; margin-right: auto;
    }
    [data-testid="stMetric"] {
        background: #263B30; border: 1px solid #3A5145; border-radius: 16px;
        padding: 0.72rem 0.9rem;
    }
    [data-testid="stMetricValue"] { color: #A9DDB9; font-size: 1.8rem; }
    [data-testid="stMetricLabel"] { color: #D4E3D9; font-size: 0.86rem; }
    [data-testid="stSpinner"] { color: #A9DDB9; }
    @media (max-width: 900px) {
        .block-container { padding: 1rem 0.8rem 2rem; }
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.8rem; }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width: 100% !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

system_info = fetch_system_info()
local_categories = load_catalog_categories()
catalog_categories = system_info["categories"] or local_categories

st.title("FruitVision AI")
st.markdown("*Sistema inteligente de clasificación de frutas mediante visión artificial.*")
left_column, right_column = st.columns([1.18, 2.42], gap="medium", vertical_alignment="top")

with left_column:
    with st.container(border=True):
        render_catalog(catalog_categories)

with right_column:
    with st.container(border=True):
        st.markdown("### Analiza una imagen")
        st.caption("Carga una fotografía para identificar su categoría y consultar la puntuación del modelo.")
        st.markdown("Selecciona un archivo o arrastra una fotografía al área de carga.")

        uploaded = st.file_uploader(
            "Selecciona o arrastra una fotografía",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
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
                set_step(verification, 1, "not_run", "No se envió ninguna solicitud al backend.")
                set_step(verification, 2, "not_run", "No se recibió respuesta HTTP porque no se llamó a la API.")
                set_step(verification, 3, "not_run", "La inferencia no se ejecutó.")
                set_step(verification, 4, "not_run", "No se recibió ningún resultado.")
                st.session_state.verification = verification
                st.error(validation_error)
                render_verification(verification)
            else:
                st.image(image_bytes, caption=uploaded.name, use_container_width=True)

                if st.button("Analizar imagen", type="primary"):
                    st.session_state.result = None
                    cached_model_info = None
                    cached_model_info_status = None
                    if system_info["model_name"] and system_info["categories"]:
                        cached_model_info = (
                            system_info["model_name"], system_info["categories"]
                        )
                        cached_model_info_status = system_info["model_info_status_code"]
                    with st.spinner("Analizando imagen con FruitVision AI..."):
                        result, verification = process_prediction(
                            image_bytes,
                            uploaded.name,
                            model_info_data=cached_model_info,
                            model_info_status_code=cached_model_info_status,
                        )
                    st.session_state.result = result
                    st.session_state.verification = verification

                result = st.session_state.result
                verification = st.session_state.verification
                if result and verification and verification["completed"]:
                    st.markdown("#### Resultado del análisis")
                    result_columns = st.columns([1.0, 1.15], vertical_alignment="center")
                    with result_columns[0]:
                        st.image(image_bytes, caption="Imagen analizada", use_container_width=True)
                    with result_columns[1]:
                        st.subheader(result["category"])
                        st.markdown(f"**Estado:** {result['state']}")
                        st.metric("Confianza del modelo", f"{result['confidence']:.2f}%")
                        st.caption(
                            "Puntuación de salida para esta imagen; no representa certeza absoluta ni rendimiento global."
                        )
                if verification:
                    render_verification(verification)

with st.container(border=True):
    render_system_panel(system_info)
