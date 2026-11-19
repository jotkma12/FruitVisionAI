"""HTTP API for live FruitVision inference."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from .config import get_settings
from .image_processing import InvalidImageError, prepare_image
from .model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    service = ModelService(settings.model_path, settings.class_names_path)
    app.state.settings = settings
    app.state.service = service
    app.state.load_error = None
    try:
        service.load()
    except Exception as exc:  # Keep the server available so health reports not-ready.
        app.state.load_error = str(exc)
    yield


app = FastAPI(title="FruitVision AI", version="1.0.0", lifespan=lifespan)


def require_service(request: Request) -> ModelService:
    service = request.app.state.service
    if not service.ready:
        raise HTTPException(status_code=503, detail="Inference model is not available.")
    return service


@app.get("/health")
def health(request: Request) -> dict[str, str | bool]:
    ready = request.app.state.service.ready
    return {"status": "ready" if ready else "unavailable", "available": ready}


@app.get("/model/info")
def model_info(request: Request) -> dict[str, str | list[str]]:
    service = require_service(request)
    return {"model": "EfficientNetB0", "categories": service.class_names}


@app.post("/predict")
async def predict(request: Request, image: UploadFile = File(...)) -> dict[str, str | float]:
    service = require_service(request)
    raw = await image.read(request.app.state.settings.max_upload_bytes + 1)
    if len(raw) > request.app.state.settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the configured upload limit.")
    try:
        batch = prepare_image(raw)
        return service.predict(batch)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed for this image.") from exc
