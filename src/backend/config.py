"""Runtime settings and project-relative model paths."""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    model_path: Path
    class_names_path: Path
    max_upload_bytes: int = 10 * 1024 * 1024


def get_settings() -> Settings:
    return Settings(
        model_path=Path(os.getenv("FRUITVISION_MODEL_PATH", PROJECT_ROOT / "models" / "fruitvision_ai_model.keras")),
        class_names_path=Path(os.getenv("FRUITVISION_CLASS_NAMES_PATH", PROJECT_ROOT / "models" / "class_names.json")),
        max_upload_bytes=int(os.getenv("FRUITVISION_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)),
    )
