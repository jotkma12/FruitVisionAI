"""Read-only loading and prediction using the original exported model."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


class ModelService:
    def __init__(self, model_path: Path, class_names_path: Path) -> None:
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model: Any = None
        self.class_names: list[str] = []

    def load(self) -> None:
        with self.class_names_path.open("r", encoding="utf-8") as file:
            class_names = json.load(file)
        if not isinstance(class_names, list) or not class_names or not all(isinstance(name, str) for name in class_names):
            raise ValueError("The class names file must contain a non-empty list of strings.")

        model = tf.keras.models.load_model(self.model_path, compile=False)
        if tuple(model.input_shape[1:]) != (224, 224, 3):
            raise ValueError(f"Unexpected model input shape: {model.input_shape}")
        if model.output_shape[-1] != len(class_names):
            raise ValueError("The model output count does not match the class names file.")
        self.model = model
        self.class_names = class_names

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, batch: np.ndarray) -> dict[str, str | float]:
        if self.model is None:
            raise RuntimeError("The model is not loaded.")
        scores = np.asarray(self.model.predict(batch, verbose=0))[0]
        index = int(np.argmax(scores))
        category = self.class_names[index]
        if category.startswith("Fresh"):
            state = "Fresh"
        elif category.startswith("Rotten"):
            state = "Rotten"
        else:
            raise ValueError(f"Unsupported class prefix: {category}")
        return {"category": category, "state": state, "confidence": round(float(scores[index]) * 100.0, 2)}
