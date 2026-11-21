"""Read-only loading and prediction using the original exported model."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)


class InvalidModelOutputError(ValueError):
    """Raised when the model returns scores that cannot form a prediction."""


class ModelLoadError(RuntimeError):
    """A safe, categorized model startup error without local path details."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ModelService:
    EFFICIENTNET_B0_LAYER_MARKERS = {
        "stem_conv",
        "block1a_dwconv",
        "block1a_project_conv",
        "top_conv",
    }

    def __init__(self, model_path: Path, class_names_path: Path) -> None:
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model: Any = None
        self.class_names: list[str] = []
        self.output_is_probability_distribution = False

    @staticmethod
    def _raise_load_error(code: str) -> None:
        logger.error("Model loading failed: %s", code)
        raise ModelLoadError(code)

    def load(self) -> None:
        try:
            with self.class_names_path.open("r", encoding="utf-8") as file:
                class_names = json.load(file)
        except FileNotFoundError:
            self._raise_load_error("class_names_file_missing")
        except json.JSONDecodeError:
            self._raise_load_error("class_names_file_invalid")
        except OSError:
            self._raise_load_error("class_names_file_unreadable")
        if not isinstance(class_names, list) or not class_names or not all(isinstance(name, str) for name in class_names):
            self._raise_load_error("class_names_invalid")

        if not self.model_path.is_file():
            self._raise_load_error("model_file_missing")
        try:
            model = tf.keras.models.load_model(self.model_path, compile=False)
        except Exception as exc:
            logger.error(
                "Model loading failed: model_load_failed (exception_type=%s)",
                type(exc).__name__,
            )
            raise ModelLoadError("model_load_failed") from exc
        if tuple(model.input_shape[1:]) != (224, 224, 3):
            self._raise_load_error("model_input_dimensions_incompatible")
        if model.output_shape[-1] != len(class_names):
            self._raise_load_error("model_class_count_mismatch")
        try:
            output_layer = model.layers[-1]
        except (AttributeError, IndexError):
            self._raise_load_error("model_load_failed")
        activation = getattr(output_layer, "activation", None)
        output_is_probability_distribution = (
            getattr(activation, "__name__", None) == "softmax"
            or isinstance(output_layer, tf.keras.layers.Softmax)
        )
        self.model = model
        self.class_names = class_names
        self.output_is_probability_distribution = output_is_probability_distribution

    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def model_name(self) -> str:
        if self.model is None:
            return "Keras model unavailable"

        layers = getattr(self.model, "layers", [])
        backbone = next(
            (
                layer
                for layer in layers
                if isinstance(layer, tf.keras.Model)
                and layer.name.lower() == "efficientnetb0"
            ),
            None,
        )
        if backbone is not None:
            backbone_layer_names = {layer.name for layer in backbone.layers}
            if self.EFFICIENTNET_B0_LAYER_MARKERS.issubset(backbone_layer_names):
                return "EfficientNetB0"

        return f"Keras {type(self.model).__name__} model"

    def predict(self, batch: np.ndarray) -> dict[str, str | float]:
        if self.model is None:
            raise RuntimeError("The model is not loaded.")
        try:
            outputs = np.asarray(self.model.predict(batch, verbose=0))
        except (TypeError, ValueError) as exc:
            raise InvalidModelOutputError("Model output must be a numeric array.") from exc

        expected_shape = (1, len(self.class_names))
        if outputs.shape != expected_shape:
            raise InvalidModelOutputError(
                f"Unexpected model output shape: {outputs.shape}; expected {expected_shape}."
            )
        if not np.issubdtype(outputs.dtype, np.number) or np.issubdtype(
            outputs.dtype, np.complexfloating
        ):
            raise InvalidModelOutputError("Model output must contain real numeric scores.")
        if not np.all(np.isfinite(outputs)):
            raise InvalidModelOutputError("Model output contains non-finite scores.")
        if np.any(outputs < 0) or np.any(outputs > 1):
            raise InvalidModelOutputError("Model scores must be within [0, 1].")
        if self.output_is_probability_distribution and not np.isclose(
            np.sum(outputs[0]), 1.0, rtol=1e-5, atol=1e-6
        ):
            raise InvalidModelOutputError(
                "Softmax model output must sum approximately to 1."
            )

        scores = outputs[0]
        index = int(np.argmax(scores))
        category = self.class_names[index]
        if category.startswith("Fresh"):
            state = "Fresh"
        elif category.startswith("Rotten"):
            state = "Rotten"
        else:
            raise ValueError(f"Unsupported class prefix: {category}")
        return {"category": category, "state": state, "confidence": round(float(scores[index]) * 100.0, 2)}
