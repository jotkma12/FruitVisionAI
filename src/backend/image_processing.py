"""Decode and prepare image uploads using the confirmed training input format."""

from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import DEFAULT_MAX_IMAGE_PIXELS


INPUT_SIZE = (224, 224)
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class InvalidImageError(ValueError):
    """Raised when an upload is empty, unsupported, or corrupt."""


def prepare_image(
    raw: bytes, max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS
) -> np.ndarray:
    if not raw:
        raise InvalidImageError("The uploaded file is empty.")
    try:
        with Image.open(BytesIO(raw)) as source:
            if source.format not in ALLOWED_FORMATS:
                raise InvalidImageError("Supported formats are JPEG, PNG, and WEBP.")
            width, height = source.size
            if width * height > max_image_pixels:
                raise InvalidImageError(
                    f"Image exceeds the configured pixel limit of {max_image_pixels} pixels."
                )
            image = source.convert("RGB").resize(INPUT_SIZE, Image.Resampling.BILINEAR)
            # Training fed float32 pixels in [0, 255]. The model contains its
            # own rescaling and ImageNet normalization layers.
            pixels = np.asarray(image, dtype=np.float32)
    except Image.DecompressionBombError as exc:
        raise InvalidImageError("Image dimensions exceed the supported pixel limit.") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("The uploaded file is not a valid image.") from exc

    if pixels.shape != (224, 224, 3):
        raise InvalidImageError("The image could not be prepared as a 224x224 RGB image.")
    return np.expand_dims(pixels, axis=0)
