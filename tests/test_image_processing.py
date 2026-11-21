from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from src.backend.image_processing import InvalidImageError, prepare_image


def png_bytes(color=(12, 90, 240, 120)):
    stream = BytesIO()
    Image.new("RGBA", (40, 30), color).save(stream, format="PNG")
    return stream.getvalue()


def test_prepare_image_returns_rgb_float32_in_training_range():
    batch = prepare_image(png_bytes())
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    np.testing.assert_allclose(batch[0, 100, 100], [12, 90, 240], atol=1)


def test_prepare_image_rejects_dimensions_over_configured_pixel_limit():
    with pytest.raises(InvalidImageError, match="configured pixel limit"):
        prepare_image(png_bytes(), max_image_pixels=1_199)


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP"])
def test_prepare_image_accepts_each_supported_format(image_format):
    stream = BytesIO()
    Image.new("RGB", (40, 30), (12, 90, 240)).save(stream, format=image_format)
    batch = prepare_image(stream.getvalue())
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert 0 <= float(batch.min()) <= float(batch.max()) <= 255


@pytest.mark.parametrize("raw", [b"", b"not an image"])
def test_prepare_image_rejects_empty_or_corrupt_data(raw):
    with pytest.raises(InvalidImageError):
        prepare_image(raw)


def test_prepare_image_rejects_unsupported_format():
    stream = BytesIO()
    Image.new("RGB", (4, 4), "red").save(stream, format="BMP")
    with pytest.raises(InvalidImageError, match="Supported formats"):
        prepare_image(stream.getvalue())
