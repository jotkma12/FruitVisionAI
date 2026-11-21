import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src.backend.model_service import InvalidModelOutputError, ModelLoadError, ModelService


CLASS_NAMES = [
    "FreshApple",
    "FreshBanana",
    "FreshBellpepper",
    "FreshCarrot",
    "FreshCucumber",
    "FreshMango",
    "FreshOrange",
    "FreshPotato",
    "FreshStrawberry",
    "FreshTomato",
    "RottenApple",
    "RottenBanana",
    "RottenBellpepper",
    "RottenCarrot",
    "RottenCucumber",
    "RottenMango",
    "RottenOrange",
    "RottenPotato",
    "RottenStrawberry",
    "RottenTomato",
]


class FixedModel:
    def __init__(self, scores):
        self.scores = np.asarray(scores)

    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 224, 224, 3)
        return self.scores


def service_for(category):
    service = ModelService(Path("unused.keras"), Path("unused.json"))
    service.class_names = CLASS_NAMES
    scores = np.zeros((1, len(CLASS_NAMES)), dtype=np.float32)
    if category == "RottenApple":
        scores[0, CLASS_NAMES.index("RottenApple")] = 0.8
        scores[0, CLASS_NAMES.index("FreshApple")] = 0.2
    else:
        scores[0, CLASS_NAMES.index("FreshApple")] = 0.8
        scores[0, CLASS_NAMES.index("RottenApple")] = 0.2
    service.model = FixedModel(scores)
    service.output_is_probability_distribution = True
    return service


def test_prediction_derives_fresh_state_from_model_score():
    result = service_for("FreshApple").predict(np.zeros((1, 224, 224, 3), dtype=np.float32))
    assert result == {"category": "FreshApple", "state": "Fresh", "confidence": 80.0}


def test_prediction_derives_rotten_state_from_model_score():
    result = service_for("RottenApple").predict(np.zeros((1, 224, 224, 3), dtype=np.float32))
    assert result == {"category": "RottenApple", "state": "Rotten", "confidence": 80.0}


def test_model_name_uses_a_descriptive_fallback_for_unrecognized_models():
    service = service_for("FreshApple")

    assert service.model_name == "Keras FixedModel model"


@pytest.mark.parametrize(
    "invalid_score",
    [float("nan"), float("inf"), -0.01, 1.01],
    ids=["nan", "infinity", "negative", "above-one"],
)
def test_prediction_rejects_non_finite_or_out_of_range_scores(invalid_score):
    scores = np.full((1, len(CLASS_NAMES)), 0.05, dtype=np.float64)
    scores[0, 0] = invalid_score
    service = service_for("FreshApple")
    service.model = FixedModel(scores)

    with pytest.raises(InvalidModelOutputError):
        service.predict(np.zeros((1, 224, 224, 3), dtype=np.float32))


@pytest.mark.parametrize(
    "scores",
    [
        np.full((2, len(CLASS_NAMES)), 0.025),
        np.full((1, len(CLASS_NAMES) - 1), 1 / (len(CLASS_NAMES) - 1)),
    ],
    ids=["wrong-batch-dimensions", "wrong-class-dimensions"],
)
def test_prediction_rejects_incorrect_output_dimensions(scores):
    service = service_for("FreshApple")
    service.model = FixedModel(scores)

    with pytest.raises(InvalidModelOutputError, match="output shape"):
        service.predict(np.zeros((1, 224, 224, 3), dtype=np.float32))


def test_prediction_rejects_non_numeric_scores():
    service = service_for("FreshApple")
    service.model = FixedModel(np.full((1, len(CLASS_NAMES)), "score"))

    with pytest.raises(InvalidModelOutputError, match="real numeric"):
        service.predict(np.zeros((1, 224, 224, 3), dtype=np.float32))


def test_softmax_prediction_rejects_scores_that_do_not_sum_to_one():
    service = service_for("FreshApple")
    service.model = FixedModel(np.full((1, len(CLASS_NAMES)), 0.1))

    with pytest.raises(InvalidModelOutputError, match="sum approximately to 1"):
        service.predict(np.zeros((1, 224, 224, 3), dtype=np.float32))


def test_non_softmax_scores_do_not_require_probability_sum():
    service = service_for("FreshApple")
    service.output_is_probability_distribution = False
    service.model = FixedModel(np.full((1, len(CLASS_NAMES)), 0.2))

    result = service.predict(np.zeros((1, 224, 224, 3), dtype=np.float32))

    assert result["confidence"] == 20.0


def write_class_names(path):
    path.write_text(json.dumps(CLASS_NAMES), encoding="utf-8")


def test_load_logs_missing_class_names_without_exposing_path(tmp_path, caplog):
    service = ModelService(tmp_path / "model.keras", tmp_path / "missing-classes.json")

    with pytest.raises(ModelLoadError) as error:
        service.load()

    assert error.value.code == "class_names_file_missing"
    assert "class_names_file_missing" in caplog.text
    assert str(tmp_path) not in caplog.text


def test_load_logs_missing_model_without_exposing_path(tmp_path, caplog):
    labels_path = tmp_path / "classes.json"
    write_class_names(labels_path)
    service = ModelService(tmp_path / "missing-model.keras", labels_path)

    with pytest.raises(ModelLoadError) as error:
        service.load()

    assert error.value.code == "model_file_missing"
    assert "model_file_missing" in caplog.text
    assert str(tmp_path) not in caplog.text


def test_load_logs_keras_load_failure_without_exception_details(tmp_path, caplog):
    labels_path = tmp_path / "classes.json"
    model_path = tmp_path / "model.keras"
    write_class_names(labels_path)
    model_path.touch()
    service = ModelService(model_path, labels_path)

    with patch(
        "src.backend.model_service.tf.keras.models.load_model",
        side_effect=ValueError(f"bad model at {model_path}"),
    ), pytest.raises(ModelLoadError) as error:
        service.load()

    assert error.value.code == "model_load_failed"
    assert "model_load_failed" in caplog.text
    assert str(model_path) not in caplog.text
    assert "bad model" not in caplog.text


@pytest.mark.parametrize(
    ("input_shape", "output_shape", "expected_code"),
    [
        ((None, 160, 160, 3), (None, 20), "model_input_dimensions_incompatible"),
        ((None, 224, 224, 3), (None, 19), "model_class_count_mismatch"),
    ],
)
def test_load_logs_model_shape_and_class_count_mismatches(
    tmp_path, caplog, input_shape, output_shape, expected_code
):
    labels_path = tmp_path / "classes.json"
    model_path = tmp_path / "model.keras"
    write_class_names(labels_path)
    model_path.touch()
    fake_model = SimpleNamespace(
        input_shape=input_shape,
        output_shape=output_shape,
        layers=[SimpleNamespace(activation=None)],
    )
    service = ModelService(model_path, labels_path)

    with patch(
        "src.backend.model_service.tf.keras.models.load_model", return_value=fake_model
    ), pytest.raises(ModelLoadError) as error:
        service.load()

    assert error.value.code == expected_code
    assert expected_code in caplog.text
    assert service.ready is False
