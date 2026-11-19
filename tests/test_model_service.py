from pathlib import Path

import numpy as np

from src.backend.model_service import ModelService


class FixedModel:
    def __init__(self, scores):
        self.scores = np.asarray([scores], dtype=np.float32)

    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 224, 224, 3)
        return self.scores


def service_for(category):
    service = ModelService(Path("unused.keras"), Path("unused.json"))
    service.class_names = ["FreshApple", "RottenApple"]
    service.model = FixedModel([0.2, 0.8] if category == "RottenApple" else [0.8, 0.2])
    return service


def test_prediction_derives_fresh_state_from_model_score():
    result = service_for("FreshApple").predict(np.zeros((1, 224, 224, 3), dtype=np.float32))
    assert result == {"category": "FreshApple", "state": "Fresh", "confidence": 80.0}


def test_prediction_derives_rotten_state_from_model_score():
    result = service_for("RottenApple").predict(np.zeros((1, 224, 224, 3), dtype=np.float32))
    assert result == {"category": "RottenApple", "state": "Rotten", "confidence": 80.0}
