import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import LABEL_ENCODER_PATH, MODEL_PATH, SCALER_PATH
from app.logger import get_logger
from app.validators import PredictionInput, ValidationError, validate_prediction_input

logger = get_logger(__name__)


class ArtifactLoadError(RuntimeError):
    pass


class PredictionService:
    """
    Encapsulates all artifact loading and inference logic.

    Keeps the Streamlit layer completely free of ML code. Artifacts are loaded
    once at construction time and reused across calls. The class is intentionally
    stateless beyond the loaded objects — no mutable shared state.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        encoder_path: Path = LABEL_ENCODER_PATH,
    ) -> None:
        self._model = self._load_artifact(model_path, "model")
        self._scaler = self._load_artifact(scaler_path, "scaler")
        self._label_encoder = self._load_artifact(encoder_path, "label encoder")
        logger.info(
            "PredictionService initialized | encoder classes: %s",
            list(self._label_encoder.classes_),
        )

    @staticmethod
    def _load_artifact(path: Path, name: str):
        if not path.exists():
            raise ArtifactLoadError(
                f"Required artifact '{name}' not found at: {path}\n"
                "Run the training notebook to generate artifacts."
            )
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            logger.debug("Loaded %s from %s", name, path)
            return obj
        except Exception as exc:
            raise ArtifactLoadError(
                f"Failed to load '{name}' from {path}: {exc}"
            ) from exc

    def predict(
        self,
        cycle_type: str,
        capacity: float,
        re: float,
        rct: float,
    ) -> float:
        """
        Run the full preprocessing + inference pipeline for a single sample.

        Returns the predicted ambient temperature (°C) as a Python float.
        Raises ValidationError for invalid inputs, RuntimeError for inference failures.
        """
        inp: PredictionInput = validate_prediction_input(cycle_type, capacity, re, rct)

        try:
            encoded_type = self._label_encoder.transform([inp.cycle_type])[0]
        except ValueError as exc:
            raise ValidationError(
                f"Encoding failed for cycle_type='{inp.cycle_type}': {exc}"
            ) from exc

        X = np.array([[encoded_type, inp.capacity, inp.re, inp.rct]], dtype=np.float64)

        try:
            X_scaled = self._scaler.transform(X)
        except Exception as exc:
            raise RuntimeError(f"Scaling failed: {exc}") from exc

        try:
            raw = self._model.predict(X_scaled)
            # Support both Keras model output (ndarray shape (1,1)) and
            # any sklearn-style model returning a 1-D array
            result = float(np.ravel(raw)[0])
        except Exception as exc:
            raise RuntimeError(f"Model inference failed: {exc}") from exc

        logger.info(
            "Prediction | type=%s capacity=%.4f re=%.6f rct=%.6f -> %.4f °C",
            inp.cycle_type,
            inp.capacity,
            inp.re,
            inp.rct,
            result,
        )
        return result
