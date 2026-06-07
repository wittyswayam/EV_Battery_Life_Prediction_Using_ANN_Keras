"""Integration tests for PredictionService.

These tests require the trained artifacts to be present in the artifacts/
directory. They are skipped automatically if artifacts are missing, so
the CI pipeline can run `test_validators.py` without a training step.
"""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.config import ARTIFACTS_DIR
from app.predictor import ArtifactLoadError, PredictionService
from app.validators import ValidationError

ARTIFACTS_PRESENT = (
    (ARTIFACTS_DIR / "battery_life_model.pkl").exists()
    and (ARTIFACTS_DIR / "scaler.pkl").exists()
    and (ARTIFACTS_DIR / "label_encoder.pkl").exists()
)

requires_artifacts = pytest.mark.skipif(
    not ARTIFACTS_PRESENT,
    reason="Trained artifacts not found. Run the notebook first.",
)


class TestArtifactLoadError:
    def test_missing_model_raises(self, tmp_path):
        fake_scaler = tmp_path / "scaler.pkl"
        fake_encoder = tmp_path / "label_encoder.pkl"

        # Create minimal valid sklearn-like objects
        scaler_mock = MagicMock()
        scaler_mock.transform = lambda X: X
        encoder_mock = MagicMock()
        encoder_mock.transform = lambda x: [0]
        encoder_mock.classes_ = ["charge", "discharge", "impedance"]

        pickle.dump(scaler_mock, open(fake_scaler, "wb"))
        pickle.dump(encoder_mock, open(fake_encoder, "wb"))

        with pytest.raises(ArtifactLoadError, match="model"):
            PredictionService(
                model_path=tmp_path / "nonexistent.pkl",
                scaler_path=fake_scaler,
                encoder_path=fake_encoder,
            )


class TestPredictionServiceWithMocks:
    """Test the service logic using lightweight mocks instead of real artifacts."""

    def _build_service(self, tmp_path, prediction_value=24.0):
        model_mock = MagicMock()
        model_mock.predict = MagicMock(return_value=np.array([[prediction_value]]))

        scaler_mock = MagicMock()
        scaler_mock.transform = MagicMock(side_effect=lambda X: X)

        encoder_mock = MagicMock()
        encoder_mock.transform = MagicMock(return_value=[1])
        encoder_mock.classes_ = ["charge", "discharge", "impedance"]

        for name, obj in (
            ("model.pkl", model_mock),
            ("scaler.pkl", scaler_mock),
            ("encoder.pkl", encoder_mock),
        ):
            pickle.dump(obj, open(tmp_path / name, "wb"))

        return PredictionService(
            model_path=tmp_path / "model.pkl",
            scaler_path=tmp_path / "scaler.pkl",
            encoder_path=tmp_path / "encoder.pkl",
        )

    def test_predict_returns_float(self, tmp_path):
        service = self._build_service(tmp_path, prediction_value=24.0)
        result = service.predict("discharge", 1.674, 0.056, 0.201)
        assert isinstance(result, float)
        assert result == pytest.approx(24.0)

    def test_invalid_cycle_type_raises_validation_error(self, tmp_path):
        service = self._build_service(tmp_path)
        with pytest.raises(ValidationError):
            service.predict("turbo_charge", 1.674, 0.056, 0.201)

    def test_nan_capacity_raises_validation_error(self, tmp_path):
        service = self._build_service(tmp_path)
        with pytest.raises(ValidationError):
            service.predict("discharge", float("nan"), 0.056, 0.201)

    def test_scaler_called_once_per_predict(self, tmp_path):
        service = self._build_service(tmp_path)
        service.predict("discharge", 1.674, 0.056, 0.201)
        service._scaler.transform.assert_called_once()

    def test_model_called_once_per_predict(self, tmp_path):
        service = self._build_service(tmp_path)
        service.predict("discharge", 1.674, 0.056, 0.201)
        service._model.predict.assert_called_once()


@requires_artifacts
class TestPredictionServiceWithRealArtifacts:
    """End-to-end tests using the actual trained artifacts."""

    @pytest.fixture(scope="class")
    def service(self):
        return PredictionService()

    def test_discharge_prediction_in_physical_range(self, service):
        result = service.predict("discharge", 1.674, 0.056, 0.201)
        assert -10.0 <= result <= 60.0, f"Unexpected prediction: {result}"

    def test_charge_prediction_is_float(self, service):
        result = service.predict("charge", 0.0, 0.056, 0.201)
        assert isinstance(result, float)

    def test_impedance_prediction_is_float(self, service):
        result = service.predict("impedance", 0.0, 0.056, 0.201)
        assert isinstance(result, float)

    def test_repeated_calls_return_same_result(self, service):
        r1 = service.predict("discharge", 1.674, 0.056, 0.201)
        r2 = service.predict("discharge", 1.674, 0.056, 0.201)
        assert r1 == pytest.approx(r2)
