"""Unit tests for input validation logic.

These tests do not load any ML artifacts — they only verify the validation
layer, which is pure Python and fast to run in any environment.
"""

import math
import pytest

from app.validators import ValidationError, validate_prediction_input


VALID_INPUTS = {
    "cycle_type": "discharge",
    "capacity": 1.674,
    "re": 0.056058,
    "rct": 0.200970,
}


class TestValidCases:
    def test_discharge_returns_dataclass(self):
        result = validate_prediction_input(**VALID_INPUTS)
        assert result.cycle_type == "discharge"
        assert math.isclose(result.capacity, 1.674)

    def test_charge_type_accepted(self):
        inp = {**VALID_INPUTS, "cycle_type": "charge"}
        result = validate_prediction_input(**inp)
        assert result.cycle_type == "charge"

    def test_impedance_type_accepted(self):
        inp = {**VALID_INPUTS, "cycle_type": "impedance"}
        result = validate_prediction_input(**inp)
        assert result.cycle_type == "impedance"

    def test_zero_capacity_accepted(self):
        inp = {**VALID_INPUTS, "capacity": 0.0}
        result = validate_prediction_input(**inp)
        assert result.capacity == 0.0

    def test_negative_re_accepted(self):
        # Re can be negative in some edge measurement cases
        inp = {**VALID_INPUTS, "re": -0.001}
        result = validate_prediction_input(**inp)
        assert result.re == -0.001

    def test_integer_inputs_coerced_to_float(self):
        result = validate_prediction_input(
            cycle_type="discharge", capacity=2, re=0, rct=0
        )
        assert isinstance(result.capacity, float)
        assert isinstance(result.re, float)
        assert isinstance(result.rct, float)


class TestInvalidCycleType:
    def test_unknown_type_raises(self):
        inp = {**VALID_INPUTS, "cycle_type": "fast_charge"}
        with pytest.raises(ValidationError, match="cycle_type"):
            validate_prediction_input(**inp)

    def test_empty_string_raises(self):
        inp = {**VALID_INPUTS, "cycle_type": ""}
        with pytest.raises(ValidationError):
            validate_prediction_input(**inp)

    def test_case_sensitive_rejection(self):
        # The encoder was fit on lowercase labels
        inp = {**VALID_INPUTS, "cycle_type": "Discharge"}
        with pytest.raises(ValidationError):
            validate_prediction_input(**inp)


class TestInvalidNumerics:
    def test_nan_capacity_raises(self):
        inp = {**VALID_INPUTS, "capacity": float("nan")}
        with pytest.raises(ValidationError, match="finite"):
            validate_prediction_input(**inp)

    def test_inf_re_raises(self):
        inp = {**VALID_INPUTS, "re": float("inf")}
        with pytest.raises(ValidationError, match="finite"):
            validate_prediction_input(**inp)

    def test_negative_inf_rct_raises(self):
        inp = {**VALID_INPUTS, "rct": float("-inf")}
        with pytest.raises(ValidationError, match="finite"):
            validate_prediction_input(**inp)

    def test_negative_capacity_raises(self):
        inp = {**VALID_INPUTS, "capacity": -0.5}
        with pytest.raises(ValidationError, match="negative"):
            validate_prediction_input(**inp)

    def test_none_raises(self):
        with pytest.raises(ValidationError):
            validate_prediction_input(cycle_type="discharge", capacity=None, re=0.056, rct=0.2)  # type: ignore
