import math
from dataclasses import dataclass
from typing import Optional

from app.config import VALID_CYCLE_TYPES


@dataclass
class PredictionInput:
    cycle_type: str
    capacity: float
    re: float
    rct: float


class ValidationError(ValueError):
    pass


def validate_prediction_input(
    cycle_type: str,
    capacity: float,
    re: float,
    rct: float,
) -> PredictionInput:
    """
    Validate and sanitize inputs before they reach the model.

    Raises ValidationError with a human-readable message on any failure.
    Returns a clean PredictionInput on success.
    """
    if cycle_type not in VALID_CYCLE_TYPES:
        raise ValidationError(
            f"cycle_type must be one of {VALID_CYCLE_TYPES}, got '{cycle_type}'."
        )

    for name, val in (("Capacity", capacity), ("Re", re), ("Rct", rct)):
        if val is None:
            raise ValidationError(f"{name} cannot be None.")
        if not isinstance(val, (int, float)):
            raise ValidationError(f"{name} must be a number, got {type(val).__name__}.")
        if math.isnan(val) or math.isinf(val):
            raise ValidationError(f"{name} must be a finite number, got {val}.")

    if capacity < 0:
        raise ValidationError(f"Capacity cannot be negative, got {capacity}.")

    return PredictionInput(
        cycle_type=cycle_type,
        capacity=float(capacity),
        re=float(re),
        rct=float(rct),
    )
