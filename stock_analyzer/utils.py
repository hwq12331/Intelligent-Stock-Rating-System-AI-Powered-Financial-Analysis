"""Small defensive numeric helpers used throughout the analyzer."""

from __future__ import annotations

from typing import Any

import numpy as np


def safe_compare(value: Any, operator: str, threshold: float, default: bool = False) -> bool:
    """Safely compare values that may be None, NaN, or infinity."""
    if value is None:
        return default

    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return default

        try:
            if operator == ">":
                return value > threshold
            if operator == "<":
                return value < threshold
            if operator == ">=":
                return value >= threshold
            if operator == "<=":
                return value <= threshold
            if operator == "==":
                return value == threshold
            if operator == "!=":
                return value != threshold
        except Exception:
            return default

    return default


def safe_get(value: Any, default: float = 0.0) -> float:
    """Return a finite float or the provided default."""
    if value is None:
        return default

    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return default
        return float(value)

    try:
        parsed = float(value)
        if np.isnan(parsed) or np.isinf(parsed):
            return default
        return parsed
    except Exception:
        return default


def safe_divide(numerator: Any, denominator: Any, default: float | None = 0.0) -> float | None:
    """Safely divide two values with None and zero handling."""
    num = safe_get(numerator, 0)
    den = safe_get(denominator, 1)

    if den == 0:
        return default

    try:
        result = num / den
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except Exception:
        return default
