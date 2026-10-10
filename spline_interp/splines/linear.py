"""Linear interpolation: the control polygon itself (C0)."""

from __future__ import annotations

import numpy as np

from .common import curve_function, polyline


@curve_function
def linear(points: np.ndarray, closed: bool, samples: int) -> np.ndarray:
    """Piecewise linear interpolation through all points."""
    return polyline(points, closed, samples)
