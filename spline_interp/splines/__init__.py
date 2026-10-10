"""Pure curve functions and the method registry used by the UI.

Every curve function takes ``(points, closed=False, samples=16, **options)`` and returns
the sampled curve as an ``(m, 2)`` NumPy array. ``samples`` is the number of samples per
segment (between two consecutive control points).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..params import Params
from .bezier import bezier, composite_bezier
from .bspline import bspline, nurbs
from .cubic import END_CONDITIONS, cubic_spline
from .hermite import cardinal, catmull_rom, kochanek_bartels
from .lagrange import lagrange
from .linear import linear


@dataclass(frozen=True)
class Method:
    key: str
    name: str
    func: Callable[..., np.ndarray]
    interpolating: bool          # passes through every control point?
    continuity: str
    local: bool                  # moving one point only changes the curve nearby?
    uses: tuple[str, ...]        # parameter names from Params that affect this method
    description: str
    options: Callable[[Params], dict] = lambda p: {}

    def evaluate(self, points, params: Params, weights=None) -> np.ndarray:
        opts = dict(self.options(params))
        if self.key == "nurbs":
            opts["weights"] = weights
        return self.func(points, closed=params.closed, samples=params.samples, **opts)


METHODS: tuple[Method, ...] = (
    Method("linear", "Linear", linear, True, "C0", True, ("samples",),
           "Straight segments - the control polygon."),
    Method("lagrange", "Lagrange polynomial", lagrange, True, "C\u221e", False, ("samples",),
           "One global polynomial; oscillates wildly with many points (Runge)."),
    Method("catmull_rom", "Catmull-Rom", catmull_rom, True, "C1", True, ("samples", "alpha"),
           "Hermite tangents from neighbours; alpha 0 / 0.5 / 1 = uniform / centripetal / chordal.",
           lambda p: {"alpha": p.alpha}),
    Method("cardinal", "Cardinal", cardinal, True, "C1", True, ("samples", "tension"),
           "Catmull-Rom with scaled tangents; tension 1 = straight-ish, -1 = loose.",
           lambda p: {"tension": p.tension}),
    Method("kochanek_bartels", "Kochanek-Bartels", kochanek_bartels, True, "C1 (C0 if c\u22600)",
           True, ("samples", "tension", "bias", "continuity"),
           "TCB spline: tension, bias and continuity shape every tangent.",
           lambda p: {"tension": p.tension, "bias": p.bias, "continuity": p.continuity}),
    Method("cubic", "Cubic spline", cubic_spline, True, "C2", False, ("samples", "end"),
           "Global C2 spline (chord-length); natural/clamped ends, periodic when closed.",
           lambda p: {"end": END_CONDITIONS[p.end_index]}),
    Method("bezier", "B\u00e9zier (single)", bezier, False, "C\u221e", False, ("samples",),
           "One curve of degree n-1; touches only the first and last point."),
    Method("composite_bezier", "Composite cubic B\u00e9zier", composite_bezier, False,
           "C0 (G1 if aligned)", True, ("samples",),
           "Every 3rd point is an anchor on the curve, the two between are handles."),
    Method("bspline", "Cubic B-spline", bspline, False, "C2", True, ("samples",),
           "Uniform cubic B-spline; smooth, stays near the control polygon."),
    Method("nurbs", "NURBS", nurbs, False, "C2", True, ("samples", "weight"),
           "Rational cubic B-spline; a point's weight pulls the curve towards it."),
)

METHOD_BY_KEY = {m.key: m for m in METHODS}

__all__ = [
    "END_CONDITIONS",
    "METHODS",
    "METHOD_BY_KEY",
    "Method",
    "bezier",
    "bspline",
    "cardinal",
    "catmull_rom",
    "composite_bezier",
    "cubic_spline",
    "kochanek_bartels",
    "lagrange",
    "linear",
    "nurbs",
]
