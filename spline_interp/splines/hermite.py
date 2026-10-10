"""Cubic Hermite family: Catmull-Rom, Cardinal and Kochanek-Bartels splines.

All of them interpolate every control point and differ only in how the tangent at each
point is chosen. Each segment ``P1 -> P2`` uses its neighbours ``P0`` and ``P3``; for open
curves the missing outer neighbours are mirrored phantom points (``2 P0 - P1``).
"""

from __future__ import annotations

import numpy as np

from .common import curve_function, unit_steps


def neighbour_quads(points: np.ndarray, closed: bool):
    """Return arrays ``(P0, P1, P2, P3)`` of shape ``(segments, 2)`` for every segment."""
    n = len(points)
    if closed:
        idx = np.arange(n)
        return (points[(idx - 1) % n], points[idx], points[(idx + 1) % n], points[(idx + 2) % n])
    ext = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])
    s = n - 1
    return ext[0:s], ext[1:s + 1], ext[2:s + 2], ext[3:s + 3]


def hermite(p1: np.ndarray, p2: np.ndarray, m1: np.ndarray, m2: np.ndarray,
            samples: int) -> np.ndarray:
    """Sample cubic Hermite segments from ``p1`` to ``p2`` with tangents ``m1`` / ``m2``."""
    t = unit_steps(samples)
    t2, t3 = t * t, t * t * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    out = (h00[None, :, None] * p1[:, None, :] + h10[None, :, None] * m1[:, None, :]
           + h01[None, :, None] * p2[:, None, :] + h11[None, :, None] * m2[:, None, :])
    return np.vstack([out.reshape(-1, 2), p2[-1:]])


@curve_function
def catmull_rom(points: np.ndarray, closed: bool, samples: int, alpha: float = 0.5) -> np.ndarray:
    """Catmull-Rom spline with knot parameterisation exponent ``alpha``.

    ``alpha = 0`` is the uniform, ``0.5`` the centripetal (no cusps or self-intersections
    within a segment) and ``1`` the chordal variant.
    """
    p0, p1, p2, p3 = neighbour_quads(points, closed)
    d01 = np.maximum(np.linalg.norm(p1 - p0, axis=1) ** alpha, 1e-9)[:, None]
    d12 = np.maximum(np.linalg.norm(p2 - p1, axis=1) ** alpha, 1e-9)[:, None]
    d23 = np.maximum(np.linalg.norm(p3 - p2, axis=1) ** alpha, 1e-9)[:, None]
    m1 = ((p1 - p0) / d01 - (p2 - p0) / (d01 + d12) + (p2 - p1) / d12) * d12
    m2 = ((p2 - p1) / d12 - (p3 - p1) / (d12 + d23) + (p3 - p2) / d23) * d12
    return hermite(p1, p2, m1, m2, samples)


def _kb_tangents(p0, p1, p2, p3, tension, bias, continuity):
    a = 1.0 - tension
    out1 = (a * (1 + bias) * (1 + continuity) / 2) * (p1 - p0) \
        + (a * (1 - bias) * (1 - continuity) / 2) * (p2 - p1)
    in2 = (a * (1 + bias) * (1 - continuity) / 2) * (p2 - p1) \
        + (a * (1 - bias) * (1 + continuity) / 2) * (p3 - p2)
    return out1, in2


@curve_function
def kochanek_bartels(points: np.ndarray, closed: bool, samples: int, tension: float = 0.0,
                     bias: float = 0.0, continuity: float = 0.0) -> np.ndarray:
    """Kochanek-Bartels (TCB) spline.

    * tension ``t``: 1 shortens the tangents (tight corners), -1 lengthens them (round)
    * bias ``b``: shifts the tangent towards the previous (+1) or next (-1) segment
    * continuity ``c``: != 0 gives different incoming/outgoing tangents -> visible corners
    """
    p0, p1, p2, p3 = neighbour_quads(points, closed)
    m1, m2 = _kb_tangents(p0, p1, p2, p3, tension, bias, continuity)
    return hermite(p1, p2, m1, m2, samples)


@curve_function
def cardinal(points: np.ndarray, closed: bool, samples: int, tension: float = 0.0) -> np.ndarray:
    """Cardinal spline: tangents ``(1 - tension) * (P[i+1] - P[i-1]) / 2``.

    ``tension = 0`` is the uniform Catmull-Rom spline, ``tension = 1`` gives zero tangents.
    """
    p0, p1, p2, p3 = neighbour_quads(points, closed)
    m1, m2 = _kb_tangents(p0, p1, p2, p3, tension, 0.0, 0.0)
    return hermite(p1, p2, m1, m2, samples)
