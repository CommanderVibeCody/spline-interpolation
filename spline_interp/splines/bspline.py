"""Uniform cubic B-splines and NURBS, evaluated with the Cox-de Boor recursion.

* open curves use a clamped (open-uniform) knot vector: the curve starts at the first and
  ends at the last control point, tangent to the control polygon there,
* closed curves use a periodic uniform knot vector (the first ``degree`` points are
  wrapped around), giving a C2 loop.

NURBS divide by the weighted basis sum (rational B-spline): a larger weight pulls the
curve towards its control point.
"""

from __future__ import annotations

import numpy as np

from .common import curve_function, unit_steps

DEGREE = 3


def basis_functions(knots: np.ndarray, degree: int, u: np.ndarray) -> np.ndarray:
    """B-spline basis matrix ``N[sample, control_point]`` via Cox-de Boor."""
    k = np.asarray(knots, dtype=float)
    u = np.asarray(u, dtype=float)
    lo, hi = k[:-1][None, :], k[1:][None, :]
    n = ((lo <= u[:, None]) & (u[:, None] < hi)).astype(float)
    # The right end of the domain belongs to the last non-empty knot span.
    last = np.nonzero(k[:-1] < k[1:])[0][-1]
    at_end = u >= k[last + 1]
    n[at_end] = 0.0
    n[at_end, last] = 1.0
    for d in range(1, degree + 1):
        cols = len(k) - 1 - d
        new = np.zeros((len(u), cols))
        for i in range(cols):
            den1 = k[i + d] - k[i]
            den2 = k[i + d + 1] - k[i + 1]
            if den1 > 0:
                new[:, i] += (u - k[i]) / den1 * n[:, i]
            if den2 > 0:
                new[:, i] += (k[i + d + 1] - u) / den2 * n[:, i + 1]
        n = new
    return n


def _evaluate(points: np.ndarray, weights: np.ndarray, closed: bool, samples: int) -> np.ndarray:
    count = len(points)
    if closed:
        p = DEGREE
        ctrl = np.vstack([points, points[:p]])
        w = np.concatenate([weights, weights[:p]])
        knots = np.arange(len(ctrl) + p + 1, dtype=float)
        spans = count
        u = p + np.append(np.arange(spans * samples) / samples, spans)
    else:
        p = min(DEGREE, count - 1)
        ctrl, w = points, weights
        spans = count - p
        knots = np.concatenate([np.zeros(p), np.linspace(0.0, 1.0, spans + 1), np.ones(p)])
        u = np.append((np.repeat(np.arange(spans), samples) + np.tile(unit_steps(samples), spans))
                      / spans, 1.0)
    basis = basis_functions(knots, p, u) * w[None, :]
    return (basis @ ctrl) / basis.sum(axis=1)[:, None]


@curve_function
def bspline(points: np.ndarray, closed: bool, samples: int) -> np.ndarray:
    """Uniform cubic B-spline (approximating, C2, local control)."""
    return _evaluate(points, np.ones(len(points)), closed, samples)


@curve_function
def nurbs(points: np.ndarray, closed: bool, samples: int, weights=None) -> np.ndarray:
    """Cubic NURBS: like :func:`bspline` but with a positive weight per control point."""
    w = np.clip(np.asarray(weights, dtype=float), 1e-3, None)
    return _evaluate(points, w, closed, samples)
