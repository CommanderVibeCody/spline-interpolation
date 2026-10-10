"""Global Lagrange polynomial interpolation (shows Runge's phenomenon)."""

from __future__ import annotations

import math

import numpy as np

from .common import curve_function, unit_steps

LIMIT = 1e6  # keep wildly oscillating values finite for drawing


@curve_function
def lagrange(points: np.ndarray, closed: bool, samples: int) -> np.ndarray:
    """One polynomial of degree ``n - 1`` through all points, at equally spaced nodes.

    ``x(t)`` and ``y(t)`` are interpolated separately at the nodes ``t = 0, 1, ..., n-1``
    using the numerically stable barycentric formula. With many points the polynomial
    oscillates wildly between the nodes near the ends (Runge's phenomenon).
    A closed curve simply repeats the first point at the end (the polynomial is not periodic).
    """
    pts = np.vstack([points, points[:1]]) if closed else points
    n = len(pts)
    nodes = np.arange(n, dtype=float)
    # Barycentric weights for equally spaced nodes: (-1)^j * C(n-1, j).
    w = np.array([(-1) ** j * math.comb(n - 1, j) for j in range(n)], dtype=float)
    t = (nodes[:-1, None] + unit_steps(samples)[None, :]).reshape(-1)
    t = np.append(t, nodes[-1])

    diff = t[:, None] - nodes[None, :]
    exact = np.isclose(diff, 0.0, atol=1e-12)
    diff[exact] = 1.0
    terms = w[None, :] / diff
    out = (terms @ pts) / terms.sum(axis=1)[:, None]
    hit_rows, hit_cols = np.nonzero(exact)
    out[hit_rows] = pts[hit_cols]
    return np.clip(out, -LIMIT, LIMIT)
