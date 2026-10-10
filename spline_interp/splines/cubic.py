"""Parametric cubic spline (C2) with chord-length parameterisation.

``x(t)`` and ``y(t)`` are interpolated separately with a global cubic spline, where the
knot ``t_i`` is the accumulated distance between the control points. The second
derivatives ``M_i`` at the knots are the solution of a linear system whose first and
last rows depend on the end condition:

* ``natural``: zero curvature at both ends (``M_0 = M_{n-1} = 0``)
* ``clamped``: the end tangents are prescribed - here they point along the first / last chord
* closed curves use ``periodic`` conditions (C2 everywhere, including the seam)
"""

from __future__ import annotations

import numpy as np

from .common import curve_function, unit_steps

END_CONDITIONS = ("natural", "clamped")


def _second_derivatives(y: np.ndarray, h: np.ndarray, end: str) -> np.ndarray:
    n = len(y)
    a = np.zeros((n, n))
    rhs = np.zeros((n, y.shape[1]))
    slope = np.diff(y, axis=0) / h[:, None]
    for i in range(1, n - 1):
        a[i, i - 1] = h[i - 1]
        a[i, i] = 2 * (h[i - 1] + h[i])
        a[i, i + 1] = h[i]
        rhs[i] = 6 * (slope[i] - slope[i - 1])
    if end == "clamped":
        d0, dn = slope[0], slope[-1]          # tangent along the first / last chord
        a[0, 0], a[0, 1] = 2 * h[0], h[0]
        rhs[0] = 6 * (slope[0] - d0)
        a[-1, -2], a[-1, -1] = h[-1], 2 * h[-1]
        rhs[-1] = 6 * (dn - slope[-1])
    else:                                     # natural
        a[0, 0] = a[-1, -1] = 1.0
    return np.linalg.solve(a, rhs)


def _periodic_second_derivatives(y: np.ndarray, h: np.ndarray) -> np.ndarray:
    # y has n + 1 rows (first point repeated); unknowns M_0..M_{n-1}, M_n = M_0.
    n = len(y) - 1
    a = np.zeros((n, n))
    rhs = np.zeros((n, y.shape[1]))
    slope = np.diff(y, axis=0) / h[:, None]
    for i in range(n):
        hp, hn = h[i - 1], h[i]               # h[-1] wraps to the closing segment
        a[i, (i - 1) % n] += hp
        a[i, i] += 2 * (hp + hn)
        a[i, (i + 1) % n] += hn
        rhs[i] = 6 * (slope[i] - slope[i - 1])
    m = np.linalg.solve(a, rhs)
    return np.vstack([m, m[:1]])


@curve_function
def cubic_spline(points: np.ndarray, closed: bool, samples: int,
                 end: str = "natural") -> np.ndarray:
    """Interpolating C2 cubic spline, chord-length parameterised."""
    if end not in END_CONDITIONS:
        raise ValueError(f"unknown end condition {end!r}")
    y = np.vstack([points, points[:1]]) if closed else points
    h = np.linalg.norm(np.diff(y, axis=0), axis=1)
    m = _periodic_second_derivatives(y, h) if closed else _second_derivatives(y, h, end)

    s = unit_steps(samples)[None, :, None] * h[:, None, None]   # local parameter
    hh = h[:, None, None]
    m0, m1 = m[:-1, None, :], m[1:, None, :]
    y0, y1 = y[:-1, None, :], y[1:, None, :]
    rest = hh - s
    out = (m0 * rest ** 3 + m1 * s ** 3) / (6 * hh) \
        + (y0 / hh - m0 * hh / 6) * rest + (y1 / hh - m1 * hh / 6) * s
    return np.vstack([out.reshape(-1, 2), y[-1:]])
