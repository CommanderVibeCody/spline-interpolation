"""Shared helpers for the curve functions.

Every public curve function takes an ``(n, 2)`` array-like of control points and
returns an ``(m, 2)`` float array of sampled curve points. The :func:`curve_function`
decorator normalises the input and handles the degenerate cases uniformly:

* consecutive duplicate points are merged (and, for closed curves, a last point
  equal to the first one is dropped),
* 0 points -> empty ``(0, 2)`` array, 1 point -> that point, 2 points -> straight line.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable

import numpy as np

EPS = 1e-9


def as_points(points) -> np.ndarray:
    """Convert anything array-like to a float ``(n, 2)`` array (copy)."""
    arr = np.array(points, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2))
    return arr.reshape(-1, 2)


def dedupe(points: np.ndarray, closed: bool, weights: np.ndarray | None = None,
           eps: float = EPS) -> tuple[np.ndarray, np.ndarray | None]:
    """Drop consecutive duplicate points (and the closing duplicate of a closed curve)."""
    if len(points) == 0:
        return points, weights
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > eps
    if closed and keep.sum() > 1:
        last = np.nonzero(keep)[0][-1]
        if np.linalg.norm(points[last] - points[0]) <= eps:
            keep[last] = False
    pts = points[keep]
    w = None if weights is None else weights[keep]
    return pts, w


def unit_steps(samples: int) -> np.ndarray:
    """``samples`` parameter values in ``[0, 1)`` (the segment end is the next segment's start)."""
    return np.arange(samples, dtype=float) / samples


def polyline(points: np.ndarray, closed: bool, samples: int) -> np.ndarray:
    """Sample the straight segments between consecutive points."""
    pts = np.vstack([points, points[:1]]) if closed else points
    a, b = pts[:-1], pts[1:]
    t = unit_steps(samples)[None, :, None]
    out = a[:, None, :] + (b - a)[:, None, :] * t
    return np.vstack([out.reshape(-1, 2), pts[-1:]])


def curve_function(func: Callable) -> Callable:
    """Wrap a core curve function ``func(points, closed, samples, **options)``.

    The core function only ever sees at least three distinct points. If it accepts a
    ``weights`` argument, per-point weights are passed through (default: all ones).
    """
    takes_weights = "weights" in inspect.signature(func).parameters

    @functools.wraps(func)
    def wrapper(points, closed: bool = False, samples: int = 16, **options) -> np.ndarray:
        pts = as_points(points)
        weights = options.pop("weights", None) if takes_weights else None
        if takes_weights:
            weights = (np.ones(len(pts)) if weights is None
                       else np.asarray(weights, dtype=float).reshape(-1))
            if len(weights) != len(pts):
                raise ValueError("need exactly one weight per control point")
        pts, weights = dedupe(pts, bool(closed), weights)
        samples = max(1, int(samples))
        if len(pts) == 0:
            return np.zeros((0, 2))
        if len(pts) == 1:
            return pts.copy()
        if len(pts) == 2:
            return polyline(pts, bool(closed), samples)
        if takes_weights:
            options["weights"] = weights
        return func(pts, bool(closed), samples, **options)

    return wrapper
