"""Geometric analysis of sampled curves: arc length, tangents, curvature."""

from __future__ import annotations

import numpy as np


def _clean(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=float).reshape(-1, 2)
    if len(curve) < 2:
        return curve
    keep = np.ones(len(curve), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(curve, axis=0), axis=1) > 1e-9
    return curve[keep]


def cumulative_length(curve: np.ndarray) -> np.ndarray:
    """Arc length from the start to every sample."""
    curve = np.asarray(curve, dtype=float).reshape(-1, 2)
    if len(curve) == 0:
        return np.zeros(0)
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def arc_length(curve: np.ndarray) -> float:
    """Total length of the sampled polyline."""
    cum = cumulative_length(curve)
    return float(cum[-1]) if len(cum) else 0.0


def point_at_fraction(curve: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Position and unit tangent at ``fraction`` (0..1) of the arc length."""
    curve = _clean(curve)
    if len(curve) == 0:
        return None
    if len(curve) == 1:
        return curve[0], np.array([1.0, 0.0])
    cum = cumulative_length(curve)
    s = float(np.clip(fraction, 0.0, 1.0)) * cum[-1]
    i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(curve) - 2))
    seg = curve[i + 1] - curve[i]
    length = np.linalg.norm(seg)
    local = (s - cum[i]) / length
    return curve[i] + seg * local, seg / length


def _is_loop(curve: np.ndarray) -> bool:
    return len(curve) > 4 and np.linalg.norm(curve[0] - curve[-1]) < 1e-6


def _padded(curve: np.ndarray) -> tuple[np.ndarray, slice]:
    """Pad a closed loop with its wrap-around neighbours so differences are periodic."""
    if _is_loop(curve):
        return np.vstack([curve[-3:-1], curve, curve[1:3]]), slice(2, -2)
    return curve, slice(None)


def tangents(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(points, unit_tangents)`` for the de-duplicated samples."""
    curve = _clean(curve)
    if len(curve) < 2:
        return curve, np.zeros_like(curve)
    padded, inner = _padded(curve)
    d = np.gradient(padded, axis=0)[inner]
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    return curve, d / np.maximum(norm, 1e-12)


def curvature(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed curvature at every sample: returns ``(points, kappa, unit_normals)``.

    Derivatives are taken with respect to arc length, so the result does not depend on
    how the samples are spaced. Closed loops (first sample == last) wrap around.
    """
    curve = _clean(curve)
    if len(curve) < 3:
        return curve, np.zeros(len(curve)), np.zeros_like(curve)
    padded, inner = _padded(curve)
    s = cumulative_length(padded)
    dx, dy = np.gradient(padded[:, 0], s), np.gradient(padded[:, 1], s)
    ddx, ddy = np.gradient(dx, s), np.gradient(dy, s)
    dx, dy, ddx, ddy = dx[inner], dy[inner], ddx[inner], ddy[inner]
    speed = np.maximum(np.hypot(dx, dy), 1e-12)
    kappa = (dx * ddy - dy * ddx) / speed ** 3
    normals = np.stack([-dy, dx], axis=1) / speed[:, None]
    return curve, kappa, normals
