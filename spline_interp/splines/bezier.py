"""Bezier curves: one global Bezier curve and a composite (piecewise) cubic Bezier."""

from __future__ import annotations

import numpy as np

from .common import curve_function, unit_steps


def de_casteljau(control: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate the Bezier curve with the given control points at all parameters ``t``."""
    pts = np.broadcast_to(control, (len(t),) + control.shape).copy()
    tt = t[:, None, None]
    while pts.shape[1] > 1:
        pts = (1 - tt) * pts[:, :-1] + tt * pts[:, 1:]
    return pts[:, 0]


@curve_function
def bezier(points: np.ndarray, closed: bool, samples: int) -> np.ndarray:
    """A single Bezier curve of degree ``n - 1`` using all points as its control polygon.

    It only passes through the first and last point and stays inside the convex hull
    of the control points. A closed curve repeats the first point as the last control point.
    """
    pts = np.vstack([points, points[:1]]) if closed else points
    total = samples * (len(pts) - 1)
    t = np.append(unit_steps(total), 1.0)
    return de_casteljau(pts, t)


@curve_function
def composite_bezier(points: np.ndarray, closed: bool, samples: int) -> np.ndarray:
    """Piecewise cubic Bezier: points ``0, 3, 6, ...`` are anchors, the rest are handles.

    The curve passes through the anchors; the two points in between are the handles of
    that segment. If the count does not divide evenly, the last segment has a lower
    degree (quadratic or linear). Joints are only C0 unless the handles are collinear.
    """
    pts = np.vstack([points, points[:1]]) if closed else points
    out = []
    start = 0
    while start < len(pts) - 1:
        seg = pts[start:start + 4]
        # A cubic piece spans three control-polygon edges, so give it proportionally more samples.
        out.append(de_casteljau(seg, unit_steps(samples * (len(seg) - 1))))
        start += len(seg) - 1
    out.append(pts[-1:])
    return np.vstack(out)
