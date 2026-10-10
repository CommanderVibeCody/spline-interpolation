"""Tests for arc length, tangents and curvature of sampled curves."""

import numpy as np

from spline_interp.splines import analysis, catmull_rom


def circle(r=100.0, n=721):
    t = np.linspace(0, 2 * np.pi, n)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def test_arc_length_of_line_and_circle():
    assert analysis.arc_length([[0, 0], [3, 4]]) == 5.0
    assert abs(analysis.arc_length(circle()) - 2 * np.pi * 100) < 0.01
    assert analysis.arc_length(np.zeros((0, 2))) == 0.0
    assert analysis.arc_length([[1, 1]]) == 0.0


def test_curvature_of_circle_is_one_over_r():
    _, kappa, normals = analysis.curvature(circle(r=50.0))
    np.testing.assert_allclose(kappa, 1 / 50.0, rtol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_curvature_of_line_is_zero_and_short_inputs():
    _, kappa, _ = analysis.curvature([[0, 0], [1, 1], [2, 2], [5, 5]])
    np.testing.assert_allclose(kappa, 0.0, atol=1e-12)
    _pts, kappa, _ = analysis.curvature([[0, 0], [1, 0]])
    assert len(kappa) == 2


def test_point_at_fraction():
    line = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    pos, tan = analysis.point_at_fraction(line, 0.25)
    np.testing.assert_allclose(pos, [5, 0])
    np.testing.assert_allclose(tan, [1, 0])
    pos, tan = analysis.point_at_fraction(line, 1.0)
    np.testing.assert_allclose(pos, [10, 10])
    assert analysis.point_at_fraction(np.zeros((0, 2)), 0.5) is None
    pos, _ = analysis.point_at_fraction([[2.0, 3.0]], 0.5)
    np.testing.assert_allclose(pos, [2, 3])


def test_tangents_are_unit_vectors_and_handle_duplicates():
    curve = catmull_rom([[0, 0], [100, 50], [200, 0], [300, 60]], samples=10)
    curve = np.vstack([curve[:5], curve[4:]])     # inject a duplicate sample
    pts, tan = analysis.tangents(curve)
    np.testing.assert_allclose(np.linalg.norm(tan, axis=1), 1.0)
    assert len(pts) == len(curve) - 1
