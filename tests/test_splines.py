"""Tests for the pure curve functions."""

import numpy as np
import pytest

from spline_interp.params import Params
from spline_interp.splines import (
    METHODS,
    bezier,
    bspline,
    cardinal,
    catmull_rom,
    composite_bezier,
    cubic_spline,
    kochanek_bartels,
    lagrange,
    linear,
    nurbs,
)
from spline_interp.splines.bspline import basis_functions

RNG = np.random.default_rng(42)
POINTS = RNG.uniform(0, 800, size=(9, 2))
INTERPOLATING = [m for m in METHODS if m.interpolating]
APPROXIMATING = [m for m in METHODS if not m.interpolating]
ALL_FUNCS = [linear, lagrange, catmull_rom, cardinal, kochanek_bartels, cubic_spline,
             bezier, composite_bezier, bspline, nurbs]

PARAM_VARIANTS = [
    Params(),
    Params(alpha=0.0, tension=0.5, bias=-0.5, continuity=0.7, end_index=1, samples=7),
    Params(alpha=1.0, tension=-0.8, bias=0.9, continuity=-0.6, samples=3),
]


def min_dist(curve, point):
    return np.min(np.linalg.norm(curve - point, axis=1))


def is_collinear(curve, a, b, tol=1e-6):
    d = (b - a) / np.linalg.norm(b - a)
    rel = curve - a
    return np.max(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])) < tol


# ----------------------------------------------------------------------------- registry
def test_registry_has_ten_unique_methods():
    assert len(METHODS) == 10
    assert len({m.key for m in METHODS}) == 10
    assert {m.key for m in INTERPOLATING} == {
        "linear", "lagrange", "catmull_rom", "cardinal", "kochanek_bartels", "cubic"}


# ----------------------------------------------------------------------------- interpolation
@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("params", PARAM_VARIANTS)
@pytest.mark.parametrize("method", INTERPOLATING, ids=lambda m: m.key)
def test_interpolating_methods_pass_through_every_point(method, params, closed):
    params.closed = closed
    curve = method.evaluate(POINTS, params)
    for p in POINTS:
        assert min_dist(curve, p) < 1e-6


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_catmull_rom_all_alphas_interpolate(alpha):
    curve = catmull_rom(POINTS, samples=10, alpha=alpha)
    assert all(min_dist(curve, p) < 1e-6 for p in POINTS)
    assert curve.shape == ((len(POINTS) - 1) * 10 + 1, 2)


def test_samples_land_exactly_on_control_points():
    samples = 12
    for func in (linear, lagrange, catmull_rom, cardinal, kochanek_bartels, cubic_spline):
        curve = func(POINTS, samples=samples)
        np.testing.assert_allclose(curve[::samples], POINTS, atol=1e-6)


def test_cardinal_zero_tension_equals_uniform_catmull_rom():
    np.testing.assert_allclose(cardinal(POINTS, tension=0.0), catmull_rom(POINTS, alpha=0.0))


def test_kochanek_bartels_defaults_equal_cardinal():
    for t in (-0.5, 0.0, 0.7):
        np.testing.assert_allclose(kochanek_bartels(POINTS, tension=t), cardinal(POINTS, tension=t))


def test_tension_one_gives_zero_tangents():
    curve = cardinal(POINTS, samples=50, tension=1.0)
    # with zero tangents every segment still is the straight chord (eased in/out)
    for i in range(len(POINTS) - 1):
        seg = curve[i * 50:(i + 1) * 50 + 1]
        assert is_collinear(seg, POINTS[i], POINTS[i + 1], tol=1e-6)


def test_catmull_rom_variants_differ():
    a = catmull_rom(POINTS, alpha=0.0)
    b = catmull_rom(POINTS, alpha=0.5)
    c = catmull_rom(POINTS, alpha=1.0)
    assert not np.allclose(a, b) and not np.allclose(b, c)


def test_cubic_end_conditions():
    natural = cubic_spline(POINTS, end="natural", samples=5000)
    clamped = cubic_spline(POINTS, end="clamped", samples=5000)
    # closed curves ignore the end condition (periodic)
    np.testing.assert_allclose(cubic_spline(POINTS, closed=True, end="natural"),
                               cubic_spline(POINTS, closed=True, end="clamped"))
    assert not np.allclose(natural, clamped)
    # clamped: the start tangent points along the first chord
    d = clamped[1] - clamped[0]
    chord = POINTS[1] - POINTS[0]
    assert abs(d[0] * chord[1] - d[1] * chord[0]) / (np.linalg.norm(d) * np.linalg.norm(chord)) < 1e-3
    # natural: (almost) no curvature at the ends -> the first samples are nearly collinear
    v1, v2 = natural[1] - natural[0], natural[2] - natural[1]
    assert abs(v1[0] * v2[1] - v1[1] * v2[0]) / (np.linalg.norm(v1) * np.linalg.norm(v2)) < 1e-4
    with pytest.raises(ValueError):
        cubic_spline(POINTS, end="bogus")


def test_lagrange_shows_runge_oscillation():
    x = np.linspace(0, 1000, 15)
    y = 400 + np.where(np.arange(15) % 2 == 0, 30, -30)
    pts = np.stack([x, y], axis=1)
    lag = lagrange(pts)
    cub = cubic_spline(pts)
    assert np.ptp(lag[:, 1]) > 10 * np.ptp(pts[:, 1])
    assert np.ptp(cub[:, 1]) < 2 * np.ptp(pts[:, 1])
    assert np.all(np.isfinite(lagrange(RNG.uniform(0, 800, (60, 2)))))


# ----------------------------------------------------------------------------- approximating
def _start_tangent_ok(curve, a, b):
    d = curve[1] - curve[0]
    c = b - a
    cos = np.dot(d, c) / (np.linalg.norm(d) * np.linalg.norm(c))
    return cos > 0.9999


@pytest.mark.parametrize("func", [bezier, bspline, nurbs, composite_bezier])
def test_open_endpoint_interpolation_and_tangents(func):
    curve = func(POINTS, samples=200)
    np.testing.assert_allclose(curve[0], POINTS[0], atol=1e-9)
    np.testing.assert_allclose(curve[-1], POINTS[-1], atol=1e-9)
    assert _start_tangent_ok(curve, POINTS[0], POINTS[1])
    assert _start_tangent_ok(curve[::-1], POINTS[-1], POINTS[-2])


@pytest.mark.parametrize("func", [bezier, bspline])
def test_approximating_curves_do_not_hit_interior_points(func):
    curve = func(POINTS)
    assert any(min_dist(curve, p) > 1.0 for p in POINTS[1:-1])


def test_convex_hull_property_bezier_and_bspline():
    for func in (bezier, bspline):
        curve = func(POINTS)
        assert curve[:, 0].min() >= POINTS[:, 0].min() - 1e-9
        assert curve[:, 0].max() <= POINTS[:, 0].max() + 1e-9
        assert curve[:, 1].min() >= POINTS[:, 1].min() - 1e-9
        assert curve[:, 1].max() <= POINTS[:, 1].max() + 1e-9


def test_bezier_matches_bernstein_form():
    from math import comb
    pts = POINTS[:5]
    t = np.linspace(0, 1, 9)
    n = len(pts) - 1
    expected = sum(comb(n, i) * ((1 - t) ** (n - i) * t ** i)[:, None] * pts[i] for i in range(n + 1))
    curve = bezier(pts, samples=2)   # 4 segments * 2 = 8 steps -> t = k / 8
    np.testing.assert_allclose(curve, expected)


def test_composite_bezier_passes_through_anchors_only():
    curve = composite_bezier(POINTS[:7])
    for anchor in POINTS[:7:3]:
        assert min_dist(curve, anchor) < 1e-9
    assert min_dist(curve, POINTS[1]) > 1e-3


def test_bspline_basis_partition_of_unity():
    knots = np.concatenate([np.zeros(3), np.linspace(0, 1, 6), np.ones(3)])
    u = np.linspace(0, 1, 101)
    basis = basis_functions(knots, 3, u)
    assert basis.shape == (101, 8)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0)
    assert np.all(basis >= -1e-12)


def test_bspline_local_control():
    moved = POINTS.copy()
    moved[0] += 100
    a, b = bspline(POINTS, samples=10), bspline(moved, samples=10)
    changed = np.any(np.abs(a - b) > 1e-9, axis=1)
    assert changed[0] and not changed[-1]


def test_nurbs_unit_weights_equal_bspline_and_weights_pull():
    w = np.ones(len(POINTS))
    np.testing.assert_allclose(nurbs(POINTS, weights=w), bspline(POINTS))
    np.testing.assert_allclose(nurbs(POINTS, closed=True, weights=w), bspline(POINTS, closed=True))
    w[4] = 8.0
    heavy = nurbs(POINTS, weights=w)
    assert min_dist(heavy, POINTS[4]) < min_dist(bspline(POINTS), POINTS[4])
    with pytest.raises(ValueError):
        nurbs(POINTS, weights=[1.0, 2.0])


# ----------------------------------------------------------------------------- shared properties
@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("func", ALL_FUNCS, ids=lambda f: f.__name__)
def test_collinear_input_gives_straight_line(func, closed):
    t = np.array([0.0, 0.1, 0.35, 0.4, 0.7, 1.0])
    a, b = np.array([50.0, 80.0]), np.array([600.0, 420.0])
    pts = a + t[:, None] * (b - a)
    curve = func(pts, closed=closed)
    assert is_collinear(curve, a, b, tol=1e-6 * 1000)


@pytest.mark.parametrize("func", ALL_FUNCS, ids=lambda f: f.__name__)
def test_closed_curves_close(func):
    curve = func(POINTS, closed=True)
    np.testing.assert_allclose(curve[0], curve[-1], atol=1e-6)
    open_curve = func(POINTS, closed=False)
    assert np.linalg.norm(open_curve[0] - open_curve[-1]) > 1.0


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("method", METHODS, ids=lambda m: m.key)
def test_deterministic_and_finite(method, closed):
    params = Params(closed=closed)
    a = method.evaluate(POINTS.tolist(), params, [1.0] * len(POINTS))
    b = method.evaluate(POINTS.copy(), params, [1.0] * len(POINTS))
    np.testing.assert_array_equal(a, b)
    assert a.ndim == 2 and a.shape[1] == 2 and np.all(np.isfinite(a))


@pytest.mark.parametrize("func", ALL_FUNCS, ids=lambda f: f.__name__)
def test_input_is_not_modified(func):
    pts = POINTS.copy()
    func(pts, closed=True)
    np.testing.assert_array_equal(pts, POINTS)


# ----------------------------------------------------------------------------- edge cases
@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("func", ALL_FUNCS, ids=lambda f: f.__name__)
def test_zero_one_two_points(func, closed):
    assert func([], closed=closed).shape == (0, 2)
    one = func([[3.0, 4.0]], closed=closed)
    np.testing.assert_array_equal(one, [[3.0, 4.0]])
    two = func([[0.0, 0.0], [10.0, 0.0]], closed=closed, samples=5)
    np.testing.assert_allclose(two[0], [0, 0])
    assert np.all(two[:, 1] == 0.0)
    np.testing.assert_allclose(two[:, 0].max(), 10.0)


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("func", ALL_FUNCS, ids=lambda f: f.__name__)
def test_duplicate_points(func, closed):
    pts = np.array([[0, 0], [0, 0], [100, 50], [100, 50], [100, 50], [200, -20], [300, 80], [0, 0]],
                   dtype=float)
    curve = func(pts, closed=closed)
    assert np.all(np.isfinite(curve))
    # all duplicates -> a single point
    same = func([[5.0, 5.0]] * 4, closed=closed)
    np.testing.assert_array_equal(same, [[5.0, 5.0]])


@pytest.mark.parametrize("method", INTERPOLATING, ids=lambda m: m.key)
def test_duplicates_still_interpolate(method):
    pts = np.array([[0, 0], [100, 50], [100, 50], [200, -20], [300, 80]], dtype=float)
    curve = method.evaluate(pts, Params())
    assert all(min_dist(curve, p) < 1e-6 for p in pts)


def test_three_points_minimum_paths():
    pts = [[0, 0], [100, 100], [200, 0]]
    for func in ALL_FUNCS:
        for closed in (False, True):
            assert np.all(np.isfinite(func(pts, closed=closed)))


def test_samples_are_clamped_to_at_least_one():
    curve = catmull_rom(POINTS, samples=0)
    np.testing.assert_allclose(curve, POINTS)
