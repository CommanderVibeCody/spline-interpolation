"""Headless smoke tests for the Pygame front-end (uses SDL's dummy video driver)."""

import os

import numpy as np
import pygame
import pytest

from spline_interp import app as app_module
from spline_interp.app import CANVAS_W, GRID, HEIGHT, OVERLAY, PARAM_ROWS, App
from spline_interp.splines import METHODS


@pytest.fixture
def app(tmp_path):
    assert os.environ.get("SDL_VIDEODRIVER") == "dummy"
    application = App(screenshot_dir=str(tmp_path / "shots"), seed=1)
    yield application
    pygame.quit()


def press(app, key, mod=0):
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod, unicode=""))


def down(app, pos, button=1):
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button))


def up(app, pos, button=1):
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=button))


def move(app, pos):
    app.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(0, 0, 0)))


def click(app, pos, button=1):
    down(app, pos, button)
    up(app, pos, button)


def run_frames(app, n):
    for _ in range(n):
        app.step_frame()


def test_main_loop_runs_and_draws(app):
    run_frames(app, 5)
    assert app.running
    curves = app.curves()
    assert set(curves) == {m.key for m in METHODS}
    assert all(len(c) > 2 for c in curves.values())


def test_run_exits_on_quit_event(app, monkeypatch):
    monkeypatch.setattr(app_module, "FPS", 1000)
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    app.run()
    assert not app.running


def test_events_through_the_queue(app):
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(500, 790), button=1))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(500, 790), button=1))
    n = len(app.points)
    app.step_frame()
    assert len(app.points) == n + 1


def test_mouse_add_drag_delete_in_overlay(app):
    app.clear()
    click(app, (100, 100))
    click(app, (300, 200))
    click(app, (500, 120))
    assert app.points == [[100, 100], [300, 200], [500, 120]]
    assert app.selected == 2
    # drag the middle point
    down(app, (302, 201))
    assert app.dragging == 1
    move(app, (320, 400))
    move(app, (350, 450))
    up(app, (350, 450))
    assert app.points[1] == [350, 450] and app.dragging is None
    # dragging is clamped to the canvas
    down(app, (350, 450))
    move(app, (5000, -50))
    up(app, (5000, -50))
    assert app.points[1] == [CANVAS_W, 0]
    # hover highlight
    move(app, (101, 99))
    assert app.hovered == 0
    # right click deletes, right click on empty space does nothing
    click(app, (100, 100), button=3)
    assert len(app.points) == 2
    click(app, (900, 800), button=3)
    assert len(app.points) == 2
    run_frames(app, 2)


def test_mouse_in_grid_cells(app):
    app.clear()
    press(app, pygame.K_TAB)
    assert app.view == GRID
    rects = app.cell_rects()
    for i, rect in enumerate(rects):
        # a different spot in every cell, otherwise we would hit the point added before
        pos = (rect.x + 30 + 15 * i, rect.y + 60 + 12 * i)
        click(app, pos)
        assert len(app.points) == i + 1
        wx, wy = app.cell_transform(rect).to_world(pos)
        assert app.points[-1] == [wx, wy]
    run_frames(app, 2)
    # drag in a cell
    rect = rects[5]
    pos = tuple(int(v) for v in app.cell_transform(rect).to_screen(app.points[0])[0])
    down(app, pos)
    assert app.dragging == 0
    move(app, (pos[0] + 10, pos[1] + 10))
    up(app, (pos[0] + 10, pos[1] + 10))
    assert app.points[0] != app.points[1]
    # delete in a cell
    pos = tuple(round(v) for v in app.cell_transform(rect).to_screen(app.points[0])[0])
    click(app, pos, button=3)
    assert len(app.points) == len(rects) - 1
    run_frames(app, 2)


def test_wheel_changes_point_weight(app):
    pos = tuple(int(v) for v in app.points[2])
    move(app, pos)
    app.handle_wheel(1, pos)
    assert app.selected == 2 and app.weights[2] > 1.0
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=5))
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=5))
    assert app.weights[2] < 1.0
    for _ in range(40):
        app.handle_wheel(-1, pos)
    assert app.weights[2] == pytest.approx(0.1)
    # wheel over a parameter row
    before = app.params.samples
    app.handle_wheel(1, (CANVAS_W + 50, app._param_y(0) + 8))
    assert app.params.samples > before
    app.handle_wheel(1, (CANVAS_W + 50, 5))     # over nothing
    app.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1))
    run_frames(app, 1)


def test_every_key(app):
    keys = [
        pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7,
        pygame.K_8, pygame.K_9, pygame.K_0, pygame.K_TAB, pygame.K_g, pygame.K_o, pygame.K_p,
        pygame.K_t, pygame.K_k, pygame.K_a, pygame.K_s, pygame.K_v, pygame.K_e, pygame.K_UP,
        pygame.K_DOWN, pygame.K_RIGHT, pygame.K_LEFT, pygame.K_PLUS, pygame.K_MINUS,
        pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_KP_MINUS, pygame.K_RIGHTBRACKET,
        pygame.K_LEFTBRACKET, pygame.K_HOME, pygame.K_r, pygame.K_DELETE, pygame.K_BACKSPACE,
        pygame.K_c, pygame.K_r, pygame.K_x,
    ]
    for view in (OVERLAY, GRID):
        app.view = view
        for key in keys:
            for mod in (0, pygame.KMOD_SHIFT):
                press(app, key, mod)
                app.update(0.05)
                app.draw()
    assert os.listdir(app.screenshot_dir)
    press(app, pygame.K_ESCAPE)
    assert not app.running


def test_method_keys_toggle_visibility(app):
    before = dict(app.visible)
    for i, key in enumerate(app_module.METHOD_KEYS):
        press(app, key)
        assert app.visible[METHODS[i].key] != before[METHODS[i].key]
    press(app, pygame.K_v)
    assert all(app.visible.values())
    press(app, pygame.K_v)
    assert not any(app.visible.values())
    app.draw()


def test_parameter_keys(app):
    app.param_row = PARAM_ROWS.index("tension")
    press(app, pygame.K_RIGHT)
    assert app.params.tension == pytest.approx(0.1)
    press(app, pygame.K_RIGHT, pygame.KMOD_SHIFT)
    assert app.params.tension == pytest.approx(0.6)
    for _ in range(30):
        press(app, pygame.K_LEFT)
    assert app.params.tension == pytest.approx(-1.0)
    press(app, pygame.K_e)
    assert app.params.end_index == 1
    app.selected = None
    press(app, pygame.K_RIGHTBRACKET)          # no selected point: nothing happens
    assert app.weights == [1.0] * len(app.points)
    app.selected = 0
    press(app, pygame.K_RIGHTBRACKET)
    assert app.weights[0] == pytest.approx(1.1)
    press(app, pygame.K_HOME)
    assert app.params.tension == 0 and app.weights[0] == 1.0


def test_every_button(app):
    assert len(app.buttons) >= 10 + 2 * len(PARAM_ROWS) + len(METHODS) + 1
    for view in (OVERLAY, GRID):
        app.view = view
        for _ in range(2):
            for button in list(app.buttons):
                if button.label == "Clear (C)":
                    continue
                click(app, button.rect.center)
                app.update(0.1)
                app.draw()
                move(app, button.rect.center)
    labels = {b.label for b in app.buttons}
    assert {"Overlay (Tab)", "Grid (Tab)", "Clear (C)", "Random (R)", "Closed (O)", "Polygon (P)",
            "Tangents (T)", "Curvature (K)", "Animate (A)", "Save PNG (S)"} <= labels
    clear = next(b for b in app.buttons if b.label == "Clear (C)")
    click(app, clear.rect.center)
    assert app.points == []
    app.draw()


def test_view_buttons(app):
    grid_btn = next(b for b in app.buttons if b.label == "Grid (Tab)")
    overlay_btn = next(b for b in app.buttons if b.label == "Overlay (Tab)")
    click(app, grid_btn.rect.center)
    assert app.view == GRID
    click(app, overlay_btn.rect.center)
    assert app.view == OVERLAY


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("view", [OVERLAY, GRID])
def test_all_decorations_with_degenerate_point_sets(app, view, closed):
    app.view = view
    app.params.closed = closed
    app.show_tangents = app.show_comb = app.animate = True
    for key in app.visible:
        app.visible[key] = True
    for pts in ([], [[10, 10]], [[10, 10], [500, 500]], [[10, 10], [10, 10], [10, 10]],
                [[100, 100], [400, 100], [700, 100]], [[100, 100], [300, 400], [600, 120]]):
        app.points = [list(map(float, p)) for p in pts]
        app.weights = [1.0] * len(pts)
        app.selected = None
        for _ in range(3):
            app.update(0.4)
            app.draw()


def test_extreme_lagrange_is_drawable(app):
    app.points = [[float(x), float(400 + (-1) ** i * 300)] for i, x in enumerate(np.linspace(20, 980, 40))]
    app.weights = [1.0] * 40
    app.visible["lagrange"] = True
    app.show_comb = app.show_tangents = app.animate = True
    app.draw()
    app.view = GRID
    app.draw()


def test_random_points_are_deterministic_with_seed(tmp_path):
    a = App(screenshot_dir=str(tmp_path), seed=5)
    a.random_points()
    pa = [p[:] for p in a.points]
    b = App(screenshot_dir=str(tmp_path), seed=5)
    b.random_points()
    assert pa == b.points
    assert all(0 <= x <= CANVAS_W and 0 <= y <= HEIGHT for x, y in pa)
