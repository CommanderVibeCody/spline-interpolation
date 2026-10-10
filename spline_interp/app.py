"""Pygame front-end: overlay view, side-by-side grid view and the parameter panel."""

from __future__ import annotations

import os
import random
import time

import numpy as np
import pygame

from . import rendering, ui
from .params import RANGES, Params, alpha_name
from .rendering import COLORS, VARIANT_COLORS, Transform
from .splines import (
    END_CONDITIONS,
    METHODS,
    Method,
    analysis,
    catmull_rom,
    cubic_spline,
)

# ----------------------------------------------------------------------------- layout
CANVAS_W, HEIGHT = 1000, 840
PANEL_W = 340
WIDTH = CANVAS_W + PANEL_W
FPS = 60

GRID_COLS, GRID_ROWS = 4, 3
CELL_LABEL_H = 22
HIT_RADIUS = 10           # pixels
WEIGHT_RANGE = (RANGES["weight"].lo, RANGES["weight"].hi)

OVERLAY, GRID = "overlay", "grid"
PARAM_ROWS = ("samples", "alpha", "tension", "bias", "continuity", "end", "weight")
METHOD_KEYS = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5,
               pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0)

DEFAULT_POINTS = ((130, 600), (230, 260), (380, 200), (470, 520),
                  (600, 640), (700, 330), (820, 230), (880, 560))


class App:
    def __init__(self, screenshot_dir: str = "screenshots", seed: int | None = None) -> None:
        pygame.init()
        pygame.display.set_caption("Spline Interpolation - comparison")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.fonts = ui.Fonts()
        self.rng = random.Random(seed)
        self.screenshot_dir = screenshot_dir

        self.params = Params()
        self.points: list[list[float]] = [list(p) for p in DEFAULT_POINTS]
        self.weights: list[float] = [1.0] * len(self.points)
        self.visible = {m.key: True for m in METHODS}
        self.visible["lagrange"] = False          # it tends to dominate the default view
        self.visible["bezier"] = False
        self.view = OVERLAY
        self.selected: int | None = None
        self.hovered: int | None = None
        self.dragging: int | None = None
        self.drag_transform: Transform | None = None
        self.param_row = 0
        self.show_polygon = True
        self.show_tangents = False
        self.show_comb = False
        self.animate = False
        self.phase = 0.0
        self.running = True
        self.status = ""
        self._cache_key = None
        self._curves: dict[str, np.ndarray] = {}
        self.buttons: list[ui.Button] = []
        self._build_buttons()

    # ------------------------------------------------------------------------- state
    @property
    def point_array(self) -> np.ndarray:
        return np.array(self.points, dtype=float).reshape(-1, 2)

    def curves(self) -> dict[str, np.ndarray]:
        """Sampled curves of every method, cached until points or parameters change."""
        key = (tuple(map(tuple, self.points)), tuple(self.weights), tuple(vars(self.params).items()))
        if key != self._cache_key:
            pts = self.point_array
            self._curves = {m.key: m.evaluate(pts, self.params, self.weights) for m in METHODS}
            self._cache_key = key
        return self._curves

    def add_point(self, world) -> None:
        self.points.append([float(world[0]), float(world[1])])
        self.weights.append(1.0)
        self.selected = len(self.points) - 1

    def delete_point(self, index: int | None) -> None:
        if index is None or not 0 <= index < len(self.points):
            return
        del self.points[index]
        del self.weights[index]
        if self.selected is not None:
            if self.selected == index:
                self.selected = None
            elif self.selected > index:
                self.selected -= 1
        self.hovered = None
        self.dragging = None

    def clear(self) -> None:
        self.points.clear()
        self.weights.clear()
        self.selected = self.hovered = self.dragging = None

    def random_points(self, count: int | None = None) -> None:
        count = count or self.rng.randint(5, 9)
        margin = 110
        xs = sorted(self.rng.uniform(margin, CANVAS_W - margin) for _ in range(count))
        self.points = [[x, self.rng.uniform(margin, HEIGHT - margin)] for x in xs]
        self.weights = [1.0] * count
        self.selected = self.hovered = self.dragging = None

    def toggle_closed(self) -> None:
        self.params.closed = not self.params.closed

    def set_view(self, view: str) -> None:
        self.view = view

    def toggle_view(self) -> None:
        self.view = GRID if self.view == OVERLAY else OVERLAY

    def toggle_method(self, index: int) -> None:
        if 0 <= index < len(METHODS):
            key = METHODS[index].key
            self.visible[key] = not self.visible[key]

    def toggle_all(self) -> None:
        show = not all(self.visible.values())
        for key in self.visible:
            self.visible[key] = show

    def set_weight(self, index: int | None, value: float) -> None:
        if index is not None and 0 <= index < len(self.weights):
            self.weights[index] = round(float(np.clip(value, *WEIGHT_RANGE)), 3)

    def adjust_param(self, name: str, direction: int, big: bool = False) -> None:
        if name == "end":
            self.params.end_index = (self.params.end_index + direction) % len(END_CONDITIONS)
            return
        rng = RANGES[name]
        step = (rng.big_step if big else rng.step) * direction
        if name == "weight":
            if self.selected is not None:
                self.set_weight(self.selected, self.weights[self.selected] + step)
            return
        value = float(np.clip(getattr(self.params, name) + step, rng.lo, rng.hi))
        if name == "samples":
            self.params.samples = round(value)
        else:
            setattr(self.params, name, round(value, 3))

    def reset_params(self) -> None:
        closed = self.params.closed
        self.params = Params(closed=closed)
        self.weights = [1.0] * len(self.points)

    def save_png(self) -> str:
        os.makedirs(self.screenshot_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.screenshot_dir, f"splines-{stamp}-{len(os.listdir(self.screenshot_dir))}.png")
        self.draw(flip=False)
        pygame.image.save(self.screen, path)
        self.status = f"saved {path}"
        return path

    # ------------------------------------------------------------------------- layout
    def cell_rects(self) -> list[pygame.Rect]:
        w, h = CANVAS_W // GRID_COLS, HEIGHT // GRID_ROWS
        return [pygame.Rect(c * w, r * h, w, h) for r in range(GRID_ROWS) for c in range(GRID_COLS)]

    def cell_transform(self, rect: pygame.Rect) -> Transform:
        inner = pygame.Rect(rect.x + 4, rect.y + CELL_LABEL_H, rect.w - 8, rect.h - CELL_LABEL_H - 4)
        return Transform.fit(CANVAS_W, HEIGHT, inner)

    def transform_at(self, pos) -> Transform | None:
        """Transform of the canvas area under ``pos`` (None over the side panel)."""
        if pos[0] >= CANVAS_W or not (0 <= pos[1] < HEIGHT) or pos[0] < 0:
            return None
        if self.view == OVERLAY:
            return Transform()
        for rect in self.cell_rects():
            if rect.collidepoint(pos):
                return self.cell_transform(rect)
        return None

    def point_at(self, pos, transform: Transform) -> int | None:
        if not self.points:
            return None
        screen = transform.to_screen(self.point_array)
        d = np.hypot(screen[:, 0] - pos[0], screen[:, 1] - pos[1])
        i = int(np.argmin(d))
        return i if d[i] <= HIT_RADIUS else None

    def _build_buttons(self) -> None:
        x0 = CANVAS_W + 16
        bw, bh, gap = 150, 28, 8
        col2 = x0 + bw + gap

        def btn(x, y, label, action, active=lambda: False, w=bw):
            self.buttons.append(ui.Button(pygame.Rect(x, y, w, bh), label, action, active))

        btn(x0, 70, "Overlay (Tab)", lambda: self.set_view(OVERLAY), lambda: self.view == OVERLAY)
        btn(col2, 70, "Grid (Tab)", lambda: self.set_view(GRID), lambda: self.view == GRID)
        btn(x0, 104, "Clear (C)", self.clear)
        btn(col2, 104, "Random (R)", self.random_points)
        btn(x0, 138, "Closed (O)", self.toggle_closed, lambda: self.params.closed)
        btn(col2, 138, "Polygon (P)", self._toggle("show_polygon"), lambda: self.show_polygon)
        btn(x0, 172, "Tangents (T)", self._toggle("show_tangents"), lambda: self.show_tangents)
        btn(col2, 172, "Curvature (K)", self._toggle("show_comb"), lambda: self.show_comb)
        btn(x0, 206, "Animate (A)", self._toggle("animate"), lambda: self.animate)
        btn(col2, 206, "Save PNG (S)", self.save_png)

        for i, name in enumerate(PARAM_ROWS):
            y = self._param_y(i)
            for j, (label, direction) in enumerate((("-", -1), ("+", 1))):
                rect = pygame.Rect(CANVAS_W + 262 + j * 34, y - 2, 28, 22)
                self.buttons.append(ui.Button(rect, label, self._adjuster(name, direction)))

        for i, method in enumerate(METHODS):
            rect = pygame.Rect(x0 - 4, self._legend_y(i) - 3, PANEL_W - 24, 22)
            self.buttons.append(ui.Button(rect, "", self._method_toggler(i)))
        btn(x0, self._legend_y(len(METHODS)) + 4, "Show all / none (V)", self.toggle_all, w=2 * bw + gap)

    def _toggle(self, attr: str):
        def action() -> None:
            setattr(self, attr, not getattr(self, attr))
        return action

    def _adjuster(self, name: str, direction: int):
        def action() -> None:
            self.param_row = PARAM_ROWS.index(name)
            self.adjust_param(name, direction)
        return action

    def _method_toggler(self, index: int):
        return lambda: self.toggle_method(index)

    @staticmethod
    def _param_y(i: int) -> int:
        return 270 + i * 26

    @staticmethod
    def _legend_y(i: int) -> int:
        return 482 + i * 24

    # ------------------------------------------------------------------------- events
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            self.handle_key(event.key, getattr(event, "mod", 0))
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.handle_mouse_down(event.pos, event.button)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = None
                self.drag_transform = None
        elif event.type == pygame.MOUSEMOTION:
            self.handle_mouse_motion(event.pos)
        elif event.type == pygame.MOUSEWHEEL:
            self.handle_wheel(event.y)

    def handle_key(self, key: int, mod: int = 0) -> None:
        big = bool(mod & pygame.KMOD_SHIFT)
        if key in METHOD_KEYS:
            self.toggle_method(METHOD_KEYS.index(key))
        elif key == pygame.K_ESCAPE:
            self.running = False
        elif key in (pygame.K_TAB, pygame.K_g):
            self.toggle_view()
        elif key == pygame.K_o:
            self.toggle_closed()
        elif key == pygame.K_c:
            self.clear()
        elif key == pygame.K_r:
            self.random_points()
        elif key == pygame.K_p:
            self.show_polygon = not self.show_polygon
        elif key == pygame.K_t:
            self.show_tangents = not self.show_tangents
        elif key == pygame.K_k:
            self.show_comb = not self.show_comb
        elif key == pygame.K_a:
            self.animate = not self.animate
        elif key == pygame.K_s:
            self.save_png()
        elif key == pygame.K_v:
            self.toggle_all()
        elif key == pygame.K_e:
            self.adjust_param("end", 1)
        elif key == pygame.K_UP:
            self.param_row = (self.param_row - 1) % len(PARAM_ROWS)
        elif key == pygame.K_DOWN:
            self.param_row = (self.param_row + 1) % len(PARAM_ROWS)
        elif key in (pygame.K_RIGHT, pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.adjust_param(PARAM_ROWS[self.param_row], 1, big)
        elif key in (pygame.K_LEFT, pygame.K_MINUS, pygame.K_KP_MINUS):
            self.adjust_param(PARAM_ROWS[self.param_row], -1, big)
        elif key == pygame.K_RIGHTBRACKET:
            self.adjust_param("weight", 1, big)
        elif key == pygame.K_LEFTBRACKET:
            self.adjust_param("weight", -1, big)
        elif key in (pygame.K_DELETE, pygame.K_BACKSPACE):
            self.delete_point(self.selected)
        elif key == pygame.K_HOME:
            self.reset_params()

    def handle_mouse_down(self, pos, button: int) -> None:
        if button in (4, 5):          # legacy wheel events
            self.handle_wheel(1 if button == 4 else -1, pos)
            return
        if button == 1:
            for b in self.buttons:
                if b.handle_click(pos):
                    return
        transform = self.transform_at(pos)
        if transform is None:
            return
        hit = self.point_at(pos, transform)
        if button == 1:
            if hit is None:
                self.add_point(transform.to_world(pos))
                hit = len(self.points) - 1
            self.selected = hit
            self.dragging = hit
            self.drag_transform = transform
        elif button == 3 and hit is not None:
            self.delete_point(hit)

    def handle_mouse_motion(self, pos) -> None:
        if self.dragging is not None and self.drag_transform is not None:
            x, y = self.drag_transform.to_world(pos)
            self.points[self.dragging] = [float(np.clip(x, 0, CANVAS_W)), float(np.clip(y, 0, HEIGHT))]
            return
        transform = self.transform_at(pos)
        self.hovered = self.point_at(pos, transform) if transform else None

    def handle_wheel(self, dy: int, pos=None) -> None:
        pos = pos if pos is not None else pygame.mouse.get_pos()
        transform = self.transform_at(pos)
        target = self.point_at(pos, transform) if transform else None
        if target is not None:
            self.selected = target
            self.set_weight(target, self.weights[target] * (1.25 if dy > 0 else 0.8))
        elif pos[0] >= CANVAS_W:
            for i in range(len(PARAM_ROWS)):
                if abs(pos[1] - (self._param_y(i) + 8)) <= 12:
                    self.param_row = i
                    self.adjust_param(PARAM_ROWS[i], 1 if dy > 0 else -1)

    # ------------------------------------------------------------------------- loop
    def update(self, dt: float) -> None:
        if self.animate:
            self.phase = (self.phase + dt * 0.12) % 1.0

    def step_frame(self, dt: float = 1 / FPS) -> None:
        for event in pygame.event.get():
            self.handle_event(event)
        self.update(dt)
        self.draw()

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.step_frame(dt)
        pygame.quit()

    # ------------------------------------------------------------------------- drawing
    def draw(self, flip: bool = True) -> None:
        self.screen.fill(ui.BG)
        if self.view == OVERLAY:
            self._draw_overlay()
        else:
            self._draw_grid()
        self._draw_panel()
        if flip:
            pygame.display.flip()

    def _decorate(self, curve, transform: Transform, color) -> None:
        if self.show_comb:
            rendering.draw_comb(self.screen, curve, transform, color)
        if self.show_tangents:
            rendering.draw_tangents(self.screen, curve, transform, color)

    def _draw_overlay(self) -> None:
        canvas = pygame.Rect(0, 0, CANVAS_W, HEIGHT)
        self.screen.fill(ui.CANVAS_BG, canvas)
        rendering.draw_background(self.screen, canvas)
        t = Transform()
        pts = self.point_array
        if self.show_polygon:
            rendering.draw_polygon(self.screen, pts, t, self.params.closed)
        curves = self.curves()
        shown = [m for m in METHODS if self.visible[m.key]]
        for m in shown:
            self._decorate(curves[m.key], t, COLORS[m.key])
        for m in shown:
            rendering.draw_curve(self.screen, curves[m.key], t, COLORS[m.key], 2)
        if self.animate:
            for m in shown:
                rendering.draw_marker(self.screen, curves[m.key], self.phase, t, COLORS[m.key])
        rendering.draw_points(self.screen, pts, t, self.selected, self.hovered, 6,
                              self.fonts.small, self.weights)
        if len(self.points) < 2:
            ui.draw_text(self.screen, self.fonts.normal,
                         "Left click to add control points - or press R for random points",
                         (24, HEIGHT - 34), ui.TEXT_DIM)

    def _cell_title(self, rect, title, color, tag=None, length=None) -> None:
        ui.draw_text(self.screen, self.fonts.small, title, (rect.x + 8, rect.y + 5), color)
        if tag is not None:
            text, colour = ("interpolating", ui.INTERP) if tag else ("approximating", ui.APPROX)
            y = rect.bottom - 22
            ui.draw_text(self.screen, self.fonts.small, text, (rect.x + 8, y), colour)
            if length is not None:
                ui.draw_text_right(self.screen, self.fonts.small, f"length {length:.0f}px",
                                   rect.right - 8, y, ui.TEXT_FAINT)

    def _draw_cell(self, rect, transform, curves_and_colors) -> None:
        pts = self.point_array
        if self.show_polygon:
            rendering.draw_polygon(self.screen, pts, transform, self.params.closed)
        for curve, color in curves_and_colors:
            self._decorate(curve, transform, color)
        for curve, color in curves_and_colors:
            rendering.draw_curve(self.screen, curve, transform, color, 2)
        if self.animate:
            for curve, color in curves_and_colors:
                rendering.draw_marker(self.screen, curve, self.phase, transform, color)
        rendering.draw_points(self.screen, pts, transform, self.selected, self.hovered, 3)

    def _draw_grid(self) -> None:
        curves = self.curves()
        pts = self.point_array
        rects = self.cell_rects()
        for i, rect in enumerate(rects):
            inner = rect.inflate(-4, -4)
            self.screen.fill(ui.CELL_BG, inner)
            clip = self.screen.get_clip()
            self.screen.set_clip(inner)
            transform = self.cell_transform(rect)
            if i < len(METHODS):
                m: Method = METHODS[i]
                curve = curves[m.key]
                self._draw_cell(rect, transform, [(curve, COLORS[m.key])])
                self._cell_title(rect, f"{(i + 1) % 10}  {m.name}", COLORS[m.key], m.interpolating,
                                 analysis.arc_length(curve))
            elif i == len(METHODS):
                variants = [(catmull_rom(pts, self.params.closed, self.params.samples, alpha=a), c)
                            for a, c in zip((0.0, 0.5, 1.0), VARIANT_COLORS)]
                self._draw_cell(rect, transform, variants)
                self._cell_title(rect, "Catmull-Rom alpha variants",
                                 ui.TEXT)
                self._variant_legend(rect, ("uniform", "centripetal", "chordal"))
            else:
                # closed curves always use periodic end conditions
                ends = ("natural",) if self.params.closed else END_CONDITIONS
                variants = [(cubic_spline(pts, self.params.closed, self.params.samples, end=e), c)
                            for e, c in zip(ends, VARIANT_COLORS)]
                self._draw_cell(rect, transform, variants)
                self._cell_title(rect, "Cubic spline end conditions", ui.TEXT)
                self._variant_legend(rect, ends if not self.params.closed else ("periodic (closed)",))
            self.screen.set_clip(clip)

    def _variant_legend(self, rect, names) -> None:
        y = rect.bottom - 8 - 16 * len(names)
        for name, color in zip(names, VARIANT_COLORS):
            pygame.draw.line(self.screen, color, (rect.x + 10, y + 7), (rect.x + 26, y + 7), 3)
            ui.draw_text(self.screen, self.fonts.small, name, (rect.x + 32, y), color)
            y += 16

    def _param_value(self, name: str) -> str:
        p = self.params
        if name == "samples":
            return str(p.samples)
        if name == "alpha":
            return f"{p.alpha:.2f} {alpha_name(p.alpha)}"
        if name == "end":
            return "periodic" if p.closed else END_CONDITIONS[p.end_index]
        if name == "weight":
            return "-" if self.selected is None else f"{self.weights[self.selected]:.2f}"
        return f"{getattr(p, name):+.1f}"

    def _draw_panel(self) -> None:
        x = CANVAS_W
        self.screen.fill(ui.PANEL_BG, pygame.Rect(x, 0, PANEL_W, HEIGHT))
        f = self.fonts
        ui.draw_text(self.screen, f.title, "Spline Interpolation", (x + 16, 16))
        mode = "Overlay" if self.view == OVERLAY else "Side-by-side grid"
        ui.draw_text(self.screen, f.small, f"{mode}  |  {len(self.points)} points  |  "
                     f"{'closed' if self.params.closed else 'open'}", (x + 16, 46), ui.TEXT_DIM)
        mouse = pygame.mouse.get_pos()
        for b in self.buttons:
            if b.label:
                b.draw(self.screen, f.small, mouse)

        ui.draw_text(self.screen, f.normal, "Parameters", (x + 16, 246), ui.TEXT_DIM)
        for i, name in enumerate(PARAM_ROWS):
            y = self._param_y(i)
            if i == self.param_row:
                pygame.draw.rect(self.screen, ui.SELECTED_ROW, (x + 8, y - 4, PANEL_W - 16, 25),
                                 border_radius=4)
            label = "Cubic end" if name == "end" else RANGES[name].label
            if name == "weight":
                label += " (sel. pt)"
            ui.draw_text(self.screen, f.small, label, (x + 16, y + 1))
            ui.draw_text_right(self.screen, f.small, self._param_value(name), x + 254, y + 1, ui.TEXT_DIM)
        for b in self.buttons:
            if b.label in ("-", "+"):
                b.draw(self.screen, f.normal, mouse)

        ui.draw_text(self.screen, f.normal, "Methods", (x + 16, 458), ui.TEXT_DIM)
        ui.draw_text_right(self.screen, f.small, "arc length", x + PANEL_W - 16, 461, ui.TEXT_FAINT)
        curves = self.curves()
        for i, m in enumerate(METHODS):
            y = self._legend_y(i)
            on = self.visible[m.key]
            color = COLORS[m.key] if on else ui.TEXT_FAINT
            box = pygame.Rect(x + 16, y + 1, 14, 14)
            if on:
                pygame.draw.rect(self.screen, color, box, border_radius=3)
            else:
                pygame.draw.rect(self.screen, color, box, 1, border_radius=3)
            ui.draw_text(self.screen, f.small, f"{(i + 1) % 10}", (x + 38, y + 2), ui.TEXT_FAINT)
            ui.draw_text(self.screen, f.small, m.name, (x + 52, y + 2), ui.TEXT if on else ui.TEXT_FAINT)
            tag, tag_col = ("I", ui.INTERP) if m.interpolating else ("A", ui.APPROX)
            ui.draw_text(self.screen, f.small, tag, (x + 236, y + 2), tag_col if on else ui.TEXT_FAINT)
            ui.draw_text_right(self.screen, f.small, f"{analysis.arc_length(curves[m.key]):.0f}",
                               x + PANEL_W - 16, y + 2, ui.TEXT_DIM if on else ui.TEXT_FAINT)

        y = self._legend_y(len(METHODS)) + 40
        r = ui.draw_text(self.screen, f.small, "I = interpolating", (x + 16, y), ui.INTERP)
        ui.draw_text(self.screen, f.small, "A = approximating", (r.right + 24, y), ui.APPROX)
        help_lines = [
            "Left: add / drag   Right: delete   Wheel: weight",
            "1-0 toggle methods   Arrows: parameters",
            "E end condition   Del delete selected   Esc quit",
        ]
        if self.status:
            help_lines[-1] = self.status[-50:]
        for j, line in enumerate(help_lines):
            ui.draw_text(self.screen, f.small, line, (x + 16, y + 20 + j * 18),
                         ui.TEXT_DIM if self.status and j == 2 else ui.TEXT_FAINT)
