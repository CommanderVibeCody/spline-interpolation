"""Drawing helpers for curves, control points and their decorations (pure Pygame)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from .splines import analysis

COLORS = {
    "linear": (150, 156, 172),
    "lagrange": (255, 99, 99),
    "catmull_rom": (255, 170, 60),
    "cardinal": (250, 225, 80),
    "kochanek_bartels": (150, 220, 90),
    "cubic": (70, 210, 190),
    "bezier": (90, 170, 255),
    "composite_bezier": (160, 130, 255),
    "bspline": (240, 120, 220),
    "nurbs": (255, 255, 255),
}
VARIANT_COLORS = ((255, 120, 70), (255, 200, 60), (120, 230, 120))

POINT = (240, 240, 245)
POINT_SELECTED = (98, 132, 255)
POINT_HOVER = (255, 196, 64)
POLYGON = (90, 96, 112)
GRID_LINE = (30, 33, 41)
COORD_LIMIT = 30000.0


@dataclass(frozen=True)
class Transform:
    """Maps world coordinates (the editing canvas) onto a screen rectangle."""
    scale: float = 1.0
    ox: float = 0.0
    oy: float = 0.0

    @classmethod
    def fit(cls, world_w: float, world_h: float, rect: pygame.Rect) -> Transform:
        scale = min(rect.w / world_w, rect.h / world_h)
        return cls(scale, rect.x + (rect.w - world_w * scale) / 2,
                   rect.y + (rect.h - world_h * scale) / 2)

    def to_screen(self, pts) -> np.ndarray:
        arr = np.asarray(pts, dtype=float).reshape(-1, 2)
        out = arr * self.scale + (self.ox, self.oy)
        return np.clip(out, -COORD_LIMIT, COORD_LIMIT)

    def to_world(self, pos) -> tuple[float, float]:
        return ((pos[0] - self.ox) / self.scale, (pos[1] - self.oy) / self.scale)


def lighten(color, amount: float = 0.5):
    return tuple(int(c + (255 - c) * amount) for c in color)


def darken(color, amount: float = 0.5):
    return tuple(int(c * (1 - amount)) for c in color)


def draw_background(surface: pygame.Surface, rect: pygame.Rect, spacing: int = 40) -> None:
    for x in range(rect.left, rect.right, spacing):
        pygame.draw.line(surface, GRID_LINE, (x, rect.top), (x, rect.bottom - 1))
    for y in range(rect.top, rect.bottom, spacing):
        pygame.draw.line(surface, GRID_LINE, (rect.left, y), (rect.right - 1, y))


def draw_curve(surface, curve, transform: Transform, color, width: int = 2) -> None:
    pts = transform.to_screen(curve)
    if len(pts) < 2:
        return
    seq = pts.tolist()
    if width <= 1:
        pygame.draw.aalines(surface, color, False, seq)
    else:
        pygame.draw.lines(surface, color, False, seq, width)


def draw_polygon(surface, points, transform: Transform, closed: bool) -> None:
    pts = transform.to_screen(points)
    if len(pts) >= 2:
        pygame.draw.lines(surface, POLYGON, closed and len(pts) > 2, pts.tolist(), 1)


def draw_points(surface, points, transform: Transform, selected=None, hovered=None,
                radius: int = 6, font=None, weights=None) -> None:
    pts = transform.to_screen(points)
    for i, (x, y) in enumerate(pts):
        color = POINT_SELECTED if i == selected else POINT_HOVER if i == hovered else POINT
        pygame.draw.circle(surface, (15, 16, 20), (int(x), int(y)), radius + 2)
        pygame.draw.circle(surface, color, (int(x), int(y)), radius)
        if font is not None:
            label = str(i)
            if weights is not None and abs(weights[i] - 1.0) > 1e-9:
                label += f"  w={weights[i]:.1f}"
            img = font.render(label, True, (190, 195, 210))
            surface.blit(img, (x + radius + 3, y - radius - 12))


def draw_tangents(surface, curve, transform: Transform, color, count: int = 14,
                  length: float = 30.0) -> None:
    """Short tangent ticks evenly spaced along the curve."""
    pts, tan = analysis.tangents(curve)
    if len(pts) < 2:
        return
    cum = analysis.cumulative_length(pts)
    total = cum[-1]
    if total <= 0:
        return
    idx = np.searchsorted(cum, np.linspace(0, total, count + 1)[:-1])
    idx = np.clip(idx, 0, len(pts) - 1)
    c = lighten(color, 0.35)
    for i in idx:
        p = pts[i]
        half = tan[i] * (length / 2) / transform.scale
        a, b = transform.to_screen([p - half, p + half])
        pygame.draw.line(surface, c, a, b, 1)


def draw_comb(surface, curve, transform: Transform, color, scale: float = 1500.0,
              max_len: float = 70.0, every: int = 2) -> None:
    """Curvature comb: normals scaled by curvature, with their tips connected."""
    pts, kappa, normals = analysis.curvature(curve)
    if len(pts) < 3:
        return
    lengths = np.clip(kappa * scale, -max_len, max_len)
    tips = pts - normals * lengths[:, None]
    sp, st = transform.to_screen(pts[::every]), transform.to_screen(tips[::every])
    c = darken(color, 0.35)
    for a, b in zip(sp, st):
        pygame.draw.line(surface, c, a, b, 1)
    if len(st) >= 2:
        pygame.draw.lines(surface, c, False, st.tolist(), 1)


def draw_marker(surface, curve, fraction: float, transform: Transform, color) -> None:
    """A dot moving along the curve (constant speed in arc length) with its tangent."""
    hit = analysis.point_at_fraction(curve, fraction)
    if hit is None:
        return
    pos, tan = hit
    tip = pos + tan * 26 / transform.scale
    a, b = transform.to_screen([pos, tip])
    pygame.draw.line(surface, color, a, b, 2)
    pygame.draw.circle(surface, (15, 16, 20), (int(a[0]), int(a[1])), 7)
    pygame.draw.circle(surface, color, (int(a[0]), int(a[1])), 5)
