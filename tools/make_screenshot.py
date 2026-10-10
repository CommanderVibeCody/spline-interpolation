"""Render the README screenshots headlessly: ``python tools/make_screenshot.py``."""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pygame

from spline_interp.app import GRID, OVERLAY, App

RUNGE_POINTS = [(80 + i * 84, 420 + (-1) ** i * 40 + (i % 3) * 30) for i in range(11)]


def shoot(app: App, out: str) -> None:
    app.draw(flip=False)
    pygame.image.save(app.screen, out)
    print(f"saved {out}")


def main(out_dir: str = "docs") -> None:
    os.makedirs(out_dir, exist_ok=True)
    app = App(seed=7)
    app.selected = 5
    app.weights[5] = 4.0
    shoot(app, os.path.join(out_dir, "screenshot-overlay.png"))

    app.view = GRID
    app.selected = None
    shoot(app, os.path.join(out_dir, "screenshot-grid.png"))

    app.view = OVERLAY
    app.params.closed = True
    app.show_comb = True
    app.animate = True
    app.phase = 0.3
    for key in app.visible:
        app.visible[key] = key in ("cubic", "bspline", "catmull_rom")
    shoot(app, os.path.join(out_dir, "screenshot-closed-curvature.png"))

    app.params.closed = False
    app.show_comb = app.animate = False
    app.points = [list(p) for p in RUNGE_POINTS]
    app.weights = [1.0] * len(app.points)
    for key in app.visible:
        app.visible[key] = key in ("lagrange", "cubic", "linear")
    shoot(app, os.path.join(out_dir, "screenshot-runge.png"))
    pygame.quit()


if __name__ == "__main__":
    main()
