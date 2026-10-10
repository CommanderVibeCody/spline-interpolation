"""Pygame UI helpers: colours, fonts, buttons and text."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pygame

BG = (20, 22, 28)
CANVAS_BG = (24, 26, 33)
PANEL_BG = (32, 35, 45)
CELL_BG = (27, 30, 38)
TEXT = (230, 232, 240)
TEXT_DIM = (150, 155, 170)
TEXT_FAINT = (95, 100, 115)
ACCENT = (98, 132, 255)
SELECTED_ROW = (48, 54, 74)
BUTTON = (52, 57, 72)
BUTTON_HOVER = (68, 74, 94)
INTERP = (120, 220, 140)
APPROX = (240, 170, 90)


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    action: Callable[[], None]
    is_active: Callable[[], bool] = field(default=lambda: False)
    centered: bool = True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, mouse) -> None:
        if self.is_active():
            colour = ACCENT
        elif self.rect.collidepoint(mouse):
            colour = BUTTON_HOVER
        else:
            colour = BUTTON
        pygame.draw.rect(surface, colour, self.rect, border_radius=6)
        text = font.render(self.label, True, TEXT)
        if self.centered:
            surface.blit(text, text.get_rect(center=self.rect.center))
        else:
            surface.blit(text, text.get_rect(midleft=(self.rect.x + 10, self.rect.centery)))

    def handle_click(self, pos) -> bool:
        if self.rect.collidepoint(pos):
            self.action()
            return True
        return False


class Fonts:
    def __init__(self) -> None:
        self.title = pygame.font.Font(None, 30)
        self.normal = pygame.font.Font(None, 22)
        self.small = pygame.font.Font(None, 19)


def draw_text(surface, font, text, pos, colour=TEXT) -> pygame.Rect:
    img = font.render(text, True, colour)
    return surface.blit(img, pos)


def draw_text_right(surface, font, text, right, y, colour=TEXT) -> pygame.Rect:
    img = font.render(text, True, colour)
    return surface.blit(img, (right - img.get_width(), y))
