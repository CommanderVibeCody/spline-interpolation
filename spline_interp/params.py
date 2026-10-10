"""Tunable parameters shared by all curve methods."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Params:
    samples: int = 24        # samples per segment
    alpha: float = 0.5       # Catmull-Rom parameterisation (0 uniform, .5 centripetal, 1 chordal)
    tension: float = 0.0     # Cardinal / Kochanek-Bartels
    bias: float = 0.0        # Kochanek-Bartels
    continuity: float = 0.0  # Kochanek-Bartels
    end_index: int = 0       # cubic spline end condition (index into END_CONDITIONS)
    closed: bool = False


@dataclass(frozen=True)
class Range:
    name: str
    label: str
    lo: float
    hi: float
    step: float
    big_step: float


RANGES = {
    "samples": Range("samples", "Samples / segment", 2, 64, 2, 8),
    "alpha": Range("alpha", "Catmull-Rom alpha", 0.0, 1.0, 0.05, 0.25),
    "tension": Range("tension", "Tension", -1.0, 1.0, 0.1, 0.5),
    "bias": Range("bias", "Bias", -1.0, 1.0, 0.1, 0.5),
    "continuity": Range("continuity", "Continuity", -1.0, 1.0, 0.1, 0.5),
    "weight": Range("weight", "NURBS weight", 0.1, 10.0, 0.1, 1.0),
}


def alpha_name(alpha: float) -> str:
    for value, name in ((0.0, "uniform"), (0.5, "centripetal"), (1.0, "chordal")):
        if abs(alpha - value) < 1e-6:
            return name
    return "custom"
