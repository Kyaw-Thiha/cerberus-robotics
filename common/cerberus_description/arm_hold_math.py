"""Pure helpers for the arm-hold event: offsets, clamping, limits.

Kept out of ``policy.locomotion`` on purpose: importing that package pulls Isaac Lab in, and these rules are worth
testing without a GPU (a bad offset target would silently tilt every episode).
"""

from __future__ import annotations

from collections.abc import Sequence


def wrap_to_limit(value: float, low: float, high: float) -> float:
    """Clamp into the joint limit (no wrap-around; arm joints do not cross +-pi here)."""
    return max(low, min(high, value))


def arm_offset_bounds(stowed: Sequence[float], max_offset_rad: float,
                      limits: Sequence[tuple[float, float]] | None = None) -> tuple[list[float], list[float]]:
    """Low/high target for the stowed pose +- offset, clamped to the limits when given."""
    lows, highs = [], []
    for index, value in enumerate(stowed):
        low, high = value - max_offset_rad, value + max_offset_rad
        if limits is not None:
            jl, jh = limits[index]
            low, high = wrap_to_limit(low, jl, jh), wrap_to_limit(high, jl, jh)
        lows.append(low)
        highs.append(high)
    return lows, highs
