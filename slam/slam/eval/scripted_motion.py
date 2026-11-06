from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotionSegment:
    label: str
    command: tuple[float, float, float]
    duration_seconds: float


_SCHEDULES: dict[str, list[MotionSegment]] = {
    "household": [
        MotionSegment("forward", (0.5, 0.0, 0.0), 5.0),
        MotionSegment("turn", (0.0, 0.0, 0.5), 3.0),
        MotionSegment("forward", (0.5, 0.0, 0.0), 5.0),
        MotionSegment("revisit", (-0.5, 0.0, 0.5), 6.0),
    ],
}


def canned_schedule(mission: str) -> list[MotionSegment]:
    try:
        return list(_SCHEDULES[mission])
    except KeyError:
        raise ValueError(f"unknown mission {mission!r}, available: {sorted(_SCHEDULES)}") from None


def command_at(schedule: list[MotionSegment], elapsed_seconds: float) -> tuple[float, float, float]:
    cursor = 0.0
    for segment in schedule:
        if elapsed_seconds < cursor + segment.duration_seconds:
            return segment.command
        cursor += segment.duration_seconds
    return (0.0, 0.0, 0.0)
