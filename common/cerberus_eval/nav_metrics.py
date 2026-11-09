"""Navigation metrics: per-episode outcomes, SPL, and the aggregates the H4 report quotes.

Pure Python (stdlib only). The pod driver hands in a trace of :class:`StepState` rows -- one per control step,
recorded from the simulator -- and this module decides the outcome. Keeping that decision here, not in the
driver, is what makes "success counted despite a fall" a test case rather than a code-review question.

Outcome precedence (strict, deliberate):

1. ``nan`` -- a non-finite pose was seen at or before the first goal hit. A run whose numbers are not numbers
   cannot be a success.
2. ``fall`` -- the base was in contact with static geometry / the fall term fired at ANY point in the episode,
   including after the goal was reached. The handoff's definition is "within tolerance before timeout, **no
   fall**", so a fall after success still fails the episode; ``fell_after_success`` keeps the fact visible.
3. ``success`` -- the first step within ``tolerance_m`` of the goal.
4. ``timeout`` -- none of the above.

``planner_failure`` is not produced here: it is the driver's record for an episode that never started because A*
found no path.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

OUTCOMES = ("success", "fall", "nan", "timeout", "planner_failure", "collision")
"""``collision`` is the kinematic upper-bound outcome: the simulated base centre entered a BLOCKED cell. Physics runs
keep the handoff's success definition (goal, no fall) and report static contacts separately, so a physics run never
produces ``collision``."""
FAILURE_OUTCOMES = tuple(o for o in OUTCOMES if o != "success")

STANDING_MIN_BASE_MINUS_FOOT_M = 0.15
"""A Go2 standing on its feet keeps the base 0.25-0.35 m above the feet; <= 0.15 means it collapsed or sank
(threshold from ``scripts/arena_spike_go2.py``'s stand check)."""


@dataclass(frozen=True)
class StepState:
    """One control step of an episode, as the simulator reported it."""

    t: float
    x: float
    y: float
    yaw: float
    fallen: bool = False
    nonfinite: bool = False
    contact: bool = False
    base_minus_foot_z: float | None = None


@dataclass
class EpisodeResult:
    scene: str = ""
    seed: int = 0
    pair_index: int = 0
    outcome: str = "timeout"
    reached_goal: bool = False
    fell: bool = False
    fell_after_success: bool = False
    nonfinite: bool = False
    time_s: float = 0.0
    steps: int = 0
    path_length_m: float = 0.0
    planned_length_m: float | None = None
    spl: float | None = None
    contacts: int = 0
    final_distance_m: float | None = None
    start: tuple[float, float, float] | None = None
    goal: tuple[float, float] | None = None
    min_base_minus_foot_z: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.outcome == "success"


def standing_ok(base_minus_foot_z: float, threshold: float = STANDING_MIN_BASE_MINUS_FOOT_M) -> bool:
    """True when the base is at least ``threshold`` above the mean foot height."""
    return base_minus_foot_z > threshold


def evaluate_episode(
    steps: Sequence[StepState],
    goal: tuple[float, float],
    *,
    tolerance_m: float = 0.5,
    shortest_path_m: float | None = None,
) -> dict[str, Any]:
    """Decide an episode's outcome and SPL from its step trace (see the module docstring for precedence)."""
    if not steps:
        raise ValueError("an episode trace needs at least one step")
    if tolerance_m <= 0:
        raise ValueError("tolerance_m must be positive")

    success_idx = fall_idx = nan_idx = None
    cumulative = [0.0] * len(steps)
    for i, s in enumerate(steps):
        if i:
            cumulative[i] = cumulative[i - 1] + math.hypot(s.x - steps[i - 1].x, s.y - steps[i - 1].y)
        if nan_idx is None and (s.nonfinite or not all(map(math.isfinite, (s.t, s.x, s.y, s.yaw)))):
            nan_idx = i
        if fall_idx is None and s.fallen:
            fall_idx = i
        if success_idx is None and math.hypot(s.x - goal[0], s.y - goal[1]) <= tolerance_m:
            success_idx = i

    # "Reached the goal" means reached it while still standing: a fall ends the episode in the simulator, so a goal
    # hit after the fall step is not a navigation result.
    reached_goal = success_idx is not None and (fall_idx is None or success_idx < fall_idx)
    fell_after_success = success_idx is not None and fall_idx is not None and fall_idx > success_idx
    if nan_idx is not None and (success_idx is None or nan_idx <= success_idx):
        outcome, end = "nan", nan_idx
    elif fall_idx is not None:
        outcome, end = "fall", fall_idx
    elif reached_goal:
        outcome, end = "success", success_idx
    else:
        outcome, end = "timeout", len(steps) - 1

    # Everything is truncated at the deciding step: a trace that keeps stepping after the goal or after a fall must
    # not inflate the path length, contact count or episode time.
    end_step = steps[end]
    path_length = cumulative[end]
    contacts = sum(int(bool(s.contact)) for s in steps[: end + 1])
    base_values = [s.base_minus_foot_z for s in steps[: end + 1] if s.base_minus_foot_z is not None]
    min_base_minus_foot = min(base_values) if base_values else None
    final_distance = math.hypot(end_step.x - goal[0], end_step.y - goal[1])
    spl = None
    if shortest_path_m is not None:
        if outcome != "success":
            spl = 0.0
        elif path_length <= 0.0:
            spl = 1.0 if shortest_path_m <= 0.0 else 0.0
        else:
            spl = shortest_path_m / max(path_length, shortest_path_m)

    record = asdict(
        EpisodeResult(
            outcome=outcome,
            reached_goal=reached_goal,
            fell=fall_idx is not None,
            fell_after_success=fell_after_success,
            nonfinite=nan_idx is not None,
            time_s=end_step.t,
            steps=end + 1,
            path_length_m=path_length,
            planned_length_m=shortest_path_m,
            spl=spl,
            contacts=contacts,
            final_distance_m=final_distance,
            goal=(float(goal[0]), float(goal[1])),
            min_base_minus_foot_z=min_base_minus_foot,
        )
    )
    record["success"] = outcome == "success"
    return record


def planner_failure_result(*, notes: Iterable[str] = ()) -> dict[str, Any]:
    """The record for a start-goal pair A* could not connect (no episode is run; SPL counts as 0)."""
    record = asdict(EpisodeResult(outcome="planner_failure", spl=0.0, notes=list(notes)))
    record["success"] = False
    return record


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Rates and means over episode records (``spl`` is the standard mean over ALL episodes, failures as 0)."""
    n = len(rows)
    if not n:
        return {"episodes": 0}
    counts = {o: sum(1 for r in rows if r.get("outcome") == o) for o in OUTCOMES}
    spls = [float(r["spl"]) for r in rows if r.get("spl") is not None]
    return {
        "episodes": n,
        "success": counts["success"],
        "success_rate": counts["success"] / n,
        "spl_mean": _mean(spls),
        "fall": counts["fall"],
        "fall_rate": counts["fall"] / n,
        "timeout": counts["timeout"],
        "timeout_rate": counts["timeout"] / n,
        "nan": counts["nan"],
        "collision": counts["collision"],
        "collision_rate": counts["collision"] / n,
        "planner_failure": counts["planner_failure"],
        "planner_failure_rate": counts["planner_failure"] / n,
        "fell_after_success": sum(1 for r in rows if r.get("fell_after_success")),
        "mean_time_s": _mean([float(r["time_s"]) for r in rows if r.get("outcome") != "planner_failure"]),
        "mean_path_length_m": _mean([float(r["path_length_m"]) for r in rows if r.get("path_length_m") is not None]),
        "mean_spl_success_only": _mean([float(r["spl"]) for r in rows if r.get("outcome") == "success" and r.get("spl") is not None]),
        "total_static_contacts": sum(int(r.get("contacts") or 0) for r in rows),
        "episodes_with_contacts": sum(1 for r in rows if (r.get("contacts") or 0) > 0),
    }


def aggregate_by_scene(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """``{scene: aggregate}`` plus an ``"all"`` entry."""
    scenes = sorted({r.get("scene", "") for r in rows})
    out = {scene: aggregate([r for r in rows if r.get("scene", "") == scene]) for scene in scenes}
    out["all"] = aggregate(rows)
    return out


def write_run_json(path: str | Path, record: dict[str, Any]) -> Path:
    """One JSON per run (the clip helper writes the matching file beside each clip)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return path


CSV_COLUMNS = (
    "scene",
    "seed",
    "pair_index",
    "outcome",
    "success",
    "spl",
    "time_s",
    "path_length_m",
    "planned_length_m",
    "contacts",
    "blocked_cell_steps",
    "margin_steps",
    "final_distance_m",
    "fell",
    "fell_after_success",
    "nonfinite",
)


def write_aggregate_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    """The aggregate table the report quotes: one row per episode, stable column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "success": row.get("outcome") == "success"})
    return path
