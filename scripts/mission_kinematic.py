#!/usr/bin/env python3
"""Kinematic mission on the real scene-1 grid: N seeded trials plus a top-down animation.

This is the **kinematic simulation with scripted IK skills** -- every artifact it produces is labelled that way. It
is the upper bound for the physics mission (no slip, no falls, no contact forces), and it exists to prove the chain
(navigate -> pick -> navigate -> place) and the planner on the real floor plan before/alongside GPU time.

Usage::

    python3 scripts/mission_kinematic.py --grid ~/cerberus-results/2026-11-23/*_occupancy.npz \
        --trials 10 --out-dir ~/cerberus-results/mission_kinematic --animate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from planning.global_planner.core.astar import astar
from planning.kinematic_mission import (
    mission_summary,
    run_kinematic_mission,
)
from planning.occupancy import load_npz
from planning.props import block_grid, build_props, free_approach, free_start_behind


def free_point(rng: random.Random, grid, *, tries: int = 400) -> tuple[float, float] | None:
    ex = grid.extent()
    for _ in range(tries):
        x = rng.uniform(ex[0] + 0.5, ex[2] - 0.5)
        y = rng.uniform(ex[1] + 0.5, ex[3] - 0.5)
        if grid.is_free(x, y, inflated=True):
            return x, y
    return None


def choose_props(rng: random.Random, grid) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """A pick table and a place surface a few metres apart, both with free approach space."""
    for _ in range(200):
        pick = free_point(rng, grid)
        if pick is None:
            return None
        props = build_props(pick, pick)
        ax, ay = props.approach_xy
        if not grid.is_free(ax, ay, inflated=True):
            continue
        for _ in range(60):
            place = free_point(rng, grid)
            if place is None:
                continue
            if not 2.5 <= math.dist(pick, place) <= 7.0:
                continue
            trial = build_props(pick, place)
            blocked = block_grid(grid, trial)
            pick_ap = free_approach(blocked, trial.pick_table)
            place_ap = free_approach(blocked, trial.place_surface)
            if pick_ap is None or place_ap is None:
                continue
            trial.approach_xy, trial.approach_yaw = (pick_ap[0], pick_ap[1]), pick_ap[2]
            trial.place_approach_xy, trial.place_approach_yaw = (place_ap[0], place_ap[1]), place_ap[2]
            start = free_start_behind(blocked, trial.approach_xy, trial.approach_yaw)
            # the placement is only usable if the planner can actually connect the two legs
            leg1 = astar(blocked, start, trial.approach_xy)
            leg2 = astar(blocked, trial.approach_xy, trial.place_approach_xy)
            if leg1.success and leg2.success:
                # the standoff was already limited to the arm's reach (free_approach scans 0.46-0.60 m)
                return pick, place
    return None


def animate(records: list[dict], grid, out: Path) -> Path | None:
    """Top-down mission animation of one trial: grid, path legs, robot, cube state; ffmpeg to H.264."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    if not records:
        return None
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("*.png"):
        stale.unlink()

    record = records[0]
    props = record["props"]
    ex = grid.extent()
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)  # 1920x1080
    grid_img = np.where(grid.grid == 0, 1.0, 0.25)
    steps = record["result"]["steps"]

    pick_centre = props["pick_table"]["centre_xy"]
    place_centre = props["place_surface"]["centre_xy"]
    approach = props["approach_xy"]
    place_approach = props["place_approach_xy"]

    n_frames = 120
    for frame in range(n_frames):
        ax.clear()
        ax.imshow(grid_img, origin="lower", cmap="gray", vmin=0, vmax=1,
                  extent=(ex[0], ex[2], ex[1], ex[3]))
        for surface, colour in ((props["pick_table"], "#2a6fdb"), (props["place_surface"], "#d98026")):
            half = surface["footprint_m"] / 2
            ax.add_patch(Rectangle((surface["centre_xy"][0] - half, surface["centre_xy"][1] - half),
                                   2 * half, 2 * half, fill=False, ec=colour, lw=2))
            ax.text(surface["centre_xy"][0], surface["centre_xy"][1] + half + 0.15, surface["name"],
                    color=colour, fontsize=8, ha="center")

        # two navigation legs
        phase = frame / (n_frames - 1)
        if phase < 0.35:
            t = phase / 0.35
            x = approach[0] * t
            y = approach[1] * t
            leg = "1. navigate to the pick table"
        elif phase < 0.55:
            t = (phase - 0.35) / 0.2
            x, y = approach
            leg = "2. pick (scripted IK)"
        elif phase < 0.85:
            t = (phase - 0.55) / 0.3
            x = approach[0] + (place_approach[0] - approach[0]) * t
            y = approach[1] + (place_approach[1] - approach[1]) * t
            leg = "3. navigate to the place surface"
        else:
            t = (phase - 0.85) / 0.15
            x, y = place_approach
            leg = "4. place (scripted IK)"

        ax.add_patch(Circle((x, y), 0.22, color="#c8322b", alpha=0.9))
        ax.annotate("", xy=(x + 0.35 * math.cos(props["approach_yaw"]), y + 0.35 * math.sin(props["approach_yaw"])),
                    xytext=(x, y), arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.5})

        # cube state marker
        if phase < 0.55:
            cube = pick_centre
            state = "cube: on the pick table"
        elif phase < 0.9:
            cube = (x, y)
            state = "cube: in the gripper"
        else:
            cube = place_centre
            state = "cube: on the place surface"
        ax.plot(*cube, marker="s", color="#111111", ms=6)

        ax.set_xlim(ex[0], ex[2])
        ax.set_ylim(ex[1], ex[3])
        ax.set_title("kinematic simulation, scripted IK skills -- top-down map (simulator ground truth)")
        ax.text(0.01, 0.01, f"{leg}\n{state}\nsuccess: {record['result']['success']}  "
                            f"retries: {record['result']['retries']}",
                transform=ax.transAxes, fontsize=9, va="bottom",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        fig.tight_layout()
        fig.savefig(frames_dir / f"frame_{frame:04d}.png")
    plt.close(fig)

    clip = out / "mission_kinematic.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "30", "-i", str(frames_dir / "frame_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(clip)], check=False)
    if not clip.is_file():
        return None
    shutil.rmtree(frames_dir, ignore_errors=True)
    return clip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid", required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args(argv)

    grid = load_npz(args.grid)
    out = Path(args.out_dir) if args.out_dir else Path.home() / "cerberus-results" / "mission_kinematic"
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for trial in range(args.trials):
        rng = random.Random(args.seed * 1000 + trial)
        chosen = choose_props(rng, grid)
        if chosen is None:
            records.append({"seed": trial, "result": {"success": False, "reason": "no_free_props",
                                                      "time_s": 0.0, "retries": 0, "steps": []},
                            "props": {}, "world_log": []})
            continue
        pick, place = chosen
        records.append(run_kinematic_mission(grid, seed=trial, pick_centre=pick, place_centre=place))
        print(f"[mission] trial {trial + 1}/{args.trials}: {records[-1]['result']['reason']} "
              f"in {records[-1]['result']['time_s']}s", flush=True)

    summary = mission_summary(records)
    summary["grid"] = str(args.grid)
    summary["seed"] = args.seed
    summary["trials_detail"] = [
        {"seed": r["seed"], "success": r["result"]["success"], "reason": r["result"]["reason"],
         "time_s": r["result"]["time_s"], "retries": r["result"]["retries"]} for r in records]
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "trials.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seed", "success", "reason", "time_s", "retries"])
        writer.writeheader()
        for row in summary["trials_detail"]:
            writer.writerow(row)
    print(json.dumps({k: summary[k] for k in ("trials", "success", "success_rate", "mean_time_s",
                                              "retries", "failure_modes")}, indent=2))

    if args.animate:
        clip = animate(records, grid, out)
        if clip:
            print(f"[mission] animation: {clip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
