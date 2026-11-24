#!/usr/bin/env python3
"""Compose a split-screen clip: the RTX view on the left, a synchronised top-down map on the right.

Left: the recorded clip (scaled, letterboxed). Right: the scene's occupancy grid, the planned path, the robot pose
taken from the clip's per-frame pose log, and a caption naming what the right panel is (ground-truth occupancy from
the planner, not a camera).

Usage::

    pixi run python scripts/compose_split_screen.py --clip walk.mp4 --poses walk.poses.json \\
        --grid occupancy.npz --out walk_split.mp4 --label "RTX view | top-down map (ground truth)"

Every frame is a real frame of the clip; the right panel is drawn at the same simulation time as the logged pose.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_grid(path: Path):
    import numpy as np

    data = np.load(path)
    grid = data["grid"] if "grid" in data else data[next(iter(data.keys()))]
    meta = {k: data[k].item() if getattr(data[k], "shape", ()) == () else data[k].tolist()
            for k in data.files if k != "grid"}
    return np.asarray(grid), meta


def draw_map(ax, grid, meta, path_xy, pose, step_label: str, path_label: str = "planned path"):
    import numpy as np

    resolution = float(meta.get("resolution_m", meta.get("resolution", 0.05)) or 0.05)
    origin = meta.get("origin_xy", meta.get("origin", [0.0, 0.0]))
    ax.set_facecolor("#5a5a5a")
    ax.imshow(grid, cmap="gray", origin="lower", vmin=-1.6, vmax=2.6,
              extent=(origin[0], origin[0] + grid.shape[1] * resolution,
                      origin[1], origin[1] + grid.shape[0] * resolution))
    if path_xy is not None and len(path_xy) > 1:
        xs = [p[0] for p in path_xy]
        ys = [p[1] for p in path_xy]
        ax.plot(xs, ys, "-", color="#1f77b4", linewidth=1.6, label=path_label)
    x, y, yaw = pose
    ax.plot([x], [y], "o", color="#d62728", markersize=5)
    ax.arrow(x, y, 0.35 * np.cos(yaw), 0.35 * np.sin(yaw), head_width=0.12, color="#d62728")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color="#c8c8c8")
    ax.set_ylabel("y (m)", color="#c8c8c8")
    ax.tick_params(colors="#909090", labelsize=7)
    ax.set_title("top-down map (planner occupancy, ground truth)", fontsize=9, color="#e0e0e0")
    if step_label:
        ax.text(0.02, 0.02, step_label, transform=ax.transAxes, fontsize=8, va="bottom", color="#e0e0e0")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clip", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--path", default=None, help="JSON with a list of [x, y] waypoints (the planned path)")
    parser.add_argument("--path-from-poses", action="store_true",
                        help="draw the robot's own logged trajectory instead (say so in --label)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="RTX view | top-down map")
    parser.add_argument("--size", default="1920x1080")
    args = parser.parse_args(argv)

    import subprocess
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width, height = (int(v) for v in args.size.lower().split("x"))
    half = width // 2

    grid, meta = load_grid(Path(args.grid))
    path_xy = json.loads(Path(args.path).read_text()) if args.path else None
    trajectory = None
    poses = json.loads(Path(args.poses).read_text())
    frames = poses["frames"] if isinstance(poses, dict) else poses
    def base_yaw(frame):
        if "base_quat_wxyz" in frame:
            return _yaw(frame["base_quat_wxyz"])
        x, y, z, w = frame["base_quat_xyzw"]  # the pose log stores XYZW
        return _yaw([w, x, y, z])

    base = [(f["base_pos"][0], f["base_pos"][1], base_yaw(f)) for f in frames]
    if args.path_from_poses:
        trajectory = [[p[0], p[1]] for p in base]

    fps_probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=r_frame_rate", "-of", "csv=p=0", args.clip],
                               capture_output=True, text=True, check=False).stdout.strip()
    try:
        num, den = fps_probe.split("/")
        fps = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        fps = 30.0

    # One map PNG per clip frame, then ffmpeg hstacks the two streams (no imageio/ffmpeg-python needed).
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for index, pose in enumerate(base):
            fig = plt.figure(figsize=(half / 100, height / 100), dpi=100, facecolor="#5a5a5a")
            ax = fig.add_axes([0.08, 0.07, 0.9, 0.86])
            here = trajectory[: min(index + 1, len(trajectory))] if trajectory else None
            draw_map(ax, grid, meta, here if trajectory else path_xy, pose, f"t = {index / fps:5.1f} s",
                     path_label="logged trajectory" if trajectory else "planned path")
            fig.savefig(tmpdir / f"map_{index:05d}.png")
            plt.close(fig)
        subprocess.run(["ffmpeg", "-y", "-i", args.clip, "-framerate", str(fps),
                        "-i", str(tmpdir / "map_%05d.png"),
                        "-filter_complex",
                        (f"[0:v]scale={half}:{height}:force_original_aspect_ratio=decrease,"
                         f"pad={half}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x101010[l];[l][1:v]hstack"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), args.out],
                       check=True, capture_output=True)
    print(f"wrote {args.out} ({len(base)} frames, {args.label})")
    return 0


def _yaw(quat_wxyz: list[float]) -> float:
    import math

    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


if __name__ == "__main__":
    sys.exit(main())
