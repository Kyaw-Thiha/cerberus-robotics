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
    ax.imshow(grid, cmap="Greys", origin="lower",
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
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("top-down map (planner occupancy, ground truth)", fontsize=9)
    if step_label:
        ax.text(0.02, 0.02, step_label, transform=ax.transAxes, fontsize=8, va="bottom")


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

    import imageio.v2 as imageio
    import matplotlib
    import numpy as np
    from PIL import Image

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width, height = (int(v) for v in args.size.lower().split("x"))
    half = width // 2

    grid, meta = load_grid(Path(args.grid))
    path_xy = json.loads(Path(args.path).read_text()) if args.path else None
    trajectory = None
    poses = json.loads(Path(args.poses).read_text())
    frames = poses["frames"] if isinstance(poses, dict) else poses
    base = [(f["base_pos"][0], f["base_pos"][1], _yaw(f["base_quat_wxyz"])) for f in frames]
    if args.path_from_poses:
        trajectory = [[p[0], p[1]] for p in base]

    reader = imageio.get_reader(args.clip)
    fps = reader.get_meta_data().get("fps", 30)
    writer = imageio.get_writer(args.out, fps=fps, macro_block_size=None)

    try:
        for index, frame in enumerate(reader):
            pose = base[min(index, len(base) - 1)]
            fig = plt.figure(figsize=(half / 100, height / 100), dpi=100)
            ax = fig.add_axes([0.08, 0.07, 0.9, 0.86])
            here = trajectory[: min(index + 1, len(trajectory))] if trajectory else None
            draw_map(ax, grid, meta, here if trajectory else path_xy, pose, f"t = {index / fps:5.1f} s",
                     path_label="logged trajectory" if trajectory else "planned path")
            fig.canvas.draw()
            map_rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            plt.close(fig)

            image = Image.fromarray(frame)
            image.thumbnail((half, height))
            canvas = Image.new("RGB", (width, height), (16, 16, 16))
            canvas.paste(image, ((half - image.width) // 2, (height - image.height) // 2))
            canvas.paste(Image.fromarray(map_rgb).resize((half, height)), (half, 0))
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        reader.close()
    print(f"wrote {args.out} ({index + 1} frames, {args.label})")
    return 0


def _yaw(quat_wxyz: list[float]) -> float:
    import math

    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


if __name__ == "__main__":
    sys.exit(main())
