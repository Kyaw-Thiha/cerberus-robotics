#!/usr/bin/env python3
"""Data panels for the half-split showcase clips, drawn from the record-once tour log.

Panel types:
  map    occupancy grid (ground truth), planned A* path, travelled trail, goals, robot arrow, progress
  lidar  the 25x60 spherical depth image JEPLO sees, turbo colour scale in metres, plus a bird's-eye hit strip

One panel frame per output frame at ``--fps``; write a PNG sequence and an H.264 mp4 (no audio). Each half is
rendered natively at the requested size (default 960x1080), never cropped from a 16:9 frame.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/TTF/DejaVuSans.ttf"


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _yaw_from_quat_xyzw(q) -> float:
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _grid_rgb(grid: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*grid.shape, 3), dtype=np.uint8)
    rgb[grid == 0] = (238, 238, 234)      # free
    rgb[grid == 1] = (54, 58, 66)         # blocked
    rgb[grid == -1] = (150, 150, 146)     # unknown
    return rgb


def _fit(rgb: np.ndarray, box_w: int, box_h: int) -> tuple[Image.Image, float, float, float]:
    h, w = rgb.shape[:2]
    scale = min(box_w / w, box_h / h)
    out = Image.fromarray(rgb).resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return out, scale, (box_w - out.width) / 2.0, (box_h - out.height) / 2.0


def render_map_frame(log: np.lib.npyio.NpzFile, index: int, width: int, height: int) -> Image.Image:
    grid = log["grid"]
    origin = log["grid_origin_xy"]
    res = float(log["grid_resolution_m"][0])
    poses = log["pose_xyzw"]
    goals = log["goals_xy"]
    goal_index = log["goal_index"]
    path_xy = log["path_xy"] if "path_xy" in log else np.zeros((0, 2), dtype=np.float32)
    path_leg = log["path_leg"] if "path_leg" in log else np.zeros((0,), dtype=np.int16)

    W, H = width, height
    img = Image.new("RGB", (W, H), (24, 26, 30))
    draw = ImageDraw.Draw(img)
    top, bottom = 64, 120
    base, scale, ox, oy = _fit(_grid_rgb(grid), W - 40, H - top - bottom)
    img.paste(base, (20 + int(ox), top + int(oy)))

    def to_px(x: float, y: float) -> tuple[float, float]:
        px = 20 + ox + (x - origin[0]) / res * scale
        py = top + oy + base.height - (y - origin[1]) / res * scale
        return px, py

    # planned path of the active leg
    leg = int(goal_index[min(index, len(goal_index) - 1)])
    active = np.argwhere(path_leg == leg).reshape(-1)
    if active.size >= 2:
        pts = [to_px(float(path_xy[i, 0]), float(path_xy[i, 1])) for i in active]
        draw.line(pts, fill=(70, 150, 235), width=3)
    # travelled trail
    trail = poses[: index + 1]
    if len(trail) >= 2:
        pts = [to_px(float(x), float(y)) for x, y, *_ in trail]
        draw.line(pts, fill=(240, 150, 60), width=2)
    # goals
    for gi, (gx, gy) in enumerate(goals):
        px, py = to_px(float(gx), float(gy))
        r = 9
        colour = (90, 210, 120) if gi < leg else ((250, 210, 90) if gi == leg else (180, 180, 180))
        draw.ellipse([px - r, py - r, px + r, py + r], outline=colour, width=3)
        draw.text((px + 11, py - 9), str(gi + 1), font=_font(20), fill=colour)
    # robot arrow
    rx, ry, _rz, qx, qy, qz, qw = poses[min(index, len(poses) - 1)]
    px, py = to_px(float(rx), float(ry))
    yaw = _yaw_from_quat_xyzw((qx, qy, qz, qw))
    L = 16
    tip = (px + L * math.cos(yaw), py - L * math.sin(yaw))
    left = (px + 9 * math.cos(yaw + 2.5), py - 9 * math.sin(yaw + 2.5))
    right = (px + 9 * math.cos(yaw - 2.5), py - 9 * math.sin(yaw - 2.5))
    draw.polygon([tip, left, right], fill=(235, 80, 80), outline=(255, 255, 255))

    caption = "Occupancy grid (ground truth, planner input)"
    draw.text((20, 18), caption, font=_font(22), fill=(235, 235, 235))
    t = float(log["t"][min(index, len(log["t"]) - 1)])
    clock = f"{int(t // 60):02d}:{t % 60:04.1f}"
    draw.text((20, H - 58), f"{clock}   goal {leg + 1}/{len(goals)}", font=_font(22), fill=(235, 235, 235))
    frac = (index + 1) / max(1, len(poses))
    bar_y = H - 26
    draw.rectangle([20, bar_y, W - 20, bar_y + 8], outline=(120, 120, 120))
    draw.rectangle([20, bar_y, 20 + (W - 40) * frac, bar_y + 8], fill=(90, 210, 120))
    return img


def render_lidar_frame(depth: np.ndarray, width: int, height: int, hits_xy: np.ndarray | None,
                       caption: str, clock: str, goal: str) -> Image.Image:
    W, H = width, height
    img = Image.new("RGB", (W, H), (24, 26, 30))
    draw = ImageDraw.Draw(img)
    top = 56
    img_h = int(H * 0.55)
    rgb = _turbo(np.asarray(depth, dtype=np.float32))
    panel = Image.fromarray(rgb).resize((W - 60, img_h), Image.NEAREST)
    img.paste(panel, (30, top))
    # a labelled colour scale
    for i in range(10, W - 30, 2):
        frac = (i - 30) / max(1, W - 60)
        bar = tuple(int(v) for v in _turbo(np.array([[1.0 - frac]], dtype=np.float32))[0, 0])
        draw.line([i, top + img_h + 12, i, top + img_h + 30], fill=bar)
    draw.text((30, top + img_h + 38), "2.0 m", font=_font(18), fill=(210, 210, 210))
    draw.text((W - 90, top + img_h + 38), "0.0 m", font=_font(18), fill=(210, 210, 210))
    draw.text((W // 2 - 50, top + img_h + 62), "range (m)", font=_font(18), fill=(210, 210, 210))

    if hits_xy is not None:
        strip_top = top + img_h + 100
        strip_h = H - strip_top - 70
        if strip_h > 40:
            panel = _render_bev(hits_xy, W - 60, strip_h)
            img.paste(panel, (30, strip_top))
            draw.text((30, strip_top - 24), "LiDAR hits (bird's-eye, body frame)", font=_font(18),
                      fill=(210, 210, 210))

    draw.text((20, 16), caption, font=_font(20), fill=(235, 235, 235))
    draw.text((20, H - 40), f"{clock}   {goal}", font=_font(20), fill=(235, 235, 235))
    return img


def _turbo(values: np.ndarray) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1).mul(255).astype(np.uint8) if hasattr(r, "mul") else (
        np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _render_bev(hits_xy: np.ndarray, width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height), (18, 20, 24))
    draw = ImageDraw.Draw(img)
    lim = 2.0
    cx, cy = width / 2, height - 12

    def pt(x, y):
        return cx + x / lim * (width / 2 - 10), cy - y / lim * (height - 20)

    draw.ellipse([*pt(-lim, lim), *pt(lim, -lim)], outline=(60, 60, 60))
    for x, y in np.asarray(hits_xy).reshape(-1, 2)[:4000]:
        draw.point(pt(float(x), float(y)), fill=(120, 220, 255))
    draw.polygon([pt(0.3, 0), pt(-0.1, 0.15), pt(-0.1, -0.15)], fill=(235, 80, 80))
    return img


def render_lidar_panel(log: np.lib.npyio.NpzFile, index: int, step_index: int, width: int, height: int,
                       caption: str) -> Image.Image:
    depth = np.asarray(log["lidar_depth"][step_index], dtype=np.float32)
    scale_m = float(log["lidar_depth_scale"][0]) if "lidar_depth_scale" in log else 2.0
    ranges = depth * scale_m
    V, H = ranges.shape
    theta = np.radians(-4.5 + (np.arange(V) + 0.5) * (50.0 / V))[:, None]
    phi = np.radians(-60.0 + (np.arange(H) + 0.5) * (120.0 / H))[None, :]
    rr = ranges
    x = (rr * np.cos(theta) * np.cos(phi)).reshape(-1)
    y = (rr * np.cos(theta) * np.sin(phi)).reshape(-1)
    hits_xy = np.stack([x, y], axis=1)
    goals = log["goals_xy"]
    leg = int(log["goal_index"][min(index, len(log["goal_index"]) - 1)])
    t = float(log["t"][min(index, len(log["t"]) - 1)])
    clock = f"{int(t // 60):02d}:{t % 60:04.1f}"
    return render_lidar_frame(ranges, width, height, hits_xy, caption, clock,
                              f"goal {leg + 1}/{len(goals)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True)
    parser.add_argument("--panel", choices=["map", "lidar"], default="map")
    parser.add_argument("--caption", default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", required=True, help="mp4 path; frames go to <out>/../<stem>_frames/")
    args = parser.parse_args(argv)

    log = np.load(args.log, allow_pickle=True)
    steps = len(log["pose_xyzw"])
    times = np.asarray(log["t"], dtype=float)
    duration = float(times[-1] - times[0])
    frames = max(1, round(duration * args.fps) + 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = out.parent / f"{out.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for k in range(frames):
        t = times[0] + k / args.fps
        index = max(0, min(int(np.searchsorted(times, t, side="right") - 1), steps - 1))
        if args.panel == "map":
            image = render_map_frame(log, index, args.width, args.height)
        else:
            image = render_lidar_panel(log, index, index, args.width, args.height,
                                       args.caption or "LiDAR depth as JEPLO sees it (sensed, 25x60, 0-2 m)")
        image.save(frame_dir / f"frame_{k:05d}.png")
    cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", str(args.fps),
           "-i", str(frame_dir / "frame_%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "20", "-preset", "veryfast", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    print(f"[panel] wrote {out} ({frames} frames) and {frame_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
