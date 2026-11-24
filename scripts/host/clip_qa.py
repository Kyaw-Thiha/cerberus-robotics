#!/usr/bin/env python3
"""Clip QA gate: no clip counts until this passes.

Every video is judged on numbers taken from the file itself (ffprobe), from five frames decoded at 0/25/50/75/100
percent, and -- when the recorder logged them -- from the camera and base poses of every frame.

Still checks (each of the five frames, 0-255 RGB), Gate v2:
  * mean luminance in [60, 240] (wide band; a blank or blown frame still fails);
  * luminance std >= 25 (a washed-out frame fails; calibrated under the owner-approved ego frames, which
    measure 34.5 and 50.2 -- the old [70, 170] band alone passed the uniform grey still_ego.png with std 0);
  * at most 5 percent of pixels clipped at 255;
  * at least 3 hue bins of 30 degrees each holding at least 5 percent of the pixels
    (a single-colour scene, e.g. the flat red-orange render, fails here).

Data halves (map, depth image, SLAM, plots) use ``data_half_checks``: not blank (luminance std >= 3), the same
frame count / fps / duration as the render half, and non-empty captions. The render-half still checks do not apply
to a schematic panel.

Pose checks (``--poses``; schema ``cerberus.poses.v1``, written next to the clip by the recorder):
  * ``ego``      -- the camera moves, and its transform relative to the base is constant within 2 cm and 2 degrees;
  * ``chase``    -- yaw-only frame: WORLD pitch/roll stay constant (2 degrees), yaw is free; the horizontal boom is
                    pulled in for walls (0.5-2.0 m) and its height scales so the look angle is unchanged;
  * ``overview`` -- the camera does not move (1 cm / 0.5 degrees);
  * ``closeup``  -- treated like ``overview`` for the robot-mounted fix, i.e. constant relative to the base.

Robot visibility (``--robot-visible``, chase frames): the base position is projected with the logged intrinsics and
pose; the projection must land inside the frame and the patch around it must differ from the frame's border colour.
That is a heuristic, stated as such: it cannot see occlusion by a wall.

The five decoded frames are kept (``--frames-dir``) so a human can look at exactly the frames the gate judged.

Usage::

    python3 scripts/host/clip_qa.py clip.mp4 --camera ego --poses clip.poses.json
    python3 scripts/host/clip_qa.py clip.mp4 --camera overview --json clip.qa.json
    python3 scripts/host/clip_qa.py clip.mp4 --camera none --still-check-only

Exit 0 = every check passed, 1 = at least one failed, 2 = the file could not be read.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ---- still checks -----------------------------------------------------------------------------------------------

LUM_MIN, LUM_MAX = 60.0, 240.0
LUM_STD_MIN = 25.0
CLIP_LEVEL = 255
CLIP_MAX_FRAC = 0.05
HUE_BIN_DEG = 30
MIN_HUE_BINS = 3
MIN_HUE_FRAC = 0.05
# Near-grey pixels carry no usable hue. 0.10 rather than 0.15: pale interior surfaces (white walls, light wood)
# still have a measurable tint, and the flat/single-colour frames the rule exists to reject have saturation ~0.
SAT_FLOOR = 0.10


def luma(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luminance of an (H, W, 3) uint8/float array."""
    rgb = rgb.astype(np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def mean_luminance(rgb: np.ndarray) -> float:
    return float(luma(rgb).mean())


def clipped_fraction(rgb: np.ndarray, level: int = CLIP_LEVEL) -> float:
    """Fraction of pixels with a channel at the clipping level (>= level, i.e. 255 after encode)."""
    return float((rgb >= level).any(axis=-1).mean())


def hue_bin_fractions(rgb: np.ndarray, sat_floor: float = SAT_FLOOR) -> list[float]:
    """12 fractions, one per 30 degree hue bin, over the pixels with enough saturation."""
    rgb = rgb.astype(np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    delta = mx - mn
    sat = np.where(mx > 1e-6, delta / np.maximum(mx, 1e-6), 0.0)
    hue = np.zeros_like(mx)
    nz = delta > 1e-6
    with np.errstate(invalid="ignore", divide="ignore"):
        rmask = nz & (mx == r)
        gmask = nz & (mx == g) & ~rmask
        bmask = nz & (mx == b) & ~rmask & ~gmask
        hue[rmask] = ((g[rmask] - b[rmask]) / delta[rmask]) % 6
        hue[gmask] = (b[gmask] - r[gmask]) / delta[gmask] + 2
        hue[bmask] = (r[bmask] - g[bmask]) / delta[bmask] + 4
    hue = (hue * 60.0) % 360.0
    usable = sat > sat_floor
    total = float(usable.sum())
    if total <= 0:
        return [0.0] * 12
    bins = np.floor(hue[usable] / HUE_BIN_DEG).astype(int) % 12
    counts = np.bincount(bins, minlength=12).astype(float)
    return (counts / total).tolist()


def still_checks(rgb: np.ndarray) -> dict[str, Any]:
    mean = mean_luminance(rgb)
    spread = float(luma(rgb).std())
    clipped = clipped_fraction(rgb)
    fractions = hue_bin_fractions(rgb)
    strong = [i for i, f in enumerate(fractions) if f >= MIN_HUE_FRAC]
    checks = {
        "mean_luminance": round(mean, 2),
        "mean_luminance_ok": LUM_MIN <= mean <= LUM_MAX,
        "luminance_std": round(spread, 2),
        "luminance_std_ok": spread >= LUM_STD_MIN,
        "clipped_fraction": round(clipped, 4),
        "clipped_ok": clipped <= CLIP_MAX_FRAC,
        "hue_fractions": [round(f, 3) for f in fractions],
        "hue_bins_over_5pct": len(strong),
        "hue_ok": len(strong) >= MIN_HUE_BINS,
    }
    checks["passed"] = bool(checks["mean_luminance_ok"] and checks["luminance_std_ok"]
                            and checks["clipped_ok"] and checks["hue_ok"])
    return checks


# ---- media ----------------------------------------------------------------------------------------------------

def ffprobe_info(path: Path) -> dict[str, Any]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames,codec_name:format=duration,size", "-of", "json", str(path)],
        capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.strip()[:300]}")
    data = json.loads(out.stdout or "{}")
    streams = data.get("streams") or [{}]
    stream = streams[0]
    fmt = data.get("format") or {}
    num, _, den = str(stream.get("r_frame_rate", "0/1")).partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": round(fps, 3),
        "codec": stream.get("codec_name"),
        "frames": int(stream.get("nb_frames") or 0),
        "duration_s": round(float(fmt.get("duration") or 0.0), 3),
        "bytes": int(fmt.get("size") or 0),
    }


def extract_frame(path: Path, at_s: float, width: int, height: int) -> np.ndarray:
    """One frame at ``at_s``; seeking exactly at/after the last frame yields nothing, so fall back to a decode
    from the start (slow but correct) before giving up."""
    for seek in (at_s, max(at_s - 0.2, 0.0), None):
        cmd = ["ffmpeg", "-v", "error"]
        if seek is not None:
            cmd += ["-ss", f"{seek:.3f}"]
        cmd += ["-i", str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        out = subprocess.run(cmd, capture_output=True, check=False)
        if out.returncode == 0 and len(out.stdout) == width * height * 3:
            break
    else:
        raise RuntimeError(f"ffmpeg could not decode a frame at {at_s:.2f}s: {out.stderr[:200]!r}")
    arr = np.frombuffer(out.stdout, dtype=np.uint8)
    if arr.size != width * height * 3:
        raise RuntimeError(f"decoded {arr.size} bytes, expected {width * height * 3}")
    return arr.reshape(height, width, 3)


def sample_times(duration_s: float) -> list[float]:
    """0, 25, 50, 75 and 100 percent, pulled a hair inside the end so the last frame exists."""
    end = max(duration_s - 0.05, 0.0)
    return [end * f for f in (0.0, 0.25, 0.5, 0.75, 1.0)]


# ---- poses ----------------------------------------------------------------------------------------------------

def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrices_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """roll, pitch, yaw in degrees from a rotation matrix (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def relative_pose(base_pos: np.ndarray, base_quat: np.ndarray, cam_pos: np.ndarray,
                  cam_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Rb = quat_to_matrix(base_quat)
    Rc = quat_to_matrix(cam_quat)
    return Rb.T @ (cam_pos - base_pos), Rb.T @ Rc


def pose_checks(poses: dict[str, Any], camera: str) -> dict[str, Any]:
    frames = poses.get("frames") or []
    if len(frames) < 2:
        return {"passed": False, "reason": "fewer than 2 logged frames"}
    base_pos = np.array([f["base_pos"] for f in frames], dtype=float)
    base_quat = np.array([f["base_quat_xyzw"] for f in frames], dtype=float)
    cam_pos = np.array([f["cam_pos"] for f in frames], dtype=float)
    cam_quat = np.array([f["cam_quat_xyzw"] for f in frames], dtype=float)

    cam_travel = float(np.linalg.norm(cam_pos[-1] - cam_pos[0]))
    base_travel = float(np.linalg.norm(base_pos[-1] - base_pos[0]))

    rel_p = np.array([relative_pose(base_pos[i], base_quat[i], cam_pos[i], cam_quat[i])[0]
                      for i in range(len(frames))])
    rel_R = [relative_pose(base_pos[i], base_quat[i], cam_pos[i], cam_quat[i])[1] for i in range(len(frames))]
    rel_spread = float(np.max(np.linalg.norm(rel_p - rel_p[0], axis=1)))
    eul = np.array([matrices_to_euler(R) for R in rel_R])
    # unwrap yaw so a pass through +-180 does not look like a 350 degree jump
    eul[:, 2] = np.degrees(np.unwrap(np.radians(eul[:, 2])))
    pitch_spread = float(np.abs(eul[:, 1] - eul[0, 1]).max())
    roll_spread = float(np.abs(eul[:, 0] - eul[0, 0]).max())
    yaw_spread = float(np.abs(eul[:, 2] - eul[0, 2]).max())

    out: dict[str, Any] = {
        "frames": len(frames),
        "camera_travel_m": round(cam_travel, 3),
        "base_travel_m": round(base_travel, 3),
        "relative_position_spread_m": round(rel_spread, 4),
        "relative_pitch_spread_deg": round(pitch_spread, 3),
        "relative_roll_spread_deg": round(roll_spread, 3),
        "relative_yaw_spread_deg": round(yaw_spread, 3),
    }
    if camera in {"ego", "closeup"}:
        out["checks"] = {"moves": cam_travel > 0.05,
                         "rigid": rel_spread <= 0.02 and pitch_spread <= 2.0 and roll_spread <= 2.0
                         and yaw_spread <= 2.0}
        out["passed"] = all(out["checks"].values())
    elif camera == "chase":
        # The chase frame orbits with the base, so the camera's displacement is not the base's; what must hold is
        # a constant offset (distance and pitch/roll in the base frame) and motion whenever the base moves.
        moved = (cam_travel > 0.05) if base_travel > 0.05 else True
        # The chase boom is collision-aware (may pull IN for a wall; the height scales with it), and the frame is
        # yaw-only by design: it does NOT inherit the base's roll/pitch, so the base-relative pitch varies with the
        # robot's own tilt. What must hold is a plausible horizontal boom and a constant WORLD pitch/roll.
        xy = np.linalg.norm((cam_pos - base_pos)[:, :2], axis=1)
        world_eul = np.array([matrices_to_euler(quat_to_matrix(q)) for q in cam_quat])
        out["boom_xy_min_m"] = round(float(xy.min()), 3)
        out["boom_xy_max_m"] = round(float(xy.max()), 3)
        out["world_pitch_spread_deg"] = round(float(np.abs(world_eul[:, 1] - world_eul[0, 1]).max()), 3)
        out["world_roll_spread_deg"] = round(float(np.abs(world_eul[:, 0] - world_eul[0, 0]).max()), 3)
        out["checks"] = {"moves_with_base": moved,
                         "boom_not_collapsed": float(xy.min()) >= 0.5,
                         "boom_plausible": float(xy.max()) <= 2.0,
                         "world_pitch_roll_constant": float(np.abs(world_eul[:, 1] - world_eul[0, 1]).max()) <= 2.0
                         and float(np.abs(world_eul[:, 0] - world_eul[0, 0]).max()) <= 2.0}
        out["passed"] = all(out["checks"].values())
        out["note"] = "yaw is free on the chase frame (yaw-only, low-passed); its spread is reported, not gated"
    elif camera == "overview":
        spread = float(np.max(np.linalg.norm(cam_pos - cam_pos[0], axis=1)))
        R0 = quat_to_matrix(cam_quat[0])
        ang = [math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R0.T @ quat_to_matrix(q)) - 1) / 2))))
               for q in cam_quat]
        out["position_spread_m"] = round(spread, 4)
        out["orientation_spread_deg"] = round(max(ang), 3)
        out["checks"] = {"fixed_position": spread <= 0.01, "fixed_orientation": max(ang) <= 0.5}
        out["passed"] = all(out["checks"].values())
    else:
        out["passed"] = True
        out["note"] = "no pose rule for this camera kind"
    return out


# ---- robot visibility (heuristic) ------------------------------------------------------------------------------

def project_point(point: np.ndarray, cam_pos: np.ndarray, cam_quat: np.ndarray, hfov_deg: float,
                  width: int, height: int) -> tuple[int, int, bool]:
    """World point -> pixel, using a pinhole model with the logged horizontal FOV; ``in_front`` is the third value."""
    R = quat_to_matrix(cam_quat)
    # USD cameras look down -Z with +Y up and +X right
    local = R.T @ (point - cam_pos)
    depth = -local[2]
    if depth <= 0.05:
        return 0, 0, False
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    u = width / 2.0 + f * (local[0] / depth)
    v = height / 2.0 - f * (local[1] / depth)
    inside = 0 <= u < width and 0 <= v < height
    return round(u), round(v), bool(inside)


def robot_visible_in_frame(frame: np.ndarray, pixel: tuple[int, int], depth_ok: bool,
                  hfov_deg: float) -> dict[str, Any]:
    """Heuristic: the projection is inside the frame and the patch there differs from the border background."""
    h, w = frame.shape[:2]
    u, v = pixel
    if not depth_ok:
        return {"visible": False, "reason": "base is behind the camera", "contrast": 0.0,
                "patch_luminance": 0.0, "border_luminance": 0.0, "patch_std": 0.0, "pixel": [int(u), int(v)]}
    # robot body half-height ~0.35 m at the logged depth; use a generous patch
    half = max(int(0.06 * w), 20)
    y0, y1 = max(v - half, 0), min(v + half, h)
    x0, x1 = max(u - half, 0), min(u + half, w)
    if y1 <= y0 or x1 <= x0:
        return {"visible": False, "reason": "projection outside the frame", "contrast": 0.0,
                "patch_luminance": 0.0, "border_luminance": 0.0, "patch_std": 0.0, "pixel": [int(u), int(v)]}
    patch = frame[y0:y1, x0:x1]
    border = np.concatenate([frame[0, :, :], frame[-1, :, :], frame[:, 0, :], frame[:, -1, :]], axis=0)
    patch_l = luma(patch)
    border_l = luma(border)
    contrast = abs(float(patch_l.mean()) - float(border_l.mean()))
    spread = float(patch_l.std())
    return {
        "visible": bool(contrast >= 12.0 or spread >= 18.0),
        "patch_luminance": round(float(patch_l.mean()), 1),
        "border_luminance": round(float(border_l.mean()), 1),
        "contrast": round(contrast, 1),
        "patch_std": round(spread, 1),
        "pixel": [int(u), int(v)],
        "method": "projection + local contrast (heuristic; cannot detect occlusion by walls)",
    }


# ---- driver ---------------------------------------------------------------------------------------------------

def run_gate(clip: Path, *, camera: str = "none", poses_path: Path | None = None,
             frames_dir: Path | None = None, min_duration_s: float = 1.0,
             max_duration_s: float | None = None, robot_visible: bool = True) -> dict[str, Any]:
    info = ffprobe_info(clip)
    report: dict[str, Any] = {"clip": str(clip), "camera": camera, "media": info, "problems": []}
    if info["width"] <= 0 or info["height"] <= 0:
        report["passed"] = False
        report["problems"].append("no video stream")
        return report
    if info["duration_s"] < min_duration_s:
        report["problems"].append(f"duration {info['duration_s']}s < {min_duration_s}s")
    if max_duration_s is not None and info["duration_s"] > max_duration_s:
        report["problems"].append(f"duration {info['duration_s']}s > {max_duration_s}s")
    if info["height"] < 720:
        report["problems"].append(f"resolution {info['width']}x{info['height']} below 720p")
    if not 20.0 <= info["fps"] <= 65.0:
        report["problems"].append(f"fps {info['fps']} outside 20-65")

    keeps = Path(frames_dir) if frames_dir else Path(tempfile.mkdtemp(prefix="clipqa_"))
    keeps.mkdir(parents=True, exist_ok=True)
    stills = []
    for fraction, at_s in zip((0, 25, 50, 75, 100), sample_times(info["duration_s"]), strict=True):
        frame = extract_frame(clip, at_s, info["width"], info["height"])
        checks = still_checks(frame)
        png = keeps / f"{clip.stem}_{fraction:03d}.png"
        Image.fromarray(frame).save(png)
        checks.update({"at_s": round(at_s, 3), "percent": fraction, "png": str(png)})
        stills.append(checks)
    report["stills"] = stills
    failed_stills = [s["percent"] for s in stills if not s["passed"]]
    if failed_stills:
        report["problems"].append(f"still check failed at {failed_stills} percent")

    if poses_path is not None:
        poses = json.loads(Path(poses_path).read_text())
        pc = pose_checks(poses, camera)
        report["poses"] = pc
        if not pc.get("passed"):
            report["problems"].append("pose checks failed")
        if robot_visible and camera == "chase":
            hfov = float(poses.get("hfov_deg") or 90.0)
            vis = []
            for index, fraction in enumerate((0, 25, 50, 75, 100)):
                # one frame per still, paired with the nearest logged pose
                frame = extract_frame(clip, sample_times(info["duration_s"])[index], info["width"], info["height"])
                entry = poses["frames"][round((len(poses["frames"]) - 1) * fraction / 100.0)]
                u, v, in_front = project_point(np.array(entry["base_pos"]), np.array(entry["cam_pos"]),
                                               np.array(entry["cam_quat_xyzw"]), hfov, info["width"], info["height"])
                vis.append({"percent": fraction, **robot_visible_in_frame(frame, (u, v), in_front, hfov)})
            report["robot_visibility"] = vis
            if not all(item["visible"] for item in vis):
                report["problems"].append("robot not visible in every chase frame")

    report["passed"] = not report["problems"]
    return report


def still_gate(paths: list[Path]) -> dict[str, Any]:
    """Apply the still checks to PNGs (the P0 look gate, before any clip exists)."""
    from PIL import Image

    results = []
    for path in paths:
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        checks = still_checks(rgb)
        checks["png"] = str(path)
        results.append(checks)
    return {"stills": results, "passed": bool(results) and all(r["passed"] for r in results)}


# ---- data halves (map, depth image, SLAM, plots) ---------------------------------------------------------------

DATA_STD_MIN = 3.0


def panel_checks(rgb: np.ndarray) -> dict[str, Any]:
    """A schematic panel must not be blank: some luminance variation, no photo-style band."""
    spread = float(luma(rgb).std())
    return {"panel_luminance_std": round(spread, 2), "panel_not_blank": spread >= DATA_STD_MIN}


def halves_match(render: Path, data: Path, duration_tol: float = 0.05) -> dict[str, Any]:
    """The two halves must cover the same frames: equal frame count, fps and duration."""
    a = ffprobe_info(render)
    b = ffprobe_info(data)
    out = {
        "render": {"frames": a["frames"], "fps": a["fps"], "duration_s": a["duration_s"]},
        "data": {"frames": b["frames"], "fps": b["fps"], "duration_s": b["duration_s"]},
        "same_frames": a["frames"] == b["frames"],
        "same_fps": abs(a["fps"] - b["fps"]) <= 0.01,
        "same_duration": abs(a["duration_s"] - b["duration_s"]) <= duration_tol,
    }
    out["passed"] = out["same_frames"] and out["same_fps"] and out["same_duration"]
    return out


def data_half_checks(data: Path, render: Path | None = None, labels: list[str] | None = None,
                     frames_dir: Path | None = None) -> dict[str, Any]:
    """Gate a data half: not blank, aligned with the render half, and captioned."""
    problems: list[str] = []
    report: dict[str, Any] = {"data": str(data), "problems": problems}
    if data.suffix.lower() == ".png":
        rgb = np.asarray(Image.open(data).convert("RGB"), dtype=np.uint8)
        checks = panel_checks(rgb)
        report["panels"] = [checks]
    else:
        info = ffprobe_info(data)
        report["media"] = info
        keeps = Path(frames_dir) if frames_dir else Path(tempfile.mkdtemp(prefix="clipqa_data_"))
        keeps.mkdir(parents=True, exist_ok=True)
        panels = []
        for fraction, at_s in zip((0, 25, 50, 75, 100), sample_times(info["duration_s"]), strict=True):
            frame = extract_frame(data, at_s, info["width"], info["height"])
            checks = panel_checks(frame)
            png = keeps / f"{data.stem}_data_{fraction:03d}.png"
            Image.fromarray(frame).save(png)
            checks.update({"percent": fraction, "png": str(png)})
            panels.append(checks)
        report["panels"] = panels
    if not all(item["panel_not_blank"] for item in report["panels"]):
        problems.append("a data panel is blank")
    caption_ok = bool(labels) and all(str(label).strip() for label in labels)
    report["labels"] = list(labels or [])
    report["labels_present"] = caption_ok
    if not caption_ok:
        problems.append("no captions supplied for the data half")
    if render is not None:
        match = halves_match(render, data)
        report["match"] = match
        if not match["passed"]:
            problems.append("data half and render half are not frame-aligned")
    report["passed"] = not problems
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clip QA gate (see the module docstring)")
    parser.add_argument("clip", nargs="?")
    parser.add_argument("--stills", nargs="+", default=None,
                        help="apply the still checks to PNG files instead of a clip")
    parser.add_argument("--data-half", default=None, help="gate a data half (map/depth/SLAM/plot) instead of a clip")
    parser.add_argument("--against", default=None, help="the render-half clip the data half must align with")
    parser.add_argument("--labels", default=None, help="comma-separated captions expected on the data half")
    parser.add_argument("--camera", default="none",
                        choices=["ego", "chase", "overview", "closeup", "none"])
    parser.add_argument("--poses", default=None, help="pose JSON written by the recorder")
    parser.add_argument("--frames-dir", default=None, help="keep the five judged frames here")
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--no-robot-visibility", dest="robot_visibility", action="store_false", default=True)
    parser.add_argument("--media-only", action="store_true",
                        help="only the ffprobe checks (schematic animations: a 2D map is not a rendered photo)")
    args = parser.parse_args(argv)

    if args.data_half:
        data = Path(args.data_half)
        if not data.is_file():
            print(f"clip_qa: {data} is not a file", file=sys.stderr)
            return 2
        labels = [part.strip() for part in args.labels.split(",")] if args.labels else None
        try:
            report = data_half_checks(data, Path(args.against) if args.against else None, labels,
                                      Path(args.frames_dir) if args.frames_dir else None)
        except RuntimeError as exc:
            print(f"clip_qa: {exc}", file=sys.stderr)
            return 2
        for panel in report["panels"]:
            print(f"  data {panel.get('percent', '-'):>3}% std={panel['panel_luminance_std']} -> "
                  f"{'ok' if panel['panel_not_blank'] else 'BLANK'}")
        print(f"  labels={report['labels']} present={report['labels_present']}")
        if "match" in report:
            print(f"  halves: {report['match']}")
        print(("PASS " if report["passed"] else "FAIL ") + str(data))
        for problem in report["problems"]:
            print(f"  problem: {problem}")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
        return 0 if report["passed"] else 1

    if args.stills:
        paths = [Path(p) for p in args.stills]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            print(f"clip_qa: missing stills {missing}", file=sys.stderr)
            return 2
        report = still_gate(paths)
        for still in report["stills"]:
            print(f"{Path(still['png']).name}: lum={still['mean_luminance']:6.1f} "
                  f"clipped={still['clipped_fraction']:.3f} hue_bins={still['hue_bins_over_5pct']} -> "
                  f"{'ok' if still['passed'] else 'FAIL'}")
        print(("PASS " if report["passed"] else "FAIL ") + f"{len(paths)} stills")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
        return 0 if report["passed"] else 1

    if not args.clip:
        print("clip_qa: give a clip path or --stills", file=sys.stderr)
        return 2
    clip = Path(args.clip)
    if not clip.is_file():
        print(f"clip_qa: {clip} is not a file", file=sys.stderr)
        return 2
    try:
        if args.media_only:
            info = ffprobe_info(clip)
            problems = []
            if info["width"] <= 0 or info["height"] <= 0:
                problems.append("no video stream")
            if info["duration_s"] < args.min_duration:
                problems.append(f"duration {info['duration_s']}s < {args.min_duration}s")
            if info["height"] < 720:
                problems.append(f"resolution {info['width']}x{info['height']} below 720p")
            if not 20.0 <= info["fps"] <= 65.0:
                problems.append(f"fps {info['fps']} outside 20-65")
            report = {"clip": str(clip), "media": info, "problems": problems, "passed": not problems,
                      "mode": "media-only (schematic animation)"}
            print(f"{clip.name}: {info['width']}x{info['height']} {info['fps']} fps {info['duration_s']}s")
            print(("PASS " if report["passed"] else "FAIL ") + str(clip))
            for problem in problems:
                print(f"  problem: {problem}")
            if args.json_out:
                Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
            return 0 if report["passed"] else 1
        report = run_gate(clip, camera=args.camera, poses_path=Path(args.poses) if args.poses else None,
                          frames_dir=Path(args.frames_dir) if args.frames_dir else None,
                          min_duration_s=args.min_duration, max_duration_s=args.max_duration,
                          robot_visible=args.robot_visibility)
    except RuntimeError as exc:
        print(f"clip_qa: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(report, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    media = report["media"]
    print(f"{clip.name}: {media['width']}x{media['height']} {media['fps']} fps {media['duration_s']}s "
          f"({media['bytes'] / 1e6:.1f} MB, {media['codec']})")
    for still in report["stills"]:
        print(f"  {still['percent']:3d}% lum={still['mean_luminance']:6.1f} clipped={still['clipped_fraction']:.3f} "
              f"hue_bins={still['hue_bins_over_5pct']} -> {'ok' if still['passed'] else 'FAIL'}")
    if "poses" in report:
        poses = report["poses"]
        print(f"  poses: travel cam={poses['camera_travel_m']} base={poses['base_travel_m']} "
              f"rel_spread={poses['relative_position_spread_m']} m -> {'ok' if poses['passed'] else 'FAIL'}")
    for item in report.get("robot_visibility", []):
        print(f"  robot@{item['percent']:3d}% contrast={item['contrast']} pixel={item['pixel']} -> "
              f"{'visible' if item['visible'] else 'NOT VISIBLE'}")
    print(("PASS " if report["passed"] else "FAIL ") + str(clip))
    for problem in report["problems"]:
        print(f"  problem: {problem}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
