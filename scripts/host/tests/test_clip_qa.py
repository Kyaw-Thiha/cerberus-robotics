"""Clip QA gate: the still checks must fail a flat red-orange frame and pass a varied one; poses must be judged
from the logged numbers. Synthetic frames, no Isaac, no GPU.

Evidence: the rejected 2026-11-23 demo clip rendered the whole scene as one flat red-orange, which is exactly what
the hue-diversity rule catches.
"""

from __future__ import annotations



import subprocess
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clip_qa


def varied_frame(size: int = 96) -> np.ndarray:
    """Four colour quarters plus a grey strip: several hues, mid luminance, nothing clipped."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    q = size // 2
    frame[:q, :q] = (150, 150, 150)   # grey (hue ignored)
    frame[:q, q:] = (120, 140, 90)    # olive
    frame[q:, :q] = (110, 150, 160)   # blue-grey
    frame[q:, q:] = (165, 120, 110)   # brown
    return frame


def flat_red_frame(size: int = 96) -> np.ndarray:
    return np.full((size, size, 3), (188, 66, 44), dtype=np.uint8)


def test_varied_frame_passes_the_still_checks():
    checks = clip_qa.still_checks(varied_frame())
    assert checks["passed"], checks
    assert checks["hue_bins_over_5pct"] >= 3


def test_flat_red_frame_fails_on_hue_diversity():
    checks = clip_qa.still_checks(flat_red_frame())
    assert not checks["passed"]
    assert not checks["hue_ok"]
    assert checks["hue_bins_over_5pct"] <= 1


def test_dark_and_blown_frames_fail_luminance_or_clipping():
    dark = np.full((32, 32, 3), 20, dtype=np.uint8)
    assert not clip_qa.still_checks(dark)["mean_luminance_ok"]
    blown = np.full((32, 32, 3), 255, dtype=np.uint8)
    checks = clip_qa.still_checks(blown)
    assert not checks["clipped_ok"]
    assert not checks["mean_luminance_ok"]


def test_hue_bins_cover_the_circle():
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    colours = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255),
               (0, 0, 255), (128, 0, 255), (255, 0, 255), (128, 64, 0), (0, 128, 64),
               (64, 0, 128), (0, 64, 128)]
    for i, colour in enumerate(colours):
        frame[i, :] = colour
    fractions = clip_qa.hue_bin_fractions(frame)
    assert sum(f > 0 for f in fractions) >= 10


def _pose(i: int, *, cam_offset=(0.3, 0.0, 0.08), yaw: float = 0.0, move: float = 0.0):
    """Base at (move*i, 0, 0.4) yawing with ``yaw*i``; camera rigidly offset in the base frame."""
    import math as _m
    base_pos = (move * i, 0.0, 0.4)
    half = (yaw * i) / 2
    base_quat = (0.0, 0.0, _m.sin(half), _m.cos(half))
    Rb = clip_qa.quat_to_matrix(np.array(base_quat))
    cam_pos = np.array(base_pos) + Rb @ np.array(cam_offset)
    cam_quat = base_quat
    return {"t": float(i), "base_pos": list(base_pos), "base_quat_xyzw": list(base_quat),
            "cam_pos": list(cam_pos), "cam_quat_xyzw": list(cam_quat)}


def test_ego_rigid_camera_passes_and_detached_camera_fails():
    poses = {"schema": "cerberus.poses.v1", "camera": "ego", "hfov_deg": 90.0,
             "frames": [_pose(i, move=0.2) for i in range(10)]}
    assert clip_qa.pose_checks(poses, "ego")["passed"]

    drifting = {"frames": [_pose(i, cam_offset=(0.3 + 0.01 * i, 0.0, 0.08), move=0.2) for i in range(10)]}
    assert not clip_qa.pose_checks(drifting, "ego")["passed"]


def test_overview_must_not_move():
    fixed = {"frames": [{"t": float(i), "base_pos": [0.5 * i, 0, 0.4], "base_quat_xyzw": [0, 0, 0, 1],
                         "cam_pos": [5.0, 5.0, 2.0], "cam_quat_xyzw": [0, 0, 0, 1]} for i in range(5)]}
    assert clip_qa.pose_checks(fixed, "overview")["passed"]
    for entry in fixed["frames"]:
        entry["cam_pos"] = [5.0 + 0.02 * entry["t"], 5.0, 2.0]
    assert not clip_qa.pose_checks(fixed, "overview")["passed"]


def test_chase_follows_the_base_and_allows_yaw():
    frames = []
    for i in range(10):
        yaw = 0.15 * i
        base = _pose(i, yaw=yaw, move=0.2)
        Rb = clip_qa.quat_to_matrix(np.array(base["base_quat_xyzw"]))
        cam_pos = np.array(base["base_pos"]) + Rb @ np.array([-1.6, 0.0, 1.3])
        frames.append({**base, "cam_pos": list(cam_pos)})
    assert clip_qa.pose_checks({"frames": frames}, "chase")["passed"]


def test_still_frame_extraction_from_a_real_ffmpeg_clip(tmp_path):
    clip = tmp_path / "t.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                    "-t", "1", "-pix_fmt", "yuv420p", str(clip)], check=True)
    info = clip_qa.ffprobe_info(clip)
    assert info["width"] == 160 and info["height"] == 120
    frame = clip_qa.extract_frame(clip, 0.5, info["width"], info["height"])
    assert frame.shape == (120, 160, 3)
    report = clip_qa.run_gate(clip, camera="none", min_duration_s=0.5, frames_dir=tmp_path / "frames")
    assert len(report["stills"]) == 5
    assert (tmp_path / "frames" / "t_000.png").is_file()


def test_gate_fails_a_flat_red_video(tmp_path):
    clip = tmp_path / "red.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=0xBC422C:size=160x120:rate=10", "-t", "1", "-pix_fmt", "yuv420p", str(clip)],
                   check=True)
    report = clip_qa.run_gate(clip, camera="none", min_duration_s=0.5)
    assert not report["passed"]
    assert all(s["mean_luminance_ok"] is not None for s in report["stills"])


def test_project_point_puts_a_point_straight_ahead_at_the_centre():
    # camera at origin looking down -Z (identity rotation), point 2 m ahead
    u, v, inside = clip_qa.project_point(np.array([0.0, 0.0, -2.0]), np.array([0.0, 0.0, 0.0]),
                                         np.array([0.0, 0.0, 0.0, 1.0]), 90.0, 1920, 1080)
    assert inside and abs(u - 960) <= 2 and abs(v - 540) <= 2
