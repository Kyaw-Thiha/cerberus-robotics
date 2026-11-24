"""Half-split composition and the data-panel renderer: sizes, alignment guards, readable output."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import compose_half_split
import render_data_panel


def _clip(path: Path, size=(320, 360), duration=1.0, source="testsrc"):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"{source}=size={size[0]}x{size[1]}:rate=30:duration={duration}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)


def test_compose_stacks_two_native_halves(tmp_path):
    left, right = tmp_path / "l.mp4", tmp_path / "r.mp4"
    _clip(left)
    _clip(right, source="smptebars")
    out = tmp_path / "out.mp4"
    result = compose_half_split.compose(left, right, out, left_text="Chase camera (render)",
                                        right_text="Occupancy grid (ground truth)")
    assert result["size"] == [640, 360] and result["frames"] == 30
    info = compose_half_split.probe(out)
    assert (info["width"], info["height"]) == (640, 360)


def test_compose_refuses_misaligned_halves(tmp_path):
    left, right = tmp_path / "l.mp4", tmp_path / "r.mp4"
    _clip(left, duration=1.0)
    _clip(right, duration=1.7)
    with pytest.raises(SystemExit):
        compose_half_split.compose(left, right, tmp_path / "out.mp4", left_text="a", right_text="b")


def test_map_panel_draws_a_readable_frame(tmp_path):
    grid = np.zeros((40, 60), dtype=np.int8)
    grid[10:12, 5:55] = 1
    poses = np.zeros((60, 7), dtype=np.float32)
    poses[:, 0] = np.linspace(-2.0, 2.0, 60)
    times = np.arange(60) * 0.02
    log = {
        "grid": grid, "grid_origin_xy": np.array([-3.0, -2.0]), "grid_resolution_m": np.array([0.1]),
        "pose_xyzw": poses, "t": times, "goals_xy": np.array([[2.0, 1.0], [-2.0, 1.5]]),
        "goal_index": np.zeros(60, dtype=np.int16),
        "path_xy": np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]], dtype=np.float32),
        "path_leg": np.zeros(3, dtype=np.int16),
    }
    path = tmp_path / "log.npz"
    np.savez_compressed(path, **log)
    panel = render_data_panel.render_map_frame(np.load(path), 30, 240, 270)
    assert panel.size == (240, 270)
    rgb = np.asarray(panel.convert("RGB"), dtype=np.float32)
    assert rgb.std() > 10.0                       # not blank
    assert rgb.mean() > 20.0


def test_showcase_builds_a_short_film_from_synthetic_chapters(tmp_path):
    import make_showcase

    clips = []
    for name in ("a.mp4", "b.mp4"):
        path = tmp_path / name
        _clip(path, size=(320, 180), duration=2.0)
        clips.append(path)
    plan = {
        "title": "Test film", "subtitle": "synthetic",
        "chapters": [
            {"clip": str(clips[0]), "title": "One", "desc": "first", "caption": "left | right", "start": 0.0, "duration": 1.0},
            {"clip": str(clips[1]), "title": "Two", "desc": "second", "caption": "left | right", "start": 0.0, "duration": 1.0},
        ],
        "credits": ["synthetic"],
        "timing": {"title_s": 1.0, "card_s": 0.5, "end_s": 1.0, "fade": 0.2},
    }
    result = make_showcase.build(plan, tmp_path / "out")
    info = compose_half_split.probe(Path(result["film"]))
    assert (info["width"], info["height"]) == (1920, 1080)
    assert 3.0 <= info["duration_s"] <= 7.0
    assert Path(result["gif"]).stat().st_size <= 8 * 1024 * 1024
