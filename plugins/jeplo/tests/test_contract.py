"""JEPLO adapter contract: sensor geometry, history/rate bookkeeping, and no privileged height scan."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from plugins.jeplo.backend import (
    DEFAULT_JOINT_POS,
    JOINT_ORDER,
    JeploBackend,
    JeploObservationError,
)
from plugins.jeplo.sensor import MAX_RANGE_M, H, SphericalDepth, V, quat_wxyz_to_matrix

MODEL_DIR = Path(__file__).resolve().parents[3] / "third_party" / "jeplo" / "training" / "logs" / "rsl_rl" / "go2_loco" / "pretrained" / "exported"


def _ray_directions() -> np.ndarray:
    """The authors' pattern, in the sensor frame (x forward, y left, z up)."""
    # their ray_spherical_slice_pattern uses bin centres, not the FOV endpoints
    v_res = (45.5 - (-4.5)) / V
    h_res = (60.0 - (-60.0)) / H
    v = np.linspace(-4.5 + v_res / 2.0, 45.5 - v_res / 2.0, V)
    h = np.linspace(-60.0 + h_res / 2.0, 60.0 - h_res / 2.0, H)
    vert, horiz = np.meshgrid(np.radians(v), np.radians(h), indexing="ij")
    x = np.cos(vert) * np.cos(horiz)
    y = np.cos(vert) * np.sin(horiz)
    z = np.sin(vert)
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def test_projection_puts_a_known_range_in_the_known_bin():
    sensor = SphericalDepth(noise_std_m=0.0)
    directions = _ray_directions()
    origin = np.array([0.0, 0.0, 0.5])
    quat = (1.0, 0.0, 0.0, 0.0)
    # even columns sit exactly on a bin centre; odd ones are half-bin edges and torch.round's
    # bank-to-even maps them to a neighbour (their own pipeline has the same convention)
    for ray in (0, 120, 1498, 750):
        hits = origin + directions[ray] * 1.0
        grid = sensor.project(hits[None, :], origin, quat)
        assert grid.shape == (V, H)
        hit = grid < MAX_RANGE_M
        assert hit.sum() == 1, (ray, int(hit.sum()))
        row, col = np.argwhere(hit)[0]
        assert grid[row, col] == pytest.approx(1.0, abs=1e-6)
        # the bin is one of the two the ray sits between (their pattern's bin centres fall on edges)
        assert abs(row - ray // H) <= 1 and col == ray % H


def test_frames_are_normalised_and_aggregated_by_minimum():
    sensor = SphericalDepth(noise_std_m=0.0, seed=0)
    origin = np.zeros(3)
    quat = (1.0, 0.0, 0.0, 0.0)
    direction = _ray_directions()[12 * H + 30]   # a middle ray: the bottom-row dropout covers rows 0-1
    out = None
    for step in range(15):
        # alternate 1.0 m and 0.5 m hits: the published image must keep the closest
        hits = (origin + direction * (1.0 if step % 2 else 0.5))[None, :]
        published = sensor.update(hits, origin, quat)
        if published is not None:
            out = published
    assert out is not None
    assert out.shape == (V, H) and out.dtype == np.float32
    assert np.isfinite(out).all()
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
    assert out.min() == pytest.approx(0.5 / MAX_RANGE_M, abs=1e-5)
    assert sensor.depth.shape == (V, H)


def test_rotation_round_trip_is_the_inverse_transform():
    R = quat_wxyz_to_matrix((0.0, 1.0, 0.0, 0.0))
    p = np.array([1.0, 2.0, 3.0])
    assert np.allclose(R @ p, np.array([1.0, -2.0, -3.0]), atol=1e-9)


def _observation(depth=None):
    return {
        "base_ang_vel": [0.0, 0.0, 0.1],
        "projected_gravity": [0.0, 0.0, -1.0],
        "joint_pos": {name: DEFAULT_JOINT_POS[name] for name in JOINT_ORDER},
        "joint_vel": {name: 0.0 for name in JOINT_ORDER},
        "command": {"vx": 0.4, "vy": 0.0, "wz": 0.0},
        "depth": np.full((V, H), 0.75, dtype=np.float32) if depth is None else depth,
    }


def test_backend_shapes_rate_and_joint_targets():
    pytest.importorskip("onnxruntime")
    if not (MODEL_DIR / "policy.onnx").is_file():
        pytest.skip("released JEPLO ONNX is not present (third_party clone)")
    backend = JeploBackend()
    backend.load(str(MODEL_DIR))
    obs = _observation()
    for step in range(100):
        result = backend.act(obs)
        actions = result["actions"]
        assert actions.shape == (12,) and np.isfinite(actions).all()
        targets = result["joint_targets"]
        assert set(targets) == set(JOINT_ORDER)
        for name in JOINT_ORDER:
            assert math.isfinite(targets[name])
    # 100 policy steps at 50 Hz = 2 s = 20 sensor-estimator calls at 10 Hz
    assert backend.estimator_calls == 20


def test_backend_refuses_a_height_scan():
    pytest.importorskip("onnxruntime")
    if not (MODEL_DIR / "policy.onnx").is_file():
        pytest.skip("released JEPLO ONNX is not present (third_party clone)")
    backend = JeploBackend()
    backend.load(str(MODEL_DIR))
    obs = _observation()
    obs["height_scan"] = np.zeros(187, dtype=np.float32)
    with pytest.raises(JeploObservationError):
        backend.act(obs)


def test_command_goes_through_the_contract_limiter():
    backend = JeploBackend()
    first = backend.set_command({"vx": 5.0, "vy": 0.0, "wz": 0.0})
    assert first["vx"] <= 1.0 + 1e-6
    for _ in range(200):
        command = backend.set_command({"vx": 5.0, "vy": 0.0, "wz": 0.0})
    assert command["vx"] == pytest.approx(1.0, abs=1e-6)      # contract max, never 5
