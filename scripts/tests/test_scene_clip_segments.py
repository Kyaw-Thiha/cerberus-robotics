"""Segment state save/resume for crash-tolerant clip rendering (fake articulation and a stub tensor module).

No torch and no Kit: the point under test is that the state a segment writes is exactly what the next segment
reads back, and that the resume writes pose, velocity and joint state (never only the joints).
"""

import json
import sys
import types
from pathlib import Path

import pytest
import scene_clip


class _FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, col = index
            assert row == 0
            return _Scalar(self.values[col])
        return _Row(self.values)


class _Row:
    """What ``tensor[0]`` returns: still tensor-like, so ``.detach().cpu().tolist()`` works."""

    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def __eq__(self, other):
        return self.value == other

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return repr(self.value)


class _Data:
    def __init__(self):
        self.root_pos_w = _FakeTensor([1.0, 2.0, 0.35])
        self.root_quat_w = _FakeTensor([1.0, 0.0, 0.0, 0.0])  # wxyz
        self.root_lin_vel_w = _FakeTensor([0.4, -0.1, 0.0])
        self.root_ang_vel_w = _FakeTensor([0.0, 0.0, 0.2])
        self.joint_pos = _FakeTensor([float(i) for i in range(20)])
        self.joint_vel = _FakeTensor([0.0] * 20)


class _Robot:
    def __init__(self):
        self.data = _Data()
        self.written = {}

    def write_root_pose_to_sim(self, value):
        self.written["root_pose"] = value

    def write_root_velocity_to_sim(self, value):
        self.written["root_vel"] = value

    def write_joint_state_to_sim(self, pos, vel):
        self.written["joint_pos"] = pos
        self.written["joint_vel"] = vel


def _stub_tensor(values, device=None, dtype=None):
    """Like torch.tensor: a single [[...]] row is one sample, as the articulation API expects."""
    values = list(values)
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        return _FakeTensor(list(values[0]))
    return _FakeTensor(values)


def test_capture_state_round_trips_through_a_file(tmp_path: Path):
    state = scene_clip.capture_state(_Robot())
    path = tmp_path / "seg0.state.json"
    path.write_text(json.dumps(state))

    loaded = scene_clip.load_state(path)
    assert loaded["root_pos_w"] == pytest.approx([1.0, 2.0, 0.35])
    assert loaded["root_quat_wxyz"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert len(loaded["joint_pos"]) == 20


def test_apply_state_writes_pose_velocity_and_joints(monkeypatch):
    stub = types.SimpleNamespace(
        float32="float32",
        tensor=_stub_tensor,
    )
    monkeypatch.setitem(sys.modules, "torch", stub)

    state = scene_clip.capture_state(_Robot())
    target = _Robot()
    scene_clip.apply_state(target, state, "cpu")

    assert target.written["root_pose"].shape == (1, 7)
    assert target.written["root_pose"].values[:3] == pytest.approx([1.0, 2.0, 0.35])
    assert target.written["root_pose"].values[3:] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert target.written["root_vel"].shape == (1, 6)
    assert target.written["joint_pos"].shape == (1, 20)
    assert target.written["joint_pos"].values[19] == pytest.approx(19.0)


def test_load_state_rejects_an_incomplete_file(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text('{"root_pos_w": [0, 0, 0]}')
    with pytest.raises(ValueError, match="root_quat_wxyz"):
        scene_clip.load_state(path)
