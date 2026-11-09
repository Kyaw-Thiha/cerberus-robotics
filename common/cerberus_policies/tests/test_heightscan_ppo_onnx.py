"""Tests for the ONNX-backed height-scan policy backend and its observation contract."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from common.cerberus_policies.backends.heightscan_ppo_onnx import (
    CONTRACT_PATH,
    HeightScanPpoOnnxBackend,
    _declares_external_data,
    _load_contract,
)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pixi.toml").exists())
CANDIDATE = REPO_ROOT / "policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate.onnx"
TERMS = ("base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions", "height_scan")


class FakeSession:
    def __init__(self, action_dim=12, fill=0.5):
        self.action_dim = action_dim
        self.feeds = []
        self.fill = fill

    def get_inputs(self):
        return [SimpleNamespace(name="obs")]

    def get_outputs(self):
        return [SimpleNamespace(shape=[1, self.action_dim])]

    def run(self, output_names, feeds):
        self.feeds.append(feeds)
        return [np.full((1, self.action_dim), self.fill, dtype=np.float32)]


def loaded_backend(monkeypatch, tmp_path: Path, action_dim: int = 12, fill: float = 0.5):
    graph = tmp_path / "policy.onnx"
    graph.write_bytes(b"\x08\x01graph-without-weights")
    backend = HeightScanPpoOnnxBackend()
    fake = FakeSession(action_dim=action_dim, fill=fill)
    import onnxruntime as ort

    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **k: fake)
    backend.load(str(graph), None)
    return backend, fake, graph


def named_observation(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "base_lin_vel": rng.normal(0, 0.1, 3),
        "base_ang_vel": rng.normal(0, 0.2, 3),
        "projected_gravity": np.array([0.0, 0.0, -1.0]) + rng.normal(0, 0.01, 3),
        "velocity_commands": rng.uniform(-1, 1, 3),
        "joint_pos": rng.normal(0, 0.05, 12),
        "joint_vel": rng.normal(0, 0.5, 12),
        "actions": rng.normal(0, 0.2, 12),
        "height_scan": rng.uniform(-1, 1, 187),
    }


def test_contract_matches_the_training_run():
    c = _load_contract(CONTRACT_PATH)
    assert [t["name"] for t in c["observation"]["terms"]] == list(TERMS)
    assert [t["dim"] for t in c["observation"]["terms"]] == [3, 3, 3, 3, 12, 12, 12, 187]
    assert sum(t["dim"] for t in c["observation"]["terms"]) == 235 == c["observation"]["total_dim"]
    assert c["action"] == {**c["action"], "dim": 12, "scale": 0.25, "use_default_offset": True}
    assert len(c["joints"]["order"]) == 12
    assert c["joints"]["order"][0] == "FL_hip_joint" and c["joints"]["order"][-1] == "RR_calf_joint"
    assert c["height_scan"]["grid"] == {"rows": 17, "cols": 11}
    assert c["control"] == {**c["control"], "sim_dt_s": 0.005, "decimation": 4, "policy_rate_hz": 50}
    assert c["commands"]["ranges"] == {"vx_mps": [-1.0, 1.0], "vy_mps": [-1.0, 1.0], "wz_radps": [-1.0, 1.0]}
    # the one thing the contract refuses to assert
    assert c["height_scan"]["convention_status"] == "unverified"


def test_assembles_named_terms_in_contract_order_and_runs_one_row(monkeypatch, tmp_path):
    backend, fake, _ = loaded_backend(monkeypatch, tmp_path)
    obs = named_observation()
    out = backend.act(obs)
    feed = fake.feeds[0]["obs"]
    assert feed.shape == (1, 235) and feed.dtype == np.float32
    expected = np.concatenate([obs[name].astype(np.float32) for name in TERMS])
    np.testing.assert_array_equal(feed[0], expected)
    assert out["actions"].shape == (12,)
    assert set(out["joint_targets"]) == set(_load_contract(CONTRACT_PATH)["joints"]["order"])


def test_accepts_a_preassembled_vector_and_rejects_bad_shape_and_dtype(monkeypatch, tmp_path):
    backend, fake, _ = loaded_backend(monkeypatch, tmp_path)
    backend.act(np.zeros(235, dtype=np.float32))
    assert fake.feeds[-1]["obs"].shape == (1, 235)
    with pytest.raises(ValueError, match="shape \\(235,\\)"):
        backend.act(np.zeros(234, dtype=np.float32))
    with pytest.raises(ValueError, match="dtype must be float32"):
        backend.act(np.zeros(235, dtype=np.float64))
    incomplete = named_observation()
    del incomplete["height_scan"]
    with pytest.raises(ValueError, match="missing terms.*height_scan"):
        backend.act(incomplete)
    obs = named_observation()
    obs["joint_vel"] = np.zeros(11)
    with pytest.raises(ValueError, match="joint_vel.*12 values"):
        backend.act(obs)


def test_action_to_joint_target_mapping_is_by_name(monkeypatch, tmp_path):
    backend, _, _ = loaded_backend(monkeypatch, tmp_path, fill=2.0)
    targets = backend.actions_to_joint_targets(np.full(12, 2.0))
    defaults = _load_contract(CONTRACT_PATH)["joints"]["default_joint_pos_rad"]
    for name in targets:
        assert targets[name] == pytest.approx(defaults[name] + 0.25 * 2.0)
    # the first joint's default is +0.1, the fifth -0.1: by-name, not by-position, matters
    assert targets["FL_hip_joint"] == pytest.approx(0.6)
    assert targets["FR_hip_joint"] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="expected 12 actions"):
        backend.actions_to_joint_targets(np.zeros(11))


def test_missing_external_weights_is_a_clear_error(monkeypatch, tmp_path):
    # an export that references weights outside the graph, without its .onnx.data sibling
    graph = tmp_path / "policy.onnx"
    graph.write_bytes(b"location" + b"policy.onnx.data")
    with pytest.raises(FileNotFoundError, match=r"missing external weights file 'policy\.onnx\.data'"):
        HeightScanPpoOnnxBackend().load(str(graph), None)
    # a self-contained graph (no external reference) loads without a .data file
    backend, _, _ = loaded_backend(monkeypatch, tmp_path)
    assert backend.act(np.zeros(235, dtype=np.float32))["actions"].shape == (12,)
    # and the real export is detected as external-weight carrying (skip without the gitignored file)
    if CANDIDATE.is_file():
        assert _declares_external_data(CANDIDATE, CANDIDATE.name + ".data")


def test_missing_graph_and_contract_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="ONNX graph not found"):
        HeightScanPpoOnnxBackend().load(str(tmp_path / "nope.onnx"), None)
    with pytest.raises(ValueError, match="needs a .onnx checkpoint path"):
        HeightScanPpoOnnxBackend().load(None, None)
    with pytest.raises(FileNotFoundError, match="observation contract not found"):
        _load_contract(tmp_path / "nope.yaml")


def test_act_before_load_raises():
    with pytest.raises(RuntimeError, match="call load\\(\\) before act"):
        HeightScanPpoOnnxBackend().act(np.zeros(235, dtype=np.float32))


def test_determinism(monkeypatch, tmp_path):
    backend, _, _ = loaded_backend(monkeypatch, tmp_path)
    obs = named_observation(seed=3)
    first, second = backend.act(obs), backend.act(obs)
    np.testing.assert_array_equal(first["actions"], second["actions"])
    assert first["joint_targets"] == second["joint_targets"]


def test_backend_module_imports_without_torch_or_ros():
    script = (
        "import sys\n"
        "import common.cerberus_policies.backends.heightscan_ppo_onnx\n"
        "bad = [m for m in ('torch', 'rclpy', 'cerberus_msgs', 'onnxruntime', 'numpy', 'isaaclab', 'isaacsim', 'omni') if m in sys.modules]\n"
        "print('BAD:' + ','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, env={"PYTHONPATH": str(REPO_ROOT), "PATH": ""}, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("BAD:"), out.stdout


@pytest.mark.skipif(
    not (CANDIDATE.is_file() and Path(str(CANDIDATE) + ".data").is_file()),
    reason="GA candidate ONNX (+ .onnx.data) is gitignored and not present locally",
)
def test_real_candidate_smoke_100_observations():
    backend = HeightScanPpoOnnxBackend()
    backend.load(str(CANDIDATE), None)
    contract = _load_contract(CONTRACT_PATH)
    rng = np.random.default_rng(0)
    for _ in range(100):
        parts = []
        for term in contract["observation"]["terms"]:
            if term["name"] == "projected_gravity":
                g = np.array([0.0, 0.0, -1.0]) + rng.normal(0, 0.02, 3)
                parts.append(g / np.linalg.norm(g))
            elif term["name"] == "height_scan":
                parts.append(rng.uniform(-0.2, 0.2, 187).astype(np.float32))
            else:
                parts.append(rng.normal(0, 0.3, term["dim"]).astype(np.float32))
        obs = np.concatenate(parts).astype(np.float32)
        out = backend.act(obs)
        assert out["actions"].shape == (12,)
        assert np.all(np.isfinite(out["actions"]))
        assert all(np.isfinite(v) for v in out["joint_targets"].values())
