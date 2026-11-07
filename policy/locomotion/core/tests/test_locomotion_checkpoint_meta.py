"""Tests for the checkpoint sidecar helpers (pure python, no torch, no Isaac).

These guard the rule the H2 handoff states as non-negotiable -- never overwrite a checkpoint -- and
the attribution fields the Phase 0.5 gate depends on (physics backend, config hash, wall clock). The
tensorboard wall-time reader is verified against a real event file when one is present: it is the
mechanism used to state the RUN_002 training time, and therefore the 2x retrain cap, without guessing.
"""

import importlib.util
import struct
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "checkpoint_meta.py"
_spec = importlib.util.spec_from_file_location("_checkpoint_meta_under_test", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

REPO_ROOT = Path(__file__).resolve().parents[4]
SEPTEMBER_EVENTS = REPO_ROOT / "policy" / "locomotion" / "runs" / "2026-09-08_19-28-32_analysis" / "events.out.tfevents.1788895745.cbacbf2829c4.886.0"


def make_meta(**overrides):
    base = {
        "checkpoint": "go2_locomotion_ga_v1_candidate.pt",
        "task_id": "Isaac-Velocity-Rough-Unitree-Go2-Cerberus-v0",
        "physics_backend": "isaacsim_physx",
        "iterations": 1500,
        "seed": 42,
        "num_envs": 4096,
        "wall_clock_s": 5306.8,
        "final_reward": 19.78,
        "config_hash_value": "0123456789abcdef",
        "git_hash_value": "deadbeef",
        "onnx": "go2_locomotion_ga_v1_candidate.onnx",
        "env_report": "docs/reports/env_report_2026-11-07.json",
        "terrain_levels": {"mean": 5.9, "max": 9, "frac_at_max": 0.147},
        "run_dir": "policy/locomotion/checkpoints/unitree_go2_rough/2026-11-07_10-00-00",
    }
    base.update(overrides)
    return _mod.build_meta(**base)


def write_event_file(path: Path, wall_times, truncate_last: bool = False):
    """Minimal TFRecord writer: only the framing + Event.wall_time field the reader cares about."""
    with open(path, "wb") as f:
        for wall in wall_times:
            payload = b"\x09" + struct.pack("<d", wall)  # field 1, wire type 1 (64-bit double)
            if truncate_last and wall is wall_times[-1]:
                f.write(struct.pack("<Q", len(payload)) + b"\0\0\0\0" + payload[:3])
                break
            f.write(struct.pack("<Q", len(payload)) + b"\0\0\0\0" + payload + b"\0\0\0\0")
    return path


class TestNoOverwrite:
    def test_ensure_new_paths_returns_free_paths(self, tmp_path):
        target = tmp_path / "free.pt"
        assert _mod.ensure_new_paths([target]) == [str(target)]
        assert not target.exists()  # it checks only; it must not create anything

    def test_ensure_new_paths_refuses_existing(self, tmp_path):
        existing = tmp_path / "model_1500.pt"
        existing.write_bytes(b"weights")
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            _mod.ensure_new_paths([tmp_path / "new.pt", existing])

    def test_write_meta_refuses_existing_sidecar(self, tmp_path):
        path = tmp_path / "candidate.meta.json"
        _mod.write_meta(path, make_meta())
        with pytest.raises(FileExistsError):
            _mod.write_meta(path, make_meta())

    def test_write_meta_requires_its_checkpoint_to_exist(self, tmp_path):
        # A sidecar naming a checkpoint that is not on disk reads as provenance but proves nothing.
        with pytest.raises(FileNotFoundError, match="do not exist"):
            _mod.write_meta(tmp_path / "final.meta.json", make_meta(), checkpoint_paths=[tmp_path / "gone.pt"])

    def test_write_meta_accepts_an_existing_checkpoint(self, tmp_path):
        checkpoint = tmp_path / "go2_locomotion_ga_v1_candidate.pt"
        checkpoint.write_bytes(b"weights")
        path = _mod.write_meta(tmp_path / "candidate.meta.json", make_meta(), checkpoint_paths=[checkpoint])
        assert Path(path).is_file()


class TestMetaValidation:
    def test_build_meta_is_complete(self):
        meta = make_meta()
        for field in _mod.REQUIRED_FIELDS:
            assert meta[field] is not None
        assert meta["created_at"]

    def test_missing_fields_are_named(self):
        meta = make_meta()
        del meta["physics_backend"]
        del meta["seed"]
        with pytest.raises(ValueError, match="physics_backend"):
            _mod.validate_meta(meta)

    def test_zero_iterations_rejected(self):
        # build_meta validates on construction: an invalid sidecar is never even produced.
        with pytest.raises(ValueError, match="iterations must be positive"):
            make_meta(iterations=0)

    def test_non_positive_wall_clock_rejected(self):
        # A zero wall-clock means it was never measured -- not that it was instant.
        with pytest.raises(ValueError, match="wall_clock_s must be positive"):
            make_meta(wall_clock_s=0)

    def test_empty_physics_backend_rejected(self):
        meta = make_meta()
        meta["physics_backend"] = ""
        with pytest.raises(ValueError, match="physics_backend"):
            _mod.validate_meta(meta)

    def test_sidecar_roundtrip(self, tmp_path):
        path = tmp_path / "x.meta.json"
        meta = make_meta()
        _mod.write_meta(path, meta)
        assert _mod.load_meta(path) == meta
        assert not list(tmp_path.glob("*.tmp"))  # atomic write leaves no debris


class TestConfigHash:
    def test_same_content_same_hash(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        for d in (a, b):
            d.mkdir()
            (d / "env.yaml").write_text("sim:\n  dt: 0.005\n")
        assert _mod.config_hash(a) == _mod.config_hash(b)

    def test_changed_content_changes_hash(self, tmp_path):
        first = tmp_path / "first"
        first.mkdir()
        (first / "env.yaml").write_text("sim:\n  dt: 0.005\n")
        before = _mod.config_hash(first)
        (first / "env.yaml").write_text("sim:\n  dt: 0.01\n")
        assert _mod.config_hash(first) != before

    def test_file_names_matter(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        for d, name in ((a, "env.yaml"), (b, "agent.yaml")):
            d.mkdir()
            (d / name).write_text("same bytes\n")
        assert _mod.config_hash(a) != _mod.config_hash(b)

    def test_hash_length_is_stable(self, tmp_path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "env.yaml").write_text("x: 1\n")
        assert len(_mod.config_hash(d)) == 16


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _len_field(field: int, payload: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def summary_record(step: int, scalars: dict) -> bytes:
    """Event{step, summary: Summary{value: [{tag, simple_value}]}} -- what the trainer writes."""
    # Summary.value is a repeated Value{tag(1), simple_value(2)}: each scalar is its own Value
    # message, nested one level under Summary.value(1). A flat encoding parses as garbage.
    values = b"".join(
        _len_field(
            1,
            _len_field(1, tag.encode()) + _varint((2 << 3) | 5) + struct.pack("<f", float(v)),
        )
        for tag, v in scalars.items()
    )
    return _varint((2 << 3) | 0) + _varint(step) + _len_field(5, values)


def write_scalar_events(path: Path, rows):
    with open(path, "wb") as f:
        for step, scalars in rows:
            payload = summary_record(step, scalars)
            f.write(struct.pack("<Q", len(payload)) + b"\0\0\0\0" + payload + b"\0\0\0\0")
    return path


class TestTfeventsScalars:
    def test_reads_last_value_per_tag(self, tmp_path):
        path = write_scalar_events(
            tmp_path / "events.out.tfevents.1",
            [(1, {"Train/mean_reward": 1.5}), (2, {"Train/mean_reward": 19.78})],
        )
        out = _mod.tfevents_last_scalars(path, ["Train/mean_reward"])
        assert out["Train/mean_reward"]["value"] == pytest.approx(19.78, rel=1e-6)
        assert out["Train/mean_reward"]["step"] == 2

    def test_missing_tag_is_absent_not_none(self, tmp_path):
        path = write_scalar_events(tmp_path / "events.out.tfevents.2", [(1, {"Other/x": 2.0})])
        assert _mod.tfevents_last_scalars(path, ["Train/mean_reward"]) == {}

    def test_several_tags_in_one_event(self, tmp_path):
        path = write_scalar_events(
            tmp_path / "events.out.tfevents.3",
            [(7, {"Train/mean_reward": 3.0, "Curriculum/terrain_levels": 5.5})],
        )
        out = _mod.tfevents_last_scalars(path, ["Train/mean_reward", "Curriculum/terrain_levels"])
        assert out["Train/mean_reward"]["value"] == pytest.approx(3.0)
        assert out["Curriculum/terrain_levels"]["value"] == pytest.approx(5.5)


class TestTrainingSummary:
    def test_maps_logged_tags_to_sidecar_fields(self, tmp_path):
        path = write_scalar_events(
            tmp_path / "events.out.tfevents.9",
            [
                (
                    1499,
                    {
                        "Train/mean_reward": 19.7756,
                        "Curriculum/terrain_levels": 5.9007,
                        "Curriculum/terrain_level_max": 9.0,
                        "Curriculum/terrain_level_frac_at_max": 0.1474,
                        "Episode_Termination/base_contact": 0.1033,
                    },
                )
            ],
        )
        out = _mod.training_summary(path)
        assert out["final_reward"] == pytest.approx(19.7756, rel=1e-5)
        assert out["final_iteration"] == 1499
        assert out["terrain_levels"]["mean"] == pytest.approx(5.9007, rel=1e-5)
        assert out["terrain_levels"]["max"] == pytest.approx(9.0)
        assert out["terrain_levels"]["frac_at_max"] == pytest.approx(0.1474, rel=1e-4)
        assert out["fall_rate"] == pytest.approx(0.1033, rel=1e-4)

    def test_absent_tags_are_omitted(self, tmp_path):
        path = write_scalar_events(tmp_path / "events.out.tfevents.10", [(3, {"Train/mean_reward": 4.0})])
        out = _mod.training_summary(path)
        assert out == {"final_reward": pytest.approx(4.0), "final_iteration": 3}

    @pytest.mark.skipif(not SEPTEMBER_EVENTS.is_file(), reason="September run tree is gitignored")
    def test_reproduces_the_published_september_final_row(self):
        # RUN_002's final row: reward 19.78, terrain mean 5.90, max 9, 14.7% at max, fall rate 10.3%.
        out = _mod.training_summary(SEPTEMBER_EVENTS)
        assert out["final_reward"] == pytest.approx(19.78, abs=0.01)
        assert out["final_iteration"] == 1499
        assert out["terrain_levels"]["mean"] == pytest.approx(5.90, abs=0.01)
        assert out["terrain_levels"]["max"] == 9
        assert out["terrain_levels"]["frac_at_max"] == pytest.approx(0.147, abs=0.001)
        assert out["fall_rate"] == pytest.approx(0.103, abs=0.001)


class TestTfeventsWallTime:
    def test_span_from_synthetic_events(self, tmp_path):
        path = write_event_file(tmp_path / "events.out.tfevents.1", [100.0, 200.5, 350.25])
        out = _mod.tfevents_wall_time_span(path)
        assert out["first"] == pytest.approx(100.0)
        assert out["last"] == pytest.approx(350.25)
        assert out["span_s"] == pytest.approx(250.25)
        assert out["events"] == 3

    def test_truncated_tail_is_tolerated(self, tmp_path):
        # A run that is still writing leaves a partial record; the parsed span must still be usable.
        path = write_event_file(tmp_path / "events.out.tfevents.2", [10.0, 20.0, 30.0], truncate_last=True)
        out = _mod.tfevents_wall_time_span(path)
        assert out["span_s"] == pytest.approx(10.0)
        assert out["events"] == 2

    def test_empty_file_reports_no_span(self, tmp_path):
        path = tmp_path / "events.out.tfevents.3"
        path.write_bytes(b"")
        assert _mod.tfevents_wall_time_span(path)["span_s"] is None

    @pytest.mark.skipif(not SEPTEMBER_EVENTS.is_file(), reason="September run tree is gitignored")
    def test_real_september_event_file(self):
        out = _mod.tfevents_wall_time_span(SEPTEMBER_EVENTS)
        # RUN_002 trained for ~1.5 h; the cap for the GA retrain is 2x this span.
        assert out["span_s"] == pytest.approx(5306.8, abs=5.0)
        assert out["events"] > 40000
