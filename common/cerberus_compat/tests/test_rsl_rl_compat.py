import importlib.metadata
import sys
import types
from dataclasses import MISSING

import pytest

from common.cerberus_compat import rsl_rl_compat as rc
from common.cerberus_compat.rsl_rl_compat import (
    CheckpointFormatError,
    check_checkpoint_format,
    detect_checkpoint_format,
    export_policy,
    get_policy_for_export,
    make_runner,
    migrate_agent_cfg,
    rsl_rl_major,
)


def set_version(monkeypatch, version):
    """Make importlib.metadata.version('rsl-rl-lib') return `version` (None = not installed)."""
    real = importlib.metadata.version

    def fake(name):
        if name == "rsl-rl-lib":
            if version is None:
                raise importlib.metadata.PackageNotFoundError(name)
            return version
        return real(name)

    monkeypatch.setattr(importlib.metadata, "version", fake)


def install_fake_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


class FakeCfg:
    """Stands in for RslRlOnPolicyRunnerCfg: has .to_dict() and class_name."""

    def __init__(self, class_name="OnPolicyRunner"):
        self.class_name = class_name
        self.device = "cuda:0"

    def to_dict(self):
        return {"class_name": self.class_name, "device": self.device}


class RecordingRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env, self.train_cfg, self.log_dir, self.device = (
            env,
            train_cfg,
            log_dir,
            device,
        )


class OnPolicyRunner(RecordingRunner):
    pass


class DistillationRunner(RecordingRunner):
    pass


@pytest.fixture
def fake_runners(monkeypatch):
    install_fake_module(monkeypatch, "rsl_rl")
    install_fake_module(
        monkeypatch,
        "rsl_rl.runners",
        OnPolicyRunner=OnPolicyRunner,
        DistillationRunner=DistillationRunner,
    )


# ---- version / major dispatch -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "version, major",
    [("3.1.2", 3), ("4.0.0", 4), ("5.5.1", 5), ("5.0.0.dev1", 5), ("10.0", 10)],
)
def test_rsl_rl_major(monkeypatch, version, major):
    set_version(monkeypatch, version)
    assert rsl_rl_major() == major


def test_rsl_rl_major_missing_package(monkeypatch):
    set_version(monkeypatch, None)
    with pytest.raises(RuntimeError, match="rsl-rl-lib"):
        rsl_rl_major()


def test_rsl_rl_major_unparseable(monkeypatch):
    set_version(monkeypatch, "dev")
    with pytest.raises(RuntimeError, match="major"):
        rsl_rl_major()


# ---- make_runner ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["3.1.2", "3.9.0"])
def test_make_runner_rejects_unsupported_rsl_rl_3x(monkeypatch, fake_runners, version):
    set_version(monkeypatch, version)
    with pytest.raises(RuntimeError, match=r"not supported.*>= 4"):
        make_runner(object(), FakeCfg(), log_dir=None, device="cpu")


def test_make_runner_accepts_rsl_rl_4_and_up(monkeypatch, fake_runners):
    install_fake_module(monkeypatch, "isaaclab_rl")
    install_fake_module(monkeypatch, "isaaclab_rl.rsl_rl")  # no migration helper: skipped
    for version in ("4.0.0", "5.5.1"):
        set_version(monkeypatch, version)
        assert type(make_runner(object(), FakeCfg(), log_dir=None, device="cpu")) is OnPolicyRunner


def test_make_runner_5x_migrates_cfg_object_once(monkeypatch, fake_runners):
    set_version(monkeypatch, "5.5.1")
    seen = []

    def handle_deprecated(cfg, installed_version):
        seen.append((cfg, installed_version))
        cfg.device = "cuda:9"  # proves to_dict() runs after migration
        return cfg

    install_fake_module(monkeypatch, "isaaclab_rl")
    install_fake_module(
        monkeypatch,
        "isaaclab_rl.rsl_rl",
        handle_deprecated_rsl_rl_cfg=handle_deprecated,
    )
    cfg = FakeCfg()
    runner = make_runner(object(), cfg, log_dir=None, device="cuda:1")
    assert seen == [(cfg, "5.5.1")]
    assert runner.train_cfg["device"] == "cuda:9"
    assert (
        runner.device == "cuda:1"
    )  # explicit device argument is passed through untouched


def test_make_runner_5x_dict_is_not_migrated(monkeypatch, fake_runners):
    set_version(monkeypatch, "5.5.1")
    install_fake_module(monkeypatch, "isaaclab_rl")
    install_fake_module(
        monkeypatch,
        "isaaclab_rl.rsl_rl",
        handle_deprecated_rsl_rl_cfg=lambda *a, **k: pytest.fail(
            "a dict cannot be migrated"
        ),
    )
    cfg = {"class_name": "DistillationRunner"}
    runner = make_runner(object(), cfg, log_dir=None, device="cpu")
    assert type(runner) is DistillationRunner
    assert runner.train_cfg is cfg


def test_make_runner_5x_without_lab_helper_skips_migration(monkeypatch, fake_runners):
    set_version(monkeypatch, "5.5.1")
    install_fake_module(monkeypatch, "isaaclab_rl")
    install_fake_module(
        monkeypatch, "isaaclab_rl.rsl_rl"
    )  # no handle_deprecated_rsl_rl_cfg attribute
    runner = make_runner(object(), FakeCfg(), log_dir=None, device="cpu")
    assert type(runner) is OnPolicyRunner


def test_migrate_agent_cfg_skips_already_migrated_cfg(monkeypatch):
    set_version(monkeypatch, "5.5.1")
    install_fake_module(monkeypatch, "isaaclab_rl")
    install_fake_module(
        monkeypatch,
        "isaaclab_rl.rsl_rl",
        handle_deprecated_rsl_rl_cfg=lambda *a, **k: pytest.fail(
            "not idempotent in Lab; must be skipped"
        ),
    )
    cfg = types.SimpleNamespace(
        actor=types.SimpleNamespace(distribution_cfg=object()), critic=MISSING
    )
    assert migrate_agent_cfg(cfg) is cfg


def test_make_runner_custom_runner_cls(monkeypatch, fake_runners):
    class Best(OnPolicyRunner):
        pass

    for version in ("4.0.0", "5.5.1"):
        set_version(monkeypatch, version)
        install_fake_module(monkeypatch, "isaaclab_rl")
        install_fake_module(
            monkeypatch,
            "isaaclab_rl.rsl_rl",
            handle_deprecated_rsl_rl_cfg=lambda cfg, v: cfg,
        )
        runner = make_runner(
            object(), FakeCfg(), log_dir="x", device="cpu", runner_cls=Best
        )
        assert type(runner) is Best


def test_make_runner_unknown_class_name(monkeypatch, fake_runners):
    set_version(monkeypatch, "5.5.1")
    with pytest.raises(ValueError, match="Unsupported runner class: Weird"):
        make_runner(object(), FakeCfg("Weird"), log_dir=None, device="cpu")


# ---- get_policy_for_export / export_policy -------------------------------------------------------------


def test_get_policy_for_export_5x_uses_get_policy(monkeypatch):
    set_version(monkeypatch, "5.5.1")
    actor = object()
    runner = types.SimpleNamespace(alg=types.SimpleNamespace(get_policy=lambda: actor))
    assert get_policy_for_export(runner) is actor


def test_export_policy_5x_uses_runner_methods(monkeypatch):
    set_version(monkeypatch, "5.5.1")
    calls = []
    runner = types.SimpleNamespace(
        export_policy_to_jit=lambda path, filename: calls.append(
            ("jit", path, filename)
        ),
        export_policy_to_onnx=lambda path, filename: calls.append(
            ("onnx", path, filename)
        ),
    )
    export_policy(runner, "/out", jit_filename="a.pt", onnx_filename="a.onnx")
    assert calls == [("jit", "/out", "a.pt"), ("onnx", "/out", "a.onnx")]


# ---- checkpoint format ---------------------------------------------------------------------------------


LEGACY = {"model_state_dict": {}, "optimizer_state_dict": {}, "iter": 3, "infos": None}
SPLIT = {
    "actor_state_dict": {},
    "critic_state_dict": {},
    "optimizer_state_dict": {},
    "iter": 3,
    "infos": None,
}
DISTILL = {
    "student_state_dict": {},
    "teacher_state_dict": {},
    "optimizer_state_dict": {},
}


def test_detect_checkpoint_format():
    assert detect_checkpoint_format(LEGACY) == "legacy"
    assert detect_checkpoint_format(SPLIT) == "split"
    assert detect_checkpoint_format(DISTILL) == "split"
    assert detect_checkpoint_format({"foo": 1}) == "unknown"
    assert detect_checkpoint_format([1, 2]) == "unknown"


def fake_torch_with(monkeypatch, loaded):
    calls = []

    def load(path, map_location=None, weights_only=None):
        calls.append((path, map_location, weights_only))
        if isinstance(loaded, Exception):
            raise loaded
        return loaded

    install_fake_module(monkeypatch, "torch", load=load)
    return calls


@pytest.fixture
def ckpt(tmp_path):
    path = tmp_path / "model_0.pt"
    path.write_bytes(b"x")
    return str(path)


def test_legacy_checkpoint_on_5x_raises_naming_both_formats(monkeypatch, ckpt):
    set_version(monkeypatch, "5.5.1")
    calls = fake_torch_with(monkeypatch, LEGACY)
    with pytest.raises(CheckpointFormatError) as exc:
        check_checkpoint_format(ckpt)
    msg = str(exc.value)
    assert "model_state_dict" in msg and "actor_state_dict" in msg
    assert "LEGACY" in msg and "SPLIT" in msg
    assert "5.5.1" in msg and "3.1.2" in msg and ckpt in msg
    assert calls == [(ckpt, "cpu", False)]


def test_split_formats_pass(monkeypatch, ckpt):
    set_version(monkeypatch, "5.5.1")
    fake_torch_with(monkeypatch, SPLIT)
    assert check_checkpoint_format(ckpt) == "split"
    fake_torch_with(monkeypatch, DISTILL)
    assert check_checkpoint_format(ckpt) == "split"


def test_unreadable_or_unknown_checkpoint_is_left_to_the_runner(monkeypatch, ckpt):
    set_version(monkeypatch, "5.5.1")
    fake_torch_with(monkeypatch, RuntimeError("corrupt"))
    assert check_checkpoint_format(ckpt) == "unknown"
    fake_torch_with(monkeypatch, {"something": 1})
    assert check_checkpoint_format(ckpt) == "unknown"


def test_missing_checkpoint_file(monkeypatch, tmp_path):
    set_version(monkeypatch, "5.5.1")
    fake_torch_with(monkeypatch, LEGACY)
    with pytest.raises(FileNotFoundError):
        check_checkpoint_format(str(tmp_path / "nope.pt"))


def test_module_imports_without_isaac_or_rsl_rl():
    assert (
        "rsl_rl" not in vars(rc)
        and "isaaclab" not in vars(rc)
        and "torch" not in vars(rc)
    )
