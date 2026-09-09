import gymnasium as gym
import pytest

# NOTE on local test execution: importing this module via its normal dotted path
# (policy.locomotion.core.platforms.registry) forces Python to first import every
# ancestor package, including policy/locomotion/__init__.py, which imports
# isaaclab_tasks -- only installed in an actual Isaac Lab environment (the RunPod
# remote image), not this local dev machine. This is a real, structural constraint
# of this repo's package layout, not fixable from inside this test file (pytest's
# own collection mechanism resolves the same dotted name regardless of how this
# file imports things internally). To run these tests locally: copy this file and
# registry.py into a bare scratch directory (no __init__.py ancestry) and run
# pytest there -- verified working this way when this file was written. On a pod
# with isaaclab_tasks installed, this file runs as-is, in place, no workaround
# needed.
from policy.locomotion.core.platforms.registry import (
    PlatformCfg,
    available_platforms,
    get_platform,
    register_platform,
)


def _make_cfg(name: str) -> PlatformCfg:
    return PlatformCfg(
        name=name,
        flat_task_id=f"Test-Flat-{name}-v0",
        flat_play_task_id=f"Test-Flat-{name}-Play-v0",
        rough_task_id=f"Test-Rough-{name}-v0",
        rough_play_task_id=f"Test-Rough-{name}-Play-v0",
        flat_env_cfg_entry_point=f"some.module.{name}:FlatEnvCfg",
        rough_env_cfg_entry_point=f"some.module.{name}:RoughEnvCfg",
        flat_agent_cfg=object(),
        rough_agent_cfg=object(),
        push_peak_force_n=100.0,
        push_peak_torque_nm=10.0,
        root_body_name="trunk",
        stairs_step_height_scale=0.7,
        num_steps_per_env=24,
    )


def test_checkpoint_prefix_defaults_from_name():
    cfg = _make_cfg("testbot")
    assert cfg.checkpoint_prefix == "testbot_locomotion"


def test_checkpoint_prefix_explicit_override_kept():
    cfg = PlatformCfg(
        name="testbot2",
        flat_task_id="Test-Flat-testbot2-v0",
        flat_play_task_id="Test-Flat-testbot2-Play-v0",
        rough_task_id="Test-Rough-testbot2-v0",
        rough_play_task_id="Test-Rough-testbot2-Play-v0",
        flat_env_cfg_entry_point="some.module:FlatEnvCfg",
        rough_env_cfg_entry_point="some.module:RoughEnvCfg",
        flat_agent_cfg=object(),
        rough_agent_cfg=object(),
        push_peak_force_n=100.0,
        push_peak_torque_nm=10.0,
        checkpoint_prefix="custom_prefix",
    )
    assert cfg.checkpoint_prefix == "custom_prefix"


def test_register_platform_adds_to_available_platforms():
    cfg = _make_cfg("regtest1")
    register_platform(cfg)
    assert "regtest1" in available_platforms()


def test_register_platform_rejects_duplicate_name():
    cfg = _make_cfg("regtest2")
    register_platform(cfg)
    with pytest.raises(ValueError, match="already registered"):
        register_platform(cfg)


def test_register_platform_calls_gym_register_for_all_four_task_ids():
    cfg = _make_cfg("regtest3")
    register_platform(cfg)
    registered_ids = set(gym.registry.keys())
    assert cfg.flat_task_id in registered_ids
    assert cfg.flat_play_task_id in registered_ids
    assert cfg.rough_task_id in registered_ids
    assert cfg.rough_play_task_id in registered_ids


def test_register_platform_play_variant_appends_play_suffix_to_class_name():
    cfg = _make_cfg("regtest4")
    register_platform(cfg)
    spec = gym.registry[cfg.rough_play_task_id]
    assert spec.kwargs["env_cfg_entry_point"] == f"{cfg.rough_env_cfg_entry_point}_PLAY"
    non_play_spec = gym.registry[cfg.rough_task_id]
    assert non_play_spec.kwargs["env_cfg_entry_point"] == cfg.rough_env_cfg_entry_point


def test_register_platform_passes_agent_cfg_as_object_not_string():
    cfg = _make_cfg("regtest5")
    register_platform(cfg)
    spec = gym.registry[cfg.flat_task_id]
    assert spec.kwargs["rsl_rl_cfg_entry_point"] is cfg.flat_agent_cfg


def test_get_platform_returns_registered_cfg():
    cfg = _make_cfg("regtest6")
    register_platform(cfg)
    assert get_platform("regtest6") is cfg


def test_get_platform_unknown_name_raises_with_available_list():
    with pytest.raises(ValueError, match="unknown platform 'nope'"):
        get_platform("nope")
