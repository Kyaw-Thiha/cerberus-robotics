from pathlib import Path

import pytest
import yaml

PLATFORMS_YAML = Path(__file__).resolve().parents[1] / "config" / "platforms.yaml"

# Repo root: system/cerberus_bringup/test/test_platforms_yaml.py -> repo root
# is three parents up.
PLATFORMS_PY_DIR = Path(__file__).resolve().parents[3] / "policy" / "locomotion" / "core" / "platforms"


def _discover_python_platform_names() -> set[str]:
    """Walks policy/locomotion/core/platforms/ for the real set of
    Python-side platforms, instead of a hand-maintained hardcoded set that
    can silently drift from the actual directory contents. Needs no
    isaaclab_tasks import (pure filesystem walk), so it can run in this same
    test process.

    Convention (established in the platform-decoupling work): one directory
    per platform, each containing its own platform.py. Non-platform entries
    like registry.py, __pycache__, and __init__.py-only dirs are excluded.
    """
    names = set()
    for entry in PLATFORMS_PY_DIR.iterdir():
        if not entry.is_dir():
            continue
        if entry.name == "__pycache__":
            continue
        if (entry / "platform.py").is_file():
            names.add(entry.name)
    return names


EXPECTED_PYTHON_PLATFORM_NAMES = _discover_python_platform_names()


def _load_platforms() -> dict:
    with open(PLATFORMS_YAML) as f:
        return yaml.safe_load(f)


def test_platforms_yaml_exists_and_parses():
    platforms = _load_platforms()
    assert isinstance(platforms, dict)
    assert len(platforms) > 0


def test_discover_python_platform_names_finds_go2():
    # Regression guard on the directory-walk logic itself: today, exactly
    # {"go2"} should be found under policy/locomotion/core/platforms/.
    assert _discover_python_platform_names() == {"go2"}


def test_every_yaml_key_matches_a_known_python_platform_name():
    platforms = _load_platforms()
    unknown = set(platforms) - EXPECTED_PYTHON_PLATFORM_NAMES
    assert not unknown, f"platforms.yaml has keys with no matching Python PlatformCfg.name: {unknown}"


def test_every_known_python_platform_name_has_a_yaml_entry():
    # The more likely drift direction: a platform added under
    # core/platforms/ but forgotten in platforms.yaml.
    platforms = _load_platforms()
    missing = EXPECTED_PYTHON_PLATFORM_NAMES - set(platforms)
    assert not missing, f"platforms.yaml is missing entries for Python platforms: {missing}"


def test_go2_entry_has_required_fields():
    go2 = _load_platforms()["go2"]
    assert "urdf_relative_path" in go2
    assert "locomotion_modes" in go2
    assert "default_locomotion_mode" in go2
    assert go2["default_locomotion_mode"] in go2["locomotion_modes"]
    assert "joint_names" in go2
    assert "default_joint_pos" in go2


def test_go2_joint_names_and_default_joint_pos_are_consistent():
    go2 = _load_platforms()["go2"]
    joint_names = go2["joint_names"]
    default_joint_pos = go2["default_joint_pos"]
    assert len(joint_names) == 12
    assert len(set(joint_names)) == 12, "joint_names must not contain duplicates"
    assert len(default_joint_pos) == len(joint_names)
    assert all(isinstance(v, (int, float)) for v in default_joint_pos)


@pytest.mark.parametrize("mode_name", ["blind", "perceptive"])
def test_go2_locomotion_mode_declares_checkpoint_derived_constants(mode_name):
    go2 = _load_platforms()["go2"]
    mode = go2["locomotion_modes"][mode_name]
    assert "height_scan_size" in mode
    assert "action_scale" in mode


def test_go2_default_locomotion_mode_is_blind():
    go2 = _load_platforms()["go2"]
    assert go2["default_locomotion_mode"] == "blind"


@pytest.mark.parametrize("mode_name", ["blind", "perceptive"])
def test_go2_locomotion_mode_declares_checkpoint_and_shape(mode_name):
    go2 = _load_platforms()["go2"]
    mode = go2["locomotion_modes"][mode_name]
    assert "checkpoint" in mode
    assert mode["observation_shape"] in ("proprioception_only", "proprioception_plus_height_scan")


def test_blind_mode_is_shape_a():
    go2 = _load_platforms()["go2"]
    assert go2["locomotion_modes"]["blind"]["observation_shape"] == "proprioception_only"


def test_perceptive_mode_is_shape_b():
    go2 = _load_platforms()["go2"]
    assert go2["locomotion_modes"]["perceptive"]["observation_shape"] == "proprioception_plus_height_scan"
