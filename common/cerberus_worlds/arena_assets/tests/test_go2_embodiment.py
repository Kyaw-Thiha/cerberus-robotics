import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.cerberus_description.embodiment import EmbodimentCfg
from common.cerberus_description.go2_arx5_spec import (
    GO2_ARX5_EMBODIMENT,
    GO2_LEG_JOINTS,
)
from common.cerberus_worlds import arena_assets
from common.cerberus_worlds.arena_assets import go2_embodiment as g2


def test_go2_embodiment_shape():
    assert isinstance(g2.GO2_EMBODIMENT, EmbodimentCfg)
    assert g2.GO2_EMBODIMENT.name == "go2"
    assert (
        g2.GO2_EMBODIMENT.articulation_cfg_entry_point
        == "isaaclab_assets.robots.unitree:UNITREE_GO2_CFG"
    )
    legs = g2.GO2_EMBODIMENT.joint_groups["legs"]
    assert legs == tuple(GO2_LEG_JOINTS)
    assert len(legs) == 12 and len(set(legs)) == 12
    assert g2.GO2_EMBODIMENT.joint_names == legs
    assert g2.GO2_EMBODIMENT.checkpoint_prefix == "go2"


def test_modules_import_without_isaaclab():
    code = (
        "import sys\n"
        "import common.cerberus_worlds.arena_assets.go2_embodiment\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('isaaclab', 'isaaclab_arena', 'isaacsim', 'omni', 'pxr')]\n"
        "assert not bad, bad\n"
    )
    root = Path(__file__).resolve().parents[4]
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root)


def test_go2_arx5_hook_needs_arena_not_a_stub():
    # W2b: the hook is real now; without isaaclab/Arena installed it fails on import.
    try:
        import isaaclab_arena  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("Arena importable; covered by the pod spike")
    with pytest.raises(ImportError):
        g2.register_go2_arx5_assets()
    with pytest.raises(ImportError):
        g2.register_embodiment_cfg(GO2_ARX5_EMBODIMENT)


def test_register_embodiment_cfg_rejects_unknown():
    cfg = EmbodimentCfg(
        name="spot", articulation_cfg_entry_point="a:B", joint_groups={"legs": ("j",)}
    )
    with pytest.raises(ValueError, match="spot"):
        g2.register_embodiment_cfg(cfg)


def test_register_go2_assets_needs_arena():
    # In a plain-python env Arena is absent: the failure must be a clean ImportError, not a silent no-op.
    try:
        import isaaclab_arena  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            g2.register_go2_assets()
    else:  # pragma: no cover - only on a pod
        pytest.skip(
            "Arena importable here; registration is exercised by scripts/arena_spike_go2.py"
        )


def test_unknown_module_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        g2.DoesNotExist  # noqa: B018


# --- scene-data helpers -----------------------------------------------------------------------------------
def _make_scene(
    root: Path,
    folder: str,
    files=("start_result_navigation.usd",),
    category="home",
    materials=True,
):
    d = root / "grscenes" / category / "scenes" / folder
    d.mkdir(parents=True)
    for f in files:
        (d / f).write_text("#usda 1.0\n")
    if materials:
        (root / "grscenes" / category / "Materials").mkdir(parents=True, exist_ok=True)
    return d


def test_discover_scenes_fake_tree(tmp_path):
    _make_scene(tmp_path, "b_usd")
    _make_scene(tmp_path, "a_usd", files=("start_result_raw.usd",))
    _make_scene(
        tmp_path,
        "both_usd",
        files=("start_result_raw.usd", "start_result_navigation.usd"),
    )
    _make_scene(tmp_path, "empty_usd", files=("metadata.json",))
    scenes = arena_assets.discover_scenes("grscenes", root=tmp_path)
    assert [s.scene_id for s in scenes] == [
        "grscenes:a_usd",
        "grscenes:b_usd",
        "grscenes:both_usd",
    ]
    by_id = {s.scene_id: s for s in scenes}
    assert by_id["grscenes:a_usd"].variant == "raw"
    assert by_id["grscenes:both_usd"].variant == "navigation"  # navigation preferred
    assert by_id["grscenes:b_usd"].category == "home"
    assert (
        by_id["grscenes:b_usd"].mdl_path == tmp_path / "grscenes" / "home" / "Materials"
    )
    assert by_id["grscenes:b_usd"].usd_path.name == "start_result_navigation.usd"
    assert set(by_id["grscenes:b_usd"].to_metadata()) == {
        "folder",
        "usd_path",
        "variant",
        "mdl_path",
    }


def test_discover_scenes_missing_dir_is_empty(tmp_path):
    assert arena_assets.discover_scenes("grscenes", root=tmp_path / "nope") == []


def test_find_scene_accepts_bare_folder_and_reports_available(tmp_path):
    _make_scene(tmp_path, "x_usd")
    assert arena_assets.find_scene("x_usd", root=tmp_path).scene_id == "grscenes:x_usd"
    assert (
        arena_assets.find_scene("grscenes:x_usd", root=tmp_path).folder.name == "x_usd"
    )
    with pytest.raises(FileNotFoundError, match="grscenes:x_usd"):
        arena_assets.find_scene("grscenes:missing_usd", root=tmp_path)


def test_scenes_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(arena_assets.SCENES_DIR_ENV, str(tmp_path))
    assert arena_assets.scenes_root() == tmp_path
    monkeypatch.delenv(arena_assets.SCENES_DIR_ENV)
    assert arena_assets.scenes_root() == arena_assets.repo_root() / "data" / "scenes"
    assert (arena_assets.repo_root() / "pixi.toml").exists()


def test_join_and_ensure_mdl_system_path(monkeypatch, tmp_path):
    assert arena_assets.join_mdl_system_path(None, "/a") == "/a"
    assert arena_assets.join_mdl_system_path("/a:/b", "/a") == "/a:/b"
    assert arena_assets.join_mdl_system_path("/a", "/b") == "/a:/b"
    monkeypatch.delenv("MDL_SYSTEM_PATH", raising=False)
    assert arena_assets.ensure_mdl_system_path(tmp_path) is True
    assert arena_assets.ensure_mdl_system_path(tmp_path) is False
    assert arena_assets.ensure_mdl_system_path(None) is False
    import os

    assert os.environ["MDL_SYSTEM_PATH"] == str(tmp_path)


def test_spawn_pose_identity():
    pos, quat = arena_assets.spawn_pose_to_background_pose((0.0, 0.0, 0.0))
    assert pos == (0.0, 0.0, 0.0) or all(abs(p) < 1e-12 for p in pos)
    assert quat == (0.0, 0.0, 0.0, 1.0)


def test_spawn_pose_translation_moves_scene_opposite():
    pos, quat = arena_assets.spawn_pose_to_background_pose((3.0, -2.0, 0.0))
    assert pos == pytest.approx((-3.0, 2.0, 0.0))
    assert quat == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_spawn_pose_yaw_maps_spawn_point_to_origin():
    x, y, yaw = 2.0, 1.0, math.pi / 2
    pos, quat = arena_assets.spawn_pose_to_background_pose((x, y, yaw))
    # rotate the scene-frame spawn point by the background rotation (about z) and add the translation
    theta = 2 * math.atan2(quat[2], quat[3])
    px = math.cos(theta) * x - math.sin(theta) * y + pos[0]
    py = math.sin(theta) * x + math.cos(theta) * y + pos[1]
    assert (px, py) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert theta == pytest.approx(-yaw)


# --- waypoint functions (torch needed) ----------------------------------------------------------------------
def test_waypoint_functions_with_fake_env():
    torch = pytest.importorskip("torch")
    pos = torch.tensor([[1.0, 0.0, 0.4], [5.0, 5.0, 0.4]])
    env = SimpleNamespace(
        device="cpu",
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_pos_w=pos))},
    )
    env.scene = _Scene(env.scene, torch.tensor([[0.0, 0.0, 0.0], [3.0, 5.0, 0.0]]))
    d = g2.waypoint_distance_xy(env, (2.0, 0.0))
    assert d.tolist() == pytest.approx([1.0, 0.0])
    assert g2.waypoint_reached(env, (1.0, 0.0), 0.1).tolist() == [True, False]
    assert g2.waypoint_reward(env, (2.0, 0.0)).tolist() == pytest.approx([-1.0, 0.0])


class _Scene(dict):
    def __init__(self, items, env_origins):
        super().__init__(items)
        self.env_origins = env_origins


def test_install_ue4_mdl_modules_copies_once_and_never_overwrites(tmp_path):
    src = tmp_path / "isaac" / "Ue4"
    src.mkdir(parents=True)
    for name in arena_assets.UE4_MDL_MODULES:
        (src / name).write_text(f"// {name}")
    mats = tmp_path / "Materials"
    mats.mkdir()
    (mats / "OmniUe4Base.mdl").write_text("// mine")  # already there: must survive

    copied = arena_assets.install_ue4_mdl_modules(mats, src)
    assert copied == ["OmniUe4Function.mdl", "OmniUe4Translucent.mdl"]
    assert (mats / "OmniUe4Base.mdl").read_text() == "// mine"
    assert arena_assets.install_ue4_mdl_modules(mats, src) == []  # idempotent


def test_install_ue4_mdl_modules_fails_loudly_when_isaac_copy_is_missing(tmp_path):
    (tmp_path / "Materials").mkdir()
    with pytest.raises(FileNotFoundError, match="OmniUe4Function.mdl"):
        arena_assets.install_ue4_mdl_modules(tmp_path / "Materials", tmp_path / "empty")


def test_ensure_mdl_system_path_installs_modules_when_isaac_is_present(monkeypatch, tmp_path):
    root = tmp_path / "isaac-sim"
    ue4 = root / "kit" / "mdl" / "core" / "Ue4"
    ue4.mkdir(parents=True)
    for name in arena_assets.UE4_MDL_MODULES:
        (ue4 / name).write_text("//")
    mats = tmp_path / "Materials"
    mats.mkdir()
    monkeypatch.setenv("ISAACSIM_ROOT_PATH", str(root))
    monkeypatch.delenv("MDL_SYSTEM_PATH", raising=False)
    assert arena_assets.ensure_mdl_system_path(mats) is True
    assert all((mats / n).is_file() for n in arena_assets.UE4_MDL_MODULES)


def test_ensure_mdl_system_path_skips_the_copy_without_isaac(monkeypatch, tmp_path):
    mats = tmp_path / "Materials"
    mats.mkdir()
    monkeypatch.setenv("ISAACSIM_ROOT_PATH", str(tmp_path / "no-isaac-here"))
    monkeypatch.delenv("MDL_SYSTEM_PATH", raising=False)
    assert arena_assets.ensure_mdl_system_path(mats) is True
    assert list(mats.iterdir()) == []
