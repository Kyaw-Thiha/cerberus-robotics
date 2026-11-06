"""Pure-python tests of the two world backends (no isaaclab, no Arena)."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from common.cerberus_description.embodiment import EmbodimentCfg
from common.cerberus_worlds import registry
from common.cerberus_worlds.arena_assets import SCENES_DIR_ENV
from common.cerberus_worlds.backends import arena_backend as ab
from common.cerberus_worlds.backends import plain_usd_backend as pb
from common.cerberus_worlds.base import SceneInfo, WorldBackend

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True)
def _clean_registry():
    registry._reset_for_tests()
    yield
    registry._reset_for_tests()


def _fake_scenes(root: Path) -> None:
    """data/scenes tree as written by fetch_grscenes_from_r2.py."""
    home = root / "grscenes" / "home"
    (home / "Materials").mkdir(parents=True)
    for folder, usd in (
        ("scene_a_usd", "start_result_navigation.usd"),
        ("scene_b_usd", "start_result_raw.usd"),
    ):
        d = home / "scenes" / folder
        d.mkdir(parents=True)
        (d / usd).write_text("#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = \"Z\"\n)\n")
    (home / "scenes" / "not_a_scene").mkdir()
    (home / "scenes" / "not_a_scene" / "metadata.json").write_text("{}")


# --- kitchen id table ------------------------------------------------------------------------------------
def test_kitchen_table_is_10_by_10():
    assert (
        len(ab.LIGHTWHEEL_KITCHEN_LAYOUTS) == 10
        and len(ab.LIGHTWHEEL_KITCHEN_STYLES) == 10
    )
    assert [i for i, _, _ in ab.LIGHTWHEEL_KITCHEN_LAYOUTS] == list(range(1, 11))
    assert [i for i, _, _ in ab.LIGHTWHEEL_KITCHEN_STYLES] == list(range(1, 11))


def test_kitchen_ids_are_100_unique_and_match_arena_naming():
    ids = ab.kitchen_scene_ids()
    assert len(ids) == 100 and len(set(ids)) == 100
    assert all(re.fullmatch(r"lightwheel_kitchen_[a-z_0-9]+", i) for i in ids)
    assert ids[0] == "lightwheel_kitchen_one_wall_coastal"
    assert ids[-1] == "lightwheel_kitchen_wraparound_farmhouse2"
    assert "lightwheel_kitchen_u_shaped_with_island_scandinavian" in ids


def test_kitchen_scene_id_by_numbers_and_bounds():
    assert ab.kitchen_scene_id(1, 1) == "lightwheel_kitchen_one_wall_coastal"
    assert ab.kitchen_scene_id(10, 10) == "lightwheel_kitchen_wraparound_farmhouse2"
    for bad in ((0, 1), (1, 0), (11, 1), (1, 11)):
        with pytest.raises(ValueError):
            ab.kitchen_scene_id(*bad)


def test_kitchen_table_matches_vendored_arena_source_if_present():
    factory = (
        REPO_ROOT
        / "third_party"
        / "IsaacLab-Arena"
        / "isaaclab_arena"
        / "assets"
        / "lightwheel_kitchen_factory.py"
    )
    if not factory.is_file():
        pytest.skip("Arena not fetched (python scripts/fetch_arena.py)")
    import ast

    tree = ast.parse(factory.read_text())
    tables = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in (
                "_LIGHTWHEEL_KITCHEN_LAYOUTS",
                "_LIGHTWHEEL_KITCHEN_STYLES",
            ):
                tables[node.targets[0].id] = ast.literal_eval(node.value)
    assert tables["_LIGHTWHEEL_KITCHEN_LAYOUTS"] == ab.LIGHTWHEEL_KITCHEN_LAYOUTS
    assert tables["_LIGHTWHEEL_KITCHEN_STYLES"] == ab.LIGHTWHEEL_KITCHEN_STYLES


# --- Arena backend: scene listing -------------------------------------------------------------------------
def test_arena_list_scenes_without_grscenes_is_the_100_kitchens(tmp_path):
    scenes = ab.ArenaBackend(scenes_root=tmp_path).list_scenes()
    assert len(scenes) == 100
    assert all(isinstance(s, SceneInfo) for s in scenes)
    assert {s.category for s in scenes} == {"kitchen"} and {
        s.source for s in scenes
    } == {"lightwheel"}
    first = scenes[0]
    assert first.scene_id == "lightwheel_kitchen_one_wall_coastal"
    assert first.metadata == {
        "layout_id": 1,
        "layout": "one_wall",
        "style_id": 1,
        "style": "coastal",
    }


def test_arena_list_scenes_includes_fake_grscenes_tree(tmp_path):
    _fake_scenes(tmp_path)
    scenes = ab.ArenaBackend(scenes_root=tmp_path).list_scenes()
    assert len(scenes) == 102
    gr = [s for s in scenes if s.source == "grscenes"]
    assert [s.scene_id for s in gr] == ["grscenes:scene_a_usd", "grscenes:scene_b_usd"]
    assert {s.category for s in gr} == {"home"}
    assert (
        gr[0].metadata["variant"] == "navigation" and gr[1].metadata["variant"] == "raw"
    )
    assert gr[0].metadata["mdl_path"] == str(
        tmp_path / "grscenes" / "home" / "Materials"
    )
    assert len({s.scene_id for s in scenes}) == 102


def test_arena_scenes_root_env_override(tmp_path, monkeypatch):
    _fake_scenes(tmp_path)
    monkeypatch.setenv(SCENES_DIR_ENV, str(tmp_path))
    assert len(ab.ArenaBackend().list_scenes()) == 102


def test_arena_normalize_and_describe(tmp_path):
    _fake_scenes(tmp_path)
    b = ab.ArenaBackend(scenes_root=tmp_path)
    assert b.normalize_scene_ref("stock") == "stock"
    assert (
        b.normalize_scene_ref("lightwheel:1:1") == "lightwheel_kitchen_one_wall_coastal"
    )
    assert (
        b.normalize_scene_ref("lightwheel_kitchen_galley_rustic")
        == "lightwheel_kitchen_galley_rustic"
    )
    assert b.normalize_scene_ref("grscenes:scene_a_usd") == "grscenes:scene_a_usd"
    for bad in (
        "lightwheel:1",
        "lightwheel:a:b",
        "lightwheel:11:1",
        "nonsense",
        "lightwheel_kitchen_x_y",
    ):
        with pytest.raises(ValueError):
            b.normalize_scene_ref(bad)
    assert (
        b.describe_scene("lightwheel:3:2")["arena_asset"]
        == "lightwheel_kitchen_l_shaped_farmhouse1"
    )
    assert b.describe_scene("stock")["kind"] == "stock"
    gr = b.describe_scene("grscenes:scene_a_usd")
    assert gr["kind"] == "grscenes" and gr["usd_path"].endswith(
        "start_result_navigation.usd"
    )
    missing = b.describe_scene("grscenes:nope_usd")
    assert missing["usd_path"] is None and "nope_usd" in missing["error"]


def test_arena_configure_only_changes_given_values():
    b = ab.ArenaBackend()
    b.configure(env_spacing=12.0)
    b.configure(shared_scene=True, robot_pose=(1, 2, 0.5))
    b.configure()
    assert (b.env_spacing, b.shared_scene, b.robot_pose) == (
        12.0,
        True,
        (1.0, 2.0, 0.5),
    )


def test_arena_prepare_scene_env_sets_mdl_path(tmp_path, monkeypatch):
    _fake_scenes(tmp_path)
    monkeypatch.delenv("MDL_SYSTEM_PATH", raising=False)
    ab.ArenaBackend(scenes_root=tmp_path).prepare_scene_env("grscenes:scene_a_usd")
    import os

    assert os.environ["MDL_SYSTEM_PATH"] == str(
        tmp_path / "grscenes" / "home" / "Materials"
    )


def test_arena_go2_arx5_registration_needs_arena_not_a_stub():
    # W2b: the go2_arx5 hook is real now; without isaaclab/Arena installed it fails on import.
    try:
        import isaaclab_arena  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("Arena importable; covered by the pod spike")
    from common.cerberus_description.go2_arx5_spec import GO2_ARX5_EMBODIMENT
    from common.cerberus_worlds.arena_assets.go2_embodiment import (
        register_embodiment_cfg,
    )

    with pytest.raises(ImportError):
        register_embodiment_cfg(GO2_ARX5_EMBODIMENT)


def test_arena_isaac_side_methods_need_isaac_not_a_stub():
    try:
        import isaaclab_arena  # noqa: F401
    except ImportError:
        pass
    else:  # pragma: no cover - pod only
        pytest.skip("Arena importable; covered by the pod spike")
    b = ab.ArenaBackend()
    with pytest.raises(ImportError):
        b.load_scene("stock")
    with pytest.raises(ImportError):
        b.build_env(object(), object(), None, 1)


# --- plain USD backend ---------------------------------------------------------------------------------------
def test_plain_list_scenes_scans_fake_tree(tmp_path):
    _fake_scenes(tmp_path)
    scenes = pb.PlainUsdBackend(scenes_root=tmp_path).list_scenes()
    assert [s.scene_id for s in scenes] == [
        "grscenes:scene_a_usd",
        "grscenes:scene_b_usd",
    ]
    assert all(
        isinstance(s, SceneInfo) and s.source == "grscenes" and s.category == "home"
        for s in scenes
    )


def test_plain_list_scenes_empty_when_no_data(tmp_path):
    assert pb.PlainUsdBackend(scenes_root=tmp_path).list_scenes() == []


def test_plain_scenes_dir_env_override(tmp_path, monkeypatch):
    _fake_scenes(tmp_path)
    monkeypatch.setenv(SCENES_DIR_ENV, str(tmp_path))
    assert len(pb.PlainUsdBackend().list_scenes()) == 2


def test_plain_source_selects_subdirectory(tmp_path):
    other = tmp_path / "othersource" / "x_scene"
    other.mkdir(parents=True)
    (other / "start_result_raw.usd").write_text("#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = \"Z\"\n)\n")
    _fake_scenes(tmp_path)
    scenes = pb.PlainUsdBackend(
        source="othersource", scenes_root=tmp_path
    ).list_scenes()
    assert [s.scene_id for s in scenes] == ["othersource:x_scene"]


def test_plain_load_scene_returns_small_dict(tmp_path):
    _fake_scenes(tmp_path)
    scene = pb.PlainUsdBackend(scenes_root=tmp_path).load_scene("grscenes:scene_a_usd")
    assert scene["scene_id"] == "grscenes:scene_a_usd"
    assert scene["usd_path"].endswith("scene_a_usd/start_result_navigation.usd")
    assert scene["mdl_path"] == str(tmp_path / "grscenes" / "home" / "Materials")
    assert scene["variant"] == "navigation"
    stock = pb.PlainUsdBackend(scenes_root=tmp_path).load_scene("stock")
    assert stock["usd_path"] is None and stock["scene_id"] == "stock"
    with pytest.raises(FileNotFoundError):
        pb.PlainUsdBackend(scenes_root=tmp_path).load_scene("grscenes:missing_usd")


def test_plain_rejects_lightwheel_scenes():
    with pytest.raises(ValueError, match="Lightwheel"):
        pb.PlainUsdBackend().normalize_scene_ref("lightwheel:1:1")
    with pytest.raises(ValueError):
        pb.PlainUsdBackend().normalize_scene_ref("whatever")


def test_plain_register_embodiment_stores_cfg_and_requires_legs():
    b = pb.PlainUsdBackend()
    cfg = EmbodimentCfg(
        name="x", articulation_cfg_entry_point="a:B", joint_groups={"legs": ("j1",)}
    )
    b.register_embodiment(cfg)
    assert b._embodiment is cfg
    with pytest.raises(ValueError, match="legs"):
        b.register_embodiment(
            EmbodimentCfg(
                name="y",
                articulation_cfg_entry_point="a:B",
                joint_groups={"arm": ("j",)},
            )
        )


def test_plain_build_env_needs_isaac():
    try:
        import isaaclab  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            pb.PlainUsdBackend().build_env(
                {}, EmbodimentCfg("x", "a:B", {"legs": ("j",)}), None, 1
            )
    else:  # pragma: no cover - pod only
        pytest.skip("isaaclab importable; covered by the pod spike")


def test_plain_waypoint_params():
    d = pb.waypoint_params(None)
    assert (
        d["waypoint_xy"] == (2.0, 0.0)
        and d["radius"] == 0.5
        and d["episode_length_s"] == 10.0
    )
    assert pb.waypoint_params({"waypoint_xy": [1, 2], "radius": 0.2})[
        "waypoint_xy"
    ] == (1, 2)
    with pytest.raises(ValueError, match="bogus"):
        pb.waypoint_params({"bogus": 1})
    with pytest.raises(TypeError):
        pb.waypoint_params("nav")


def test_yaw_quaternion_helpers():
    assert pb._yaw_quat_xyzw(0.0) == (0.0, 0.0, 0.0, 1.0)
    x, y, z, w = pb._yaw_quat_xyzw(3.141592653589793)
    assert (x, y) == (0.0, 0.0) and abs(z - 1.0) < 1e-12 and abs(w) < 1e-12
    # without isaaclab_physx the installed Lab is taken to be the 2.x line: WXYZ
    if not pb._is_lab3():
        assert pb._quat_from_xyzw((0.1, 0.2, 0.3, 0.4)) == (0.4, 0.1, 0.2, 0.3)


# --- both backends: contract + registry ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,cls", [("arena", ab.ArenaBackend), ("plain_usd", pb.PlainUsdBackend)]
)
def test_backends_implement_the_contract_and_resolve_via_registry(name, cls):
    backend = registry.get_backend(name)
    assert isinstance(backend, cls) and isinstance(backend, WorldBackend)
    assert not cls.__abstractmethods__
    for helper in (
        "configure",
        "normalize_scene_ref",
        "describe_scene",
        "prepare_scene_env",
    ):
        assert callable(getattr(backend, helper))


def test_importing_backends_does_not_import_isaac():
    code = (
        "import sys\n"
        "import common.cerberus_worlds.backends.arena_backend, common.cerberus_worlds.backends.plain_usd_backend\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('isaaclab', 'isaaclab_arena', 'isaacsim', 'omni', 'pxr', 'torch')]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)


# --- spike script (dry-run is pure python) -------------------------------------------------------------------
def _load_spike():
    spec = importlib.util.spec_from_file_location(
        "arena_spike_go2", REPO_ROOT / "scripts" / "arena_spike_go2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spike_dry_run_lightwheel(capsys):
    spike = _load_spike()
    assert spike.main(["--dry-run", "--scene", "lightwheel:1:1"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["scene_id"] == "lightwheel_kitchen_one_wall_coastal"
    assert (
        plan["backend"] == "arena" and plan["num_envs"] == 16 and plan["steps"] == 200
    )
    assert len(plan["joint_groups"]["legs"]) == 12
    assert plan["problems"] == []


def test_spike_dry_run_plain_usd_stock_and_bad_args(capsys):
    spike = _load_spike()
    assert (
        spike.main(
            [
                "--dry-run",
                "--scene",
                "stock",
                "--backend",
                "plain_usd",
                "--num_envs",
                "4",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["num_envs"] == 4
    assert spike.main(["--dry-run", "--scene", "lightwheel:99:1"]) == 2
    assert (
        spike.main(["--dry-run", "--scene", "lightwheel:1:1", "--backend", "plain_usd"])
        == 2
    )
    assert spike.main(["--dry-run", "--num_envs", "0"]) == 2


def test_spike_dry_run_reports_missing_grscenes_data(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(SCENES_DIR_ENV, str(tmp_path))
    spike = _load_spike()
    assert spike.main(["--dry-run", "--scene", "grscenes:nope_usd"]) == 1
    out = capsys.readouterr()
    assert json.loads(out.out)["problems"] and "[PROBLEM]" in out.err
    _fake_scenes(tmp_path)
    registry._reset_for_tests()  # fresh backend instances
    assert spike.main(["--dry-run", "--scene", "grscenes:scene_a_usd"]) == 0


def test_spike_dry_run_go2_arx5_on_arena_is_flagged(capsys):
    spike = _load_spike()
    assert spike.main(["--dry-run", "--embodiment", "go2_arx5"]) == 1
    assert "W2" in capsys.readouterr().err


def _cm_scene(root: Path) -> None:
    """A centimetre-authored scene like GRScenes: metersPerUnit NOT authored, so USD reports its 0.01 fallback."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    folder = root / "grscenes" / "home" / "scenes" / "cm_usd"
    folder.mkdir(parents=True)
    (root / "grscenes" / "home" / "Materials").mkdir(exist_ok=True)
    stage = Usd.Stage.CreateNew(str(folder / "start_result_navigation.usd"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Root").GetPrim())
    stage.GetRootLayer().Save()


@pytest.mark.parametrize("backend_cls", [pb.PlainUsdBackend, ab.ArenaBackend])
def test_centimetre_scene_is_served_as_a_metre_wrapper(tmp_path, backend_cls):
    """Isaac does not convert units, so a cm scene spawned raw is 100x too big (a 2.4 m couch was 241 m on a pod)."""
    _cm_scene(tmp_path)
    info = backend_cls(scenes_root=tmp_path).describe_scene("grscenes:cm_usd")
    assert info["usd_path"].endswith("cm_usd/scene_metric.usd")
    assert info["source_usd_path"].endswith("cm_usd/start_result_navigation.usd")


def test_plain_load_scene_returns_the_metre_wrapper_for_a_centimetre_scene(tmp_path):
    _cm_scene(tmp_path)
    scene = pb.PlainUsdBackend(scenes_root=tmp_path).load_scene("grscenes:cm_usd")
    assert scene["usd_path"].endswith("cm_usd/scene_metric.usd")
    assert scene["source_usd_path"].endswith("cm_usd/start_result_navigation.usd")


def test_metre_scene_keeps_its_own_path(tmp_path):
    _fake_scenes(tmp_path)  # fixtures author metersPerUnit = 1
    info = pb.PlainUsdBackend(scenes_root=tmp_path).describe_scene("grscenes:scene_a_usd")
    assert info["usd_path"] == info["source_usd_path"]
