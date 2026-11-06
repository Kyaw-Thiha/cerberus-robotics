"""Pure-python tests for the Go2 + ARX5 Arena embodiment module (no isaaclab / Arena needed)."""

import argparse
import ast
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.cerberus_description.go2_arx5_spec import (
    ARX5_ARM_JOINTS,
    GO2_ARX5_EMBODIMENT,
    GO2_ARX5_TCP_PRIM_REL_PATH,
    GRIPPER_CLOSED_M,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_M,
    TCP_BODY_NAME,
)
from common.cerberus_worlds.arena_assets import go2_arx5_embodiment as emb
from common.cerberus_worlds.arena_assets import go2_embodiment as g2
from policy.manipulation.go2_arx5_stack import layout

ROOT = Path(__file__).resolve().parents[4]
W2A_ENV_CFG = ROOT / "policy" / "manipulation" / "go2_arx5_stack" / "env_cfg.py"


# --- constants against the spec ------------------------------------------------------------------------------
def test_arm_action_spec_matches_spec():
    spec = emb.ARM_ACTION_SPEC
    assert (
        spec["joint_names"]
        == list(ARX5_ARM_JOINTS)
        == [f"joint{i}" for i in range(1, 7)]
    )
    assert spec["body_name"] == TCP_BODY_NAME == "eef_link"
    assert spec["scale"] == 0.5 == layout.IK_ACTION_SCALE
    assert spec["body_offset"] is None
    assert spec["asset_name"] == "robot"
    assert spec["controller"] == {
        "command_type": "pose",
        "use_relative_mode": True,
        "ik_method": "dls",
    }


def test_gripper_action_spec_matches_spec():
    spec = emb.GRIPPER_ACTION_SPEC
    assert spec["joint_names"] == list(GRIPPER_JOINT_NAMES) == ["joint7", "joint8"]
    assert spec["open_command_expr"] == {
        "joint7": GRIPPER_OPEN_M,
        "joint8": GRIPPER_OPEN_M,
    }
    assert spec["close_command_expr"] == {
        "joint7": GRIPPER_CLOSED_M,
        "joint8": GRIPPER_CLOSED_M,
    }
    assert (GRIPPER_OPEN_M, GRIPPER_CLOSED_M) == (0.044, 0.0)


def test_no_leg_term_and_disjoint_joint_sets():
    arm, grip = (
        set(emb.ARM_ACTION_SPEC["joint_names"]),
        set(emb.GRIPPER_ACTION_SPEC["joint_names"]),
    )
    legs = set(GO2_ARX5_EMBODIMENT.joint_groups["legs"])
    assert not (arm & grip) and not (arm & legs) and not (grip & legs)
    assert arm == set(GO2_ARX5_EMBODIMENT.joint_groups["arm"])
    assert grip == set(GO2_ARX5_EMBODIMENT.joint_groups["gripper"])
    assert [emb.ARM_ACTION_NAME, emb.GRIPPER_ACTION_NAME] == [
        "arm_action",
        "gripper_action",
    ]
    assert emb.ACTION_DIM == 7


def test_frame_paths():
    assert emb.ROBOT_BASE_PRIM_PATH == "{ENV_REGEX_NS}/Robot/Geometry/base"
    assert (
        emb.EE_TARGET_PRIM_PATH == "{ENV_REGEX_NS}/Robot/" + GO2_ARX5_TCP_PRIM_REL_PATH
    )
    assert emb.EE_FRAME_NAME == "ee_frame"


def _call_keywords(tree: ast.AST, func_name: str) -> dict[str, ast.expr]:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", None)) == func_name
        ):
            return {k.arg: k.value for k in node.keywords if k.arg}
    raise AssertionError(f"no call to {func_name} in {W2A_ENV_CFG}")


def test_terms_mirror_w2a_env_cfg_source():
    tree = ast.parse(W2A_ENV_CFG.read_text())
    ik = _call_keywords(tree, "DifferentialInverseKinematicsActionCfg")
    assert ast.unparse(ik["scale"]) == "layout.IK_ACTION_SCALE"
    assert ast.unparse(ik["body_name"]) == "TCP_BODY_NAME"
    assert ast.unparse(ik["joint_names"]) == "list(ARX5_ARM_JOINTS)"
    ctrl = _call_keywords(tree, "DifferentialIKControllerCfg")
    assert (
        ast.literal_eval(ctrl["command_type"]) == "pose"
        and ast.literal_eval(ctrl["ik_method"]) == "dls"
    )
    assert ast.literal_eval(ctrl["use_relative_mode"]) is True
    grip = _call_keywords(tree, "BinaryJointPositionActionCfg")
    assert ast.unparse(grip["joint_names"]) == "list(GRIPPER_JOINT_NAMES)"
    assert "GRIPPER_OPEN_M" in ast.unparse(grip["open_command_expr"])
    assert "GRIPPER_CLOSED_M" in ast.unparse(grip["close_command_expr"])


def _class_assignments(tree: ast.AST, class_name: str) -> list[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [
                t.id
                for n in node.body
                if isinstance(n, ast.Assign)
                for t in n.targets
                if isinstance(t, ast.Name)
            ]
    raise AssertionError(class_name)


def test_observation_terms_mirror_w2a_env_cfg_source():
    tree = ast.parse(W2A_ENV_CFG.read_text())
    w2a_policy = _class_assignments(tree, "PolicyCfg")
    # the kitchen env has no cube_* terms (objects come from the Arena scene); everything else is shared
    assert [t for t in w2a_policy if not t.startswith("cube_")] == list(
        emb.OBSERVATION_TERMS
    )
    assert (
        _class_assignments(tree, "SubtaskCfg")
        == list(emb.SUBTASK_TERM_NAMES)
        == [layout.SUBTASK_GRASP_SIGNAL]
    )


# --- kitchen options and geometry ----------------------------------------------------------------------------
def _parse(*argv):
    parser = argparse.ArgumentParser()
    emb.add_environment_cli_args(parser)
    return parser.parse_args(list(argv))


def test_cli_defaults_equal_dataclass_defaults():
    assert emb.options_from_namespace(_parse()) == emb.KitchenOptions()
    opts = emb.options_from_namespace(
        _parse(
            "--robot_xy",
            "2",
            "-1.5",
            "--riser_height_m",
            "0.28",
            "--pick_object",
            "jug",
            "--plate_size_m",
            "0.2",
            "0.1",
        )
    )
    assert opts.robot_xy == (2.0, -1.5) and opts.riser_height_m == 0.28
    assert opts.pick_object == "jug" and opts.plate_size_m == (0.2, 0.1)
    assert (
        emb.options_from_namespace(SimpleNamespace()) == emb.KitchenOptions()
    )  # missing attributes: defaults


def test_pick_object_choices_are_lightwheel_sdk_objects():
    assert emb.DEFAULT_PICK_OBJECT in emb.LIGHTWHEEL_SDK_OBJECTS
    with pytest.raises(SystemExit):
        _parse(
            "--pick_object", "cracker_box"
        )  # Arena Nucleus object, not Lightwheel SDK


def test_validate_options_and_warnings():
    assert emb.validate_options(emb.KitchenOptions()) == []
    assert emb.option_warnings(emb.KitchenOptions()) == []
    bad = emb.KitchenOptions(
        pick_object="nope",
        riser_height_m=-1,
        object_height_m=0.2,
        plate_thickness_m=0.0,
    )
    assert len(emb.validate_options(bad)) == 4
    assert emb.option_warnings(emb.KitchenOptions(riser_height_m=0.5))
    assert emb.option_warnings(emb.KitchenOptions(riser_height_m=0.0))
    assert emb.validate_options(emb.KitchenOptions(episode_length_s=0))


def test_geometry_default_inside_reach_envelope():
    g = emb.kitchen_geometry(emb.KitchenOptions())
    x0, y0 = emb.DEFAULT_ROBOT_XY
    assert g["robot_pos"] == (x0, y0, layout.ROBOT_BASE_Z) and g["robot_yaw_rad"] == 0.0
    reach_y = layout.TOPDOWN_REACH_ABS_Y - layout.REACH_MARGIN_M
    for kind in ("pick", "plate"):
        lo, hi = g[f"{kind}_range_min"], g[f"{kind}_range_max"]
        eps = 1e-9
        assert (
            layout.CUBE_X_RANGE[0] - eps
            <= lo[0] - x0
            <= hi[0] - x0
            <= layout.CUBE_X_RANGE[1] + eps
        )
        assert -reach_y - eps <= lo[1] - y0 <= hi[1] - y0 <= reach_y + eps
        assert lo[2] == hi[2]
    # the two y regions must not overlap, and the plate half width must not reach the pick region
    assert g["plate_range_max"][1] < g["pick_range_min"][1]
    plate_half = g["plate_size"][1] / 2
    assert g["plate_range_max"][1] + plate_half < g["pick_range_min"][1]
    riser = g["riser"]
    assert riser["size"] == (
        layout.TABLE_DEPTH_X,
        layout.TABLE_WIDTH_Y,
        layout.TABLE_TOP_Z,
    )
    assert riser["center"][2] == pytest.approx(layout.TABLE_TOP_Z / 2)
    # objects spawn above the riser top
    assert (
        g["pick_range_min"][2] > g["support_top_z"]
        and g["plate_range_min"][2] > g["support_top_z"]
    )
    # the riser covers the whole placement region in x/y
    cx, cy, _ = riser["center"]
    sx, sy, _ = riser["size"]
    for kind in ("pick", "plate"):
        for corner in (g[f"{kind}_range_min"], g[f"{kind}_range_max"]):
            assert abs(corner[0] - cx) <= sx / 2 and abs(corner[1] - cy) <= sy / 2


def test_kitchen_pick_yaw_is_fixed_hardware_aware_test_pose():
    geometry = emb.kitchen_geometry(emb.KitchenOptions())
    expected_yaw = layout.CUBE_YAW_RANGE[0]
    assert emb.PICK_YAW_RANGE == (expected_yaw, expected_yaw) == (-0.3, -0.3)
    assert geometry["pick_yaw_range"] == (-0.3, -0.3)


def test_w2b_default_regions_are_conservatively_reachable_and_separated():
    geometry = emb.kitchen_geometry(emb.KitchenOptions())
    robot_x, robot_y = geometry["robot_pos"][:2]
    riser = geometry["riser"]
    center_x, center_y, _ = riser["center"]
    size_x, size_y, _ = riser["size"]

    for kind in ("pick", "plate"):
        lo, hi = geometry[f"{kind}_range_min"], geometry[f"{kind}_range_max"]
        corners = ((lo[0], lo[1]), (lo[0], hi[1]), (hi[0], lo[1]), (hi[0], hi[1]))
        assert max(math.hypot(x - robot_x, y - robot_y) for x, y in corners) <= 0.39
        assert center_x - size_x / 2 <= lo[0] <= hi[0] <= center_x + size_x / 2
        assert center_y - size_y / 2 <= lo[1] <= hi[1] <= center_y + size_y / 2

    # Full default object/plate footprints stay on the riser and do not intersect.
    pick_half_x, pick_half_y = (side / 2 for side in geometry["pick_object_footprint"])
    plate_half_x, plate_half_y = (side / 2 for side in geometry["plate_size"][:2])
    pick_x = (
        geometry["pick_range_min"][0] - pick_half_x,
        geometry["pick_range_max"][0] + pick_half_x,
    )
    pick_y = (
        geometry["pick_range_min"][1] - pick_half_y,
        geometry["pick_range_max"][1] + pick_half_y,
    )
    plate_x = (
        geometry["plate_range_min"][0] - plate_half_x,
        geometry["plate_range_max"][0] + plate_half_x,
    )
    plate_y = (
        geometry["plate_range_min"][1] - plate_half_y,
        geometry["plate_range_max"][1] + plate_half_y,
    )
    assert center_x - size_x / 2 <= min(pick_x[0], plate_x[0])
    assert max(pick_x[1], plate_x[1]) <= center_x + size_x / 2
    assert center_y - size_y / 2 <= min(pick_y[0], plate_y[0])
    assert max(pick_y[1], plate_y[1]) <= center_y + size_y / 2
    separated = (
        pick_x[1] <= plate_x[0]
        or plate_x[1] <= pick_x[0]
        or pick_y[1] <= plate_y[0]
        or plate_y[1] <= pick_y[0]
    )
    assert separated


def test_geometry_without_riser_and_other_robot_xy():
    g = emb.kitchen_geometry(
        emb.KitchenOptions(riser_height_m=0.0, robot_xy=(3.0, 1.0))
    )
    assert g["riser"] is None and g["robot_pos"][:2] == (3.0, 1.0)
    assert g["pick_range_min"][0] == pytest.approx(3.0 + emb.PICK_X_RANGE[0])


def test_place_reference_pos_releases_object_on_top_of_plate():
    plate = (1.0, 2.0, 0.311)
    t, h = 0.01, 0.04
    ref = emb.place_reference_pos(plate, t, h)
    assert ref[:2] == (1.0, 2.0)
    # ScriptedPickPlace releases at ref.z + cube_size(=h) + margin: the held object centre must be plate top + h/2
    release_z = ref[2] + h
    assert release_z == pytest.approx(plate[2] + t / 2 + h / 2)


class _Obj:
    pass


def test_apply_mimic_overrides_with_fakes():
    def subtask():
        s = _Obj()
        s.action_noise = 0.005
        return s

    cfg = _Obj()
    cfg.datagen_config = _Obj()
    cfg.subtask_configs = {"robot": [subtask(), subtask()]}
    assert emb.apply_mimic_overrides(cfg) is cfg
    d = cfg.datagen_config
    assert (
        d.generation_relative is True
        and d.generation_select_src_per_subtask is True
        and d.seed == 1
    )
    assert d.name == emb.MIMIC_DATAGEN_OVERRIDES["name"]
    first, second = cfg.subtask_configs["robot"]
    assert first.action_noise == second.action_noise == 0.03
    assert (
        first.num_interpolation_steps == 5
        and first.apply_noise_during_interpolation is False
    )
    assert first.description != second.description


def test_mimic_overrides_match_w2a_mimic_env_cfg():
    tree = ast.parse(
        (ROOT / "policy/manipulation/go2_arx5_stack/mimic_env_cfg.py").read_text()
    )
    source = ast.unparse(tree)
    for key in (
        "generation_relative",
        "generation_select_src_per_subtask",
        "max_num_failures",
    ):
        assert (
            f"self.datagen_config.{key} = {emb.MIMIC_DATAGEN_OVERRIDES[key]!r}"
            in source
        )
    assert f"action_noise={emb.MIMIC_SUBTASK_OVERRIDES['action_noise']}" in source
    assert (
        f"num_interpolation_steps={emb.MIMIC_SUBTASK_OVERRIDES['num_interpolation_steps']}"
        in source
    )


def test_control_rate_matches_w2a_env_cfg():
    src = W2A_ENV_CFG.read_text()
    assert f"self.decimation = {emb.DECIMATION}" in src
    assert f"self.sim.dt = {emb.SIM_DT}" in src


# --- laziness / wiring ---------------------------------------------------------------------------------------
def test_module_imports_without_isaaclab():
    code = (
        "import sys\n"
        "import common.cerberus_worlds.arena_assets.go2_arx5_embodiment\n"
        "import common.cerberus_worlds.arena_assets.go2_embodiment\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('isaaclab', 'isaaclab_arena', 'isaaclab_tasks', 'isaacsim', 'omni', 'pxr', 'torch')]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_unknown_attribute_raises_and_lazy_names_listed():
    with pytest.raises(AttributeError):
        emb.DoesNotExist  # noqa: B018
    assert set(emb._LAZY_NAMES) == {
        "Go2Arx5Embodiment",
        "ProceduralBoxObject",
        emb.ENV_FACTORY_CLASS,
    }
    assert emb.ENV_FACTORY_PATH.endswith(f"go2_arx5_embodiment:{emb.ENV_FACTORY_CLASS}")


def test_register_needs_arena_cleanly():
    try:
        import isaaclab_arena  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            emb.register_go2_arx5_assets()
        with pytest.raises(ImportError):
            g2.register_go2_arx5_assets()
    else:  # pragma: no cover - only on a pod
        pytest.skip(
            "Arena importable here; registration is exercised by scripts/arena_mimic_go2_arx5.py smoke"
        )


def test_go2_hook_calls_the_new_module(monkeypatch):
    sentinel = {"Go2Arx5Embodiment": object}
    monkeypatch.setattr(emb, "register_go2_arx5_assets", lambda: sentinel)
    assert g2.register_go2_arx5_assets() is sentinel
    assert (
        g2.register_embodiment_cfg(GO2_ARX5_EMBODIMENT)
        == g2.GO2_ARX5_ARENA_NAME
        == "go2_arx5"
    )


def test_arena_backend_registers_go2_arx5_through_the_hook(monkeypatch):
    from common.cerberus_worlds.backends.arena_backend import ArenaBackend

    calls = []
    monkeypatch.setattr(
        emb, "register_go2_arx5_assets", lambda: calls.append("go2_arx5") or {}
    )
    backend = ArenaBackend()
    backend.register_embodiment(GO2_ARX5_EMBODIMENT)
    assert calls == ["go2_arx5"]
    assert backend._arena_names == {"go2_arx5": "go2_arx5"}
    assert backend._embodiments["go2_arx5"] is GO2_ARX5_EMBODIMENT


def test_unknown_embodiment_still_rejected():
    from common.cerberus_description.embodiment import EmbodimentCfg

    cfg = EmbodimentCfg(
        name="spot", articulation_cfg_entry_point="a:B", joint_groups={"legs": ("j",)}
    )
    with pytest.raises(ValueError, match="spot"):
        g2.register_embodiment_cfg(cfg)


# --- manager term functions with fakes (torch needed; absent from the pure pixi env) -------------------------
def test_object_grasped_by_gripper_with_fakes():
    torch = pytest.importorskip("torch")

    class Data:
        pass

    def fake_env(finger_pos, obj_pos, tcp_pos):
        robot = SimpleNamespace(
            data=SimpleNamespace(joint_pos=torch.tensor([[0.0] * 6 + finger_pos])),
            find_joints=lambda names, preserve_order=True: (
                list(range(6, 8)),
                list(names),
            ),
        )
        ee = SimpleNamespace(
            data=SimpleNamespace(target_pos_w=torch.tensor([[tcp_pos]]))
        )
        obj = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([obj_pos])))
        return SimpleNamespace(scene={"robot": robot, "ee_frame": ee, "obj": obj})

    cfgs = dict(
        robot_cfg=SimpleNamespace(name="robot"),
        ee_frame_cfg=SimpleNamespace(name="ee_frame"),
        object_cfg=SimpleNamespace(name="obj"),
    )
    f = emb.object_grasped_by_gripper
    assert bool(f(fake_env([0.02, 0.02], [0, 0, 0.3], [0, 0, 0.31]), **cfgs)[0])
    assert not bool(
        f(fake_env([0.044, 0.044], [0, 0, 0.3], [0, 0, 0.31]), **cfgs)[0]
    )  # open
    assert not bool(
        f(fake_env([0.02, 0.02], [0, 0, 0.3], [0, 0, 0.5]), **cfgs)[0]
    )  # far
    assert not bool(
        f(fake_env([0.02, 0.044], [0, 0, 0.3], [0, 0, 0.31]), **cfgs)[0]
    )  # one finger open
