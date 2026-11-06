"""Pure-python checks of the W2a stack task: geometry vs reach envelope, contract names, static structure of the
Isaac-dependent modules (parsed with `ast`, never imported), and `register_all` against a fake gymnasium."""

import ast
import math
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.cerberus_description import go2_arx5_spec as spec
from policy.manipulation.go2_arx5_stack import layout, register

PKG_DIR = Path(__file__).resolve().parent.parent
ISAAC_ROOTS = {
    "isaaclab",
    "isaaclab_tasks",
    "isaaclab_mimic",
    "isaaclab_assets",
    "isaacsim",
    "omni",
    "pxr",
    "torch",
}


def _tree(name: str) -> ast.Module:
    return ast.parse((PKG_DIR / name).read_text())


def _module_level_imports(tree: ast.Module) -> set[str]:
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _assigned_names(cls: ast.ClassDef) -> set[str]:
    return {
        t.id
        for n in cls.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name
    )


# --- geometry vs reach envelope ---------------------------------------------------------------------------------


def test_arm_base_height_matches_task_brief():
    assert layout.ARM_BASE_Z == pytest.approx(
        layout.ROBOT_BASE_Z + spec.GO2_ARX5_MOUNT_XYZ[2]
    )
    assert 0.35 <= layout.ARM_BASE_Z <= 0.45


def test_base_height_is_the_go2_standing_height():
    front = layout.go2_standing_height(layout.GO2_DEFAULT_THIGH["front"])
    rear = layout.go2_standing_height(layout.GO2_DEFAULT_THIGH["rear"])
    assert front == pytest.approx(0.333, abs=1e-3)
    assert rear == pytest.approx(0.324, abs=1e-3)
    assert rear - 0.005 <= layout.ROBOT_BASE_Z <= front + 0.005
    assert layout.ROBOT_BASE_POS == (0.0, 0.0, layout.ROBOT_BASE_Z)


def test_cube_range_is_inside_the_topdown_reach_envelope_with_margin():
    x_lo, x_hi = layout.CUBE_X_RANGE
    env_lo, env_hi = layout.TOPDOWN_REACH_X
    assert x_lo >= env_lo + layout.REACH_MARGIN_M - 1e-9
    assert x_hi <= env_hi - layout.REACH_MARGIN_M + 1e-9
    y_lo, y_hi = layout.CUBE_Y_RANGE
    assert (
        max(abs(y_lo), abs(y_hi))
        <= layout.TOPDOWN_REACH_ABS_Y - layout.REACH_MARGIN_M + 1e-9
    )
    # the arm base is at x = y = 0 (mount offsets are z only), so env-frame x/y equal arm-base-frame x/y
    assert spec.GO2_ARX5_MOUNT_XYZ[:2] == (0.0, 0.0)


def test_waypoint_heights_stay_in_the_scanned_band():
    cube_top = layout.TABLE_TOP_Z + layout.CUBE_SIZE
    highest = (
        layout.TABLE_TOP_Z + 1.5 * layout.CUBE_SIZE + layout.HOVER_HEIGHT
    )  # place hover
    lowest = layout.CUBE_Z  # grasp at the cube centre
    assert lowest == pytest.approx(
        layout.TABLE_TOP_Z + layout.CUBE_SIZE / 2, abs=layout.CUBE_REST_MARGIN_M + 1e-9
    )
    assert cube_top < highest
    # offline scan covered table tops 0.28-0.30 m with exactly these four heights relative to the table
    assert 0.28 <= layout.TABLE_TOP_Z <= 0.30
    assert layout.HOVER_HEIGHT == pytest.approx(0.10)


def test_cubes_sit_on_the_table_with_margin():
    assert layout.CUBE_Z > layout.TABLE_TOP_Z + layout.CUBE_SIZE / 2 - 1e-12
    x_lo, x_hi = layout.CUBE_X_RANGE
    y_lo, y_hi = layout.CUBE_Y_RANGE
    table_x = (layout.TABLE_FRONT_X, layout.TABLE_FRONT_X + layout.TABLE_DEPTH_X)
    half_w = layout.TABLE_WIDTH_Y / 2
    margin = 0.05
    assert table_x[0] + margin <= x_lo and x_hi <= table_x[1] - margin
    assert -half_w + margin <= y_lo and y_hi <= half_w - margin
    for pos in (layout.CUBE_1_DEFAULT_POS, layout.CUBE_2_DEFAULT_POS):
        assert (
            x_lo <= pos[0] <= x_hi
            and y_lo <= pos[1] <= y_hi
            and pos[2] == layout.CUBE_Z
        )
    assert layout.TABLE_CENTER == pytest.approx(
        (layout.TABLE_FRONT_X + layout.TABLE_DEPTH_X / 2, 0.0, layout.TABLE_TOP_Z / 2)
    )
    assert layout.TABLE_SIZE[2] == layout.TABLE_TOP_Z  # solid block from the ground
    assert layout.CUBE_DROP_MIN_Z < layout.TABLE_TOP_Z


def test_table_clears_the_go2_footprint():
    assert (
        layout.TABLE_FRONT_X - layout.GO2_FRONT_EXTENT_X
        >= layout.TABLE_MIN_CLEARANCE_TO_GO2_X - 1e-9
    )


def test_two_cubes_fit_in_the_range_with_the_min_separation():
    (x_lo, x_hi), (y_lo, y_hi) = layout.CUBE_X_RANGE, layout.CUBE_Y_RANGE
    diagonal = math.hypot(x_hi - x_lo, y_hi - y_lo)
    assert diagonal > layout.CUBE_MIN_SEPARATION * 1.5  # rejection sampling has room
    assert layout.CUBE_MIN_SEPARATION > layout.CUBE_SIZE * math.sqrt(
        2
    )  # cubes never overlap, whatever the yaw
    d = math.dist(layout.CUBE_1_DEFAULT_POS, layout.CUBE_2_DEFAULT_POS)
    assert d >= layout.CUBE_MIN_SEPARATION


def test_cube_fits_the_gripper_and_thresholds_are_consistent():
    assert spec.GRIPPER_OPEN_M > spec.GRIPPER_CLOSED_M
    assert (spec.GRIPPER_OPEN_M, spec.GRIPPER_CLOSED_M) == (0.044, 0.0)
    assert (
        2 * spec.GRIPPER_OPEN_M - layout.CUBE_SIZE >= 0.04
    )  # >= 2 cm slack per side when open
    assert layout.GRIPPER_OPEN_ATOL < layout.GRIPPER_THRESHOLD < spec.GRIPPER_OPEN_M
    # a cube-holding finger (about half the cube side from the centre line) counts as "not open"
    assert spec.GRIPPER_OPEN_M - layout.CUBE_SIZE / 2 > layout.GRIPPER_THRESHOLD


def test_grasp_pitch_is_topdown():
    assert layout.GRASP_PITCH_RAD == pytest.approx(math.pi / 2)


# --- contract names from the spec ----------------------------------------------------------------------------------


def test_contract_names():
    assert spec.TCP_BODY_NAME == "eef_link"
    assert layout.EEF_PRIM_RELATIVE_PATH == spec.GO2_ARX5_TCP_PRIM_REL_PATH
    assert layout.EEF_PRIM_RELATIVE_PATH.endswith("/" + spec.TCP_BODY_NAME)
    assert layout.BASE_PRIM_RELATIVE_PATH == spec.GO2_ARX5_BASE_PRIM_REL_PATH
    assert tuple(spec.GRIPPER_JOINT_NAMES) == ("joint7", "joint8")
    assert tuple(spec.ARX5_ARM_JOINTS) == tuple(f"joint{i}" for i in range(1, 7))
    assert len(spec.GO2_ARX5_TCP_OFFSET_XYZ) == 3
    assert layout.ACTION_DIM == 7  # 6 (IK, relative pose) + 1 (gripper)
    assert layout.IK_ACTION_SCALE == 0.5


# --- static structure of the Isaac-dependent modules ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["layout.py", "waypoints.py", "register.py", "__init__.py"]
)
def test_pure_modules_have_no_isaac_or_gym_imports(name):
    assert not (
        _module_level_imports(_tree(name)) & (ISAAC_ROOTS | {"gymnasium", "gym"})
    )


def test_package_init_files_are_lightweight():
    for path in (PKG_DIR / "__init__.py", PKG_DIR.parent / "__init__.py"):
        tree = ast.parse(path.read_text())
        assert not [
            n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
        ], path


def test_register_imports_gymnasium_lazily():
    tree = _tree("register.py")
    inside = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Import) and any(a.name == "gymnasium" for a in n.names)
    ]
    assert inside, "register_all should import gymnasium inside the function"


def test_entry_points_resolve_to_classes_in_this_package():
    for dotted in (
        register.ENV_CFG_ENTRY_POINT,
        register.MIMIC_ENV_ENTRY_POINT,
        register.MIMIC_ENV_CFG_ENTRY_POINT,
    ):
        module, cls = dotted.split(":")
        assert module.startswith(register.PACKAGE + ".")
        filename = module.rsplit(".", 1)[1] + ".py"
        assert (PKG_DIR / filename).is_file()
        _class(_tree(filename), cls)  # raises StopIteration if missing
    assert PKG_DIR == _REPO_ROOT.joinpath(*register.PACKAGE.split("."))


def test_mimic_cfg_subtasks():
    tree = _tree("mimic_env_cfg.py")
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "SubTaskConfig"
    ]
    assert len(calls) == 2
    first = {kw.arg: kw.value for kw in calls[0].keywords}
    last = {kw.arg: kw.value for kw in calls[1].keywords}
    assert ast.literal_eval(first["object_ref"]) == "cube_2"
    assert isinstance(first["subtask_term_signal"], ast.Name)
    assert first["subtask_term_signal"].id == "SUBTASK_GRASP_SIGNAL"
    assert ast.literal_eval(last["object_ref"]) == "cube_1"
    assert ast.literal_eval(last["subtask_term_signal"]) is None
    assert ast.literal_eval(last["subtask_term_offset_range"]) == (0, 0)
    source = (PKG_DIR / "mimic_env_cfg.py").read_text()
    assert "MimicEnvCfg" in source and "MIMIC_EEF_NAME" in source


def test_env_cfg_defines_the_terms_mimic_needs():
    tree = _tree("env_cfg.py")
    subtask_group = _class(tree, "SubtaskCfg")
    assert layout.SUBTASK_GRASP_SIGNAL in _assigned_names(subtask_group)
    terminations = _class(tree, "TerminationsCfg")
    names = _assigned_names(terminations)
    assert {"success", "time_out"} <= names
    actions = _class(tree, "ActionsCfg")
    fields = [n.target.id for n in actions.body if isinstance(n, ast.AnnAssign)]
    assert fields == [
        "arm_action",
        "gripper_action",
    ]  # arm first: the gripper is the last action entry


def test_mimic_env_reads_only_existing_obs_groups():
    source = (PKG_DIR / "mimic_env.py").read_text()
    assert (
        'obs_buf["policy"]["eef_pos"]' in source
        and 'obs_buf["policy"]["eef_quat"]' in source
    )
    assert 'obs_buf["subtask_terms"]' in source
    env_source = (PKG_DIR / "env_cfg.py").read_text()
    assert "eef_pos = ObsTerm" in env_source and "eef_quat = ObsTerm" in env_source


# --- register_all with a fake gymnasium ------------------------------------------------------------------------------


class FakeGym:
    """Mimics gymnasium's `register` / `registry` (re-registering an id is allowed there, just noisy)."""

    def __init__(self):
        self.registry = {}
        self.calls = []

    def register(self, id, entry_point=None, **kwargs):
        self.calls.append(id)
        self.registry[id] = {"entry_point": entry_point, **kwargs}


def test_register_all_registers_both_ids():
    gym = FakeGym()
    assert register.register_all(gym) is None  # None, see the module docstring
    assert set(gym.registry) == {
        "Cerberus-Go2Arx5-Stack-Cube-IK-Rel-v0",
        "Cerberus-Go2Arx5-Stack-Cube-IK-Rel-Mimic-v0",
    }
    assert tuple(gym.calls) == register.task_ids()
    plain = gym.registry[register.TASK_ID_IK_REL]
    assert plain["entry_point"] == "isaaclab.envs:ManagerBasedRLEnv"
    assert plain["kwargs"] == {"env_cfg_entry_point": register.ENV_CFG_ENTRY_POINT}
    mimic = gym.registry[register.TASK_ID_MIMIC]
    assert mimic["entry_point"] == register.MIMIC_ENV_ENTRY_POINT
    assert mimic["kwargs"] == {
        "env_cfg_entry_point": register.MIMIC_ENV_CFG_ENTRY_POINT
    }
    assert plain["disable_env_checker"] and mimic["disable_env_checker"]


def test_register_all_is_idempotent():
    gym = FakeGym()
    for _ in range(3):
        register.register_all(gym)
    assert len(gym.calls) == 2
    # an id someone else already registered is left alone
    other = FakeGym()
    other.registry[register.TASK_ID_MIMIC] = {"entry_point": "elsewhere"}
    register.register_all(other)
    register.register_all(other)
    assert other.calls == [register.TASK_ID_IK_REL]
    assert other.registry[register.TASK_ID_MIMIC] == {"entry_point": "elsewhere"}


def test_register_all_imports_gymnasium_lazily(monkeypatch):
    fake = FakeGym()
    module = types.ModuleType("gymnasium")
    module.register = fake.register
    module.registry = fake.registry
    monkeypatch.setitem(sys.modules, "gymnasium", module)
    register.register_all()
    register.register_all()
    assert tuple(fake.calls) == register.task_ids()


def test_task_ids_are_unique_and_namespaced():
    ids = register.task_ids()
    assert len(set(ids)) == len(ids) == 2
    assert all(i.startswith("Cerberus-Go2Arx5-") and i.endswith("-v0") for i in ids)
    assert ids[1] == ids[0].replace("-v0", "-Mimic-v0")
