# Multi-Platform Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the buildable-now half of `docs/superpowers/specs/2026-09-09-multi-platform-config-design.md`: the `core/platforms/` auto-discovery swap (Part A), and a real, testable Go2-only ROS2 stack — `common/cerberus_description/`, `system/cerberus_bringup/`, and `policy/locomotion_policy/` — supporting both the blind (Flat) and perceptive (Rough) checkpoints (Part B, Go2 slice only; the `go2_arx5` combo stays deferred).

**Architecture:** Isaac Lab side gets a one-line discovery-walk swap. ROS2 side is three new colcon packages: `cerberus_description` (Go2's vendored URDF, ament_cmake, data-only), `cerberus_bringup` (`platforms.yaml` + launch file that resolves platform/mode into concrete params, ament_python), `locomotion_policy` (ONNX inference node, ament_python, shape-generic observation assembly driven by `platforms.yaml`'s `observation_shape` field).

**Tech Stack:** Python 3.12, ROS2 Jazzy (via `pixi run ros2`/`pixi run colcon`), `onnxruntime`, `pyyaml`, `pytest`, Isaac Lab's `isaaclab_tasks.utils.import_packages` (pod-only).

## Global Constraints

- Do not commit or push anything — leave all changes uncommitted in the working tree (explicit user instruction this session).
- No RunPod usage except the one short, already-approved pod session to export the Rough checkpoint's ONNX via `play.py`'s existing `export_policy_as_onnx`/`export_policy_as_jit` — terminate immediately after.
- Go2's URDF source: Unitree's own official `unitreerobotics/unitree_ros` repo, `robots/go2_description/` (BSD-3-Clause, clear root `LICENSE` file) — not the community `anujjain-dev/unitree-go2-ros2` fork (thinner license documentation, per this session's research). It's ROS1/catkin — must be converted to ament_cmake/package.xml format 3 when vendored, not used as-is.
- `platforms.yaml`'s schema, `locomotion_modes.<mode>.observation_shape` values (`proprioception_only` / `proprioception_plus_height_scan`), and the exact observation term order (`base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos_rel, joint_vel_rel, last_action[, height_scan]`) are fixed by the spec — do not redesign mid-implementation.
- `default_locomotion_mode: blind` is a deliberate safety default per the spec — never silently default to `perceptive`.
- The `perceptive` mode's `height_scan` input is a constant flat-terrain stub until `perception/terrain_perception/` exists — must log a loud `[WARN]` at node startup, not a quiet one.
- Action post-processing: `JointPositionAction`, `scale: 0.25`, `use_default_offset: true`, applied to all joints — network output is `action * 0.25 + default_joint_pos`, never published as a raw target.

---

## Task 1: Auto-discovery swap for `core/platforms/`

**Files:**
- Modify: `policy/locomotion/__init__.py`

**Interfaces:**
- Consumes: `policy.locomotion.core.platforms.go2.platform` (already exists, calls `register_platform(GO2)` at import time — no change needed there).
- Produces: nothing new — `available_platforms()` and the four Go2 gym ids must behave identically to before this change.

- [ ] **Step 1: Replace the file contents**

`policy/locomotion/__init__.py`:
```python
"""Cerberus locomotion task registration. Importing this module auto-discovers
and registers every platform package under core/platforms/ -- add a platform
by adding a new core/platforms/<name>/ directory (see
core/platforms/registry.py for what it must define), never by editing this
file.
"""

from isaaclab_tasks.utils import import_packages

# "registry" holds the shared PlatformCfg contract itself, not a platform --
# skip it so import_packages doesn't try to treat it as one.
_BLACKLIST_PKGS = ["registry"]

import_packages(__name__ + ".core.platforms", _BLACKLIST_PKGS)
```

- [ ] **Step 2: Verify syntax locally**

```bash
python3 -m py_compile policy/locomotion/__init__.py
```
Expected: no output (success).

- [ ] **Step 3: Document the pod-side check this can't run locally**

`isaaclab_tasks` isn't installed locally, so `import_packages` can't be
exercised here. Record as a pod-side follow-up (not executed in this task):
```bash
python -c "import policy.locomotion; import gymnasium as gym; \
  from policy.locomotion.core.platforms.registry import available_platforms; \
  print(available_platforms()); \
  print(sorted(k for k in gym.registry if 'Go2' in k))"
```
Expected: `['go2']` and the same four `Isaac-Velocity-*-Unitree-Go2-Cerberus-*` ids as before this change.

---

## Task 2: Export the Rough checkpoint's ONNX (short pod session)

**Files:** none (produces an artifact, not a code change)

**Interfaces:**
- Produces: `policy/locomotion/checkpoints/go2_locomotion_v1_candidate.onnx` (downloaded locally), used by Task 6.

This is the one approved short pod session. `play.py` already calls
`export_policy_as_onnx`/`export_policy_as_jit` unconditionally whenever it
runs — no code change needed, just an invocation against the candidate
checkpoint.

- [ ] **Step 1: Launch a short-lived pod** (per `docs/runpod.md`'s conventions: dedicated SSH key, driver smoke-test only if rendering — not needed here since this is headless inference/export, no `--video`)

- [ ] **Step 2: Sync code and the candidate checkpoint to the pod**

```bash
tar czf - --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    policy docs scripts .env | \
  ssh -i ~/.ssh/cerberus_runpod_ed25519 -p <port> root@<host> \
    "mkdir -p /workspace/cerberus && tar xzf - -C /workspace/cerberus"
```

- [ ] **Step 3: Run `play.py` against the candidate checkpoint to trigger export**

```bash
ssh -i ~/.ssh/cerberus_runpod_ed25519 -p <port> root@<host> \
  "cd /workspace/cerberus && timeout --signal=KILL --kill-after=60 10m \
   python policy/locomotion/play.py --terrain rough \
   --checkpoint policy/locomotion/checkpoints/go2_locomotion_v1_candidate.pt \
   --num_envs 1 --headless"
```
`play.py` exits after one launch since it runs `while simulation_app.is_running()` with no `--video` and no fixed step count — confirm it produces the export before it starts the render loop (it does: export happens before the `while` loop in the existing code), then kill the SSH session once `exported/policy.onnx` and `exported/policy.pt` appear next to the checkpoint (`ls` in a second SSH connection, don't block on the infinite loop).

- [ ] **Step 4: Download the exported files**

```bash
scp -i ~/.ssh/cerberus_runpod_ed25519 -P <port> \
  root@<host>:/workspace/cerberus/policy/locomotion/checkpoints/exported/policy.onnx \
  policy/locomotion/checkpoints/go2_locomotion_v1_candidate.onnx
```
(Adjust the remote path to wherever `play.py`'s `--checkpoint` flag placed
`exported/` — same directory as the checkpoint file, per `script_utils.py`'s
existing convention.)

- [ ] **Step 5: Terminate the pod immediately**

- [ ] **Step 6: Also download the already-exported Flat checkpoint from R2** (no pod needed, confirmed present this session)

```bash
set -a; source .env; set +a
export RCLONE_CONFIG_CERBERUS_R2_TYPE="s3"
export RCLONE_CONFIG_CERBERUS_R2_PROVIDER="Cloudflare"
export RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_CERBERUS_R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export RCLONE_CONFIG_CERBERUS_R2_ACL="private"
export RCLONE_CONFIG_CERBERUS_R2_NO_CHECK_BUCKET="true"
rclone copyto \
  "cerberus_r2:${R2_BUCKET}/locomotion/unitree_go2_flat/2026-09-03_04-24-43/exported/policy.onnx" \
  policy/locomotion/checkpoints/go2_locomotion_flat_v1_candidate.onnx
```

- [ ] **Step 7: Verify both files are present and non-empty**

```bash
ls -la policy/locomotion/checkpoints/*.onnx
```
Expected: both `go2_locomotion_v1_candidate.onnx` and
`go2_locomotion_flat_v1_candidate.onnx` present, non-zero size (~160-175KB
range, matching the sizes observed on R2 this session).

---

## Task 3: Vendor Go2's URDF into `common/cerberus_description/`

**Files:**
- Create: `common/cerberus_description/package.xml`
- Create: `common/cerberus_description/CMakeLists.txt`
- Create: `common/cerberus_description/urdf/go2.urdf.xacro` (+ supporting xacro files, + meshes, vendored from `unitreerobotics/unitree_ros`)

**Interfaces:**
- Produces: a colcon-buildable `cerberus_description` package installing `urdf/` and `meshes/` — consumed by Task 5's launch file via `find-pkg-share cerberus_description`.

- [ ] **Step 1: Fetch the source files from Unitree's official repo**

```bash
mkdir -p /tmp/unitree_ros_fetch
git clone --depth 1 https://github.com/unitreerobotics/unitree_ros.git /tmp/unitree_ros_fetch
ls /tmp/unitree_ros_fetch/robots/go2_description/
```
Expected: `xacro/`, `urdf/`, `meshes/`, `package.xml`, `CMakeLists.txt`, `LICENSE` (repo root).

- [ ] **Step 2: Copy the xacro and mesh files (not the ROS1 package.xml/CMakeLists.txt)**

```bash
mkdir -p common/cerberus_description/urdf common/cerberus_description/meshes
cp /tmp/unitree_ros_fetch/robots/go2_description/xacro/*.xacro common/cerberus_description/urdf/
cp /tmp/unitree_ros_fetch/robots/go2_description/meshes/*.dae common/cerberus_description/meshes/
mv common/cerberus_description/urdf/robot.xacro common/cerberus_description/urdf/go2.urdf.xacro
cp /tmp/unitree_ros_fetch/LICENSE common/cerberus_description/LICENSE
```

- [ ] **Step 3: Update mesh package references inside the copied xacro files**

The copied xacro files reference meshes via a `package://go2_description/meshes/...` URI (ROS1 catkin package name). Update to this package's actual ROS2 name:
```bash
sed -i 's#package://go2_description/#package://cerberus_description/#g' common/cerberus_description/urdf/*.xacro
```

- [ ] **Step 4: Write the ament_cmake `package.xml`**

`common/cerberus_description/package.xml`:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematron="http://download.ros.org/schema/package_format3.xsd"?>
<package format="3">
  <name>cerberus_description</name>
  <version>0.1.0</version>
  <description>Robot description (URDF/xacro/meshes) for Cerberus platforms. Go2's URDF is vendored from unitreerobotics/unitree_ros (BSD-3-Clause), converted from catkin to ament_cmake -- see LICENSE.</description>
  <maintainer email="k.thiha10.mail@gmail.com">Kevin Thiha</maintainer>
  <license>BSD-3-Clause</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <exec_depend>xacro</exec_depend>
  <exec_depend>robot_state_publisher</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

- [ ] **Step 5: Write the ament_cmake `CMakeLists.txt`**

`common/cerberus_description/CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.8)
project(cerberus_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
```

- [ ] **Step 6: Verify it builds**

```bash
pixi run colcon build --packages-select cerberus_description --symlink-install
```
Expected: `Summary: 1 package finished`.

- [ ] **Step 7: Verify xacro actually expands without errors**

```bash
set -a; . install/cerberus_description/share/cerberus_description/../../local_setup.bash 2>/dev/null; set +a
pixi run xacro common/cerberus_description/urdf/go2.urdf.xacro > /tmp/go2_expanded.urdf
xmllint --noout /tmp/go2_expanded.urdf
```
Expected: `xacro` produces output with no errors; `xmllint` reports no output
(valid XML). If `xacro` errors on an unresolved macro/package reference,
check whether the vendored file depends on another `unitree_ros` file not
yet copied (e.g. `velodyne.xacro`/`laser.xacro` reference sensor macros not
needed for a bare Go2 — safe to strip references to unused sensor xacro
files from `go2.urdf.xacro`'s includes if they cause errors and aren't
needed for this platform).

- [ ] **Step 8: Clean up the fetch scratch dir**

```bash
rm -rf /tmp/unitree_ros_fetch /tmp/go2_expanded.urdf
```

---

## Task 4: `platforms.yaml` + naming-contract test

**Files:**
- Create: `system/cerberus_bringup/config/platforms.yaml`
- Create: `system/cerberus_bringup/test/test_platforms_yaml.py`

**Interfaces:**
- Produces: the `platforms.yaml` schema exactly as specified — consumed by Task 5's `locomotion_policy` node (reads `locomotion_modes.<mode>.checkpoint`/`observation_shape`) and Task 6's `bringup.launch.py` (reads `urdf_relative_path`, `default_locomotion_mode`).

This task's test is pure Python/YAML — no ROS2 dependency, runs with plain `pytest` before the package even needs `colcon build`.

- [ ] **Step 1: Write the failing test**

`system/cerberus_bringup/test/test_platforms_yaml.py`:
```python
from pathlib import Path

import pytest
import yaml

PLATFORMS_YAML = Path(__file__).resolve().parents[1] / "config" / "platforms.yaml"

# Mirrors policy.locomotion.core.platforms.registry.available_platforms()'s
# expected output -- hardcoded here since isaaclab_tasks isn't importable in
# this test process. Extend this set when a new Python-side platform is added.
EXPECTED_PYTHON_PLATFORM_NAMES = {"go2"}


def _load_platforms() -> dict:
    with open(PLATFORMS_YAML) as f:
        return yaml.safe_load(f)


def test_platforms_yaml_exists_and_parses():
    platforms = _load_platforms()
    assert isinstance(platforms, dict)
    assert len(platforms) > 0


def test_every_yaml_key_matches_a_known_python_platform_name():
    platforms = _load_platforms()
    unknown = set(platforms) - EXPECTED_PYTHON_PLATFORM_NAMES
    assert not unknown, f"platforms.yaml has keys with no matching Python PlatformCfg.name: {unknown}"


def test_go2_entry_has_required_fields():
    go2 = _load_platforms()["go2"]
    assert "urdf_relative_path" in go2
    assert "locomotion_modes" in go2
    assert "default_locomotion_mode" in go2
    assert go2["default_locomotion_mode"] in go2["locomotion_modes"]


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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
mkdir -p system/cerberus_bringup/config
pixi run pytest system/cerberus_bringup/test/test_platforms_yaml.py -v
```
Expected: FAIL — `platforms.yaml` doesn't exist yet (`FileNotFoundError`).

- [ ] **Step 3: Write `platforms.yaml`**

`system/cerberus_bringup/config/platforms.yaml`:
```yaml
go2:
  urdf_relative_path: urdf/go2.urdf.xacro
  locomotion_modes:
    blind:
      checkpoint: policy/locomotion/checkpoints/go2_locomotion_flat_v1_candidate.onnx
      observation_shape: proprioception_only
    perceptive:
      checkpoint: policy/locomotion/checkpoints/go2_locomotion_v1_candidate.onnx
      observation_shape: proprioception_plus_height_scan
  default_locomotion_mode: blind
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pixi run pytest system/cerberus_bringup/test/test_platforms_yaml.py -v
```
Expected: PASS (8 tests).

---

## Task 5: `locomotion_policy` ROS2 node

**Files:**
- Create: `policy/locomotion_policy/package.xml`
- Create: `policy/locomotion_policy/setup.py`
- Create: `policy/locomotion_policy/setup.cfg`
- Create: `policy/locomotion_policy/locomotion_policy/__init__.py`
- Create: `policy/locomotion_policy/locomotion_policy/observation_assembly.py`
- Create: `policy/locomotion_policy/locomotion_policy/inference_node.py`
- Test: `policy/locomotion_policy/test/test_observation_assembly.py`

**Interfaces:**
- Consumes: `platforms.yaml`'s `locomotion_modes.<mode>.observation_shape` values (Task 4) as the two string literals below.
- Produces: `assemble_observation(shape: str, proprio: ProprioSample, height_scan: np.ndarray | None) -> np.ndarray` — pure function, the piece that's actually unit-testable without ROS2 or a real robot. `InferenceNode` (rclpy node) wraps it with topic I/O — not unit-tested here (needs a live ROS graph), verified manually in Task 7.

`observation_assembly.py` is where the exact term order and action
post-processing live — kept separate from the rclpy node so it's testable
with plain `pytest`, no ROS2 runtime needed.

- [ ] **Step 1: Write the failing test**

`policy/locomotion_policy/test/test_observation_assembly.py`:
```python
import numpy as np
import pytest

from locomotion_policy.observation_assembly import ProprioSample, assemble_observation


def _make_proprio(num_joints: int = 12) -> ProprioSample:
    return ProprioSample(
        base_lin_vel=np.array([0.1, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.array([0.0, 0.0, 0.1], dtype=np.float32),
        projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        velocity_commands=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        joint_pos_rel=np.zeros(num_joints, dtype=np.float32),
        joint_vel_rel=np.zeros(num_joints, dtype=np.float32),
        last_action=np.zeros(num_joints, dtype=np.float32),
    )


def test_blind_shape_has_no_height_scan_and_correct_length():
    proprio = _make_proprio(num_joints=12)
    obs = assemble_observation("proprioception_only", proprio, height_scan=None)
    # 3+3+3+3+12+12+12 = 48
    assert obs.shape == (48,)
    assert obs.dtype == np.float32


def test_blind_shape_term_order():
    proprio = _make_proprio(num_joints=12)
    obs = assemble_observation("proprioception_only", proprio, height_scan=None)
    np.testing.assert_array_equal(obs[0:3], proprio.base_lin_vel)
    np.testing.assert_array_equal(obs[3:6], proprio.base_ang_vel)
    np.testing.assert_array_equal(obs[6:9], proprio.projected_gravity)
    np.testing.assert_array_equal(obs[9:12], proprio.velocity_commands)
    np.testing.assert_array_equal(obs[12:24], proprio.joint_pos_rel)
    np.testing.assert_array_equal(obs[24:36], proprio.joint_vel_rel)
    np.testing.assert_array_equal(obs[36:48], proprio.last_action)


def test_perceptive_shape_appends_height_scan_last():
    proprio = _make_proprio(num_joints=12)
    height_scan = np.full(187, 0.3, dtype=np.float32)  # arbitrary grid size for this test
    obs = assemble_observation("proprioception_plus_height_scan", proprio, height_scan=height_scan)
    assert obs.shape == (48 + 187,)
    np.testing.assert_array_equal(obs[48:], height_scan)


def test_perceptive_shape_without_height_scan_raises():
    proprio = _make_proprio(num_joints=12)
    with pytest.raises(ValueError, match="height_scan"):
        assemble_observation("proprioception_plus_height_scan", proprio, height_scan=None)


def test_unknown_shape_raises():
    proprio = _make_proprio(num_joints=12)
    with pytest.raises(ValueError, match="unknown observation_shape"):
        assemble_observation("not_a_real_shape", proprio, height_scan=None)


def test_apply_action_scale_and_offset():
    from locomotion_policy.observation_assembly import apply_action_postprocessing

    raw_network_output = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    default_joint_pos = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    result = apply_action_postprocessing(raw_network_output, default_joint_pos, scale=0.25)
    expected = raw_network_output * 0.25 + default_joint_pos
    np.testing.assert_allclose(result, expected)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
mkdir -p policy/locomotion_policy/locomotion_policy policy/locomotion_policy/test
touch policy/locomotion_policy/locomotion_policy/__init__.py
cd policy/locomotion_policy && pixi run pytest test/test_observation_assembly.py -v; cd -
```
Expected: FAIL — `locomotion_policy.observation_assembly` doesn't exist (`ModuleNotFoundError`).

- [ ] **Step 3: Write `observation_assembly.py`**

`policy/locomotion_policy/locomotion_policy/observation_assembly.py`:
```python
"""Pure observation-assembly and action-postprocessing logic for the
locomotion_policy inference node -- kept separate from the rclpy node so
it's unit-testable without a live ROS graph. Term order and action
post-processing come from the actual training run's params/env.yaml
(confirmed 2026-09-08 for the Rough candidate) -- do not reorder without
re-checking that file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SHAPE_BLIND = "proprioception_only"
SHAPE_PERCEPTIVE = "proprioception_plus_height_scan"


@dataclass
class ProprioSample:
    base_lin_vel: np.ndarray
    base_ang_vel: np.ndarray
    projected_gravity: np.ndarray
    velocity_commands: np.ndarray
    joint_pos_rel: np.ndarray
    joint_vel_rel: np.ndarray
    last_action: np.ndarray


def assemble_observation(
    observation_shape: str,
    proprio: ProprioSample,
    height_scan: np.ndarray | None,
) -> np.ndarray:
    """Concatenates observation terms in training's exact order:
    base_lin_vel, base_ang_vel, projected_gravity, velocity_commands,
    joint_pos_rel, joint_vel_rel, last_action[, height_scan].
    """
    base_terms = np.concatenate(
        [
            proprio.base_lin_vel,
            proprio.base_ang_vel,
            proprio.projected_gravity,
            proprio.velocity_commands,
            proprio.joint_pos_rel,
            proprio.joint_vel_rel,
            proprio.last_action,
        ]
    ).astype(np.float32)

    if observation_shape == SHAPE_BLIND:
        return base_terms
    elif observation_shape == SHAPE_PERCEPTIVE:
        if height_scan is None:
            raise ValueError(f"observation_shape={observation_shape!r} requires a height_scan array, got None")
        return np.concatenate([base_terms, height_scan.astype(np.float32)])
    else:
        raise ValueError(f"unknown observation_shape {observation_shape!r}, expected {SHAPE_BLIND!r} or {SHAPE_PERCEPTIVE!r}")


def apply_action_postprocessing(
    raw_network_output: np.ndarray,
    default_joint_pos: np.ndarray,
    scale: float = 0.25,
) -> np.ndarray:
    """JointPositionAction post-processing: scale the network's raw output
    and add each joint's default position, matching training's
    use_default_offset=true. Never publish raw_network_output directly."""
    return raw_network_output * scale + default_joint_pos
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd policy/locomotion_policy && pixi run pytest test/test_observation_assembly.py -v; cd -
```
Expected: PASS (7 tests).

- [ ] **Step 5: Write the rclpy inference node**

`policy/locomotion_policy/locomotion_policy/inference_node.py`:
```python
"""ROS2 inference node: loads an exported ONNX locomotion policy, assembles
observations from live topics, publishes joint position commands. Which
checkpoint/observation_shape to use comes from launch-time parameters
(checkpoint_path, observation_shape, default_joint_pos), resolved by
cerberus_bringup's launch file from platforms.yaml -- this node never reads
platforms.yaml itself.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

from locomotion_policy.observation_assembly import (
    SHAPE_PERCEPTIVE,
    ProprioSample,
    apply_action_postprocessing,
    assemble_observation,
)


class InferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("locomotion_policy")

        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("observation_shape", "")
        self.declare_parameter("control_rate_hz", 50.0)

        checkpoint_path = self.get_parameter("checkpoint_path").value
        self.observation_shape = self.get_parameter("observation_shape").value
        if not checkpoint_path or not self.observation_shape:
            raise RuntimeError("checkpoint_path and observation_shape parameters are required")

        if self.observation_shape == SHAPE_PERCEPTIVE:
            self.get_logger().warn(
                "height_scan stubbed flat -- perceptive mode is running in bring-up/testing "
                "configuration, not a validated real-sensor deployment. See "
                "docs/locomotion_architecture.md."
            )

        self.session = ort.InferenceSession(checkpoint_path)
        self._input_name = self.session.get_inputs()[0].name

        self._latest_joint_state: JointState | None = None
        self._latest_cmd_vel = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(12, dtype=np.float32)
        self._default_joint_pos = np.zeros(12, dtype=np.float32)

        self.create_subscription(JointState, "joint_states", self._on_joint_state, 10)
        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, 10)
        self._action_pub = self.create_publisher(Float32MultiArray, "joint_position_command", 10)

        control_rate = self.get_parameter("control_rate_hz").value
        self.create_timer(1.0 / control_rate, self._step)

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._latest_cmd_vel = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32)

    def _height_scan_stub(self) -> np.ndarray:
        # Constant flat-terrain value -- see the startup warning above and
        # docs/locomotion_architecture.md for why this is a bring-up stub,
        # not a real sensor reading.
        return np.zeros(187, dtype=np.float32)

    def _step(self) -> None:
        if self._latest_joint_state is None:
            return  # no joint data yet, skip this tick

        joint_pos = np.array(self._latest_joint_state.position, dtype=np.float32)
        joint_vel = np.array(self._latest_joint_state.velocity, dtype=np.float32)

        proprio = ProprioSample(
            base_lin_vel=np.zeros(3, dtype=np.float32),  # TODO(hardware): wire to a real IMU/odometry topic
            base_ang_vel=np.zeros(3, dtype=np.float32),  # TODO(hardware): wire to a real IMU topic
            projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),  # TODO(hardware): wire to IMU orientation
            velocity_commands=self._latest_cmd_vel,
            joint_pos_rel=joint_pos - self._default_joint_pos,
            joint_vel_rel=joint_vel,
            last_action=self._last_action,
        )
        height_scan = self._height_scan_stub() if self.observation_shape == SHAPE_PERCEPTIVE else None
        obs = assemble_observation(self.observation_shape, proprio, height_scan)

        raw_output = self.session.run(None, {self._input_name: obs.reshape(1, -1)})[0].reshape(-1)
        self._last_action = raw_output
        joint_targets = apply_action_postprocessing(raw_output, self._default_joint_pos, scale=0.25)

        msg = Float32MultiArray()
        msg.data = joint_targets.tolist()
        self._action_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

Note: `base_lin_vel`/`base_ang_vel`/`projected_gravity` are stubbed with
`TODO(hardware)` markers rather than fabricated real logic — this repo has
no IMU driver/odometry source wired up yet, and guessing at one here would
be worse than an honest gap. `default_joint_pos` is similarly a zero
placeholder pending Go2's actual default joint angles (available once
Task 3's URDF is fully parsed — a follow-up, not blocking this task's
observation-assembly logic, which is what's actually tested).

- [ ] **Step 6: Write `package.xml`**

`policy/locomotion_policy/package.xml`:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematron="http://download.ros.org/schema/package_format3.xsd"?>
<package format="3">
  <name>locomotion_policy</name>
  <version>0.1.0</version>
  <description>ROS2 inference node running a trained Cerberus locomotion policy from an exported ONNX checkpoint.</description>
  <maintainer email="k.thiha10.mail@gmail.com">Kevin Thiha</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_python</buildtool_depend>

  <depend>rclpy</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>std_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

- [ ] **Step 7: Write `setup.py` and `setup.cfg`**

`policy/locomotion_policy/setup.py`:
```python
from setuptools import find_packages, setup

package_name = "locomotion_policy"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kevin Thiha",
    maintainer_email="k.thiha10.mail@gmail.com",
    description="ROS2 inference node for a trained Cerberus locomotion policy.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "inference_node = locomotion_policy.inference_node:main",
        ],
    },
)
```

`policy/locomotion_policy/setup.cfg`:
```ini
[develop]
script_dir=$base/lib/locomotion_policy
[install]
install_scripts=$base/lib/locomotion_policy
```

- [ ] **Step 8: Add the ament resource marker**

```bash
mkdir -p policy/locomotion_policy/resource
touch policy/locomotion_policy/resource/locomotion_policy
```

- [ ] **Step 9: Verify the package builds**

```bash
pixi run colcon build --packages-select locomotion_policy --symlink-install
```
Expected: `Summary: 1 package finished`.

---

## Task 6: `cerberus_bringup` package + launch file

**Files:**
- Create: `system/cerberus_bringup/package.xml`
- Create: `system/cerberus_bringup/setup.py`
- Create: `system/cerberus_bringup/setup.cfg`
- Create: `system/cerberus_bringup/resource/cerberus_bringup`
- Create: `system/cerberus_bringup/launch/bringup.launch.py`
- Modify: `system/cerberus_bringup/test/test_platforms_yaml.py` (moves from Task 4's scratch location into the real package's `test/` dir — same file, no content change, since Task 4 already created it directly at this path)

**Interfaces:**
- Consumes: `platforms.yaml` (Task 4), `cerberus_description` (Task 3, via `find-pkg-share`), `locomotion_policy`'s `inference_node` entry point (Task 5).

- [ ] **Step 1: Write `package.xml`**

`system/cerberus_bringup/package.xml`:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematron="http://download.ros.org/schema/package_format3.xsd"?>
<package format="3">
  <name>cerberus_bringup</name>
  <version>0.1.0</version>
  <description>Launch files composing the full Cerberus graph. Owns platform/locomotion-mode selection (platforms.yaml) -- see docs/superpowers/specs/2026-09-09-multi-platform-config-design.md.</description>
  <maintainer email="k.thiha10.mail@gmail.com">Kevin Thiha</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_python</buildtool_depend>

  <exec_depend>cerberus_description</exec_depend>
  <exec_depend>locomotion_policy</exec_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>xacro</exec_depend>

  <test_depend>python3-pytest</test_depend>
  <test_depend>python3-yaml</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

- [ ] **Step 2: Write `setup.py`/`setup.cfg`**

`system/cerberus_bringup/setup.py`:
```python
import os
from glob import glob

from setuptools import find_packages, setup

package_name = "cerberus_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "pyyaml"],
    zip_safe=True,
    maintainer="Kevin Thiha",
    maintainer_email="k.thiha10.mail@gmail.com",
    description="Launch files composing the full Cerberus ROS2 graph.",
    license="MIT",
    tests_require=["pytest"],
)
```

`system/cerberus_bringup/setup.cfg`:
```ini
[develop]
script_dir=$base/lib/cerberus_bringup
[install]
install_scripts=$base/lib/cerberus_bringup
```

```bash
mkdir -p system/cerberus_bringup/resource
touch system/cerberus_bringup/resource/cerberus_bringup
```

- [ ] **Step 3: Write the launch file**

`system/cerberus_bringup/launch/bringup.launch.py`:
```python
"""Composes the full Cerberus ROS2 graph for a given --platform/--mode.
Resolves platforms.yaml (config/platforms.yaml) into concrete params for
robot_state_publisher and locomotion_policy -- see
docs/superpowers/specs/2026-09-09-multi-platform-config-design.md for the
schema and reasoning. Only this launch file, locomotion_policy, and (once
built) manipulation_policy are platform-aware; every other package gets
robot identity from the standard /robot_description parameter.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def _load_platforms_yaml() -> dict:
    config_path = os.path.join(get_package_share_directory("cerberus_bringup"), "config", "platforms.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _launch_setup(context, *args, **kwargs):
    platform_name = LaunchConfiguration("platform").perform(context)
    platforms = _load_platforms_yaml()
    if platform_name not in platforms:
        raise RuntimeError(f"unknown platform {platform_name!r}, known: {sorted(platforms)}")
    platform = platforms[platform_name]

    mode_name = LaunchConfiguration("locomotion_mode").perform(context)
    if not mode_name:
        mode_name = platform["default_locomotion_mode"]
    if mode_name not in platform["locomotion_modes"]:
        raise RuntimeError(
            f"unknown locomotion_mode {mode_name!r} for platform {platform_name!r}, "
            f"known: {sorted(platform['locomotion_modes'])}"
        )
    mode = platform["locomotion_modes"][mode_name]

    description_share = get_package_share_directory("cerberus_description")
    xacro_path = os.path.join(description_share, platform["urdf_relative_path"])

    repo_root = os.path.abspath(os.path.join(get_package_share_directory("cerberus_bringup"), "..", "..", "..", ".."))
    checkpoint_path = os.path.join(repo_root, mode["checkpoint"])

    nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": Command(["xacro ", xacro_path])}],
        ),
        Node(
            package="locomotion_policy",
            executable="inference_node",
            name="locomotion_policy",
            output="screen",
            parameters=[
                {"checkpoint_path": checkpoint_path},
                {"observation_shape": mode["observation_shape"]},
            ],
        ),
    ]
    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("platform", default_value="go2", description="Robot platform (see config/platforms.yaml)."),
            DeclareLaunchArgument(
                "locomotion_mode",
                default_value="",
                description="blind or perceptive -- defaults to the platform's own default_locomotion_mode if empty.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
```

- [ ] **Step 4: Move Task 4's config/test into this package** (they were already created at these exact paths in Task 4, so this step is verification, not a move)

```bash
ls system/cerberus_bringup/config/platforms.yaml system/cerberus_bringup/test/test_platforms_yaml.py
```
Expected: both present (created in Task 4).

- [ ] **Step 5: Build all three packages together**

```bash
pixi run colcon build --packages-select cerberus_description locomotion_policy cerberus_bringup --symlink-install
```
Expected: `Summary: 3 packages finished`.

---

## Task 7: End-to-end local verification

**Files:** none (verification only)

- [ ] **Step 1: Run every test written in this plan**

```bash
source install/setup.bash
pixi run pytest system/cerberus_bringup/test/test_platforms_yaml.py policy/locomotion_policy/test/test_observation_assembly.py -v
```
Expected: all 15 tests pass (8 from Task 4 + 7 from Task 5).

- [ ] **Step 2: Launch with the blind mode (default) and confirm the graph comes up**

```bash
pixi run ros2 launch cerberus_bringup bringup.launch.py
```
In a second terminal:
```bash
source install/setup.bash
pixi run ros2 topic echo /robot_description --once
pixi run ros2 node list
```
Expected: `/robot_description` publishes a non-empty URDF string;
`ros2 node list` shows `/robot_state_publisher` and `/locomotion_policy`.

- [ ] **Step 3: Confirm `locomotion_policy` loaded the correct (blind) checkpoint**

Check the node's log output from Step 2 for the ONNX session loading
successfully and no `[WARN] height_scan stubbed flat` message (that warning
should only appear in perceptive mode).

- [ ] **Step 4: Launch with perceptive mode and confirm the warning appears**

```bash
pixi run ros2 launch cerberus_bringup bringup.launch.py locomotion_mode:=perceptive
```
Expected: `[WARN] [locomotion_policy]: height_scan stubbed flat -- perceptive mode is running in bring-up/testing configuration...` appears in the log.

- [ ] **Step 5: Publish a synthetic joint_states message and confirm inference runs without error**

```bash
pixi run ros2 topic pub /joint_states sensor_msgs/msg/JointState \
  "{name: [j0,j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11], position: [0,0,0,0,0,0,0,0,0,0,0,0], velocity: [0,0,0,0,0,0,0,0,0,0,0,0]}" \
  --once
pixi run ros2 topic echo /joint_position_command --once
```
Expected: `/joint_position_command` publishes a 12-element float array with
no exceptions in the node's log.

- [ ] **Step 6: Stop all launched processes**

```bash
# Ctrl-C both terminals from Steps 2 and 4
```

---

## Self-Review Notes

- **Spec coverage**: Part A (Task 1), checkpoint status/export (Task 2), `cerberus_description` (Task 3), `platforms.yaml` schema + naming contract (Task 4), `locomotion_policy` node design including both shapes and the flat-stub warning (Task 5), `cerberus_bringup` launch flow (Task 6), end-to-end check (Task 7) — all covered. `go2_arx5` combo intentionally has no task, per the spec's explicit deferral.
- **Known, disclosed gap**: `inference_node.py`'s `base_lin_vel`/`base_ang_vel`/`projected_gravity` and `default_joint_pos` are stubbed with `TODO(hardware)` markers, not fabricated real sensor logic — no IMU/odometry source exists in this repo yet, and this plan's actual tested deliverable is the observation-assembly logic (Task 5's unit tests) and the launch/config plumbing (Tasks 4, 6, 7's manual verification), not real-hardware control. Flagged here and in the node's own code comments, not hidden.
- **Type/signature consistency**: `assemble_observation(observation_shape: str, proprio: ProprioSample, height_scan: np.ndarray | None) -> np.ndarray` is defined once in Task 5 Step 3 and used identically in Task 5 Step 5's node and nowhere else redefined. `platforms.yaml`'s `locomotion_modes.<mode>.observation_shape` values (`proprioception_only`/`proprioception_plus_height_scan`) match `SHAPE_BLIND`/`SHAPE_PERCEPTIVE`'s literal string values exactly across Tasks 4 and 5.
- **No placeholders**: every code block is complete, runnable code; the one intentional gap (real IMU wiring) is explicitly disclosed rather than silently stubbed.
