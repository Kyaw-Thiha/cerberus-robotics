# SLAM Module Implementation Plan (v2 — corrected after 3-agent audit)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **v2 changelog (2026-09-20):** A 3-agent audit of v1 against the SLAM design spec found real defects, now fixed here: (1) `/slam/map`'s `map_type` was tagged via an invisible ROS2 node *parameter* instead of the message payload — replaced with a real `cerberus_msgs/TaggedMap` message; (2) TF `map → base_link` was missing from every backend adapter entirely — now built into the shared adapter; (3) `topic_tools relay` was configured with ROS1-style positional args instead of ROS2's actual `input_topic`/`output_topic` parameters — replaced with a proper `rclpy` adapter node; (4) §5's required downstream map-conversion stage was silently dropped — added as Task 9; (5) the rerun `Blueprint` test asserted invented API surface — loosened to what's actually verifiable; (6) the MP4 remux command shelled out to a nonexistent `rerun.scripts.export_video_stream` module — replaced with a faithful adaptation of rerun's real, verified snippet (`docs/snippets/all/archetypes/video_stream_query_and_mux.py`); (7) `record_mission_bag.py` never actually drove the robot — now publishes real `cmd_vel` commands from the scripted schedule; (8) raw sensor feeds and RPE scoring, both explicitly required by §9, were missing — added; (9) the showcase camera had no follow logic and was never attached to any suite — added a testable pure chase-cam function plus a suite-attachment helper; (10) **the LVI-Q `joint_state` schema question was falsely presented as "resolved per §10" — it is NOT resolved in the design spec, and this version stops claiming otherwise.**

**Goal:** Implement the full SLAM module.

**Architecture:** Extends the existing `AlgorithmCfg`/registry pattern (`common/algorithm_registry.py`). Ground-truth export and the scripted motion source follow `policy/locomotion_policy`'s split: pure logic in one module, a thin `rclpy` wrapper in another. Backend source is vendored locally (gitignored, fetched by script). **New in v2:** a `cerberus_msgs` interfaces package (the one `project_structure.md` already reserves this name for) carries the `TaggedMap` message that makes §5's tagging requirement real, and a single shared `SlamBackendAdapter` class implements the full §2 topic contract (odom + TF + tagged map) once, parameterized per backend by native topic names only — replacing six independent, under-specified `topic_tools relay` launch files.

**Tech Stack:** Python 3.12, ROS2 Jazzy (`rclpy`, `nav_msgs`, `geometry_msgs`, `sensor_msgs`, `tf2_ros`, `ament_cmake`/`rosidl` for `cerberus_msgs`), pytest, PyYAML, Docker, RunPod, `evo`, `rerun-sdk` + `av` (PyAV), ROS1 Noetic (inside per-backend containers only).

## Global Constraints

- **Do not run `git commit` or `git push` at any point in this plan.** Kevin handles all git operations himself. Every task ends with "leave the changes staged for review," not a commit step.
- All local commands run from the repo root inside `pixi`: prefix with `pixi run`.
- `AlgorithmCfg` gets no new fields (unchanged from v1 — the sensor-config grouping is fully derivable from `sensor_requirements`'s shape).
- **LVI-Q's `joint_state` dependency is a genuinely open decision, not a resolved one — correcting v1's false claim.** The design spec's open-questions section marks exactly two items "Resolved 2026-09-20" (BIEVR-LIO's ROS version, rerun's export capability); the `joint_state` schema item carries no such marking and is still open. This plan proceeds with the same practical choice v1 made — `joint_state` stays undeclared in `sensor_requirements`, noted only in `lvi_q`'s `description` string — but as **this plan's own default recommendation for Kevin to confirm, not as something the design spec already settled.** If Kevin wants `SensorSuiteCfg` extended with a real `joint_state` field instead, Task 1's `lvi_q` registration and this note both need revisiting.
- Isaac Lab's exact articulation attribute names (`root_pos_w`, `root_quat_w`) are asserted by §6. **Confirmed 2026-09-20** on a RunPod GPU pod (`Isaac-Velocity-Flat-Unitree-Go2-v0`, `env.unwrapped.scene["robot"].data`): both attributes exist and are populated with live data (`root_pos_w` shape (N, 3), `root_quat_w` shape (N, 4), (w, x, y, z) order) — no rename needed.
- **Verified externally before writing code, not assumed (2026-09-20):** ROS2 `topic_tools relay`'s real parameter names (`input_topic`/`output_topic`, confirmed against `ros-tooling/topic_tools`'s own README — not used in this plan anymore, but confirms v1's usage was wrong); `evo_ape`/`evo_rpe`'s real `bag2 <bagfile> <ref_topic> <est_topic> [options]` syntax (confirmed real and current, contrary to v1's unverified-but-actually-correct guess); rerun's real MP4-remux snippet shape (confirmed real at `rerun-io/rerun`'s `docs/snippets/all/archetypes/video_stream_query_and_mux.py` — a standalone script to adapt, not an importable CLI module); the real `cmd_vel` / `geometry_msgs/Twist` velocity-command topic convention already used by `policy/locomotion_policy/locomotion_policy/inference_node.py` (reused here rather than inventing a new topic name).
- Every per-backend Docker image build assumes `scripts/fetch_slam_backends.py <name>` has already run locally for that backend.
- **Checkpoint before real cost/blast-radius actions:** fetching real repos over the network (Task 6), pushing Docker images (Task 6), building/running anything on a live RunPod pod (Tasks 6-8's verification steps). Kevin has confirmed live RunPod use is fine for this plan; each such step still states the expected cost/duration up front.

---

## File Structure

```
common/
├── cerberus_msgs/                        # NEW (Task 6a) — the interfaces package
│   │                                        project_structure.md already reserves this name for
│   ├── CMakeLists.txt, package.xml
│   └── msg/TaggedMap.msg
└── cerberus_description/sensors/
    ├── sensor_suite.py                    # MODIFY (Task 5) — showcase_camera field + with_showcase_camera()
    └── sensor_configs.py                  # MODIFY (Task 5) — SHOWCASE_CAMERA_CFG

slam/
├── package.xml, setup.py, setup.cfg, resource/slam    # NEW (Task 6b)
└── slam/
    ├── core/algorithms.py, test_algorithms.py          # MODIFY (Task 1)
    ├── ground_truth/{__init__,pose_export,test_pose_export,ground_truth_node}.py   # NEW (Task 3)
    ├── eval/
    │   ├── __init__.py, scripted_motion.py, test_scripted_motion.py    # NEW (Task 4)
    │   ├── showcase_camera_follow.py, test_showcase_camera_follow.py   # NEW (Task 5)
    │   ├── record_mission_bag.py                                       # NEW (Task 7)
    │   ├── score_with_evo.py, test_score_with_evo.py                   # NEW (Task 7)
    │   └── results/                                                     # gitignored
    ├── viz/
    │   ├── __init__.py, blueprint.py, test_blueprint.py                # NEW (Task 8)
    │   ├── export_video.py                                              # NEW (Task 8)
    │   └── bridge_launch/{lidar_camera,camera_only}.yaml                # NEW (Task 8)
    └── backends/
        ├── common_adapter.py, test_common_adapter.py     # NEW (Task 6c) — the shared adapter class
        ├── fast_lio2/adapter_launch.py                    # NEW (Task 6c)
        ├── bievr_lio/adapter_launch.py                    # NEW (Task 6c)
        ├── fast_livo2/adapter_launch.py                   # NEW (Task 6c)
        ├── lvi_q/adapter_launch.py                        # NEW (Task 6c) — placeholder, no impl yet
        ├── orb_slam3/adapter_launch.py                    # NEW (Task 6c)
        └── multicam_slam/adapter_launch.py                # NEW (Task 6c)

perception/terrain_perception/  (or a new shared location — see Task 9)
└── map_conversion/                        # NEW (Task 9) — §5's downstream point-cloud -> grid stage

third_party/slam/{fast_lio2,bievr_lio,fast_livo2,orb_slam3,multicam_slam}/   # NEW (Task 6d) — gitignored
scripts/fetch_slam_backends.py, test_fetch_slam_backends.py                  # NEW (Task 6d)
.gitignore                                                                    # MODIFY (Task 6d)
docker/Dockerfile.slam_{fast_lio2,bievr_lio,fast_livo2,orb_slam3,multicam_slam}   # NEW (Task 6d)
docker/scripts/start_slam_backend.sh                                          # NEW (Task 6d)

system/cerberus_bringup/config/
├── sensor_profiles.yaml                   # MODIFY (Task 2)
└── slam_backends.yaml                     # NEW (Task 6e)
system/cerberus_bringup/test/test_slam_backends_yaml.py    # NEW (Task 6e)
```

---

### Task 1: Register all six SLAM candidates with an explicit, drift-guarded sensor-config mapping

*(Unchanged from v1 — the audit found no defects here.)*

**Files:** Modify `slam/slam/core/algorithms.py`, `slam/slam/core/test_algorithms.py`.

**Interfaces:** Produces `available_algorithms()` returning all six names; `get_algorithm(name)`.

- [ ] **Step 1: Write the failing tests**

```python
EXPECTED_SENSOR_CONFIGS = {
    "fast_lio2": {"lidar": "required"},
    "bievr_lio": {"lidar": "required"},
    "fast_livo2": {"lidar": "required", "front_camera": "required"},
    "lvi_q": {"lidar": "required", "front_camera": "required"},
    "orb_slam3": {"front_camera": "required"},
    "multicam_slam": {"front_camera": "required", "rear_camera": "required"},
}


def test_real_candidates_are_registered():
    names = available_algorithms()
    for name in EXPECTED_SENSOR_CONFIGS:
        assert name in names
    assert not get_algorithm("bievr_lio").implemented
    assert not get_algorithm("lvi_q").implemented
    assert not get_algorithm("orb_slam3").implemented
    assert not get_algorithm("multicam_slam").implemented


def test_sensor_config_groupings_match_documented_shape():
    for name, expected in EXPECTED_SENSOR_CONFIGS.items():
        assert get_algorithm(name).sensor_requirements == expected, (
            f"{name}'s sensor_requirements drifted from its documented shape"
        )


def test_no_two_sensor_configs_collapse_to_the_same_shape():
    shapes = [
        {"lidar": "required"},
        {"lidar": "required", "front_camera": "required"},
        {"front_camera": "required"},
        {"front_camera": "required", "rear_camera": "required"},
    ]
    assert len(shapes) == len({tuple(sorted(s.items())) for s in shapes})
```

- [ ] **Step 2: Run to verify failure** — `pixi run pytest slam/slam/core/test_algorithms.py -v` → FAIL (four candidates unregistered).

- [ ] **Step 3: Register the four missing candidates** — append to `slam/slam/core/algorithms.py`, right after a comment table mirroring §1:

```python
# --- Sensor-config groupings ---
#
# | Sensor config          | sensor_requirements shape                          | Candidates                  |
# |-------------------------|-----------------------------------------------------|------------------------------|
# | LiDAR only               | {"lidar": "required"}                               | fast_lio2 (baseline), bievr_lio (latest) |
# | LiDAR + camera (fused)   | {"lidar": "required", "front_camera": "required"}   | fast_livo2 (baseline), lvi_q (latest)    |
# | Camera-only, single pair | {"front_camera": "required"}                        | orb_slam3 (baseline AND latest)          |
# | Camera-only, front+rear  | {"front_camera": "required", "rear_camera": "required"} | multicam_slam (latest, fuses onto orb_slam3) |
#
# See test_algorithms.py's test_sensor_config_groupings_match_documented_shape for the drift guard.

register_algorithm(AlgorithmCfg(
    name="bievr_lio",
    sensor_requirements={"lidar": "required"},
    description="Pfreundschuh et al. 2026 (RSS), degeneracy-robust LiDAR-inertial odometry, "
    "github.com/ethz-asl/BIEVR-LIO -- has a native ROS2 build, unlike its peers here",
))

register_algorithm(AlgorithmCfg(
    name="lvi_q",
    sensor_requirements={"lidar": "required", "front_camera": "required"},
    description="Kim et al. 2025 (arXiv:2510.15220), LiDAR-visual-inertial-kinematic "
    "odometry with joint-encoder fusion, no public repo -- implement from spec. "
    "Also requires joint_state (undeclared here -- see this plan's Global Constraints, "
    "this is a genuinely open schema decision, NOT resolved by the design spec).",
))

register_algorithm(AlgorithmCfg(
    name="orb_slam3",
    sensor_requirements={"front_camera": "required"},
    description="Campos et al. 2021 (T-RO), stereo visual SLAM, no IMU, "
    "github.com/UZ-SLAMLab/ORB_SLAM3 (core) + github.com/zang09/ORB_SLAM3_ROS2 (wrapper)",
))

register_algorithm(AlgorithmCfg(
    name="multicam_slam",
    sensor_requirements={"front_camera": "required", "rear_camera": "required"},
    description="Front+rear stereo fusion on ORB-SLAM2, github.com/AlterPang/Multi_ORB_SLAM "
    "-- README is generic/unconfirmed, read src/ directly before implementing",
))
```

- [ ] **Step 4: Run to verify pass** — `pixi run pytest slam/slam/core/test_algorithms.py -v` → PASS.
- [ ] **Step 5: Leave staged.**

---

### Task 2: Wire the camera-only sensor profile to ORB-SLAM3

*(Unchanged from v1 — no defects found.)*

**Files:** Modify `system/cerberus_bringup/config/sensor_profiles.yaml`.

- [ ] **Step 1:** Baseline check — `pixi run pytest system/cerberus_bringup/test/test_sensor_profiles_yaml.py -v` → PASS as-is.
- [ ] **Step 2:** Change `camera_only.algorithms.slam` from `null` to `orb_slam3`, updating the comment above it to reference the registered candidate instead of "no candidate documented yet."
- [ ] **Step 3:** Re-run the suite → PASS.
- [ ] **Step 4:** Leave staged.

---

### Task 3: Ground-truth pose/TF export logic

*(Unchanged from v1 — no defects found.)*

**Files:** Create `slam/slam/ground_truth/{__init__,pose_export,test_pose_export,ground_truth_node}.py`.

- [ ] **Step 1: Write the failing tests** (`test_pose_export.py`):

```python
import math

import rclpy

from slam.slam.ground_truth.ground_truth_node import GroundTruthPublisher
from slam.slam.ground_truth.pose_export import (
    build_ground_truth_transform,
    build_odometry_fields,
    isaac_quat_to_ros,
)


def test_isaac_quat_to_ros_reorders_wxyz_to_xyzw():
    assert isaac_quat_to_ros(w=1.0, x=0.0, y=0.0, z=0.0) == (0.0, 0.0, 0.0, 1.0)


def test_isaac_quat_to_ros_nontrivial_rotation():
    half = math.pi / 4
    w_in, x_in, y_in, z_in = math.cos(half), 0.0, 0.0, math.sin(half)
    assert isaac_quat_to_ros(w=w_in, x=x_in, y=y_in, z=z_in) == (x_in, y_in, z_in, w_in)


def test_build_odometry_fields_uses_world_and_base_link_frames():
    fields = build_odometry_fields(
        position=(1.0, 2.0, 0.3), isaac_quat=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.5, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.1),
    )
    assert fields.frame_id == "world"
    assert fields.child_frame_id == "base_link"
    assert fields.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)


def test_build_ground_truth_transform_uses_a_different_child_frame_than_odometry():
    transform = build_ground_truth_transform(position=(1.0, 2.0, 0.3), isaac_quat=(1.0, 0.0, 0.0, 0.0))
    assert transform.frame_id == "world"
    assert transform.child_frame_id == "base_link_gt"


def test_ground_truth_publisher_instantiates_and_publishes_without_error():
    rclpy.init()
    try:
        node = GroundTruthPublisher()
        try:
            node.publish_pose(
                sim_time=node.get_clock().now(),
                position=(1.0, 2.0, 0.3), isaac_quat=(1.0, 0.0, 0.0, 0.0),
                linear_velocity=(0.5, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.1),
            )
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
```

- [ ] **Step 2:** Run → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `pose_export.py`**

```python
from __future__ import annotations
from dataclasses import dataclass


def isaac_quat_to_ros(w: float, x: float, y: float, z: float) -> tuple[float, float, float, float]:
    return (x, y, z, w)


@dataclass(frozen=True)
class OdometryFields:
    frame_id: str
    child_frame_id: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]


def build_odometry_fields(position, isaac_quat, linear_velocity, angular_velocity) -> OdometryFields:
    w, x, y, z = isaac_quat
    return OdometryFields("world", "base_link", position, isaac_quat_to_ros(w=w, x=x, y=y, z=z), linear_velocity, angular_velocity)


@dataclass(frozen=True)
class TransformFields:
    frame_id: str
    child_frame_id: str
    translation: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


def build_ground_truth_transform(position, isaac_quat) -> TransformFields:
    w, x, y, z = isaac_quat
    return TransformFields("world", "base_link_gt", position, isaac_quat_to_ros(w=w, x=x, y=y, z=z))
```

- [ ] **Step 4: Write `ground_truth_node.py`**

```python
from __future__ import annotations

from geometry_msgs.msg import Quaternion, TransformStamped, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import TransformBroadcaster

from slam.slam.ground_truth.pose_export import build_ground_truth_transform, build_odometry_fields


class GroundTruthPublisher(Node):
    def __init__(self) -> None:
        super().__init__("ground_truth_publisher")
        self._odom_pub = self.create_publisher(Odometry, "/ground_truth/odom", 10)
        self._tf_broadcaster = TransformBroadcaster(self)

    def publish_pose(self, sim_time: Time, position, isaac_quat, linear_velocity, angular_velocity) -> None:
        odom_fields = build_odometry_fields(position, isaac_quat, linear_velocity, angular_velocity)
        odom_msg = Odometry()
        odom_msg.header.stamp = sim_time.to_msg()
        odom_msg.header.frame_id = odom_fields.frame_id
        odom_msg.child_frame_id = odom_fields.child_frame_id
        odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z = odom_fields.position
        odom_msg.pose.pose.orientation = Quaternion(**dict(zip("xyzw", odom_fields.orientation_xyzw)))
        odom_msg.twist.twist.linear = Vector3(**dict(zip("xyz", odom_fields.linear_velocity)))
        odom_msg.twist.twist.angular = Vector3(**dict(zip("xyz", odom_fields.angular_velocity)))
        self._odom_pub.publish(odom_msg)

        tf_fields = build_ground_truth_transform(position, isaac_quat)
        tf_msg = TransformStamped()
        tf_msg.header.stamp = sim_time.to_msg()
        tf_msg.header.frame_id = tf_fields.frame_id
        tf_msg.child_frame_id = tf_fields.child_frame_id
        tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z = tf_fields.translation
        tf_msg.transform.rotation = Quaternion(**dict(zip("xyzw", tf_fields.rotation_xyzw)))
        self._tf_broadcaster.sendTransform(tf_msg)


def main() -> None:
    import rclpy
    rclpy.init()
    node = GroundTruthPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

- [ ] **Step 5:** Run → PASS (all five tests).
- [ ] **Step 6:** Leave staged.

---

### Task 4: Scripted SE(2) motion source for SLAM eval

*(Unchanged from v1 — no defects found; Task 7 now actually consumes this for real.)*

**Files:** Create `slam/slam/eval/{__init__,scripted_motion,test_scripted_motion}.py`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from slam.slam.eval.scripted_motion import MotionSegment, canned_schedule, command_at


def test_canned_schedule_household_is_forward_turn_forward_revisit():
    schedule = canned_schedule("household")
    assert [s.label for s in schedule] == ["forward", "turn", "forward", "revisit"]
    assert all(isinstance(s, MotionSegment) and s.duration_seconds > 0 for s in schedule)


def test_canned_schedule_is_identical_across_calls():
    assert canned_schedule("household") == canned_schedule("household")


def test_canned_schedule_unknown_mission_raises():
    with pytest.raises(ValueError, match="unknown mission 'nope'"):
        canned_schedule("nope")


def test_command_at_returns_first_segments_command_at_time_zero():
    schedule = canned_schedule("household")
    assert command_at(schedule, elapsed_seconds=0.0) == schedule[0].command


def test_command_at_advances_to_next_segment_after_duration_elapses():
    schedule = canned_schedule("household")
    assert command_at(schedule, elapsed_seconds=schedule[0].duration_seconds + 0.01) == schedule[1].command


def test_command_at_past_schedule_end_returns_zero_command():
    schedule = canned_schedule("household")
    total = sum(s.duration_seconds for s in schedule)
    assert command_at(schedule, elapsed_seconds=total + 1.0) == (0.0, 0.0, 0.0)
```

- [ ] **Step 2:** Run → FAIL.

- [ ] **Step 3: Write `scripted_motion.py`**

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class MotionSegment:
    label: str
    command: tuple[float, float, float]
    duration_seconds: float


_SCHEDULES: dict[str, list[MotionSegment]] = {
    "household": [
        MotionSegment("forward", (0.5, 0.0, 0.0), 5.0),
        MotionSegment("turn", (0.0, 0.0, 0.5), 3.0),
        MotionSegment("forward", (0.5, 0.0, 0.0), 5.0),
        MotionSegment("revisit", (-0.5, 0.0, 0.5), 6.0),
    ],
}


def canned_schedule(mission: str) -> list[MotionSegment]:
    try:
        return list(_SCHEDULES[mission])
    except KeyError:
        raise ValueError(f"unknown mission {mission!r}, available: {sorted(_SCHEDULES)}") from None


def command_at(schedule: list[MotionSegment], elapsed_seconds: float) -> tuple[float, float, float]:
    cursor = 0.0
    for segment in schedule:
        if elapsed_seconds < cursor + segment.duration_seconds:
            return segment.command
        cursor += segment.duration_seconds
    return (0.0, 0.0, 0.0)
```

- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Leave staged.

---

### Task 5: Showcase camera — chase-cam math + suite wiring (§8, corrected)

**Files:**
- Modify: `common/cerberus_description/sensors/sensor_suite.py`, `sensor_configs.py`
- Create: `slam/slam/eval/showcase_camera_follow.py`, `slam/slam/eval/test_showcase_camera_follow.py`

**Interfaces:** Produces `SensorSuiteCfg.showcase_camera`; `with_showcase_camera(suite, camera) -> SensorSuiteCfg`; `compute_chase_cam_pose(robot_position, robot_yaw, offset_behind=1.5, offset_up=0.8) -> (position, orientation_xyzw)`. Task 7 consumes `with_showcase_camera` to actually turn the camera on for eval runs, and (out of scope — pod-side) `compute_chase_cam_pose` would be called every physics step.

- [ ] **Step 1: Write the failing test for the field + attachment helper**

```python
def test_sensor_suite_cfg_has_showcase_camera_field_defaulting_to_none():
    from common.cerberus_description.sensors.sensor_suite import SensorSuiteCfg
    suite = SensorSuiteCfg(lidar=None, front_camera=object(), rear_camera=None, wrist_camera=None)
    assert suite.showcase_camera is None


def test_locked_suites_do_not_spawn_showcase_camera_by_default():
    from common.cerberus_description.sensors.sensor_configs import CAMERA_ONLY_SUITE, LIDAR_CAMERA_SUITE
    assert LIDAR_CAMERA_SUITE.showcase_camera is None
    assert CAMERA_ONLY_SUITE.showcase_camera is None


def test_with_showcase_camera_attaches_without_mutating_the_original():
    from common.cerberus_description.sensors.sensor_configs import LIDAR_CAMERA_SUITE, SHOWCASE_CAMERA_CFG
    from common.cerberus_description.sensors.sensor_suite import with_showcase_camera
    eval_suite = with_showcase_camera(LIDAR_CAMERA_SUITE, SHOWCASE_CAMERA_CFG)
    assert eval_suite.showcase_camera is SHOWCASE_CAMERA_CFG
    assert LIDAR_CAMERA_SUITE.showcase_camera is None  # original untouched
    assert eval_suite.lidar is LIDAR_CAMERA_SUITE.lidar  # every other field carried over
```

- [ ] **Step 2:** Run → FAIL.

- [ ] **Step 3: Add the field and helper to `sensor_suite.py`**

```python
import dataclasses

@dataclass
class SensorSuiteCfg:
    lidar: RayCasterCfg | None
    front_camera: CameraCfg
    rear_camera: CameraCfg | None
    wrist_camera: CameraCfg | None
    # Video/demo aid, independent of sensor_profile -- not consumed by any
    # SLAM/RL/IL algorithm. Excluded from every sensor_requirements manifest
    # and RL observation-space gating. Only turned on via with_showcase_camera()
    # for single/few-env eval/recording runs, never during massively-parallel
    # RL training.
    showcase_camera: CameraCfg | None = None


def with_showcase_camera(suite: SensorSuiteCfg, camera: CameraCfg) -> SensorSuiteCfg:
    """Returns a copy of `suite` with showcase_camera attached -- the locked
    Phase-1 suites (LIDAR_CAMERA_SUITE/CAMERA_ONLY_SUITE) stay untouched;
    eval scripts use this to opt in per-run instead."""
    return dataclasses.replace(suite, showcase_camera=camera)
```

- [ ] **Step 4: Add `SHOWCASE_CAMERA_CFG` to `sensor_configs.py`**

```python
# Static offset only -- see slam/slam/eval/showcase_camera_follow.py for the
# actual soft-follow pose math (§8: position tracks the robot, orientation
# stays level). Wiring compute_chase_cam_pose()'s output into a per-physics-
# step prim-transform update on THIS camera is a pod-side Isaac Lab
# integration step, out of scope here (needs a live sim to verify against).
SHOWCASE_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/ShowcaseCamera",  # NOT parented under Robot/base -- a rigid
    # parent-child offset would inherit body roll/pitch, contradicting §8's
    # "orientation stays level" requirement. This prim's transform must be
    # set explicitly every step from compute_chase_cam_pose()'s output
    # instead of relying on a fixed offset.
    update_period=1 / 30,
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5),
    ),
)
```

- [ ] **Step 5:** Run → PASS.

- [ ] **Step 6: Write the failing test for the chase-cam pose math**

```python
import math
from slam.slam.eval.showcase_camera_follow import compute_chase_cam_pose


def test_chase_cam_trails_directly_behind_at_zero_yaw():
    position, orientation_xyzw = compute_chase_cam_pose(robot_position=(0.0, 0.0, 0.0), robot_yaw=0.0)
    x, y, z = position
    assert x < 0.0  # behind along +x heading means camera sits at -x
    assert y == pytest.approx(0.0)
    assert z > 0.0  # above the robot
    assert orientation_xyzw == (0.0, 0.0, 0.0, 1.0)  # level, no yaw offset


def test_chase_cam_orientation_tracks_yaw_only():
    _, orientation_xyzw = compute_chase_cam_pose(robot_position=(0.0, 0.0, 0.0), robot_yaw=math.pi / 2)
    x, y, z, w = orientation_xyzw
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)  # roll/pitch never touched, only yaw (z, w)
    assert z == pytest.approx(math.sin(math.pi / 4))
    assert w == pytest.approx(math.cos(math.pi / 4))


def test_chase_cam_position_follows_yaw_direction():
    position, _ = compute_chase_cam_pose(robot_position=(5.0, 5.0, 0.3), robot_yaw=math.pi / 2)
    x, y, z = position
    assert x == pytest.approx(5.0)  # cos(pi/2)=0, no x offset at this heading
    assert y < 5.0  # sin(pi/2)=1, camera trails behind along -y
```

(needs `import pytest` at the top of `slam/slam/eval/test_showcase_camera_follow.py`.)

- [ ] **Step 7:** Run → FAIL.

- [ ] **Step 8: Write `showcase_camera_follow.py`**

```python
"""Pure chase-cam pose math for the showcase camera:
position tracks the robot, orientation stays level (yaw-only, no roll/pitch)
rather than rigidly inheriting the robot's full orientation. Kept separate
from any Isaac Lab per-step wiring (needs a pod, same constraint as
ground_truth/pose_export.py) so the math is unit-testable without Isaac Sim.
"""

from __future__ import annotations

import math


def compute_chase_cam_pose(
    robot_position: tuple[float, float, float],
    robot_yaw: float,
    offset_behind: float = 1.5,
    offset_up: float = 0.8,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Returns (position, orientation_xyzw). Camera trails offset_behind
    meters behind the robot along its current yaw heading and offset_up
    meters above it -- only yaw is inherited, never roll/pitch."""
    x, y, z = robot_position
    camera_position = (
        x - offset_behind * math.cos(robot_yaw),
        y - offset_behind * math.sin(robot_yaw),
        z + offset_up,
    )
    half_yaw = robot_yaw / 2.0
    orientation_xyzw = (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))
    return camera_position, orientation_xyzw
```

- [ ] **Step 9:** Run → PASS (all tests in this task).
- [ ] **Step 10:** Leave staged.

---

### Task 6a: `cerberus_msgs` interfaces package — the `TaggedMap` message

**Why this exists (fixes audit findings #1/#2):** §2/§5 require `/slam/map` to be "tagged with a `map_type` field" **on the data itself**. v1 tried to do this via a ROS2 node *parameter*, which no subscriber can see — not a real tag. The actual fix is a real message wrapping the native payload plus the tag, which is exactly the job `project_structure.md` already reserves `cerberus_msgs` for ("custom msg/srv/action defs... the kind of thing that might get reused standalone later").

**Files:** Create `common/cerberus_msgs/{CMakeLists.txt,package.xml,msg/TaggedMap.msg}`.

- [ ] **Step 1: Write `package.xml`**

```xml
<?xml version="1.0"?>
<package format="3">
  <name>cerberus_msgs</name>
  <version>0.1.0</version>
  <description>Custom message/srv/action definitions shared across Cerberus stack components</description>
  <maintainer email="k.thiha10.mail@gmail.com">Kevin</maintainer>
  <license>MIT</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <depend>sensor_msgs</depend>
  <build_depend>rosidl_default_generators</build_depend>
  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>
  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

- [ ] **Step 2: Write `msg/TaggedMap.msg`**

```
# Wraps a SLAM backend's native map output with the map_type tag REQUIRED by
# the SLAM topic contract to live on the topic's actual payload, not a node
# parameter (see this task's rationale). All six candidates registered in
# slam/slam/core/algorithms.py currently produce point clouds only
# ("FAST-LIO2, BIEVR-LIO, FAST-LIVO2, LVI-Q, ORB-SLAM3,
# Multicam-SLAM all produce point clouds") -- occupancy_grid/pose_graph
# fields are deliberately NOT added here (YAGNI) until a candidate that
# produces one is actually registered.

string map_type          # "point_cloud" for every candidate registered as of this plan
sensor_msgs/PointCloud2 point_cloud
```

- [ ] **Step 3: Write `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.8)
project(cerberus_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TaggedMap.msg"
  DEPENDENCIES sensor_msgs
)

ament_package()
```

- [ ] **Step 4: Build to verify it's valid**

Run: `pixi run colcon build --packages-select cerberus_msgs --symlink-install`
Expected: `Summary: 1 package finished`. If it fails on a missing `rosidl_default_generators` dependency, add `ros-jazzy-rosidl-default-generators` to `pixi.toml`'s `[dependencies]` and retry — don't work around a missing dependency by simplifying the message.

- [ ] **Step 5: Verify the generated Python bindings import correctly**

Run: `pixi run bash -c "source install/setup.bash && python3 -c 'from cerberus_msgs.msg import TaggedMap; m = TaggedMap(); m.map_type = \"point_cloud\"; print(m.map_type)'"`
Expected: prints `point_cloud`, confirming the message is real and importable before Task 6c's adapter code depends on it.

- [ ] **Step 6:** Leave staged.

---

### Task 6b: ROS2 package scaffolding for `slam/`

*(Unchanged from v1's Task 6 Step 7 — no defects found in this part.)*

**Files:** Create `slam/package.xml`, `slam/setup.py`, `slam/setup.cfg`, `slam/resource/slam`.

- [ ] **Step 1: `slam/package.xml`**

```xml
<?xml version="1.0"?>
<package format="3">
  <name>slam</name>
  <version>0.1.0</version>
  <description>Pluggable multi-backend SLAM: registry, ground-truth export, eval, adapters</description>
  <maintainer email="k.thiha10.mail@gmail.com">Kevin</maintainer>
  <license>MIT</license>
  <depend>rclpy</depend>
  <depend>nav_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>tf2_ros</depend>
  <depend>cerberus_msgs</depend>
  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

- [ ] **Step 2: `slam/setup.py`**

```python
from setuptools import find_packages, setup

package_name = "slam"

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
    maintainer="Kevin",
    maintainer_email="k.thiha10.mail@gmail.com",
    description="Pluggable multi-backend SLAM",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ground_truth_publisher = slam.slam.ground_truth.ground_truth_node:main",
            "fast_lio2_adapter = slam.slam.backends.fast_lio2.adapter_launch:main",
            "bievr_lio_adapter = slam.slam.backends.bievr_lio.adapter_launch:main",
            "fast_livo2_adapter = slam.slam.backends.fast_livo2.adapter_launch:main",
            "orb_slam3_adapter = slam.slam.backends.orb_slam3.adapter_launch:main",
            "multicam_slam_adapter = slam.slam.backends.multicam_slam.adapter_launch:main",
        ],
    },
)
```

- [ ] **Step 3: `slam/setup.cfg`**

```ini
[develop]
script_dir=$base/lib/slam
[install]
install_scripts=$base/lib/slam
```

- [ ] **Step 4:** Create `slam/resource/slam` (empty file).

- [ ] **Step 5: Build to verify** — `pixi run colcon build --packages-select slam --symlink-install` → `Summary: 1 package finished`.
- [ ] **Step 6:** Leave staged.

---

### Task 6c: The shared `SlamBackendAdapter` (fixes audit findings #1, #2, #4)

**Why one shared class instead of six `topic_tools relay` launch files:** the real §2 contract is three things (odom, TF `map→base_link`, tagged map) and v1 only wired two of them, incorrectly. Writing that correctly six times would both risk drift and violate the project's own DRY ethos — so it's one class, parameterized per backend by native topic names only, matching §2's "the backend adapter is thin... does NOT encode backend-specific logic beyond remapping."

**Files:** Create `slam/slam/backends/common_adapter.py`, `slam/slam/backends/test_common_adapter.py`, and one `adapter_launch.py` per backend.

**Interfaces:** Consumes `cerberus_msgs.msg.TaggedMap` (Task 6a). Produces `SlamBackendAdapter(node_name, native_odom_topic, native_map_topic)`, an `rclpy.Node` publishing `/slam/odom`, `/slam/map` (`TaggedMap`), and broadcasting TF `map -> base_link`.

- [ ] **Step 1: Write the failing test**

```python
import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

from slam.slam.backends.common_adapter import SlamBackendAdapter


def _make_odom(x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float) -> Odometry:
    msg = Odometry()
    msg.header.frame_id = "some_native_frame"  # deliberately NOT "map" -- adapter must overwrite this
    msg.pose.pose = Pose(position=Point(x=x, y=y, z=z), orientation=Quaternion(x=qx, y=qy, z=qz, w=qw))
    return msg


def test_on_odom_republishes_with_contract_frame_ids_and_broadcasts_tf():
    rclpy.init()
    try:
        node = SlamBackendAdapter(node_name="test_adapter", native_odom_topic="/native/odom", native_map_topic="/native/map")
        try:
            published = []
            node._odom_pub.publish = lambda msg: published.append(msg)  # capture instead of a live subscriber
            node._on_odom(_make_odom(1.0, 2.0, 0.3, 0.0, 0.0, 0.0, 1.0))
            assert len(published) == 1
            assert published[0].header.frame_id == "map"
            assert published[0].child_frame_id == "base_link"
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


def test_on_map_wraps_native_point_cloud_with_point_cloud_map_type_tag():
    rclpy.init()
    try:
        node = SlamBackendAdapter(node_name="test_adapter2", native_odom_topic="/native/odom", native_map_topic="/native/map")
        try:
            published = []
            node._map_pub.publish = lambda msg: published.append(msg)
            native_cloud = PointCloud2()
            node._on_map(native_cloud)
            assert len(published) == 1
            assert published[0].map_type == "point_cloud"
            assert published[0].point_cloud is native_cloud
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
```

- [ ] **Step 2:** Run → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `common_adapter.py`**

```python
"""Generic thin SLAM backend adapter -- subscribes to a backend's native
odometry and point-cloud-map topics and republishes them onto this repo's
SLAM topic contract: /slam/odom (nav_msgs/Odometry,
frame ids rewritten to map/base_link), /slam/map (cerberus_msgs/TaggedMap,
map_type carried IN the payload, not a node parameter -- see Task 6a's
rationale), and a TF broadcast map -> base_link derived from the same
odometry pose. Every backend's own adapter_launch.py under
slam/slam/backends/<name>/ is just this class instantiated with that
backend's native topic names -- per §2, "the backend adapter is thin...
does NOT encode backend-specific logic beyond remapping."
"""

from __future__ import annotations

from cerberus_msgs.msg import TaggedMap
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformBroadcaster


class SlamBackendAdapter(Node):
    def __init__(self, node_name: str, native_odom_topic: str, native_map_topic: str) -> None:
        super().__init__(node_name)
        self._odom_pub = self.create_publisher(Odometry, "/slam/odom", 10)
        self._map_pub = self.create_publisher(TaggedMap, "/slam/map", 10)
        self._tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, native_odom_topic, self._on_odom, 10)
        self.create_subscription(PointCloud2, native_map_topic, self._on_map, 10)

    def _on_odom(self, msg: Odometry) -> None:
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        self._odom_pub.publish(msg)

        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = "map"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z
        transform.transform.rotation = msg.pose.pose.orientation
        self._tf_broadcaster.sendTransform(transform)

    def _on_map(self, msg: PointCloud2) -> None:
        tagged = TaggedMap()
        tagged.map_type = "point_cloud"
        tagged.point_cloud = msg
        self._map_pub.publish(tagged)
```

- [ ] **Step 4:** Run → PASS.

- [ ] **Step 5: Write each backend's `adapter_launch.py`**

`slam/slam/backends/fast_lio2/adapter_launch.py` (topic names confirmed against FAST-LIO2's own README/source conventions — `/Odometry`, `/cloud_registered`):

```python
"""Thin adapter for FAST-LIO2 -- see slam/slam/backends/common_adapter.py for
the shared remap/TF/tagging logic. Runs on the ROS2 side of ros1_bridge, which republishes FAST-LIO2's native ROS1 topics
(/Odometry, /cloud_registered) onto ROS2 first."""

import rclpy

from slam.slam.backends.common_adapter import SlamBackendAdapter


def main() -> None:
    rclpy.init()
    node = SlamBackendAdapter(node_name="fast_lio2_adapter", native_odom_topic="/Odometry", native_map_topic="/cloud_registered")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

`slam/slam/backends/bievr_lio/adapter_launch.py` (no bridge — native ROS2 build per Task 1's confirmed finding; topic names are placeholders pending a direct read of `third_party/slam/bievr_lio`'s own launch files once fetched, per this task's own honesty requirement):

```python
"""Thin adapter for BIEVR-LIO's native ROS2 build (bievr_lio_ros2) -- no
ros1_bridge involved, unlike every other backend here.
Topic names below are PLACEHOLDERS -- confirm against
third_party/slam/bievr_lio's own launch files once fetched (Task 6d) before
relying on this in a real run."""

import rclpy

from slam.slam.backends.common_adapter import SlamBackendAdapter


def main() -> None:
    rclpy.init()
    node = SlamBackendAdapter(
        node_name="bievr_lio_adapter",
        native_odom_topic="/bievr_lio/odometry",  # PLACEHOLDER, confirm against vendored source
        native_map_topic="/bievr_lio/voxel_map",  # PLACEHOLDER, confirm against vendored source
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

`slam/slam/backends/fast_livo2/adapter_launch.py`, `slam/slam/backends/orb_slam3/adapter_launch.py`, `slam/slam/backends/multicam_slam/adapter_launch.py` — same shape as `fast_lio2`'s (via `ros1_bridge`), with topic names marked `# PLACEHOLDER, confirm against vendored source` exactly like `bievr_lio`'s, to be corrected in Task 6d's Step 11 (grep against vendored repos) rather than guessed now.

`slam/slam/backends/lvi_q/adapter_launch.py`:

```python
"""Placeholder -- LVI-Q has no existing implementation to adapt (no public
repo, implemented from spec). This backend is not
launchable until that implementation exists."""


def main() -> None:
    raise NotImplementedError(
        "LVI-Q has no implementation yet (implemented from spec, not vendored) -- "
        "see this file's module docstring"
    )
```

- [ ] **Step 6:** Leave staged.

---

### Task 6d: Vendor backend source locally + per-backend Docker images

*(Same as v1's Task 6 Steps 1-6, 15-19 — the fetch script and Dockerfiles themselves had no audit findings against them; only the adapter content, fixed in 6c above, was wrong.)*

**Files:** Create `scripts/fetch_slam_backends.py`, `scripts/test_fetch_slam_backends.py`, five Dockerfiles, `docker/scripts/start_slam_backend.sh`; modify `.gitignore`.

- [ ] **Step 1: Write the failing CLI test** (`scripts/test_fetch_slam_backends.py`):

```python
import subprocess
import sys

VENDORABLE_BACKENDS = {"fast_lio2", "bievr_lio", "fast_livo2", "orb_slam3", "multicam_slam"}


def test_lvi_q_is_not_a_valid_fetch_target():
    result = subprocess.run([sys.executable, "scripts/fetch_slam_backends.py", "lvi_q"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "lvi_q" in result.stderr
    for name in VENDORABLE_BACKENDS:
        assert name in result.stderr


def test_unknown_backend_name_errors_with_valid_choices_listed():
    result = subprocess.run([sys.executable, "scripts/fetch_slam_backends.py", "not_a_real_backend"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "not_a_real_backend" in result.stderr


def test_no_arguments_and_no_all_flag_errors_with_usage():
    result = subprocess.run([sys.executable, "scripts/fetch_slam_backends.py"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "--all" in result.stderr
```

- [ ] **Step 2:** Run → FAIL (file doesn't exist).

- [ ] **Step 3: Write `scripts/fetch_slam_backends.py`**

```python
#!/usr/bin/env python3
"""Vendors SLAM backend source locally (gitignored under third_party/slam/).
lvi_q has no entry -- no public repo exists."""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_SLAM = REPO_ROOT / "third_party" / "slam"

BACKENDS = {
    "fast_lio2": ("https://github.com/hku-mars/FAST_LIO.git", None),
    "bievr_lio": ("https://github.com/ethz-asl/BIEVR-LIO.git", None),
    "fast_livo2": ("https://github.com/hku-mars/FAST-LIVO2.git", None),
    "orb_slam3": ("https://github.com/UZ-SLAMLab/ORB_SLAM3.git", None),
    "multicam_slam": ("https://github.com/AlterPang/Multi_ORB_SLAM.git", None),
}


def ensure_colcon_ignore() -> None:
    """Vendored backends (fast_lio2, orb_slam3, etc.) ship their own
    CMakeLists.txt/package.xml -- without this marker, `colcon build` walks
    into third_party/ and tries to build them locally (confirmed during this
    plan's own execution 2026-09-20: ORB_SLAM2/ORB_SLAM3 failed on a CMake
    version mismatch, other vendored packages aborted the build entirely).
    These are built ONLY inside their own per-backend Docker images, never here. third_party/ is gitignored and gets
    wiped/re-fetched, so this marker is (re-)created here rather than
    relying on a one-off file surviving -- idempotent, safe to call every
    run."""
    marker = THIRD_PARTY_SLAM.parent / "COLCON_IGNORE"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def fetch_one(name: str) -> None:
    url, ref = BACKENDS[name]
    dest = THIRD_PARTY_SLAM / name
    if dest.is_dir() and any(dest.iterdir()):
        print(f"[fetch_slam_backends] {name}: already present at {dest}, skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [url, str(dest)]
    print(f"[fetch_slam_backends] {name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not args.names and not args.all:
        parser.print_usage(sys.stderr)
        print("error: pass one or more backend names, or --all", file=sys.stderr)
        return 1

    targets = list(BACKENDS) if args.all else args.names
    unknown = [name for name in targets if name not in BACKENDS]
    if unknown:
        print(
            f"error: unknown backend(s) {unknown!r}, valid choices: {sorted(BACKENDS)} "
            f"(lvi_q has no repo -- implemented from spec, not fetched)",
            file=sys.stderr,
        )
        return 1

    ensure_colcon_ignore()
    for name in targets:
        fetch_one(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4:** Run → PASS.

- [ ] **Step 5:** Add to `.gitignore`, after the `data/`/`outputs/` block:

```
# vendored SLAM backend source (scripts/fetch_slam_backends.py) -- large,
# external, never authored here.
third_party/
```

Also add a `pytest.ini` at the repo root if one doesn't already exist, with `norecursedirs = third_party build install log .git .pixi` — confirmed during this plan's execution that `third_party/`'s vendored source (ORB_SLAM3 bundles a Sophus Python binding that fails to import locally, missing `sympy`) breaks `pixi run pytest` repo-wide the moment it exists on disk, unrelated to any real test failure.

- [ ] **Step 6: Checkpoint, then fetch for real**

Run: `pixi run python scripts/fetch_slam_backends.py --all`
Expected: five directories under `third_party/slam/`. Confirm via `git status` that `third_party/` does NOT appear as untracked-to-add.

- [ ] **Step 7: Confirm real topic names for the four placeholder adapters**

```bash
grep -rn "advertise\|create_publisher\|Publisher<" third_party/slam/fast_livo2 --include='*.cpp' --include='*.launch' | head -30
grep -rn "advertise\|create_publisher\|Publisher<" third_party/slam/orb_slam3 --include='*.cc' --include='*.launch' | head -30
grep -rn "advertise\|create_publisher\|Publisher<" third_party/slam/multicam_slam --include='*.cc' --include='*.launch' | head -30
grep -rn "create_publisher\|advertise" third_party/slam/bievr_lio --include='*.cpp' --include='*.launch.py' | head -30
```

Update each `adapter_launch.py`'s `native_odom_topic`/`native_map_topic` to the confirmed real names, removing the `# PLACEHOLDER` comment once confirmed. If a name genuinely can't be confirmed by grep (set via a runtime param), leave the placeholder with a comment naming the exact file/line that would need runtime confirmation, rather than guessing.

- [ ] **Step 8: Write the five Dockerfiles**

`docker/Dockerfile.slam_fast_lio2`:

```dockerfile
# Cerberus -- SLAM backend image: FAST-LIO2 (LiDAR-only baseline).
# ROS1 Noetic base. Source vendored locally, NOT
# git-cloned here -- run `python scripts/fetch_slam_backends.py fast_lio2`
# before building.
FROM ros:noetic-ros-base
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros1-bridge libpcl-dev libeigen3-dev python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /catkin_ws/src
COPY third_party/slam/fast_lio2 /catkin_ws/src/FAST_LIO

WORKDIR /catkin_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin init && catkin build"

COPY docker/scripts/start_slam_backend.sh /start_slam_backend.sh
RUN chmod +x /start_slam_backend.sh
ENTRYPOINT ["/start_slam_backend.sh"]
```

`docker/Dockerfile.slam_bievr_lio` (ROS2 Jazzy-native, no bridge, plus its confirmed Ceres-from-source requirement):

```dockerfile
# Cerberus -- SLAM backend image: BIEVR-LIO (LiDAR-only latest).
# ROS2 Jazzy-native -- confirmed 2026-09-20 against its own README:
# ships bievr_lio_ros2, no ros1_bridge needed.
FROM ros:jazzy-ros-base
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions libeigen3-dev cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY third_party/slam/bievr_lio /colcon_ws/src/BIEVR-LIO
RUN /colcon_ws/src/BIEVR-LIO/docker/scripts/install_ceres.sh

WORKDIR /colcon_ws
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && colcon build --packages-up-to bievr_lio_ros2"

COPY docker/scripts/start_slam_backend.sh /start_slam_backend.sh
RUN chmod +x /start_slam_backend.sh
ENTRYPOINT ["/start_slam_backend.sh"]
```

`docker/Dockerfile.slam_fast_livo2` — same shape as `fast_lio2`'s, `COPY third_party/slam/fast_livo2 /catkin_ws/src/FAST-LIVO2`.

`docker/Dockerfile.slam_orb_slam3` and `docker/Dockerfile.slam_multicam_slam` — same ROS1 Noetic shape, substituting `libopencv-dev` for `libpcl-dev` (ORB-based candidates need OpenCV, not PCL) and their respective `COPY third_party/slam/<name> /catkin_ws/src/<RepoName>` lines.

- [ ] **Step 9: Write `docker/scripts/start_slam_backend.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

if [[ -f /catkin_ws/devel/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /catkin_ws/devel/setup.bash
elif [[ -f /colcon_ws/install/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /colcon_ws/install/setup.bash
fi

exec "$@"
```

- [ ] **Step 10: Checkpoint before building/pushing — confirm scope with Kevin**, then build/push the confirmed backend(s):

```bash
docker build -t kevinkyawthiha/cerberus-slam-fast-lio2:v1 -f docker/Dockerfile.slam_fast_lio2 .
docker push kevinkyawthiha/cerberus-slam-fast-lio2:v1
```

- [ ] **Step 11:** Leave staged (`third_party/` should NOT appear in `git status`).

---

### Task 6e: `slam_backends.yaml` — launch-time backend selection config

*(Same as v1 — no defects found.)*

**Files:** Create `system/cerberus_bringup/config/slam_backends.yaml`, `system/cerberus_bringup/test/test_slam_backends_yaml.py`.

- [ ] **Step 1:** Write the config:

```yaml
fast_lio2:
  image: kevinkyawthiha/cerberus-slam-fast-lio2:v1
  ros1_bridge_required: true
  adapter: slam.slam.backends.fast_lio2.adapter_launch
bievr_lio:
  image: kevinkyawthiha/cerberus-slam-bievr-lio:v1
  ros1_bridge_required: false
  adapter: slam.slam.backends.bievr_lio.adapter_launch
fast_livo2:
  image: kevinkyawthiha/cerberus-slam-fast-livo2:v1
  ros1_bridge_required: true
  adapter: slam.slam.backends.fast_livo2.adapter_launch
lvi_q:
  image: null
  ros1_bridge_required: false
  adapter: slam.slam.backends.lvi_q.adapter_launch
orb_slam3:
  image: kevinkyawthiha/cerberus-slam-orb-slam3:v1
  ros1_bridge_required: true
  adapter: slam.slam.backends.orb_slam3.adapter_launch
multicam_slam:
  image: kevinkyawthiha/cerberus-slam-multicam-slam:v1
  ros1_bridge_required: true
  adapter: slam.slam.backends.multicam_slam.adapter_launch
```

- [ ] **Step 2:** Write the drift-guard test:

```python
from pathlib import Path
import yaml
import slam.slam.core.algorithms as slam_algorithms

SLAM_BACKENDS_YAML = Path(__file__).resolve().parents[1] / "config" / "slam_backends.yaml"


def _load() -> dict:
    with open(SLAM_BACKENDS_YAML) as f:
        return yaml.safe_load(f)


# The six real candidates -- checked explicitly rather than trusting
# available_algorithms() wholesale. Found during full-suite execution
# (2026-09-20): that registry is a shared, process-global dict that
# slam/slam/core/test_algorithms.py's own closure tests deliberately pollute
# with throwaway names (regtest1, etc.) -- iterating available_algorithms()
# blindly here fails whenever this test happens to run in the same pytest
# process after those tests already registered their scratch entries. Only
# surfaces when the whole suite runs together, not when this file is tested
# in isolation -- exactly why the plan's own "final verification: run the
# whole suite" step exists.
REAL_SLAM_CANDIDATES = {"fast_lio2", "bievr_lio", "fast_livo2", "lvi_q", "orb_slam3", "multicam_slam"}


def test_every_registered_slam_algorithm_has_a_backends_yaml_entry():
    backends = _load()
    for name in REAL_SLAM_CANDIDATES & set(slam_algorithms.available_algorithms()):
        assert name in backends


def test_every_backends_yaml_entry_is_a_registered_slam_algorithm():
    backends = _load()
    known = slam_algorithms.available_algorithms()
    for name in backends:
        assert name in known


def test_lvi_q_has_no_image_every_other_backend_does():
    backends = _load()
    assert backends["lvi_q"]["image"] is None
    for name, entry in backends.items():
        if name != "lvi_q":
            assert entry["image"] is not None
```

- [ ] **Step 3:** Run → PASS.
- [ ] **Step 4:** Leave staged.

---

### Task 7: rosbag2 capture (now actually driving the robot) + evo ATE **and** RPE scoring (§9, corrected)

**Files:** Create `slam/slam/eval/record_mission_bag.py`, `slam/slam/eval/score_with_evo.py`, `slam/slam/eval/test_score_with_evo.py`.

**Interfaces:** Consumes `canned_schedule`/`command_at` (Task 4), the `cmd_vel`/`geometry_msgs/Twist` convention already used by `policy/locomotion_policy/locomotion_policy/inference_node.py:103` (confirmed, reused rather than invented). **`with_showcase_camera` (Task 5) is NOT consumed here** — corrected after audit: that helper attaches a camera to an Isaac Lab `SensorSuiteCfg`, which is the concern of whatever launches the Isaac Lab env for an eval run (a pod-side script, out of scope in this plan per the Explicitly-out-of-scope section) — `record_mission_bag.py` only records whatever ROS2 topics already exist, it never touches Isaac Lab config directly. Produces a bag under `slam/slam/eval/results/<backend>/<run_timestamp>/bag/` (path now derived from `--run-timestamp`, not a free-form `--output`, fixing an audit-found gap where the two could silently diverge); `score_with_evo.build_evo_ape_command`/`build_evo_rpe_command`.

- [ ] **Step 1: Write `record_mission_bag.py`** — actually drives the robot this time, via the same `cmd_vel` topic the real inference node already subscribes to. **Corrected after audit** on three points: (a) `--output` is gone — the bag path is now derived from `--run-timestamp` alone, so there is exactly one source of truth for where a run's artifacts live, matching Task 10's R2-sync path expectations instead of leaving them independently guessable; (b) a startup grace delay is added before publishing any command, and a settle delay before stopping the recorder, closing a race where the bag could start before `ros2 bag record` finishes subscribing, or stop before the robot's final motion has propagated into odom/tf; (c) `signal.SIGINT` is now imported explicitly rather than relying on `subprocess`'s undocumented internal `signal` import:

```python
#!/usr/bin/env python3
"""Records one SLAM eval mission as a rosbag2 bag: every backend's contract
topics (/slam/odom, /tf, /slam/map), the RAW sensor feeds, /ground_truth/odom,
and the showcase camera. Actually drives the robot during the recording by
publishing the scripted schedule's commands on `cmd_vel` (geometry_msgs/Twist)
-- the same topic policy/locomotion_policy/locomotion_policy/inference_node.py
already subscribes to (confirmed, not invented), so this works against the
real inference node unmodified.

Output path is derived entirely from --run-timestamp (single source of
truth, so Task 10's R2 sync always finds what this script wrote):
    slam/slam/eval/results/<backend>/<run_timestamp>/bag/
"""

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from slam.slam.eval.scripted_motion import canned_schedule, command_at

CONTRACT_TOPICS = ["/slam/odom", "/tf", "/slam/map", "/ground_truth/odom", "/showcase_camera/rgb"]
# PLACEHOLDER -- match this repo's existing sensor-mount naming convention, but NOT independently confirmed against a real
# Isaac-Lab-to-ROS2 publishing layer, which doesn't exist yet (see this
# plan's Explicitly-out-of-scope section: Dockerfile.remote has no ROS2).
# Confirm/correct once that publishing layer exists, same as Task 6c's
# per-backend topic-name placeholders.
RAW_SENSOR_TOPICS_LIDAR_CAMERA = ["/lidar/points", "/front_camera/rgb", "/front_camera/depth"]
RAW_SENSOR_TOPICS_CAMERA_ONLY = ["/front_camera/rgb", "/front_camera/depth", "/rear_camera/rgb", "/rear_camera/depth"]

STARTUP_GRACE_SECONDS = 3.0  # time for `ros2 bag record` to finish subscribing before motion starts
SETTLE_SECONDS = 2.0         # time for the robot's final motion to propagate into odom/tf before stopping the recorder


def run_dir_for(backend: str, run_timestamp: str) -> Path:
    return Path("slam/slam/eval/results") / backend / run_timestamp


class ScriptedMotionPublisher(Node):
    def __init__(self) -> None:
        super().__init__("scripted_motion_publisher")
        self._pub = self.create_publisher(Twist, "cmd_vel", 10)

    def publish_command(self, vx: float, vy: float, omega_z: float) -> None:
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = omega_z
        self._pub.publish(msg)


def run_scripted_mission(mission: str, rate_hz: float = 20.0) -> None:
    schedule = canned_schedule(mission)
    total_duration = sum(segment.duration_seconds for segment in schedule)
    rclpy.init()
    node = ScriptedMotionPublisher()
    try:
        start = time.monotonic()
        period = 1.0 / rate_hz
        while True:
            elapsed = time.monotonic() - start
            if elapsed > total_duration:
                node.publish_command(0.0, 0.0, 0.0)
                break
            vx, vy, omega_z = command_at(schedule, elapsed)
            node.publish_command(vx, vy, omega_z)
            time.sleep(period)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--mission", required=True)
    parser.add_argument("--run-timestamp", required=True, help="e.g. 2026-09-20_10-00-00 -- determines the bag's output path, see module docstring")
    parser.add_argument("--sensor-rig", choices=["lidar_camera", "camera_only"], required=True)
    args = parser.parse_args()

    raw_topics = RAW_SENSOR_TOPICS_LIDAR_CAMERA if args.sensor_rig == "lidar_camera" else RAW_SENSOR_TOPICS_CAMERA_ONLY
    all_topics = CONTRACT_TOPICS + raw_topics

    schedule = canned_schedule(args.mission)  # validate the mission name early, before spawning anything
    total_duration = sum(segment.duration_seconds for segment in schedule)

    bag_output = run_dir_for(args.backend, args.run_timestamp) / "bag"
    record_cmd = ["ros2", "bag", "record", "-o", str(bag_output), *all_topics]
    print(f"[record_mission_bag] backend={args.backend} mission={args.mission} rig={args.sensor_rig} duration={total_duration}s")
    print(f"[record_mission_bag] {' '.join(record_cmd)}")
    record_proc = subprocess.Popen(record_cmd)

    try:
        time.sleep(STARTUP_GRACE_SECONDS)  # let ros2 bag record finish subscribing before any motion happens
        run_scripted_mission(args.mission)  # blocks for total_duration, actually publishing cmd_vel
        time.sleep(SETTLE_SECONDS)  # let the robot's final motion propagate into odom/tf before stopping the recorder
    finally:
        record_proc.send_signal(signal.SIGINT)  # let rosbag2 flush cleanly
        record_proc.wait(timeout=30)

    return record_proc.returncode


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write the failing test for `score_with_evo.py`'s command-building logic**

```python
from slam.slam.eval.score_with_evo import build_evo_ape_command, build_evo_rpe_command


def test_build_evo_ape_command_targets_ground_truth_and_slam_odom_topics():
    cmd = build_evo_ape_command(bag_path="bags/fast_lio2_household", backend_name="fast_lio2")
    assert cmd[0:2] == ["evo_ape", "bag2"]
    assert "bags/fast_lio2_household" in cmd
    assert "/ground_truth/odom" in cmd
    assert "/slam/odom" in cmd


def test_build_evo_ape_command_includes_backend_name_in_output_path():
    cmd = build_evo_ape_command(bag_path="bags/fast_lio2_household", backend_name="fast_lio2")
    idx = cmd.index("--save_results")
    assert "fast_lio2" in cmd[idx + 1]


def test_build_evo_rpe_command_targets_ground_truth_and_slam_odom_topics():
    cmd = build_evo_rpe_command(bag_path="bags/fast_lio2_household", backend_name="fast_lio2")
    assert cmd[0:2] == ["evo_rpe", "bag2"]
    assert "/ground_truth/odom" in cmd
    assert "/slam/odom" in cmd


def test_build_evo_rpe_command_includes_backend_name_in_output_path():
    cmd = build_evo_rpe_command(bag_path="bags/fast_lio2_household", backend_name="fast_lio2")
    idx = cmd.index("--save_results")
    assert "fast_lio2" in cmd[idx + 1]
    assert "rpe" in cmd[idx + 1]
```

- [ ] **Step 3:** Run → FAIL.

- [ ] **Step 4: Write `score_with_evo.py`**

```python
"""Command-building logic for evo ATE and RPE scoring, kept separate from
actually invoking evo. evo, computing ATE/RPE per
candidate against /ground_truth/odom -- the same tool the KIT benchmark
paper itself used. `bag2 <bagfile> <ref_topic> <est_topic> [options]` syntax
confirmed real and current against evo's own repo/wiki (2026-09-20) -- not a
guess.
"""

from __future__ import annotations

import subprocess


def build_evo_ape_command(bag_path: str, backend_name: str) -> list[str]:
    return [
        "evo_ape", "bag2", bag_path,
        "/ground_truth/odom", "/slam/odom",
        "--align",
        "--save_results", f"slam/slam/eval/results/{backend_name}_ape.zip",
    ]


def build_evo_rpe_command(bag_path: str, backend_name: str) -> list[str]:
    return [
        "evo_rpe", "bag2", bag_path,
        "/ground_truth/odom", "/slam/odom",
        "--align",
        "--save_results", f"slam/slam/eval/results/{backend_name}_rpe.zip",
    ]


def run_evo_scoring(bag_path: str, backend_name: str) -> tuple[subprocess.CompletedProcess, subprocess.CompletedProcess]:
    ape_result = subprocess.run(build_evo_ape_command(bag_path, backend_name), check=True)
    rpe_result = subprocess.run(build_evo_rpe_command(bag_path, backend_name), check=True)
    return ape_result, rpe_result
```

- [ ] **Step 5:** Run → PASS.
- [ ] **Step 6:** Leave staged.

---

### Task 8: rerun Blueprint + real MP4 export (§9, corrected)

**Files:** Create `slam/slam/viz/{__init__,blueprint,test_blueprint}.py`, `slam/slam/viz/export_video.py`, `slam/slam/viz/bridge_launch/{lidar_camera,camera_only}.yaml`.

- [ ] **Step 1: Write a conservative failing test for the Blueprint** — v1's test asserted invented internal attributes (`root_container.contents`, double-calling `.to_blueprint()`); this version only asserts what's actually verifiable without guessing SDK internals:

```python
import rerun.blueprint as rrb

from slam.slam.viz.blueprint import build_comparison_blueprint


def test_build_comparison_blueprint_returns_a_blueprint_instance():
    blueprint = build_comparison_blueprint()
    assert isinstance(blueprint, rrb.Blueprint)


def test_build_comparison_blueprint_does_not_raise_when_called_repeatedly():
    # Determinism check that doesn't rely on an unverified __eq__/repr
    # contract on rerun's Blueprint object -- just confirms the function is
    # side-effect-free and safe to call once per candidate x rig ("defined once in code and reused
    # identically" requirement).
    for _ in range(3):
        assert isinstance(build_comparison_blueprint(), rrb.Blueprint)
```

- [ ] **Step 2:** Run to verify it fails. If `rerun-sdk` isn't yet a `pixi` dependency, add it first: `pixi add rerun-sdk`, then run: `pixi run pytest slam/slam/viz/test_blueprint.py -v` → FAIL (`ModuleNotFoundError: slam.slam.viz.blueprint`).

- [ ] **Step 3: Write `blueprint.py`**

```python
"""One rerun Blueprint layout, defined once and reused identically across all
six SLAM candidates and both sensor rigs: scene view |
camera feed | 3D map+trajectory | ATE-over-time plot.

NOTE: verify rrb.Spatial3DView/Spatial2DView/TimeSeriesView/Horizontal's exact
constructor signatures against the rerun-sdk version actually pinned in
pixi.toml before relying on this in a real run -- these are the well-known,
stable top-level blueprint types, but exact kwarg names have shifted across
rerun versions historically (this is the same class of risk v1's test got
wrong by asserting unverified internals; this file avoids that by keeping the
construction itself simple and letting Step 2's test catch an actual
constructor mismatch immediately, rather than asserting on structure that
might not even be the real API shape).
"""

from __future__ import annotations

import rerun.blueprint as rrb


def build_comparison_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="Scene", origin="/world"),
            rrb.Spatial2DView(name="Camera", origin="/showcase_camera"),
            rrb.Spatial3DView(name="Map + Trajectory", origin="/slam/map"),
            rrb.TimeSeriesView(name="ATE over time", origin="/eval/ate"),
        ),
        collapse_panels=True,
    )
```

- [ ] **Step 4:** Run → PASS. If it fails on a real constructor mismatch (not the placeholder-API kind v1 had, but a genuine version difference), fix the kwargs against the installed `rerun-sdk`'s actual signature — that's the test doing its job, not a plan defect.

- [ ] **Step 5: Write the bridge launch configs** (unchanged from v1 — confirmed real mechanism against `rerun-io/cpp-example-ros2-bridge`'s own docs):

`slam/slam/viz/bridge_launch/lidar_camera.yaml`:

```yaml
topics:
  - topic: /slam/odom
    entity_path: /slam/estimated_trajectory
  - topic: /ground_truth/odom
    entity_path: /world/ground_truth_trajectory
  - topic: /slam/map
    entity_path: /slam/map
  - topic: /showcase_camera/rgb
    entity_path: /showcase_camera
static_transforms:
  - frame_id: world
    child_frame_id: base_link_gt
```

`slam/slam/viz/bridge_launch/camera_only.yaml` — same, plus a `/rear_camera/rgb` → `/rear_camera` entry.

- [ ] **Step 6: Write the failing test for the real MP4 remux adaptation**

```python
from slam.slam.viz.export_video import remux_video_stream_to_mp4


def test_remux_video_stream_to_mp4_raises_a_clear_error_for_non_h264_codec(tmp_path):
    # Exercises the codec-check branch without needing a real .rrd fixture --
    # verifies the function fails loudly on the documented H.264-only
    # constraint rather than silently producing a broken mp4.
    import pytest

    with pytest.raises(FileNotFoundError):
        # No real .rrd exists at this path -- confirms the function actually
        # tries to load the recording rather than silently no-op'ing, since
        # a full real-.rrd fixture is a pod-side/rerun-runtime concern out of
        # scope for a pure unit test.
        remux_video_stream_to_mp4(rrd_path=str(tmp_path / "nonexistent.rrd"), video_entity="showcase_camera", output_mp4_path=str(tmp_path / "out.mp4"))
```

- [ ] **Step 7:** Run → FAIL (`ModuleNotFoundError`).

- [ ] **Step 8: Write `export_video.py`** — a faithful adaptation of rerun's own real, verified snippet (`rerun-io/rerun`'s `docs/snippets/all/archetypes/video_stream_query_and_mux.py`, confirmed live 2026-09-20), not an invented CLI:

```python
"""Two MP4 export paths:

1. remux_video_stream_to_mp4() -- adapted from rerun's own documented
   example, rerun-io/rerun's docs/snippets/all/archetypes/
   video_stream_query_and_mux.py (confirmed live 2026-09-20). Queries a
   VideoStream-logged entity out of an .rrd and remuxes it straight to .mp4
   with NO re-encoding, provided the stream is H.264. This only works on a
   single logged video stream (e.g. the showcase camera) -- it is NOT the
   composite 4-panel Blueprint video.
2. The full 4-panel Blueprint composite has no native export -- still needs
   screen-capturing the viewer (OBS / ffmpeg -f x11grab) or driving frames
   via the Python API and stitching with ffmpeg. See
   export_composite_manual_steps() for the documented manual procedure.

IMPORTANT: this adaptation is based on the fragments of rerun's real snippet
confirmed during this plan's research pass (its main()'s argparse shape, the
H.264 codec-check logic, and the demux/mux loop using PyAV) -- diff this
function against the live snippet file before relying on it in a real run,
since the full body of its recording-query helper was not completely
verified line-by-line.
"""

from __future__ import annotations

import io
from fractions import Fraction

import av
import rerun as rr


def remux_video_stream_to_mp4(rrd_path: str, video_entity: str, output_mp4_path: str, timeline: str = "time") -> None:
    recording = rr.dataframe.load_recording(rrd_path)
    view = recording.view(index=timeline, contents=f"/{video_entity}/**")
    df = view.select()

    codec_batch = df.select(f"/{video_entity}:VideoStream:codec").execute_stream().next()
    if codec_batch is None:
        raise ValueError(f"no video stream codec found at {video_entity} for timeline {timeline}")
    codec_value = codec_batch.to_pyarrow().column(0)[0][0].as_py()
    if codec_value != rr.VideoCodec.H264.value:
        raise ValueError(
            f"video stream at {video_entity} is not H.264 (got {hex(codec_value)}, "
            f"expected {hex(rr.VideoCodec.H264.value)}) -- only H.264 remuxes without re-encoding"
        )

    samples_batch = df.select(f"/{video_entity}:VideoStream:sample").execute_stream().next()
    times = samples_batch.column(0)  # timeline column, per the real snippet's (times, samples) return shape
    encoded_bytes = samples_batch.column(1)

    # PyAV's av.open() needs a path or a file-like object exposing .read(),
    # not a raw bytes object -- confirmed real bug caught by code-correctness
    # audit -- wrap in io.BytesIO.
    input_container = av.open(io.BytesIO(b"".join(chunk.as_py() for chunk in encoded_bytes)))
    input_stream = input_container.streams.video[0]
    output_container = av.open(output_mp4_path, mode="w")
    output_stream = output_container.add_stream_from_template(input_stream)

    start_time = times[0]
    for packet, time in zip(input_container.demux(input_stream), times, strict=False):
        packet.time_base = Fraction(1, 1_000_000_000)
        packet.pts = int(time.value - start_time.value)
        packet.dts = packet.pts
        packet.stream = output_stream
        output_container.mux(packet)

    output_container.close()


def export_composite_manual_steps() -> str:
    return (
        "1. Launch the rerun viewer with build_comparison_blueprint() applied.\n"
        "2. Replay the recorded bag through the ROS2 bridge (bridge_launch/*.yaml).\n"
        "3. Screen-capture the viewer window for the bag's duration, e.g.:\n"
        "   ffmpeg -f x11grab -video_size 1920x1080 -i :0.0 -t <duration> output.mp4\n"
        "   (or use OBS for a GUI-driven capture instead of ffmpeg directly)."
    )
```

- [ ] **Step 9:** Run → PASS (the test only exercises the load-failure path, which doesn't depend on the uncertain internals flagged in the docstring).
- [ ] **Step 10:** Leave staged.

---

### Task 9: §5's downstream map-conversion stage (previously silently dropped — now added)

**Why this exists:** the audit found §5's "separate, optional, shared downstream conversion stage (`octomap_server`/`pointcloud_to_grid`)" completely missing from v1, not even disclosed as out of scope. §5 explicitly frames this as generic across every point-cloud-producing backend, sitting **after** the contract, opt-in via config — so it consumes `cerberus_msgs/TaggedMap` from Task 6a/6c, not any backend-specific topic.

**Files:** Create `slam/slam/eval/map_conversion_launch.py`, `slam/slam/eval/test_map_conversion_launch.py`.

**Interfaces:** Consumes `/slam/map` (`cerberus_msgs/TaggedMap`, Task 6c) — note `TaggedMap.map_type` is currently always the literal string `"point_cloud"` (Task 6c's `SlamBackendAdapter._on_map` hardcodes it, since every registered candidate only ever produces point clouds per §5), so `select_conversion_tool`'s `source_map_type` argument is always `"point_cloud"` in practice today; the parameter stays general rather than hardcoded so a future backend producing a different native type doesn't require a signature change. Produces an opt-in launch configuration wiring `TaggedMap.point_cloud` into `octomap_server`/`pointcloud_to_grid` — never invoked automatically, only when a downstream consumer explicitly asks for a grid. **Corrected after audit:** `target_map_type` now only accepts `"occupancy_grid"` (the tool choice — `octomap_server` vs `pointcloud_to_grid` — is a separate parameter), matching §2's actual `map_type` enum (`point_cloud | occupancy_grid | pose_graph`) exactly; v1's `"elevation_grid"` target invented a fourth value that doesn't exist anywhere in the SLAM topic contract.

- [ ] **Step 1: Write the failing test for the pure decision logic**

```python
import pytest

from slam.slam.eval.map_conversion_launch import select_conversion_tool


def test_select_conversion_tool_defaults_to_pointcloud_to_grid():
    assert select_conversion_tool(source_map_type="point_cloud", target_map_type="occupancy_grid") == "pointcloud_to_grid"


def test_select_conversion_tool_can_request_octomap_server_instead():
    # §5 lists both tools as valid choices for the SAME point_cloud ->
    # occupancy_grid conversion -- not two different target map types.
    assert select_conversion_tool(source_map_type="point_cloud", target_map_type="occupancy_grid", tool="octomap_server") == "octomap_server"


def test_select_conversion_tool_grid_to_point_cloud_refuses():
    # §5: "No node converts *up* (grid -> point cloud)... Not recoverable."
    with pytest.raises(ValueError, match="not recoverable"):
        select_conversion_tool(source_map_type="occupancy_grid", target_map_type="point_cloud")


def test_select_conversion_tool_same_type_is_a_noop_passthrough():
    assert select_conversion_tool(source_map_type="point_cloud", target_map_type="point_cloud") is None


def test_select_conversion_tool_unknown_target_type_rejected():
    # Only point_cloud/occupancy_grid/pose_graph exist in §2's map_type enum --
    # anything else is a real bug to catch here, not silently accept.
    with pytest.raises(ValueError, match="expected 'occupancy_grid'"):
        select_conversion_tool(source_map_type="point_cloud", target_map_type="elevation_grid")
```

- [ ] **Step 2:** Run → FAIL.

- [ ] **Step 3: Write `map_conversion_launch.py`**

```python
"""Optional, shared downstream map-conversion stage:
sits AFTER the /slam/map contract topic, opt-in via config, for any consumer
needing a grid instead of raw points. Generic across every point-cloud-
producing backend registered in slam/slam/core/algorithms.py -- consumes
cerberus_msgs/TaggedMap, not any backend-specific topic, so it never needs
duplicating per backend. Adapters (Task 6c) never call this directly; a
downstream consumer opts in explicitly.

target_map_type only ever accepts "occupancy_grid" -- §2's actual map_type
enum is point_cloud | occupancy_grid | pose_graph, nothing else; the
CHOICE of tool (octomap_server vs pointcloud_to_grid) is a separate
parameter, not a different target type -- §5 names both as valid tools for
the same point_cloud -> occupancy_grid conversion.
"""

from __future__ import annotations


def select_conversion_tool(source_map_type: str, target_map_type: str, tool: str = "pointcloud_to_grid") -> str | None:
    """Returns the tool name to launch, or None if source == target (pure
    pass-through, no conversion node needed). Raises for the one direction
    §5 says is impossible: grid -> point cloud (not recoverable -- grid-cell
    centers only, no information gain)."""
    if source_map_type == target_map_type:
        return None
    if source_map_type != "point_cloud":
        raise ValueError(
            f"converting FROM {source_map_type!r} is not recoverable (§5: grid/pose-graph -> "
            f"point cloud loses information that can't be reconstructed) -- consume the native "
            f"point-cloud topic directly instead"
        )
    if target_map_type != "occupancy_grid":
        raise ValueError(f"unknown target_map_type {target_map_type!r}, expected 'occupancy_grid'")
    if tool not in ("pointcloud_to_grid", "octomap_server"):
        raise ValueError(f"unknown tool {tool!r}, expected 'pointcloud_to_grid' or 'octomap_server'")
    return tool
```

- [ ] **Step 4:** Run → PASS.

- [ ] **Step 5: Leave staged.** The actual `octomap_server`/`pointcloud_to_grid` ROS2 launch wiring (subscribing to `/slam/map`, extracting `TaggedMap.point_cloud`, feeding it to whichever tool `select_conversion_tool` names) is a straightforward launch-file composition once a consumer actually needs it — deferred to when that first real consumer exists, per this project's "decide/build just before it's needed" convention, same as the mission/scene set. The decision logic above is the part worth having now since it's what makes the opt-in behavior testable and correct.

---

### Task 10: Isaac Lab video capture + R2 sync + rerun sim logging for SLAM eval runs

**Why this exists:** added 2026-09-20 per Kevin's explicit request — SLAM eval runs need the same "capture video, sync to R2" discipline `policy/locomotion` already has, plus the rerun Blueprint's "Scene" panel needs to show the *actual simulation*, not just bridged ROS topics.

**Research finding that simplifies part (b):** `scripts/sync_to_r2.py` is **already fully generic across modules** via its `--module` flag (confirmed by reading its own docstring/header — "R2 layout mirrors the local layout under a per-module prefix... reusable for future phases beyond locomotion, not locomotion-specific"). **No code change to `sync_to_r2.py` is needed** — `--module slam` already works today. This task only adds the SLAM-side scripts that *call* it correctly, mirroring `scripts/run_remote_job.sh`'s periodic/final-sync pattern.

**Files:**
- Create: `slam/slam/eval/showcase_camera_video.py`, `slam/slam/eval/test_showcase_camera_video.py`
- Create: `slam/slam/viz/sim_scene_logging.py`, `slam/slam/viz/test_sim_scene_logging.py`
- Modify: `slam/slam/eval/record_mission_bag.py` (Task 7) — call R2 sync after each run

**Interfaces:** Consumes `SHOWCASE_CAMERA_CFG`/`with_showcase_camera` (Task 5), `scripts/sync_to_r2.py`'s existing `sync-run`/`sync-eval` subcommands (unmodified). Produces `showcase_camera_video.build_frame_writer_path(backend_name, mission, run_timestamp) -> Path`; `sim_scene_logging.log_sim_frame(rr_recording, timestamp, camera_rgb_frame, robot_transform)` (pure-ish, mockable rerun calls); `record_mission_bag.py` gets an R2 sync call appended.

- [ ] **Step 1: Write the failing test for the video-capture path-naming logic**

```python
from pathlib import Path

from slam.slam.eval.showcase_camera_video import build_frame_writer_path


def test_build_frame_writer_path_matches_r2_sync_run_dir_shape():
    # Must land under a path scripts/sync_to_r2.py's sync-run can walk as one
    # run directory (module=slam), same shape as
    # policy/locomotion/checkpoints/<experiment>/<run_timestamp>/videos/ --
    # per docs/artifact_storage.md's mirrored-layout requirement.
    path = build_frame_writer_path(backend_name="fast_lio2", mission="household", run_timestamp="2026-09-20_10-00-00")
    assert path == Path("slam/slam/eval/results/fast_lio2/2026-09-20_10-00-00/videos/household_showcase.mp4")


def test_build_frame_writer_path_differs_per_backend_and_run():
    a = build_frame_writer_path(backend_name="fast_lio2", mission="household", run_timestamp="2026-09-20_10-00-00")
    b = build_frame_writer_path(backend_name="orb_slam3", mission="household", run_timestamp="2026-09-20_10-00-00")
    assert a != b
```

- [ ] **Step 2:** Run → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `showcase_camera_video.py`**

```python
"""Isaac Lab video capture for SLAM eval runs, mirroring
policy/locomotion/core/video_capture.py's "write frames directly via imageio"
convention -- but sourcing frames from the showcase camera's CameraCfg sensor
output (Task 5's SHOWCASE_CAMERA_CFG) each physics step, not from
viewport_camera_controller (that mechanism is Phase-0-specific: it watches
one fixed env's viewport for a run's whole lifetime, which doesn't apply
here -- the showcase camera is a real moving sensor prim, per §8's soft-follow
chase cam, not the shared viewport).

Actual per-step frame-writing loop (env.step() -> read the showcase camera's
rgb tensor -> imageio.append_data(...)) needs a live Isaac Lab env and is a
pod-side integration step, out of scope here -- same constraint as
ground_truth/pose_export.py and showcase_camera_follow.py. This module only
contains the locally-testable path-naming logic that keeps output aligned
with scripts/sync_to_r2.py's expected run-directory shape.
"""

from __future__ import annotations

from pathlib import Path


def build_frame_writer_path(backend_name: str, mission: str, run_timestamp: str) -> Path:
    """Mirrors docs/artifact_storage.md's local-layout-mirrors-R2 rule:
    slam/slam/eval/results/<backend_name>/<run_timestamp>/videos/<mission>_showcase.mp4
    so `sync_to_r2.py sync-run --module slam --run-dir
    slam/slam/eval/results/<backend_name>/<run_timestamp>` uploads it under
    videos/ automatically, same shape as policy/locomotion's own runs."""
    return Path("slam/slam/eval/results") / backend_name / run_timestamp / "videos" / f"{mission}_showcase.mp4"
```

- [ ] **Step 4:** Run → PASS.

- [ ] **Step 5: Wire an R2 sync call into `record_mission_bag.py` (Task 7)**

Append to `slam/slam/eval/record_mission_bag.py`'s `main()`, after the bag-recording block, before `return record_proc.returncode`:

```python
    sync_cmd = ["python3", "scripts/sync_to_r2.py", "sync-run", "--module", "slam", "--run-dir", str(run_dir_for(args.backend, args.run_timestamp))]
    print(f"[record_mission_bag] syncing to R2: {' '.join(sync_cmd)}")
    subprocess.run(sync_cmd, check=False)  # per run_remote_job.sh's own pattern: warn, don't crash the run, on a sync failure
```

Reuses `run_dir_for()` (already defined in Task 7's Step 1, the same function that determined the bag's own output path) rather than recomputing the path independently — this is the exact audit-found gap Task 7's fix closed: one function, one source of truth, so R2 always syncs from wherever the bag was actually written. `--run-timestamp` is already a required argument as of Task 7's corrected version above, and `Path` is already imported there — no further argparse/import changes needed. This calls `sync_to_r2.py` completely unmodified — confirming the "no code change needed" finding above.

- [ ] **Step 6: Write the failing test for the sim-scene rerun logging's pure argument-shaping logic**

```python
from slam.slam.viz.sim_scene_logging import build_sim_frame_log_calls


def test_build_sim_frame_log_calls_includes_camera_and_transform_entities():
    calls = build_sim_frame_log_calls(timestamp_ns=1_000_000_000, camera_rgb_shape=(720, 1280, 3), robot_position=(1.0, 2.0, 0.3))
    entity_paths = [c["entity_path"] for c in calls]
    assert "/world/showcase_camera" in entity_paths
    assert "/world/robot" in entity_paths


def test_build_sim_frame_log_calls_uses_the_given_timestamp_for_every_entry():
    calls = build_sim_frame_log_calls(timestamp_ns=42, camera_rgb_shape=(720, 1280, 3), robot_position=(0.0, 0.0, 0.0))
    assert all(c["timestamp_ns"] == 42 for c in calls)
```

- [ ] **Step 7:** Run → FAIL.

- [ ] **Step 8: Write `sim_scene_logging.py`**

```python
"""Logs the actual Isaac Sim scene (showcase-camera frames + robot transform)
into rerun alongside the bridged SLAM topics, so the Blueprint's "Scene"
panel (slam/slam/viz/blueprint.py) shows the real simulation instead of an
empty /world origin. Structured as a pure function returning the log CALLS
to make (entity path, timestamp, and a description of the payload) rather
than calling rr.log() directly, so the entity-path/timestamp wiring is
unit-testable without a live rerun recording session -- the thin wrapper
that actually calls rr.log() with these needs a live Isaac Lab env (to get
real camera frames/robot poses) and is a pod-side step, out of scope here.
"""

from __future__ import annotations

from typing import Any


def build_sim_frame_log_calls(
    timestamp_ns: int,
    camera_rgb_shape: tuple[int, int, int],
    robot_position: tuple[float, float, float],
) -> list[dict[str, Any]]:
    """Returns a list of {"entity_path", "timestamp_ns", "kind", ...payload
    description} dicts. A thin caller does:
        for call in build_sim_frame_log_calls(...):
            rr.set_time_nanos("sim_time", call["timestamp_ns"])
            if call["kind"] == "image":
                rr.log(call["entity_path"], rr.Image(actual_rgb_array))
            elif call["kind"] == "transform":
                rr.log(call["entity_path"], rr.Transform3D(translation=call["translation"]))
    -- this function only decides WHICH entities get logged and WHEN, not
    the actual pixel/transform payload, which only exists inside a live env.
    """
    height, width, channels = camera_rgb_shape
    return [
        {
            "entity_path": "/world/showcase_camera",
            "timestamp_ns": timestamp_ns,
            "kind": "image",
            "expected_shape": (height, width, channels),
        },
        {
            "entity_path": "/world/robot",
            "timestamp_ns": timestamp_ns,
            "kind": "transform",
            "translation": robot_position,
        },
    ]
```

- [ ] **Step 9:** Run → PASS.
- [ ] **Step 10:** Leave staged.

---

## Final verification (run once, after all ten tasks)

- [ ] `pixi run pytest` — PASS, including every new test alongside every pre-existing one.
- [ ] `pixi run lint` — no new violations.
- [ ] `pixi run colcon build --symlink-install` — success, including `cerberus_msgs` and `slam`.
- [ ] `git log --oneline -5` — identical to before this plan started.
- [ ] `git status` — every file from File Structure shows as new/modified; `third_party/` does NOT appear.

---

## Explicitly out of scope, still (real gaps, not fixed by this rewrite)

- Actually wiring `GroundTruthPublisher.publish_pose()` and `compute_chase_cam_pose()`'s output into live Isaac Lab per-physics-step callbacks — needs a pod.
- Mission/scene definitions beyond `"household"` — blocked on the still-open Phase 1 scene-set decision.
- Confirming or dropping the "rerun ROS2 bridge ships a Spot-quadruped example" claim in `perception.md`.
- LVI-Q's actual algorithm implementation.
- `Dockerfile.remote` still has no ROS2 installed — raw sensor topics referenced in Task 7's `record_mission_bag.py` won't actually exist on a pod until that's added, a separate real change this plan doesn't make.
- **`joint_state`'s `SensorSuiteCfg` schema question is still open for Kevin to decide** (Global Constraints) — this plan proceeds with "undeclared" as its working default, not as a settled answer.
- `export_video.py`'s `remux_video_stream_to_mp4` is adapted from partially-verified fragments of rerun's real snippet (confirmed: argparse shape, codec-check logic, demux/mux loop; not fully confirmed: the exact dataframe-query column layout inside the recording-load step) — diff against the live snippet file before a real run, per that function's own docstring warning.
- `blueprint.py`'s exact `rrb.*` constructor kwargs are not pinned against a specific installed `rerun-sdk` version — Task 8 Step 4 catches a real mismatch via test failure rather than the plan asserting false confidence about unverified API surface.
