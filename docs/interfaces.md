# Cerberus — Interfaces: three command boundaries and the policy backend layer

Frozen in H1 (2026-11-06). Spec source: `modular_stack_options.md`, "Design principle: three command boundaries". Messages are in `common/cerberus_msgs/msg/`; the Python layer is `common/cerberus_policies/`. Swappability comes from freezing the messages between layers, not from picking one algorithm.

## The three boundaries

| Boundary | Message | Rate | Producers | Consumers |
|---|---|---|---|---|
| B1 locomotion command | `LocoCommand` | 5–50 Hz | global/local planner, navigation policy, VLA base head, scripted sources (`null_loco`, `scripted_loco`) | any locomotion backend: blind, map-based, depth E2E, world-model, raw-LiDAR |
| B2 arm command | `ArmCommand` | 5–30 Hz | grasp planner + cuRobo, BC / Diffusion Policy / ACT, VLA | arm IK / joint trajectory controller, or a whole-body controller |
| B3 body command | `BodyCommand` (B1 + B2) | 5–10 Hz (chunked) | whole-body IL policy, loco-manip VLA | unified whole-body controller |

Rules: (1) backends declare their boundary and sensor needs, and a config is valid only if every boundary in use has exactly one producer and one consumer; (2) the safety filter sits on B1/B2/B3 and on joint targets, so it is paradigm-agnostic. A joint-level end-to-end VLA would bypass B1–B3 and is a non-goal.

## Messages

`LocoCommand`: `Header header`, `float32 vx, vy, wz` (base twist, base frame), `float32 body_pitch`, `float32 body_height`, `bool has_body_pitch`, `bool has_body_height`. The optional fields are valid only when their flag is true (flags, not NaN sentinels).

`ArmCommand`: `Header header`, `uint8 mode` (`EE_POSE=0`, `EE_TRAJECTORY=1`, `JOINT_POSITION=2`), `geometry_msgs/Pose[] ee_poses` (one entry for `EE_POSE`, one per step for `EE_TRAJECTORY`), `float32[] joint_positions` flattened row-major `[step][joint]` with `uint32 num_joints`, `float32[] gripper` (one per step), `float32 dt` (seconds between chunk steps). A single command is a one-step chunk.

`BodyCommand`: `Header header`, `LocoCommand loco`, `ArmCommand arm`, `float32 dt`. The IIT Go2 VLA 11-D command (twist 3 + pitch 1 + 6 arm joints + gripper 1) is a one-step chunk: `loco.{vx,vy,wz}`, `loco.body_pitch` with `has_body_pitch=true`, `arm.mode=JOINT_POSITION` with `num_joints=6` and 6 `joint_positions`, and one `gripper` value. A test (`test_iit_11d_command_is_a_one_step_chunk`) pins this.

Python code passes the same fields as plain dicts (`base.LOCO_COMMAND_KEYS`, `ARM_COMMAND_KEYS`, `BODY_COMMAND_KEYS`, `check_command`); headers are added by the ROS adapter.

## Policy backends (`common/cerberus_policies/`)

Mirrors `common/cerberus_worlds/`: a lazy `module:Class` registry, selection by config, no silent default, and nobody imports a backend module directly (an AST-scan test enforces it).

- `base.PolicyBackend` ABC: class attributes `boundary` (`Boundary.B1_LOCO | B2_ARM | B3_BODY`), `role` (`CONSUMER` default, or `PRODUCER`), `required_sensors` (sensor ids), `rate_hz`; abstract `load(checkpoint_path, embodiment_cfg)` and `act(observation) -> dict`. Missing or invalid attributes fail at class-definition time. No torch/numpy/ROS/Isaac import at module level.
- `registry.get_backend(slot, name=None, config=None)`: explicit name, else `LOCO_BACKEND` / `ARM_BACKEND` / `BODY_BACKEND`, else `ValueError` listing what is available. `config` kwargs build a fresh, uncached instance.
- `validate_config.validate_config(backends, sensor_suite, external_producers, external_consumers)`: one producer and one consumer per boundary in use, and `required_sensors` a subset of the sensor suite (`sensor_suite_from_profile("lidar_camera")` reads `system/cerberus_bringup/config/sensor_profiles.yaml`). Reports every problem at once.
- `backend_card` + `docs/backend_card.schema.yaml` + `cards/*.yaml`: validated cards; see below.

## Modular vs fused backends

Both plug in at the same boundary (B1 for locomotion); they differ in what `act` reads from the observation.

- **Modular**: consumes a map or other upstream module output (elevation map, height scan). Example: `height_map_miki`.
- **Fused**: consumes raw sensor streams, with the perception encoder trained jointly inside the policy (JEPLO on raw LiDAR, DAWN/WMP on depth, Omni-Perception). It lists the raw streams in `required_sensors` and reads them from `observation["sensors"][sensor_id]`. The ABC does not assume a pre-processed map.

Observation convention (not enforced): `{"t", "proprio": {...}, "sensors": {sensor_id: raw}, "command": <dict at this boundary>}`. Training happens in each backend's own environment; deployment loads an exported ONNX/TorchScript policy behind the boundary.

## Support levels

Every backend has a card in `common/cerberus_policies/cards/`. `support_level` is what the evidence proves today; `target_support_level` is ambition. The generated table is `backend_support.md` (`pixi run backend-table`; a test fails when it is stale).

| Level | Meaning | Evidence required |
|---|---|---|
| L0 | Documented | Backend card only, no code |
| L1 | Integrated | Adapter exists and passes contract tests (loads, `act()` returns correct shapes, sensor needs validate against a rig) |
| L2 | Evaluated | Run on our scenes or shared evals with released weights or without training; numbers recorded |
| L3 | Trained by us | Trained and evaluated by us; result recorded in a report |

Validation: L1 and above need non-empty `evidence`; L2 and L3 need a `results` entry and a `report` evidence entry; L3 also needs a result with `trained_by_us: true`; `unknown` placeholders are allowed only at L0; `target_support_level` must be at least `support_level`. Evidence entries are `{kind: file|report|commit, ref}`, and `file` refs must exist on disk (checked in the all-cards test).

## Design choices (where the spec was silent)

1. **Card slots match registry slots** (`loco`, `arm`, `body`), and a card's `boundary` must agree with its slot.
2. **Producer vs consumer is explicit** (`role`). Real policies (JEPLO, blind PPO) are consumers of B1; `null_loco` and `scripted_loco` are B1 *producers* (command sources), which is how "returns zero velocity" / "constant command from config" is read. They live in the `loco` slot because the slot is keyed by boundary.
3. **Architecture for a blind policy is `modular`**: it has no trained perception encoder inside the policy.
4. **`joint_encoders` is an always-available sensor id** alongside the rig ids in `sensor_profiles.yaml` (lidar, front_camera, rear_camera, wrist_camera, imu).
5. **The sensor check is standalone**, not the bringup validator, because `cerberus_bringup` is only importable once colcon-built; TODO to unify (see `validate_config.py`).
6. **Endpoints that are not policy backends** (a ROS planner, the actuation layer) are declared through `external_producers` / `external_consumers`.
7. **Evidence is typed** (`file|report|commit`) so level rules can be checked mechanically; `unknown` is a literal string permitted only on `required_sensors`, `rate_hz`, `license`, `checkpoint_url`, `code_url` at L0.
8. **L0 cards cite `modular_stack_options.md` as their source** and carry `rate_hz: 50` (the doc states all locomotion paradigms emit leg targets at ~50 Hz); anything the doc does not state is `unknown`.
