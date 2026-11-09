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

`HeightScan` (H3): `Header header`, `uint32 rows`, `uint32 cols`, `float32 resolution`,
`float32[] data` — a row-major grid of relative heights around the robot base, in the convention the
rough policy was trained on (Isaac Lab `height_scan`; see the contract below). The convention is
**unverified** against the live term, so producers must emit it knowingly; the policy cannot detect a
mismatch.

`BodyCommand`: `Header header`, `LocoCommand loco`, `ArmCommand arm`, `float32 dt`. The IIT Go2 VLA 11-D command (twist 3 + pitch 1 + 6 arm joints + gripper 1) is a one-step chunk: `loco.{vx,vy,wz}`, `loco.body_pitch` with `has_body_pitch=true`, `arm.mode=JOINT_POSITION` with `num_joints=6` and 6 `joint_positions`, and one `gripper` value. A test (`test_iit_11d_command_is_a_one_step_chunk`) pins this.

Python code passes the same fields as plain dicts (`base.LOCO_COMMAND_KEYS`, `ARM_COMMAND_KEYS`, `BODY_COMMAND_KEYS`, `check_command`); headers are added by the ROS adapter.

## B1 rate limiting (H3)

`LocoCommand` is accepted at 5–50 Hz, and nothing upstream is trusted to be smooth. `CommandRateLimiter`
(`common/cerberus_policies/command_limiter.py`, pure Python) sits between the command input and a
locomotion backend: per-axis `max_rate` (vx, vy in m/s²; wz in rad/s²) plus per-axis `max_abs` (the
training command ranges, ±1.0 m/s and ±1.0 rad/s from the saved GA params), applied over the caller's
`dt`. `body_pitch` / `body_height` pass through clamped only. `reset()` forgets the previous command;
the next `apply` adopts its clamped target rather than ramping from zero.

Why it exists: the Phase 0.5 adversary breaks the candidate through this channel (pinned at the clamp,
sign flips ~3.4/s). **The limiter's effect is untested** — the shipped values in
`common/cerberus_policies/config/command_limiter.yaml` are placeholders, not tuning, and no claim is
made that they pass the gate. It is unrelated to `adversarial_robustness/attack_math.RateLimitedChannel`
(which bounds the adversary per sim step); the gate artifacts stay reproducible and unchanged.

Cards may record the limiter with the optional fields `max_command_rate` (`{axis: {value, unit}}`) and
`command_limiter_applied` (bool); the schema documents both, `unknown` is allowed only at L0.

## Policy backends (`common/cerberus_policies/`)

Mirrors `common/cerberus_worlds/`: a lazy `module:Class` registry, selection by config, no silent default, and nobody imports a backend module directly (an AST-scan test enforces it).

- `base.PolicyBackend` ABC: class attributes `boundary` (`Boundary.B1_LOCO | B2_ARM | B3_BODY`), `role` (`CONSUMER` default, or `PRODUCER`), `required_sensors` (sensor ids), `rate_hz`; abstract `load(checkpoint_path, embodiment_cfg)` and `act(observation) -> dict`. Missing or invalid attributes fail at class-definition time. No torch/numpy/ROS/Isaac import at module level.
- `registry.get_backend(slot, name=None, config=None)`: explicit name, else `LOCO_BACKEND` / `ARM_BACKEND` / `BODY_BACKEND`, else `ValueError` listing what is available. `config` kwargs build a fresh, uncached instance.
- `validate_config.validate_config(backends, sensor_suite, external_producers, external_consumers)`: one producer and one consumer per boundary in use, and `required_sensors` a subset of the sensor suite (`sensor_suite_from_profile("lidar_camera")` reads `system/cerberus_bringup/config/sensor_profiles.yaml`). Reports every problem at once.
- `backend_card` + `docs/backend_card.schema.yaml` + `cards/*.yaml`: validated cards; see below.

### The first real backend: `heightscan_ppo` (H3)

- `contracts/heightscan_ppo_v1.yaml`: the observation/action/control contract for the rough policy.
  Term order and dims from the GA run's saved params: `base_lin_vel`(3), `base_ang_vel`(3),
  `projected_gravity`(3), `velocity_commands`(3), `joint_pos`(12), `joint_vel`(12), `actions`(12),
  `height_scan`(187) = **235**; action 12, scale 0.25, default offsets by name, decimation 4 at
  dt 0.005 (50 Hz), command ranges ±1 m/s and ±1 rad/s. The height-scan sign/offset is marked
  `unverified` and is deferred to a pod check.
- `backends/heightscan_ppo_onnx.py`: loads the `.onnx` graph plus its external `<name>.onnx.data`
  weights (a clear error when the sibling is missing), runs a **fixed batch of 1 row by row**,
  validates a 235-vector (or assembles named terms in contract order), and maps raw actions to joint
  targets by name (`default + scale * action`). onnxruntime/numpy are imported lazily.
- `command_limiter.py` sits in front of it at B1 (see above).

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

Validation: L1 and above need non-empty `evidence`; **levels are cumulative — L2 and L3 also need the L1 evidence**, a `file` entry for the adapter under `common/cerberus_policies/backends/` and one for a `test_*.py` contract test; L2 and L3 need a `results` entry and a `report` evidence entry; L3 also needs a result with `trained_by_us: true`; `unknown` placeholders are allowed only at L0; `target_support_level` must be at least `support_level`. So a backend with good training numbers but no adapter is L0, not L2/L3. Evidence entries are `{kind: file|report|commit, ref}`, and `file` refs must exist on disk (checked in the all-cards test).

## Design choices (where the spec was silent)

1. **Card slots match registry slots** (`loco`, `arm`, `body`), and a card's `boundary` must agree with its slot.
2. **Producer vs consumer is explicit** (`role`). Real policies (JEPLO, the height-scan PPO and the flat blind PPO) are consumers of B1; `null_loco` and `scripted_loco` are B1 *producers* (command sources), which is how "returns zero velocity" / "constant command from config" is read. They live in the `loco` slot because the slot is keyed by boundary.
3. **Architecture for a blind policy is `modular`**: it has no trained perception encoder inside the policy. The height-scan policy is also `modular`, conditioning on a simulator-provided scan (paradigm B with a privileged map), not on raw sensors.
4. **`joint_encoders` is an always-available sensor id** alongside the rig ids in `sensor_profiles.yaml` (lidar, front_camera, rear_camera, wrist_camera, imu). **`height_scan` is an upstream module output**, not a rig sensor (the sim ray-caster or terrain perception produces it), so `validate_config` treats it as available while any other unlisted id still has to be on the rig.
5. **The sensor check is standalone**, not the bringup validator, because `cerberus_bringup` is only importable once colcon-built; TODO to unify (see `validate_config.py`).
6. **Endpoints that are not policy backends** (a ROS planner, the actuation layer) are declared through `external_producers` / `external_consumers`.
7. **Evidence is typed** (`file|report|commit`) so level rules can be checked mechanically; `unknown` is a literal string permitted only on `required_sensors`, `rate_hz`, `license`, `checkpoint_url`, `code_url` at L0.
8. **L0 cards cite `modular_stack_options.md` as their source** and carry `rate_hz: 50` (the doc states all locomotion paradigms emit leg targets at ~50 Hz); anything the doc does not state is `unknown`.
