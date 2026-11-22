# Platform and embodiment configuration

How "which robot" is selected on the training side and on the deployment side, how the two stay in sync, and what to add
for a new robot or a new arm.

The rule behind all of it: platform identity is configuration, never code. A component that needs to know the robot reads
it from a config file or from `/robot_description` and TF, and never hardcodes a joint list, a URDF or a checkpoint.

## The three surfaces

The string `go2` (and later `go2_arx5`) has to mean the same thing in three independent places:

| surface | where | used for |
|---|---|---|
| training | `policy/locomotion/core/platforms/<name>/platform.py`, a `PlatformCfg` instance registered as `go2` | gym task ids, environment configs, push curriculum, checkpoint prefix |
| deployment | `system/cerberus_bringup/config/platforms.yaml`, the top-level key `go2` | URDF path, locomotion modes and their checkpoints, manipulation checkpoint |
| the command line | `--platform go2` (training) and `platform:=go2` (launch) | selects either of the above |

A test loads `platforms.yaml` and checks every top-level key against the platform names the training side knows, so a
typo or an orphaned entry fails in CI instead of at launch.

## Training side

`policy/locomotion/core/platforms/registry.py` defines the contract:

```python
@dataclass(frozen=True)
class PlatformCfg:
    name: str
    flat_task_id: str; flat_play_task_id: str
    rough_task_id: str; rough_play_task_id: str
    flat_env_cfg_entry_point: str; rough_env_cfg_entry_point: str
    flat_agent_cfg: Any; rough_agent_cfg: Any
    push_peak_force_n: float; push_peak_torque_nm: float
    root_body_name: str = "base"
    stairs_step_height_scale: float = 1.0
    num_steps_per_env: int = 24
    checkpoint_prefix: str = ""          # defaults to "<name>_locomotion"
```

`policy/locomotion/__init__.py` discovers platform packages instead of importing them one by one: it calls
`import_packages("<module>.core.platforms", ["registry"])`, which walks and imports every subpackage; `registry` is
blacklisted because it holds the contract, not a platform. Each platform's `platform.py` registers itself at import time,
so adding a robot touches no existing file.

Task ids, environment configs and agent configs are platform data. The push curriculum is too: its peak force and torque
are tuned per robot and must be re-derived for a new one, not copied.

## Deployment side

`system/cerberus_bringup/config/platforms.yaml` holds selection and composition, which is why it lives in the launch
package rather than in `common/cerberus_description/` (that package is shared, dependency-only, no standalone behaviour).
Same shape as Autoware's `vehicle_model` launch argument.

```yaml
go2:
  urdf_relative_path: urdf/go2.urdf.xacro
  joint_names: [FL_hip_joint, ...]          # training-time order, not URDF order
  default_joint_pos: [0.1, 0.8, -1.5, ...]  # one per joint, same order
  locomotion_modes:
    blind:
      checkpoint: .../go2_locomotion_flat_v1_candidate.onnx
      observation_shape: proprioception_only
    perceptive:
      checkpoint: .../go2_locomotion_v1_candidate.onnx
      observation_shape: proprioception_plus_height_scan
  default_locomotion_mode: blind
```

Joint values are remapped by name, never by position: the trained articulation order is not the URDF order, and a test
keeps `joint_names` in sync with the policy's contract file in `common/cerberus_policies/contracts/`.

A locomotion mode declares its own observation shape, so the node never hardcodes "this policy needs a height scan".

- `proprioception_only`: base linear and angular velocity, projected gravity, velocity command, relative joint
  positions, relative joint velocities, last action. Needs IMU, joint encoders and a command topic.
- `proprioception_plus_height_scan`: the same seven terms with `height_scan` appended last, matching training's
  concatenation order. Until terrain perception exists, the scan is a constant flat-terrain stub, logged as such at
  start-up: legitimate for bring-up, not a validated sensor deployment.

Both modes run the same node; only the observation assembly and the ONNX file differ, and the ONNX file is what the
runtime loads (no Isaac Lab or RSL-RL at inference). Training-time noise is off at inference, actions are
`JointPositionAction` with scale 0.25 and the default offset, so the network output is scaled and added to each joint's
default position before it is published as a position command.

Launch:

```bash
ros2 launch cerberus_bringup bringup.launch.py platform:=go2 locomotion_mode:=blind
```

The launch file resolves the xacro path for `robot_state_publisher` and the checkpoint and observation-shape parameters
for `locomotion_policy`, and starts `manipulation_policy` only when the resolved block has a manipulation checkpoint.

Only three things in the ROS2 graph are platform-aware: `cerberus_bringup` (the composer), `locomotion_policy` and
`manipulation_policy` (one checkpoint path each). Terrain and object perception, SLAM, the planner, the orchestrator,
the safety filter and actuation get robot identity from `/robot_description` and TF and need no change when the platform
changes.

## Adding a robot

1. `policy/locomotion/core/platforms/<name>/platform.py` with a `PlatformCfg` (task ids, env config entry points, agent
   configs, push force and torque, root body, terrain scale) and `env_cfg.py` for the environments.
2. Re-derive the push curriculum for the new mass and geometry: anchoring to another robot's numbers is the mistake the
   Go2 values exist to avoid.
3. A `locomotion_modes` block under the same name in `platforms.yaml`, with the URDF and the joints in training order.
4. Train, evaluate and gate the checkpoint on this robot before it becomes a deployment mode (see
   `docs/adversarial_gate_method.md`).

## Adding an arm

The Go2 + ARX5 embodiment is a composition, not a new robot:

1. Merge the assets (`go2_arx5.urdf.xacro` or the composed USD, see `docs/embodiment_asset_composition.md`). A single
   robot needs no merge, which is why the Go2 slice came first.
2. Add the `go2_arx5` block to `platforms.yaml` with the merged URDF, the extended joint list and a
   `manipulation_checkpoint` for the arm policy.
3. Keep the arm's sensors independent of the navigation rig choice (see `docs/sensor_setup.md`), and settle the rear
   camera mount the arm displaces.
4. Do not assume the legs-only checkpoint is valid on the combined body: the arm adds about 3.35 kg and shifts the centre
   of mass. Measure the transfer first (standing, commands, push recovery with the arm stowed), then re-run the
   adversarial gate on the combined embodiment before calling it robust.

Arm policy options, and the reason the current choice is task-space control: see `docs/worlds_integration_spec.md` and
`docs/manipulator.md`.

## Open decisions

- Whether `manipulation_checkpoint` stays a single path or becomes a block of candidate checkpoints (BC, diffusion, ACT)
  depends on the manipulation evaluation work.
- The exact xacro authoring for the combined body depends on the asset merge running first.
- A third observation shape (onboard encoder or depth input) would add an observation-assembly function and a new
  `observation_shape` value; it is not a node restructure.
