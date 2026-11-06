# Cerberus — Sensor Setup Handoff (coding-agent reference)

> **Status: locked for Phase 1 implementation**, unlike `manipulator.md`. The
> two rigs below (LiDAR+camera primary, camera-only ablation) are the actual
> configs to implement, not a research placeholder. The wrist/arm camera
> section is the exception — it's Phase 2 and stays a placeholder until that
> phase's own research pass, per the project's normal just-in-time rule.
> Sourcing/reasoning behind these decisions lives in the chat history and is
> summarized in `perception.md`'s Sensor Modalities table; this doc is
> implementation detail only, not justification.

## 1. The two rigs

| Sensor | LiDAR+camera (primary) | Camera-only (ablation) |
|---|---|---|
| LiDAR | ✅ spinning, `RayCasterCfg` | ❌ absent |
| Front stereo camera | ✅ | ✅ |
| Rear stereo camera | ❌ absent | ✅ |
| Wrist RGB-D (Phase 2+) | ✅ | ✅ (both rigs get it — arm sensing is independent of the nav sensor choice) |

Everything below is written so that switching between these two rigs — or a
future third rig a fork adds — is a **config + launch-arg change**, never a
code change in `slam/`, `planning/`, or `policy/`.

## 2. Isaac Lab sensor configs

### 2.1 LiDAR (primary rig only)

```python
lidar = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/lidar_cage",
    update_period=1 / 10,  # 10 Hz, matches Livox Mid-360 real hardware rate
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),  # near stock Go2 L1 mount height
    mesh_prim_paths=["/World/ground"],  # extend per-scene: add scene mesh prims from `cerberus_worlds` (docs/worlds.md) once Phase 1/2 scenes are live
    attach_yaw_only=True,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=100,
        vertical_fov_range=[-25, 15],   # asymmetric, biased downward — near-field/underbody coverage
        horizontal_fov_range=[-180, 180],
        horizontal_res=1.0,
    ),
    debug_vis=False,
)
```

Known gotcha (confirmed via GitHub issue search): `RayCasterCfg` has no
`spawn` field the way `CameraCfg` does, so it can't auto-spawn a prim before
the robot exists. Use the `prim_path="{ENV_REGEX_NS}/Robot/base"` pattern
directly (no separate lidar prim needed) rather than trying to pre-spawn a
cage prim — this is the workaround the IsaacLab community discussion landed
on for this exact Go2+LiDAR case.

`mesh_prim_paths` needs to be extended per scene — it's a static list of
meshes the LiDAR raycasts against, not a live scene query, so household scene
meshes (`cerberus_worlds` scene USDs, see docs/worlds.md) must be added explicitly once Phase 1's nav scenes
are wired up. Flag this as a thing that will silently return "no hits" if
forgotten, not an error — worth a startup sanity check in `slam/` bringup.

### 2.2 Front stereo camera (both rigs)

```python
front_camera_left = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/front_cam_left",
    update_period=1 / 30,
    height=480, width=640,
    data_types=["rgb", "depth"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.32, -0.06, 0.05),  # ~12cm stereo baseline, half-offset each side
        rot=(0.5, -0.5, 0.5, -0.5),  # ~20° downward tilt, ros convention
        convention="ros",
    ),
)
front_camera_right = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/front_cam_right",
    # identical except pos=(0.32, +0.06, 0.05)
    ...
)
```

Isaac Lab depth here is privileged ground-truth (`distance_to_image_plane`),
not reconstructed via actual stereo disparity matching — so in sim, "stereo"
mainly matters if downstream SLAM code expects two image streams to compute
disparity itself (e.g. a from-scratch ORB-SLAM3-style stereo frontend). If
you're only ever consuming the depth channel directly, a single `CameraCfg`
with `data_types=["rgb", "depth"]` gives equivalent information more cheaply
— **use two only if the SLAM candidate you actually implement wants raw
stereo pairs**, not by default. Note this in whichever SLAM package gets
implemented first.

**Cost note:** RGB rendering (TiledCamera or Camera with `"rgb"` in
`data_types`) is the expensive sensor in this whole rig — expect this to cap
parallel env count well before LiDAR does (see `isaac_lab_workflow.md`
compute notes). For Phase 1 work that only needs depth (terrain
understanding, obstacle avoidance), consider `RayCasterCamera` instead of
`CameraCfg`/`TiledCamera` — same Warp-based cheap raycasting as the LiDAR,
depth-only, no RTX render cost. Save `TiledCamera`'s real RGB for when
Phase 2 object perception actually needs it.

### 2.3 Rear stereo camera (camera-only ablation only)

Identical config to 2.2, mirrored: `pos=(-0.32, ±0.06, 0.05)`,
`rot` flipped 180° yaw. Only ever instantiated when `sensor_profile ==
camera_only` (see §4) — never spawn it in the primary rig, even inactive,
to keep VRAM budgeting between rigs directly comparable.

### 2.4 Wrist-adjacent RGB-D (Phase 2 placeholder)

```python
# PLACEHOLDER — do not implement until Phase 2 research pass.
# Reserve the field in SensorSuiteCfg (see §3) so the composition pattern
# doesn't need restructuring when this gets filled in.
wrist_camera: CameraCfg | None = None
# When implemented: mount ~10cm proximal from the ARX5 wrist joint (not on
# the gripper itself) to keep the grasp workspace in view without the
# gripper/fingers occluding it during closing.
```

## 3. Sensor-suite config module

Follows the exact pattern `common/cerberus_description/` already uses for
`UNITREE_GO2_CFG` + `ARX5_CFG` → `GO2_ARX5_CFG` — this is that same
composition idea, one layer over, for sensors instead of robot morphology.

```python
# common/cerberus_description/sensors/sensor_suite.py

@dataclass
class SensorSuiteCfg:
    lidar: RayCasterCfg | None
    front_camera: CameraCfg          # always present, both rigs
    rear_camera: CameraCfg | None    # camera-only rig only
    wrist_camera: CameraCfg | None   # Phase 2+, both rigs once implemented

LIDAR_CAMERA_SUITE = SensorSuiteCfg(
    lidar=LIDAR_CFG, front_camera=FRONT_CAM_CFG,
    rear_camera=None, wrist_camera=None,
)
CAMERA_ONLY_SUITE = SensorSuiteCfg(
    lidar=None, front_camera=FRONT_CAM_CFG,
    rear_camera=REAR_CAM_CFG, wrist_camera=None,
)
```

A fork adding a third rig (say, LiDAR + stereo + thermal) adds one more
`SensorSuiteCfg` instance here — nothing downstream needs to change unless
it specifically consumes the new modality.

`InteractiveSceneCfg` construction reads this config and conditionally
attaches sensors:

```python
if sensor_suite.lidar is not None:
    scene.lidar = sensor_suite.lidar
if sensor_suite.rear_camera is not None:
    scene.rear_camera = sensor_suite.rear_camera
```

## 4. Modularity contract: required-sensor manifests

This is the actual mechanism that makes "swap sensor rigs without rewriting
`slam/` or `planning/`" true instead of aspirational. Extends
`project_structure.md` rule 6 (embodiment identity is config, never
hardcoded) one layer up, to sensors.

**Every ROS2 package whose behavior depends on a sensor modality declares
what it needs**, in a small manifest sitting alongside its `package.xml`:

```yaml
# slam/slam/sensor_requirements.yaml
requires:
  lidar: required
  front_camera: required
# planner/local_planner/sensor_requirements.yaml (SCAN-Planner/VOP-Nav path)
requires:
  lidar: required
# perception/terrain_understanding/sensor_requirements.yaml (DELTA/HiPAN path)
requires:
  front_camera: required
  rear_camera: optional   # DELTA wants it, HiPAN doesn't — package-internal branch, not a bringup concern
```

**`cerberus_bringup` validates before spawning any node**, not at runtime
when a node discovers a topic is missing:

```python
# system/cerberus_bringup/validate_sensor_suite.py
def validate(active_suite: SensorSuiteCfg, package_requirements: dict):
    missing = []
    for pkg, reqs in package_requirements.items():
        for sensor, level in reqs.items():
            if level == "required" and getattr(active_suite, sensor) is None:
                missing.append((pkg, sensor))
    if missing:
        raise SensorSuiteMismatchError(
            f"Active sensor profile is missing sensors required by: "
            f"{', '.join(f'{pkg} needs {s}' for pkg, s in missing)}. "
            f"Either switch --sensor_profile or don't launch these packages."
        )
```

This turns "I forked this and swapped sensor profiles without checking
compatibility" from a confusing runtime hang (a node subscribed to a topic
that never publishes) into a startup error naming exactly which package
wanted exactly which missing sensor. This is the actual point of the
exercise — a fork should get a clear failure, not a silent one.

## 5. Launch-time profile switching

```bash
ros2 launch cerberus_bringup bringup.launch.py sensor_profile:=lidar_camera
ros2 launch cerberus_bringup bringup.launch.py sensor_profile:=camera_only
```

`sensor_profile` selects both the `SensorSuiteCfg` (§3, Isaac Lab side) and
the ROS2 node set (§4 validates the pairing is consistent) — e.g.
`camera_only` launches `terrain_understanding` (DELTA/HiPAN-capable) instead
of `slam` + `local_planner`'s LiDAR-only nodes. Add profiles to
`cerberus_bringup`'s launch arg choices as new `SensorSuiteCfg` entries get
added in §3 — the two should always be added together, one PR.

## 6. Observation-space consequences for RL/IL (flag, not solved here)

Switching sensor profile changes what's physically available to feed a
policy — a terrain-understanding policy trained on LiDAR-derived elevation
input cannot just be dropped into a camera-only observation space; that's a
retrain, not a config swap, same as an embodiment swap requires retraining
per `project_structure.md`. The `ManagerBasedRLEnvCfg`'s observation terms
should gate on the same `SensorSuiteCfg` fields (an observation term reading
`front_camera` fine in both rigs; one reading `lidar` only valid when
`lidar is not None`) so that pointing a training run at the wrong profile
raises an observation-space error immediately, not a confusing shape
mismatch mid-training. Not implemented yet — flagging so whoever writes
Phase 1's env configs builds this in from the start rather than retrofitting.

## 7. Open items

- Exact stereo baseline and resolution not yet tuned against a specific
  real-hardware reference (D430 vs D435i-class) — using RealSense-typical
  values as placeholders above.
- Whether front SLAM/VO actually needs true stereo pairs or just the single
  privileged-depth camera (§2.2) depends on which SLAM candidate from
  `perception.md`'s ablation table gets implemented first — decide there,
  not here.
- `mesh_prim_paths` for the LiDAR needs the actual scene mesh (scenes now come from `cerberus_worlds`, see docs/worlds.md)
  paths once Phase 1 scene loading is wired up — currently a placeholder.
- Wrist camera (§2.4) — fully deferred to Phase 2 per project ethos; the
  `SensorSuiteCfg` field exists now so Phase 2 doesn't need to restructure
  this file, but nothing else here should assume its shape.
