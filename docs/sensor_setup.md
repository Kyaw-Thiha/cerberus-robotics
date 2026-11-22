# Sensor setup

The two sensor rigs, the Isaac Lab sensor configs that implement them, and the contract that lets a deployed stack swap
a rig without editing code in `slam/`, `planning/` or `policy/`.

## The two rigs

| sensor | LiDAR + camera (primary) | camera-only (ablation) |
|---|---|---|
| LiDAR, spinning | yes | absent |
| front stereo camera | yes | yes |
| rear stereo camera | absent | yes |
| wrist RGB-D | phase 2 | phase 2 |

Both rigs are implementation targets, not research placeholders. Anything below is written so that switching rig, or
adding a third one in a fork, is a config and launch-argument change.

## Isaac Lab sensor configs

### LiDAR (primary rig)

```python
lidar = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    update_period=1 / 10,                      # 10 Hz, the Livox Mid-360 hardware rate
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),   # the stock Go2 L1 mount height
    mesh_prim_paths=["/World/ground"],         # extend per scene: the scene meshes from cerberus_worlds
    attach_yaw_only=True,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=100,
        vertical_fov_range=[-25, 15],          # asymmetric, biased downward for near-field and underbody
        horizontal_fov_range=[-180, 180],
        horizontal_res=1.0,
    ),
    debug_vis=False,
)
```

`RayCasterCfg` has no `spawn` field, so it cannot create a prim before the robot exists. Attach it to
`{ENV_REGEX_NS}/Robot/base` directly instead of pre-spawning a cage prim.

`mesh_prim_paths` is a static list of meshes to raycast against, not a scene query. Every scene must be added to it, and
forgetting to is silent: the scan returns no hits rather than raising. Start-up sanity check it.

### Front stereo camera (both rigs)

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
)
```

RGB rendering is the expensive sensor: only spawn cameras for runs that need them, never for massively parallel
training.

### Rear stereo pair

Camera-only rig only, mounted at the rear of the base. The arm occupies the rear of the back once it is flown (see
`docs/worlds_integration_spec.md`), so the mount has to be settled before the arm is mounted.

### Wrist RGB-D

Phase 2. The field is reserved in `SensorSuiteCfg` so the composition does not need restructuring later. Mount it about
10 cm proximal from the ARX5 wrist rather than on the gripper, so the grasp workspace stays in view while the fingers
close.

## Sensor suite config

`common/cerberus_description/sensors/sensor_suite.py` holds one `SensorSuiteCfg` per rig. `InteractiveSceneCfg`
construction reads the active suite and attaches only the sensors it defines, so an absent sensor is the config saying
"this rig does not have one" rather than a special case in the scene code.

## Required-sensor manifests

Every ROS2 package whose behaviour depends on a sensor modality declares it in a small manifest beside its
`package.xml`:

```yaml
# slam/slam/sensor_requirements.yaml
requires:
  lidar: required
  front_camera: required
```

Levels are `required` and `optional`. `system/cerberus_bringup` validates the pairing before spawning any node: a
required sensor missing from the active suite is a start-up error naming the package and the sensor, not a node that
subscribes to a topic nobody publishes.

## Profile switching

```bash
ros2 launch cerberus_bringup bringup.launch.py sensor_profile:=lidar_camera
ros2 launch cerberus_bringup bringup.launch.py sensor_profile:=camera_only
```

`sensor_profile` picks both the Isaac Lab `SensorSuiteCfg` and the ROS2 node set, and the same launch validates that the
two agree. A new profile is added to both in the same change.

## Consequence for learning

Switching rig changes what a policy can see, so it is a retrain, not a config swap, exactly as an embodiment swap is.
Observation terms gate on the same `SensorSuiteCfg` fields, so pointing a training run at the wrong profile fails
immediately on the observation space instead of producing a shape mismatch mid-training.

## Open items

- The stereo baseline and resolution are placeholders until they are tuned against a specific real sensor.
- Whether visual odometry needs true stereo pairs or only the single depth camera depends on which SLAM backend is
  built first (see `docs/slam_design.md`).
- `mesh_prim_paths` for the LiDAR needs the real scene meshes once scene loading is wired up.
