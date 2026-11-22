# SLAM design

Backends, the topic contract they share, map-type policy, ground truth and the evaluation pipeline. Companion to
`docs/perception.md` (which candidates and why) and `docs/sensor_setup.md` (the rigs).

## Backends against rigs

| sensor config | rig | baseline | latest |
|---|---|---|---|
| LiDAR only | `lidar_camera`, camera unused | FAST-LIO2 | BIEVR-LIO |
| LiDAR + camera | `lidar_camera`, fused | FAST-LIVO2 | GS-LIVO |
| camera only | `camera_only` | ORB-SLAM3 (stereo, no IMU) | ORB-SLAM3 + Multicam-SLAM |

Five backends are registered: `fast_lio2`, `bievr_lio`, `fast_livo2`, `gs_livo`, `orb_slam3`, `multicam_slam`. GS-LIVO
replaced LVI-Q, which had no public code. Almost every candidate is ROS1-rooted; BIEVR-LIO is the exception that ships
native ROS1 and ROS2 builds. The ROS1-ness stays inside each backend's own container: the rest of the stack is ROS2.

## Architecture

`slam/slam/` is a generic package name and stays one, whatever wins the ablation, for the same reason
`policy/locomotion/` is not named after PPO.

- Backend selection is launch-time config (`slam_backend:=fast_lio2`, ...), never a hardcoded import.
- The topic contract is the interface boundary. Every backend publishes:
  - `/slam/odom` (`nav_msgs/Odometry`)
  - `tf`: `map` to `base_link`
  - `/slam/map` (native format) tagged with `map_type` (`point_cloud`, `occupancy_grid`, `pose_graph`)
- Downstream (`planning/global_planner` and everything above) depends on that contract and never on which backend is
  active.
- The adapter is thin: it remaps names and frame conventions and tags the map. It does no map conversion and holds no
  backend logic.

## Images and source

One Docker image per backend, each candidate's dependency stack isolated inside it (PCL, Eigen, GTSAM and Sophus
versions differ; do not share one environment across them). The ROS1 side publishes onto the ROS2 contract through
`ros1_bridge`, with the adapter on the ROS2 side of the bridge. A native ROS2 backend skips the bridge.

Backend source is fetched to the local host first and copied into the image, rather than cloned inside the
Dockerfile:

```bash
scripts/fetch_slam_backends.py <name> [<name> ...] | --all
```

It clones shallowly into `third_party/slam/<name>/` (gitignored, never committed), idempotently, and lists the valid
names on an unknown one. Each image copies from there, which keeps the source browsable and editable at the cost of one
manual step before a build. Builds and runs only ever happen on the remote host either way.

## Map types

Pass through each backend's native output tagged with `map_type`; never convert inside the adapter.

`map_type` is a free-form string on `cerberus_msgs/TaggedMap`, not a strict enum. Every registered backend currently
tags `point_cloud`, including GS-LIVO, whose distinguishing Gaussian-splat map is a separate file artifact with no live
topic, so it does not flow through `/slam/map`.

| conversion | loss | tooling |
|---|---|---|
| point cloud to occupancy or elevation grid | real but intentional | `octomap_server`, `pointcloud_to_grid`, `grid_map` |
| grid to point cloud | not recoverable | none exists; do not build it |

A grid consumer therefore uses an optional shared downstream conversion stage after the contract, generic across all
point-cloud backends, rather than a per-backend converter. A Gaussian-splat message type is deferred until a consumer
needs it; the `.ply` sidecar is the contract until then.

## Ground truth

Scoring every backend needs a noise-free reference. Isaac Lab's articulation data provides it: `root_pos_w` and
`root_quat_w` per environment, every physics step, no noise (verified on the pinned version, quaternions in `w, x, y, z`
order). It is published as `nav_msgs/Odometry` on `/ground_truth/odom` with `frame_id: world`, `child_frame_id:
base_link`, sim-timestamped, and broadcast as `world` to `base_link_gt` — deliberately a different frame from SLAM's
`map` to `base_link`, so `evo` and `rerun` never confuse estimate with truth.

## Motion for evaluation

Evaluation needs a trajectory, not a planner. Interactive bring-up uses Isaac Lab's `Se2Keyboard`, which emits SE(2)
velocity commands for quadrupeds. Evaluation runs use a scripted source that emits the same `(vx, vy, wz)` tensor on a
fixed schedule per mission (forward, turn, forward, revisit), so every backend is scored against identical motion and
human variance is not a confound between them.

## Evaluation pipeline

- **Capture:** `rosbag2 record` per mission on the remote host: the contract topics, the raw sensor feeds,
  `/ground_truth/odom` and the showcase camera, all on the sim clock so replay is time-aligned.
- **Scoring:** `evo`, ATE and RPE against `/ground_truth/odom` — the tool the KIT benchmark itself used.
- **Video:** `rerun` with its ROS2 bridge, a bridge script subscribing to the replayed topics and logging, plus a
  blueprint layout (scene, camera feed, map and trajectory, ATE over time). A single logged video stream can be
  remuxed out of the `.rrd`; the composite panel video has no native export and needs a screen capture.
- **Quick checks:** RViz2 during development.

The showcase camera is a video aid, independent of `sensor_profile`, excluded from every sensor manifest and from
observation gating, and only spawned for single or few-env evaluation runs.

## Open items

- Which scenes make one mission (the KIT benchmark's mission structure is the reference) is not decided; it blocks
  running the harness against a real scene set.
- Stereo baseline and resolution are still placeholders (see `docs/sensor_setup.md`).
- LiDAR `mesh_prim_paths` needs the real scene meshes, which depends on the mission decision above.
- Multicam-SLAM's README documents only the generic example; its multi-camera entry point has to be read out of its
  source when that backend is implemented.
