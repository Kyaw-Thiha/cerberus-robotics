# Cerberus — SLAM Module Handoff (coding-agent reference)

> **Status: architecture locked, candidates confirmed, several implementation
> specifics still open** (marked in §9). Sourcing/reasoning behind the sensor
> configs and candidate picks lives in the chat history and is summarized in
> `perception.md`'s SLAM section; this doc is implementation detail only, not
> justification. Companion to `sensor_setup_handoff.md` (physical sensor rigs)
> and `isaac_lab_workflow.md` (remote/local split, still applies unchanged).

## 1. The three sensor configs and six candidates

| Sensor config | Rig (see `sensor_setup_handoff.md`) | Baseline | Latest |
|---|---|---|---|
| LiDAR only | `lidar_camera` rig, camera unused by this backend | FAST-LIO2 | BIEVR-LIO |
| LiDAR + camera | `lidar_camera` rig, fused | FAST-LIVO2 | GS-LIVO (replaces LVI-Q, 2026-09-21 — see below) |
| Camera-only | `camera_only` rig | ORB-SLAM3 (stereo, no IMU) | ORB-SLAM3 + Multicam-SLAM fusion |

| Candidate | Repo | ROS | Notes |
|---|---|---|---|
| FAST-LIO2 | `github.com/hku-mars/FAST_LIO` | ROS1 (Melodic+, catkin) | Community ROS2 (Foxy/Humble) ports exist if a bridge-free path is ever wanted |
| BIEVR-LIO | `github.com/ethz-asl/BIEVR-LIO` | Not directly confirmed — ETH ASL/RSL's lab pattern (COIN-LIO, terrain-navigation, etc.) is uniformly ROS1-first; near-certain ROS1 | **Confirm directly once cloned** |
| FAST-LIVO2 | `github.com/hku-mars/FAST-LIVO2` | ROS1 (catkin, Ubuntu 22.04) | Community ROS2 Humble port exists (`chaoo0/FAST-LIVO2`) |
| ~~LVI-Q~~ GS-LIVO | ~~No public repo~~ `github.com/HKUST-Aerial-Robotics/GS-LIVO` (fork: `Kyaw-Thiha/cerberus-GS-LIVO`) | Native ROS2 Jazzy ament_cmake (`src/gs-livo/CMakeLists.txt`) — was ROS1 catkin, migrated by commits 40c16fd (build system) + af3d2f5 (rclcpp port) | **Replaced LVI-Q 2026-09-21**: LVI-Q (Kim et al. 2025, arXiv:2510.15220) had no public repo anywhere — implementing it would have meant a from-scratch VIKO+LIKO alternating-optimization system with zero reference code, deprioritized in favor of a real working alternative. GS-LIVO (Hong et al. 2025, arXiv:2501.08672) extends FAST-LIVO2's own IESKF backbone (shared author, Fu Zhang) with real-time 3D Gaussian-splat mapping — 490 GitHub stars, real buildable CUDA/C++ code confirmed. **GS-map persistence (correctly characterized):** the Gaussian map is purely in-process memory upstream (`sub_GSMap` + octree in `gsmap_manager->gs_map_`, no ROS publisher, no file save, no service). This fork adds a gated async persistence path (`publish.gs_save_en`, default `false`, writes 3DGS PLY to `Log/GSMap/`; snapshot under mutex, write on detached thread → zero SLAM accuracy impact). With that flag off, behavior is byte-identical to upstream (in-memory only, lost on shutdown). With it on, you get a .ply per `gs_save_interval_frames` (default 500) + `gs_map_final.ply` at shutdown — loadable in rerun for post-hoc photorealistic viewing. `map_type` for the per-frame `PointCloud2` adapter output stays `"point_cloud"` (the GS map flows as a separate .ply sidecar, not through `/slam/map`); extending `TaggedMap.msg` to carry a real Gaussian-splat field is deferred until a Phase 2+ downstream consumer is defined — see §5. |
| ORB-SLAM3 | `github.com/UZ-SLAMLab/ORB_SLAM3` | Core lib is ROS-agnostic C++; official example wrapper is ROS1 (Melodic) | Mature community ROS2 wrappers exist (e.g. `zang09/ORB_SLAM3_ROS2`, Foxy) |
| Multicam-SLAM | `github.com/AlterPang/Multi_ORB_SLAM` | ROS1 (built on ORB-SLAM2's Hydro+ ROS1 examples) | |

**The finding that matters:** every candidate with actual code is ROS1-rooted — **with one confirmed exception: BIEVR-LIO**. Verified directly against its README (2026-09-20): it ships *native* ROS1 Noetic (catkin, package `bievr_lio_ros`) **and** native ROS2 Jazzy (colcon, package `bievr_lio_ros2`, own `interfaces/ros2` folder) build paths — separate Docker scripts for each (`run_docker_ros1.sh` / `run_docker_ros2.sh`). Practical implication: BIEVR-LIO can skip the `ros1_bridge` entirely and publish straight onto the ROS2 topic contract from §2, unlike its own "baseline" pairing (FAST-LIO2, confirmed ROS1-only, no ROS2 port). Also needs Ceres 2.2.0 built from source (script provided in-repo) and only supports the Livox gen2 driver on its ROS2 side (gen1 is ROS1-only). Every other candidate below is still ROS1-only as stated — this is decision-simplifying for the rest — see §3.

**Also verified:** `AlterPang/Multi_ORB_SLAM` (the Multicam-SLAM candidate) is real and live, and is confirmed built on ORB-SLAM2 (not ORB-SLAM3) as this table states — but its README is a near-verbatim copy of stock ORB-SLAM2's own README (generic placeholder clone URL, only documents the standard `rgbd_tum` example, nothing multi-camera-specific called out). Whoever implements this backend will need to read `src/`/`Examples/` directly to find the actual multi-cam entry point rather than trusting the README.

RGB-D was evaluated as a possible fourth config and rejected — see `perception.md` §1 for the reasoning (KIT/Freiburg benchmark evidence).

**Rest of the project stays ROS2, no exceptions.** Nothing here changes `project_structure.md`'s ROS2 colcon workspace, the ROS2 nature of `perception/`, `planning/`, `policy/`, `safety_filter/`, `actuation/`, or the coding-agent dev loop. The ROS1-ness described below is confined entirely inside each backend's own container.

---

## 2. Architecture: pluggable backend, not a locked-in algorithm

`slam/slam/` stays a generic package name permanently — never renamed after whichever candidate wins an ablation. Same logic as `policy/locomotion/` not being named after PPO. See chat history for the full reasoning; the practical shape:

- **Backend selection is a launch-time config**, not a hardcoded import — `slam_backend:=fast_lio2` / `bievr_lio` / `fast_livo2` / `gs_livo` / `orb_slam3` / `multicam_slam`, same shape as `sensor_profile` in `sensor_setup_handoff.md` §5.
- **The topic contract is the actual interface boundary.** Every backend, regardless of language/ROS version/dependency stack, ultimately publishes:
  - `/slam/odom` (`nav_msgs/Odometry`)
  - `tf`: `map → base_link`
  - `/slam/map` (native format — see §5) tagged with a `map_type` field (`point_cloud`, `occupancy_grid`, `pose_graph`)
- `planning/global_planner` and everything else downstream depends only on this contract, never on which backend is active.
- **The backend adapter is thin.** Its job is remapping topic names and frame conventions to the contract above — it does **not** do map-type conversion (see §5) and does **not** encode backend-specific logic beyond that remapping.
- Nothing prevents keeping more than one backend "live" long-term (e.g. if the sensor-rig ablation concludes both `lidar_camera` and `camera_only` should ship) — the pluggable structure supports that for free.

---

## 3. Docker / ROS bridge strategy

Since every real candidate is ROS1, standardize on one pattern for all six rather than mixing strategies per backend:

- **One Docker image per backend**, ROS1 Noetic base, each candidate's own dependency stack isolated inside (PCL/Eigen/GTSAM/Sophus versions differ per candidate — do not try to share an environment across them).
- **`ros1_bridge` is the translation layer** inside (or alongside) each container: it takes the candidate's native ROS1 topics and republishes them onto the ROS2 topic contract from §2. The adapter logic (remapping, `map_type` tagging) lives on the ROS2 side of the bridge.
- A future ROS2-native backend (e.g. LVI-Q if implemented from spec) skips the bridge entirely and publishes directly onto the contract.
- Reuse the `docker/remote/Dockerfile` pattern from `isaac_lab_workflow.md` per-backend rather than trying to get six heterogeneous stacks coexisting in one image.

**Source vendoring, decided 2026-09-20 — diverges from `docker/remote/Dockerfile`'s own pattern on purpose.** Unlike `docker/remote/Dockerfile` (which `RUN git clone`s IsaacLab directly inside the Dockerfile, never touching the host), each SLAM backend's source is fetched to the **local host first**, then `COPY`'d into its image:

- `scripts/fetch_slam_backends.py <name> [<name> ...] | --all` — `git clone --depth 1` per named backend into `third_party/slam/<name>/`, idempotent (skips a backend already present), errors with the valid name list on an unknown one. No entry for `lvi_q` (no repo exists — its code is original-to-repo, see §2's `slam/slam/backends/lvi_q/`).
- `third_party/` is a new top-level, gitignored directory (same role as `data/`/`outputs/` — pulled from external sources, not authored here, never committed).
- Each `docker/Dockerfile.slam_<name>` does `COPY third_party/slam/<name> /workspace/<name>` instead of `RUN git clone` — the build context is the repo root (same as every existing `docker build -f docker/Dockerfile.... .` invocation in `runpod.md`), so `scripts/fetch_slam_backends.py <name>` must run once locally before that backend's image builds.
- Rationale: browsable/editable local source beats a black-box image layer, at the cost of one extra manual step before each backend's build — accepted tradeoff, not a correctness concern (build/run still only ever happens remotely either way, per `isaac_lab_workflow.md`'s remote-only rule for anything needing Isaac Sim; these backends don't need Isaac Sim at all, but do need ROS1, which isn't in the local `pixi.toml` either, so local execution was never on the table regardless of vendoring choice).

---

## 4. Per-backend sensor manifests

Extends `sensor_setup_handoff.md` §4's `sensor_requirements.yaml` + `cerberus_bringup` validation mechanism — that mechanism is already correct and shouldn't be reinvented, it just needs to be **per-backend**, not one static file for the whole `slam` package (a single file can't represent six different backends' needs).

```yaml
# slam/slam/backends/fast_lio2/sensor_requirements.yaml
requires: {lidar: required}

# slam/slam/backends/bievr_lio/sensor_requirements.yaml
requires: {lidar: required}

# slam/slam/backends/fast_livo2/sensor_requirements.yaml
requires: {lidar: required, front_camera: required}

# slam/slam/backends/lvi_q/sensor_requirements.yaml
requires: {lidar: required, front_camera: required}
# + joint_state — not currently a field in SensorSuiteCfg (it's proprioception,
# not an optical/range sensor). Open item, see §9.

# slam/slam/backends/orb_slam3/sensor_requirements.yaml
requires: {front_camera: required}

# slam/slam/backends/multicam_slam/sensor_requirements.yaml
requires: {front_camera: required, rear_camera: required}
```

`validate_sensor_suite.py` needs a small change: look up the manifest under the **active `slam_backend`'s** subfolder, keyed off both `slam_backend` and `sensor_profile`, rather than one package-root file.

---

## 5. Map type / conversion policy

**Principle: pass through each backend's native output untouched, tagged with `map_type`. Never silently convert inside the adapter.**

`map_type` is a free-form `string` field on `cerberus_msgs/TaggedMap` (`common/cerberus_msgs/msg/TaggedMap.msg:1-10`), not a strict enum. The convention is `"point_cloud"` for every backend currently registered (FAST-LIO2, BIEVR-LIO, FAST-LIVO2, ORB-SLAM3, Multicam-SLAM); the common adapter hard-codes this tag at `common_adapter.py:49`. GS-LIVO also passes through its per-frame registered scan as `"point_cloud"` (`/cloud_registered`, see its adapter_launch.py citation block) — the Gaussian Splat map that actually distinguishes GS-LIVO is a **separate artifact** that has no live ROS topic interface upstream (see GS-LIVO's §1 row above), so it does not flow through `/slam/map`. Persistence for the GS map is a fork-side add (file save to `Log/GSMap/`), gated by the backend's own `publish.gs_save_en` parameter; when it lands in R2 it's a `.ply` 3DGS file loaded by rerun, not a `/slam/map` message.

Conversion loss is real and asymmetric:

| Direction | Loss | Tooling |
|---|---|---|
| Point cloud → occupancy/elevation grid | Real but intentional — standard practice | `octomap_server`, `pointcloud_to_grid`, ANYbotics `grid_map` |
| Occupancy/elevation grid → point cloud | Not recoverable — grid-cell centers only, no information gain | No tooling exists for this; don't build it |
| Pose graph (RAEM-style, N/A to SLAM here but same principle) → dense map | Doesn't apply — nothing to convert | N/A |

- Adapter passes through native output + `map_type` tag.
- A **separate, optional, shared downstream conversion stage** (`octomap_server` / `pointcloud_to_grid`) sits after the contract, opt-in via config, for any consumer needing a grid instead of raw points. This stage is generic across every point-cloud-producing backend (FAST-LIO2, BIEVR-LIO, FAST-LIVO2, LVI-Q, ORB-SLAM3, Multicam-SLAM all produce point clouds) — it does not need to be duplicated per-backend.
- No node converts *up* (grid → point cloud). A consumer needing point-cloud detail should consume the native topic directly.
- **Extending `map_type` to a real Gaussian-splat field** (a `GaussianSplatMap` sibling msg carrying x,y,z + SH + opacity + scale + rotation, filled from the fork's `Log/GSMap/*.ply` files at startup or via a separate service) is **deferred** until a concrete downstream Phase 2+ consumer is defined. Until then, the .ply sidecar is the contract for the GS map.

---

## 6. Ground truth pose export

Needed to score every candidate with `evo` — nothing can be benchmarked without this.

- Isaac Lab's articulation data already computes this: `root_pos_w` / `root_quat_w` (world-frame position + orientation, per env, every physics step, no noise). **Confirmed 2026-09-20** against the pinned Isaac Lab version on a RunPod GPU pod (`Isaac-Velocity-Flat-Unitree-Go2-v0`, `env.unwrapped.scene["robot"].data`): both attributes exist, are populated with live data (`root_pos_w` shape (N, 3), `root_quat_w` shape (N, 4) in (w, x, y, z) order), and match the assumption exactly — no rename needed.
- Publish as `nav_msgs/Odometry` on `/ground_truth/odom`, `frame_id: world`, `child_frame_id: base_link`, sim-timestamped (`use_sim_time`).
- Also broadcast as TF: `world → base_link_gt` — deliberately a **different** frame id than SLAM's own estimated `map → base_link` (§2), so `evo`/`rerun` never confuse estimate vs. truth.
- Publish every physics step — it's free, no rendering cost involved.

---

## 7. Scripted motion source for SLAM eval

The planner doesn't exist yet (Planning & Orchestration is a later phase) and SLAM eval doesn't need it to — `evo` just needs a representative trajectory to score against.

- Isaac Lab ships `isaaclab.devices.Se2Keyboard` (and `Se2Gamepad`), built explicitly "for mobile base (such as quadrupeds)" — emits SE(2) velocity commands `(vx, vy, ωz)` as a tensor. Usable as-is for interactive bring-up.
- For actual eval runs, don't use the interactive device — write a small scripted source that emits the same `(vx, vy, ωz)` tensor shape on a fixed, canned schedule per mission (forward, turn, forward, revisit — matching the KIT benchmark's own mission structure, see `perception.md`). This keeps every candidate scored against **identical** motion, run to run — more rigorous than freehand teleoperation, which would introduce human-motion variance as a confound between backend comparisons.

---

## 8. Showcase camera (video/demo aid, not an algorithm input)

For the side-by-side comparison videos (§9 of the earlier chat, see rerun plan below) — purely visual, not consumed by any SLAM/RL/IL algorithm.

- New `showcase_camera: CameraCfg | None` field in `SensorSuiteCfg`, independent of `sensor_profile` — it's a video aid, not a sensor-suite choice.
- `data_types=["rgb"]` only, no depth needed.
- Mount as a soft-follow "chase cam": position tracks the robot, orientation stays level rather than rigidly inheriting body roll/pitch (unlike a real sensor mount) — this is what makes the footage watchable rather than sensor-accurate.
- **Explicitly excluded** from every `sensor_requirements.yaml` (§4) and from the Phase 0/1 RL observation-space gating in `sensor_setup_handoff.md` §6, so it never accidentally becomes a dependency.
- Only spawn it for single/few-env SLAM-eval and video-recording runs, **never** during massively-parallel RL training — RGB rendering is the expensive sensor per `sensor_setup_handoff.md` §2.2's cost note.

---

## 9. Evaluation & visualization pipeline

- **Capture:** `rosbag2 record` per mission run, on the remote box — every backend's contract topics (§2), the raw sensor feeds, `/ground_truth/odom`, and the showcase camera, all on the same sim clock so replay is automatically time-aligned.
- **Quantitative scoring:** `evo`, computing ATE/RPE per candidate against `/ground_truth/odom` — the same tool the KIT benchmark paper itself used, so results are comparable in kind.
- **Qualitative side-by-side video:** `rerun`, following their official ROS2 bridge (`rerun-io/cpp-example-ros2-bridge`) — confirmed live and real: a generic C++ bridge subscribing to all supported ROS2 topics, with topic-remap and static-transform config via an external launch-time file. **The "documented Spot-quadruped example" claim from `perception.md` could not be independently confirmed** in this pass — the repo reads as a general proof-of-concept with launch-file-driven examples, not obviously Spot-branded; check its `launch/` directory directly before leaning on that precedent. Mechanism: a small bridge script subscribes to bag-replayed topics and calls `rr.log(...)`; a **Blueprint** layout (scene view | camera feed | 3D map+trajectory | ATE-over-time plot) is defined once in code and reused identically across all six candidates and both rigs, so every comparison video has the same shape.
  - **MP4 export, split into two paths — confirmed 2026-09-20, updates the "no confirmed export" note below:**
    1. **Any single logged video stream** (e.g. the showcase camera) has a real, documented, no-re-encode export: log it via the `VideoStream` archetype (`rr.log(..., rr.VideoStream(...))`, pre-encoded e.g. H.264) rather than raw per-frame `rr.Image`, then remux straight out of the `.rrd` to `.mp4` via rerun's documented Python snippet (`rerun.io/docs/reference/video` → "Export MP4 from RRD"). No screen capture needed for this piece.
    2. **The actual §9 deliverable — the full 4-panel Blueprint as one video** — has no equivalent native export; that composite still needs screen-capturing the viewer (OBS / `ffmpeg -f x11grab`) or driving frames via the Python API and stitching with `ffmpeg`, exactly as originally planned. Don't conflate the two — path 1 alone doesn't produce the side-by-side comparison video.
- **RViz2** stays the zero-setup option for quick sanity checks during dev; not the tool for the polished comparison output.

---

## 10. Open items

- ~~**BIEVR-LIO's ROS version** — inferred from ETH ASL/RSL's consistent lab pattern, not directly confirmed from its own README. Confirm once cloned.~~ **Resolved 2026-09-20** — see §1: confirmed native ROS1 *and* native ROS2, not ROS1-only as assumed.
- ~~**LVI-Q's `joint_state` requirement** doesn't fit `SensorSuiteCfg`'s current schema...~~ **Moot as of 2026-09-21** — LVI-Q was replaced by GS-LIVO, which has no joint-kinematics fusion, so this specific schema question no longer applies. GS-LIVO instead needs `TaggedMap`'s `map_type` enum extended for its Gaussian-splat map output — see the candidate table above and §5.
- **Mission/scene definitions** — which scenes (now sourced from `cerberus_worlds`, see `docs/worlds.md`) constitute one "mission" (matching the KIT benchmark's own Table I structure) — not decided; blocks the eval harness from running against a real set.
- **Stereo camera baseline/resolution** — still a placeholder in `sensor_setup_handoff.md` §7; several candidates' reported metrics are baseline-sensitive.
- **`mesh_prim_paths` for the LiDAR** — needs real scene mesh paths (scenes now come from `cerberus_worlds`, see `docs/worlds.md`) once the mission/scene decision above is made (circular dependency, resolve together).
- ~~**rerun's video export capability** — not confirmed natively as of this pass; recheck before committing to a specific export workflow.~~ **Resolved 2026-09-20** — see §9: single-stream export via `VideoStream` remux is real and documented; the composite 4-panel comparison video still needs screen capture, no native export exists for that.
- **Multi_ORB_SLAM's actual multi-cam entry point** — its README is a near-verbatim copy of stock ORB-SLAM2's, documenting only the generic `rgbd_tum` example. Confirmed the repo is real and ORB-SLAM2-based, but the multi-camera-specific build target/usage isn't documented anywhere — will need to read `src/`/`Examples/` directly once this backend is implemented.
- **Whether rerun's ROS2 bridge actually ships a Spot-quadruped example** — `perception.md` cites this as "the closest public precedent," but it couldn't be independently confirmed in this pass; the repo reads as a generic proof-of-concept. Check `rerun-io/cpp-example-ros2-bridge`'s `launch/` directory directly before relying on this, or drop the claim from `perception.md` if it doesn't hold up.
