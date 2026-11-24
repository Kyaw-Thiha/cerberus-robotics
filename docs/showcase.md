# Showcase

What this repository can show today, what each artifact actually is, and what is missing. Every clip named here is
a real render of a real run; the label says whether it is a policy driving the robot or scripted motion, and whether
the scene's own materials are on.

Status legend: **done** (gated and shipped), **pending** (build in this session), **missing** (not produced; why).

## Artifacts

| artifact | what it shows | what is simulated / ground truth / scripted | status |
|---|---|---|---|
| `mission_kinematic.mp4` (1920x1080, 30 fps, 4 s) | top-down map of the full household mission: navigate to the table, pick with scripted IK, navigate to the place surface, place | **kinematic simulation, scripted IK skills**: A* + pure pursuit on a unicycle (no physics, no slip, no falls) on the scene-1 occupancy grid; the arm solves IK waypoints; the robot/cube poses are the simulator's own state | done |
| `arm_chase.mp4`, `arm_closeup.mp4` (1920x1080, 30 fps, 20 s) | the Go2 **with the ARX5** standing in the real GRScenes home while the arm reaches, turns the wrist and opens/closes the gripper | RTX render of the physics sim; legs held by the retrained `go2_arx5_locomotion_ga_v1_candidate` checkpoint, arm by scripted joint targets; camera poses logged per frame; **scene materials are the dataset's own textures rebuilt as UsdPreviewSurface** (the MDL shaders themselves do not compile on this stack) | rendered in session 3 |
| `walk_ego.mp4` (1920x1080, 30 fps, 20 s) | the arm robot walking a scripted path in the home | RTX render; velocity commands are scripted, the joints come from the retrained checkpoint; pose log per frame; passes the clip QA gate (hue bins 3-5, luminance 122) | rendered and gated in session 3 |
| `arm_ego.mp4` (1920x1080, 30 fps, 20 s) | the arm robot standing while the arm reaches, turns the wrist and opens/closes the gripper | RTX render; legs held by the retrained checkpoint, arm scripted | rendered in session 3 |

**Scene look, stated honestly.** The materials are the dataset's own textures, rebuilt as `UsdPreviewSurface` and bound
in scope (the root cause of the earlier flat grey was a USD reference-scope rule, see
`docs/reports/render_unblock.md`). What is *not* solved on this stack: exposure. The RTX tonemapper setting and a
0.45 global scale on all 137 scene lights change these views by less than 10 %, so the wide views of this white-toned
home read a mean luminance of 173-235 and fail the gate's 70-170 band (the EGO view reads 134 and passes). No
threshold was changed to hide that.
| `walking_30s.mp4` (30 s, 1280x720, 20 fps) | **JEPLO authors' demo**: their released checkpoint walking the Go2 in their own MuJoCo sim-to-sim loop, headless | their simulator (`unitree_mujoco` python route) and their checkpoint; depth comes from a port of their publisher, inference on CUDA EP (not TensorRT); **not our pipeline, not our scene**; the clip-QA still checks fail its hue diversity because MuJoCo's grid is grey (media checks pass) | done (labelled: authors' demo) |
| `demo_scene1.mp4` (previous session) | 90 s of the legs-only robot walking a scripted path | **rejected**: the scene rendered as one flat colour (MDL materials unresolved) and the camera was fixed, not following | not shown |

## Scene look: what was wrong and what was fixed

The previous session's demo rendered the whole home as a single flat red-orange. Measured here, on the scene USD the
run loads (`start_result_navigation.usd`): **3,282 MDL shader prims**, 5,013 asset inputs, **0 missing files**, and
`MDL_SYSTEM_PATH` empty -- because `run_session.sh` sets `CERBERUS_SKIP_MDL_PATH=1`. The flat colour is an
unresolved MDL material, not a broken USD. The fix is to point Kit at the scene's `Materials` root (810 MDL files,
including `KooPbr.mdl`, which 280 of them import) **before** the app launches.

## Rendering: what did not work, and why

Two independent problems, both evidenced in `docs/reports/first_scene_session.md`:

1. **MDL materials do not compile.** The scene USD binds 3,282 MDL shader nodes and every referenced file exists,
   but the renderer reports `MdlModule is NOT in the DB` and `USD_MDL: ... is Invalid` for 29 modules in one launch,
   so each material falls back to a default colour -- the flat look. Setting `MDL_SYSTEM_PATH` (which
   `run_session.sh` had been skipping) is necessary but not sufficient: the modules themselves fail to compile.
2. **Camera/RTX launches do not survive.** 159 of 164 launch attempts failed (14-20 s), `carb.windowing-glfw` fails
   headless, and the launches that reached the scene build exited 0 without their outputs. No clip from this host
   passed the gate; nothing is claimed.

The kinematic mission animation above is produced by matplotlib + ffmpeg, not by RTX, which is why it exists.

## Metrics

| measurement | value | kind |
|---|---|---|
| kinematic mission, N=10 seeded trials, real scene-1 grid | 4/10 success; failure modes: navigation to table 1, navigation to place 2, place unreachable 2, pick unreachable 1 | kinematic upper bound with a scripted arm |
| navigation, kinematic, real grid (previous session) | 10/10, SPL 1.0, 5.62 m mean path | kinematic upper bound, unicycle |
| arm locomotion retrain | see `docs/reports/first_scene_session.md` | PPO, 4096 envs, seed 42 |

## What is missing, and why

- **Physics navigation and the physics household mission**: the host class crashes Kit's loader on most launches and
  exits 0 after long sim blocks; see `docs/reports/kit_reliability.md` and `docs/reports/kit_crash_report.md`.
- **A hero clip with the retrained policy**: pending the retrain's evaluation.
- **SLAM**: FAST-LIO2 / ORB-SLAM3 had no time box in either session; the rosbags and the docker recipes are untouched.
- **JEPLO**: ran (attempt 2) in the authors' own MuJoCo demo; see `docs/reports/jeplo_spike.md`. It is not integrated:
  adapting it needs their sensor simulation re-implemented on our stack (3-4 weeks, their estimate in that report).

## Credits

- **GRScenes-100** home scenes: CC BY-NC-SA 4.0, non-commercial use. Dataset: <https://github.com/OpenRobotLab/GRScenes>
- **Isaac Sim / Isaac Lab** (NVIDIA) for the simulator and the RL stack.
- **JEPLO** (ASIG-X) is referenced as the *authors' demo with their released checkpoint*, not as our pipeline; it is
  GPLv3 and lives outside this repository.
