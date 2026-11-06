# Arena Integration Log

Log of the W0-W4 gates of the Arena integration. W labels are specific to this
integration and differ from the Cerberus Phases (0-4) in `docs/cerberus_project_context.md`.

## W0 - Remove OmniGibson

Date: 2026-10-04

Removed:
- `common/cerberus_scenes/`
- `docker/remote/patch_omnigibson_device.py` and its test
- `docker/remote/fetch_omnigibson_assets.py`
- Dockerfile OmniGibson steps
- `start.sh` dataset fetch and version marker
- `scripts/smoke_og_*.py` and `_remote_launch_smoke.sh`
- SLAM eval OmniGibson hooks in `slam/slam/eval/sim_env_cfg.py` and
  `slam/slam/ground_truth/sim_ground_truth_loop.py`
- RunPod network-volume references; the volume itself (`8ey9ahvll1`, 50 GB, EU-RO-1) was deleted
  on 2026-10-05, along with the OmniGibson-era `cerberus-remote` Docker Hub tags v2, v3, v5, v6,
  v7 and v9. The new image was then retagged `v3` and `v4` was removed; v1 predates OmniGibson and
  was kept. Old local images and the OmniGibson dangling builds were removed too
- OmniGibson docs

Deviations from the plan:
- SLAM eval coupling was not in the plan. It now runs on stock flat terrain until W1.
- OmniGibson docs were deleted per user decision instead of getting SHELVED banners.
- The `worlds/` scripts named in the plan did not exist.
- `self.events.push_disturbance_clear = None` in `policy/locomotion/core/platforms/go2/env_cfg.py`
  was kept: it is a real bug fix unrelated to OmniGibson (only its comment was reworded).

Gate result: PASS.

Versions pinned: Isaac Sim 5.1.0, Isaac Lab v2.3.2, rsl-rl-lib 3.1.2, unchanged.

Checks:
- `grep -rniE 'omnigibson|behavior-?1k'` over code, Docker and scripts: no matches.
  Remaining mentions are in this log and one-line history notes in docs.
- Image build: `docker build -f docker/remote/Dockerfile` from a minimal context (`pixi.toml`,
  `pixi.lock`, `docker/remote/*`; the repo root has no `.dockerignore` and would ship about 8 GB
  of `.pixi` and `third_party`). Built `kevinkyawthiha/cerberus-remote` (18.7 GB, down from
  20.3 GB), first pushed as `v4` and then moved to `v3`; digest
  `sha256:b9b8381c88a0d4d82ac1c835d6436ad194a4e37d87cb833e6e04ebe8d2cc1706`.
- Pod: RTX 4090, SECURE, US-NC-1, driver 570.195.03. `start.sh` ran cleanly and
  `/etc/cerberus_env.sh` exports `RUNPOD_*` and `ISAACLAB_PATH`; no OmniGibson directories.
- Train: `train.py --config smoke_test --headless --logger tensorboard` exited 0. Three
  iterations, finite mean reward, checkpoints `model_0.pt`, `model_2.pt`, `model_best_easy.pt`,
  params and git diff written.
- Play: `play.py --headless --video --video_length 100 --num_envs 8` exited 0. Checkpoint loaded,
  policy exported to `policy.pt` and `policy.onnx`, 100-step video written.
- Pod deleted afterwards; `pod list` is empty.

Observation, not a gate failure: at the end of the play run, Isaac Lab's own
`isaaclab/envs/ui/viewport_camera_controller.py` printed `AttributeError: 'ManagerBasedRLEnv'
object has no attribute 'scene'` from a Kit callback during teardown. The run still exited 0
and finished all steps. It sits in Isaac Lab UI code, not code we touched. It was not compared
against the previous OmniGibson-era image, so it is unproven whether it pre-dates W0.

## W0.5 - cerberus_worlds skeleton

Files created per the contract:
- `common/cerberus_description/embodiment.py`: `EmbodimentCfg`, `compose_embodiments`
- `common/cerberus_worlds/base.py`: `SceneInfo`, `WorldBackend`
- `common/cerberus_worlds/registry.py`: `register_backend`, `get_backend`, `available_backends`,
  `WORLD_BACKEND_ENV`
- `common/cerberus_worlds/backends/arena_backend.py` and `plain_usd_backend.py`: stubs raising
  `NotImplementedError` until W1
- colocated tests, including an AST-scan test that no package imports `backends.*` directly

Deviations from the plan:
- No default backend: `get_backend()` raises `ValueError` unless an argument or `WORLD_BACKEND`
  is given.
- Repo-wide pytest cannot collect locally because `policy/locomotion/__init__.py` imports
  `isaaclab_tasks`. Use scoped runs, e.g. `pixi run python -m pytest common/...`.

Gate result: PASS (local).

Commands run:
- `pixi run python -m pytest -q common docker system slam/slam/eval slam/slam/ground_truth scripts`:
  138 passed, 5 skipped.
- `pixi run ruff check` on the new files: clean (7 autofixes applied, mostly `collections.abc`
  imports).
- Architecture test: no package outside `common/cerberus_worlds/` imports `backends.*`.

Baseline note: before W0, the same scoped run had 3 failures, all in the deleted
`test_omnigibson_scenes.py`. `slam/slam/backends` tests cannot be collected locally either way,
because `cerberus_msgs` needs a colcon build (pre-existing, unrelated).

## W1.5a - ARX5 assets, local prep (no pod run yet)

Date: 2026-10-04. Status: code written and unit-tested locally; the gate itself (spawn, stand with a
stowed arm, joints by name, 16 envs) has NOT been run. It needs a GPU pod.
Update 2026-10-05 (overnight): the gate was run later (see "W1.5a - pod run" and "W1.5a follow-ups"); the 18 DoF / no-gripper finding below is superseded by the 20 DoF X5A fused USD.

Written:
- `scripts/fetch_arx5_assets.py`: sparse-clones `real-stanford/arx5-sdk` `models/` into gitignored
  `third_party/arx5-sdk/` (MIT; commit `c160db7` recorded in `.cerberus_fetch.json`).
- `common/cerberus_description/go2_arx5_spec.py`: pure-python contract (joint names, limits, link
  masses, stowed pose, arm gains, mount, USD path, `GO2_/ARX5_/GO2_ARX5_EMBODIMENT`).
- `common/cerberus_description/go2_arx5_cfg.py`: `GO2_ARX5_CFG`, derived with `.replace()` from the
  stock `UNITREE_GO2_CFG`, so the legs keep the Phase 0 body and actuators. Arm = one
  `ImplicitActuatorCfg` per joint.
- `common/cerberus_description/urdf_patch.py`: URDF patch (massless `eef_link` gets 1 g; mesh paths made
  absolute) and USD joint-name validation.
- `scripts/compose_go2_arx5.py` (pod): `UrdfConverter` on the arm, then `isaacsim.robot_setup.assembler`
  onto the stock `go2.usd`, then validation. `--dry-run` works locally.
- `scripts/spawn_hold_go2_arx5.py` (pod): 16-env PD hold at the default pose with a JSON report.

Findings from the LeggedManip_Lab reference USD (inspected locally with `usd-core`, no Isaac Sim):
- **No actuated gripper** in the USD or in its MuJoCo model (MuJoCo `link7`/`link8` are static bodies).
  The arm is 6 DoF, 18 DoF in total. `cerberus_project_context.md` says "6-DoF + parallel gripper";
  X5.urdf has none either. A gripper is a prerequisite for W2 pick-place and is not solved here.
- Arm mass 3.454 kg in the USD, 3.52 kg in X5.urdf, 3.35 kg in the spec. Go2 trunk + legs 16.095 kg.
- Mount: `link0` sits at (0, 0, 0.07) on the trunk in the USD but at 0.06 in its MuJoCo model. Spec uses
  0.07; Unitree's mounting spec has not been checked.
- Joint limits differ: X5.urdf has j1 +-10 rad, j5 +-1.67, j6 +-1.57. The USD has +-3, +-1.5, +-3.
  Spec follows the SDK URDF.
- Their USD has a massless end-effector body with a -inf centre of mass (hence the eef mass patch). Its
  `config.yaml` says `merge_fixed_joints: true`, yet the fixed joints are still present.
- Their arm drives are zero in the USD; gains live in the ArticulationCfg (50/3, 50/2, 80/3, 30/3,
  30/2.5, 20/1). Their effort limits are commented out; the spec uses their MuJoCo ctrlrange
  (20, 20, 20, 7, 5, 5 N*m). The ARX datasheet has not been checked.
- Upstream `X5.urdf` references `./meshes/base_link.STL`, which does not exist (the file is
  `meshes/X5/base_link.STL`). `patch_arm_urdf` repairs a unique-basename match and logs it.

Deviations from the plan:
- The plan says "URDF to USD via UrdfConverter, fixed joint on the Go2 back". The arm is converted that
  way, but the Go2 body is the stock Isaac `go2.usd` rather than the repo's ROS xacro, so legs-only
  baselines stay comparable. The repo xacro has rotor links, a `base`/`trunk` split and sensors.
- `embodiment_asset_composition.md` named `RobotAssembler.assemble_articulations()`. The Isaac Sim 5.1
  docs list `assemble_rigid_bodies(base_path, attach_path, base_mount_frame, attach_mount_frame, ...)`;
  the script uses that.

Unverified until a pod run (each marked `# UNVERIFIED:` in code):
1. `assemble_rigid_bodies` yields ONE articulation (the script raises if the count of
   `PhysicsArticulationRootAPI` prims is not 1, and does not strip the arm root itself).
2. Enabling the assembler extension after app start; the stage the assembler acts on.
3. `go2.usd` has a `defaultPrim`; the mount-frame arguments are relative to the given paths; the
   assembler accepts a plain Xform and an arm without the Isaac robot schema.
4. Relative asset paths in `go2.usd` survive `Flatten()`; `rotateXYZ` order equals URDF rpy.
5. `effort_limit_sim` on `ImplicitActuatorCfg` in Lab 2.3.2; `root_physx_view.get_masses()` shape.
6. Spawn-test thresholds are provisional until first calibrated.

If (1) fails, the plan's rule applies: stop and report; no patching of Isaac internals.

Checks run locally:
- `pixi run python -m pytest -q common docker system slam/slam/eval slam/slam/ground_truth scripts`:
  198 passed, 6 skipped (the 6 need isaaclab).
- `pxr`-gated tests in `test_urdf_patch.py` ran in a throwaway `usd-core` venv: 14 passed. They skip in
  pixi, which has no `usd-core`.
- `ruff check` on the new files: clean.
- `caplog` is empty in the pixi ROS pytest env (pytest 7.4.4), reproduced with a 3-line test, so the
  one logging test spies on the logger instead.

Follow-up: `usd-core` (26.8.x) added as a PyPI dependency in the default pixi env only (`pixi add --pypi
usd-core`; not on conda-forge). The Docker environments (`sim-ros2`, `slam-*`) use
`no-default-feature`, so it does not enter the pod or SLAM images; the pod uses Isaac Sim's own `pxr`.
The `pxr`-gated tests in `test_urdf_patch.py` now run in the normal suite: 199 passed, 5 skipped (the 5
remaining need isaaclab).

## W1.5a - pod run

Date: 2026-10-05 (pod clock). Pod: RTX 4090, SECURE, US-NC-1, driver 595.91.07, image
`kevinkyawthiha/cerberus-remote:v3`, Isaac Sim 5.1.0, Isaac Lab 2.3.2 (`isaaclab` 0.54.2). About 80
minutes, about $0.80; the pod was deleted afterwards and `pod list` is empty.

Gate result: PASS on flat ground. The "spawn in a W1 scene" half waits for W1.

Commands (all on the pod, via `$ISAACLAB_PATH/isaaclab.sh -p`, headless):
- `scripts/compose_go2_arx5.py --force`: exit 0 in 11 s. `UrdfConverter` on the patched `X5.urdf`, then
  `RobotAssembler.assemble_rigid_bodies` onto the stock Nucleus `go2.usd`, then validation OK.
- `scripts/spawn_hold_go2_arx5.py` (16 envs, 10 s PD hold at the default pose): exit 0, `passed: true`.

Fused USD, checked locally with `usd-core`: one `PhysicsArticulationRootAPI` (`/go2_arx5/base`), 18
revolute joints (12 leg + `joint1..6`), mount joint `/go2_arx5/arx5_arm/base_link/AssemblerFixedJoint`
between `/go2_arx5/base/arm_mount` and the arm `base_link`, no sublayers, no external asset refs
(self-contained, 38 MB, gitignored under `data/assets/go2_arx5/`).

Hold metrics (fused, 16 identical envs): base height 0.236 m, settle drift 1.7e-5 m, arm joint error
max 0.042 rad (joint4), arm mass 3.521 kg, total 18.54 kg (stock Go2 15.02 kg + arm), joint names OK, eef
body present, all finite.

Observations, not gate failures:
- Legs sag under PD hold with no policy (stock gains, stiffness 25): max error 0.656 rad on `RL_calf`, mean
  0.20 rad. The reference USD sagged 0.42 rad. Hence the leg bound is loose (0.8 rad) and the arm bound is the
  real "stowed arm holds its pose" check (0.10 rad).
- Fused tilt is larger than the reference: pitch -11.4 deg, roll -5.2 deg (max tilt 12.5 deg) against 1.3 deg.
  Checked afterwards from the fused USD: the arm's CoM at the stowed pose is (-0.007, 0.000, 0.127) m in the
  trunk frame and the whole-robot CoM x moves from +0.009 to +0.006 m, so the arm shifts the CoM by about 3 mm
  in x and 0 in y. A fore/aft or left/right CoM shift from the arm does NOT explain the lean (it does raise the
  CoM by about 0.03 m). The rear calves sag 2-3x more than the front ones, so cause is still unattributed; a legs-only stock run through the same harness would separate arm from Go2. W4 trains with the
  load, so this is context rather than a blocker. It may argue for a different stow pose.
  Superseded 2026-10-05: the lean was attributed to stock Go2 PD sag; see "W1.5a follow-ups".
- The reference LeggedManip_Lab USD, through the same harness, sets `passed: false` only on harness
  assumptions (our body names and a 0.124 rad `joint4` sag against the 0.10 bound); it spawns and settles.

Resolved from the "unverified" list:
1. `assemble_rigid_bodies` yields ONE articulation: yes.
2. Extension enable after app start, `new_stage()`, `defaultPrim`, `Flatten()`: all fine.
3. Mount-frame arguments: NOT relative as the docstring says. `robot_assembler.py` uses them verbatim as
   absolute stage paths (first attempt failed with `Path must be an absolute path`). The script now passes
   absolute paths.
4. `effort_limit_sim` on `ImplicitActuatorCfg` in Lab 2.3.2: accepted. `get_masses()` is
   `(num_envs, num_bodies)`.
Still provisional: the hold thresholds (arm 0.10 rad, leg 0.80 rad, tilt 25 deg, height band).

Bugs found only by running on the pod, fixed:
- `isaaclab.sh` needs `TERM` set when launched over a non-tty ssh (`'ansi+tabs': unknown terminal type`);
  every step exited at once.
- `SimulationApp.close()` ends the process with exit code 0, so `finally: close()` swallowed tracebacks and
  made failed runs look like successes. It also hung for more than 10 minutes at shutdown after the work was
  done (twice). Both scripts now write results first, then flush and `os._exit(code)` with the real code. A
  `faulthandler` watchdog (`--watchdog_s`) prints thread stacks if a run stalls.
- The assembler path bug above.

Driver note: 595.91.07 is the branch `docs/runpod.md` marks as bad for RTX rendering. Headless physics with no
cameras ran fine on it. Rendering on that driver is still untested.
Update 2026-10-05: rendering on 595.91.07 with Sim 5.1 segfaults; see "W1.5a follow-ups".

Artifacts (local, gitignored): `data/assets/go2_arx5/go2_arx5.usd` and `.compose.json`,
`data/reports/hold_final.json`, `hold_fused2.json`, `hold_reference2.json`, `data/reports/w15a_logs/`.
Not pushed to R2.

Next: W1 (Arena spike) is still the go/no-go; W1.5a's scene half re-runs there. Open before W2: an actuated
gripper (none exists in any asset we have).
Update 2026-10-05: W1 ran on the Lab 3.0 line, the gripper exists (X5A, 20 DoF); see "W1 - new-stack gates" and "W1.5a follow-ups".

## W1 - decisions before the spike

Date: 2026-10-05. No code written yet; this records what was checked and decided.

Arena compatibility (README badges, `pyproject.toml` and file trees fetched from GitHub):
- `release/0.3.0` (last commit 2026-09-15): Isaac Lab 3.0.0, Isaac Sim 6.0.0, Python >=3.12; `pyproject` pins
  `isaaclab==3.0.0b2` and depends on `lightwheel-sdk`; has the Lightwheel kitchen-bench content.
- `feature/arena_v0.2_on_lab_2.3` (last commit 2026-03-31): Isaac Sim 5.1.0, Isaac Lab 2.3.0, Python >=3.10; README says
  "has not been SQA tested". No Lightwheel files and no `lightwheel-sdk` dependency. This is evidence from file names and
  `pyproject`, not from running it.
- Consequence for W3: the plan assumed Lab 3.0 GA on Sim 6.1. Arena's 0.3.0 release targets Lab 3.0.0b2 on Sim 6.0,
  so the W3 target may end up being whatever Arena supports. To be settled when W3 starts.
Superseded 2026-10-05: W3 target resolved to Sim 6.1.0 with Arena `main` and Lab `bb0c8e1b9`, then Lab `v3.0.0-EA`; see "W3".

Decisions (user):
- Gate A runs on the Lab 2.3 branch with the existing pod image. If it fails there, a time-boxed `release/0.3.0` spike is
  done before Arena is declared unable to host a quadruped, because the failure could be the stale branch.
- Gate B (Lightwheel) cannot run on 2.3, so it moves to the 0.3.0 stack, after W3.
- W2 is split: W2a (fixed-base Mimic, stock Isaac Lab scene, Lab 2.3.2) and W2b (Lightwheel kitchen, after W3). The gripper
  decision still has to come before W2a.
  Superseded 2026-10-05: Gate A did not run on the Lab 2.3 branch (W1 ran on the Lab 3.0 line) and Gate B ran on the new stack before W3 closed; see the plan revision at the end of this section and "W1 - new-stack gates".
- No new git branch; nothing committed or pushed. Pre-W1 snapshot (tracked diff, untracked tarball, base commit) in
  `data/backups/pre_w1_20261005_0025/`, gitignored.

GRScenes access: Hugging Face `InternRobotics/GRScenes` reports `gated: false` (checked via the API without a token), so
there is no approval step. GRScenes-100 is `scenes/GRScenes-100/`: `commercial_scenes.zip` 23.8 GB and `home_scenes` as a
split zip (21.5 + 21.5 + 8.0 GB), about 74.7 GB compressed. The split zip must be joined before extraction. Extracted size
is not measured; peak disk during download plus extraction is probably about 150 GB.

Storage decision (user, 2026-10-05): no network volume. A network volume would cost $0.07/GB/month (50 GB was $3.50, 150 GB
$10.50) and ties pods to one datacenter. Instead, one throwaway CPU pod downloads and extracts the GRScenes home set, then
copies the smallest scene plus the shared `models/` and `Materials/` files it references to a private R2 prefix. GPU pods pull
only that subset (R2 egress is free; first 10 GB-month free, $0.015/GB-month after). The R2 bucket was 1.54 GiB with top-level
prefixes `locomotion/` and `slam/` when checked. `rclone` is not in the remote image yet. Not started.
Update 2026-10-05: the extraction ran on an A5000 GPU pod (Runpod CPU pods cap container disk at 20-30 GB), not a CPU pod; see "W3" (GRScenes subset) and "Run record".

Lightwheel SDK access check (2026-10-05, from the local machine, no credentials, isolated `HOME`, since removed):
- Arena `release/0.3.0` loads its Lightwheel RoboCasa kitchens only through `lightwheel_sdk` (`floorplan_loader.get_usd` in
  `background_library.py`, `object_loader.acquire_by_registry` for objects).
- With no `LoaderUserName`/`LoaderToken` and no cache file, `floorplan_loader.get_usd(scene="robocasakitchen", layout_id=1,
  style_id=1)` against the SDK default host `https://api.lightwheel.net` returned a 57 MB scene with no login prompt. The fixture
  `Microwave039` and the object `apple` (`Apple016`) also downloaded anonymously.
- Earlier note that Gate B/W2b "need an account" was an unverified inference and is retracted for these assets.
- Not checked: other layout/style pairs, other objects, rate limits or quotas, the `api-dev.lightwheel.net` endpoint that Arena's
  Dockerfile sets (the test used the SDK default), and the license terms of the assets. The SDK docs only say "refer to the
  platform's license terms".
- Original RoboCasa (UT Austin, `github.com/robocasa/robocasa`): code MIT, assets CC BY 4.0, about 10 GB through
  `robocasa.scripts.download_kitchen_assets`, no login mentioned. Whether Arena's Lightwheel kitchens are the same assets was not
  verified.

Gripper source found (2026-10-05, read from GitHub, nothing built):
- ARX's own `ARXroboticsX/ARX_Model` repo has `X5/X5A/urdf/X5A.urdf` with a real 2-finger gripper: prismatic `joint7`/`joint8`, 0-0.044 m each,
  `link7`/`link8` 0.0648 kg each. The arm joints in that file have placeholder limits (+-10 rad) and effort/velocity (100/1000).
- The SDK `X5.urdf` we use now has no fingers: `link6` carries a 0.147 m x 0.18 m mesh at 0.581 kg (stock gripper body, inferred from the
  mesh plot and mass, not seen rendered) and `eef_link` is an empty frame 0.145 m out. `ARX_X5/.../x5_2025.urdf` (BSD-3-Clause) also has no fingers.
- X5A total mass is 3.454 kg, matching the 3.45 kg we recorded for the LeggedManip_Lab USD, so that USD was probably X5A with fingers frozen.
- X5A is not the same geometry as SDK X5 (joint5/joint6 origins differ by 2-4 mm; link masses differ).
- `ARX_Model` declares no license. Do not commit or share its meshes; keep in gitignored `third_party/`.
- Real hardware: the SDK treats the gripper as one motor with `gripper_width` 0.082 m, consistent with two 0.044 m fingers (0.088 m total).

Plan revision (user, 2026-10-05): the user will not wait for Lab 3.0 GA (targeted end of October) and wants to update now. Plan rule
"no EA unless GA slips" is overridden by that explicit decision.
- Arena `main` (last commit 2026-09-11): Python >=3.12, Docker base `nvcr.io/nvidia/isaac-sim:6.1.0`, Lab submodule `bb0c8e1b9`, which
  is 228 commits behind Lab `v3.0.0-EA` (EA itself is 210 commits behind `release/3.0.0`). `release/0.3.0` is the Sim 6.0.1 / beta2 line.
- New target: Sim 6.1.0 + Python 3.12 + Arena `main` with its pinned Lab commit, then try Lab `v3.0.0-EA` / `release/3.0.0` at a recorded SHA.
  Whether Arena `main` runs on the EA tag is untested.
- The old 5.1 image is kept only for the lean-attribution checks that need the Lab 2.3.2 checkpoint and for composing the arm USD with the
  existing script. W3 code migration moves before the new-stack pod sessions. W2a is done on the new stack.

## W1 - new-stack gates

Date: 2026-10-05 (overnight orchestrated run). Per the plan revision above, W1 ran on the Lab 3.0 line. Raw reports are under `data/reports/overnight/<session>/`; running notes in `data/reports/overnight/NOTES.md`. Nothing was committed or pushed.

Pinned stack (Arena `main` @ `311d0f0fc6724eb953313f04765f76d947982c40`, Isaac Lab `bb0c8e1b9af381bf13064ec3303e17db79e4b6ef`, isaaclab pkg 16.4.0, Sim 6.1.0-rc.26, Python 3.12.13, rsl-rl-lib 5.4.1, torch 2.11.0+cu128). Pod RTX 4090, driver 570.195.03, about $0.54:
- Render smoke: PASS (Isaac RTX camera, RGB returned).
- Gate A, stock Arena scene: PASS, 16 envs x 200 steps, 40.9 ms/step mean (p95 45.7), peak 3263 MiB (peak memory from the running notes, not re-checked in the JSON). Observation, not explained: the report's mean foot z is -0.77 m with base-minus-foot 0.216 m, i.e. the robot stands on something about 0.77 m below the origin; the cause was not checked (compare with the EA run below, where the robot fell through).
- Gate B, Lightwheel kitchen 1:1: PASS 98 ms/step (57.19 MB SDK download); layout 6:8: PASS 102 ms/step (112.19 MB).
- Fixed-base Go2+ARX5 in the Lightwheel kitchen (Arena mimic smoke): PASS, 16 envs, arm drift 7.6e-5 rad after gravity was disabled on the fixed-base robot.
- `spawn_hold` flat ground, 20 DoF: PASS.
- `train.py` smoke, 3 iterations: PASS. It needed legacy task-path aliases (`common/cerberus_compat/isaaclab_tasks.py`) and thread caps for OpenBLAS.
- Compose (`RobotAssembler`): FAIL on Sim 6.1; see "W3".

Lab `v3.0.0-EA` (`ae37b028ea415c91ea2bc32609efcd759ed2b974`, isaaclab pkg 17.0.2; pods RTX 4090 and L40S, drivers 570-580):
- GRScenes fetch from R2: 1.31 GiB in 48.6 s, OK.
- Gate A, stock Arena scene: exit 0 and `passed: true` in the report, BUT the robot fell through the Galileo background floor (root z -0.73..-0.49 m; report foot z mean -0.60, base-minus-foot -0.04). This pass is NOT meaningful for stock-scene physics.
- Gate A, GRScenes home scene (`MVUHLWYKTKJ5EAABAAAAACY8_usd`) through Arena: FAILED. It stalled about 17 minutes inside PhysX warmup (`ArenaEnvBuilder.make_registered`) with 237 "TriangleMesh: triangles are too big" warnings; the run ended with exit 1.
- Gate A, same GRScenes scene through the `plain_usd` backend (the documented fallback): PASS, 16 envs x 200 steps, 26.9 s, final z 0.253 m.
- Gate B Lightwheel 1:1: PASS (72.8 s wall, 90 ms/step mean). `spawn_hold` on EA: PASS (pitch -15.5 deg). Compose on EA: FAIL.

Net: Arena hosts the Go2 on the stock and Lightwheel scenes at 16 envs without patching. For GRScenes the working path is `plain_usd`. The honest gaps are the hollow EA stock pass, the failed Arena+GRScenes gate, and a second unexplained floor offset on the pinned stock scene. Whether a collision-simplified GRScenes scene would pass through Arena is untested.

## W1.5a follow-ups

Date: 2026-10-05. Old-stack session (pod `cerb-old-fused`, image `cerberus-remote:v3`, Sim 5.1.0 / Lab 2.3.2, RTX 4090, driver 595.91.07, about 1 h, about $0.7).

- X5A arm switch and 20 DoF fused USD: the arm was rebuilt from ARX's X5A model (`ARXroboticsX/ARX_Model`; no license file, gitignored `third_party/arx-model`, do not redistribute) with `joint7`/`joint8` prismatic fingers. Fused USD: 20 DoF, arm 3.455 kg, total 18.47 kg. This replaces the 18 DoF / no-gripper robot from "W1.5a - ARX5 assets" and "W1.5a - pod run".
- Hold metrics (16 envs, 2000 steps): free base height 0.240 m, pitch -12.4 deg, roll -2.6 deg, tilt max 12.6 deg, drift 3.5e-5 m, leg error max 0.58 rad (RL_calf), arm error max 0.042 rad (joint4), gripper closed/open error 1.4e-4 / 1.3e-4 m. Fixed base: height 0.400 m, tilt 0, arm error max 0.044 rad, gripper error 2e-6 m.
- Lean attribution: the stock legs-only Go2 through the same harness leans pitch -11.9 deg, roll -0.07 deg (RL_calf error 0.47 rad, height 0.261 m). The lean is therefore stock Go2 PD sag at stock gains, not the arm (report verdict `go2_pd_hold`, stock pitch 104% of fused pitch). Residual: fused roll -2.6 deg vs -0.07 deg stock was not attributed. Do not move the mount or the stow pose for this.
- Frozen policy `flat_v1_candidate` (ONNX) on the fused robot, driving only the 12 leg joints by name, vs stock Go2 (48 measured envs each): verdict `transfers`. Standing pitch -1.07 deg stock vs -1.33 fused; push recovery 1.0 at 40, 80 and 120 N on both (16 trials each); fall rate 0 everywhere. Weaker spot: 0.5 rad/s yaw command, fused roll mean 6.0 deg, tilt max 21.3 deg (stock 2.8 / 11.5), with lower yaw-rate error on the fused robot (0.22 vs 0.47). Harness caveat: the default fall rule (base height 0.15 m) is an artifact because this policy crouches to 0.11-0.17 m while moving, so the verdict is from a run with `--fall_height_m 0.05`; verdict thresholds are provisional; one pod, one seed; flat ground only. This is not the Phase 0.5 gate.
- Regression: a `train.py` smoke on rsl-rl 3.1.2 exited 0 after the adapter changes.
- Rendering: on driver 595.91.07 with Sim 5.1, RTX rendering SEGFAULTS in `librtx.scenedb.plugin.so` at Hydra engine creation; `play --video` is blocked by a guard. The same render smoke PASSED with Sim 6.1 on drivers 570.195.03 and 570.211.01. `play.py` without `--video` has no end condition (it was killed by its timeout, as expected).

## W2a

Date: 2026-10-05. Fixed-base Go2+ARX5, stock table scene, cubes, on the pinned new stack (pod RTX 4090, driver 570.211.01, $0.74/h). Gate result: PASS with caveats.
- Franka control pipeline (Lab stack-cube Mimic task): 10/10 annotated, 10 generated in 28 attempts (35.7%).
- Our env: 16 envs, fixed base, 20 DoF, 7D action; differential IK error <1e-5 m; gripper reaches 0.044 m open and ~0 m closed within 10 steps.
- Scripted demos: 10 successes in 21 attempts. Auto-annotation: 7/10 (3 skipped because replay did not complete the task).
- Mimic: 20 successes in 155 attempts (12.9%), HDF5 3.37 MB, 20 demos, first action shape (216, 7). The 200-trial expansion was skipped because 12.9% was not promising.
- On EA: 10/10 source, 7/7 annotated, 20/20 generated (3.36 MB). Lab's `annotate_demos.py` reads `args_cli.headless`, which no longer exists on the Lab 3.0 line, so our in-process annotate port is used; `generate_dataset.py` through the Arena interop callback works when the `/tmp` arena-root path exists.

Tuning fixes that made grasps work: gravity disabled on the fixed-base robot (the stowed arm sagged 0.974 rad in 60 steps; EEF drift 0.228 m -> 5e-7 m); gripper stiffness 1000 -> 5000 N/m; cube friction 1 -> 5 (cubes slipped at 1, 0/21); close pause 15 -> 40 steps (fingers stopped at ~0.025 m with 15).

Caveats: friction 5 is unrealistically high; gravity is disabled, which is not valid for a legged base; the gains are provisional. A duplicated invocation once truncated an HDF5 file (lock race); the clean rerun is the one used.

## W2b

Date: 2026-10-05. Status: BLOCKED. The gate (Mimic dataset in the Lightwheel kitchen) was NOT passed.

Setup: fixed-base Go2+ARX5 in the Lightwheel kitchen through Arena, pick object `sweet_potato`, procedural riser and plate. Env smoke PASS at 4 and 16 envs.

Attempt history (scripted recording, 0 successful demos in about 55 attempts over 6 pod runs):
1. Default pick region: 15/15 `descend_pick` timeouts, about 3.15 cm residual.
2. +0.03 m dz: 15/15 done without success.
3. Region moved to x 0.32-0.36 with dz 0.015: 0/2; dz 0.02: 0/2; dz 0.03: 0/5 (fingers open, `grasp_1` false, object not lifted).
4. Yaw alignment added: 0/5, all timeouts in `above_place` with joint6 saturated at -1.57 rad (rotation error 0.13-0.37).
5. Last run: 0/5 (3 done without success, 2 `above_place` timeouts). The object yaw was NOT actually fixed at -0.3 in that run (observed -17.2..+2.3 deg), so the fixed-yaw hypothesis is untested.

Diagnosis: the sweet potato is 125 mm x 52.7 mm (SweetPotato003.usd bbox) against an 88 mm jaw opening, so only a narrow yaw window grasps it, and aligning to that window pushes joint6 into its limit. This is not a stack or robot failure: W2a shows the gripper can grasp, lift and place cubes. The Lab generator error `UnboundLocalError randomized_subtask_boundaries` appears only with an empty annotated file (a consequence of 0 demos, not a cause). The last pod agent stalled past its cap; the orchestrator killed it and deleted pod `o8f3115bx4alq1`, and its artifacts were only partially pulled (`data/reports/overnight/W2bLast/`).

Recommended next step: swap to a cube-like procedural pick object in the kitchen to complete the pipeline; real irregular objects need teleop demos or a grasp planner (headless teleop is unsolved). If the sweet potato stays, first make the yaw randomization actually fixed and re-test.

## W3

Date: 2026-10-05. Target stack: Sim 6.1.0 container (`nvcr.io/nvidia/isaac-sim:6.1.0`, pullable anonymously), Python 3.12.13, Arena `main`, first with its own pinned Lab, then Lab `v3.0.0-EA`. No GA yet.

Versions and SHAs:
- Pinned: Lab `bb0c8e1b9af381bf13064ec3303e17db79e4b6ef` (isaaclab 16.4.0), Arena `311d0f0fc6724eb953313f04765f76d947982c40`, rsl-rl-lib 5.4.1, torch 2.11.0+cu128, lightwheel-sdk 1.0.3, numpy 2.3.5.
- EA: Lab `v3.0.0-EA` `ae37b028ea415c91ea2bc32609efcd759ed2b974` (isaaclab 17.0.2).

Images (built with `docker/remote_v6/build.sh`; the pixi stage was dropped because rclpy is not needed):
- `kevinkyawthiha/cerberus-remote-v6:v6-dev`, 35.4 GB, `sha256:81d8ff200a5952e417743cf9ad4da2b4a16c5ad47af26759546e3b87e9a13929`.
- `kevinkyawthiha/cerberus-remote-v6:v6-ea-dev`, 35.0 GB, `sha256:4377062efde6d1b83798374db8efba5b8233ddbc1f4348442f4c656abefb99c3`.
- `cerberus-remote:v3` (Sim 5.1.0 / Lab 2.3.2 / Python 3.11) is kept: it is still needed for fused-USD composition. Dockerfile fixes found: `ARG ISAACSIM_VERSION` must be redeclared after `FROM`; version-JSON distribution-name normalization.
- Local check before the push: scoped pytest 479 passed / 11 skipped.

Migration findings (Lab 3.0): quaternions are XYZW everywhere (`euler_xyz_from_quat`, `Camera.set_world_poses`, `OffsetCfg.rot`); `data.*` values are Warp ProxyArrays (`common/cerberus_compat/isaac_data.as_tensor`); `Imu` has no `quat_w` on EA (fallback `root_quat_w`); `AppLauncher` no longer registers `--headless`/`--enable_cameras` (`--viz` instead); `isaaclab_tasks.utils.hydra` moved; `cli_args` moved to `isaaclab_rl.entrypoints.backends.cli_args_rsl_rl`; the rsl-rl >=4 runner has no `.log`; old rsl-rl 3.1.2 checkpoints (`model_state_dict`) likely do not load on 5.x (checkpoint-format check added; a real load was not tested); `get_masses` returns Warp arrays; legacy `isaaclab_tasks.manager_based` task paths need aliases. Code moved behind `common/cerberus_compat/{quat,isaac_data,rsl_rl_compat,launcher,isaaclab_tasks}.py`; policy train/play/cli_common/best_checkpoint_runner and the eval scripts use the adapters. Old-stack behaviour was checked (train smoke exit 0 on rsl-rl 3.1.2).

Compose on Sim 6.1: OPEN (later superseded: bypassed by the URDF path, code removed; see "Composition decision confirmed and executed"). `RobotAssembler` no longer has `assemble_rigid_bodies`; the public `begin_assembly`/`assemble`/`finish_assemble` path failed at runtime on pods in four adapter iterations:
1. Anonymous stage: "Cannot create prim at path /go2_arx5 in null or expired layer" in `finish_assemble` `stop_sublayer` (direct-edit branch when the default prim equals the base prim).
2. `Layer.SaveAs` does not exist.
3. Export plus `open_stage`: the same null/expired layer error.
4. Placeholder `/World` default prim: the fused structure appears but the session sublayer is never removed within 120 s.
The USD built on the old stack (`go2_arx5.usd`, 20 DoF, 18.47 kg) loads and holds on Sim 6.1 (pinned and EA), so only rebuilding on 6.1 is blocked. Keep `cerberus-remote:v3` for composition and back up the fused USD. (Superseded: the URDF path no longer needs it.) No internals were patched.

GRScenes subset: R2 prefix `worlds/grscenes/home/` (private; CC BY-NC-SA 4.0) holds scene `MVUHLWYKTKJ5EAABAAAAACY8_usd`: HF revision `4760b9031881c985e3582075cb2d8dbde1530a68`, 1052 objects, 1,406,493,050 B, manifest sha256 `77876dc639285dd77d7129dd2b94a8c41af73e851f69eaa473abafe03adc1357`, 0 unresolved USD dependencies. Extraction ran on an A5000 pod (160 GB, $0.27/h, about 3 h) because CPU pods cap container disk at 20-30 GB; 7z opens the split zip directly (170,029 members). Consumer: `scripts/fetch_grscenes_from_r2.py` (links plus an `env.sh` with `MDL_SYSTEM_PATH`). Lightwheel SDK worked anonymously for the tested assets.

W3 checklist status (from the plan):
- Arena compat for the chosen Lab/Sim: DONE. Arena `main` with its pinned Lab `bb0c8e1b9` runs; Arena `main` on the EA tag also ran (Gate B, W2a pass; stock Gate A there is hollow).
- rsl-rl version and API: DONE for 5.4.1 (adapters); old checkpoints probably unloadable on 5.x, unverified.
- Python 3.12, XYZW, `PhysxCfg` import path, `--headless` replacement: DONE in `common/cerberus_compat/` (verified by the scoped tests and the pod smokes; the `PhysxCfg` move itself was not separately recorded).
- GRScenes tooling on the new Sim: PARTIAL. The scene loads through `plain_usd`; the Arena path fails (PhysX warmup stall). `preprocess.py -n` on Sim 6.x was not recorded.
- Lightwheel assets: DONE (Gate B on both stacks).
- Mimic API: DONE for W2a (with the in-process annotate port on Lab 3.0); W2b blocked.
- ROS2 bridge and distro support in Sim 6.x: NOT CHECKED (rclpy dropped from the image).
- `isaaclab-go2-locomotion` reference version: NOT CHECKED.
- Docker base image and driver/CUDA: DONE for Sim 6.1 image on drivers 570-580; avoid 595 for rendering (Sim 5.1 segfault; not tested with Sim 6.1).

Gate result: NOT fully passed. W1 Gates A and B and the W2a demo pass on the pinned stack. Open: compose on Sim 6.1, W2b, Gate A via Arena on GRScenes, hollow EA stock Gate A, no GA.

## Run record

Pods (all of ours deleted; other-project `dqn-v2-*` pods were never touched):
- `cerb-old-fused`, RTX 4090, driver 595.91.07: about 1 h, about $0.7 (old-stack fused session).
- New-stack gates pod, RTX 4090, driver 570.195.03: about $0.54.
- W2a pod `g3lsuv9jse9of4` (`cerb-new-w2a`), RTX 4090, driver 570.211.01: $0.74/h; duration not recorded in the notes.
- EA sessions (RTX 4090 and L40S, drivers 570-580), W2b retry/yaw/final pods (6 runs for W2b), pod `o8f3115bx4alq1` killed: durations not recorded.
- GRScenes extraction, A5000: about 3 h at $0.27/h (about $0.8).
- Each GPU session was under about $1.1. The account balance went from 35.30 to 16.00 overnight but the account also runs 6-18 other-project pods (about $1.5-3.9/h), so that delta is not attributable to us.
- Pre-W1 snapshot: `data/backups/pre_w1_20261005_0025/`. Nothing committed or pushed; many files are untracked.

Security note: the Runpod API key was printed once in an agent's tool transcript, by a `cat` of `/etc/cerberus_env.sh` on an old-image pod. That file contains the `RUNPOD_*` environment, including the key, because `start.sh` writes it there. The key must be rotated by the user (console.runpod.io/user/credentials); treat it as compromised. No value is recorded here. Avoid it by never cat-ing or printing env files (use `grep -c` or key names only), and by excluding `RUNPOD_API_KEY` from what `start.sh` writes to `/etc/cerberus_env.sh`.

## URDF-merge experiment (composition decision)

Date: 2026-10-05 (overnight, `data/reports/overnight/UrdfPath/`). Question: build the fused Go2+ARX5 by merging URDFs and converting with Isaac Lab `UrdfConverter`, instead of the Isaac RobotAssembler. The decision below was an orchestrator recommendation; the user confirmed it later (see "Composition decision confirmed and executed").

Why:
- The Sim 6.1 public RobotAssembler workflow failed 4 times. NVIDIA made `assemble_rigid_bodies` private in the assembler changelog 3.3.2.
- Hypothesis (strong, unproven): the public begin/assemble/finish workflow targets variant-structured `Robots_Multiphysics` assets, and our flat stock Go2 lacks the IsaacRobotAPI relationships that `finish_assemble` writes.

Alternatives searched:
- Isaac Lab 3.0 has no robot-attach tool. Arena composes at environment level only. Lab issue #2280 (arm on a quadruped) has no maintainer answer. Isaac Sim's own importer pipeline is plain pxr stage operations.
- The one real alternative was URDF merge.

Built offline (all tested, 619 pytest pass):
- `common/cerberus_description/urdf/go2.urdf.xacro`: new argument `sim` (default false; default ROS expansion proven unchanged by a golden test). `sim:=true` gives root link `base`, no rotors/sensors/gazebo/transmission, foot mass 0.04; Go2 total 15.017 kg vs stock 15.019 (Head_upper/Head_lower 0.001 kg each omitted).
- `common/cerberus_description/xacro_expand.py`: expands xacro without an ament index.
- `scripts/build_go2_arx5_urdf.py`: merges sim Go2 + patched X5A arm + fixed `arm_mount_joint` into one `go2_arx5` URDF (27 links, 26 joints, 20 actuated, 18.472 kg).
- `scripts/convert_go2_arx5_urdf_to_usd.py` (pod): converts and checks. `scripts/compare_go2_urdf_to_usd.py`: offline comparison.

Bug found and fixed by the comparison: the vendored Go2 xacro (and upstream unitree_ros xacro) used the FRONT thigh limits (-1.5708..3.4907 rad) for all four legs. Unitree's `go2_description.urdf` and the stock Isaac Go2 use -0.5236..4.5379 for the rear legs. Fixed in `leg.xacro`/`const.xacro`; the real-robot ROS description was affected too. The golden test was updated deliberately (only the RL/RR thigh limits differ).

Offline comparison, merged URDF vs stock-built USD:
- PASS: mass 4e-8 relative, CoM 2e-9 m, inertia within 1e-5 absolute, limits 3e-7 rad.
- Differences: foot inertia 6.4e-6 (URDF) vs 9.6e-6 (stock kept the 0.06 kg foot inertia); fixed foot joints named `*_foot_fixed` vs `*_foot_joint`; calf effort/velocity 35.55/20.06 (URDF) vs 45.43/15.7 (USD); the ArticulationCfg actuators override the latter.

Pod results (pods `cerb-urdf-ea` on the EA stack and `cerb-urdf-old` on the old stack, RTX 4090):
- `UrdfConverter` works on BOTH Lab 2.3.2 and the Lab 3.0 line EA: 1 articulation root, 20 expected joints, no world root joint (floating base works), 18.472 kg. Output is flattened to a self-contained file (the converter writes payload/instance references).
- TCP prim path is `Geometry/base/base_link/link1/.../link6/eef_link`; the stock-built USD has `arx5_arm/eef_link`. `EEF_PRIM_RELATIVE_PATH` and any code assuming the old hierarchy must become name-based.
- Converted-vs-stock-built comparison on the converted USD: PASS on both stacks.

Hold test (EA, 16 envs, 2000 steps, includes gripper close/open), all pass:
- Stock-built free base: pitch -15.51 deg, roll -0.59 deg, base height 0.234 m.
- Converted free base: pitch -15.07 deg, roll -1.14 deg, base height 0.216 m.
- Fixed base: identical (height 0.400 m, tilt 0).
- Mass: 18.474 kg (stock-built) vs 18.472 kg (converted).

Mount discrepancy (new, important):
- Every arm body in the stock-built USD sits exactly 0.07 m LOWER than in the URDF-built one (max body position diff 0.0700 m for base_link, link1..link8; identical on both stacks).
- The spec `GO2_ARX5_MOUNT_XYZ` is (0, 0, 0.07) and the URDF applies it, so the old Sim 5.1 assembler result effectively mounted the arm base at the Go2 `base` origin (z 0), not at 0.07.
- A URDF variant with mount z=0 (`mountz0`) reproduces the stock-built behaviour: height 0.231 m vs 0.234, pitch -15.56 vs -15.51 deg.
- Which z is physically right is NOT settled (Unitree's mounting spec was never checked). `layout.py` (`ARM_BASE_Z = ROBOT_BASE_Z + GO2_ARX5_MOUNT_XYZ[2]`) and the W2a/W2b reach scans assumed 0.07, while the USD the W2a/W2b pods actually used had 0.0. W2a still passed empirically with the stock-built USD.

Frozen-policy test (`scripts/eval_frozen_policy_on_fused.py`, EA stack) caveat:
- NOT a valid discriminator for URDF vs USD: even the stock-built USD gives verdict `fails` on EA (fused standing tilt max 59.7 deg; the stock Go2 itself 22.4 deg standing and 22.7 deg at wz 0.5, vs 2.94 / about 11 on the old stack). The Lab 2.3.2-trained checkpoint behaves worse on Lab 3.0 EA even for the stock Go2.
- Converted vs stock-built on the SAME stack are equivalent: standing tilt max 59.6 vs 59.7 deg, push recovery 1.0 at 40/80/120 N for both, vx 0.5/1.0 fall rate 0.02 vs 0. The `mountz0` variant: standing tilt 38.2 deg.
- The old-stack policy eval for the converted USD was NOT completed (pod killed at the cost cap): "policy transfers on the converted USD on the old stack" is UNVERIFIED. The earlier old-stack result for the stock-built USD (`transfers`) stands.

Cost overrun and cleanup: the session ran past its stated caps. Two pods ran about 2.5 h and 1.7 h at $0.74/h, about $3.1-3.4 of GPU time vs a $3.00 cap. The orchestrator deleted both pods; none of ours remain. Balance 11.41 USD (the account also runs other-project pods, so the balance is not attributable to us).

Decision recommendation (orchestrator; later CONFIRMED, see the section below):
- Adopt URDF merge (xacro `sim` variant + build script + `UrdfConverter`) as the canonical way to build fused embodiments: it works on both stacks, needs no Isaac assembler, is one description for ROS and sim, and exposed the rear-limit bug and the mount discrepancy.
- Keep `scripts/compose_go2_arx5.py` (Sim 5.1 RobotAssembler and the failed Sim 6.1 adapter) as legacy; do not invest further in the Sim 6.1 RobotAssembler path. A plain-USD assembler stays an optional later adapter.

Open decisions for the user (all answered in "Composition decision confirmed and executed"): (1) which mount z is correct (0.07 spec vs 0.0 effective); (2) whether to switch the default fused USD to the converted one and update `layout.py` and the W2 reach scans; (3) re-verify the policy on the converted USD on the old stack and decide what to do about the Lab 3.0 behaviour of the frozen policy.

## Composition decision confirmed and executed

Date: 2026-10-05 (after the URDF-merge experiment). Decisions by the user:
- The URDF path (xacro `sim:=true` + `scripts/build_go2_arx5_urdf.py` + `scripts/convert_go2_arx5_urdf_to_usd.py`, Isaac Lab `UrdfConverter`) is the default way to build fused embodiments.
- Arm mount z is fixed at the spec value `GO2_ARX5_MOUNT_XYZ = (0, 0, 0.07)`. Evidence: the old assembler USD had the arm base at z=0, fully inside the Go2 body shell whose mesh top is 0.089 m; at 0.07 the base bottom is 19 mm below that top and 13 mm above the xacro trunk box top at 0.057 m (if that box is the collision shape; unchecked); LeggedManip_Lab uses 0.07 (USD) and 0.06 (MuJoCo). The real mounting-plate thickness was NOT checked in hardware docs because Unitree's payload page is script-rendered.

Executed:
- Default fused USD `data/assets/go2_arx5/go2_arx5.usd` (+ `.convert.json`) is now the URDF-converted file (Lab v3.0.0-EA, floating base, 20 DoF, 18.472 kg, flattened, self-contained). The assembler-built USDs moved to `data/assets/go2_arx5/legacy/` and are kept only as the golden reference for `scripts/compare_go2_urdf_to_usd.py`.
- The RobotAssembler path was REMOVED entirely per the no-dead-code rule: `scripts/compose_go2_arx5.py` (including its Sim 5.1 branch and the failed Sim 6.1 adapter) and its test. Its version/import helpers moved to `common/cerberus_compat/urdf_converter.py`.
- Prim paths are now constants in `go2_arx5_spec.py` (`GO2_ARX5_BASE_PRIM_REL_PATH = "Geometry/base"`, `GO2_ARX5_TCP_PRIM_REL_PATH`, the `eef_link` under `base_link/link1/.../link6`), used by the W2a/W2b configs, checked by the convert script, and checked by an offline test against the default USD. Body and joint names are unchanged.

Verification gap (explicit):
- NO new pod run was done for this switch (cost decision). The W2a gate result (10/10, 7/7, 20/20 on EA) belongs to the OLD z=0 asset and is NOT re-verified on the new default (arm 7 cm higher, `layout.py` finally consistent with the asset).
- Structure, hold and gripper tests on the 0.07 URDF-converted USD did pass (numbers in "URDF-merge experiment" above).
- Hypothesis, unverified: the z mismatch between the scans (0.07) and the asset used (0.0) contributed to W2b's reach and wrist-limit failures. W2b stays BLOCKED.
- The frozen-policy check on the converted USD on the old stack was NOT completed and is deliberately not needed to close the arm+scenes phase: W4 retrains on `go2_arx5` on the final stack, and rsl-rl 5.x cannot load 3.x checkpoints.

## Open items

- Rotate the Runpod API key (user); change `start.sh` to stop writing `RUNPOD_API_KEY` to `/etc/cerberus_env.sh`.
- W2b: cube-like pick object in the kitchen, then recording and the Mimic chain; fixed-yaw hypothesis untested; teleop or grasp planner for irregular objects.
- Fused-USD composition on Sim 6.1 (four failed iterations): SUPERSEDED, not solved, by the URDF-merge path (see "URDF-merge experiment"); the decision is confirmed and the RobotAssembler code was removed (see "Composition decision confirmed and executed"). `cerberus-remote:v3` is no longer needed to build the fused USD.
- URDF merge decisions (1) mount z = 0.07 and (2) converted USD as default: DONE (see "Composition decision confirmed and executed"). (3) frozen-policy check on the converted USD on the old stack: not completed and not needed; the frozen policy's worse behaviour on Lab 3.0 EA (even for the stock Go2) is handled by the W4 retrain.
- NEW: W2a gate not re-verified on the new default z=0.07 USD (needs one pod run); W2b reach/wrist-limit hypothesis (z mismatch) untested; real mounting-plate thickness unchecked in hardware docs.
- RESOLVED: the prim paths are now `GO2_ARX5_BASE_PRIM_REL_PATH` / `GO2_ARX5_TCP_PRIM_REL_PATH` in `go2_arx5_spec.py`, used by the W2a/W2b configs (old note: code assuming `arx5_arm/eef_link` had to change before the converted USD was used).
- Foot inertia (6.4e-6 URDF vs 9.6e-6 stock), `*_foot_fixed` vs `*_foot_joint` naming and calf effort/velocity limits differ between the URDF-built and stock USD; actuator config overrides the limits, the rest is unreviewed.
- Rear thigh limit fix changed the ROS description of the real robot: review before real-robot use.
- Gate A via Arena on GRScenes: stall in PhysX warmup with oversized triangle meshes; EA stock-scene floor fall-through, and the pinned stock-scene floor offset, unexplained.
- Unfinished W3 items: ROS2 bridge on Sim 6.x, `isaaclab-go2-locomotion` reference, old checkpoint loading on rsl-rl 5.x, GA bump.
- W2a: friction 5, disabled gravity and provisional gains must be revisited before any legged use; Mimic yield 12.9%.
- Hold thresholds still provisional; fused free-base roll -2.6 deg unattributed; yaw-rate roll of the frozen policy on the fused robot (21 deg tilt) deserves a look before W4.
- ARX X5A model has no license: keep out of git and images pushed publicly.
- Commit/push decision pending with the user.
- NEW 2026-10-06: run `--min-cuda-version 13.0` pods for `v6-rel` (torch 2.12.0+cu130 needs driver >= 580);
  driver 580.159.04 on an A5000 renders and steps fine.
- NEW 2026-10-06: Arena on Lab `release/3.0.0` (this image) was NOT re-exercised — Gate A/B and the W2a mimic
  smoke last ran on the EA/pinned Lab. Arena main's own Lab pin (`28a386f0`) is not on `release/3.0.0`.
- NEW 2026-10-06 (supersedes several items above): GRScenes houses are centimetre-authored and Isaac does not convert
  units; use `scene_metric.usd` (automatic through the backends). Re-run anything measured on GRScenes before this fix.
- RESOLVED 2026-10-12 (pod-verified): the admission `render` phase is reimplemented on Isaac Lab's `Camera` sensor and
  PASSES on a pod -- `frame_std` 49.98 against a 2.0 threshold, PNG a properly lit textured room, 0 MDLC errors. It
  needed three fixes found by running it (fully-qualified scene id, physics reset to fire `PHYSICS_READY`, one env for
  one camera prim); see "Render phase on release/3.0.0" and its "PASSED" subsection. `--render` stays off by default.
- RESOLVED 2026-10-07, EXTENDED 2026-10-12: `data/scenes/grscenes/admission.json` now holds **18 home scenes, 18 pass /
  0 fail** (8 from 2026-10-07, 10 more from the 2026-10-11/12 batch, one of which crashed once and passed on a retry --
  a flake). SLAM and Phase 1 read `--spawn-index`/the manifest instead of an `--spawn`. The rclone-1.58.1 problem is
  FIXED: `cerberus-remote-v6:v6-rel2` (built/pushed 2026-10-11) pins rclone 1.75.1, so no per-pod workaround. Still
  open: the other 51 home scenes and all 30 commercial scenes are unswept, and `--combined-env-phases` (worth ~1/3 off
  per-scene time) is implemented but unverified on a pod.
- NEW 2026-10-06: GRScenes needs a per-scene `lidar_proxy.usd` for any LiDAR rig (`scripts/build_lidar_proxy.py`,
  cached); the RAW scene USD cannot be used as a ray-cast target. Proxies are scene-local and gitignored with the
  rest of `data/scenes/`.
- NEW 2026-10-12: the retraining target is REOPENED (`plans/arena_integration_handoff.md` section 3) -- the user may
  train a perception-conditioned / visual-navigation policy instead of the stock flat+rough recipe. That changes whether
  scenes are training data (so the train/held-out split must be locked before the run), the observation interface, and
  the Phase 0.5 attack surface; see that section and "Scene-pool selection" below.
- NEW 2026-10-12: scene-pool selection facts, free for all 69 homes from 2 KB metadata each: wall instances 104-545 and
  pickable objects 2-89, **uncorrelated (r = 0.054)**; ID prefixes are NOT layout families, so no prefix group needs
  holding out. Table `data/reports/scene_admission/scene_pool.csv`. The current 18 homes are object-poor (median 5
  pickables vs 15 for the rest) because selection is smallest-nav-USD-first, which is a weak proxy (44% spread).
- NEW 2026-10-12: GRScenes ships manipulation data we do not use (every home has `start_result_interaction.usd` +
  `interactive_obj_list.json`; rigid bodies authored there, none in the navigation USD we load). This reopens the
  "GRScenes = navigation, Lightwheel = manipulation" split for the chained home mission. See `docs/worlds.md`.
- NEW 2026-10-12: `--combined-env-phases` is implemented (flag OFF by default, 5 tests) but NOT verified on a pod; the
  `.pt` load path's API is verified against rsl-rl 5.5.1 in the image but a real `.pt` load is still untested.
- NEW 2026-10-06: `pin_physx()` is now called by the Cerberus Go2 cfgs, so any *future* training also runs PhysX +
  the legacy actuator path unless someone overrides it; the stock Lab 3.0 reward-weight changes are still inherited
  and unreviewed.

## W4-prep — SLAM readiness on the pinned Lab 3.0 line

Date: 2026-10-06. Goal (user): make the codebase able to run SLAM missions on the new stack in new GRScenes
scenes, testing only what is needed. No retraining. Pod: RTX A5000, driver 580.159.04, SECURE (created with
`--min-cuda-version 13.0` because the new image's torch needs it), about 2.5 h, about $0.7.

### Stack and image

`kevinkyawthiha/cerberus-remote-v6:v6-rel`, 35.1 GB, digest
`sha256:c3443f44003dff5389e29d3d386c292e7a7330db602902207de6ab1960833e5c`:

- Isaac Lab `release/3.0.0` @ `dcabdcdaca0caf3ebc37ae7f92948479c6288239` (isaaclab pkg 27.0.1), not the EA tag.
- Arena `main` @ `34bb35a85c4c3daf8d7c5e521dbeeb6e58dd5943` (isaaclab_arena 0.3.0). Arena's own Lab submodule pin is
  `28a386f0` (EA + 3 commits); that pin and `release/3.0.0` have diverged, so Arena-on-release is a combination this
  image is the first to test — the Gate B/Arena runs above were on EA/pinned, and Arena was NOT re-exercised here.
- rsl-rl-lib 5.5.1, torch 2.12.0+cu130, numpy 2.3.5, warp-lang 1.17.0, Python 3.12.13.
- The pixi `sim-ros2` stage is back (dropped on 2026-10-05): rclpy 7.1.11 (robostack-jazzy), pyzmq, msgpack, and
  `evo` 1.37.1 pip-installed into the same env. `docker/remote_v6/build.sh` now copies `pixi.toml`/`pixi.lock`.
- `start.sh` (both images) no longer writes `RUNPOD_API_KEY` into `/etc/cerberus_env.sh` (the 2026-10-05 leak).
  Key rotation is still the user's to do.

Deviation: `release/3.0.0` moves daily, so the SHA above is the pin; bump it deliberately.

### The "frozen policy tilts on Lab 3.0" mystery: solved, it was Newton

`isaaclab_tasks/core/velocity/config/go2/flat_env_cfg.py` sets `self.sim.physics.default = newton_mjwarp`, and the
Go2 rough cfg sets `self.sim.use_newton_actuators = True`. Nothing in `policy/locomotion/{train,play}.py`, the eval
scripts or the SLAM loop overrode either, so every Lab 3.0 run so far (including the 2026-10-05 EA sessions that
produced "stock Go2 standing tilt 22.4 deg, fused 59.7 deg, verdict fails") was a PhysX-trained checkpoint driving
Newton physics with a different actuator path.

Fix: `common/cerberus_compat/physics.py` (`pin_physx()`), called from the Cerberus Go2 flat/rough env cfgs; it sets
`sim.physics.default` to the `isaacsim_physx` preset and `use_newton_actuators = False`, and is a no-op on Lab 2.x.

Evidence on the pod: the SLAM run's metadata reports `physics_cfg: PhysxCfg` and `use_newton_actuators: false`, and
the frozen old-stack ONNX policy then walked the house with max tilt 6.75 deg and base height 0.135 m above the
floor (previous Newton-era numbers on the same class of run were 22-60 deg). So the old checkpoint is usable on the
new stack for SLAM/plumbing purposes; whether to retrain is now a quality decision, not a compatibility one.
Remaining Lab 3.0 differences that a retrain would inherit (unreviewed): the stock task's reward weights changed
(`feet_air_time`, `action_rate_l2` and others), root/joint reset randomisation differs from 2.3.2.

### GRScenes geometry: empty meshes, centimetres, and the LiDAR proxy

Two blocking facts, both found by running:

- ~~The scene stage is `meters_per_unit = 0.01`; Isaac consumes the USD as a reference and scales by that factor, so the
  proxy must declare the same units.~~ **WRONG, corrected below ("GRScenes is spawned 100x too large").** The first
  half is true (GRScenes is centimetre-authored, and does not even author `metersPerUnit`, so USD reports its 0.01
  fallback); the second half was an unverified assumption and the measurement contradicts it.
- 1115 of the scene's 3230 `Mesh` prims are empty (`Get()` on `points`/`faceVertexCounts` is None; not
  time-sampled): 432/648 under `/Root/Meshes/Base` and 683/1427 under `/Root/Meshes/BaseAnimation`; `Furnitures`
  and `Animation` are fully valid. Isaac Lab's `MultiMeshRayCaster` descends the target prim, hits the first empty
  mesh and dies with `TypeError: iteration over a 0-d array` (`utils/mesh.create_trimesh_from_geom_mesh`). The raw
  scene therefore cannot be a ray-cast target, and the empty meshes are scattered, so no subtree can be excluded.

Fix: a cached LiDAR proxy. `scripts/build_lidar_proxy.py` (pxr only, no Kit app, ~5 s per scene) gathers the 2115
valid meshes in world frame (`common/cerberus_worlds/scene_geometry.py`, which skips empty meshes, handles instance
proxies, XformCache and mesh winding) and writes `<scene folder>/lidar_proxy.usd` (133 MB for this house): one
static `Xform` `/LidarProxy` with a single `Mesh` child, no physics/collision APIs, plus a JSON sidecar
(schema v3, vertex/triangle counts, skipped-empty count, source mtime) used as the cache key.
`slam/slam/eval/scene_env_cfg.apply_scene` spawns it at `/World/lidar_proxy` (invisible) and retargets the LiDAR to
`MultiMeshRayCasterCfg(mesh_prim_paths=[RaycastTargetCfg(prim_expr="/World/lidar_proxy", ...)])`.

Second failure worth recording: the first proxy version made the Mesh itself the USD's default prim, which Isaac
mapped onto `/World/lidar_proxy` such that the prim existed but had no Mesh child, giving `RuntimeError: No mesh
prims found at path: /World/lidar_proxy with supported types: [...]`. The child traversal in
`get_all_matching_child_prims` includes the root prim, so the wrapper (Xform root, mesh below) is what made the
target resolve.

### Remote extraction: Hugging Face 429 killed the home archive

The first limited admission run lost the entire home fetch (74.7 GB split zip, read by HTTP range requests) to
`HTTPError: HTTP Error 429: Too Many Requests` after 8-way parallel reads. `scripts/remote_zip.py` now paces
request starts with a shared token bucket (`$REMOTE_ZIP_RPS`, default 6/s), honours `Retry-After` on 429/503, adds
jitter to the backoff, and defaults to 4 workers. The refetch of 8 home scenes then completed cleanly.
Also fixed: `--link-to` refused to merge into a real directory (the pod already had one scene pulled from R2), so
`link_tree` now merges — symlinking new scene folders and shared dirs into the existing layout, idempotently.

### SLAM smoke: PASS

Command (pod, from `/workspace/cerberus`; the script builds the proxy first if it is missing):

```
bash scripts/run_slam_eval_remote.sh --smoke --mission household --sensor-rig lidar_camera \
  --checkpoint policy/locomotion/checkpoints/go2_locomotion_flat_v1_candidate.onnx \
  --scene grscenes:MVUHLWYKTKJ5EAABAAAAACY8_usd --spawn 0,0,0,0 --timeout 30m
```

Result: exit 0, `/ground_truth/odom`, `/tf`, `/imu/data`, `/front_camera/rgb`, `/front_camera/depth` all 1050
messages (one per control step, 21 s mission at 50 Hz), `/lidar/points` 202 messages (10 Hz over 20.94 s), all
stamps 0.0-20.98 s, bag 1.83 GB `.mcap`, showcase video written. Robot state at the end: base height 0.135 m above
the floor, max tilt 6.75 deg. Artifacts: `slam/slam/eval/results/smoke/2026-10-06_17-46-12/`
(`smoke_check.json`, `sim_loop_report.json`, `bag/bag_0.mcap`, `videos/`).

This is the first end-to-end evidence that the simulated-sensor path (scene USD + walking policy + LiDAR/IMU/cameras
-> zmq -> rclpy relay -> ROS2 bag) works on the new stack in a GRScenes house. No SLAM backend was on the pod
(backends and their images are unchanged by this work).

Caveats: one scene, one seed, `--spawn 0,0,0,0` given explicitly (the manifest did not exist yet); the smoke mode
records no scoring and syncs nothing; the state check thresholds are provisional; `onnxruntime` 1.30.0 ships in the
image's Isaac Python (verified), while the new-stack rsl-rl 5.x `.pt` load path remains UNVERIFIED.

### GRScenes is spawned 100x too large (found by measuring, fixed with a metre wrapper)

This is the most important correction of the session. I had assumed Isaac scales a referenced stage by its
`metersPerUnit`; it does not (USD composition never converts units, and neither does Isaac Lab's USD spawner). On a
pod, spawning the scene exactly as the SLAM loop does and measuring prims in the live stage gave: stage
`metersPerUnit` 1.0, `/World/ground` scale (1, 1, 1), couch **241.1 x 97.3 x 93.7** units, table 200 x 95 x 76.8,
sofa chair 126.9 x 125.4 x 81.9. Those are centimetres (a 2.4 m couch), so in the sim the couch was 241 m long, the
house about 1.3 km wide, and the Go2 a 0.3 m ant inside it.

Consequences, all now understood:
- The 2026-10-06 SLAM smoke "PASS" proved the PLUMBING (scene loads, policy walks, every sensor topic reaches the
  bag) but was physically meaningless: LiDAR ranges, tilt-in-clutter and the 4.9 m walk were all in a 100x world.
- The admission `sample` phase worked in centimetre numbers while treating them as metres (footprint "1317 x 758 m",
  a spawn at x = 477, 2 m grid cells that were really 2 cm, clearance thresholds of "0.8 m" that were 0.8 cm). Its
  spawn poses were therefore wrong for the sim.
- The earlier PhysX `TriangleMesh: triangles are too big` warnings (237 of them) and the Arena stall in PhysX
  warmup on this scene on 2026-10-05 are now explained by the same cause: the triangles really were 100x too big.
  Today's `env` admission child also sat in `build_env` for >6 min. Whether the stall disappears at the right scale
  is being re-tested (see below).
- A frame rendered from a 0.3 m robot in a 100x house looked plausible, which is why the smoke video did not give
  this away.

Fix: `common/cerberus_worlds/scene_geometry.metric_scene_usd()` writes (and caches, keyed on the source's mtime and
units) `scene_metric.usd` beside any scene whose stage is not 1 m per unit: an `Xform` `/Scene` scaled by
`metersPerUnit`, referencing the original by relative path (so its own model/material references keep resolving),
declared 1 m per unit, same up axis. `arena_assets.sim_usd_path()` returns it, and both backends' `describe_scene`
and `load_scene` now put it in `usd_path` (the original stays in `source_usd_path`), so the SLAM loop, the proxy
builder, the admission sampler and `plain_usd`'s `build_env` all see metres. A scene that cannot be opened raises
instead of silently falling back to the wrong scale; without `pxr` (no Isaac, no usd-core) the original path is
returned. The LiDAR proxy is built from the wrapper, so it is metric too (stage 1 m/unit); old cm proxies are
rebuilt automatically because the cache key is the wrapper's mtime. Verified on a pod after the fix: couch
2.411 x 0.973 x 0.937 m, table 2.0 x 0.95 x 0.768 m, sofa chair 1.269 x 1.254 x 0.819 m, `/World/ground` scale 0.01.
Note the prim paths below the scene root change (`/World/ground/Source/Meshes/...` instead of `/World/ground/Meshes/...`).

Everything measured about GRScenes before this fix (Gate A on EA, the smoke's walk distance, the manifest rows) was
at the wrong scale and should be re-run; the Lightwheel/Arena kitchens are not affected by this bug.

### GRScenes materials render flat red: root cause found, fix applied, VERIFIED on a pod

The smoke video's frames are uniformly red (`data/reports/slamready/smoke_run/frame_1.png`), and the runtime logged
`[EntityResolver] FAILED .::OmniUe4Function` / `.::OmniUe4Base` for the scene's `models/...` materials. Cause: the
dataset's own `Materials/*.mdl` all start with `using .::OmniUe4Function import *;` (and 2 of 4 also import
`OmniUe4Translucent`, 2 import `OmniUe4Base`). That is a RELATIVE import, so the modules must sit in the same folder
as the material, but GRScenes ships no `OmniUe4*.mdl` at all (0 of them in the R2 copy, which mirrors the release).
Isaac Sim does ship them, at `/isaac-sim/kit/mdl/core/Ue4/`. A material that cannot compile falls back to red.
Fix: `arena_assets.install_ue4_mdl_modules()` copies the three modules from Isaac Sim into the scene's `Materials/`,
and `ensure_mdl_system_path()` (called by every backend before the app launches) now does it automatically when Isaac
Sim is present. **Verified 2026-10-06:** the metric-scale smoke's showcase video (`data/reports/slamready/smoke_metric/`)
shows a fully textured, correctly scaled home (white shelving with books and vases, a dining table with chairs, a pendant
light, a wall TV, a plant, a marble ledge, a pale-blue wall) and no red. LiDAR/IMU rigs never depended on it.

### SLAM smoke at metre scale: PASS (this supersedes the first smoke's physical meaning)

Same command shape as above, but with the metre wrapper, the metric proxy and the materials fix, from a spawn the metric
sampler produced (`--spawn -4.157,1.327,0,0`):

- Exit 0. `/ground_truth/odom`, `/tf`, `/imu/data`, `/front_camera/rgb`, `/front_camera/depth`: 1050/1050 messages each
  (need >= 525 / 315); `/lidar/points`: 202 (need >= 105); stamps 0.0-20.98 s; bag 2.49 GB; 1050 control steps.
- Physics: `PhysxCfg` / `PhysxManager`, `use_newton_actuators: false` (the `pin_physx` fix, confirmed again).
- LiDAR proxy built from `scene_metric.usd`: `meters_per_unit 1.0`, 2115 meshes used, 1115 empty skipped, 4.5 s, sky dome
  at +-258 m (it was +-25,817 in the 100x world).
- The robot walked from (-4.157, 1.327) to (-0.83, 1.27): about 3.3 m through a 15.2 x 9.6 m house. Final base height
  0.123 m above the floor, final tilt 5.5 deg, MAX tilt 14.9 deg (the 100x run reported 6.75 deg; the difference is
  plausible, real clutter and floor contact now matter, but it is one run of one frozen policy).
- The showcase video is textured and correctly scaled (see the materials section). Artifacts:
  `data/reports/slamready/smoke_metric/` (`household_showcase.mp4`, `smoke_check.json`, `sim_loop_report.json`,
  `lidar_proxy.json`) and `data/reports/slamready/admission_run/` (child JSON and logs of the admission run).

So the simulated-sensor path is now verified end to end at the correct physical scale, with working materials. Still one
scene and one seed; no SLAM backend was on the pod.

### Scene admission (limited): verified at the right scale

Pod run 2026-10-06 (A5000, driver 580.173.02), scene `MVUHLWYKTKJ5EAABAAAAACY8_usd`, after the metre wrapper and the
cameras-on child launch. Every number below is from `data/reports/scene_admission/grscenes/*.json` on the pod:

| Phase | At 100x scale (before) | At metre scale (after) |
|---|---|---|
| `sample` (geometry, spawn search, LiDAR sweep) | 306 s; footprint "1317 x 758 m"; LiDAR 28845/36000 | **44 s**; footprint **15.2 x 9.6 m**; 835 free 0.25 m cells; LiDAR **35999/36000** |
| `env` at 4 envs, 200 steps | sat in `build_env` > 6 min (PhysX warmup), then killed | **ok, 100 s** |
| `env` at 16 envs | never reached | **ok, 90 s** |
| `render` | n/a | **failed**: replicator `Orchestrator failed to schedule a capture after 1000 app updates` |

So the scale bug (above) was also the cause of the PhysX stall: at the right scale the same scene builds and steps at 4
and 16 envs, the spawn search returns plausible metre poses (e.g. (-4.157, 1.327) with 1.12 m clearance, floor z 0.0),
and the whole sample phase is 7x faster. The earlier "child phases die silently" failure was real and is fixed by the
camera-enabled launch (children now run to completion).

Open in this section: (1) the admission `render` phase used `omni.replicator` (`rep.orchestrator.step`), which did not
work here; it was reimplemented on Isaac Lab's `Camera` sensor on 2026-10-11 (see that section) and still needs a pod
run (the phase is optional and off by default); (2) no `admission.json` has been produced by THIS run because it was stopped
before it finished, so only one scene has been measured; (3) the commercial scenes and the other 7 home scenes were
not run at the corrected scale.

Other bugs found while doing this: `run_slam_eval_remote.sh` forwarded `--spawn VALUE`, which breaks for a negative x
(argparse reads a leading `-` as a flag); it now forwards `--spawn=VALUE`. The script's own parser still wants
`--spawn VALUE` with a space.

Spawn poses now come from `common/cerberus_worlds/admission.py`'s manifest
(`data/scenes/<source>/admission.json`), and `apply_scene`/the loop take `--spawn-index` from it or an explicit
`--spawn` override; there is deliberately no silent (0,0,0) fallback for a scene.

### Scene admission sweep, home, 8 scenes — 2026-10-07 (step 1 of the agreed next steps)

Date: 2026-10-07. Pod `cerb-admission` (`4b137lub1zmg3c`), RTX A5000, SECURE, CA-MTL-1, driver 580.178.04, image
`kevinkyawthiha/cerberus-remote-v6:v6-rel` (digest `sha256:c3443f44…`; Lab `release/3.0.0` `dcabdcda`, Arena
`34bb35a85`), 100 GB container disk, created with `--min-cuda-version 13.0`. About 1 h 40 m of GPU time, about $0.45
(plus about $0.15 for the re-runs). Pod deleted; `pod list` empty.

Gate result: **PASS — 8 of 8 home scenes admitted** at the corrected metre scale. Manifest
`data/scenes/grscenes/admission.json` (schema 1, generated 2026-10-07T05:43:37Z, `tool_versions` from
`/etc/cerberus_versions.json`), 40 spawn poses total. R2: the 8 scenes uploaded to the private prefix
`worlds/grscenes/home/` (8.19 GiB, 7861 files, `refs_unresolved: 0`, HF rev `4760b9031881c985e3582075cb2d8dbde1530a68`)
and the manifest to `worlds/grscenes/admission.json`.

Command (pod, `/workspace/cerberus`): `bash scripts/run_scene_admission_pod.sh --categories home --limit 8`
(no `--render`; the 8 smallest home scenes). Artifacts pulled locally to `data/reports/scene_admission/`
(`scene_admission.log`, `finish.log`, `upload.log`, `grscenes/*.json|*.log`, `validate_scenes_grscenes.json`).

Results (all `status: pass`, `shared_scene: true`, phases `sample` + `env` at 4 and 16 envs):

| scene | samp s | build4 s | build16 s | step4 ms | step16 ms | peak GPU MiB | tilt deg | root z min–max m | footprint m |
|---|---|---|---|---|---|---|---|---|---|
| MVUCSQAKTKJ5EAABAAAAACI8 | 33 | 212 | 181 | 20.5 | 20.2 | 6609 | 14.7 | 0.239–0.283 | 14.6 x 13.1 |
| MVUCSQAKTKJ5EAABAAAAADY8 | 37 | 249 | 181 | 22.6 | 21.4 | 6841 | 2.8 | 0.284–0.284 | 14.6 x 12.5 |
| MVUCSQAKTKJ5EAABAAAAAEI8 | 42 | 230 | 189 | 24.7 | 21.6 | 6664 | 15.2 | 0.239–0.263 | 10.1 x 18.2 |
| MVUHLWYKTKJ5EAABAAAAACY8 | 46 | 143 | 145 | 20.0 | 21.0 | 6862 | 16.0 | 0.236–0.253 | 15.2 x 9.6 |
| MWAX5JYKTKJZ2AABAAAAABI8 | 32 | 193 | 167 | 23.3 | 22.4 | 6289 | 15.7 | 0.237–0.278 | 13.9 x 12.4 |
| MWAX5JYKTKJZ2AABAAAAADQ8 | 37 | 167 | 161 | 22.0 | 20.8 | 6943 | 15.7 | 0.234–0.276 | 14.8 x 12.8 |
| MWAX5JYKTKJZ2AABAAAAAEA8 | 32 | 147 | 148 | 25.5 | 19.1 | 7017 | 15.6 | 0.239–0.264 | 15.2 x 11.9 |
| MWBGLKQKTKJZ2AABAAAAAAQ8 | 30 | 190 | 182 | 19.9 | 23.6 | 6691 | 9.9 | 0.241–0.334 | 13.8 x 12.3 |

LiDAR hits 35998–36000 of 36000 in every scene; observations finite; no fall-through.

First pass and the slow-host finding (important for anyone re-running this):
- The first pass returned 0 pass / 8 fail: every scene `env4: timeout after 480.0 s at stage 'build_env'`. This was
  **not** a scene failure. On this host the stage-open was 213 s against 9 s in the 2026-10-06 session (23x), the
  sample phase 253 s against 44 s, and `env` `build_env` 143–249 s against 86 s (`reset_s` 120 s against 0.72 s).
  A single-scene retry with `--build-timeout-s 1200` passed with physics numbers identical to the 2026-10-06 run
  (tilt 16.04 deg, root z 0.236–0.253 m, obs finite), so the default budget (`--build-timeout-s 300` +
  `STARTUP_ALLOWANCE_S 180` = 480 s) is calibrated for a fast host and is too tight on a slow/cold one. Feeding the
  same pod the same 7 first-pass failures with `--build-timeout-s 1200` and a warm page cache gave 8/8 pass. Cost of
  the two extra runs: about $0.15. Recommendation: raise `--build-timeout-s` (or the `STARTUP_ALLOWANCE_S`) in
  `run_scene_admission_pod.sh`, and treat a timeout as retryable rather than as a scene verdict.

Bugs found while doing this (both outside the gate):
1. **The image's `rclone` is v1.58.1, too old for the R2 upload.** `scripts/sync_to_r2.py` documents a v1.59+
   requirement; with 1.58.1 the upload failed with `s3 provider "Cloudflare" not known - please set correctly`, so
   `run_scene_admission_pod.sh`'s upload step reported FAILED even though only the tool was at fault. Installing
   rclone v1.75.1 on the pod made the same command work (8.19 GiB in 4 m 43 s). Fix: bump rclone in the image (or
   pin a known-good provider string for 1.58).
2. **Empty admitted-list guard is wrong.** `run_scene_admission_pod.sh` builds `admitted_<cat>.txt` with
   `print("\n".join(...))`, which is a 1-byte newline when nothing is admitted; `[[ -s "$ids" ]]` is then true, so
   `--from-tree` is called with zero ids and exits with `SystemExit: --from-tree needs --scene-id /
   --scene-ids-file`. Only bites when nothing was admitted (as in the first pass). Fix: test the parsed count, not
   the file size.

Deviation from the plan: the sweep was completed on the same pod after the failed first pass (cached scenes, larger
build budget) instead of re-rolling to another host; no re-roll was needed, and no scene needed re-extraction.

### Fixes landed 2026-10-07 (same session, uncommitted)

- **Timeout budget.** `validate_scenes.py` now has `DEFAULT_BUILD_TIMEOUT_S = 900.0` (was an argparse literal 300)
  with the measurement above as its comment, so a slow/cold host no longer converts a passing scene into a `fail`.
  The wrapper's header records that a timeout row is redone with `--retry-failed`.
- **Empty-admitted-list guard.** `admission.write_scene_ids_file(path, ids) -> int` writes one id per line and an
  EMPTY file for an empty list; `run_scene_admission_pod.sh` branches on the returned count, not on the file size.
  Unit-tested in `common/cerberus_worlds/test_admission.py` (empty case asserted to be 0 bytes).
- **rclone preflight.** `sync_to_r2.check_rclone()` (min 1.59) is called from `_require_env`, the single entry point
  for every R2 write, and duplicated in `fetch_grscenes_to_r2.py`'s standalone fallback. A stale rclone now fails in
  seconds with an explicit message instead of a config-looking provider error at the end of a run.
- **Dockerfile (built 2026-10-11 as `v6-rel2`, see the end of this log).** `docker/remote_v6/Dockerfile` installs a pinned, sha256-verified rclone 1.75.1
  (amd64/arm64) instead of the distro package, and pins `lightwheel-sdk==1.0.3` (what v6-rel resolved) so a rebuild
  differs only by what we changed on purpose. Until that image is pushed, pods need
  `curl -fsSL https://rclone.org/install.sh | bash` once before an R2 write; `docs/runpod.md` says so.
- Verified: scoped `pytest common docker system slam/slam/eval slam/slam/ground_truth scripts` 792 passed / 12
  skipped; `ruff check` clean; both `--dry-run` paths clean; the preflight's fail-fast and the empty-manifest path
  exercised directly.

## Collision/clearance gate for SLAM runs — 2026-10-11 (code only, no pod run)

Reason: handoff next-step 2 ("**add before scoring anything:** a collision check (or per-scene waypoint paths)").
The canned `household` schedule is open loop and generic, so a run can drive into a wall or furniture in a small
room; ATE/RPE from such a run measure the collision, not the backend. The collision check is the version that was
implemented, so waypoint paths are no longer a prerequisite for the first real eval.

What landed (all uncommitted, local):

- `slam/slam/ground_truth/collision_monitor.py` (new, pure numpy): `offending_bodies()` (feet ignored by suffix,
  everything else above `CONTACT_FORCE_THRESHOLD_N = 5.0` N is a collision), `band_clearance_m()` (horizontal
  distance to the nearest LiDAR hit inside `CLEARANCE_BAND_ABOVE_FLOOR_M = 0.05-0.45 m` above the floor, so the
  floor itself is not an "obstacle"), and `CollisionMonitor` (rising-edge collision events, first collision time,
  bodies involved, min clearance, and the `valid_for_scoring` verdict + its reasons).
- `sim_ground_truth_loop.py`: `_contact_offenders()` reads the stock task's existing **unfiltered** `contact_forces`
  sensor (the Go2 cfg keeps it; `base_contact` termination uses it), `_lidar_clearance()` reuses
  `_publish_sensors()`'s staleness trick so clearance is computed only on a genuinely new scan (10 Hz LiDAR vs
  50 Hz control step). The report gains a `collision` block, written before the first step and after every step, so
  a crashed or killed run still carries the verdict.
- `smoke_check.py`: a missing `collision` block, or a report where `contact_sensor` is false, is now a smoke
  failure (the check could not run); actual collisions are printed as `INVALID (collision)` and recorded in
  `smoke_check.json` but do not fail the plumbing smoke, since the schedule is generic by design.
- `scripts/run_slam_eval_remote.sh` (non-smoke path): refuses to score when `valid_for_scoring` is not true
  (`no:collision` etc.), syncs the unscored run to R2, exits 3 and does not self-terminate.
- Tests: `slam/slam/ground_truth/test_collision_monitor.py` (new, 10 tests, runs without torch);
  `test_sim_ground_truth_loop.py` +3 helper tests (torch-gated, so they skip locally and run in the sim env);
  `test_smoke_check.py` +2 failure cases, +1 collision-reporting test, fixture gained the block.
  Verified locally: scoped pytest 806 passed / 15 skipped; `ruff check` clean on every changed file;
  `bash -n scripts/run_slam_eval_remote.sh` clean.

Thresholds are provisional and the whole path is unverified against a live sim: the contact sensor has never been
read on a pod, and the crawl-height crouch of this policy (0.11-0.17 m while moving) is close enough to the
clearance band's lower edge that a calf rubbing the floor could still register as a collision. The first real eval
should be treated as the calibration run; the report lists the offending bodies so a false positive is visible.

Not done here, deliberately: persistent free-cell data in `admission.json` and geometry-derived waypoint paths
(they need pod-side ray casting and a manifest schema bump). The collision gate makes them an improvement in run
quality rather than a prerequisite for a scoreable run.

### Admission `render` phase reimplemented on Isaac Lab's `Camera` (same session, code only)

`scripts/validate_scenes.py::_phase_render` no longer uses `omni.replicator` (`rep.orchestrator.step` had been
failing with "Orchestrator failed to schedule a capture after 1000 app updates"). It now mirrors the SLAM loop's
working sensor path: build the env, construct a `Camera` from `_admission_camera_cfg()` (its prim is authored in
`Camera.__init__` because `cfg.spawn` is set -- verified in the pinned source), run one `env.step()` so the
physics-manager callback builds the render product and the view exists, set the pose with
`set_world_poses_from_view(eye, target)`, then take `RENDER_FRAMES` frames and keep the last one (the first RTX
frames are empty). The child JSON key `frame_std` is unchanged, so `scene_admission_logic.evaluate_render` and the
manifest's `render_frame_std` metric are untouched.

The frame statistics moved into `_frame_stats()` so the child's numbers are unit-testable without a GPU; two tests
cover RGB-only measurement (a blank frame reads 0, a textured one clears `Thresholds.render_min_std`, and a
fully-transparent RGBA frame still reads blank). Verified: `pytest scripts/test_validate_scenes.py
scripts/test_scene_admission_logic.py` 75 passed; `ruff check` clean; `py_compile` clean. Not verified: the sensor on
a real stack -- `--render` remains off by default until a pod run shows a textured frame.


## `cerberus-remote-v6:v6-rel2` built, verified and pushed — 2026-10-11

Why: the pinned rclone 1.75.1 and `lightwheel-sdk==1.0.3` sat in the Dockerfile but in no image, so every pod needed
`curl -fsSL https://rclone.org/install.sh | bash` before an R2 write (a forgotten curl fails the final sync, which
`run_slam_eval_remote.sh` treats as fatal, leaving a billing pod).

Prep: removed local Docker clutter first (superseded `v6-dev`, the dangling image, and the `dqn-ssp-centering` images, at
the user's say-so): free disk 23 GB -> 67 GB. That also pruned the 46 GB build cache, so the build ran from scratch.

First build FAILED at the rclone step, which is the point of building it: the checksums verified (`OK`), but
`rclone version` aborted with `Invalid value when setting --version from environment variable RCLONE_VERSION="1.75.1"`.
A Dockerfile `ARG` is an environment variable inside `RUN`, and rclone reads every `RCLONE_*` env var as a flag. Fix:
the args are now `CERB_RCLONE_RELEASE`, `CERB_RCLONE_SHA_AMD64`, `CERB_RCLONE_SHA_ARM64` (comment in the Dockerfile).

Second build: `TAG=v6-rel2 PUSH=1 docker/remote_v6/build.sh` (Dockerfile default refs = the v6-rel ones), about 32 min wall
time for the rest, pushed. Image `kevinkyawthiha/cerberus-remote-v6:v6-rel2`, 35.1 GB, digest
`sha256:5ab14c76a0230f4ef533991ff5be3b7682c3d4195e91485eaabc8a4a72a81b1b`, id `sha256:e4d51ad63726e5fe3b531f31746152ab20f49a8a4b69f8106f2313c47f2b6235`. Log: `data/reports/docker/build_v6_rel2.log`.

Verified by running the image: `rclone v1.75.1`; `/etc/cerberus_versions.json` byte-identical to `v6-rel`'s (Sim 6.1.0, Lab
`dcabdcda`, Arena `34bb35a85`, rsl-rl 5.5.1, torch 2.12.0+cu130, lightwheel-sdk 1.0.3, Python 3.12.13); the pixi `sim-ros2`
env imports `rclpy`, `zmq`, `msgpack` and `evo_ape` runs; `docker manifest inspect` finds the tag in the registry.
NOT verified: a pod boot from this image (GPU, driver >= 580), an actual R2 upload from it, and the later Dockerfile steps
beyond what the above exercises. `v6-rel` stays on Docker Hub (and locally) as the rollback.

### Render phase on release/3.0.0: three real findings (2026-10-11, pod run)

Attempting to verify the rewritten `_phase_render` on a live pod exposed things the source I had read locally could
not: **the IsaacLab copy in `third_party/IsaacLab-Arena/submodules/IsaacLab` is Arena's pin, NOT the `release/3.0.0`
installed on the pod (`/workspace/isaaclab`, `dcabdcd`)**. For anything Isaac-side, read the POD's source. Two of the
three failures below came from trusting the wrong revision.

1. **A child render run needs the fully-qualified scene id.** `validate_scenes.py --child render --scene-id X` fails
   before Kit starts with `ValueError: Unknown scene 'X'; use 'stock' or 'grscenes:<folder>'`. The parent always
   passes `plan.scene_id`, which is `grscenes:<folder>`; a hand-written invocation must too. (My invocation bug; the
   sweep is unaffected because the parent builds the id itself.)
2. **A `Camera` created after the env is built never initializes.** Sensors register for
   `PhysicsEvent.PHYSICS_READY` (`sensor_base.py::_register_callbacks`), which is emitted once, when physics becomes
   ready (`isaaclab_physx/physics/physx_manager.py:492` inside `reset()`, and again at 1045). A standalone sensor
   created after that event gets no callback, so `is_initialized` stays False forever -- 100 `env.step()`s did not
   help. The first symptom was `AttributeError: 'Camera' object has no attribute '_device'` when calling
   `set_world_poses_from_view`, because `_device` is assigned in `SensorBase._initialize_impl` (line 253 of the pod's
   copy), i.e. only once the callback runs. Fix: `SimulationContext.reset()` re-runs the physics manager's
   `reset(soft=False)` + `play()`, re-emitting `PHYSICS_READY`, so resetting physics after creating the camera
   initializes it. That is public API -- no Isaac Lab file is patched. The in-scene sensors (the SLAM rigs) never hit
   this because the scene manager creates them before `play()`.
3. **Measured exit codes through a pipe are wrong.** `timeout ... isaaclab.sh ... | tail -25; echo $?` reports
   `tail`'s status (0), which hid a child failure. Redirect to a log and read `$?` directly instead.

Also confirmed correct on this revision: `CameraData.output` is `dict[str, ProxyArray] | None` of shape (N, H, W, C)
and the repo's own access pattern `camera.data.output[data_type].torch` matches what `_frame_stats` expects.

#### Render phase: PASSED on a pod (2026-10-11)

Fourth attempt on the A5000: exit 0, `ok: true`, stage `render`, frame `(240, 320, 3)`, `frame_mean` 196.2,
`frame_std` 49.98 against `Thresholds.render_min_std` 2.0, PNG written (80 KB). `render.png` was pulled and eyeballed:
a properly lit, textured room (white shelving with books and vases, a dining table with chairs, a pendant light, a
plant), i.e. no flat red and so no missing-MDL fallback. The scene had been fetched fresh from R2 on this pod, so the
materials path is verified end to end for the admission route too.

Two further fixes beyond the three findings above:
4. **The single camera prim requires a single env.** With the default `--num-envs 4`, `Camera`'s initialization
   raises `Number of camera prims in the view (1) does not match the number of environments (4)`. `_phase_render` now
   sets `args.num_envs = 1` itself rather than depending on the caller (the parent passes `--num-envs 1` too).
5. `set_world_poses_from_view` is called only after `is_initialized`, and a bounded check turns a non-initializing
   sensor into a clear `RuntimeError` instead of an `AttributeError` about `_device`.

Also seen at every Kit startup on this image, harmless here: `Could not load the dynamic library ...
libnvrtc-builtins.so ... cannot open shared object file`. The RTX renderer still loads
(`IsaacRtxRenderer`) and frames come back textured.

### Why GRScenes scenes take ~18 min to validate: a missing MDL module the extractor never pulled (2026-10-11)

Symptom: on the 2026-10-11 sweep the first scene sat in the `env` phase for 10+ minutes and was killed by its per-child
budget, while the *same* scene's `sample` phase took 35 s. The child log held 579 `[rtx.neuraylib.plugin]
[MDLC:COMPILER] comp error` lines and 224 `[omni.rtx.materials]` errors.

Diagnosis (all measured, not inferred):

1. The failing files are the dataset's object materials, e.g.
   `scenes/<id>_usd/models/object/articulated/hearth/<hash>/Materials/MI_*.mdl`, and they fail on lines 6-8, column 10 --
   exactly the `import ::KooPbr::KooMtl;` / `::KooPbr_maps::*` lines (`import ::state::normal;` on line 5 resolves).
   `C120` = module not found.
2. Scale: **280 of the 299** `.mdl` files in the tree import `::KooPbr`, 225 import `::KooPbr_maps`; all 299 import
   `::state` and `::anno`, which do resolve.
3. `KooPbr` was absent from the pod's scene tree AND from Isaac Sim (`/isaac-sim/kit/mdl/core/` ships Base, Blender,
   MaterialX, Nonvisual, OpenPBR, Ue4, VRay, Volume, mdl -- no KooPbr).
4. **It is in the archive.** Listing the home archive's central directory (222,703 members) gives
   `home_scenes/Materials/KooPbr.mdl` and `home_scenes/Materials/KooPbr_maps.mdl`, and all 1,677 `.mdl` members live in
   that one directory. Our selective extraction simply never asked for them: the reference walk resolves **USD
   references**, and these modules are imported by *other MDL source files*, never by a USD or a metadata.json. The
   `UNRESOLVED_FALLBACK_FRACTION` escape hatch (extract the whole `Materials/`) did not fire because every USD
   reference resolved.

Consequence: every material that imports the module fails to compile, RTX retries on each render pass, and a failing
compile is never cached -- so **every launch pays the full cost again**, which is why the same scenes were 3-5 min per
scene on the 2026-10-07 pod (warm cache from repeated runs) and ~18 min here. Not our code's fault and not a
regression, but it made the sweep look like a host problem when it was an extraction gap.

Fix (2026-10-11, `scripts/fetch_grscenes_to_r2.py`): the extraction closure now also follows **MDL module imports**.
`mdl_module_imports()` parses `import ::Mod::Sym;` / `import .::Mod import *;` out of each extracted `.mdl`, and
`module_member()` resolves `Mod` to `<root>Materials/<Mod>.mdl` (falling back to any `/<Mod>.mdl`). The result reports
`mdl_modules`, and the extract-only/upload summaries print it. This supplies what the dataset actually ships, instead
of the two alternatives considered and rejected:
- **stubbing a fake `KooPbr`**: invents an asset the ethos says to borrow, and silently changes the rendered appearance,
  which is an input to the camera-based SLAM backends (`fast_livo2`, `orb_slam3`, `multicam_slam`) -- harness-level
  reward hacking in a project whose contribution is honest evaluation;
- **launching the physics-only phases without cameras**: TESTED and it does not work -- the child hung at `launch` and
  was killed at the 600 s bound with zero MDL errors, confirming (rather than inferring) the note that every launch on
  this image needs `enable_cameras`.

A tree extracted before this fix needs one re-run of the fetch to pick the modules up; `extract_needed()` skips
everything already on disk, so it costs a central-directory listing plus two small files.

#### Second bug, found while verifying: the merge skipped shared directories (same session)

Re-running the fetch with the module closure logged the right thing -- `mdl module closure: 2 module file(s) pulled in
by imports: ['KooPbr.mdl', 'KooPbr_maps.mdl']`, 31.7 KiB -- and then `created 0 relative symlink(s)` /
`merged 0 new entr(ies)`, and the module was still absent from the tree the backends read:

```
data/scenes/grscenes/home/Materials        (real dir)  301 entries
grscenes/home/extract/home_scenes/Materials (real dir) 815 entries   <- KooPbr.mdl is here
```

`merge_into_tree()` only linked TOP-LEVEL entries and per-scene dirs: when the destination already had a `Materials/`
it skipped the directory wholesale, so the 514 material files the extraction had just added were never linked, and
`MDL_SYSTEM_PATH` (which points at the destination) could not resolve `::KooPbr`. **514 of 815 material files were
invisible.**

Trigger: mixing sources on one pod. The 2026-10-07 sweep fetched from HF into a fresh tree, so `Materials/` did not
exist yet and was linked as one symlink (complete). This session pulled one scene from R2 first (for the render gate),
which created `Materials/` and `models/` as REAL directories; the later HF extraction then merged into that layout and
skipped both. So the bug needs an R2 pull followed by an HF extract -- which is exactly what a gate-then-sweep
sequence does.

Fix: `merge_into_tree()` now merges a shared directory FILE BY FILE via the new `link_missing_files()` (recursing into
real subdirs, skipping anything already present, never clobbering, and leaving a symlink it finds alone), so the
destination becomes a complete union of every source that has been fetched and the merge stays idempotent. Whole new
shared dirs are still linked as a single entry. One new test (`test_merge_links_new_files_inside_an_existing_shared_dir`)
covers both the existing-dir and new-dir paths and the idempotency; `scripts/test_fetch_grscenes.py` is 38 passing.

## ~~IN FLIGHT~~ DONE 2026-10-12T01:35Z: unattended admission sweep on pod `2cki2jonaxyye8`

Written so this survives the driving agent's session ending. **COMPLETE**: the supervisor finished at 01:35Z
(upload rc=0, report sync rc=0), the failed scene was retried and passed (18 pass / 0 fail), the corrected
manifest and reports were re-synced, and the pod is idle awaiting deletion. Kept for the record.

- **Pod**: `cerb-admission-a5`, id `2cki2jonaxyye8`, RTX A5000, $0.27/h, image `cerberus-remote-v6:v6-rel2`,
  ssh `root@63.141.33.141 -p 22148` with the automation key. Created 20:31Z; it has NO Runpod API key, so it
  **cannot stop itself**. Terminate it with `runpodctl pod remove 2cki2jonaxyye8` (or the console) once the status
  file says DONE -- an idle pod bills at $0.27/h.
- **What is running**: `/workspace/supervisor.sh` (detached, PID ~12436), which waits for the in-flight chain, then
  continues validation with `--resume --retry-failed --limit 18 --max-runtime-h 3`, re-uploads admitted scenes and
  `admission.json`, syncs the reports to R2, and writes a status file. Bounded to ~02:35Z.
- **Read the status**: `/workspace/cerberus/data/reports/scene_admission/SUPERVISOR_STATUS.txt` (also synced to R2).
  Logs: `/workspace/supervisor.log`, `/tmp/sweep2.log`, `$R/supervisor_validate.log`, `$R/supervisor_upload.log`.
- **Progress at the time of writing**: manifest `data/scenes/grscenes/admission.json` = **10 pass / 10 rows** (the 8
  from 2026-10-07 plus 2 new: `MWAX5JYKTKJZ2AABAAAAACA8_usd`, `MV7J6NIKTKJZ2AABAAAAADQ8_usd`). 8 of the 10-scene
  batch remain. Pace measured this session: 10-26 min per scene, `build_env` alone 291-759 s.
- **Artifacts already saved to R2**: admitted scenes + manifest under `worlds/grscenes/`, and this session's reports
  under `scene_admission/reports/scene_admission/` (19 MB, verified rc=0 at 22:57Z).
- **Also landed this session and NOT yet committed**: the collision/clearance gate, the render-phase rewrite (verified
  on a pod), the `KooPbr` MDL module-closure fix, and the file-level `merge_into_tree` fix. See the sections above.

### `.pt` load path: API verified against the installed rsl-rl (2026-10-11, no pod time)

The rsl-rl 5.x checkpoint path in `slam/slam/ground_truth/sim_ground_truth_loop.py::_load_policy` had never been run.
Rather than discover a mismatch on a paid pod, it was checked against the library actually installed in `v6-rel2`
(`/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.12/site-packages/rsl_rl`, rsl-rl-lib 5.5.1):

- `OnPolicyRunner.__init__(env, train_cfg: dict, log_dir: str | None = None, device: str = 'cpu')` -- matches
  `common/cerberus_compat/rsl_rl_compat.make_runner`'s `runner_cls(env, train_cfg, log_dir=log_dir, device=device)`.
- `OnPolicyRunner.load(path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None)
  -> dict` -- our `runner.load(path)` is valid.
- `OnPolicyRunner.get_inference_policy(device: str | None = None) -> MLPModel` -- our
  `runner.get_inference_policy(device=env.unwrapped.device)` is valid.

So the API shape is right; a `.pt` run should now fail only for substantive reasons. The two remaining unknowns are
(a) the agent-cfg -> dict migration (`migrate_agent_cfg` + `_cfg_to_dict`, whose `handle_deprecated_rsl_rl_cfg` step is
marked UNVERIFIED in the compat layer) and (b) whether a real Lab-3.0-trained `.pt` loads under `strict=True`.

### Next-session task: halve the admission cost by merging the two env phases (design, not implemented)

Measured this session: **10-26 min per scene, and 95% of every launch is `build_env`** (291-759 s), because each
scene runs THREE separate Kit launches (`sample`, `env` at 4 envs x 200 steps, `env` at 16 envs x 50 steps). The two
env launches exist only because the verdict logic labels them differently: `evaluate_env(res, th, "env4")` runs the
stability checks (steps completed, obs finite, root z band, tilt, drift) and `evaluate_env(res, th, "env16",
check_scale=True)` adds the speed/memory limits (`max_step_ms`, `max_gpu_mib`). The stability checks are shared, so
**one run at the scale env count with the longer step count satisfies both** -- and is a strictly stronger test
(16 envs for 200 steps instead of 16 for 50).

Design, kept additive so the 10 existing manifest rows stay valid:

1. `validate_scenes.py`: add a phase `env_combined` (16 envs, `--steps` 200) that records everything both runs record
   today, and stop launching `env_4`/`env_16` when it is enabled. Keep the old phases reachable behind a flag
   (e.g. `--legacy-env-phases`) until the combined one has a track record.
2. `scene_admission_logic.py`: teach `evaluate_scene` that an `env_combined` result satisfies BOTH labels -- apply the
   stability checks and then the scale checks to the same dict -- while still honouring `env4`/`env16` keys for rows
   written before the change. Add the new key to `collect_metrics`; a row is either `env4`+`env16` (old) or
   `env_combined` (new).
3. Tests: the existing `test_scene_admission_logic.py` fixtures make this cheap -- one row built from a combined
   result must produce the same verdict as the equivalent env4+env16 pair, and a combined result that fails a
   stability check must still fail.

Expected gain: **one fewer launch per scene, ~35-45% off the sweep** (3 launches -> 2, or 4 -> 3 with `--render`).
At this session's pace that is 10-26 min -> roughly 7-17 min per scene, i.e. the full 61-home remainder drops from
10-26 h to about 7-18 h. It does not address the `build_env` cost itself; instrumenting `build_env` (stage open vs
PhysX warmup vs RTX init) is the follow-on question, and it needs a pod.

**IMPLEMENTED 2026-10-11 (flag-gated, local tests green, pod verification pending).** `--combined-env-phases` on
`validate_scenes.py` (and the same flag passed through `run_scene_admission_pod.sh`) runs one env launch at the scale
env count for the LONG step count, and `scene_admission_logic.evaluate_scene` applies the stability rules and then the
scale rules to that single result (labels `env` / `env_scale`). Rows keep their shape: a combined row records
`env_phase: combined` with `build_s`/`step_ms` and deliberately omits `build_s_4`/`build_s_16`/`step_ms_4`/`step_ms_16`
so it never claims a 4-env build it did not do; split rows keep `env_phase: split` and the old keys, so the 15 existing
rows stay valid. Default is OFF, so nothing changes until it is asked for. Tests: 5 new (2 in `test_validate_scenes.py`
for the launch sequence and the render skip, 3 in `test_scene_admission_logic.py` for the combined verdict, the
stability/scale failures and the unchanged split metrics); `pytest scripts/test_validate_scenes.py
scripts/test_scene_admission_logic.py` = 80 passed, ruff clean, `bash -n` clean, and `--dry-run` shows the flag
reaching the child. NOT yet verified: an actual combined run on a pod (one scene is enough: same verdict, one fewer
`build_env`).

### Scene-pool questions checked, 2026-10-11 (no GPU; archive metadata + the pod's extracted files)

Prompted by "how many scenes do we need, and does manipulation use these scenes?". Results, with what is NOT known:

1. **Selection bias, corrected.** An earlier hypothesis (smallest-first picks simple homes) was mostly wrong on size
   (nav USD 1.107-1.595 MB over 69 homes, a 44% spread) but RIGHT on objects: the 18 smallest homes have 2-19
   pickable objects (median 5) vs median 15 / max 89 for the rest. Matters if the same homes host pick tasks.
2. **GRScenes carries manipulation data.** All 69 homes: `interactive_obj_list.json` with 32-172 articulated and 2-89
   pickable prims; 45 `toy` and no mail across the dataset. See `docs/worlds.md` for the physics findings (rigid bodies
   authored only in the interaction USD, nested under static groups; articulated assets have joints but no
   articulation root). Our backend ignores all of it today. This reopens the "GRScenes = navigation only, Lightwheel =
   manipulation" split in `cerberus_project_context.md`: the chained home mission needs pickable objects inside a
   home, and the seam between the two pools is undesigned.
3. **Prefix groups are not shown to be layouts.** Test was uninformative by construction (asset reuse 1.00), except
   that `MVUHL5YKTK`/`MVUHLWYKTK` share 209 objects. Needs a geometric test before the held-out split is locked.
4. **The retrain may be scene-trained.** The user may train a perception-conditioned / visual-navigation policy, which
   contradicts the "locomotion trains on procedural terrain only" assumption in handoff section 3 ("recipe DECIDED").
   If so, the split must be locked before training, and scene loading cost matters (build_env 291-759 s per scene on
   this pod; the backend builds one scene per env, no rotation).

### The one failed home scene: segfault in the first step, not a materials or geometry problem (2026-10-12)

Of the 18 home scenes validated on 2026-10-11/12, **17 passed and 1 failed**:
`grscenes:MV7J6NIKTKJZ2AABAAAAADY8_usd`, `env4: error at stage 'step': child exited 1 without a result`.

What the logs actually show (all from `data/reports/scene_admission/grscenes/grscenes_MV7J6NIKTKJZ2AABAAAAADY8_usd_env_4.log`,
also mirrored to R2):

- The faulthandler watchdog dumped three times. The first two (at 240 s and 480 s of process life) put the main
  thread in `build_env` -> `gym.make` -> `ManagerBasedEnv.__init__` -> `_init_sim` ->
  `SimulationContext.reset()` -> `physx_manager.play()` (line 577). Then the log prints
  `[INFO]: Time taken for simulation start Last: 697.499248 s`, i.e. **`play()` took 697 s and completed**.
- The third dump is in the `step` stage, main thread inside `asset_base.py:294 assert_shape_and_dtype` (the
  write-validation helper, called from line 325), and the log ends mid-dump with
  `Segmentation fault (core dumped)`. A dump interrupted by the crash points at native-side corruption, not a Python
  exception: there is no traceback, no `error` in the child JSON, and the `step` stage simply stops.
- The scene is otherwise healthy: the sample phase passed in 47.9 s with 18,542,954 triangles, a 13.7 x 13.2 m
  footprint, 913 free cells and 36000/36000 LiDAR hits.

Ruled out:

- **Materials / the KooPbr bug**: 0 `MDLC:COMPILER` errors in this child's log, unlike the pre-fix runs (579).
- **Scene geometry pathology**: triangle counts across the 18 scenes span 14.4M-35.1M and the FAILED scene sits
  mid-range at 18.5M; the largest scene (35.1M) passed. Sample-phase metrics are unremarkable.
- **A timeout**: the per-child budget is 900 s + 180 s; the crash came after ~717 s of process life, and the reason
  string says `exited 1 without a result`, not `timeout`.

What is NOT known yet: whether this is a flake (host/GPU/PhysX) or a property of this scene. One retry decides it
(`--scene grscenes:MV7J6NIKTKJZ2AABAAAAADY8_usd --resume --retry-failed`, run 2026-10-12T01:36Z). Supporting context
for the flake reading: this host is slow (env4 `build_s` spans 143-885 s across scenes, and this scene had the slowest
build of the session at 706 s), and the crash came immediately after that unusually long `play()`.

**RESOLVED 2026-10-12T01:48Z: it was a flake.** The same scene, same settings, re-run with
`--scene grscenes:MV7J6NIKTKJZ2AABAAAAADY8_usd --resume --retry-failed` (726 s wall) passed both env phases --
`env_4` build 306 s / 200 steps, `env_16` build 310 s / 50 steps, both `ok: true` -- and the row flipped to `pass`.
The manifest is now **18 pass / 0 fail / 0 skip over the 18 home scenes**. Note the same scene's `build_s` swung
706 s -> 306 s between attempts, which is the host variance that made this sweep's pace unpredictable (143-885 s
across scenes). Practical rule for this tool: a single `fail` row is worth one `--retry-failed` before it is believed,
especially when the reason is a crash or a timeout rather than a threshold violation.

### Scene-pool selection: prefixes answered, and two free stratification axes (2026-10-12, no GPU)

Follow-up to the scene-pool questions, all from the archive's `metadata.json` + `interactive_obj_list.json` (2 KB per
scene, read over range requests; no pod, no extraction):

- **Prefixes are not layout families.** Composition similarity (counts of referenced model kinds) is 0.868 within a
  prefix vs 0.872 across prefixes (465 vs 1881 pairs) -- no signal -- and wall counts span 104-381 inside `MVUCSQAKTK`
  and 125-545 inside `MWAX5JYKTK`. A shared floor plan would cluster. So scene-level holdout is safe with respect to the
  id, and the earlier worry (plus the 209 shared objects between `MVUHL5YKTK`/`MVUHLWYKTK`) does not generalize.
  The root layers author no transforms (`xformOp` count 0), so placements live in the referenced assets; an exact
  geometric test would mean composing them, which was not judged worth the bandwidth given the above.
- **Two uncorrelated stratification axes, free for all 69 homes:** wall instances 104-545 (median 206) and pickable
  objects 2-89 (median 9), r = 0.054. Navigation-USD bytes -- what `fetch_grscenes_to_r2.py --scenes N` selects on --
  is a weak proxy (1.107-1.595 MB, 44% spread). Table: `data/reports/scene_admission/scene_pool.csv`.
- **Consequence for the next batch** (when the scene pool is revisited): select stratified on wall count AND pickable
  count, not smallest-first. The current 18 homes are skewed on the second axis (median 5 pickables vs 15 for the rest).

## Lab 3.0 GA upgrade: done and pod-verified (2026-10-25)

Lab 3.0 went GA on 2026-10-20 (`v3.0.0` = `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`). We were on a pre-GA `release/3.0.0` SHA (`dcabdcda`, 147 commits
behind) with Sim 6.1.0. Now on: Sim **6.2.0**, Lab **v3.0.0**, Arena `main` @ `276d9731356e77e974173f6c47cf53d719a764a9` (its `submodules/IsaacLab` pin IS
`e3f7d9ff`, so Lab and Arena finally agree), torch 2.11.0+cu130 (was 2.12.0), rsl-rl 5.5.1, Warp 1.17, Python 3.12.14, isaaclab pkg 28.0.0.
Image `kevinkyawthiha/cerberus-remote-v6:v6-ga`, 34.4 GB, digest `sha256:da1dc4cf73cb1955f062921d35147bf8fc7519d8c7bca26dbde790a16367b19c`
(`/etc/cerberus_versions.json` checked). Rollback: `v6-rel2` (Sim 6.1 / Lab `dcabdcda` / Arena `34bb35a85`). Build log:
`data/reports/ga_upgrade/docker_build_v6_ga.log`.

Pod runs (RTX A5000, driver **595.91.07** on all three pods; about $2 in total; all pods deleted, `pod list` empty):

| Check | Result |
|---|---|
| `train.py --config smoke_test` | exit 0, 3 iterations, finite rewards (Hydra shim + deprecated `AppLauncher` adapter work) |
| `arena_spike_go2.py --scene stock`, 16 envs, 100 steps | PASS (build 13 s, reset 32 s) |
| `arena_mimic_go2_arx5.py smoke` (fused Go2+ARX5, Lightwheel kitchen, Arena mimic env) | PASS |
| Scene render: `validate_scenes.py --child render` on GRScenes `MVUHLWYKTKJ5EAABAAAAACY8_usd` | PASS 2 of 3 launches: frame_std 50.04 and 50.26 (old stack 49.98, threshold 2.0); PNG is a lit, textured room; 0 `[Error]` lines. The 1st launch (cold pod) hung at the `reset` stage until its 900 s cap. |
| `play.py --headless --video --video_length 100 --num_envs 8` | PASS after the video port below: 100 frames, 1280x720 h264, visually correct (Go2s, shadows, command arrows) |
| `push_recovery_showcase_flat.py` (shared multi-clip capture) | PASS after the port: two clips written |

Breakages found by running (all fixed; none were visible from reading the diffs):
1. **Arena `registry_name`** (#1507): `register_asset` asserts `"registry_name" in vars(cls)`. Both embodiments now set `registry_name` next to `name`.
2. **Arena `TaskTerminationCfg`**: `get_termination_cfg` must return `TaskTerminationCfg(timeout_s, success=[CompletionCriteria(...)])`; `WaypointReachTask` ported
   (`partial(waypoint_reached, ...)` is the predicate), the dead termination configclass removed.
3. **Schema fragments**: `spawn.articulation_props` is a list of fragments on GA, so `props.replace(fix_root_link=True)` raised. `fix_root_link` is now a spawner
   field (`go2_arx5_cfg.py` uses it when present, the Lab 2.x path otherwise).
4. **Arena progress tracking**: task success now flows through `env.update_task_progress()` / `progress_tracker`. Our hand-rolled
   `ArenaCompatibleGo2Arx5StackMimicEnv` shim lacked them; it now inherits Arena's `IsaacLabArenaManagerBasedRLMimicEnv` plus our `Go2Arx5StackMimicEnv` methods.
5. **Lab GA reset selector vs Arena** (upstream bug, NOT patched): `env.reset()` passes `env_ids=slice(None)`; Arena's `ResetBackgroundPhysics`
   (`isaaclab_arena/terms/events.py:276`, active on backgrounds with nested physics such as the Lightwheel kitchens) does `torch.as_tensor(env_ids)` and raises
   `TypeError: 'slice' object cannot be interpreted as an integer`. Arena has a TODO for it (Lab PR #8030). Workaround: `common/cerberus_compat/reset.py::reset_all`
   (explicit id tensor), used by `arena_mimic_go2_arx5.py` and `arena_spike_go2.py`. Lab Mimic's own `env.reset()` calls (annotate/generate) are NOT covered: expect the
   same error there until Arena fixes it.
6. **Video / render_mode (the important one)**: on GA `env.render()` returns None for `render_mode="rgb_array"`. `play.py --video` exited 0 with the log line
   "Ignored saving a video as there were zero frames to save" and wrote nothing; `video_capture.record_condition_clips` skipped `None` frames and would have written empty
   clips. Fixes: `script_utils.configure_video_recorders` (Lab's `pre_launch_video_config` + `apply_video_recording`, `env_cfg.video_recorders`), used by `train.py` and
   `play.py`; `capture_viewport_frame` (`sim.visualizers[i].render_rgb_array()`, raises instead of returning None) and `point_viewport_at_env` (the removed
   `viewport_camera_controller` is replaced by the Kit visualizer's `origin_env_index`) used by `video_capture.py`, the terrain/push showcases and the progression video.
   `ensure_capture_visualizer(env_cfg)` injects a headless Kit visualizer in those scripts.
7. **Pre-existing, found on the way**: the showcase and eval scripts never called `install_legacy_task_path_aliases()` (ModuleNotFoundError `isaaclab_tasks.manager_based`
   on any Lab 3.x). Added to all five.

Driver 595: every pod this session (A5000 CA-MTL-1, 4090 RO, A5000 SE) came up on 595.91.07, the branch that segfaulted RTX on Sim 5.1 and that NVIDIA's tracker lists
for Isaac Sim start crashes. torch cu130 needs >= 580, so there was no 570 option and no 580 host was available. Observed on 595 + Sim 6.2: on one pod about half of the
Kit launches crashed ~0.2 s after start (Vulkan path, then hung writing the crash report) or hung at `SimulationApp` start; on the pod used for all rendering results
only one launch in about ten hung (the cold first render). Rendering itself, once Kit is up, was correct every time. `policy/cli_common.check_gpu_driver_for_rendering`
still blocks 595 for `--video`; set `CERBERUS_ALLOW_BAD_DRIVER_RENDER=1` to bypass it (used for the video checks above). Unattended jobs on 595 need a hard
`timeout` and a retry: a hung launch is a stall, not an error. Prefer a pod on 580.x when one is offered.

Changelog items that matter for rendering (Sim 6.2 release notes, Lab 3.0 release): Sim 6.2 known issue "camera gizmos can block RTX sensor rendering" (workaround
`--/persistent/app/viewport/Viewport/Viewport0/scene/cameras/visible=false`; not needed in our runs); `NonVisualMaterial` no longer authors a preview surface;
`get_sensor_reading()` is now an owned snapshot; Lab documents silent black/corrupt RTX images past a buffer limit and fixed an OVRTX eco-mode freeze.
Lab removed `mdp.image` (use `image_rgb` etc.); we use no image observation terms.

NOT verified: other GRScenes homes (only the one known-good scene was re-rendered; the 18-scene admission sweep was not re-run), the SLAM eval end to end
(Camera sensor and LiDAR proxy paths on Sim 6.2), Lightwheel Mimic annotate/generate (the `env.reset()` issue above), `play_terrain_showcase.py` and
`play_training_progression.py` (patched the same way as the flat showcase, but there is no rough checkpoint to run them with), `train.py --video`, the ROS2 bridge on
Sim 6.x, old Lab 2.3.2 checkpoints under rsl-rl 5.x, any render on a 580.x driver. The `body_ids` `isinstance(..., list)` pattern in `push_impulse_event.py` /
`push_recovery_event.py` is a latent bug at the old pin and at GA (Lab finalizes `body_ids` to a tensor for manager terms); not changed.
