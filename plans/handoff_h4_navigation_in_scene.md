# Handoff H4: Locomotion + Navigation in GRScenes (ground-truth pose and map)

Two parts: **Part 1 is local (no GPU, no Isaac) and is most of the work; stop at Gate L and wait for the owner's go-ahead.** **Part 2 is one short remote pod session** on a short-lived branch `h4-nav-in-scene` from a fresh `origin/main` clone, merged via PR. Local work goes straight to `main`. Start every session with `git fetch && git pull --ff-only` (stop if it cannot fast-forward); push after each package; never force-push; never read or print `.env`.

## Goal

The frozen Go2 rough policy (`go2_locomotion_ga_v1_candidate`; it failed the adversarial gate, which is recorded and irrelevant here) navigates from A to B in GRScenes household scenes under a classical planner, with **ground-truth pose and a ground-truth-derived occupancy map**. Deliverable: navigation success rate, path efficiency, fall rate and static-collision count over seeded start-goal pairs in 2-3 scenes, plus video clips. This is the Phase 1 navigation core. SLAM-derived pose/map (H4c) plugs in later through the interfaces defined here. **Do not claim onboard-sensing navigation**; state "ground-truth pose and map; simulator-provided height scan" in the report and README.

## Budget and rules

- Pod: **at most about 3 GPU-hours total, one session**. If the Kit launcher crash seen in H2 (driver-595 flake) repeats after one retry, sync what you have, stop, and report.
- **No retraining, no policy changes, no limiter tuning, no robustification.**
- Every eval records video clips (rule below) and syncs them.
- Anything that contradicts this handoff: report and stop at the next safe point.

## Read first

`docs/reports/scene_eval_phase0.md` (what failed in H2, the recorded resume point), `docs/h2_report.md`, `docs/h3_report.md`, `docs/worlds.md`, `docs/perception.md` (planner candidates), `common/cerberus_worlds/` (`base.py`, `backends/plain_usd_backend.py`), `planning/global_planner/` and `planning/local_planner/` (existing candidate registries and tests), `policy/locomotion/core/platforms/go2/env_cfg.py`, `common/cerberus_policies/contracts/heightscan_ppo_v1.yaml`, `common/cerberus_policies/command_limiter.py`, `policy/locomotion_policy/` core, `policy/locomotion/adversarial_robustness/attack_env.py` (example of driving the frozen policy and injecting commands), existing rollout-recording code under `policy/locomotion/eval/`, `scripts/preflight_local.sh`, `scripts/arena_spike_go2.py` (standing check), `docs/runpod.md`.

## Key design point (the H2 blocker)

The existing `plain_usd` env builds a 33-D observation with no velocity command; the policy needs its 235-D training observation (3+3+3+3+12+12+12 + 187 height-scan cells). **Do not reshape the policy.** Build the scene env from the Cerberus Go2 rough **Play** env cfg and swap its terrain for the preprocessed scene USD (terrain importer USD type), keeping the observation group, noise-off play settings, action term and decimation identical. Point the height-scan ray-caster at the scene's static geometry (floor, walls, furniture) so obstacles appear as height changes. Inject the velocity command externally each step through a command hook (as in `attack_env.py`). If `cerberus_worlds` cannot express this, add a `loco_velocity` task variant to `plain_usd_backend.build_env`; policy and planning code must never import a backend directly. All envs share one global terrain: confirm inter-robot collision filtering or use few envs with distant spawn points.

## Part 1: Local (no GPU)

### Package A: Env builder, scripted path, metrics, video, assertions

- A1. `docs/reports/loco_in_scene_design.md` (one page): env composition, which prims the ray-caster scans, the command hook, per-env reset to a start pose, metrics, and what is verified locally vs on the pod.
- A2. Env builder and driver (`scene_eval.py`-style) for the pod: `--scene`, `--num_envs`, `--seed`, `--steps`; per-env reset to a given (x, y, yaw); loads the policy the same way the H2 attack env did successfully (say which path); outputs `assertions.json`.
- A3. Pure-Python **scripted velocity path generator** (waypoints to command time series, ramps via `CommandRateLimiter`, per-axis limits from the contract, deterministic by seed). Used as a smoke test on the pod.
- A4. Pure-Python **metrics** module: fall (base-contact/orientation termination), path completion, time, static-geometry contact count if available, standing check (`base_minus_foot_z <= 0.15` style, as in the arena spike), NaN detection; JSON per scene and aggregate CSV.
- A5. **Free assertions**, recorded in the first sim step: (1) articulation joint order vs contract and `platforms.yaml` (fail loudly on mismatch); (2) observation dim exactly 235 and term order per contract; (3) height-scan convention on flat ground (record the constant and clip bounds hit; report the measured value, do not silently edit the contract); (4) no NaN/inf in a 200-step zero-command stand.
- A6. **Reusable video helper** used by every driver you write: N short low-res headless clips per run (default 3), follow camera, always keeps the first success and the first failure if both occur, named `<scene>_<seed>_<outcome>.mp4` plus a metrics JSON, configurable count/resolution, off-switch flag for pure-metrics runs. Reuse existing recording code; do not write new logic if it exists. Add a "Run artifacts" rule to `docs/runpod.md`: every pod eval records clips and syncs them; failure clips are mandatory.
- A7. Extend `scripts/preflight_local.sh` with import/dry-run checks for the new modules. Local tests for A3, A4, A6 naming/selection, and the assertion logic (mocked joint lists).

### Package B: Scene geometry to occupancy map (local, no Isaac)

- B1. Add `usd-core` (`pxr`) as an optional pixi feature (for example `geometry`) so the default env is not bloated; re-lock; keep CI green. If it cannot install for Python 3.12 here, say so and fall back to building the grid inside the Isaac eval (ray-casts) while keeping the same output format.
- B2. Fetch 2-3 already-preprocessed GRScenes homes to gitignored `data/scenes/` using the existing fetch scripts (R2 credentials come from the environment; never print them). Do not commit scene assets.
- B3. `planning/` module `occupancy.py`: read static mesh geometry from the scene USD, transform to world space, and rasterize to a 2D grid (default 5 cm cells). A cell is **blocked** if geometry intersects a configurable body-clearance band above the local floor (suggest 0.12 m to 0.45 m; parameters in config, not hardcoded), **free** otherwise, **unknown** if outside the scene extent. Output `.npz` (grid, origin, resolution, band params, scene id) next to the scene, and a small PNG preview. Handle floor height changes reasonably, and say what it does with multi-level scenes (single-floor selection is acceptable; document it).
- B4. **Inflation**: dilate blocked cells by a robot radius parameter (Go2 about 0.7 x 0.31 m standing; start with a circle radius from config and report the choice).
- B5. Local tests on synthetic grids and, when scene files are present, a smoke test on the real grids (skip otherwise). Commit only the small PNG previews and the grid parameters, not the scenes.

### Package C: Planning interfaces and global planner

- C1. Tiny interfaces in `planning/` (pure Python, no ROS/Isaac): `PoseSource` (returns x, y, yaw) and `MapSource` (returns the occupancy grid), each with a ground-truth implementation now. **These are the seams where SLAM plugs in later (H4c).** Document them in `docs/interfaces.md`.
- C2. Real global planner: **A\*** on the inflated grid (8-connected, optional path shortcutting/smoothing) registered in the existing `planning/global_planner` candidate registry, with tests (paths exist, are collision-free on the grid, optimal on toy grids, correct failure when unreachable).
- C3. Seeded **start-goal sampler**: free-space pairs with a minimum geodesic distance (for example 3-10 m), reachable per the planner. Record how many pairs were rejected as unreachable per scene.
- C4. Local path-planning visualizations: grid, inflated grid, sampled paths overlaid; commit a few small PNGs.

### Package D: Local controller (pure Python)

- D1. **Pure-pursuit** path follower registered in `planning/local_planner`: input pose + path, output a `LocoCommand`-style dict (vx, vy, wz) with lookahead, slowdown near turns and at the goal, and goal tolerance (default 0.5 m, configurable). Output goes through `CommandRateLimiter`.
- D2. **Closed-loop kinematic test** (no physics): simulate a unicycle model with the limiter in the loop and run the controller over the A\* paths on the real grids; report kinematic success rate and path efficiency. This validates planner plus controller before any GPU time and is an upper bound for the physics runs.
- D3. Tests for lookahead selection, goal detection, limits and the limiter interaction.

### Package E: Navigation eval driver and docs

- E1. Nav eval driver for the pod: for each scene and seeded start-goal pair, reset to start, plan with A\*, follow with pure pursuit through the policy, and record: success (within tolerance before timeout, no fall), SPL-style path efficiency, fall, timeout, planner failure, static-geometry contact count, time. JSON per run, aggregate CSV, clips via the A6 helper.
- E2. Update `docs/project_structure.md`, `docs/interfaces.md`, `docs/perception.md` (planner status), `README.md` (Phase 1 row: "in progress, ground-truth pose and map, simulator height scan"), and `docs/h4_report.md` (interim: Part 1 results incl. kinematic success rate, planner stats per scene, what Part 2 will run and its expected GPU minutes).
- **Gate L:** pytest, lint, backend-table/other freshness checks, preflight, colcon build of changed ROS packages (if any) all green; previews committed; pushed to `main`. **STOP and tell the owner Part 1 is ready.** Do not launch a pod. The owner must have pushed H3 and topped up the balance first.

## Part 2: Remote (only after the owner says go)

- **R0. Bring-up.** Per `docs/runpod.md` and `docker/remote_v6`; fresh clone of `origin/main`; branch `h4-nav-in-scene`; write `outputs/env_report.json`; run the preflight checks on the pod; pull the candidate checkpoint/ONNX and the scene assets and occupancy grids from R2 or rebuild them locally-equivalent (grids are small).
- **R1. Stand test**, 1 scene, 16 envs, zero command, 200 steps: `assertions.json`. **Gate R1:** joint order OK, 235-D OK, no NaN, robot stands (no sinking or fall-through). If not, record the exact diagnosis (scene collision, spawn height, physics preset) and stop; do not tune.
- **R2. Scripted-path smoke**, 1 scene, short: validates the env, the command hook and the video helper. Cap 10 minutes.
- **R3. Navigation eval**, 2-3 scenes, N seeded start-goal pairs per scene (start with N=20; reduce if GPU time is tight), same seeds for every scene. Cap 90 minutes of GPU time.
- **R4. Wrap-up.** Update `docs/reports/scene_eval_phase0.md` and write the final `docs/h4_report.md` (per-scene success/SPL/fall/timeout/contact numbers, kinematic upper bound vs physics result, assertions measured, GPU minutes vs the 3-hour cap, failure diagnoses with clips, deferred items, contradictions). Sync everything with `scripts/sync_to_r2.py` and print the manifest. Tell the owner the pod is safe to terminate; **do not terminate it and do not leave it idle**.

## Reporting rules

- Report poor results plainly; do not tune or retrain to improve them here.
- If a result looks too good or too bad, state what you checked to rule out a measurement bug (for example, success counted despite a fall, or the controller reaching the goal through geometry).
- Report the gap between the kinematic upper bound and the physics result; that gap is the finding.

## Out of scope

SLAM or any estimated pose/map (H4c), dynamic or unmapped obstacles, learned navigation policies, ROS wrappers for the planners, SCAN-Planner/VOP-Nav/HiPAN/RAEM integration, estimated-map or corrupted-scan experiments, retraining or robustifying any policy, `go2_locomotion_final.pt`, Arena/Lightwheel, arm or `go2_arx5`.
