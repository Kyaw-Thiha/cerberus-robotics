# Handoff H4: Locomotion + Navigation in GRScenes (ground-truth pose and map)

Two parts: **Part 1 is local (no GPU, no Isaac) and is most of the work; stop at Gate L and wait for the owner's go-ahead.** **Part 2 is one short remote pod session** on a short-lived branch `h4-nav-in-scene` from a fresh `origin/main` clone, merged via PR. Local work goes straight to `main`. Start every session with `git fetch && git pull --ff-only` (stop if it cannot fast-forward); push after each package; never force-push; never read or print `.env`.

## Goal

The frozen Go2 rough policy (`go2_locomotion_ga_v1_candidate`; it failed the adversarial gate, which is recorded and irrelevant here) navigates from A to B in GRScenes household scenes under a classical planner, with **ground-truth pose and a ground-truth-derived occupancy map**. Deliverable: navigation success rate, path efficiency, fall rate and static-collision count over seeded start-goal pairs in 2-3 scenes, plus video clips. This is the Phase 1 navigation core. SLAM-derived pose/map (H4c) plugs in later through the interfaces defined here. **Do not claim onboard-sensing navigation**; state "ground-truth pose and map; simulator-provided height scan" in the report and README.

## Budget and rules

- Pod: **at most about 3 GPU-hours total, one session.** Scene fetch, preprocessing and the occupancy grids (R0c) are CPU work in that same session; they do not add a session, but they must not idle the pod. If the Kit launcher crash seen in H2 (driver-595 flake) repeats after one retry, sync what you have, stop, and report.
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
  **Moved to the pod (2026-11-09 owner decision):** scene assets are downloaded, preprocessed and read on the pod only; nothing scene-sized comes to the laptop, and the scene scripts refuse to run without `CERBERUS_ON_POD=1`. The fetch and the real-grid build now happen in R0c, in the same pod session.
- B3. `planning/` module `occupancy.py`: read static mesh geometry from the scene USD, transform to world space, and rasterize to a 2D grid (default 5 cm cells). A cell is **blocked** if geometry intersects a configurable body-clearance band above the local floor (suggest 0.12 m to 0.45 m; parameters in config, not hardcoded), **free** otherwise, **unknown** if outside the scene extent. Output `.npz` (grid, origin, resolution, band params, scene id) next to the scene, and a small PNG preview. Handle floor height changes reasonably, and say what it does with multi-level scenes (single-floor selection is acceptable; document it).
- B4. **Inflation**: dilate blocked cells by a robot radius parameter (Go2 about 0.7 x 0.31 m standing; start with a circle radius from config and report the choice).
- B5. Local tests on synthetic grids; the real-grid smoke test and the kinematic run on real grids are R0c on the pod (scene files exist nowhere else). Commit only the small PNG previews and the grid parameters, not the scenes.

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

- **R0. Bring-up.** Per `docs/runpod.md` and `docker/remote_v6`; fresh clone of `origin/main`; branch `h4-nav-in-scene`; write `outputs/env_report.json`; run the preflight checks on the pod; pull the candidate checkpoint/ONNX from R2.
- **R0a. Host gate, then launch probe -- before any download.** `scripts/pod_driver_gate.sh` first (allowlist in `scripts/pod_driver_allowlist.json`: 580.x >= 580.95.05, or exactly 595.58.03). On reject it records the verdict and exits 2; the pod holds no data, so terminate it. Then `scripts/pod_launch_probe.sh`: smallest headless Go2 launch, 5-minute cap, verdict OK/SEGFAULT/TIMEOUT/OTHER in `outputs/launch_probe.csv`. A non-OK verdict stops the session; two non-OK verdicts on different hosts stop everything -- do not keep retrying hosts.
- **R0b. Credentials check FIRST, before downloading anything.** On the pod, verify R2 access with a listing or a HEAD of one small object. If it returns 403, **STOP immediately**, sync nothing, and report: do not burn GPU time diagnosing it. H2's pod fetch worked earlier, so a 403 means the token or the bucket permissions changed, not that the code is wrong.
- **R0c. Scene fetch, preprocessing and the real occupancy grids (CPU only, no simulator).** In the same pod session as everything below -- do not open a second session. In order:
  1. fetch the 2-3 scenes (`scripts/fetch_grscenes_from_r2.py`, `CERBERUS_ON_POD=1`);
  2. run the existing preprocess (the `plain_usd` metric wrapper, as H2 did);
  3. run `scripts/build_occupancy.py` on each scene: writes `<scene>/occupancy.npz` and a PNG preview;
  4. commit the small PNG previews, the `.npz` files (they are small) and the grid parameters; **never** commit scene USD;
  5. run the kinematic closed-loop test on the REAL grids (`scripts/kinematic_eval.py --grid <npz>`, CPU, seconds) and record per scene: reachability, rejected pairs, kinematic success, mean path length.
  - **Gate G.** Look at the previews. A grid is clearly wrong if walls are missing, the floor reads as blocked or unknown, doorways are closed, or reachability is under 50%. Diagnose from the previews; fix on the pod if it is quick (under about 15 minutes), otherwise sync and stop and report. **Do not start the physics runs on a bad grid.**
  - The grid stage is CPU work on a rented pod: keep it short and do not let the pod idle while results are inspected.
- **R1. Stand test and the scene-2 diagnosis in parallel, two tmux sessions.** After R0a passes:
  - session A (Kit): scene 1 stand test, **1 env** (4 envs cost ~12 minutes to build on a real house and leave no room for a retry), zero command, 200 steps -> `assertions.json`. **Gate R1:** joint order OK, 235-D OK, no NaN, robot stands. If not, record the exact diagnosis and stop; do not tune.
  - session B (CPU, no simulator): `scripts/build_occupancy.py --diagnose` on scene 2 -> reason-coloured preview, up-facing height histogram and JSON counts. Read it against the Gate G failure already recorded (whole rooms blocked; 36/79 pairs unreachable).
  - **Gate G rule:** physics runs only on scenes whose previews pass. If scene 2 is not fixed within 15 minutes of starting session B, substitute a third cached scene (the list is in `docs/h4_report.md`) or continue with scene 1 only. Do not tune the policy or the planner.
  - Rationale: the build and the diagnosis are different resources (GPU vs CPU), so running them together is the only way to keep the pod from idling while a grid is inspected.
- **R2. Scripted-path smoke**, on the scene(s) that passed Gate G, short: validates the env, the command hook and the video helper. Cap 10 minutes.
- **R3. Navigation eval**, the scenes that passed Gate G (2-3 if available), N seeded start-goal pairs per scene (start with N=20; reduce if GPU time is tight), same seeds for every scene. Cap 90 minutes of GPU time.
- **R4. Wrap-up.** Update `docs/reports/scene_eval_phase0.md` and write the final `docs/h4_report.md` (per-scene success/SPL/fall/timeout/contact numbers, kinematic upper bound vs physics result, assertions measured, GPU minutes vs the 3-hour cap, failure diagnoses with clips, deferred items, contradictions). Sync everything with `scripts/sync_to_r2.py` and print the manifest. Tell the owner the pod is safe to terminate; **do not terminate it and do not leave it idle**.

## Reporting rules

- Report poor results plainly; do not tune or retrain to improve them here.
- If a result looks too good or too bad, state what you checked to rule out a measurement bug (for example, success counted despite a fall, or the controller reaching the goal through geometry).
- Report the gap between the kinematic upper bound and the physics result; that gap is the finding.

## Out of scope

SLAM or any estimated pose/map (H4c), dynamic or unmapped obstacles, learned navigation policies, ROS wrappers for the planners, SCAN-Planner/VOP-Nav/HiPAN/RAEM integration, estimated-map or corrupted-scan experiments, retraining or robustifying any policy, `go2_locomotion_final.pt`, Arena/Lightwheel, arm or `go2_arx5`.
