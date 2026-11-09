> **ARCHIVED 2026-11-08 — byte-identical duplicate of `docs/status_report_2026-11-06.md`, which is the
> canonical copy (`README.md` links it). Kept only so no history is lost; do not update.**

# Cerberus Status Report (2026-11-06)
Machine: local (Arch Linux, no Isaac/GPU run)   Branch: main   HEAD: 6b2ed37   Last commit: 2026-11-06 17:15:50 -0500

## Summary (max 10 lines)
- Phase 0: **trained** — Flat and Rough Go2 checkpoints, eval CSVs, video, tensorboard exist locally.
- Phase 0.5: **not started** — no `adversarial_robustness/`, only `plans/handoff_phase0.5.md`.
- **cerberus_worlds**: implemented (ABC + registry + both backends, 7 test modules); no `PolicyBackend`.
- **cerberus_msgs / description**: msgs has only `TaggedMap.msg`; description has `GO2_ARX5_CFG` but no `ARX5_CFG` symbol.
- **Arena W1**: 2026-11-06 spike JSON says `passed: true` but the robot sank (`robot_z_min_max [-0.73,-0.48]`); the log's new standing check makes that a FAIL, and the smoke default moved to `plain_usd`.
- **Docs**: 14 stale/contradictory items listed below.
- Biggest blocker I can see: Phase 0.5 is unbuilt, so there is no `go2_locomotion_final.pt` to freeze for Phase 1.

## Findings by section (1-9 above)

### 1. Repo basics — DONE
- `git status --porcelain`: empty (clean). Branch `main` tracks `origin/main`.
- Remote: `git@github.com:Kyaw-Thiha/cerberus-robotics.git` (origin).
- HEAD `6b2ed37 smoke: run the spike step on the ground-plane control, not Arena's Galileo background`; last commit 2026-11-06 17:15:50 -0500.
- Last 30 commits include the 2026-11-06 GA-compat/arena-smoke series, 2026-10-25 GA pins, 2026-10-11 scene/SLAM work, 2026-10-06 URDF merge.
- No uncommitted or non-ignored untracked files (`git status --porcelain -uall` = 0 lines).
- Top level: `.claude/ .env .env.example .gitattributes .gitignore .pixi/ .pytest_cache/ .ruff_cache/ .superpowers/ README.md artifacts/ build/ common/ data/ docker/ docs/ install/ log/ perception/ pixi.lock pixi.toml planning/ plans/ policy/ pytest.ini scripts/ slam/ system/ third_party/`.
- Requested paths:
  - `common/cerberus_policies/` — **MISSING**
  - `policy/whole_body/` — **MISSING**
  - `common/cerberus_worlds/` — **EXISTS**
  - `common/cerberus_msgs/` — **EXISTS**
  - `common/cerberus_description/` — **EXISTS**
- Tree vs `docs/project_structure.md`:
  - Exists and matches: `common/{cerberus_msgs,cerberus_description,cerberus_worlds,cerberus_compat}`, `perception/terrain_perception`, `slam/slam`, `planning/{global_planner,local_planner}`, `policy/locomotion/{train.py,play.py,eval,checkpoints,core}`, `policy/manipulation/go2_arx5_stack`, `policy/locomotion_policy`, `system/cerberus_bringup`, `docker/{remote,remote_v6,slam}`, `scripts/`, `docs/`.
  - In doc, missing on disk: `colcon_defaults.yaml` (line 92), `docs/architecture.svg` (line 163), `perception/object_perception/` (121), `planning/task_orchestrator/` (128), `policy/manipulation/{demo_collection,algorithms,continual_learning,world_model,eval}/` (140-144), `policy/manipulation_policy/` (146), `safety_filter/` (148-153), `actuation/` (155-156), `system/{scripts,systemd}/`.
  - Exists, not in the doc tree: `plans/`, `third_party/`, `artifacts/`, `.claude/`, `.superpowers/`, `docs/superpowers/`, `docs/archive/`, `policy/locomotion/core/{platforms,tests}/`, `slam/slam/{ground_truth,eval,backends}/`, `policy/locomotion/runs/`.

### 2. Tooling and environment — PARTIAL
- `pixi.toml` **present**: `python = "3.12.*"`, ROS Jazzy base, colcon, ruff, pytest `<8`, matplotlib, pandas, xacro, onnxruntime, opencv, msgpack, pyzmq. Tasks: `build = colcon build --symlink-install`, `lint = ruff check .`, `test = pytest`. SLAM features/environments (`slam-fast-lio2`, `slam-bievr-lio`, `slam-orb-slam3`, `slam-multicam-slam`, `slam-fast-livo2`, `slam-gs-livo`, `sim-ros2`) and default env.
- `pixi.lock` **present** (1.1 MB). Lock contents not enumerated.
- `colcon_defaults.yaml` **MISSING** (doc claims it exists).
- `docker/remote/Dockerfile` **present** (old stack): `ISAACSIM_VERSION=5.1.0`, `ISAACLAB_VERSION=v2.3.2`, base `nvcr.io/nvidia/isaac-sim:5.1.0`, rsl-rl-lib `3.1.2` via Lab's `isaaclab_rl/setup.py`.
- `docker/remote_v6/Dockerfile` **present** (new stack): `ISAACSIM_VERSION=6.2.0`, `ISAACLAB_REF=e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4` (v3.0.0 GA), `ARENA_REF=276d9731356e77e974173f6c47cf53d719a764a9`, `ISAACLAB_INSTALL_SPEC=all`, `lightwheel-sdk` pinned by ARG. Torch `2.11+cu130` and rsl-rl `5.5.1` are **comment-stated transitive Lab pins**, not explicit Dockerfile pins (no `torch==`/`rsl-rl==` grep hit).
- `scripts/sync_to_r2.py` **present** (226 lines); `docs/artifact_storage.md` **present**.
- CI: **none** (`.github/`, `.gitlab-ci.yml` absent). Pre-commit: **none**. Ruff config file: **none** (`ruff.toml`/`pyproject.toml` absent; lint runs with ruff defaults).
- Local tests run with `pixi run pytest -q`: **839 passed, 15 skipped, 2 warnings, 3 collection errors in 25.37 s**.
  - Errors (missing optional deps, not test failures): `policy/locomotion/core/platforms/tests/test_registry.py` (`No module named 'gymnasium'`), `slam/slam/backends/tests/test_common_adapter.py` (`No module named 'cerberus_msgs'`), `system/cerberus_bringup/test/test_validate_sensor_suite.py` (`No module named 'cerberus_bringup'`).

### 3. Docs inventory and staleness — PARTIAL
Last commit date per file (all in `docs/`):
- 2026-11-06 `2a84c1f`: `archive/superpowers_plans/2026-09-09-multi-platform-config.md`, `archive/superpowers_plans/2026-09-09-platform-decoupling.md`, `archive/superpowers_plans/2026-09-20-slam-foundations.md`, `artifact_storage.md`, `cerberus_project_context.md`, `embodiment_asset_composition.md`, `isaac_lab_workflow.md`, `modular_stack_options.md`, `perception.md`, `superpowers/specs/2026-09-08-platform-decoupling-design.md`, `superpowers/specs/2026-09-09-multi-platform-config-design.md`, `worlds.md`
- 2026-11-06 `6b2ed37`: `arena_integration_log.md`
- 2026-11-06 `2c201f9`: `project_structure.md`, `runpod.md`
- 2026-09-14 `c4661c0`: `locomotion_architecture.md`
- 2026-09-08 `4c052c2`: `manipulator.md`

Requested docs:
- `sensor_setup_handoff.md` — exists at `plans/sensor_setup_handoff.md` (not in `docs/`).
- `slam_handoff.md` — `plans/slam_handoff.md`.
- `arena_integration_handoff.md` — `plans/arena_integration_handoff.md`.
- `omnigibson_handoff.md` — **absent** (removed in W0).
- `worlds.md`, `artifact_storage.md`, `manipulator.md`, `perception.md`, `isaac_lab_workflow.md` — all in `docs/`.
- Modular Stack Options doc — `docs/modular_stack_options.md`.

### 4. Common components — PARTIAL
- `cerberus_msgs`: only `msg/TaggedMap.msg` (`string map_type`, `sensor_msgs/PointCloud2 point_cloud`). `LocoCommand`, `ArmCommand`, `BodyCommand` are **not defined**; they appear only as a proposed interface in `docs/modular_stack_options.md:67-69`.
- `cerberus_description`:
  - `UNITREE_GO2_CFG` is **imported**, not defined here (`go2_arx5_cfg.py:12`, from `isaaclab_assets.robots.unitree`).
  - `GO2_ARX5_CFG` and `GO2_ARX5_FIXED_BASE_CFG` exist (`go2_arx5_cfg.py:90-91`); `ARX5_CFG` as a named symbol **does not exist** (arm groups are `_ARM_ACTUATORS`, `_GRIPPER_ACTUATORS`, spec constants in `go2_arx5_spec.py`).
  - Joint indexing is by name: `joint_groups` in `embodiment.py`, joint-name lists in `go2_arx5_spec.py`, actuator groups keyed by joint name.
  - Assets: Go2 xacro + DAE meshes are committed (`urdf/*.xacro`, `meshes/*.dae`, BSD-3 Unitree LICENSE); fused URDF/USD are generated, gitignored under `data/assets/go2_arx5/` (present locally). Arm source is `third_party/arx-model` (gitignored, no license file).
- `cerberus_worlds`: `base.py` is a real ABC (4 abstract methods, `SceneInfo`), `registry.py` real (lazy `module:Class`, no silent default), `backends/arena_backend.py` 284 lines and `backends/plain_usd_backend.py` 303 lines contain real code, **no `NotImplementedError`/TODO** hits. Tests: `test_worlds_base.py`, `test_worlds_registry.py`, `test_admission.py`, `test_no_direct_backend_imports.py`, `backends/tests/test_backends_pure.py`, plus two arena-asset tests.
- `cerberus_policies`: **does not exist**. `PolicyBackend` appears only as a proposal in `docs/modular_stack_options.md:201`.

### 5. Phase 0: locomotion — DONE
- `policy/locomotion/train.py` and `play.py` exist with real code; `eval/` has `push_recovery_eval.py`, `push_recovery_event.py`, `plot_push_recovery.py`, two showcase scripts, and `rough_push_recovery_curve.png`.
- Task IDs (`policy/locomotion/core/platforms/go2/platform.py:30-33`): `Isaac-Velocity-Flat-Unitree-Go2-Cerberus-v0`, `...-Play-v0`, `Isaac-Velocity-Rough-Unitree-Go2-Cerberus-v0`, `...-Play-v0`.
- Versus stock (`core/platforms/go2/env_cfg.py`): adds Hwangbo push-disturbance curriculum and clear event, pins PhysX on Lab 3.0 (stock defaults Newton), rescales `pyramid_stairs`/`pyramid_stairs_inv` step heights for Go2, adds two diagnostic terrain-level terms, play config disables pushes and uses an `asset_root` follow camera. Module docstring says rewards, PPO hyperparameters and network architecture are untouched.
- Checkpoints (`policy/locomotion/checkpoints/`, all gitignored/untracked):
  - `go2_locomotion_flat_v1_candidate.pt` 984 KB, 2026-09-03; `.onnx` 165 KB, 2026-09-10.
  - `go2_locomotion_v1_candidate.pt` 6.6 MB, 2026-09-08; `.onnx` 1.1 MB, 2026-09-09.
  - `unitree_go2_flat/2026-11-06_21-34-15/exported/policy.{pt,onnx,onnx.data}` and `..._21-49-42/...` (GA smoke exports).
  - Naming scheme: `_v1_candidate` exists; `_robustified` and `go2_locomotion_final.pt` **do not exist**.
- Results: `runs/2026-09-08_19-28-32_analysis/eval/eval_rough_full_sweep.csv` (100 trials/cell) has 94% at 0 N/level 0, 93% at 80 N/level 0, 89% at 120 N/level 0, 61% at 120 N/level 8, 62% at 200 N/level 8. `RUN_001.md` reports Flat 93% at 120 N vs the 87.5% reference and Rough terrain curriculum plateauing at mean level ~5.9 with 14.7% of envs at level 9. `RUN_002.md` records 1500-iteration convergence, stair fix reproduced (`pyramid_stairs` 83.3%, `pyramid_stairs_inv` 66.4%), `boxes` weakest (63.6% pooled).
- Tensorboard event files, 29+29 push-recovery clips, 11 progression clips and 16 terrain clips are present under `policy/locomotion/runs/` (174 MB total).
- Verdict: **trained** (Flat and Rough), not yet gated/frozen.

### 6. Phase 0.5: adversarial gate — MISSING
- `policy/locomotion/adversarial_robustness/` **does not exist**; no `train_adversary.py`, `attack_env.py`, or nested `eval/`.
- Only artifact: `plans/handoff_phase0.5.md` (spec: adversary MLP, force+command attack, gate versus Phase 0 push range, robustify loop).
- No results, checkpoints or notes for it anywhere (`find -iname '*adversar*'` returns nothing outside the handoff).
- Verdict: **not started**.

### 7. Phase 1+ components — quick inventory
- `perception/`: exists; **no `package.xml`**; 4 Python files; only candidate registries (`terrain_perception/core/algorithms.py`, 44 lines) plus tests — **skeleton/config-only**, no perception implementation.
- `slam/`: exists; **`package.xml` present**; 57 Python files; real backend adapters for 6 algorithms, ground-truth bridge, eval/recording scripts — **real code** (algorithms themselves are external, `third_party/`).
- `planning/`: exists; **no `package.xml`**; 8 Python files; `global_planner`/`local_planner` registries only (44/51 lines), no planner implementation — **skeleton**.
- `policy/manipulation/`: exists; **no `package.xml`**; 10 Python files; `go2_arx5_stack` env config, Mimic env, layout, waypoints, register — **partial real code**; no `demo_collection`, `algorithms`, `continual_learning`, `world_model`, `eval`.
- `policy/locomotion_policy/`: exists; **`package.xml` present**; 6 Python files; real inference node + observation assembly, with hardware IMU TODOs — **real code**.
- `policy/manipulation_policy/`: **MISSING**.
- `safety_filter/`: **MISSING**.
- `actuation/`: **MISSING**.
- `system/`: exists; **`package.xml` present** (`cerberus_bringup`); 8 Python files; launch file, YAML configs, sensor-suite validator — **partial real code**; no `system/scripts`, `system/systemd`.

### 8. Arena / worlds spike status — PARTIAL
- Gate definitions in `docs/arena_integration_log.md`: W0 = remove OmniGibson (PASS); W1 = Arena spike (Gate A stock/GRScenes, Gate B Lightwheel); W2a = fixed-base Mimic in stock scene; W2b = Lightwheel kitchen Mimic; W3 = stack/version closure. W labels differ from project Phases.
- W1 attempted multiple times:
  - 2026-10-05: Gate A stock on EA "passed" but robot fell through the floor; GRScenes via Arena failed to progress (stall ~17 min in PhysX warmup); GRScenes via `plain_usd` passed.
  - 2026-11-06: `data/reports/ga_smoke_20261106T214925Z/arena_spike_go2.json` has `"passed": true`, but metrics show `robot_z_min_max [-0.7296,-0.4822]`, `foot_z_mean -0.506`, `base_minus_foot_z_mean -0.181`. The arena log's 2026-11-06 "standing check" says this is a FAIL (threshold `<= 0.15`), and `run_ga_smoke_pod.sh` now runs the spike as `--backend plain_usd --scene stock`. Commit `6b2ed37` is that change.
  - Result: **no meaningful Arena W1 pass**; `plain_usd` ground-plane pass only.
- `data/scenes/`: only `data/scenes/grscenes/admission.json` (36 KB); no scene USD locally. Fetch scripts exist: `scripts/fetch_grscenes_from_r2.py`, `fetch_grscenes_to_r2.py`, `validate_scenes.py`, `build_lidar_proxy.py`.
- GRScenes access: `docs/arena_integration_log.md:252` says HF reports `gated: false`, so **no approval step**; a home zip was obtained. Contradicted by `docs/worlds.md:165` (see mismatches).

### 9. Anything surprising
- `.env` exists in the working tree (mode 644) with live-looking `WANDB_API_KEY`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_TOKEN_VALUE`, `RUNPOD_API_KEY`. It is **gitignored and not tracked**; values not printed. Other files reference the variable names only.
- No CI, no pre-commit, no ruff config file, despite `ruff` and `pytest` tasks in `pixi.toml`.
- `docs/superpowers/specs/2026-09-08-platform-decoupling-design.md` and `2026-09-09-multi-platform-config-design.md` duplicate files in `docs/archive/superpowers_plans/` (the archive copies are longer).
- `plans/` (5 handoff docs, 100 KB) is not mentioned in `docs/project_structure.md`'s tree.
- 21 MB of committed Go2 DAE meshes (largest `base.dae` 11 MB) — intentional asset, not an accident.
- `third_party/IsaacLab-Arena` is a full vendored checkout present locally but gitignored.
- Only 3 TODO/FIXMEs in the repo, all the hardware IMU wiring in `policy/locomotion_policy/locomotion_policy/inference_node.py:153-155`.
- `build/` and `install/` contain stale colcon outputs (`build/{slam,locomotion_policy,cerberus_bringup}`, `install/...`); both are gitignored.
- Local `policy/locomotion/runs/` has ~174 MB of clips/events; ignored by `.gitignore`.

## Doc/code mismatches
1. `README.md:2` — describes "quadruped loco-manipulator ... on Isaac Sim"; no mission domain, no GRScenes/Arena, no Go2+ARX5, no phase status. It is a 2-line README.
2. `docs/project_structure.md:92` — `colcon_defaults.yaml` listed; absent on disk.
3. `docs/project_structure.md:163` — `docs/architecture.svg` listed; absent.
4. `docs/project_structure.md:121,128,140-144,146,148-156` — `object_perception`, `task_orchestrator`, manipulation subpackages, `manipulation_policy`, `safety_filter`, `actuation` listed; all missing.
5. `docs/worlds.md:43` — "Nothing is committed yet"; backends are committed (`7653b8c`).
6. `docs/worlds.md:165` — "GRScenes access-gate approval process ... still open"; `docs/arena_integration_log.md:252` says HF `gated: false`, no approval step.
7. `docs/cerberus_project_context.md:159` — "SLAM approach and tooling ... not yet confirmed"; six SLAM backends with adapters, configs and Dockerfiles exist in `slam/slam/backends/`.
8. `docs/cerberus_project_context.md:172` — "exact implementation [of embodiment config] is not yet written"; `common/cerberus_description/go2_arx5_cfg.py` + `go2_arx5_spec.py` + `embodiment.py` exist.
9. `docs/cerberus_project_context.md:46` and `docs/worlds.md:55` — GRScenes "access gate"; see item 6.
10. `docs/artifact_storage.md:126-127` — rclone "not yet in `docker/remote/Dockerfile`"; `docker/remote_v6/Dockerfile` installs rclone 1.75.1 (old image still true).
11. `docs/isaac_lab_workflow.md:37` — points at `docker/remote/Dockerfile` (Lab 2.3.2); the active stack is `docker/remote_v6/Dockerfile` (Sim 6.2/Lab 3.0 GA).
12. `docs/manipulator.md:79-80` — MobileVLA-R1 is "the leading candidate" for the orchestrator; `docs/modular_stack_options.md:169` makes joint-level VLA a non-goal and `:47` keeps QUART out, recommending π0.5/X-VLA/SmolVLA/MolmoAct2 instead. The two docs disagree on the VLA family to use (MobileVLA-R1 is high-level, not joint-level, so the conflict is about which candidate is still live).
13. `docs/project_structure.md:119-121,126-128` — labels `terrain_perception`, `object_perception`, `global_planner` and `task_orchestrator` "ROS2 pkg"; the first two and `global_planner` have **no `package.xml`** (only `slam/`, `policy/locomotion_policy/`, `system/cerberus_bringup/`, `common/cerberus_msgs/`, `common/cerberus_description/` do).
14. `data/reports/ga_smoke_20261106T214925Z/arena_spike_go2.json` — `"passed": true` conflicts with the standing check added in `docs/arena_integration_log.md` (same date) that makes it a fail.

## Not checked
- Isaac Sim/Lab execution, physics steps, training, Arena, or scene loading — local machine cannot run them and no remote pod was used.
- `git fetch`/remote state; only local `origin/main` tracking was read.
- ROS2/colcon build and node runtime (no build or launch run).
- `pixi.lock` resolved versions (only file presence and `pixi.toml` declarations).
- Contents of `third_party/` (vendored, gitignored) beyond directory names.
- Whether `.env` credentials are live/valid; only their presence and file mode were checked.
- Checkpoint correctness/accuracy beyond file names, sizes, dates and recorded eval CSVs.
- Full contents of `docs/arena_integration_log.md` (1279 lines); W-gate sections and the tail were read, middle sections were not.
- `data/reports/overnight/` and `data/reports/slamready/` raw artifacts were not audited beyond the GA smoke reports.

## Suggested next 3 actions (my opinion, not repo facts)
1. Re-run W1 Arena with a scene where the robot can stand (e.g. `--backend arena --scene lightwheel:1:1`) and the standing criterion active, then record the result; the current `passed: true` JSON is contradicted by the log's own metric and by the smoke script's move to `plain_usd`.
2. Implement Phase 0.5 from `plans/handoff_phase0.5.md` (`attack_env.py`, `train_adversary.py`, `eval/`) so the candidate checkpoint can become `go2_locomotion_final.pt`; Phase 1 composition has nothing frozen to build on today.
3. Decide and write the interface layer before more modules land: `LocoCommand`/`ArmCommand`/`BodyCommand` in `cerberus_msgs`, `package.xml` for `perception`/`planning`, and either `common/cerberus_policies` or an explicit note that `PolicyBackend` is deferred.
