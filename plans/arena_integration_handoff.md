# Cerberus — Arena / Worlds Integration Handoff (for coding agent)

Audience: a coding agent working in the Cerberus repo. Dense on purpose. Read fully before acting.
Context docs: `cerberus_project_context.md`, `project_structure.md`, `worlds.md`, `isaac_lab_workflow.md`,
`manipulator.md`, `perception.md`. OmniGibson docs were deleted in W0 (history in git).

## 0. Goal and ground rules

Replace OmniGibson with a modular world layer (`common/cerberus_worlds/`) backed by Isaac Lab Arena
(Apache-2.0, alpha) and a generic plain-USD backend, validate it with GPU physics at 16+ parallel envs,
then add the ARX5 arm, then upgrade Isaac Sim/Lab. Impressive non-warehouse scenes (homes, commercial)
and modularity are the two goals.

Ground rules:
1. **No monkeypatching Isaac Lab, Isaac Sim or Arena internals.** If something only works with a patch, stop and report. That is what killed OmniGibson (six fixes plus a device patcher, still CPU only).
2. **Do not run expensive training until the stack is final** (after W3). W0–W2 need no RL training.
3. **Each phase has a gate.** If a gate fails, stop, write findings to `docs/arena_integration_log.md`, do not improvise a workaround that changes scope.
4. **Verify before trusting this doc.** Grep the repo for the named artifacts; versions and repo state below may be stale.
5. **Leave changes uncommitted by default** unless the user says otherwise. Never add attribution lines the user didn't ask for beyond the session's commit rules.
6. Label integration phases **W0–W4** in all docs. Cerberus Phases 0–4 and Phase 0.5 (adversarial gate) are a different numbering.
7. Keep Cerberus-specific wrappers thin around Arena. Arena is alpha, so interfaces will move.

## 1. Verified facts and versions (as of 2026-10-04)

| Item | Fact |
|---|---|
| Current Cerberus stack | Isaac Sim 5.1.0, Isaac Lab v2.3.2 (rsl-rl-lib 3.1.2), BEHAVIOR-1K v3.9.2 (to be removed) |
| Isaac Sim 6.0 | GA June 4 2026, 6.0.1 June 22 |
| Isaac Lab 3.0 | v3.0.0-EA Sept 16 targets **Sim 6.1**; release/3.0.0 branch also 6.1; GA targeted end Oct 2026; EA→GA is bug-fix only; beta2 targeted Sim 6.0. `main` was still the 2.x line (Sim 4.5/5.0/5.1) |
| Lab 3.0 migration | XYZW quaternions; `PhysxCfg` moved to `isaaclab_physx`; `--headless` deprecated; Python 3.12; IMU/PVA changes; 2.x style works with deprecation warnings until 4.0; PhysX default, Newton beta |
| Arena | Composes Scene/Embodiment/Task via `ArenaEnvBuilder` into a `ManagerBasedRLEnvCfg`; `@register_asset` registry; "Arena in Your Repository" guide for out-of-tree assets; Mimic interop (PR #6690 on 3.0 line); only `isaaclab_arena_g1` embodiment exists, no quadruped; catalog is tabletop/kitchen skewed |
| Arena versions | README (possibly stale): `main` → Lab 3.0.0 / Sim 6.0.0 / Python ≥3.12; branch `feature/arena_v0.2_on_lab_2.3` → Lab 2.3.0 / Sim 5.1.0. v0.3 announced Sept 10 2026 with `release/0.3.0`; **compat table not found, read that branch's README first** |
| GRScenes-100 | 69 home + 30 commercial plain-USD scenes, hashed IDs, no catalog. Per scene: `start_result_raw/navigation/interaction.usd`, `metadata.json`, `interactive_obj_list.json`. Tools: `preprocess.py` (`-n` nav, `-i` interaction), `get_metadata.py`, `play_scene.py`. Needs `MDL_SYSTEM_PATH`. Tooling pinned to Sim 4.2.0.2. CC BY-NC-SA 4.0 + user-agreement gate |
| Lightwheel via Arena | RoboCasa/LIBERO: 100 kitchens (10 layouts × 10 styles), 130+138 tasks, LW-BenchHub Apache-2.0 |
| InternScenes | ~40k scenes but only Real2Sim released, Sim 4.1.0. Watch item, do not integrate |
| LeggedManip_Lab | zzzJie-Robot, Apache-2.0, ~195 stars. GO2-ARX5 ("ARX-X5") platform; Flat and WBC (whole-body EE tracking) modes. Needs Isaac Lab `main` + rsl-rl ≥5.0.1 + Python 3.11 + Sim 5.1.0, **not v2.3.2**. Policies are whole-body, so only the asset/config layer is reusable |
| Isaac Lab Mimic | Loco-manip APIs (PR #3992: base-as-EEF, nav controller, custom recorders) are in v2.3.2. G1 example = RL lower body + IK upper body. Requires task-space (EE) actions. `get_object_poses()` reads `scene.get_state(is_relative=True)["rigid_object"]`, so tracked objects must be Isaac Lab `RigidObject`s (inference from source, untested) |

## 2. Unknowns (resolve by test, not assumption)

- Arena v0.3 ↔ Lab/Sim compat; which Arena branch to pin.
- Registering a Go2 quadruped as an Arena embodiment (no precedent).
- Arena Scene/Background handling a full multi-room GRScenes home with 16+ envs.
- GRScenes preprocessing on Sim 5.1 / 6.x; access-gate approval timing.
- ROS2 bridge/distro support in Sim 6.x.
- Mimic with GO2-ARX5 (EE actions, FK, RigidObject targets).
- `isaaclab-go2-locomotion` version compatibility.
- Fused GO2-ARX5 USD mass/inertia/PD gains; ARX-X5 vs ARX5 variant (spec: 3.35 kg, 1.5 kg payload).

## 3. Phases and gates

### W0 — Remove OmniGibson (no new functionality)
Inventory first with `grep -ri omnigibson\|behavior1k\|behavior-1k` across repo, Docker, scripts, docs. Expected items (verify):
- `common/cerberus_scenes/omnigibson_scenes.py` (Fixes 1–6)
- `docker/remote/patch_omnigibson_device.py` (comprehensive device patcher)
- Dockerfile steps: BEHAVIOR-1K clone, patcher COPY/RUN, lerobot import strip, MDL symlinks, `--no-deps` install, `fetch_omnigibson_assets.py`
- `docker/fetch_omnigibson_assets.py`; `start.sh` dataset fetch and version marker
- RunPod network volume `8ey9ahvll1` mount; env vars `OMNIGIBSON_DATA_PATH`, `BEHAVIOR1K_PATH` in `/etc/cerberus_env.sh`
- `worlds/` OmniGibson fetch scripts

**KEEP** `policy/locomotion/core/platforms/go2/env_cfg.py`'s `self.events.push_disturbance_clear = None` (real asymmetry bug, unrelated to OmniGibson).
Do not delete the archive docs: add the SHELVED banner to `docs/omnigibson.md` and `docs/omnigibson_handoff.md`. History stays in git.
**Gate:** repo has zero OmniGibson imports, Docker image builds, existing Go2 flat locomotion task still trains a few iterations and plays.

### W0.5 — `cerberus_worlds` skeleton
```
common/cerberus_worlds/
  base.py                 # WorldBackend ABC: list_scenes, load_scene, register_embodiment, build_env
  registry.py             # name -> backend, chosen by config / WORLD_BACKEND env var
  backends/arena_backend.py
  backends/plain_usd_backend.py   # generic USD loader, configurable preprocess step
```
Embodiment composition lives in `common/cerberus_description/`: `UNITREE_GO2_CFG` + `ARX5_CFG` → `GO2_ARX5_CFG`; name-based joint indexing; `go2_arx5_` checkpoint prefix. Interface only, with stub backends and a unit test for the registry. No sim needed.
**Gate:** `pytest` passes; no other package imports a backend directly.

### W1 — Arena spike (the go/no-go)
No new git branch (user decision 2026-10-05): work on `main`, uncommitted, nothing pushed; a local patch snapshot of the pre-W1 tree is in `data/backups/` (gitignored). Use a **fresh env**: check Arena out under gitignored `third_party/`, not into the repo. Pin per the "Version strategy" paragraph below (decided 2026-10-05); the `release/0.3.0` compat table is only the README badges (Lab 3.0.0 / Sim 6.0.0 / Python >=3.12, `pyproject` pins `isaaclab==3.0.0b2`).
1. Register a bare-minimum Go2 as an Arena embodiment (out-of-tree, per "Arena in Your Repository"). Reuse `UNITREE_GO2_CFG`. Thin wrapper.
2. Import GRScenes-100's smallest home scene as a custom Arena Background (run `preprocess.py -n`; set `MDL_SYSTEM_PATH`).
3. Compose with a trivial task (reach waypoint) via `ArenaEnvBuilder`.
4. `env.reset()` / `env.step()` at **16+ parallel envs, GPU physics on, zero monkeypatching**.
**Gate A:** step 4 works. **Gate B:** the same pipeline loads one Lightwheel kitchen scene via Arena with Go2.
If A fails on multi-room scale only, fall back to `plain_usd_backend` for GRScenes (Isaac Lab cloning, no Arena) and keep Arena for kitchens. If Arena cannot host a quadruped at all, drop Arena embodiment use and keep `plain_usd_backend` as primary; report before continuing.
**Version strategy (decided 2026-10-05):** run Gate A on `feature/arena_v0.2_on_lab_2.3` (Lab 2.3 / Sim 5.1, current pod image). That branch has been frozen since 2026-03-31 and is not SQA tested, so a Gate A failure there is ambiguous. Before declaring Arena unable to host a quadruped, do a time-boxed spike on `release/0.3.0` (Lab 3.0.0b2 / Sim 6.0 / Python 3.12); the cause may be the stale branch. Gate B (Lightwheel) cannot run on the 2.3 branch (no Lightwheel content and no `lightwheel-sdk` dependency there; found by file-tree and `pyproject` inspection, not by running it), so Gate B moves to the 0.3.0 stack, after W3. The 0.3.0 spike also informs W3: Arena's `release/0.3.0` pins Lab 3.0.0b2 on Sim 6.0, not the GA / Sim 6.1 assumed earlier, so the W3 target may end up being whatever Arena supports.
**REVISION (user, 2026-10-05, later): do not wait for Lab 3.0 GA, and do not spend the Arena spike on the frozen Lab 2.3 branch.** Move to the Lab 3.0 line now. The old image stays only for checks that need the Lab 2.3.2 checkpoint (W1.5a caveat 2). New stack: Isaac Sim 6.1.0 container, Python 3.12, Arena `main` (its Dockerfile base is `isaac-sim:6.1.0`; its Lab submodule commit `bb0c8e1b9` is 228 commits behind the `v3.0.0-EA` tag, which is Sim 6.1). Arena `release/0.3.0` (Lab beta2, Sim 6.0.1) is not needed. First run Arena with its own pinned Lab commit; then try bumping Lab to `v3.0.0-EA` or `release/3.0.0` at a recorded SHA. Moving to the EA line is the user's explicit decision; bump to GA when it ships. Untested: whether Arena `main` runs on the EA tag.
**GRScenes access:** `InternRobotics/GRScenes` on Hugging Face reports `gated: false`, so there is no approval step. GRScenes-100 is `scenes/GRScenes-100/`: `commercial_scenes.zip` 23.8 GB, `home_scenes` as a split zip (`.z01` 21.5 GB, `.z02` 21.5 GB, `.zip` 8.0 GB, join before extracting), about 74.7 GB compressed in total. `scenes/demo_scenes/` is a single 30 MB file and is no use for step 2.

**Status W1 (2026-10-05 overnight run; details and flaws in the log, "W1 - new-stack gates"):** run on the new stack, not on the Lab 2.3 branch. On Arena `main` @ `311d0f0fc` with its pinned Lab `bb0c8e1b9` (Sim 6.1.0 container, Py 3.12.13, driver 570.195.03): **Gate A (stock Arena scene) PASS** 16 envs x 200 steps, 40.9 ms/step; **Gate B (Lightwheel 1:1 and 6:8) PASS** (98 / 102 ms/step). On Lab `v3.0.0-EA` (`ae37b028e`): Gate A on the stock scene exited 0 but the robot fell through the Galileo floor (root z -0.73..-0.49 m), so that pass is NOT meaningful; **Gate A on the GRScenes home scene via Arena FAILED** (stalled ~17 min in PhysX warmup with 237 "TriangleMesh: triangles are too big" warnings); the documented fallback `plain_usd` backend on the same GRScenes scene PASSED (16 envs x 200 steps, final z 0.253 m). Gate B on EA PASS. Net: Arena hosts the Go2 for stock and Lightwheel scenes; GRScenes goes through `plain_usd`. Not tested: Arena on a multi-room scene with a tuned collision approximation.

### W1.5a — ARX5 assets (no training)
- Pull the fused GO2-ARX5 USD/`ArticulationCfg` from LeggedManip_Lab as **reference**, do not take its policies or requirement for Isaac Lab `main`. Rebuild or adapt on the pinned stack (URDF → USD via `UrdfConverter`, fixed joint on Go2's back).
- Verify mass/inertia, joint limits, PD gains; confirm ARX-X5 vs ARX5 variant.
- Check name-based joint indexing works; spawn in a W1 scene, hold default pose stable for N seconds at 16 envs.
**Gate:** GO2-ARX5 spawns, stands with a stowed arm, joints addressable by name.

**Status (2026-10-05): the flat-ground half of the gate PASSED on a pod.** Code and results: `scripts/fetch_arx5_assets.py`, `scripts/spawn_hold_go2_arx5.py` (the original assembler script `scripts/compose_go2_arx5.py` has since been removed, see the composition decision below), `common/cerberus_description/{go2_arx5_spec,go2_arx5_cfg,assembly}.py`; run record in `docs/arena_integration_log.md` ("W1.5a - pod run"). Still to do: re-run the hold test inside a W1 scene once W1 exists (`spawn_hold_go2_arx5.py` builds its own flat-ground scene today).

**Status update (2026-10-05 overnight):** rebuilt as a 20 DoF fused USD from X5A (gripper included) on the old image; hold on flat ground PASS on the old stack, the pinned new stack and EA (EA free-base pitch -15.5 deg). In a W1 scene only the fixed-base robot was run (Arena Lightwheel kitchen mimic smoke, 16 envs, arm drift 7.6e-5 rad after disabling gravity on the fixed-base robot, and the W2b env smoke); a free-base hold inside a Lightwheel or GRScenes scene was not run. Caveats 1-4 and 6 below are resolved (notes under each); 5 and 7 are unchanged.

**Known caveats and open checks (a future agent should test these, not assume them):**
1. **The fused robot leans at rest and the cause is unknown.** In the PD-hold test (default pose, no policy, 16 envs) it settles at pitch -11.4 deg, roll -5.2 deg (max tilt 12.5 deg); the reference LeggedManip_Lab USD through the same harness showed about 1.3 deg. The rear calves sag 2-3x more than the front (`RL_calf_joint` 0.656 rad vs `FL_calf_joint` 0.234). The arm's CoM was ruled out as the fore/aft or lateral cause: at the stowed pose it is (-0.007, 0.000, 0.127) m in the trunk frame and the whole-robot CoM x shifts only from +0.009 to +0.006 m (it does raise the CoM about 3 cm). Do not "fix" it by moving the mount point (`GO2_ARX5_MOUNT_XYZ`, z 0.07; the LeggedManip MuJoCo model uses 0.06) or the stow pose before it is attributed.
   **Resolved 2026-10-05 (old-stack session, 20 DoF fused USD):** the lean is the stock Go2 PD sag, not the arm. A legs-only stock Go2 through the same hold harness leans pitch -11.9 deg, roll -0.07 deg (RL_calf error 0.47 rad); the fused 20 DoF robot leans pitch -12.4, roll -2.6 (free base, height 0.240 m). Pitch is fully explained by stock gains (the report's verdict is `go2_pd_hold`, based on pitch); the ~2.6 deg fused roll vs 0.07 stock is small and was not separately attributed. With the base fixed the fused robot holds height 0.400 m, tilt 0.
2. **Attribute it with two cheap, inference-only checks (no training, allowed before W3):**
   - a legs-only stock Go2 through the same hold harness (the fused cfg lists arm joints in `init_state`, so the harness needs a legs-only variant), to see whether the lean is just PD sag at stock gains under load;
   - the existing frozen policy `policy/locomotion/checkpoints/go2_locomotion_flat_v1_candidate.pt` on the fused robot with the arm stowed, versus the same policy on stock Go2, measured the same way (pitch/roll while standing, velocity tracking, push recovery via `policy/locomotion/eval/`). The policy expects 12 leg joints and the fused robot has 18, so the eval env must select and drive only the leg joints by name (`GO2_ARX5_EMBODIMENT.joint_groups['legs']`) in both observations and actions.
   This also tests the claim in §4.2 that `go2_locomotion_final.pt` "does not transfer", which is an inference from the 22% mass increase and has not been measured. Record the result in the log and in §4.2.
   **Resolved 2026-10-05:** the frozen `go2_locomotion_flat_v1_candidate` (ONNX, 48 inputs) driving only the 12 leg joints of the fused 20 DoF robot (arm and gripper held) gets verdict `transfers` against the stock Go2: standing pitch -1.07 deg stock vs -1.33 fused, push recovery 1.0 at 40/80/120 N on both (16 trials each), fall rate 0 in standing and all commands. Weaker spot: yaw-rate command 0.5 rad/s, fused roll mean 6.0 deg and tilt max 21.3 deg (stock 2.8 / 11.5), though yaw-rate error was smaller on the fused robot. **Harness caveat:** the default fall rule (base height 0.15 m) is an artifact because this policy crouches to 0.11-0.17 m while moving; the verdict is from a run with `--fall_height_m 0.05`. Verdict thresholds are provisional; 48 measured envs; one pod, one seed.
3. **Hold-test thresholds are provisional and one was fitted to the data.** Arm 0.10 rad, leg 0.80 rad (loosened after seeing 0.656 rad on `RL_calf`), tilt 25 deg, base-height band 0.15-0.45 m. Stock PD gains (stiffness 25) with no policy sag the legs under body weight. The passing evidence is: the arm holds its pose (max 0.042 rad), the base stops moving (drift 1.7e-5 m), all 18 joints are addressable by name, and the arm weighs 3.52 kg. It is not evidence of good balance. Recalibrate the thresholds from item 2's baselines.
   **Resolved 2026-10-05 (numbers only):** baselines now measured on the same harness: stock legs-only hold leg error max 0.47 rad, mean 0.15; fused 20 DoF leg error max 0.58, arm max 0.042 rad (joint4), gripper error ~1e-4 m, drift 3e-5 m. The thresholds themselves are unchanged and still provisional (the 0.80 rad leg bound is still loose); whether to recalibrate them was not decided.
4. **Gripper: the SDK `X5.urdf`, the LeggedManip_Lab USD and its MuJoCo model have no actuated gripper, but ARX's own model does** (found 2026-10-05). `ARXroboticsX/ARX_Model` (master, last commit 2026-07-14), `X5/X5A/urdf/X5A.urdf` + meshes, has `joint1..6` revolute plus `joint7`/`joint8` prismatic fingers (`link7`/`link8`, 0.0648 kg each, range 0-0.044 m each, mirrored axes +y/-y, no `<mimic>` tag, effort 100 / velocity 1000 placeholders). The SDK `X5.urdf` instead bakes the stock gripper body into `link6` (0.581 kg vs 0.441 kg in X5A) with an empty `eef_link`. Total X5A mass 3.454 kg, which equals the LeggedManip_Lab USD figure, so that USD was probably built from X5A with the fingers frozen. Caveats: X5A geometry is not identical to the SDK X5 (joint5/joint6 offsets differ by 2-4 mm); all X5A arm joint limits are placeholder +-10 rad; `ARX_Model` has no LICENSE file (GitHub reports none), so keep its files in gitignored `third_party/` and do not redistribute them; meshes use `package://X5A/meshes/` paths. Plan: use X5A as the arm source (Option A: two-finger gripper, one open/close action driving both fingers) and supply joint limits ourselves. Not yet built or tested.
   **Resolved 2026-10-05:** the fused USD was rebuilt from ARX's X5A model (`ARXroboticsX/ARX_Model`, not redistributable, gitignored `third_party/arx-model`): 20 DoF (12 legs, `joint1..6`, `joint7`/`joint8` prismatic fingers), arm 3.455 kg, total 18.47 kg, gripper tracks open/closed targets within ~1e-4 m (1.3e-4 free base, 2e-6 fixed base). The "18 DoF / no gripper" statements in the log's earlier sections are superseded.
5. **Pod-run gotchas** (also in `docs/runpod.md`): set `TERM=xterm` before `isaaclab.sh` over non-tty ssh; `SimulationApp.close()` exits 0 (hiding failures) and has hung 10+ minutes at shutdown, so one-shot scripts write results, flush, then `os._exit(code)`; the mount-frame arguments of `RobotAssembler.assemble_rigid_bodies` are absolute prim paths despite its docstring.
6. **Rendering on driver 595.91.07 is untested.** Headless physics worked on it, but `docs/runpod.md` lists 595.x as a crash for RTX rendering. Run the render smoke test before any pod session that records video.
   **Resolved 2026-10-05:** RTX rendering on driver 595.91.07 with Isaac Sim 5.1 SEGFAULTS (`librtx.scenedb.plugin.so` at Hydra engine creation); `play --video` is blocked by a guard on that combination. The render smoke PASSED with Sim 6.1 on drivers 570.195.03 and 570.211.01 (Isaac RTX camera RGB), and EA sessions ran on 570/580. Avoid driver 595 for any rendering pod; headless physics without cameras still worked on it.
7. **Mass and limit sources disagree** (arm 3.35 kg spec / 3.45 kg LeggedManip USD / 3.52 kg `X5.urdf`; arm joint limits differ between the URDF and the USD; effort limits 20/20/20/7/5/5 N*m are taken from a MuJoCo ctrlrange, not an ARX datasheet). The spec follows `X5.urdf`; check the ARX datasheet before relying on torque limits in W4.
8. **Arm mount z discrepancy (found 2026-10-05).** `GO2_ARX5_MOUNT_XYZ` is (0, 0, 0.07) and the URDF-merge build applies it, but every arm body in the stock-built USD (Sim 5.1 assembler) sits exactly 0.07 m lower, so that USD effectively mounted the arm base at the Go2 `base` origin (z 0). A URDF variant with mount z=0 reproduces the stock-built hold (height 0.231 m vs 0.234, pitch -15.56 vs -15.51 deg). Which z is physically right is NOT settled (Unitree's mounting spec was never checked). `layout.py` (`ARM_BASE_Z`) and the W2a/W2b reach scans assumed 0.07 while the W2a/W2b pods used the z=0 USD; W2a still passed. Also, the rear thigh limits in the vendored Go2 xacro were wrong (front limits on all legs) and were fixed; see the log.
   **RESOLVED (user, 2026-10-05):** the spec value `GO2_ARX5_MOUNT_XYZ = (0, 0, 0.07)` is final. Evidence: with z=0 the arm base sits fully inside the Go2 body shell (mesh top 0.089 m); at 0.07 the base bottom is 19 mm below that top and 13 mm above the xacro trunk box top (0.057 m, if that box is the collision shape; unchecked); LeggedManip_Lab uses 0.07 (USD) and 0.06 (MuJoCo). The real mounting-plate thickness was NOT checked (Unitree's payload page is script-rendered). The default fused USD is now the URDF-converted one at z=0.07, so `layout.py` is finally consistent with the asset. **Unverified:** the W2a gate was passed on the OLD z=0 asset and was not re-run on the new default (no pod run, cost decision). The hypothesis that the z mismatch contributed to W2b's reach/wrist-limit failures is also untested.

### W2a — Fixed-base Mimic demo, stock Isaac Lab scene (Lab 2.3.2)
Test the Mimic compat chain with the base fixed first, on the current stack with a stock Isaac Lab scene (cabinet or table). Mimic needs EE (task-space) actions via differential IK and objects as Isaac Lab `RigidObject`s, not Lightwheel. Record a few demos, annotate, generate with Mimic. **The gripper decision (caveat 4 above) must be made before W2a starts.**
**Gate:** Mimic produces a valid generated dataset for one pick-place with GO2-ARX5 (base fixed). Moving-base (loco-manip) Mimic is a later step.

**Status W2a (2026-10-05): PASS with caveats, on the new stack (Sim 6.1 / Lab `bb0c8e1b9`), not Lab 2.3.2.** Fixed-base Go2+ARX5, stock table scene, cubes, gripper decision made (X5A fingers). Franka control pipeline 10/10 annotated and 10 generated (28 attempts); our env at 16 envs OK, IK error <1e-5 m; scripted demos 10 successes in 21 attempts; auto-annotation 7/10; Mimic generated 20 successes in 155 attempts (12.9%), HDF5 3.37 MB. On EA: 10/10 source, 7/7 annotated, 20/20 generated (3.36 MB). **Caveats:** cube friction raised 1 -> 5 (unrealistically high), gravity disabled on the fixed-base robot (the stowed arm otherwise sagged 0.974 rad), gripper stiffness 1000 -> 5000 N/m, close pause 15 -> 40 steps; all gains provisional. The 12.9% yield was judged not promising, so the 200-trial expansion was skipped. Lab's `annotate_demos.py` reads `args_cli.headless` (gone on the Lab 3.0 line), so we use an in-process annotate port. **Asset caveat (2026-10-05, later):** this gate passed on the OLD assembler-built USD (arm base at z=0); it is NOT re-verified on the new default URDF-converted USD (arm 7 cm higher). Only structure, hold and gripper tests ran on the new asset.

### W2b — Lightwheel kitchen (after W3)
Repeat the W2a flow in a Lightwheel kitchen via Arena. Lightwheel needs the Arena `release/0.3.0` stack (Lab 3.0 line), so this runs after W3 and shares Gate B with W1.

**Status W2b (2026-10-05): BLOCKED, gate NOT passed.** Fixed-base Go2+ARX5 in the Lightwheel kitchen via Arena (sweet_potato pick object, procedural riser/plate): env smoke PASS at 4 and 16 envs, but scripted recording produced 0 successful demos in about 55 attempts over 6 pod runs (default region: 15/15 `descend_pick` timeouts, ~3.15 cm residual; +0.03 m dz: 15/15 done without success; moved region and dz 0.015/0.02/0.03: 0/2, 0/2, 0/5, fingers open and `grasp_1` false; yaw alignment: 0/5, `above_place` timeouts with joint6 saturated at -1.57 rad; last run 0/5). Diagnosis: the object is 125 mm x 52.7 mm against an 88 mm jaw opening and joint6 hits its limit when aligning; this is an object/planner problem, not a stack or robot failure (W2a grasps cubes). The fixed-yaw hypothesis is UNTESTED: in the last run the yaw was not actually fixed (observed -17.2..+2.3 deg). **Next step:** swap to a cube-like procedural pick object in the kitchen; irregular real objects need teleop demos or a grasp planner (headless teleop is unsolved). See the log, "W2b".

### W3 — Isaac Sim/Lab upgrade
Per the REVISION note in W1 (user decision 2026-10-05), W3 is no longer gated on GA or on W1/W2a passing. Target the Lab 3.0 line on Isaac Sim 6.1.0 now (Arena `main`'s pinned Lab commit first, then `v3.0.0-EA`/`release/3.0.0` at a recorded SHA); bump to GA when it ships. W3 code migration comes before the new-stack pod sessions, and W2a runs on the new stack so the Mimic work is not done twice.
Dependency checklist (check each, record result in the log):
- Arena compat for the chosen Lab/Sim (README/branch table).
- rsl-rl version and API (Lab 2.3.2 uses rsl-rl-lib 3.1.2; LeggedManip_Lab wants ≥5.0.1).
- Python 3.12; XYZW quaternion migration in all Cerberus code; `PhysxCfg` import path; `--headless` replacement.
- GRScenes tooling on the new Sim; Lightwheel assets; Mimic API.
- ROS2 bridge and distro support in Sim 6.x.
- `isaaclab-go2-locomotion` reference version.
- Docker base image, driver/CUDA requirement on the rented GPU.
**Gate:** W1 pipeline re-run at 16+ envs on the new stack with zero patches, W1 Gate B (Lightwheel kitchen with Go2) passes, and the W2a Mimic demo still works.

**Status W3 (2026-10-05): migration and image done, gate NOT fully passed.** Done: Sim 6.1.0 + Py 3.12.13 image `kevinkyawthiha/cerberus-remote-v6:v6-dev` (Lab `bb0c8e1b9`, Arena `311d0f0fc`, rsl-rl-lib 5.4.1, torch 2.11.0+cu128) and `:v6-ea-dev` (Lab `v3.0.0-EA` `ae37b028e`); code migrated through `common/cerberus_compat/` adapters; scoped local pytest 479 passed / 11 skipped; W1 Gates A and B and the W2a Mimic demo pass on the pinned stack (see W1/W2a above). Open: **fused-USD composition on Sim 6.1 does not work** (public `RobotAssembler` begin/assemble/finish; four adapter iterations failed), so the first arm USD was built on the old image `cerberus-remote:v3`; the URDF path (below) is now the confirmed, default way to build it on either stack and the assembler code was removed; the W2b gate (Lightwheel kitchen Mimic) is blocked; Gate A on GRScenes via Arena fails (fallback `plain_usd`); old rsl-rl 3.1.2 checkpoints likely do not load on rsl-rl 5.x (a format check was added; a real load is unverified); ROS2 bridge on Sim 6.x and the `isaaclab-go2-locomotion` reference version were not checked; no GA yet (EA tag in use). Checklist item by item is in the log, "W3". The "zero patches" wording holds only for what ran, and the stock-scene Gate A on EA is a hollow pass.

**Composition decision (2026-10-05): CONFIRMED by the user and EXECUTED.** Fused embodiments are built by merging URDFs and converting with Isaac Lab `UrdfConverter`, as a three-script workflow: `scripts/build_go2_arx5_urdf.py` (merges `go2.urdf.xacro` with `sim:=true`, the patched X5A arm and the mount joint), `scripts/convert_go2_arx5_urdf_to_usd.py` (pod; converts and checks) and `scripts/compare_go2_urdf_to_usd.py` (offline). Documented in `docs/embodiment_asset_composition.md`. It works on both Lab 2.3.2 and the Lab 3.0 EA (1 articulation root, 20 joints, floating base, 18.472 kg) and needs no Isaac assembler. `scripts/compose_go2_arx5.py` (Sim 5.1 assembler, failed Sim 6.1 adapter) and its test were REMOVED (no dead code); the composition problem on Sim 6.1 is bypassed, not solved. Mount z is settled at 0.07 (caveat 8). The default fused USD `data/assets/go2_arx5/go2_arx5.usd` is the converted one; the old assembler USDs are archived under `data/assets/go2_arx5/legacy/` only as the golden reference of the compare tool. Prim paths live in `go2_arx5_spec.py` (`GO2_ARX5_BASE_PRIM_REL_PATH`, `GO2_ARX5_TCP_PRIM_REL_PATH`). Not done: W2a re-run on the new default, and the frozen-policy check on the converted USD on the old stack (not needed to close arm+scenes; W4 retrains on the final stack). Details: `docs/arena_integration_log.md`, "Composition decision confirmed and executed".

**Status W4-prep (2026-10-06): the simulated-sensor path for SLAM is verified end to end on the new stack, at the correct
physical scale, with textured materials. No retraining was done, no SLAM backend has run yet, and the scene
admission sweep has not produced a manifest.** Full record: `docs/arena_integration_log.md`, "W4-prep". Evidence:
`data/reports/slamready/` (`smoke_metric/`, `smoke_run/` for the superseded wrong-scale run, `admission_run/`).

**Verified (pod, 2026-10-06)**
- Image `kevinkyawthiha/cerberus-remote-v6:v6-rel` (digest `sha256:c3443f44...`; superseded 2026-10-11 by `v6-rel2`, same stack plus the rclone pin) = Sim 6.1.0 + Lab `release/3.0.0` @
  `dcabdcda` + Arena `main` @ `34bb35a85`, rsl-rl-lib 5.5.1, torch 2.12.0+cu130, plus the pixi `sim-ros2` env (rclpy,
  pyzmq, msgpack, `evo`). It needs a host driver >= 580: create pods with `--min-cuda-version 13.0` (an RTX A5000,
  driver 580.x, is fine and costs ~$0.27/h).
- **Physics.** The 2026-10-05 "frozen policy tilts on Lab 3.0" result was Newton physics, not a bad checkpoint: the stock
  Lab 3.0 Go2 task sets `sim.physics.default = newton_mjwarp` and `use_newton_actuators = True`, and nothing we ran
  overrode it. `common/cerberus_compat/physics.pin_physx()` (called from the Cerberus Go2 cfgs) pins PhysX + the legacy
  actuator path. The SLAM run reports `PhysxCfg`/`PhysxManager`/`use_newton_actuators: false`.
- **Scale.** GRScenes is centimetre-authored and Isaac does NOT convert units: the raw house was spawned 100x too large
  (a couch measured 241 m on a pod; the first smoke walked a 0.3 m robot through a 1.3 km house). Fix:
  `scene_geometry.metric_scene_usd()` writes a cached `scene_metric.usd` (an Xform scaled by `metersPerUnit` that
  references the original) and both backends return it as `usd_path` (`source_usd_path` is the original). Re-measured:
  couch 2.41 x 0.97 x 0.94 m, house footprint 15.2 x 9.6 m. It also cured the PhysX warmup stall (and so very likely the
  2026-10-05 Arena stall: "triangles too big" was literal). **Everything measured about GRScenes before this fix is at the
  wrong scale** (Gate A on EA, the first smoke, any earlier spawn poses).
- **LiDAR.** 1115 of 3230 mesh prims in the sample house are empty, which crashes Isaac Lab's `MultiMeshRayCaster` on the
  raw scene. A cached `lidar_proxy.usd` (`scripts/build_lidar_proxy.py`, valid geometry only, Xform root + Mesh child,
  built from the METRE wrapper, ~5 s) is spawned invisibly at `/world/lidar_proxy`-style path by `apply_scene` and the LiDAR
  targets it. Runs with a LiDAR rig build it automatically.
- **Materials.** The dataset's `.mdl` files import `OmniUe4Function`/`Translucent`/`Base` relatively and GRScenes does not ship
  them; `ensure_mdl_system_path()` now copies them from `/isaac-sim/kit/mdl/core/Ue4/` into the scene's `Materials/`.
  Verified: the smoke video shows a fully textured home.
- **SLAM smoke (metre scale): PASS.** `scripts/run_slam_eval_remote.sh --smoke --mission household --sensor-rig lidar_camera
  --checkpoint <onnx|pt> --scene grscenes:<id> --spawn x,y,yaw,floor_z`: pose/TF/IMU/RGB/depth 1050/1050 messages, LiDAR
  202 messages (10 Hz), bag 2.49 GB, the frozen ONNX policy walked ~3.3 m (max tilt 14.9 deg, final base height 0.123 m).
  Pass `--spawn VALUE` with a SPACE to the script (it forwards `--spawn=VALUE`; a leading `-` breaks the Python parser).
- **Scene admission children** now run to completion. They used to die at Kit startup because they launched without
  cameras (inferred; the crash dumps were lost), and `validate_scenes._launch_app` now always enables cameras. At metre
  scale one scene gave: `sample` 44 s, footprint 15.2 x 9.6 m, 835 free 0.25 m cells, LiDAR 35999/36000, `env` at 4 envs
  ok (100 s), at 16 envs ok (90 s).
- Hugging Face rate-limits the range-read extractor (HTTP 429 killed a whole home fetch once). `scripts/remote_zip.py` now
  paces requests (`$REMOTE_ZIP_RPS`, default 6/s), honours `Retry-After`, jitters, and uses 4 workers.

**Known gaps (not done)**
1. **`admission.json` now exists (RESOLVED 2026-10-07).** The 8 smallest home scenes were swept at the corrected metre
   scale: 8 pass / 0 fail, 40 spawn poses, `data/scenes/grscenes/admission.json` (also in R2 at
   `worlds/grscenes/admission.json`; the 8 scenes are uploaded under `worlds/grscenes/home/`). Details and per-scene
   metrics: `docs/arena_integration_log.md`, "Scene admission sweep, home, 8 scenes". The two problems that run
   exposed are FIXED in code (same day, see the log's "Fixes landed 2026-10-07"): the `--build-timeout-s` default
   (now 900 s + allowance, was 480 s effective, too tight on a slow/cold host), the empty-admitted-list guard, and
   an rclone version preflight. The Dockerfile now pins rclone 1.75.1 and `lightwheel-sdk==1.0.3`, and that image is
   built and pushed as `cerberus-remote-v6:v6-rel2` (2026-10-11), so pods no longer need the rclone workaround. Still open: commercial scenes and the other 61 home scenes are unswept.
2. **Admission `render` phase: reimplemented on Isaac Lab's `Camera` sensor and VERIFIED on a pod (2026-10-11,
   `frame_std` 49.98, PNG textured).** It used `omni.replicator` (`rep.orchestrator.step`), which failed with "Orchestrator failed to schedule a
   capture after 1000 app updates". `scripts/validate_scenes.py::_phase_render` now builds the env, creates a
   `Camera` (its prim is authored in `Camera.__init__`), steps once so the physics-manager callback builds its render
   product, sets the pose with `set_world_poses_from_view`, then takes `RENDER_FRAMES` frames. The child JSON still
   carries `frame_std`, so the verdict logic is unchanged. It stays off by default. Two gotchas found by running it:
   a sensor only initializes on `PhysicsEvent.PHYSICS_READY` (so physics is reset after the camera is created), and
   one camera prim needs exactly one env. See the log's "Render phase on release/3.0.0" section.
3. **No SLAM backend has been run.** Backend images exist on Docker Hub (`cerberus-slam-<name>:v1`); artifacts are layered
   onto a pod with `scripts/extract_slam_backend_artifact.sh` (runs locally, needs Docker). The non-smoke path
   (backend + relay + bag + `evo` scoring) is therefore unexercised on the new stack.
4. **Arena on Lab `release/3.0.0` is untested** (Arena main's own Lab pin `28a386f0` is not on that branch). Not needed for SLAM.
5. The new-stack rsl-rl 5.x `.pt` checkpoint load path is unverified; the SLAM loop runs the old ONNX policy fine.
6. Only 1 of 69 home scenes, 0 of 30 commercial and 2 of 100 Lightwheel kitchens have ever been run; the commercial pair
   failed once at `open_stage` (before the camera fix) and was never re-run.
7. Rotate the Runpod key (user). Pod ops lesson: pull logs and artifacts BEFORE deleting a pod.
8. **What the smoke test does and does not show.** It ran a blind, scripted 21 s velocity schedule (`household` in
   `slam/slam/eval/scripted_motion.py`: forward 0.5 m/s 5 s, turn 0.5 rad/s 3 s, forward 5 s, back-and-turn 6 s), open loop,
   one run, one scene, one start; the robot stayed upright (max tilt 14.9 deg, base 0.12 m) and every sensor stream was
   recorded. It does NOT show navigation (no goals, planner or obstacle avoidance: Phase 1), collision behaviour (not
   measured, the robot may have pushed against furniture or a wall), long runs, other scenes, or floors with rugs or
   thresholds (the policy is flat-terrain). Do not read it as "the robot walks around houses fine".

**Pod-ops gotchas learned this session** (also in `docs/runpod.md`): never `pkill -f`/`pgrep -f` a pattern that appears in your
own ssh command (it kills the session; use numeric PIDs); sync with `tar czf - ... | ssh ... tar xzf -` in small batches with
`ServerAliveInterval` (large syncs to a loaded host reset the connection); `--wait` on `runpodctl pod create` can block 10+
min on the 35 GB image pull, so poll ssh in a bounded loop instead; minimal Isaac scripts must launch with cameras enabled
on this image.

**Next steps (agreed with the user, 2026-10-06; in this order)**
1. **Admission sweep — DONE 2026-10-07.** `scripts/run_scene_admission_pod.sh --categories home --limit 8` on an
   A5000: 8 pass / 0 fail, manifest `data/scenes/grscenes/admission.json`, 8 scenes + manifest mirrored to R2. About
   $0.60 and 1 h 40 m (the first pass was 0/8 on `build_env` timeouts from a slow/cold host; re-running the failures
   with `--build-timeout-s 1200` on the warm pod gave 8/8 — see the log, "Scene admission sweep, home, 8 scenes").
   The sweep spec, for the commercial scenes and the remaining 61 homes: load at metre
   scale, find spawn poses by ray-casting to the floor with ~1 m clearance, spawn the Go2 and step 200 zero-action steps
   (stands, no fall-through, tilt ok), LiDAR sweep hits geometry, repeat at 16 envs; each scene in its own process with a
   timeout so a hang becomes a recorded `fail`. Output `data/scenes/<source>/admission.json` (pass/fail, reasons, spawn poses),
   which SLAM runs read for `--spawn-index` and Phase 1 reads for its train/held-out lists.
2. **First real SLAM eval.** One LiDAR-only backend (`fast_lio2`) on 1-2 admitted scenes with the existing ONNX policy and the
   `lidar_camera` rig, scored with `evo` (ATE/RPE); this exercises the non-smoke path (backend + relay + bag + scoring) for the
   first time. **Done 2026-10-11 (code, no pod run yet): the collision half of "add before scoring anything".** The sim loop
   now records a `collision` block in `sim_loop_report.json` (`slam/slam/ground_truth/collision_monitor.py`: non-foot contact
   above 5 N, plus LiDAR band clearance), `smoke_check.py` fails on a missing block or a missing contact sensor, and
   `run_slam_eval_remote.sh` refuses to score an invalid run. Per-scene waypoint paths were NOT built (the handoff offered
   the collision check *or* waypoints); they remain a quality improvement, needing free cells persisted in the manifest and
   pod-side ray casting. Until this lands, the first eval's thresholds are the calibration run.

   *First-eval runbook (next pod session, in order):* (a) locally, Docker is required —
   `python scripts/fetch_slam_backends.py fast_lio2` then `scripts/extract_slam_backend_artifact.sh fast_lio2`;
   (b) create the pod from `kevinkyawthiha/cerberus-remote-v6:v6-rel2`, an RTX A5000, SECURE, container disk >= 80 GB,
   `--min-cuda-version 13.0`, automation ssh key; (c) on the pod `export TERM=xterm` (no rclone install needed on `v6-rel2`); (d) sync the code with
   `tar czf - ... | ssh ... tar xzf -` (the `checkpoints` exclusion means copying the ONNX policy explicitly);
   (e) launch detached, never in the foreground:
   `nohup timeout --signal=KILL --kill-after=2m 45m bash scripts/run_slam_eval_remote.sh --backend fast_lio2
   --mission household --sensor-rig lidar_camera --checkpoint policy/locomotion/checkpoints/
   go2_locomotion_flat_v1_candidate.onnx --scene grscenes:<id> --spawn-index 0 --timeout 40m --no-terminate &`;
   (f) confirm the process is detached (`ps aux` shows `?` in the TTY column), the GPU is busy and the log is
   progressing; (g) pull `slam/slam/eval/results/fast_lio2/<run_timestamp>/` (`smoke_check.json`,
   `sim_loop_report.json`, the evo results) BEFORE deleting the pod, then delete and verify with `runpodctl pod list`.
   Read `collision.valid_for_scoring` before trusting any ATE/RPE: an invalid run is not scored at all (exit 3).
3. **Retraining.** ~~The recipe is DECIDED (user)~~ **REOPENED 2026-10-12 (user): the retrain may be
   perception-conditioned / visual navigation, not a plain flat+rough run, so the target is not settled.** What was decided
   and still holds if the plain path is taken: accept Lab 3.0's new training defaults (the stock Go2 task's reward weights
   and reset randomisation) and keep `pin_physx()` on top (PhysX + the legacy actuator path: the scenes, cameras, collision
   and SLAM sim all run on PhysX, so the physics backend is deliberately NOT part of "the new defaults"). What the reopened
   question changes, and what to settle before spending on a run:
   - **Are scenes training data?** A perception-conditioned policy trained in homes makes the GRScenes set training data,
     not evaluation-only, so the train/held-out split must be locked BEFORE the run (`docs/worlds.md` currently says to
     lock it "when the Phase 1 scene set is final"), and scene count becomes a real variable rather than a nicety.
   - **The candidate's training distribution decides the scene pool.** From `docs/perception.md`: AME-2 / DELTA look
     procedural-terrain-trained (scene count irrelevant), HiPAN is depth-conditioned with a path-guided curriculum in
     unstructured 3D (scene-like), VOP-Nav is a crowded-dynamic LiDAR policy. Verify each paper's setup before choosing.
   - **The observation interface changes** (the frozen policy is blind, 48 inputs), so the new checkpoint re-opens the SLAM
     eval and the Phase 0.5 gate.
   - **Loading scenes during training is expensive**: `build_env` measured 143-885 s per scene on one A5000, and the
     `plain_usd` backend builds one scene per env with no rotation. That is an unsolved engineering item, not a detail.
   Timing remains independent of SLAM readiness. The result is a new lineage: name it accordingly and re-run the SLAM eval
   with it before swapping it in. It is also the natural start of W4 (`go2_arx5`, Phase 0.5 gate), which can run as its own
   track.

### W4 — Arm-loaded locomotion + arm control
Decision (see §4). Recommended Option A. Train locomotion on `go2_arx5` (new checkpoint lineage), re-run Cerberus Phase 0.5 adversarial gate on it, then arm IK under motion. First expensive training happens here, after the stack is final.
**Gate:** Phase 0.5 gate passes on `go2_arx5` with arm stowed and arm moving.

**W4 prerequisites (2026-10-05):** (1) the frozen-policy transfer test was measured as `transfers` on standing, velocity commands and pushes at 40/80/120 N (see caveat 2 above, with the harness caveat on the fall-height rule and the weaker 0.5 rad/s yaw case), so retraining is not forced by mass alone; the Phase 0.5 adversarial re-gate on `go2_arx5` is still required as written. (2) Fused-USD rebuilds do not need the old image: the URDF-merge path is confirmed and is the default (it builds on Lab 2.3.2 and Lab 3.0 EA). (2b) The frozen policy behaves worse on Lab 3.0 EA even for the stock Go2 (see §4.2 note), so decide the training/deploy stack before W4 and note that the frozen-policy check on the converted USD on the old stack was not completed and is deliberately not needed (W4 retrains on `go2_arx5` on the final stack; rsl-rl 5.x cannot load 3.x checkpoints). (3) Old rsl-rl 3.1.2 checkpoints may not load on rsl-rl >=4; retraining on the new stack implies new checkpoints anyway. (4) Fixed-base arm tuning (gravity disabled, friction 5) must not be carried into legged training.

## 4. Phase 1.5 decisions (arm integration)

1. **Arm role during locomotion training.** A (recommended): legs-only policy, arm stowed or randomized as a load, arm driven by differential-IK task-space control. Matches Mimic's EE-action requirement and the G1 example. B: whole-body EE-tracking controller (LeggedManip_Lab WBC style), harder to train and to gate. Re-evaluate only if A fails under arm motion.
2. **Checkpoint lineage.** Arm adds 3.35 kg to ~15 kg (~22%) plus a CoM shift. `go2_locomotion_final.pt` is assumed not to transfer (unmeasured; see the W1.5a caveat 2 for the inference-only test). Re-train and re-gate on `go2_arx5`; keep the legs-only checkpoints as baseline for comparison.
   **Correction 2026-10-05:** the "assumed not to transfer" claim was measured for `go2_locomotion_flat_v1_candidate` (the final checkpoint name was not used): it transfers on standing, commands and push recovery at 40-120 N (fused 20 DoF, arm stowed, leg joints driven by name). Harness caveat: needs `--fall_height_m 0.05`; the default 0.15 m fall rule misreads the policy's 0.11-0.17 m crouch. This is a short flat-ground inference test, not the Phase 0.5 gate; the re-gate is still required and retraining is a decision for W4, not forced by the mass increase.
   **Note 2026-10-05 (Lab 3.0 EA):** the Lab 2.3.2-trained frozen checkpoint behaves worse on Lab 3.0 EA even for the stock Go2 (standing tilt 22.4 deg vs 2.94 on the old stack; fused 59.7 deg, verdict `fails`), so the `transfers` result holds for the old stack only. Converted and stock-built USD behave the same on the same stack; the policy eval of the converted USD on the old stack was not completed (unverified).
   **Superseded 2026-10-06:** that tilt was Newton physics, not the checkpoint; with PhysX pinned the frozen ONNX policy walks a
   house on Lab 3.0 (max tilt 14.9 deg in clutter). The `fails` verdict above should be treated as a Newton result.
3. **Version coupling.** rsl-rl API differs across Lab versions; train and deploy on the same pinned stack.
4. **Camera conflict.** The arm sits on the Go2's back, conflicting with `perception.md`'s rear stereo camera on the camera-only ablation rig. Options: move the rear camera, mount it lower, or restrict that rig to front stereo plus side cameras. Flag to the user when W4 starts; do not silently change `perception.md`.

## 4b. Downstream modules that must move to the new stack (added 2026-10-05)

The new stack (Isaac Sim 6.1 / Lab 3.0 line / Python 3.12, images `cerberus-remote-v6:v6-dev` and `v6-ea-dev`) is the target for everything that touches the simulator. Only these places import isaaclab/isaacsim/omni (grep, non-test files): `policy/` (19 files), `common/` (8), `scripts/` (7), `slam/` (2: `slam/slam/eval/sim_env_cfg.py`, `slam/slam/ground_truth/sim_ground_truth_loop.py`). `perception/`, `planning/` and `system/` have none. Status per module:

- **SLAM sim eval (not migrated, not verified on the new stack).** It still runs on Isaac Lab's stock flat ground (`/World/ground`) with the old `UnitreeGo2FlatCerberusEnvCfg_PLAY`. Open work: (1) resolve `TODO(W1)` in `sim_env_cfg.py`: load a scene through `cerberus_worlds` (`registry.get_backend`) and re-point `scene.lidar.mesh_prim_paths` at the scene mesh prim (the lidar raycasts against scene geometry; the camera-only rigs do not); (2) the SLAM remote scripts (`scripts/run_slam_eval_remote.sh`) assume the OLD image's pixi `sim-ros2` environment, which supplies `rclpy` for the sim-to-ROS2 bridge relay; `docker/remote_v6` dropped that stage, so it must be re-added to the new image (or the relay moved to a separate container) before SLAM can run there; (3) `sim_ground_truth_loop.py` was migrated for XYZW quaternions and ProxyArray `.data` reads and unit-tested only; the Lab 3.0 EA `Imu` has no `quat_w` (the loop falls back to `root_quat_w`): run it once on a pod. Until then SLAM stays on the old image `cerberus-remote:v3` on flat ground, where it is unaffected by the arm work (SLAM does not use the fused robot).
- **Locomotion policy/eval (code migrated, behaviour not verified on the new stack).** `train.py`, `play.py`, `cli_common.py`, `best_checkpoint_runner.py` and the eval scripts go through the adapters in `common/cerberus_compat/` (smoke train exit 0 on rsl-rl 5.4.1 on the pinned image). The Lab 2.3.2-trained frozen checkpoint behaves worse on Lab 3.0 EA even for the stock Go2 (section 4.2 note); the cause is not diagnosed. See the open diagnostic below. rsl-rl 5.x probably cannot load the 3.x checkpoint format (inferred; `check_checkpoint_format` raises a clear error).
- **Scenes / worlds.** `plain_usd_backend` is the working GRScenes route (one scene validated of 69; one scene in R2); Arena hosts Lightwheel kitchens. Phase 1 navigation code must go through `registry.get_backend`, never import a backend directly.
- **Manipulation (W2a/W2b code).** Written for and run on the new stack; W2a passed on the OLD z=0 asset and was not re-run on the new default (see W2a status); W2b is blocked.
- **Open diagnostic (cheap, no training): why does the frozen policy tilt on Lab 3.0 EA?** Run the same checkpoint on a standing stock Go2 on both stacks, dump the policy input vector (term by term), the actions and the applied joint torques for the first ~200 steps, and diff. Different observations = a migration bug in our env config or adapters (fix it, no retraining needed). Identical observations but different torques/dynamics = a real physics or actuator difference (retrain on the new stack; this is already the W4 plan for `go2_arx5`). Candidate causes, none verified: actuator/effort-limit semantics changed in Lab 3.0 (migration guide: explicit actuator groups keep the authored solver effort limit, which can change clipping); our ported `UnitreeGo2FlatCerberusEnvCfg_PLAY`; Sim 5.1 to 6.1 solver/contact defaults. Do this before any retraining so a migration bug is not baked into the new policy.
  **RESOLVED 2026-10-06:** it was Newton physics + Newton actuators (the Lab 3.0 Go2 task default), not a migration bug and
  not a bad checkpoint; see the W4-prep status above. The diagnostic scripts (`diag_frozen_policy_obs.py`,
  `diag_compare.py`) were written but never needed on a pod and were deleted on 2026-11-06 along with the other solved-problem diagnostics.
- **Scene coverage open item:** Arena stalled building the GRScenes house with 16 per-env copies (`shared_scene=False`, 237 PhysX oversized-triangle warnings) while `plain_usd` with one shared copy (`shared_scene=True`) passed; only the smallest scene was tested. Untested hypothesis: per-env cloning of the large house mesh. Cheap checks: Arena with a shared scene / 1 env, and `plain_usd` on 2-3 more scenes (needs re-running `scripts/fetch_grscenes_to_r2.py --scenes N`, about 3 h on a 160 GB pod).

## 4c. Scene admission test (spec for a future agent; NOT built yet; added 2026-10-05)

**Why.** Only 1 of 69 GRScenes home scenes (smallest by `start_result_navigation.usd`, id `MVUHLWYKTKJ5EAABAAAAACY8_usd`, the one in R2 under `worlds/grscenes/home/`) and 2 of 100 Lightwheel kitchens (layouts 1:1 and 6:8) have been run. The mechanism works (plain_usd backend for houses, Arena for kitchens), but we cannot claim every scene works. Failures so far were deterministic, not random: Arena stalled on that house in `PhysxManager` warmup (17 min, 237 'TriangleMesh: triangles are too big' warnings) with 16 per-env copies (`shared_scene=False`), while `plain_usd` with one shared copy passed. The stock Arena `galileo` background let the Go2 fall through its floor on the EA image (its Gate A 'pass' is not meaningful for floor physics). Do not assume a scene works because the dataset is curated; admit scenes by test.

**Units (added 2026-10-06).** GRScenes is centimetre-authored; spawn poses, clearances and footprints are only meaningful in
METRES, so the admission sampler must open the metre wrapper (`describe_scene()['usd_path']`, i.e. `scene_metric.usd`), never the
raw scene USD. The backends already return it.

**Deliverable.** `scripts/validate_scenes.py` (pod script, same conventions as `scripts/arena_spike_go2.py`: report JSON written first, flush, `os._exit(code)`, faulthandler watchdog, `--dry-run`, access scenes only through `common.cerberus_worlds.registry.get_backend`) plus a colocated pure-python test for the verdict logic. It writes one manifest per source, e.g. `data/scenes/<source>/admission.json`, with one row per scene: `scene_id`, `backend`, `status` (`pass`/`fail`/`skip`), `reasons`, metrics, tool versions (Lab/Sim/Arena SHAs from `/etc/cerberus_versions.json`), date. Optionally mirror to R2 under the scene prefix (private; GRScenes is CC BY-NC-SA, never publish).

**Checks per scene** (headless physics first; rendering as a second, optional pass):
1. Load on the intended backend (`plain_usd` for GRScenes, `arena` for Lightwheel) with a small env count (4), then once at 16 to catch scale problems; enforce a per-scene build timeout (e.g. 5 min) so a stall becomes a recorded `fail`, not a hung pod.
2. Find a valid spawn: sample candidate xy positions on a grid over the scene footprint, cast a downward ray against the scene collision mesh, keep points with a floor hit and clear head/torso volume (no overlap with furniture), store the chosen spawn pose(s) per scene in the manifest. The default spawn (0,0,0) is NOT validated per scene (`GRScenesBackground(spawn_pose)` default); Phase 1 must read spawns from the manifest.
3. Spawn the Go2 (and, where relevant, the fixed-base Go2+ARX5) at the chosen spawn, step zero actions for ~200 steps: observations finite, root z within a band of the floor hit, no fall-through (this would have caught the Galileo case), base tilt below a threshold, step time and peak GPU memory recorded.
4. Lidar: one ray cast sweep from the robot lidar height must hit scene geometry (needed by SLAM; `scene.lidar.mesh_prim_paths` must be pointed at the scene mesh prim, see section 4b).
5. Optional render pass (needs a driver that renders; 570/580 passed on Sim 6.1): one RGB camera frame is non-empty, to catch missing MDL materials (`MDL_SYSTEM_PATH` must be set before app launch; sourced from `data/scenes/grscenes/home/env.sh` written by `scripts/fetch_grscenes_from_r2.py`).
6. Record `shared_scene` mode and whether the Arena path also works with `shared_scene=True` / 1 env (tests the per-env-cloning hypothesis from section 4b; if it holds, make Arena's `GRScenesBackground` default to a shared scene for houses).

**Data prerequisite.** Only one scene is in R2. Re-run `scripts/fetch_grscenes_to_r2.py --scenes N` on a pod with a 160 GB container disk (Runpod CPU pods cap the disk at 20-30 GB; the earlier extraction took about 3 h on an A5000 at $0.27/h, about $0.80). Start with N around 10 (by size order) and grow if the pass rate is low. Check the script's selection and symlink handling again with several scenes (it was only exercised on one).

**Use.** Phase 1 (navigation) samples train and held-out scenes only from `status=pass` rows, uniformly per category (home/commercial) as in `docs/worlds.md`; failed scenes stay listed with reasons. The held-out scene set is locked only after this manifest exists. Same script covers Lightwheel kitchens (sweep a stratified subset of the 10x10 layout/style grid, not all 100, to bound cost; the SDK downloads 57-112 MB per kitchen anonymously).

**Cost guide.** About 30-40 min of pod time per ~10 scenes (about $0.40 at $0.74/h) plus the extraction above. Follow `docs/runpod.md`: name pods `cerb-*`, delete explicitly, verify with `runpodctl pod list`, never touch other projects' pods, keep a spend cap in the task.

## 5. Anti-patterns

- Patching Isaac Lab/Arena/Sim files in the Docker image to make something work.
- Assuming Sim 6.0 vs 6.1, or that README version tables are current.
- Taking LeggedManip_Lab's policies or its Lab `main` requirement wholesale.
- Training before W3 completes.
- Letting policy/perception/safety import a backend directly.
- Deleting archive docs or the `push_disturbance_clear = None` fix.

## 6. Deliverables per phase

Update `docs/arena_integration_log.md` after each gate with: versions pinned, commands run, pass/fail, any deviation. Update `docs/worlds.md` open items as they resolve. Report back to the user at every gate, especially W1 (go/no-go) and any fallback taken.
