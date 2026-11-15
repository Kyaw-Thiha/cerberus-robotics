# Newton feasibility spike — kit-less Isaac Lab on the Newton backend

**Verdict: Conditional go as a second backend for *training and evaluation* on procedural terrain; no-go
for running the existing PhysX checkpoints on it; the scene path needs a lite asset pipeline before it can
even be evaluated.**

The kit-less stack installs and runs from a 0.25 GB base image on a plain RTX 4090 (and an RTX A5000) with
the *same* driver build (595.91.07) that crashes Kit, trains the stock Go2 rough task, records video through
its headless Newton GL viewer, and builds the full Cerberus Go2 task with the exact 235-D contract (order,
dims, joint order, timing, commands). What it does **not** do is reproduce the PhysX-trained policy: every one
of 384 environments falls within ~1.5 s. The two explanations that phase 2 could test are now closed:

* **joint/body ordering** — the public joint order on Newton already equals the contract order, and passing the
  explicit list changes nothing (fall rate 1.0 in both, medians 74 vs 77 steps);
* **the backend-native actuator path** — switching `use_newton_actuators` off leaves the transfer result
  unchanged (1.0 vs 1.0, medians 76 vs 71 steps) and leaves the rest pose identical to 0.05 deg, so the
  reported Newton double-PD issue (#5806) is not what breaks this policy;
* **the height scan hitting the robot** — it *did* hit it (5 % of cells, all under the body and feet), and
  preserving the stock `global_world_only` flag makes the flat-ground scan match the formula **187/187 cells
  within 1 cm**, but the transfer result does not move (1.0 vs 1.0, medians 79 vs 78 steps).

Scene loading still has no result, but the reason is now measured rather than suspected: one home mirror is
**24.9 GB**, of which **20.2 GB is the shared model geometry** and **4.7 GB is textures**, and the fetch target
`/workspace` is a **20 GB volume** — the phase-1 "120 GB container disk" never mattered.

Branch `newton-spike`. Code: `spikes/newton/` (new files only). Everything else in the repository is
unchanged; where a change to an existing file would be needed it is described below, not made.

## Answer per question

| # | question | answer | evidence |
|---|---|---|---|
| 1 | kit-less install + stock task | **yes** | install 820 s from a clean boot; 50 Newton iterations in 84 s at 128 envs, params finite, no NaN; **5/5** consecutive launches ok (~31 s each) |
| 2 | video on a headless pod | **yes** | 1920x1080 / 50 fps / 80 frames / 1.6 s mp4 from `--viz newton_gl --video`; no `$DISPLAY`, `dev_dri` present, libEGL/libGL/libOpenGL/libvulkan present |
| 3 | Cerberus task parity on Newton | **yes, MDP-identical** | 235-D obs, term order equal to the contract, joint order equal to the contract by name, decimation 4 @ dt 0.005, action dim 12, command ranges ±1, no NaN, 500/500 steps standing |
| 4 | policy transfer | **no**, and the two explanations on the table are closed | 100 % fall rate at levels 0–5 in four runs; ordering none/contract → medians 74/77 steps; actuators on/off → medians 76/71 steps; rest pose identical across actuator paths (≤0.05 deg) |
| 5 | scene load | **not reached**, cause measured | home mirror 24.9 GB (models geometry 20.2 GB, Materials textures 4.7 GB, per-scene USD 3.2 MB) against a 20 GB `/workspace` volume; a geometry-only fetch of the shared model library (20.2 GB) still does not fit |
| 6 | height-scan convention | **verified, and the self-hit contamination is fixed** | formula from the installed source: `pos_w_z - ray_hits_z - 0.5`; contaminated scan 94.8 % of cells within 1 cm (11 outliers under the torso and feet); with the stock `global_world_only=True`, **187/187 within 1 cm** and hits all at z = 0 |

## Host and environment

Two sessions, three pods created in total (phase 1: 2, the second because the first could not fetch its code
bundle; phase 2: 2, the second because the first pod's volume filled and the host then had no free GPU to
restart on — the per-session cap is 2, and the phase-2 session spent both). Every other boot was a
`POST /pods/{id}/start` resumption of the same pod, or an unsuccessful attempt at one.

Phase 2 host (all phase-2 numbers below come from it):

| | |
|---|---|
| GPU | NVIDIA RTX A5000, 24 564 MiB, **$0.27/h** |
| driver | 595.91.07 |
| image pull + boot | **~7 s** |
| install (`apt` + clone + `uv sync`) | **44 s** on this host (network-bound: 820 s on the phase-1 host for the identical command) |
| `/workspace` | **20 GB volume** (19 G free at boot) — not the requested 200 GB container disk |
| spend | ≈ **$0.25** of GPU time for the whole phase (pod 1 ~30 min at $0.27/h, pod 2 ~7 min at $1.09/h), far under the 1.5 USD / 90 min cap |

Phase 1 host (stage 1–3 numbers and the video come from it):

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 564 MiB |
| driver | **595.91.07** (CUDA 13.2 shown) — the build that crashes Kit at startup for the PhysX path |
| price | $0.89/h (secure cloud) |
| image | `nvidia/cuda:13.0.0-base-ubuntu24.04` (public Docker Hub tag, ~0.25 GB) |
| image pull + boot | **~3–5 s** (pod created 03:57:55Z → the job's first log line 03:57:58Z) |
| Isaac Lab | tag `v3.0.0` = **`e3f7d9ff`**, the commit the Cerberus `v6-ga` image pins |
| install (apt + clone + `uv sync`) | **820 s** first boot (deps 44 s, clone ~5 s, `uv sync` 820 s); ~4–5 min on the later boots |
| launches ok | **5/5** |
| python deps | isaaclab 28.0.0, isaaclab_newton 9.1.4, newton 1.6.1, warp-lang 1.17.0, torch 2.11.0+cu130, usd-exchange 3.0.1 (**OpenUSD 26.08**), rsl-rl-lib 5.5.1 |
| spend | **≈ $0.90** of GPU time (≈ 60 min at $0.89/h) plus ~$0.02 for the discarded first pod; under the 2.5 USD / 2 h cap |

The install command, at this tag:

```bash
git clone --depth 1 --branch v3.0.0 https://github.com/isaac-sim/IsaacLab.git /workspace/isaaclab
cd /workspace/isaaclab
VIRTUAL_ENV=/opt/labenv UV_PROJECT_ENVIRONMENT=/opt/labenv UV_PYTHON=/usr/bin/python3 \
UV_PYTHON_PREFERENCE=only-system uv venv --seed /opt/labenv
uv sync --frozen --inexact --extra video --extra importers
```

**The handoff's `./isaaclab.sh -i newton,tasks,assets,rl[rsl_rl]` does not exist at `v3.0.0`.** At that
commit `isaaclab.sh` is a 104-line deprecation shim; the documented kit-less install there is the one above,
which is what Isaac Lab's own `docker/Dockerfile.kitless` does (minus the `ov` extra, which brings OVRTX).
Newton, rsl-rl and `usd-exchange` are base dependencies of the tag, so no physics extra is needed.

## Stage 1 — boot and the stock Go2 rough task (pass)

```bash
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-UnitreeGo2 \
    --max_iterations 50 --viz none physics=newton_mjwarp env.scene.num_envs=128
```

* `Training time: 84.15 s` for 50 iterations at 128 envs, steady-state **0.38–0.41 s/iteration** (the first
  iterations carry compilation and CUDA-graph capture). This is a smoke run, not a learning result: mean
  reward drifts from −0.89 to −8.64 and `Episode_Termination/base_contact` is 0.98, i.e. an untrained policy
  still falls over. What it does show is that the step loop, the logger and the checkpoints are finite.
* `model_49.pt` written, `params_finite: true`, no `nan` anywhere in the log.
* **5/5** consecutive launches succeeded (31, 32, 31, 31, 31 s each), each a fresh `isaaclab train` process.
* Logs: `jobs/newton-20261115T035749Z/logs/stage1_train.log`, `stage1_launch_{1..5}.log`,
  `outputs/stage1/{smoke.json,launches.csv}`.

Nothing needed Kit: no `Simulation App Starting`, no RTX, no crash. The 595.91.07 driver that fails roughly
three launches in four on the PhysX path is irrelevant here.

## Stage 2 — video on a headless pod (pass)

```bash
isaaclab train ... --max_iterations 8 --viz newton_gl --video --video_length 80 --video_interval 0
```

* `clip_0000.mp4`: **1920x1080, 50 fps, 80 frames, 1.6 s, 269 KB**. Frame means ~(187, 191, 196) with a
  mid-clip std of ~(38, 33, 29) — a real rendered scene, not a black or static frame.
* Container graphics context: no `$DISPLAY`, `/dev/dri` present, `libEGL.so.1`, `libGL.so.1`,
  `libOpenGL.so.0`, `libGLX.so.0`, `libvulkan.so.1` all resolvable after installing `libegl1 libgl1
  libopengl0 libxrender1 libvulkan1 libx11-6 libxext6`. That is the whole requirement: `NewtonGLVisualizer`
  auto-selects the pyglet EGL backend when there is no display. No Xvfb, no RTX, no `--enable_cameras`.
* The GL probe itself (`spikes/newton/gl_probe.py`) failed to answer the "offscreen window" question on this
  run because it called `pyglet.canvas` without importing the submodule; fixed in the branch after the run
  (`1.6.x` pyglet requires the explicit import). The mp4 is the authoritative answer — video recording works.
* Logs: `logs/stage2_video.log`, `outputs/stage2/{gl_probe.json,video.json,clip_0000.mp4}`.

## Stage 3 — the Cerberus Go2 task on Newton (pass, MDP-identical)

`spikes/newton/tasks.py` subclasses `UnitreeGo2RoughCerberusEnvCfg_PLAY` (imported read-only) and changes
exactly two things: the physics preset (`RoughPhysicsCfg` → its concrete `newton_mjwarp` member, with
`use_newton_actuators=True`, as Lab 3.0's own Go2 cfg does) and the height scanner
(`RayCasterCfg` → `isaaclab_newton.sensors.NewtonRaycastSensorCfg`, which casts against the live Newton BVH
and ignores `mesh_prim_paths`). Task id `Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Newton-Play-v0`.

64 envs, 500 steps, zero action, `--viz none` (log: `logs/stage3_parity.log`, data:
`outputs/stage3/parity.json`):

| check | result |
|---|---|
| observation dim | **235** (`[64, 235]`) |
| observation term order | `base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions, height_scan` — **identical to the contract** |
| per-term dims | the live manager reports the group total `[235]`, not 8 per-term entries, so the equality check reports `False` as a granularity artefact; 3+3+3+3+12+12+12+187 = 235 is confirmed by the total |
| joint names and order | `FL_hip, FL_thigh, FL_calf, FR_…, RL_…, RR_…` — **identical to the contract**, and `backend_joint_names` is identical too, i.e. Go2 needs no ordering override at all |
| decimation / dt / step dt | 4 / 0.005 s / 0.02 s |
| action | `total_action_dim 12`, space `[64, 12]`, cfg scale 0.25 with default offset |
| command ranges | ±1.0 m/s x,y; ±1.0 rad/s yaw; 10 s resampling; 2 % standing |
| height scan on Newton | `NewtonRaycastSensor`, 187 rays, `ray_hits_w` shape `[64, 187]`, **100 % finite hits**, hit z in [−0.77, 1.18] m |
| physics backend actually used | `NewtonCfg` on `cuda:0` |
| 500 steps at zero action | no NaN, **0 base contacts**, no time-outs |
| speed | env build 22 s; 500 steps in 26 s (19 steps/s = 1 240 env-steps/s) |

One number to flag for the PhysX comparison: at zero action the base settles at **0.255 m** mean height
with a worst-case 0.20 m sink over the 10 s. The Go2 spawns at 0.40 m. Phase 2 measured that settlement
joint by joint (below) and it is gravity, not the actuator path.

## Stage 4 — policy transfer (fail, and ordering is not the reason)

The PhysX-trained candidate `go2_locomotion_ga_v1_candidate` (`model_1499.pt`, split rsl-rl format) loads
without rsl-rl: `spikes/newton/policy_loader.py` rebuilds the actor from `actor_state_dict`
(`mlp.0/2/4/6`, 235→512→256→128→12, ELU) and returns a callable. It reports no normaliser tensors, matching
the contract. 384 envs (64 per level 0–5), pinned terrain levels, one fixed forward command 0.8 m/s,
1000 steps, no tuning:

| ordering flag | fall rate | median steps to first fall | mean abs vx err | mean abs vy | mean abs wz |
|---|---|---|---|---|---|
| `none` (backend order) | **100 %** (384/384) | 74 | 0.823 m/s | 0.308 m/s | 1.29 rad/s |
| `contract` (explicit 12-name list) | **100 %** (384/384) | 77 | 0.831 m/s | 0.308 m/s | 1.29 rad/s |

Per level, both runs are flat: fall rate 1.0 at every level 0–5, median first fall 69–83 steps, mean alive
time 80–88 steps. Death is not terrain-dependent, which is the signature of a policy-facing mismatch rather
than of terrain difficulty — and the handoff's diagnostic ("falls within a few dozen steps in every
environment signals an ordering mismatch") fires. But **ordering is ruled out** by stage 3: the public joint
order equals the backend order equals the contract order, and `joint_ordering_active` normalises to `False`
(no permutation installed) even when the explicit list is passed. Passing the flag changes nothing (fall
rates 1.0 vs 1.0, medians 74 vs 77).

Phase 2 ran the same rollout a second time with the other actuator path
(`transfer_eval --use_newton_actuators false`, the wired `stage4b`), on the phase-2 host:

| actuator path | fall rate | median steps to first fall | mean abs vx err | steps/s |
|---|---|---|---|---|
| `use_newton_actuators = true` (stock Newton path) | **100 %** (384/384) | 76 | 0.824 m/s | 15.2 |
| `use_newton_actuators = false` (PhysX-style implicit actuators) | **100 %** (384/384) | 71 | 0.829 m/s | 14.6 |

Per-level medians 68–81 steps in both, i.e. 1.4–1.6 s. **The actuator path is ruled out**, and the rest-pose
test below shows why it cannot matter for this robot: the two paths produce the same joint steady state to
within 0.05 deg. That is also the answer to "run the stock solver preset": every rollout in both sessions ran
on the stock `RoughPhysicsCfg.newton_mjwarp` preset (recorded to `outputs/solver_params.json`: `njmax 1000`,
`nconmax 300`, `cone pyramidal`, `impratio 1.0`, `integrator implicitfast`, `use_mujoco_contacts false`,
`num_substeps 2`, `shape margin 0 / ke 160 000 / kd 1100`). Those are the preset's values; nothing was tuned,
and there is no second "stock" configuration to bisect against. `njmax`/`nconmax` are contact *budgets*, not
dynamics parameters.

What remains, in the order I would test it next:

1. **the height scan sees the robot itself.** The Ray Caster page says it plainly: with Newton "the sensor
   queries the live scene BVH and therefore sees every collision shape in its environment, global terrain,
   **and dynamic bodies**", and `mesh_prim_paths` is ignored. The Go2 is the only dynamic body in the scan
   volume, so rays under the base can hit its own legs instead of the ground, which corrupts the 187-cell
   observation for exactly the cells the policy uses to place its feet. Phase 1's probe recorded hits up to
   **+1.175 m** above the ground plane on rough terrain, which a 0.3 m-high base cannot explain — consistent
   with self-hits, though stairs are also possible. One run settles it: record, per cell, which prim each
   ray hits (or compare the scan against a PhysX scan of the same terrain at the same pose, which the contract
   already calls `unverified`).
2. **height-scan convention** — the formula is now verified from the installed source (next section), but the
   *values* under Newton have not been compared with PhysX on identical terrain.
3. **solver contact/friction parameters** (`ke`, `kd`, `cone`, substeps) — last, because the policy falls
   identically at level 0 (flat-ish boxes) and level 5 (stairs), which is not how a friction/contact mismatch
   usually scales.

PhysX reference at 0 N push, the only unperturbed column in the existing sweeps
(`policy/locomotion/runs/2026-09-08_19-28-32_analysis/eval/eval_rough_full_sweep.csv` and
`data/reports/ga_retrain/eval_rough_ga_candidate.csv`, 100 trials/cell):

| terrain level | GA candidate (PhysX, 0 N) | September (PhysX, 0 N) | Newton (this spike) |
|---|---|---|---|
| 0 | 88 % | 94 % | 0 % |
| 4 | 83 % | 86 % | 0 % |
| 8 | 76 % | 65 % | not run (levels 0–5 only) |

The protocols are not identical — the sweeps are push-recovery trials with a command drawn per episode,
this is a fixed 0.8 m/s forward command for 20 s — so read the gap as "no transfer at all", not as a
percentage-point comparison. Levels 1, 2, 3, 5 have no PhysX counterpart in the existing CSVs at all.

Logs: `logs/stage4_{none,contract}.log`, `outputs/stage4_{none,contract}/{transfer.json,per_level.csv}`.
Cost of the whole stage: 2 x (19 s build + 125 s rollout) — Newton ran 8.5 policy-steps/s at 384 envs.

## Stage 5 — rest pose: where the joints actually settle (phase 2)

No policy. 16 envs, zero action (so the action term commands `default_joint_pos` exactly), 200 steps at 50 Hz,
settled joint positions averaged over the envs, compared with the contract's default angles. Run twice, once
per actuator path:

| joint | commanded (contract) | settled, actuators on | offset | settled, actuators off | offset |
|---|---|---|---|---|---|
| FL_hip | +0.100 | +0.2235 | **+7.08°** | +0.2242 | +7.12° |
| FL_thigh | +0.800 | +0.7780 | −1.26° | +0.7776 | −1.29° |
| FL_calf | −1.500 | −1.7186 | **−12.52°** | −1.7186 | −12.52° |
| FR_hip | −0.100 | −0.2229 | −7.04° | −0.2239 | −7.10° |
| FR_thigh | +0.800 | +0.7640 | −2.06° | +0.7645 | −2.03° |
| FR_calf | −1.500 | −1.7152 | −12.33° | −1.7148 | −12.31° |
| RL_hip | +0.100 | +0.2894 | **+10.85°** | +0.2898 | +10.88° |
| RL_thigh | +1.000 | +0.9412 | −3.37° | +0.9406 | −3.41° |
| RL_calf | −1.500 | −1.7620 | **−15.01°** | −1.7627 | −15.05° |
| RR_hip | −0.100 | −0.2742 | −9.98° | −0.2745 | −10.00° |
| RR_thigh | +1.000 | +0.9377 | −3.57° | +0.9376 | −3.57° |
| RR_calf | −1.500 | −1.7574 | −14.75° | −1.7568 | −14.72° |

Base height 0.4553 m at spawn → **0.3039 m** after 4 s (0.3038 m with the other path).

* **max |offset| 15.01° / 15.05°, mean |offset| 8.32° / 8.33°** — the two actuator paths agree to **0.05°** on
  every joint, so the double-PD issue (#5806) is not active here and cannot be the cause of the falls.
* The pattern is the gravity-loaded steady state of a position-controlled quadruped: both calves sag 12–15°
  (they carry the body), the hips abduct 7–11° (mirrored left/right, i.e. the legs fold outward), the thighs
  move 1–3.6°. It is **not** a uniform offset in one direction, which is what a re-enabled target stiffness
  or an offset-bias bug would produce.
* The spread across the 16 envs is 0.04–0.14 rad (2–8°): the envs stand on different rough-terrain cells, so
  the settlement is terrain-dependent — expected, and worth knowing when reading a PhysX-vs-Newton standing
  comparison.
* Evidence: `outputs/rest_pose_{true,false}/rest_pose.json`, `logs/rest_pose_{true,false}.log` under
  `jobs/newton-20261115T153614Z/`.

What this means for the report: the 0.20 m sink seen in stage 3 is real but ordinary, and it is reproduced
identically with PhysX-style actuators — so it explains why a PhysX policy *starts* lower on Newton than it
expects, but not why it falls within 1.5 s.

## Stage 6 — height-scan convention: the formula (phase 2)

The observation term the environment actually imports, read from the installed package **on the pod**
(`inspect.getsource`, saved verbatim to `outputs/height_scan/height_scan_source.txt`):

```python
def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w.torch[:, 2].unsqueeze(1) - sensor.data.ray_hits_w.torch[..., 2] - offset
```

So, at this tag: **height = sensor_frame_z − ray_hit_z − 0.5**, no clip and no sign flip inside the function;
`pos_w` is the parent frame (the base link), and the offset default is 0.5. The contract's `clip: [-1.0, 1.0]`
belongs to the observation term, not to this function. That is consistent with the contract file's stated
formula, including the sign — i.e. the *convention* is not misread.

The flat-ground value was not measured in phase 2 (`height_scan_check.py` failed twice on my own bugs, then
the pod could not be restarted). **Phase 3 measured it** and the result is in stage 8: with a clean sensor the
187 cells are all within 1 cm of `base_z − 0.5` and every hit is at z = 0.

## Stage 7 — scene payloads and the container volume (phase 2)

Measured on the R2 side before downloading anything (`spikes/newton/scene_assets.py --list`, 22 482 objects):

| component | geometry | textures | material defs | other | total |
|---|---|---|---|---|---|
| `models/` (shared by all 18 homes) | **20.2 GB** | 0 | 0 | 4.7 MB | 20.2 GB |
| `Materials/` | 0 | **4.7 GB** | 1.2 MB | 0 | 4.7 GB |
| one scene dir (`scenes/<id>_usd/`) | **3.0–3.3 MB** | 0–339 KB | 0 | 55–91 KB | ~3.2–3.7 MB |
| **whole mirror** | 20.2 GB | 4.7 GB | 1.2 MB | 5.9 MB | **24.9 GB** |

The deployment fact that explains phase 1's `ENOSPC`: **`/workspace` is a 20 GB volume** (`df` in the pod:
19 G total, 99 M used at boot), not the 120 GB container disk the pod was created with. The fetch target is
`/workspace/...`, so the 24.9 GB home mirror never could fit, and a larger container disk would not have
helped.

The geometry-only attempt (`--fetch-lite`: `scenes/<id>/**` + `models/**` + `manifest.json`, USD/USDZ/JSON
only, `*.mdl` excluded) still asks for **20.2 GB** of shared model geometry against 19 GB free, and was
stopped by its 20-minute box without producing a scene to parse. The per-scene USD is only 3.2 MB; the cost is
entirely the model library, most of which a given home does not reference.

**Proposal (not built) — a geometry-only "lite" scene USD.** For each scene: (1) resolve the model references
from `start_result_navigation.usd` with `pxr` (the same `usd-exchange` runtime Newton uses) and copy only those
model folders from R2; (2) flatten the stage (`usdcat --flatten`) and prune everything that is not collision
geometry — drop `UsdShade`/MDL bindings, texture readers and preview surfaces, keep `UsdPhysics.CollisionAPI`
meshes with `physics:approximation="convexHull"` (concave for floors) — then (3) verify no external asset
paths and no material bindings remain, (4) write it to R2 per scene as `<scene>_lite.usd`, expected tens of MB
rather than GB, and (5) run the existing parse probe plus a 200-step stand on it. That is what makes stage 5
(and the navigation lane) testable on a 20 GB volume at all; it needs no change to any existing file, and the
colliders are the one thing that must survive.

## Stage 8 — ray attribution and the scan fix (phase 3)

Flat ground, the Go2 held at its default joint targets (100 steps to settle), 4 envs, the 187-cell scan dumped
(`spikes/newton/ray_attribution.py`). On a plane the formula predicts the same value in every cell:
`base_z − 0 − 0.5 = 0.2681 − 0.5 = −0.2319`.

| sensor config | cells within 1 cm | hit z range | scan value |
|---|---|---|---|
| `global_world_only = false` (phases 1–2) | **176–178 of 187 per env (94.8 % pooled)** | 0.0 – 0.391 m | −0.54 m in the bad cells |
| `global_world_only = true` (stock cfg flag, restored) | **187/187 in every env** | 0.0 m exactly | −0.2319 m, i.e. `base_z − 0.5` |

The 11 outliers per env, in the 17 x 11 grid (row-major, 0.1 m spacing, rows run along x, cols along y):

```
  ...........        rows 5-11 of 17 -> x in [-0.3, +0.3] m
  ......X....        cols 1-9 of 11  -> y in [-0.4, +0.4] m
  ........X..        i.e. directly under the torso, both sides
  .X.........
  ....XXX.X..
  .........X.
  ..X........
  ...X....X..
```

* **8 of the 11 outliers hit at z = 0.310–0.333 m** — the underside/top of the base link's own collision
  geometry (the base origin sits at 0.268 m, the torso surface extends above and below it). These are quarters
  and fore/aft parts of the body.
* **3 of the 11 hit at z = 0.025–0.066 m** — feet and lower legs.
* **No hit identifiers exist to report.** `NewtonRaycastSensorData` exposes only `pos_w`, `quat_w`,
  `ray_hits_w`, `ray_distances`, `ray_normals_w`, and the sensor calls
  `newton.intersect_ray(..., out_dist, out_normal)` — distance and normal only, no shape or prim id. The
  attribution above therefore comes from the hit *height* and the cell's position; both are unambiguous, since
  a ground hit on a plane can only be z = 0.

**The fix is a flag I had dropped, not a Newton limitation.** `NewtonRaycastSensorCfg` inherits
`RayCasterCfg.global_world_only`, and the stock velocity cfg sets it to `True` for exactly this reason. With
it, the sensor passes `ray_worlds = -1` and `enable_global_world = False` to `newton.intersect_ray`, so rays
query the global world (terrain, plane) only and every dynamic body — including the robot carrying the sensor
— is excluded. Restored in `spikes/newton/tasks.py` (`global_world_only=True`, the default), verified above:
**187/187 cells within 1 cm, hits at z = 0 for every ray.**

Excluding the robot by *shape* instead (a per-shape ray filter) is not exposed by this sensor config or by
`newton.intersect_ray`; `global_world_only` is the supported mechanism at this tag, and it is the one the
stock task already used.

## Stage 9 — transfer with a clean scan (phase 3), and what is left (phase 3)

Same rollout as before (384 envs, levels 0–5, fixed 0.8 m/s, 1000 steps), 4090, both sensor settings on the
same host:

| scan | fall rate | median steps to first fall | mean abs vx err | per-level medians |
|---|---|---|---|---|
| contaminated (`global_world_only=false`) | **100 %** (384/384) | 78 | 0.828 m/s | 77, 77, 79, 79, 77, 78 |
| **clean** (`global_world_only=true`, 187/187 correct) | **100 %** (384/384) | **79** | 0.818 m/s | 85, 70, 82, 81, 72, 87 |

Against the PhysX reference (88 % at level 0, 0 N; 83 % at level 4): Newton is at 0 % on the same levels. The
scan was genuinely contaminated and is now genuinely clean, and the policy falls exactly as before — so the
height scan is **not** the cause either.

Five explanations are now closed by measurement, across three sessions: joint/body ordering, the
backend-native actuator path, the double-PD issue #5806, the height-scan formula, and height-scan
contamination. Remaining suspects, cheapest test first:

| # | suspect | one test | cost |
|---|---|---|---|
| 1 | **observation *values* other than the scan** (projected gravity / base velocity frames or scales differ between backends even though the layout matches) | dump both backends' 235-D observation for the same pose and command, diff term by term | ~5 min of pod time, one run |
| 2 | **action path**: does `default_joint_pos + 0.25 x` land where the policy expects (offset sign, joint target ordering inside the action term) | one rollout with the policy's first action recorded and compared against the PhysX run's | ~10 min |
| 3 | **does it fall while standing?** zero forward command with the policy in the loop, 500 steps | 1 run: if it falls standing, the gait is irrelevant and the actuator/contact model is | ~5 min |
| 4 | **contact model** (`ke 160 000`, `kd 1100`, `cone pyramidal`, `impratio 1.0`): the stock preset only; any change is tuning and out of scope here | substepping (`num_substeps 2 -> 4`) is the one lever the preset names | ~5 min per variant |
| 5 | **the policy simply does not transfer** and needs a Newton-conditioned retrain | one Newton retrain + sweep (cost below) | ~1–1.5 h of GPU |

**Estimated cost of one Newton retrain** (Go2 rough, Cerberus config, stock recipe, 1500 iterations), from the
measured iteration time: phase 1 ran **0.38–0.41 s/iteration at 128 envs** on a 4090 (steady state, after
warm-up). Scaling linearly in env count to 4096 (32x):

* 12.2–13.1 s/iteration → **5.1–5.5 h** of 4090 time → **$4.5–4.9** at $0.89/h, or **$1.4–1.5** on an A5000
  class host at $0.27/h;
* that linear scaling is an **upper bound**: at 128 envs the GPU is mostly idle, and the same recipe on PhysX
  at 4096 envs took 4009 s (1.1 h, `docs/h2_report.md`). The honest range is therefore **1.1–5.5 h**, which
  is too wide to plan with;
* pin it with one **50-iteration probe at 4096 envs** (~2 min of install already paid, ~10 min of run time at
  the upper-bound rate, <$0.15) before committing to a retrain.

No retrain was run, and nothing was tuned.

## Decision table

| axis | verdict | evidence |
|---|---|---|
| install and boot reliability | **Go** | 820 s from a clean boot on one host, **44 s** on another for the identical command (network-bound); no Kit, no RTX, no startup crash; **5/5** launches on driver 595.91.07 where the PhysX path manages ~1 in 4; ~3–7 s image pull |
| video recording | **Go** | 1080p50 mp4 through the Newton GL viewer with no display and no RTX; only `libegl1/libgl1/libopengl0/libxrender1/libvulkan1` needed |
| Cerberus task parity | **Go** | 235-D, contract term order, contract joint order (no permutation needed), decimation/dt, action dim, command ranges all identical; height scan on the Newton sensor with 100 % finite hits; 500/500 standing steps |
| standing / actuator behaviour | **Go, with a caveat** | rest pose settles 8.3° mean / 15.0° max from the commanded defaults and the base sinks 15 cm — gravity, identical (≤0.05°) with and without the backend-native actuator path; no sign of #5806 |
| height-scan convention | **Go** | formula and offset from the installed source; contaminated scan 94.8 % of cells within 1 cm (11 outliers under the torso and feet), clean scan **187/187 within 1 cm** once the stock `global_world_only` flag is kept |
| policy transfer | **No-go (as shipped)** | 100 % falls in six runs: ordering 74/77 steps, actuators 76/71, contaminated/clean scan 78/79; all six agree |
| scene loading | **No-go at this volume** | home mirror 24.9 GB vs a 20 GB `/workspace`; per-scene USD 3.2 MB, shared model geometry 20.2 GB; a lite geometry USD is proposed, not built |

Overall: **Conditional go** for a Newton *training and evaluation* lane on procedural terrain — install, video,
task parity, standing and the height scan all measure clean, so nothing about the *task* blocks a from-scratch
training run. **No-go** for claiming the existing PhysX checkpoints run on Newton: six runs across three
sessions collapse the same way, and the five explanations the spike could test are closed. **No-go** for scenes
until a lite asset path exists.

## Effort estimate for adoption

Two separate lanes, because they do not share a blocker.

### Minimal lane: train and evaluate on procedural terrain (no scenes)

Estimates from what this spike actually measured: 44–820 s install (network-bound), ~20 s env build, 1 240
env-steps/s at 64 envs and ~3 300 at 384 (4090), ~70 min for a 4096-env retrain on the existing PhysX recipe.

| item | estimate |
|---|---|
| physics plug-in: `common/cerberus_physics/registry.py` (`PHYSICS_BACKEND`, lazy factories, `physx` + `newton_mjwarp`, no silent default — the `cerberus_worlds.registry` pattern), with `pin_physx` becoming the `physx` adapter | 2 days |
| Newton adapter: physics preset, actuator path, height-scan sensor swap, ordering presets, recorded in saved params | 2 days |
| certify Go2 flat/rough × train/play + the push-recovery eval on Newton (same MDP, the existing contract check plus a zero-command standing check in front of each run) | 2–3 days |
| settle the transfer question: term-by-term observation diff against PhysX, a standing-with-policy run, then one Newton retrain and its sweep (the ray-attribution and scan-convention parts are done) | 1–2 days + 1–2 GPU sessions (1.1–5.5 h each, see stage 9) |
| PP/PN/NN/NP evaluation columns and report tooling (the sweep axes stay) | 1–2 days |
| kit-less CI lane + a ~3 GB image with the `uv` environment baked in (44–820 s of install per boot is the only recurring cost) | 2 days |

**≈ 2 weeks (9–11 person-days)** for a minimal, certified second backend on procedural terrain, including
one Newton retrain. That is the lane I would fund: it needs no scene, no camera work and no new task code.
Budget the retrain at **$1.4–4.9 and 1.1–5.5 h** until a 50-iteration probe at 4096 envs pins the rate.

### Scene lane (separate, gated)

| item | estimate |
|---|---|
| lite geometry USD pipeline (reference resolution, flatten, material strip, R2 per-scene artifact) | 3–5 days |
| volume/asset budget: 20 GB is the default `/workspace`; a real home needs the lite artifact plus a stated volume size | 0.5 day + infra decision |
| Newton USD import + spawn-pose verification + the existing occupancy/scan proxies on the lite asset | 3–5 days |
| `nav_eval` / `scene_env` on Newton, clips through a Newton renderer | 5–10 days |

**≈ 4–6 weeks**, and it should not start before the lite asset exists — every previous attempt to reach the
scene path died on payload size, not on physics.

### What would need retraining

Any policy that must be certified on Newton should be trained on Newton (NN) or with solver randomization
(friction, armature/damping, substeps, actuator response). The shipped PhysX checkpoints can be *evaluated* on
Newton today (that is what stage 4 did) but this spike gives no reason to believe they will behave, and the
height-scan question has to be answered before a Newton retrain is even meaningful.

## Limitations and what could not be verified

* **One host, one GPU** (RTX 4090, driver 595.91.07) and one driver build. The driver rule allowed 580+; no
  other build was seen, so "Newton works on 595.x" is one measurement, not a distribution.
* **The transfer cause is narrowed, not identified.** Ordering (phase 1) and the actuator path plus the
  double-PD issue (phase 2) are ruled out by measurement. The leading remaining hypothesis — the Newton ray
  caster hitting the robot's own collision shapes — is stated from the documentation and one suspicious hit
  height, and was **not** tested. The other open item is the height-scan *value* comparison against PhysX.
* **No policy retrain on Newton** in either session (in phase 2 the height-scan convention should be settled
  first, or the retrain inherits the same suspect input channel).
* **The flat-ground height-scan value was measured in phase 3 only** (stage 8); the phase-2 attempt failed on my
  own script bugs before the pod could not be restarted.
* **No scene parse, load, or OpenUSD crash verdict.** An OpenUSD 26.08 runtime means the known
  collider-rich-body crash is not expected, and `PXR_WORK_THREAD_LIMIT=1` is the documented workaround, but
  neither was exercised.
* **No PhysX number was re-measured on this pod.** The comparison uses the committed GA and September sweeps,
  whose protocol differs from this rollout (see stage 4).
* **Height-scan convention** is still `unverified` in the contract; this spike confirmed the formula and the
  0.5 offset from the installed source and that the sensor produces 187 finite hits — not that the values
  match PhysX on the same terrain.
* **Video quality** is judged from frame statistics and the file, not by watching the clip.
* **Multi-env scenes, camera sensors, perception, SLAM** were out of scope and untouched.
* **Container state across stop/start:** `/opt/labenv` and `/workspace/isaaclab` were gone after a stop (a
  full reinstall was needed), while data written to the `/workspace` **volume** survived — which is how a
  filled volume blocked the next boot. Budget a cold install on every boot (44–820 s depending on the host's
  network) or bake the image.
* **Phase 2 used both of its pod creations**, and the second pod could not be restarted
  (`start pod: There are not enough free GPUs on the host machine`); the phase-3 session retried the flat scan
  as a separate pod instead of a restart, and got it.
* **The scan fix changes nothing about the transfer result.** It is a real bug fix (5 % of the observation was
  wrong on flat ground, more on rough) but it is not the cause of the collapse, and the report does not claim
  otherwise.

## Mistakes made here (and what they cost)

Recorded because they are the kind of thing that eats a two-hour budget:

1. The first pod could not fetch its code bundle: my submit script addressed R2 as `cerberus_r2/spike/...`
   (a local path) instead of `cerberus_r2:<bucket>/spike/...`. One creation lost, ~$0.02.
2. The stock task id is `Isaac-Velocity-Rough-UnitreeGo2` at this tag (no hyphen before `Go2`); the stage-1
   and stage-2 commands used the older spelling and exited in 5 s each. Fixed and re-run.
3. Two of my stage scripts assumed Isaac Lab 3.0 shapes that differ (`active_terms` is a list on the action
   manager, not a dict; tensor bookkeeping on the wrong device). Both were caught by the first real run and
   fixed; the parity and transfer results above come from the runs after the fixes.
4. `session_guard.py --provider runpod` cannot stop a RunPod pod: it requires `VAST_API_KEY` before it looks
   at the provider, and then passes that key to the RunPod stop. `spikes/newton/pod_guard.py` reuses its
   decision function and state file, and stops through `runpod_provider.stop_pod`. **Suggested change in
   `scripts/session_guard.py`: read the key per provider and pass `RUNPOD_API_KEY` on the runpod branch.**
5. `scripts/watch_job.py` mixes `scripts.x` and bare imports, so it only runs with
   `PYTHONPATH=<repo>:<repo>/scripts` as a module; my first spawn crashed twice before that was clear. Not
   fixed (existing file): run it as `python3 -m scripts.watch_job` with both paths on `PYTHONPATH`.
6. The base image ships `rclone` 1.58, which rejects the `Cloudflare` S3 provider — the same trap
   `scripts/sync_to_r2.py` documents. The driver now installs the pinned 1.75.1 even when a binary exists.
   Before that fix the R2 calls worked by falling back to generic S3, so the symptom was a warning, not an
   error.
7. `scripts/create_pod.py` defaults are PhysX-shaped (35 GB image, 100 GB disk, port 22). Nothing needed
   changing for this spike (the job path lives in `spikes/newton/`), but a Newton lane should default to the
   small image and no ports.
8. **Phase 2, height scan:** the check shipped with a duplicate `--steps` flag, then with a term lookup that
   assumed the observation manager exposes term objects (this build exposes a list of names). Both are fixed
   in the branch; the stage cost two boots and was still not measured.
9. **Phase 2, disk:** I asked for a 200 GB container disk but `/workspace` is a 20 GB **volume**, so the
   container-disk size never applied to the fetch target. The size listing should have come before the first
   fetch in phase 1 — it is one `rclone lsf` and it would have prevented the 25 GB download into 19 GB.
10. **Phase 3, flat terrain:** swapping the terrain for a plane without disabling `curriculum.terrain_levels`
    crashed both attribution runs at the first `env.reset()` (the stock curriculum indexes
    `terrain_generator.size`, which a plane does not have). One pod boot lost (~$0.12); fixed and re-run as a
    second pod rather than a restart, per the session rule.

**Hygiene:** a stop/patch response body echoes the pod's full `env` map, including the base64 bootstrap and
`RUNPOD_API_KEY`. Phase 1 printed one such body (truncated by the client); phase 2 never prints or stores a
response body — `spikes/newton/submit_run.py:safe_body` strips `env`/`dockerStartCmd`/`dockerEntrypoint` and
masks the rest through the repo's redaction helper, and `pod_guard.py` records only the HTTP status. Rotate
`RUNPOD_API_KEY` if the phase-1 output was captured anywhere.

## Artifacts

* Code and this report: branch `newton-spike`, commits `8c08b2b..HEAD` (local; not pushed — SSH auth is
  broken). Pod code travels as a git bundle.
* Phase 3 R2 prefixes:
  * `cerberus-robotics/jobs/newton-20261115T175131Z/` — `outputs/{host.json,transfer_gwo_true,transfer_gwo_false}`
    (the clean-scan transfer and the contaminated comparison), `logs/{ray_false,ray_true,transfer_gwo_true,
    transfer_gwo_false}.log`. The two `ray_*` runs in this prefix failed on the plane/curriculum bug.
  * `cerberus-robotics/jobs/newton-20261115T180119Z/` — `outputs/ray_{false,true}/{ray_attribution.json,
    grid_env0.txt}`, `logs/ray_{false,true}.log`, `host.json` (the attribution measurement).
* Phase 2: `jobs/newton-20261115T153614Z/` (rest pose, transfer A/B, scene sizes) and
  `jobs/newton-20261115T160621Z/` (the installed `mdp.height_scan` source). Phase 1:
  `jobs/newton-20261115T035749Z/` (install, stock-task smoke, video, parity, ordering transfer).
* Pods `rwmhwno9bflsbf`, `ofcqe4o85uqsjh`, `62f3leopdsfgu1`, `dwf452id9j7f6e`, `5v3r6d7nl7l3xu` are all
  **stopped** (`EXITED`), not deleted: the owner deletes them. The phase-1 pod's volume still holds its partial
  scene fetch.
* Reproduce: `python3 spikes/newton/submit_run.py --dry-run`, then `--start-watch`; phase 3 uses
  `--bootstrap spikes/newton/newton_job3.sh --set CERBERUS_ONLY=ray` (attribution only) or without `--set`
  (attribution + transfer); results with `python3 spikes/newton/collect_results.py --job-id <job>`.
* Phase-3 spend: ≈ **$0.18** of GPU time across two pods (12 min at $0.89/h, 6 min at $0.59/h), inside the
  1.0 USD / 60 min cap. All pod logs are the containers' own files; no API response body is printed or stored
  (`submit_run.py:safe_body` strips the pod `env` map and masks through `scripts/redact.py`).
