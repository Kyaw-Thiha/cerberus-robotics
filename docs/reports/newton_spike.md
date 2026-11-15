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
  reported Newton double-PD issue (#5806) is not what breaks this policy.

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
| 6 | height-scan convention | **formula verified, value not measured** | installed `mdp.height_scan` reads `pos_w_z - ray_hits_z - offset` (offset 0.5, no clip in the function); the flat-ground measurement did not run (script bugs, then the pod could not be restarted) |

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

## Stage 6 — height-scan convention (formula verified, value not measured)

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

The measured flat-ground value was **not obtained**. The check (`spikes/newton/height_scan_check.py`) builds
the task with a plane terrain and compares the height-scan slice of the observation cell by cell with the
formula. It failed twice on my own bugs — a duplicate `--steps` flag, then a term lookup that assumed the
observation manager exposes term *objects* while this build exposes a list of names — and by the time both
were fixed the pod could not be restarted (its host had no free GPU) and the session's creation budget was
spent. What the check would print is `base_z − 0 − 0.5` per cell (≈ **−0.20 m** at the settled 0.30 m base
height), which is what the formula gives; that number is a prediction, not a measurement, and is not used as
evidence anywhere in this report.

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

## Decision table

| axis | verdict | evidence |
|---|---|---|
| install and boot reliability | **Go** | 820 s from a clean boot on one host, **44 s** on another for the identical command (network-bound); no Kit, no RTX, no startup crash; **5/5** launches on driver 595.91.07 where the PhysX path manages ~1 in 4; ~3–7 s image pull |
| video recording | **Go** | 1080p50 mp4 through the Newton GL viewer with no display and no RTX; only `libegl1/libgl1/libopengl0/libxrender1/libvulkan1` needed |
| Cerberus task parity | **Go** | 235-D, contract term order, contract joint order (no permutation needed), decimation/dt, action dim, command ranges all identical; height scan on the Newton sensor with 100 % finite hits; 500/500 standing steps |
| standing / actuator behaviour | **Go, with a caveat** | rest pose settles 8.3° mean / 15.0° max from the commanded defaults and the base sinks 15 cm — gravity, identical (≤0.05°) with and without the backend-native actuator path; no sign of #5806 |
| height-scan convention | **Conditional** | formula and offset verified from the installed source; the flat-ground value was not measured, and Newton's sensor sees dynamic bodies (self-hits) which is now the leading suspect |
| policy transfer | **No-go (as shipped)** | 100 % falls in every run: ordering none/contract 74/77 steps, actuators on/off 76/71 steps; four independent runs agree |
| scene loading | **No-go at this volume** | home mirror 24.9 GB vs a 20 GB `/workspace`; per-scene USD 3.2 MB, shared model geometry 20.2 GB; a lite geometry USD is proposed, not built |

Overall: **Conditional go** for a Newton *training and evaluation* lane on procedural terrain — install, video,
task parity and standing all measure clean, and none of the five things that would block a from-scratch
training run is missing. **No-go** for claiming the existing PhysX checkpoints run on Newton (four runs, same
collapse; ordering and actuators ruled out). **No-go** for scenes until a lite asset path exists.

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
| settle the transfer question: ray-hit attribution / self-hit test, scan-vs-PhysX comparison at one pose, then one Newton retrain and its sweep | 2 days + 1–2 GPU sessions (~1 h each) |
| PP/PN/NN/NP evaluation columns and report tooling (the sweep axes stay) | 1–2 days |
| kit-less CI lane + a ~3 GB image with the `uv` environment baked in (44–820 s of install per boot is the only recurring cost) | 2 days |

**≈ 2 weeks (10–12 person-days)** for a minimal, certified second backend on procedural terrain, including
one Newton retrain. That is the lane I would fund: it needs no scene, no camera work and no new task code.

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
* **The flat-ground height-scan value was not measured** (two of my own script bugs, then the pod could not be
  restarted); only the formula and its offset were verified, from the installed source.
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
  (`start pod: There are not enough free GPUs on the host machine`), so the last two checks (flat height scan,
  scene parse) could not be retried in-session.

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

**Hygiene:** a stop/patch response body echoes the pod's full `env` map, including the base64 bootstrap and
`RUNPOD_API_KEY`. Phase 1 printed one such body (truncated by the client); phase 2 never prints or stores a
response body — `spikes/newton/submit_run.py:safe_body` strips `env`/`dockerStartCmd`/`dockerEntrypoint` and
masks the rest through the repo's redaction helper, and `pod_guard.py` records only the HTTP status. Rotate
`RUNPOD_API_KEY` if the phase-1 output was captured anywhere.

## Artifacts

* Code and this report: branch `newton-spike`, commits `8c08b2b..HEAD` (local; not pushed — SSH auth is
  broken). Pod code travels as a git bundle.
* Phase 1 R2 prefix `cerberus-robotics/jobs/newton-20261115T035749Z/`:
  `outputs/{host.json,stage1/,stage2/,stage3/,stage4_none/,stage4_contract/,versions.json}`;
  `logs/{uv_sync,import_probe,stage1_train,stage1_launch_1..5,stage2_video,stage3_parity,stage4_none,
  stage4_contract,stage5_fetch}.log`; `rl_logs/` (the smoke checkpoints); `job.log`, `status.json`,
  `log_tail.txt`. Input side `spike/newton-20261115T035749Z/{code/cerberus-newton.bundle,
  inputs/checkpoint/model_1499.pt}`.
* Phase 2 R2 prefixes:
  * `cerberus-robotics/jobs/newton-20261115T153614Z/` — `outputs/{host.json,rest_pose_true,rest_pose_false,
    transfer_true,transfer_false,height_scan,scene_assets,solver_params.json}`,
    `logs/{rest_pose_true,rest_pose_false,transfer_true,transfer_false,height_scan,scene_sizes,
    scene_fetch_lite}.log`, `job.log`.
  * `cerberus-robotics/jobs/newton-20261115T160621Z/` — `outputs/height_scan/height_scan_source.txt` (the
    installed `mdp.height_scan` source, read on the pod) and `host.json`.
* Pods `rwmhwno9bflsbf`, `ofcqe4o85uqsjh`, `62f3leopdsfgu1` are all **stopped** (`EXITED`), not deleted: the
  owner deletes them. The phase-1 pod's volume still holds its partial scene fetch.
* Reproduce: `python3 spikes/newton/submit_run.py --dry-run`, then `--start-watch`; phase 2 uses
  `--bootstrap spikes/newton/newton_job2.sh --disk-gb 200 --wall-s 5400 --usd-cap 1.5`; results with
  `python3 spikes/newton/collect_results.py --job-id <job>`.
