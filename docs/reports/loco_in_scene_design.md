# Locomotion in a scene: env composition, metrics and what is verified where

H4 Part 1 design note (2026-11-09). Companion to `docs/reports/scene_eval_phase0.md` (the H2 blocker) and
`docs/h4_report.md` (results). Scope: navigate A→B in a GRScenes home with **ground-truth pose and a
ground-truth-derived occupancy map**, with the frozen `go2_locomotion_ga_v1_candidate` policy executing velocity
commands. No onboard sensing is claimed.

## Env composition

`policy/locomotion/eval/scene_env.py` builds the task the policy was trained in and changes only what the scene
requires:

| piece | source | why |
|---|---|---|
| task | `Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0` | the Play cfg's observation group, noise-off settings, `JointPositionAction` (scale 0.25), decimation 4 at dt 0.005 (50 Hz) and PhysX are what the checkpoint expects; nothing is reshaped |
| terrain | `TerrainImporterCfg(terrain_type="usd", usd_path=<scene>)` | the preprocessed metre-scale scene USD replaces the generator terrain |
| height scanner | same `RayCasterCfg` (rate, pattern, offset, `ray_alignment: yaw`), target swapped to `/World/scan_proxy` | the raw house cannot be ray-cast (empty Mesh prims) and a downward ray would hit the ceiling; the proxy is one clean triangle mesh with ceilings removed (`scripts/build_scan_proxy.py`) |
| curriculum terrain terms | `None` | they index the generator terrain that no longer exists |
| command manager | `heading_command=False`, `resampling_time_range=(1e9, 1e9)`, `rel_standing_envs=0` | one command source only: the hook |
| reset events | pose/velocity/joint ranges zeroed | the driver teleports each env to its own start pose |

The scene is resolved through `common.cerberus_worlds.registry.get_backend("plain_usd")`; no policy or planning
module imports a backend directly. A `loco_velocity` variant inside `plain_usd_backend.build_env` was considered
and rejected: it would make the world layer import the locomotion task registry (policy code), and the composition
is policy-side by nature. The registry seam keeps the same decoupling.

## Ray-cast target

`scan_proxy.usd` is built by `scripts/build_scan_proxy.py` from the scene's world triangles, in metres. The filter
(`scene_geometry.filter_scan_triangles`) drops **large high horizontal slabs** (ceilings, roofs, upper floors) and
keeps everything else — floors, walls *including their top faces*, furniture. Keeping wall tops matters: a vertical
ray registers a wall only through a horizontal surface, so "drop everything above 0.6 m" would make walls
invisible and read the floor through them. Multi-level scenes therefore read through an upper floor slab to the
level below; that matches the single-floor occupancy assumption and is not a multi-floor claim.

## Command hook

`scene_env.VelocityCommandHook` writes `command_manager.get_term("base_velocity").vel_command_b[:, :3]` every step,
the same buffer `adversarial_robustness.attack_env.ManagerBasedHooks.apply_command` writes for x/y (confirmed in
H2), extended to the yaw rate. The observation is recomputed after the write (`env.get_observations()`) so the
policy sees the command on the step it is applied, not one step late.

## Per-env reset

`scene_env.reset_envs_to(env, poses)` resets the envs, then writes each robot's root pose `(x, y, yaw[, floor_z])`
and default joints to the sim, and recomputes the observation. `init_state` is one value per articulation cfg, so a
teleport is the only way to give each env its own start-goal pair. All envs share one scene prim, so spawns must be
at least `MIN_SPAWN_SEPARATION_M` (0.8 m) apart — checked before the physics step
(`scene_assertions.check_spawn_separation`); Isaac Lab's per-env collision filtering on a shared USD terrain is
unverified, which is why the default env count is small (4 validated spawn poses, not 16).

## Metrics

`common/cerberus_eval/nav_metrics.py`, from a per-step trace: outcome (`success` / `fall` / `nan` / `timeout` /
`planner_failure`), SPL (shortest / travelled, capped at 1, failures 0), time, path length, static-geometry contact
count, final distance, minimum `base_minus_foot_z` (standing check, threshold 0.15 m). A fall **after** the goal is
still a failure (the handoff's "no fall"), and `fell_after_success` keeps the fact visible. The trace is truncated
at the deciding step so trailing steps cannot inflate path length or contacts.

## Assertions (first sim step)

`common/cerberus_eval/scene_assertions.py`, written to `assertions.json`:

1. articulation joint order vs the contract **and** `platforms.yaml` — hard failure, because positional joint data
   would silently corrupt every run;
2. observation exactly 235-D and the contract term order, with the injected command found in the
   `velocity_commands` slice (offset 9–12; the full layout is 0–3, 3–6, 6–9, 9–12, 12–24, 24–36, 36–48, 48–235);
3. height-scan constant on the floor: **recorded, not enforced** — the measured mean, spread, clip saturation and
   which candidate conventions it matches; the contract stays `unverified` until the pod reports;
4. no NaN/inf in a 200-step zero-command stand, plus the standing check and a no-fall check.

## Verified locally vs on the pod

| item | locally (no GPU) | on the pod |
|---|---|---|
| scripted path generator, metrics, assertions, clip selection | unit tests (`common/cerberus_eval/tests/`) | used as-is |
| occupancy grid, A\*, pure pursuit | unit + synthetic-USD tests; kinematic closed loop | used as-is |
| scan proxy build | synthetic USD test (`pxr`) | built once per scene from the real house |
| env composition, hook, teleport, video | `--dry-run` only | R1 stand + R2 scripted smoke |
| scene frames, collision filtering, scan convention | not verifiable | R1 `assertions.json` |

## Proposal (not implemented): a collision-only lite scene USD

The real-house build is the session bottleneck: 745 s for 4 envs on `MV7J…ADQ8_usd`, and 12+ minutes observed on the
pod. Most of that is spawning the 20.7M-triangle render meshes per env; the scan proxy is a separate 13 s pass and the
articulation init is small. A lite USD that keeps only collision proxies (and drops render-only meshes) should cut the
build and the GPU memory, with clips rendered from the full USD in a separate visual pass. It is a proposal only: it
must first be shown that the collision geometry and the height-scan proxy behave identically (same contacts, same
187-cell scan on flat ground), otherwise the physics numbers stop being comparable. Not attempted in this session.
