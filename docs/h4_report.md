# H4 report — locomotion + navigation in GRScenes (interim: Part 1)

Status: **Part 1 (local) implemented and tested; Part 2 (pod) not run.** No navigation number in this document is a
physics result. Ground-truth pose and map; the height scan is simulator-provided. This is **not** onboard-sensing
navigation.

## What exists now

| package | what | where |
|---|---|---|
| A | scene env composition (rough Play cfg + scene USD + scan proxy), command hook, per-env teleport, scripted path, metrics, assertions, clips | `policy/locomotion/eval/scene_env.py`, `scripts/scene_eval.py`, `common/cerberus_eval/` |
| B | occupancy grid from the scene's static meshes (+ inflation), config, PNG preview, CLI | `planning/occupancy.py`, `planning/config/occupancy.yaml`, `scripts/build_occupancy.py` |
| C | `PoseSource` / `MapSource` seams, A* (registered `astar`), seeded pair sampler, previews | `planning/sources.py`, `planning/global_planner/core/astar.py`, `planning/sampling.py`, `planning/viz.py` |
| D | pure pursuit (registered `pure_pursuit`) + closed-loop unicycle test | `planning/local_planner/core/pure_pursuit.py`, `planning/kinematic_sim.py`, `scripts/kinematic_eval.py` |
| E | navigation eval driver (Isaac backend + local mock backend) | `scripts/nav_eval.py`, `common/cerberus_eval/nav_loop.py` |

Design note: `docs/reports/loco_in_scene_design.md`. Run-artifact rule: `docs/runpod.md` ("Run artifacts").

## Local verification

- `pixi run pytest -q` passes, including 130+ new tests: scripted paths, metrics (including "success after a fall is
  a fall" and "truncated trace"), assertions with mocked joint lists, clip selection, occupancy band/step/inflation,
  A* optimality/unreachable/corner-cutting, sampler determinism, pure pursuit limits, kinematic episodes, the
  navigation loop, and the "success through geometry" guard.
- `scripts/preflight_local.sh --skip-install`: all checks PASS (shell syntax, repo tests, H4 imports, scene-env
  compile, `scene_eval --dry-run`, `build_scan_proxy --help`, R2 credential presence).
- `scripts/scene_eval.py --dry-run --num_envs 2` resolves the scene and validated spawn poses without Isaac.
- Previews committed: `docs/reports/figures/h4_occupancy_synthetic.png`, `h4_paths_synthetic.png`,
  `h4_path_single_synthetic.png`, plus `h4_plan_preview_summary.json`.

## Kinematic upper bound (synthetic flat, not a GRScenes home)

`scripts/kinematic_eval.py --pairs 20 --seed 0`:

- grid: 163 x 243 cells at 5 cm, free 95.7%, inflated 32.3%;
- sampling: 20 pairs from 30 attempts, 10 rejected on the 3–10 m geodesic window, 0 unreachable;
- **20/20 success, SPL 1.0, 0 collisions**, mean travelled length 5.53 m, mean episode 7.8 s.

Read that as the ceiling, not a result: the unicycle has no dynamics, no slip and no fall, and SPL is capped at 1.0
because the episode stops at the 0.5 m goal tolerance (travelled length is *shorter* than the reference). The
`margin_steps` counter (time inside the inflation band but not inside geometry) is recorded per episode; the
aggregate above shows 0 real collisions.

Two planner defects were found and fixed while producing these numbers, both caught by inspection of the committed
preview rather than by a metric:

1. a 0.1 m wall at 5 cm cells left an unblocked 1-cell channel between its two face rows, and A* threaded through
   it; blocked cells are now thickened by one cell (`_thicken`);
2. the surface sampler capped each triangle at 4096 samples, so a 12 x 8 m floor slab left ~90% of its cells
   unsampled and read as UNKNOWN; sampling now scales the pitch globally to the scene's total area, with a
   neighbour-fill for residual holes.

The preview is the check that matters here: a success rate alone would have hidden both.

## Blocked locally (needs the owner)

1. **Scene fetch.** `scripts/fetch_grscenes_from_r2.py` fails with `403 AccessDenied` on `ListObjectsV2` using the
   credentials the repo's own documented pattern loads from `.env` (preflight's credential check passes because it
   only verifies presence). Separately, the disk has **3.3 GB free**; one scene's shared `Materials`/`models` are
   tens of GB (H2 recorded ~25 GB for the first scene). So B2/B3's real-scene grids and B5's real-scene smoke test
   are **not done**, and every number above is synthetic.
2. **Git remote.** `git fetch` fails with `Permission denied (publickey)` (no SSH identity loaded in this session).
   All commits are local; the Gate L push is pending.

## Deviations from the handoff (each deliberate)

- **B1:** `usd-core` is already a default `pypi-dependencies` entry (`pixi.toml`, used by the existing
  `scene_geometry`/`build_lidar_proxy` tests), so no optional `geometry` feature was added and nothing was re-locked.
  Moving it would have removed `pxr` from the environment the existing tests use.
- **R1 env count:** the driver defaults to **4 envs**, not 16. All envs share one scene prim and per-env collision
  filtering on a shared USD terrain is unverified; the admission manifest has 5 validated spawn poses per scene, and
  spawns closer than 0.8 m are refused. `--num_envs` still accepts 16, but only with 16 validated, separated poses.
- **`loco_velocity` backend variant:** not added. Composing the rough Play task inside `plain_usd_backend.build_env`
  would make the world layer import the locomotion task registry (policy code); the composition lives in
  `policy/locomotion/eval/scene_env.py` and reaches the scene through `cerberus_worlds.registry`. The decoupling
  rule ("policy/planning never import a backend directly") holds.
- **Inflation radius:** 0.25 m, not 0.35 m. 0.35 m is half the body length but closes a 0.9 m doorway to a 0.2 m
  corridor, making ordinary homes unnavigable for the *planner*; 0.25 m covers the half-width plus margin. The
  tradeoff is in `planning/config/occupancy.yaml`.

## What Part 2 will run, and the expected cost

Per `docs/runpod.md` and `docker/remote_v6`, on a short-lived `h4-nav-in-scene` branch from `origin/main`:

| step | what | GPU minutes (estimate) |
|---|---|---|
| R0 | bring-up, `outputs/env_report.json`, preflight, checkpoint + scenes + grids | 10–15 |
| R1 | stand test, 1 scene, 4 envs, 200 zero-command steps, `assertions.json` | 10 |
| R2 | scripted-path smoke, short | 10 |
| R3 | nav eval, 2–3 scenes x 20 seeded pairs, 4 envs, 3 clips/scene | 20–40 |
| R4 | wrap-up, sync, report | 10 |

Total estimate ~60–85 minutes, inside the 3-hour cap. R1 is the gate: if the joint order, the 235-D observation, the
height-scan constant or the standing check fail, Part 2 stops with the diagnosis instead of tuning.

## Deferred (unchanged from the handoff)

SLAM-derived pose/map (H4c), dynamic or unmapped obstacles, learned navigation policies, ROS wrappers for the
planners, SCAN-Planner/VOP-Nav/HiPAN/RAEM integration, estimated-map or corrupted-scan experiments, retraining or
robustifying any policy, `go2_locomotion_final.pt`, Arena/Lightwheel, arm or `go2_arx5`.
