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

## Real scenes moved to the pod (owner decision, 2026-11-09)

Scene assets are downloaded, preprocessed and read on the pod only; nothing scene-sized is ever fetched to this
machine, and the scene scripts refuse to run without `CERBERUS_ON_POD=1` (`scripts/pod_guard.py`).

The reason is measured, not precautionary: `scripts/fetch_grscenes_from_r2.py` returns `403 AccessDenied` on
`ListObjectsV2` with the credentials the repo's documented `.env` pattern loads, and the disk has **2.6 GB free**
against tens of GB per scene (H2 recorded ~25 GB for the first scene). So the grids, the real-grid smoke test and
the kinematic run on real scenes all happen in **R0c**, the first CPU stage of the pod session, and only the small
`.npz`/PNG outputs come back. **Every number in this report is synthetic**; there is no local real-scene result and
none is implied.
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
| R0 | bring-up, `outputs/env_report.json`, preflight, checkpoint | 5–10 |
| R0b | R2 credential check first; 403 stops the session before anything is downloaded | 2 |
| R0c | fetch + preprocess 2–3 scenes, build occupancy grids, kinematic run on real grids, Gate G (CPU, no simulator) | 15–25 |
| R1 | stand test, 1 scene, 4 envs, 200 zero-command steps, `assertions.json` | 10 |
| R2 | scripted-path smoke, short | 10 |
| R3 | nav eval, 2–3 scenes x 20 seeded pairs, 4 envs, 3 clips/scene | 20–40 |
| R4 | wrap-up, sync, report | 10 |

Total estimate ~70–105 minutes, inside the 3-hour cap; R0c is CPU work in the same session. R1 is the gate: if the joint order, the 235-D observation, the
height-scan constant or the standing check fail, Part 2 stops with the diagnosis instead of tuning.

## Deferred (unchanged from the handoff)

SLAM-derived pose/map (H4c), dynamic or unmapped obstacles, learned navigation policies, ROS wrappers for the
planners, SCAN-Planner/VOP-Nav/HiPAN/RAEM integration, estimated-map or corrupted-scan experiments, retraining or
robustifying any policy, `go2_locomotion_final.pt`, Arena/Lightwheel, arm or `go2_arx5`.

## Part 2: pod session, 2026-11-10

Ran on a fresh clone of the pushed `main` (`fb90e28`) on `cerberus-remote-v6:v6-ga` (A40 first, then a 4090 after the
first host never published port 22). Three hours were budgeted; the session stopped after ~1h45 of pod time.

### Credentials: no change was needed

R0b initially failed on a 403, but the probe was wrong: `worlds` is a **prefix inside the `cerberus-robotics`
bucket**, not a bucket name. The scenes are readable at `cerberus-robotics/worlds/grscenes/home`, and
`fetch_grscenes_from_r2.py` already builds that full path. Re-running the check against the real path passed
(listing 4 entries, object read 1). **No credential or bucket change is needed**; the earlier "R2 is 403" note in
this report was a probe bug, not a permission change.

### Real grids (R0c)

Both target homes fetched, preprocessed and rasterised. `planning/occupancy.py` needed two fixes that only real
scenes exposed:

1. the sampling pitch was derived from the scene's total triangle area; a ~500 m GRScenes backdrop plane dominated
   that area and coarsened the pitch past the 5 cm cell, leaving **71% of cells unknown**. The pitch is now fixed at
   2 cm.
2. a chunk of large triangles could multiply the per-triangle sample cap into billions of samples, which the
   container OOM-killed. A hard per-chunk sample budget now bounds it.

| scene | grid | free | blocked | unknown | traversable | largest component |
|---|---|---|---|---|---|---|
| `MV7J…ADQ8_usd` | 138 x 337 @ 5 cm (6.9 x 16.8 m) | 35% | 50% | 15% | 6 970 cells (15%) | 6 582 cells (94% of traversable) |
| `MV7J…ADY8_usd` | — | — | — | — | — | — |

Scene 1's preview shows rooms, walls, doorways and furniture, with unknown only outside the footprint. Gate G
passed on that scene.

**Scene 2 fails Gate G.** Its preview (`docs/reports/occupancy/MV7J…ADY8_usd_occupancy.png`) shows several rooms as
solid blocked rectangles, and the sampler rejected **36 of 79 attempts as unreachable (46%)** against 2 of 41 on
scene 1. The navigable space is real but fragmented; the likely cause is a large up-facing surface inside the
0.12–0.45 m band (a raised floor or platform) being marked blocked. Do not run physics on this scene until a per-cell
reason dump explains it — this is a diagnosis, not a tuning change.

### Kinematic upper bound on the real grids

| scene | pairs | success | mean path | SPL | rejections |
|---|---|---|---|---|---|
| `MV7J…ADQ8_usd` | 20 | **20/20** | 5.99 m | 1.0 (capped) | 39 attempts: 2 unreachable, 17 outside 3–10 m, 2 too close |
| `MV7J…ADY8_usd` | 20 | **20/20** | 4.44 m | 1.0 (capped) | — |

As before, SPL 1.0 is a cap artefact: episodes stop at the 0.5 m goal tolerance, so the travelled length is shorter
than the reference. The mean path length is the number to compare with physics.

### Physics runs: blocked by the Kit launcher

Every launch of the scene env crashed at app start with `libcarb.tasking` SIGSEGV (exit 139), the driver-595 flake
recorded for this image. Three launches were tried (the documented one-retry allowance plus one): the 4-env stand
crashed immediately, a second 4-env run survived 18+ minutes without ever writing `assertions.json` (this scene's
recorded 4-env build is ~745 s, so it was still building when cut), and a 1-env navigation launch crashed at start.

Consequences, stated plainly:

- **No physics navigation numbers, no assertions, no clips.** The R2 prefix holds the grids, the kinematic metrics
  and the branch bundle; `assertions/` and `*.mp4` do not exist because no run got that far.
- **The kinematic-vs-physics gap cannot be reported this session.** The kinematic ceiling is 100% on both scenes;
  whether the policy tracks it is untested here. Nothing was tuned.
- The session did **not** reach R1's gate, so R2/R3 were never validly attempted.

### Delivered

- R2: `cerberus-robotics/migration/h4_nav/20261110T0857Z/` — 53 files: both grids (`occupancy/*/occupancy.npz`),
  scan-proxy sidecars, both occupancy PNGs, per-pair kinematic JSON + aggregate CSV for both scenes,
  `h4-nav-in-scene.bundle`. Manifest check: every local file present, `missing: none`.
- Branch `h4-nav-in-scene` on the pod at commit `4c7ff31` (occupancy fixes + `docs/reports/occupancy/`), exported
  as that bundle because the pod has no GitHub write credential. From a clone of `main`: `git fetch
  <bundle>` then push the branch.
- Pod `6vbmvper6uwjry` **stopped** (`EXITED`), not deleted: no artifacts remain only on it.

### Next session needs

A host where Kit launches. The provisioner is not the problem (one 4090 host worked, a second 4090 host never
published port 22); the failing part is Isaac Sim's startup on driver 595. Try a different GPU/datacenter, or the
`v6-dev`/`v6-rel2` image whose driver line was previously stable for app launches, and only then run R1 -> R2 -> R3.
