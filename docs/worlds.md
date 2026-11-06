# Cerberus — Worlds Reference

Short reference for where scenes come from, how they're split, and how backends plug in.
Complements `cerberus_project_context.md` (Mission domain decision). Implementation detail and
phase gates are in `arena_integration_log.md`. Replaces the dropped OmniGibson integration.

## Why this exists

The OmniGibson integration only worked on CPU physics in our Isaac Lab setup, so it was
dropped (see git history). Scene sources now sit behind `common/cerberus_worlds/`, so a future user can add a
sim env or scene library without touching policy, perception or safety code.

## Backend interface

`WorldBackend` (`common/cerberus_worlds/base.py`) is an ABC with four abstract methods:
`list_scenes() -> list[SceneInfo]`, `load_scene(scene_id) -> Any`,
`register_embodiment(cfg: EmbodimentCfg) -> None`,
`build_env(scene, embodiment, task, num_envs) -> Any`.
`SceneInfo` is a frozen dataclass: `scene_id`, `category=""`, `source=""`, `metadata={}`.

Embodiments are described by `EmbodimentCfg` in `common/cerberus_description/embodiment.py`
(pure Python, no isaaclab imports at module level): `name`, `articulation_cfg_entry_point`
(lazy `"module:ATTR"` string), `joint_groups` (group name -> joint names, addressed by name),
`checkpoint_prefix` (defaults to `name`), `joint_names` property, and `resolve()` which
lazily imports the ArticulationCfg. `compose_embodiments(name, legs, arm,
articulation_cfg_entry_point)` merges a legs and an arm embodiment (e.g. `go2_arx5`) and raises
`ValueError` on any joint-name collision.

Backend selection (`common/cerberus_worlds/registry.py`): `get_backend(name=None)` resolves
explicit argument > `WORLD_BACKEND` env var (`arena` or `plain_usd`) > `ValueError` listing the
available backends. There is no silent default. `register_backend(name, factory)` takes a
lazy `"module:Class"` string or a callable (duplicate name -> `ValueError`); the built-ins
`arena` and `plain_usd` are registered lazily by string. Instances are cached per name;
`available_backends()` returns the sorted names.

Rule: policy, perception, safety and all other packages never import
`common.cerberus_worlds.backends.*` directly; they go through `registry.get_backend`. An
AST-scan test enforces this. Isaaclab/arena imports live inside method bodies.

Status (2026-10-05, overnight run): both backends are implemented
(`common/cerberus_worlds/backends/arena_backend.py`, `plain_usd_backend.py`). Arena embodiments live in
`common/cerberus_worlds/arena_assets/go2_embodiment.py` (legs-only Go2) and `go2_arx5_embodiment.py`
(fixed-base Go2+ARX5 with a gripper). Nothing is committed yet; results are in
`data/reports/overnight/NOTES.md`.

| Backend | Wraps | Use |
|---|---|---|
| `arena_backend` | Isaac Lab Arena (Scene + Embodiment + Task via `ArenaEnvBuilder`) | Lightwheel kitchens, manipulation, Mimic demos. Works for Go2 and fixed-base Go2+ARX5. Does not work (stalls) on the multi-room GRScenes scene |
| `plain_usd_backend` | Any plain USD scene, configurable preprocess step | GRScenes-100 homes/commercial, navigation. The documented fallback when Arena stalls |

## Scene sources

| Source | Content | License | Role |
|---|---|---|---|
| GRScenes-100 | 69 home + 30 commercial plain-USD scenes | CC BY-NC-SA 4.0 + access gate | Phase 1 navigation, secondary missions |
| Lightwheel RoboCasa/LIBERO via Arena | 100 kitchen scenes (10 layouts × 10 styles) | Apache-2.0 | Phase 2 manipulation |
| InternScenes | ~40k scenes, Real2Sim only released, Sim 4.1 | CC BY-NC-SA 4.0 | Watch item, not planned |

Measured results (Lab v3.0.0-EA and the pinned Sim 6.1 stack, 16 envs unless noted):

- GRScenes via `plain_usd`: Gate A passes, 16 envs x 200 steps, 26.9 s, final robot z 0.253 m (scene `MVUHLWYKTKJ5EAABAAAAACY8_usd`).
- GRScenes via Arena: Gate A FAILED. It stalled about 17 min inside PhysxManager warmup (`ArenaEnvBuilder.make_registered`)
  with 237 `TriangleMesh: triangles are too big` warnings. Arena is not viable for this multi-room scene; use `plain_usd`.
- Arena stock scene (Galileo background): Gate A exit 0 on the EA stack, but the robot fell through the floor
  (root z -0.73 to -0.49 m), so that pass says nothing about stock-scene physics. On the pinned Sim 6.1 stack Gate A stock passed
  16x200, 40.9 ms/step (no fall-through check recorded in NOTES.md; unverified).
- Lightwheel kitchens via Arena: Gate B passes. Pinned stack 1:1 at 98 ms/step (57.19 MB SDK download), 6:8 at 102 ms/step
  (112.19 MB); EA 1:1 in 72.8 s. The Lightwheel SDK works anonymously, no account needed for the tested assets.
- Fixed-base Go2+ARX5 in a Lightwheel kitchen (Arena mimic smoke): 16 envs pass, arm drift 7.6e-5 rad (gravity disabled on the fixed-base robot).

GRScenes data in R2: private prefix `worlds/grscenes/home/` holds scene `MVUHLWYKTKJ5EAABAAAAACY8_usd`
(HF revision `4760b9031881c985e3582075cb2d8dbde1530a68`, 1052 objects, 1,406,493,050 B, manifest sha256
`77876dc639285dd77d7129dd2b94a8c41af73e851f69eaa473abafe03adc1357`, 0 unresolved USD dependencies). Consumer:
`scripts/fetch_grscenes_from_r2.py` (links the scene and writes an `env.sh` with `MDL_SYSTEM_PATH`); fetch took
1.31 GiB in 48.6 s on an EA pod. Only the home zip was extracted so far, one scene validated. Extraction ran on an A5000 GPU pod
(7z reads the split zip directly, 170,029 members); see `docs/runpod.md`.

License caveats: GRScenes is CC BY-NC-SA 4.0 and the R2 copy must stay private (no redistribution). The ARX official
model (`ARXroboticsX/ARX_Model` X5/X5A) has no license file: do not redistribute, keep gitignored at `third_party/arx-model`.

**GRScenes specifics found by running (2026-10-06, see `arena_integration_log.md`):**

- Scene stages are **centimetres** (`meters_per_unit = 0.01`), Z-up. Anything built alongside a scene (e.g. the
  LiDAR proxy) must declare the same units and write source-unit coordinates; Isaac scales the reference itself.
- 1115 of one house's 3230 `Mesh` prims are empty, which kills Isaac Lab's `MultiMeshRayCaster` if the raw scene
  USD is a ray-cast target. A LiDAR rig therefore needs a cached per-scene `lidar_proxy.usd` next to the scene USD,
  built by `scripts/build_lidar_proxy.py` (valid geometry only, ~5 s, 133 MB for a 1.4 GB house). Runs built on the
  `plain_usd` backend do this automatically when the rig has a LiDAR.
- Robot spawn poses are per-scene data, not a constant: they come from `data/scenes/<source>/admission.json`
  (`common/cerberus_worlds/admission.py`), written by `scripts/validate_scenes.py`. A scene run without a manifest
  entry needs an explicit `--spawn x,y,yaw,floor_z`; there is no silent (0,0,0) fallback.
- The simulated-sensor path (walking policy + LiDAR/IMU/cameras -> ROS2 bag) passed end to end in a home scene on
  the pinned Lab 3.0 stack (2026-10-06 smoke: 1050/1050 messages on pose, TF, IMU and both cameras; 202 lidar
  points, 10 Hz; bag 1.83 GB).

- The dataset's material MDLs import a module library the dataset ships but the *extractor* used to skip:
  `Materials/KooPbr.mdl` and `Materials/KooPbr_maps.mdl` are referenced only by other `.mdl` files (`import
  ::KooPbr::KooMtl;`), never by a USD or `metadata.json`, so the USD-reference walk never asked for them. 280 of the
  dataset's 299 `.mdl` files import `::KooPbr`. A missing module makes every such material fail to compile
  (`MDLC comp error ... C120`), which the RTX renderer retries on every launch -- and a failed compile is never cached,
  so each launch paid the full cost again (10+ min per scene on a 2026-10-11 pod). Fixed 2026-10-11: the extraction
  closure follows MDL imports too (`mdl_module_imports()` / `module_member()` in `scripts/fetch_grscenes_to_r2.py`).
  `KooPbr` is NOT in Isaac Sim; do not look for it there. Camera-based SLAM backends evaluate against these materials,
  so a tree extracted before that fix renders with a partial material set.
- Mixing scene sources on one pod used to leave assets unlinked: `merge_into_tree()` linked only top-level entries, so
  when the destination already had a `Materials/` (e.g. from an earlier single-scene R2 pull) the later Hugging Face
  extraction was skipped wholesale -- 514 of 815 material files invisible on 2026-10-11. Fixed the same day by merging
  shared directories file by file (`link_missing_files()`), so the destination is a complete union of every source.

- GRScenes also ships **interaction data we do not use** (measured 2026-10-11 over all 69 homes): every home has
  `start_result_interaction.usd` and `interactive_obj_list.json` with `articulated` (32-172 per scene, median 55:
  cabinets, doors, windows, faucets, ...) and `pickable` (2-89 per scene, median 9, never 0: bottle 266, cup 254,
  plate 241, pan 102, pot 87, lamp 73, bowl 56, toy 45, tray 36, ... across the dataset; no mail/letters). The
  interaction USD authors `PhysicsRigidBodyAPI` on 115 prims in the one scene inspected (the navigation USD: 0), and
  every pickable prim has a rigid-body descendant. Articulated model assets carry joints (357 of 400 sampled) but no
  `ArticulationRootAPI` as shipped. Our `plain_usd` backend loads `start_result_navigation.usd` as static terrain
  (`TerrainImporterCfg(terrain_type="usd")`), so none of this is usable today; the rigid bodies are nested under static
  `Instance/Group_*` prims, which is not an Isaac Lab `RigidObject` (Mimic needs those). NOT verified: that the
  interaction scene behaves in a run, or that a standalone `models/object/others/<kind>/<hash>/instance.usd` spawns
  as a `RigidObject` the ARX5 can grasp (W2b: irregular objects against an 88 mm jaw failed).
- **Smallest-first selection is biased on the manipulation axis, not on size.** Navigation-USD size spans only
  1.107-1.595 MB over the 69 homes (a weak complexity proxy), but the 18 smallest homes have 2-19 pickable objects
  (median 5) against a median of 15 (max 89) for the other 51. Choose later batches stratified, not smallest-first.
- **ID prefixes are NOT layout families (answered 2026-10-12).** Six prefix groups (18/18/15/9/6/3), but composition
  similarity -- the counts of model kinds a scene references -- is the same within a prefix as across prefixes
  (mean 0.868 vs 0.872 over 465 / 1881 pairs), and wall-instance counts vary wildly inside one family
  (`MVUCSQAKTK` 104-381, `MWAX5JYKTK` 125-545). A shared floor plan would cluster tightly. So a scene-level holdout
  does not leak layouts through the id, and no prefix group needs holding out wholesale. (Asset-overlap could not
  answer this: reuse is ~1.00, 51,565 distinct of 51,775 references, and no wall/floor asset is shared.)
- **Free complexity and manipulability proxies for all 69 homes** (no extra fetch; from each scene's `metadata.json`
  and `interactive_obj_list.json`, which are 2 KB each): wall-instance count spans **104-545** (median 206) and
  pickable-object count spans **2-89** (median 9), and the two are **uncorrelated (r = 0.054)**. So they are separate
  axes: a batch chosen for one says nothing about the other, and a stratified selection has to stratify on both.
  Full per-scene table: `data/reports/scene_admission/scene_pool.csv` (69 rows: walls, layout/object asset counts,
  pickable, articulated, nav-USD bytes). Navigation-USD byte size, which the current smallest-first selection uses,
  is a weak proxy (1.107-1.595 MB, a 44% spread).

GRScenes has hashed IDs and only home vs commercial labels. Retail/restaurant/office need manual
classification of the 30 commercial scenes before stratified sampling is possible.

## Split and sampling

- One policy across all categories, not one per category.
- Scene-level holdout: hold out whole scenes, never parts of a scene seen in training.
- Sample category uniformly, then scene uniformly within it, so the largest category doesn't
  dominate gradients. Add DRATS-style adaptive upweighting only if proficiency is uneven.
- Held-out scenes are not locked yet. Decide when the Phase 1 scene set is final.

## Remote workflow

Scene assets are fetched to gitignored `data/scenes/` (from R2 with `scripts/fetch_grscenes_from_r2.py`; correction 2026-10-05: no
network volume is used, and the old OmniGibson volume text no longer applies), not into git. Cerberus code stays outside the image and is git-synced per pod, as in
`isaac_lab_workflow.md`. GRScenes needs the `MDL_SYSTEM_PATH` env var and its own preprocess
step (`preprocess.py -n` for navigation, `-i` for interaction).

## Open items

Answered (2026-10-05):
- Arena's Scene abstraction on a full multi-room GRScenes home: no. It stalled in PhysX warmup on the one scene tried; use `plain_usd`.
  Other GRScenes scenes were not tried.
- GRScenes tools on Sim 5.1/6.x: the scene USDs load on Sim 6.1 (EA and pinned) through `plain_usd`. The GRScenes preprocess tools
  themselves (`preprocess.py`) were not run; unverified.

Still open:
- GRScenes access-gate approval process and timing (the home zip was obtained, but the gate process itself is not documented here).
- Scene admission: **18 of 69 home scenes have a `pass` row, 0 failures** (8 on 2026-10-07; a 10-scene batch finished
  2026-10-12T01:35Z, where one scene crashed on its first attempt and passed on one retry -- a flake, see the log).
  The remaining 51 homes and all 30 commercial scenes are unswept. Measured cost on an A5000: **10-26 min per scene**,
  almost all of it `build_env` (143-885 s per launch on the same host, three launches per scene), so this is
  host/scene-setup bound, not a defect. Manifest + scenes are mirrored to R2 (`worlds/grscenes/`).
- Commercial-scene classification (retail/restaurant/office) is still not done.
- Held-out scene split is not locked.
