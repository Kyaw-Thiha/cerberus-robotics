# Worlds and scene integration

How scenes are loaded, what makes a scene admissible, and which layer owns what. Companion to `docs/worlds.md`
(scene sources and the GRScenes data itself) and `docs/arena_integration_log.md` (the record of each gate as it ran).

## Layer

`common/cerberus_worlds/` is the world layer, with one backend per source behind a registry:

| backend | source | used for |
|---|---|---|
| `plain_usd` | GRScenes-100 (69 home, 30 commercial plain-USD scenes) | navigation; one shared scene copy works with 16+ envs |
| `arena` | Lightwheel RoboCasa/LIBERO kitchens through Isaac Lab Arena | manipulation and Mimic demos |

Consumers go through `registry.get_backend`; no module outside the world layer imports a backend directly. Wrappers
around Arena stay thin because Arena is alpha and its interfaces move.

Rules that came out of the first attempt at this layer (OmniGibson, removed):

- Do not patch Isaac Lab, Isaac Sim or Arena internals to make a scene load. If something only works with a patch, stop
  and record the failure instead: the six local fixes plus a device patcher behind OmniGibson were what made it
  CPU-only and unmaintainable.
- Versions and README tables go stale; verify against the installed stack, not the doc.
- Scenes are downloaded, preprocessed and read on the GPU host only. Nothing scene-sized lands on a laptop.
- GRScenes is centimetre-authored. Spawn poses, clearances and footprints are only meaningful in metres, so anything
  that samples them opens the metre wrapper (`describe_scene()["usd_path"]`, i.e. `scene_metric.usd`), never the raw
  scene USD. The backends already return it.

## Phase gates

The integration ran as gates, labelled W0-W4 to keep them distinct from the Cerberus phases 0-4:

| gate | content |
|---|---|
| W0 | remove OmniGibson, no new functionality |
| W0.5 | `cerberus_worlds` skeleton |
| W1 | Arena spike, the go/no-go |
| W1.5a | ARX5 assets, no training |
| W2a / W2b | fixed-base Mimic demo on a stock scene / Lightwheel kitchen |
| W3 | Isaac Sim and Isaac Lab upgrade |
| W4 | arm-loaded locomotion plus arm control |

A gate that fails is recorded in `docs/arena_integration_log.md`; it is not worked around in a way that changes scope.

Arm integration decisions that shape the world layer:

- **Arm role during locomotion training:** legs-only policy with the arm stowed or randomised as a load, arm driven by
  differential-IK task-space control. This matches Mimic's end-effector action requirement. A whole-body
  end-effector-tracking controller is the alternative, harder to train and to gate; revisit only if the first approach
  fails under arm motion.
- **Checkpoint lineage:** the arm adds about 3.35 kg to 15 kg and shifts the centre of mass. The legs-only checkpoint
  was measured to transfer on standing, velocity commands and 40-120 N push recovery with the arm stowed (fused 20-DoF,
  leg joints driven by name), but the adversarial gate still has to be re-run on the combined embodiment before it can
  be called robust. The evaluation harness needs `--fall_height_m 0.05`: the default 0.15 m fall rule misreads the
  policy's 0.11-0.17 m crouch.
- **Version coupling:** the reinforcement-learning API differs across Isaac Lab versions; train and deploy on the same
  pinned stack.
- **Camera conflict:** the arm occupies the Go2's back, where the camera-only ablation rig mounts its rear stereo pair.
  Resolve it (move the camera, mount it lower, or restrict that rig) before the arm is flown.

## Scene admission

Scene admission answers one question per scene: does this scene load, hold the robot and give the sensors something to
see? It exists because a mechanism that works on one scene is not evidence for 99 others, and because failures so far
were deterministic rather than random: Arena stalled in `PhysxManager` warmup on a large house with per-env scene
copies (17 minutes, 237 oversized-triangle warnings) while the same house loaded with one shared copy; a stock Arena
background let the Go2 fall through its floor.

`scripts/validate_scenes.py` implements it. It writes one manifest per source (`data/scenes/<source>/admission.json`),
one row per scene with `scene_id`, `backend`, `status` (`pass` / `fail` / `skip`), `reasons`, the measured metrics, the
tool versions and the date, and optionally mirrors it to the private `worlds/grscenes/` prefix on R2.

Checks per scene, physics first and rendering as a second optional pass:

1. Load on the intended backend at 4 envs, then once at 16 to catch scale problems, with a per-scene build timeout so
   a stall is recorded as `fail` rather than hanging the host.
2. Find a valid spawn: sample candidate positions on a grid over the footprint, cast a downward ray against the scene
   collision mesh, keep points with a floor hit and clear head and torso volume, and store the chosen spawn per scene
   in the manifest. The `(0,0,0)` default is not validated per scene; navigation reads spawns from the manifest.
3. Spawn the robot at that spawn and step zero actions for about 200 steps: observations finite, root height within a
   band of the floor hit, no fall-through, base tilt under threshold; record step time and peak GPU memory.
4. Cast one lidar sweep from the mounted height and require it to hit scene geometry. `mesh_prim_paths` is a static
   mesh list, so a scene that is not added to it returns "no hits" silently rather than failing — check it explicitly.
5. Optional render pass: one RGB frame non-empty, which catches missing MDL materials (`MDL_SYSTEM_PATH` has to be set
   before the app starts).
6. Record the shared-scene mode, and whether the Arena path also works with one shared copy and 1 env, which tests the
   per-env-cloning hypothesis above.

Navigation samples training and held-out scenes only from `pass` rows, uniformly per category; failed scenes stay listed
with their reasons. The held-out set is locked only once the manifest exists. The same script covers Lightwheel
kitchens by sweeping a stratified subset of the 10x10 layout/style grid rather than all 100.

Cost guide: about 30-40 minutes of host time per 10 scenes, plus the extraction of the scenes into the private cache.
