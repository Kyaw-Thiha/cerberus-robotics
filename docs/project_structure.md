# Cerberus — Project Structure

Reference doc for how the repo is organized and why. Mirrors the pattern used by
Artemis (self-driving perception stack) and Autoware Universe: top-level folders
are **stack components**, not code-type buckets. Repo root doubles as the colcon
workspace — no separate `ros2_ws/`.

## Guiding rules

1. **One directory per stack component**, matching the architecture diagram
   exactly (Perception, SLAM, Planning, Policy, Safety filter, Actuation).
   If the tree stops matching the diagram, the diagram is the source of truth —
   fix the tree, not the other way around.
2. **Research code and ROS2 nodes live side by side within the owning
   component**, not split into parallel `research/` and `ros2_ws/` trees.
   Isaac Lab training code (plain Python, no `package.xml`) sits next to the
   ROS2 package that consumes its output (a frozen checkpoint). colcon simply
   skips anything without a `package.xml`, so there's no build-system conflict.
3. **No `cerberus_` prefix by default.** Package names describe what they do
   (`terrain_perception`, `global_planner`), not what project they belong to —
   matches both Artemis and Autoware Universe convention. Prefix only:
   - `cerberus_msgs`, `cerberus_description` — generic names that would collide
     with other packages if unprefixed, and the kind of thing that might get
     reused standalone later.
   - `cerberus_bringup` — `xxx_bringup` is the actual ROS convention, kept for
     recognizability, not collision-avoidance.
4. **`eval/` is nested per module**, next to the code that produces the
   artifact, not centralized at root. Matches Artemis's `prediction/eval/`
   pattern. A module that both *trains something* and *evaluates something*
   (e.g. an adversarial robustness gate, which trains a small adversary policy
   as part of producing its eval artifact) gets its own subdirectory with its
   own nested `eval/`, sibling to — not nested under — the module's other
   `eval/` directories, since it's a materially different artifact produced by
   a materially different process (see `policy/locomotion/adversarial_robustness/`
   below).
5. **Split a component into multiple packages when the sub-parts are runtime
   peers** (e.g. `terrain_perception` vs `object_perception` — different
   nodes, different jobs). **Keep sub-parts nested under one package when
   one is hierarchically composed inside the other** (locomotion is a frozen
   low-level controller under the manipulation/navigation policy — not a
   peer, so it nests under `policy/`, not promoted to root).
6. **Embodiment identity is a config, never a hardcoded assumption**, so the
   quadruped and/or arm can be swapped later without restructuring this tree
   or rewriting `train.py`, the ROS2 inference nodes, or the safety filter.
   See "Embodiment configuration pattern" below for the concrete practices.

## Embodiment configuration pattern (for cross-platform modularity)

The current combo is Go2 (legs) + ARX5 (arm), but the codebase should not
assume this combo is permanent — retraining on a swap is expected and fine,
rewriting the codebase on a swap is not. Three practices make that true,
following the pattern LeggedManip_Lab already demonstrates across its 7
supported platform combos (one `train.py`, a `--task` string selects the
platform):

1. **Compose embodiment configs, don't write one monolithic robot config.**
   `common/cerberus_description/` defines separable `UNITREE_GO2_CFG` (legs)
   and `ARX5_CFG` (arm) objects and composes them into `GO2_ARX5_CFG`. A
   future arm swap edits only the arm sub-config; a future leg-platform swap
   (e.g. to ANYmal, the likely candidate if an inspection/intervention
   mission is added later) edits only the leg sub-config.
2. **Index joints by name, never by array position**, in both the RL
   observation/action space (`policy/locomotion/`, `policy/manipulation/`)
   and the ROS2 inference nodes (`policy/locomotion_policy/`,
   `policy/manipulation_policy/`). Isaac Lab's `ArticulationCfg` supports
   this natively — it costs nothing to do now and is a real refactor to
   retrofit later.
3. **ROS2 inference nodes stay generic**: `sensor_msgs/JointState` keyed by
   name, joint list loaded from a launch-time YAML/URDF param rather than
   hardcoded in node source. `safety_filter/core/`'s CBF constants (joint
   limits, self-collision geometry, torque limits) are likewise loaded from
   the embodiment config rather than inlined — the CBF/QP *formulation* code
   survives a swap even though its numeric parameters don't.

URDF/xacro is the canonical embodiment description, feeding both ROS and
sim: `common/cerberus_description/urdf/go2.urdf.xacro` (with its `sim`
argument) is merged with the arm by `scripts/build_go2_arx5_urdf.py` and
converted to USD with Isaac Lab `UrdfConverter`. See
`docs/embodiment_asset_composition.md`.

Checkpoint naming follows the same principle: prefix with the embodiment id
(`go2_arx5_...`, not just `go2_...`) once the arm is in the loop, so a future
combo's checkpoints don't collide with or shadow this one's (see "Checkpoint
versioning" below for the current, legs-only-so-far naming).

## Tree

```
cerberus/
├── pixi.toml                      # single root env: colcon, ruff, pytest, plotting libs
├── pixi.lock
├── colcon_defaults.yaml           # colcon build args/paths
├── build/ install/ log/           # colcon artifacts — gitignored
│
├── common/                        # shared, dependency-only, no standalone behavior
│   ├── cerberus_msgs/               # custom msg/srv/action defs
│   └── cerberus_description/        # UNITREE_GO2_CFG + ARX5_CFG, composed into
│       │                               # GO2_ARX5_CFG — see embodiment config pattern above
│       ├── embodiment.py              # EmbodimentCfg (lazy entry point, joint_groups by name)
│       │                                 # + compose_embodiments(); pure python
│       ├── urdf/go2.urdf.xacro        # Go2 description; `sim` arg (default false = ROS; true =
│       │                                 # root link `base`, no rotors/sensors/gazebo, for sim)
│       └── xacro_expand.py            # expands xacro without an ament index; pure python
│
│   ├── cerberus_worlds/             # world-backend abstraction — see docs/worlds.md
│   │   ├── base.py                    # WorldBackend: list_scenes, load_scene,
│   │   │                                 # register_embodiment, build_env
│   │   ├── registry.py                # get_backend(): explicit arg > WORLD_BACKEND env > error,
│   │   │                                 # no silent default; lazy "module:Class" registration
│   │   ├── admission.py               # scene admission manifest (pass/fail + validated spawn poses)
│   │   ├── arena_assets/              # out-of-tree Arena embodiments/backgrounds + scene discovery
│   │   └── backends/                    # never imported outside registry
│   │       ├── arena_backend.py         # Isaac Lab Arena (Lightwheel kitchens etc.)
│   │       └── plain_usd_backend.py     # generic USD scenes (GRScenes-100), configurable preprocess
│
│   ├── cerberus_compat/             # Isaac Lab 2.x <-> 3.0 adapters (quat, Warp data, rsl-rl, launcher);
│   │                                   # all Isaac-touching code goes through these
│
├── perception/
│   ├── terrain_perception/          # ROS2 pkg: terrain/obstacle perception for nav
│   └── object_perception/           # ROS2 pkg: object detection + 6D pose for grasp targets
│
├── slam/
│   └── slam/                        # ROS2 pkg — name placeholder, approach not yet decided
│
├── planning/
│   ├── global_planner/              # ROS2 pkg: A*/D*/RRT over the SLAM map
│   └── task_orchestrator/           # ROS2 pkg: state machine, fault recovery/retry
│
├── policy/
│   ├── locomotion/                  # Phase 0 (+ Phase 1 velocity-command composition)
│   │   ├── train.py / play.py         # Isaac Lab, plain python — NOT a colcon package
│   │   ├── eval/                      # push-recovery robustness curve (standard tests)
│   │   ├── adversarial_robustness/    # Phase 0.5 gate — sibling to eval/, not nested under it
│   │   │   ├── train_adversary.py       # trains adversary policy vs. frozen candidate checkpoint
│   │   │   ├── attack_env.py            # wrapper env: frozen locomotion policy in the loop
│   │   │   └── eval/                    # adversarial vulnerability report, robustify/reattack results
│   │   └── checkpoints/               # candidate → robustified → final, versioned
│   ├── manipulation/                # Phase 2, owns the Phase 4 training loop
│   │   ├── demo_collection/           # Isaac Lab Mimic
│   │   ├── algorithms/                # BC, Diffusion Policy, ACT
│   │   ├── continual_learning/        # Phase 4: naive fine-tune vs LoRA/adapter
│   │   ├── world_model/               # Phase 4: diffusion world model, policy refinement
│   │   └── eval/                      # demo-efficiency plot, forgetting curve, refinement success
│   ├── locomotion_policy/           # ROS2 pkg: inference node, frozen (final) checkpoint
│   └── manipulation_policy/         # ROS2 pkg: inference node, frozen checkpoint
│
├── safety_filter/
│   ├── core/                        # CBF/shielding math, plain python, pytest-testable
│   │   └── eval/                      # safety filter ablation table (filter on vs off);
│   │                                   # reuses policy/locomotion/adversarial_robustness/
│   │                                   # infra, retargeted at policy+filter, see Phase 3 notes
│   └── cerberus_safety_filter/      # ROS2 pkg: thin node wrapper around core/
│
├── actuation/
│   └── cerberus_actuation/          # ROS2 pkg: low-level joint cmd interface
│
├── system/                          # deployment composition, not a stack stage
│   ├── cerberus_bringup/              # launch files, composes the full graph
│   ├── scripts/
│   └── systemd/
│
├── docs/                             # architecture.svg, pipeline diagram, this file,
│                                       # isaac_lab_workflow.md, manipulator.md (VLM/VLA plan),
│                                       # worlds.md (scene sources/splits/backends),
│                                       # arena_integration_log.md (W0-W4 gate log)
├── docker/
│   ├── remote/                       # OLD stack image (Sim 5.1 / Lab 2.3.2 / Py 3.11): Dockerfile, start.sh
│   ├── remote_v6/                    # NEW stack image (Sim 6.2 / Lab v3.0.0 GA / Py 3.12 + Arena main + pixi sim-ros2)
│   └── slam/                         # one SLAM backend image per subdirectory
├── scripts/                          # cross-cutting ops tooling, not owned by one stack component
│   ├── sync_to_r2.py                   # permanent artifact storage — see docs/artifact_storage.md
│   ├── build_go2_arx5_urdf.py          # merge sim Go2 + patched X5A arm -> go2_arx5 URDF (offline)
│   ├── convert_go2_arx5_urdf_to_usd.py # UrdfConverter -> flattened USD + checks (pod)
│   ├── compare_go2_urdf_to_usd.py      # URDF vs USD mass/CoM/inertia/limits (offline)
│   ├── run_ga_smoke_pod.sh             # one-session GA-stack smoke: train, play+video, Arena spike
│   └── tests/                          # colocated tests for the scripts above
├── data/                             # gitignored — demos, trajectories
├── outputs/                          # gitignored — checkpoints, logs, videos, wandb runs
└── README.md
```

## Test layout

Tests live in a `tests/` subdirectory of the thing they test — one directory,
not a repo-wide `tests/` tree. `common/cerberus_worlds/backends/tests/`,
`slam/slam/eval/tests/`, `scripts/tests/`, and so on. ROS2/ament packages keep
`<pkg>/test/` instead, because colcon and `setup.py` already expect that name.

Rules that keep this working without `__init__.py` files or a plugin:

- **Test module basenames must be unique repo-wide.** `pytest`'s default
  `prepend` import mode keys modules by basename, so two `test_algorithms.py`
  files collide. This is why the per-package ones are named
  `test_<package>_algorithms.py`.
- **`pytest.ini` sets `pythonpath = . scripts`**, so `from common.x import y`
  (repo root) and `import build_lidar_proxy` (bare script imports) both resolve
  regardless of the current directory.
- **Moving a test one level deeper changes its `__file__` anchors.** Every
  `Path(__file__).parents[n]` and `Path(__file__).parent` used to reach repo
  root, the package, or a sibling module needs `n + 1` / `.parent.parent`.
- `policy/locomotion_policy/conftest.py` is intentionally empty: it exists only
  to put that package on `sys.path` for its `locomotion_policy.*` imports. Do
  not switch pytest to `--import-mode=importlib` without replacing it.

## Why `locomotion` and `manipulation` nest under `policy/` instead of root

The architecture diagram draws **one** box — "Policy layer: RL locomotion + IL
manipulation" — not two. Phase 1 explicitly composes the Phase 0(.5) locomotion
policy as a frozen low-level controller under a high-level policy that outputs
velocity commands; that's hierarchical composition, not two independent
systems. Promoting them to root-level siblings of `perception/`, `planning/`,
`safety_filter/` would break the one rule the whole tree is organized around:
component boundary = directory boundary, diagram = source of truth.

The Phase 4 training loop (continual learning + world model) nests under
`manipulation/` specifically, not generically under `policy/`, because the
project doc ties it to manipulation only — the world model trains on Phase 2's
collected manipulation trajectories, and continual learning is scoped to new
terrains/objects for that same policy. Locomotion has no training-loop
counterpart in the current doc, other than the adversarial robustness gate
described next.

## Why `adversarial_robustness/` lives under `policy/locomotion/`, and why it's not just another `eval/` entry

Phase 0.5's adversarial robustness gate is not a pure evaluation — it *trains*
a small adversary policy against the frozen-candidate locomotion checkpoint,
then evaluates against it, and may trigger a robustify/reattack loop that
retrains the locomotion policy itself. That's meaningfully more than what
lives in a bare `eval/` directory elsewhere in this tree (which just runs a
trained artifact and produces plots/tables). It's kept as a sibling directory
to `eval/`, nested under `locomotion/` rather than promoted anywhere else,
because:

- It is tightly coupled to `locomotion/`'s own frozen checkpoint and physics
  setup — the wrapper environment (`attack_env.py`) internally runs the
  frozen locomotion policy as part of its step function, so it has no
  meaningful existence independent of this module.
- It happens strictly *before* the Phase 1 freeze-and-compose step, which is
  the entire reason for its existence (see `cerberus_project_context.md`) —
  keeping it under `locomotion/` rather than under a new top-level directory
  keeps that "gate before freeze" relationship visible in the tree itself.
- Its infra is reused later, unmodified in kind, by `safety_filter/core/eval/`
  (same adversary-training method, retargeted at policy+filter instead of the
  bare policy) — noted at that location too, so the reuse relationship is
  discoverable from either side.

## Checkpoint versioning within `policy/locomotion/checkpoints/`

Introduced alongside the Phase 0.5 gate, since freezing now happens after a
possible robustify loop rather than immediately after Phase 0 training:

- `go2_locomotion_v1_candidate.pt` — output of Phase 0 training, not yet
  gated.
- `go2_locomotion_v1_robustified.pt` — only exists if the Phase 0.5 gate
  failed on the candidate and a robustify/reattack pass was run. Numbering
  increments (`_v2_candidate`, etc.) if more than one robustify pass is
  needed.
- `go2_locomotion_final.pt` — the checkpoint that passed the Phase 0.5 gate.
  This, and only this, is what Phase 1's ROS2 `locomotion_policy` inference
  node and any composed high-level policy are ever built against.

This naming is legs-only and stays that way through Phase 0.5, since the arm
isn't in the loop yet. Once Phase 2 attaches ARX5, the manipulation-phase
checkpoints under `policy/manipulation/` switch to an embodiment-prefixed
scheme (`go2_arx5_...`) per the "Embodiment configuration pattern" section
above, so a future arm or platform swap gets its own checkpoint lineage
rather than colliding with this one's.

### The `_ga_` tag (checkpoints retrained on Isaac Lab 3.0 GA)

`go2_locomotion_ga_v1_candidate.{pt,onnx,meta.json}` is a **stack-identified** checkpoint: the same Go2
rough recipe as RUN_002, retrained on Isaac Sim 6.2 / Isaac Lab 3.0 GA / rsl-rl 5.x with PhysX pinned
explicitly. It exists because a `.pt` records weights, not a physics engine — the September files were
produced on the retired 2.3.2 stack and are a different artifact class, not an older version of the
same thing. Rules:

- `_ga_` marks **which stack produced the weights**; `v1`/`v2` and `candidate`/`robustified`/`final`
  keep their Phase 0.5 meaning unchanged. Combined: `go2_locomotion_ga_v1_candidate` → gate →
  `go2_locomotion_final`.
- September files keep their names and bytes forever: they are the parity reference
  (`docs/reports/phase0_ga_parity.md`). Never rename, overwrite or convert them in place.
- The tag appears only where a stack distinction matters. The promoted checkpoint stays
  `go2_locomotion_final.pt` (no `_ga_`) because its sidecar records the stack and the gate verdict;
  Phase 1 freezes one artifact, and that artifact is identified by its sidecar, not its filename.
- Every `_ga_` checkpoint ships its `.meta.json` (`policy/locomotion/core/checkpoint_meta.py`): git
  hash, task id, resolved physics backend, iterations, seed, num_envs, config hash, wall clock, final
  reward, terrain-level stats. A checkpoint without a sidecar is not a candidate.

## Rule 7: world backends are pluggable

Scene source and sim-env composition sit behind `common/cerberus_worlds/`. Policy, perception
and safety code never import a backend directly, so a new scene library or env framework is a
new file in `backends/`, not a rewrite. Scene assets are fetched to gitignored `data/scenes/`.
Nobody imports `common.cerberus_worlds.backends.*` directly; everything goes through
`registry.get_backend`, and an AST-scan test enforces it. OmniGibson was evaluated and dropped
(see `docs/worlds.md`).

## Open item

`slam/slam/` keeps a placeholder name until a SLAM approach is actually
chosen — rename the package once it's decided, same way `edgegate_slam` would
have named itself, had that been the real answer (it isn't — SLAM tooling is
still genuinely undecided as of this doc).
