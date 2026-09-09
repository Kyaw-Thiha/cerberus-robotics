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
│                                       # GO2_ARX5_CFG — see embodiment config pattern above
│
├── worlds/                        # mission environment scenes — pulled from external
│   │                                 # libraries, not authored here; gitignored like data/
│   ├── household/                   # BEHAVIOR-1K / OmniGibson scene USDs — Phase 1/2 primary
│   ├── retail_restaurant_office/    # same OmniGibson/BEHAVIOR-1K asset family — secondary
│   │                                 # missions, near-zero marginal cost over household
│   └── fetch_scenes.py                # pulls/caches scene USDs; no scene binaries committed
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
│   │   └── checkpoints/               # candidate → robustified → final, versioned (see Phase 0.5 handoff)
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
│                                       # isaac_lab_workflow.md, manipulator.md (VLM/VLA plan)
├── docker/
│   └── Dockerfile.remote             # Isaac Lab base image + pinned extras, built on rented GPU box
├── scripts/                          # cross-cutting ops tooling, not owned by one stack component
│   └── sync_to_r2.py                   # permanent artifact storage — see docs/artifact_storage.md
├── data/                             # gitignored — demos, trajectories
├── outputs/                          # gitignored — checkpoints, logs, videos, wandb runs
└── README.md
```

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

## Open item

`slam/slam/` keeps a placeholder name until a SLAM approach is actually
chosen — rename the package once it's decided, same way `edgegate_slam` would
have named itself, had that been the real answer (it isn't — SLAM tooling is
still genuinely undecided as of this doc).
