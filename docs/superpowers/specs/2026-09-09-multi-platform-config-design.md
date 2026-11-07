> **Shorter copy.** The longer version of this document is `docs/archive/superpowers_plans/2026-09-09-multi-platform-config.md`.

# Multi-Platform Config Design

Date: 2026-09-09

## Problem

Two related but separately-buildable gaps in how "which robot platform" gets
selected across the codebase, found while extending the platform-decoupling
work from `2026-09-08-platform-decoupling-design.md`:

**Part A — Isaac Lab training side.** `policy/locomotion/__init__.py`
currently imports each platform's `platform.py` with one hand-written line
per platform. Real precedent (LeggedManip_Lab, the project this repo's docs
already cite) instead auto-discovers every platform package via a filesystem
walk, so adding a platform touches zero existing files. Buildable now —
`core/platforms/` and its one `go2` entry already exist.

**Part B — cross-language platform selection.** `docs/project_structure.md`'s
"Embodiment configuration pattern" already states the right principle
("joint list loaded from a launch-time YAML/URDF param rather than hardcoded
in node source") but doesn't specify where that YAML lives, its schema, or
how checkpoint selection threads through it. Research this session
(Autoware's `vehicle_model` launch arg, Clearpath's central `robot.yaml`,
and the hard constraint that C++ ROS2 nodes cannot import Python objects)
found the previously-drafted plan of ROS2 nodes importing
`common/cerberus_description/` Python objects directly doesn't work as
stated.

**Correction from this spec's first draft**: `ros2`/`colcon` ARE available
locally — `pixi.toml` already declares `ros-jazzy-ros-base` +
`colcon-common-extensions`, installed under `.pixi/envs/default/` and usable
via `pixi run ros2 ...`/`pixi run colcon ...`. Only checked with a bare
`which ros2` the first time, which missed pixi's environment. This changes
scope: a **Go2-only slice of Part B is buildable and testable now**, not
deferred — no arm merge needed (Go2 alone doesn't need
`docs/embodiment_asset_composition.md`'s Robot Assembler step, that's only
for combining two robots), and a real `locomotion_policy` ROS2 node is
already-flagged outstanding work (`RUN_002.md`'s recommendations). The
**`go2_arx5` combo case stays deferred** to Phase 2 — it genuinely needs the
asset merge and an ARX5 asset that doesn't exist yet.

## Part A: auto-discovery for `core/platforms/`

### Current state

```python
# policy/locomotion/__init__.py
from policy.locomotion.core.platforms.go2 import platform as _go2_platform  # noqa: F401
```

One import line per platform, in a file every new platform's author has to
remember to edit.

### Target state

```python
"""Cerberus locomotion task registration. Importing this module auto-discovers
and registers every platform package under core/platforms/ -- add a platform
by adding a new core/platforms/<name>/ directory (see
core/platforms/registry.py for what it must define), never by editing this
file.
"""

from isaaclab_tasks.utils import import_packages

# "registry" holds the shared PlatformCfg contract itself, not a platform --
# skip it so import_packages doesn't try to treat it as one.
_BLACKLIST_PKGS = ["registry"]

import_packages(__name__ + ".core.platforms", _BLACKLIST_PKGS)
```

`isaaclab_tasks.utils.import_packages(package_name, blacklist_pkgs)` recursively
walks and imports every subpackage under the given package — confirmed this
session by reading LeggedManip_Lab's actual `tasks/__init__.py`, which uses
this exact call for the same purpose. Since it's already a dependency
(`isaaclab_tasks` is imported elsewhere in this codebase), this adds no new
dependency.

Each platform's `platform.py` already calls `register_platform(GO2)` at
import time (built in the prior session) — the walk just needs to *import*
that module; the registration side effect already exists and needs no change.

### Verification

`import_packages` needs `isaaclab_tasks` (not installed locally, same
constraint as the rest of `policy/locomotion/`). Locally verifiable only via
`py_compile` (syntax). Functionally verifiable on a pod: import
`policy.locomotion`, confirm `available_platforms()` still returns `["go2"]`
and all four gym ids still register — same check the prior plan's Task 11
already documents as a pod-side follow-up, extended to cover this change too.

## Part B: cross-language platform selection

### Checkpoint status (confirmed on R2 this session)

Two trained Go2 locomotion checkpoints exist, corresponding to the two
runtime "shapes" described in `docs/locomotion_architecture.md`:

- **Flat** (`unitree_go2_flat`, run `2026-09-03_04-24-43`, full 1500
  iterations) — Shape A (proprioception only, no exteroception). **Already
  has `exported/policy.onnx` and `exported/policy.pt`** from a prior
  `play.py` invocation — no pod run needed, just a download.
- **Rough** (`go2_locomotion_v1_candidate.pt`, already local under
  `policy/locomotion/checkpoints/`) — Shape B (proprioception +
  `height_scan`). No exported ONNX/JIT exists yet for this one — needs a
  short pod run (`play.py`'s existing `export_policy_as_onnx`/
  `export_policy_as_jit` calls, already built, just need to be invoked
  against this checkpoint), per the earlier agreement: short, necessary,
  terminate immediately after.

Both are legitimate deployment targets per `docs/locomotion_architecture.md`
— not "one real, one fallback." The design below supports loading either.

### Where it lives

`system/cerberus_bringup/config/platforms.yaml` — not inside
`common/cerberus_description/`. Reasoning: `common/` is documented as
"shared, dependency-only, no standalone behavior," and this file's job is
selection/composition (which URDF variant, which checkpoints) across
multiple *other* components (`locomotion_policy`, `manipulation_policy`,
`robot_state_publisher`) — that's exactly what `cerberus_bringup` is already
documented to do ("launch files, composes the full graph"). Matches
Autoware's shape: the `vehicle_model` launch arg and its resolution logic
live in the top-level launch package (`autoware_launch`), not inside the
vehicle description package itself.

### Schema — deliberately thin, but checkpoints declare their own shape

`locomotion_checkpoint` becomes a *map of named modes* instead of one path,
each declaring the observation shape it needs (per
`docs/locomotion_architecture.md`'s Shape A/B/C taxonomy) — so the node
never hardcodes "the policy needs a height_scan," it reads that per mode:

```yaml
go2:
  urdf_relative_path: urdf/go2.urdf.xacro
  locomotion_modes:
    blind:
      checkpoint: policy/locomotion/checkpoints/go2_locomotion_flat_v1_candidate.onnx
      observation_shape: proprioception_only          # Shape A
    perceptive:
      checkpoint: policy/locomotion/checkpoints/go2_locomotion_v1_candidate.onnx
      observation_shape: proprioception_plus_height_scan   # Shape B
  default_locomotion_mode: blind

go2_arx5:
  urdf_relative_path: urdf/go2_arx5.urdf.xacro
  locomotion_modes:
    blind:
      checkpoint: policy/locomotion/checkpoints/go2_locomotion_flat_v1_candidate.onnx
      observation_shape: proprioception_only
  default_locomotion_mode: blind
  manipulation_checkpoint: policy/manipulation/checkpoints/go2_arx5_manipulation_final.pt
```

`default_locomotion_mode: blind` is a deliberate safety default — launching
without explicitly asking for `perceptive` gets the fully-tested, no-open-risk
mode (per `docs/locomotion_architecture.md`'s honest framing: perceptive is
legitimate but its noise-realism gap is untested). `go2_arx5` reuses the same
`blind` checkpoint as `go2` — the concrete payoff of locomotion being frozen
and composed, not retrained, when the arm is added (per
`project_structure.md`'s own reasoning).

**Still deliberately excludes joint names, joint limits, and self-collision
geometry**, even though `project_structure.md`'s rule 3 mentions loading
"CBF constants (joint limits, self-collision geometry, torque limits)... from
the embodiment config." Those already live in the URDF itself (`<limit
effort=... lower=... upper=...>` tags, `<collision>` geometry are standard
URDF elements) — duplicating them into `platforms.yaml` would create a
second source of truth that can drift from the URDF, exactly the "god
config" failure mode flagged earlier this session. Instead:

- `robot_state_publisher` loads the resolved xacro (per-platform, via
  `urdf_relative_path`) and publishes it on the standard `/robot_description`
  parameter/topic.
- `safety_filter/core/` (and any other node needing joint limits/collision
  geometry) parses `/robot_description` directly at startup — standard ROS2
  practice (`urdf_parser_py`/`kdl_parser`), not a `cerberus_description`
  Python import and not a second YAML.
- `platforms.yaml` only carries what genuinely doesn't exist anywhere else:
  *which* URDF variant, *which trained checkpoint(s)*, and *what shape each
  checkpoint's observations are* — selection data, not physical data.

### `locomotion_policy` node design

Runs inference from the exported ONNX (via `onnxruntime`, not the full
Isaac Lab/RSL-RL runtime — matches why `play.py` exports `policy.onnx` in
the first place). Observation assembly, read from `params/env.yaml` of the
run that produced each checkpoint (already confirmed locally for the Rough
candidate):

- **Shape A (blind)**: `base_lin_vel, base_ang_vel, projected_gravity,
  velocity_commands, joint_pos_rel, joint_vel_rel, last_action` — no sensor
  dependency beyond IMU + joint encoders + a commanded velocity topic.
- **Shape B (perceptive)**: the same seven terms, **plus `height_scan`**
  appended last (matches training's `concatenate_dim: -1` order). Needs a
  `height_scan` input topic. Until `perception/terrain_perception/` exists,
  this mode's `height_scan` input is a **constant flat-terrain stub**,
  clearly logged as such at node startup (`"[WARN] height_scan stubbed flat
  -- perceptive mode is running in bring-up/testing configuration, not a
  validated real-sensor deployment"`) — matches
  `docs/locomotion_architecture.md`'s honest framing: legitimate for
  bring-up, not yet a tested real-sensor deployment.
- Noise/corruption terms from training (`enable_corruption: true` at train
  time) are **off** at inference, matching existing Play-mode behavior
  already built into `*CerberusEnvCfg_PLAY`.
- Actions: `JointPositionAction`, `scale: 0.25`, `use_default_offset: true`,
  applied to all joints (`.*` pattern) — network output gets scaled and
  added to each joint's default position before being published as a
  position command, not used as a raw target directly.

Both modes share the same node code — only the observation-assembly function
and the loaded ONNX file differ, selected by `locomotion_modes.<mode>` from
`platforms.yaml`. Adding a future Shape C checkpoint means adding a new
observation-assembly function and a new `observation_shape` value, not
restructuring the node.

### Launch flow

```
ros2 launch cerberus_bringup bringup.launch.py platform:=go2 locomotion_mode:=blind
```

`bringup.launch.py`: declares a `platform` launch argument (default `go2`)
and a `locomotion_mode` argument (default resolved from the platform's own
`default_locomotion_mode` if not passed), reads `platforms.yaml` at launch
time (an `OpaqueFunction`, matching how ROS2 launch files conditionally
branch on argument values), resolves:

- `xacro:=$(find-pkg-share cerberus_description)/<urdf_relative_path>` passed
  to `robot_state_publisher`.
- `checkpoint_path` and `observation_shape` parameters on the
  `locomotion_policy` node, from `locomotion_modes.<locomotion_mode>`.
- `checkpoint_path` parameter on the `manipulation_policy` node, only
  launched at all if the resolved block has a `manipulation_checkpoint` key
  — so a legs-only platform like `go2` never starts a manipulation node with
  nothing to load.

**Only three things in the whole ROS2 graph are platform-aware**:
`cerberus_bringup` (the composer), `locomotion_policy`, and
`manipulation_policy` (each needs one checkpoint path). Every other package
in the tree (`terrain_perception`, `object_perception`, `slam`,
`global_planner`, `task_orchestrator`, `cerberus_safety_filter`,
`cerberus_actuation`) stays fully generic — they get robot identity from
`/robot_description`/TF like any standard ROS2 node, needing zero code
changes when the platform changes. This is the concrete fulfillment of
`project_structure.md` rule 3's "ROS2 inference nodes stay generic."

### Naming contract across three surfaces

The string `"go2"` (and later `"go2_arx5"`) has to mean the same thing in
three independent places: `PlatformCfg.name` (Python, training side),
`platforms.yaml`'s top-level key (ROS2 side), and whatever a user types
after `--platform`/`platform:=`. Since Part B's Go2 slice is buildable now,
**this check gets built**, not deferred: a small test loads `platforms.yaml`
and confirms every top-level key matches a name that would come from Isaac
Lab's `available_platforms()` (hardcoded expected set for now, since
`isaaclab_tasks` isn't importable in this same test process — see Part A's
own local-testability constraint) — catches a typo'd or orphaned entry on
either side. Extending it to `go2_arx5` once that combo exists is a one-line
addition to the expected set, not a rewrite.

### What's genuinely undecided, on purpose

- **Whether `manipulation_checkpoint` should be a single path or itself a
  small block** (e.g. separate BC/Diffusion/ACT checkpoints being compared
  per `project_structure.md`'s Phase 2 `algorithms/` list) — depends on
  Phase 2's own eval-selection decisions, not decidable from here.
- **Exact xacro authoring** (how `go2_arx5.urdf.xacro` itself references the
  merged USD or a separately-authored URDF) — depends on
  `docs/embodiment_asset_composition.md`'s Robot Assembler script actually
  running, which hasn't happened. `go2.urdf.xacro` (buildable now) has no
  such dependency — Go2's own URDF is sourced directly (e.g. Unitree's own
  ROS2 packages, or a community port), no merge step needed.

## Testing / verification plan

**Part A**: local `py_compile` now (syntax only, `isaaclab_tasks` unavailable
locally); pod-side functional check deferred to the existing plan's Task 11
pattern (import `policy.locomotion`, confirm `available_platforms()` and all
four gym ids unchanged).

**Part B, Go2-only slice**: genuinely buildable and testable now, via
`pixi run colcon build` / `pixi run ros2 launch ...` locally:
- `platforms.yaml` schema validity + the naming-contract check above: plain
  Python/YAML, no ROS2 dependency, testable immediately.
- `robot_state_publisher` actually parses the resolved Go2 xacro and
  publishes `/robot_description` — testable via `ros2 launch` + `ros2 topic
  echo` once Go2's URDF is sourced.
- `locomotion_policy` node: loading the ONNX and running one inference step
  against a synthetic observation vector is testable without any real robot
  or Isaac Sim (pure `onnxruntime` + fabricated input). End-to-end (real
  joint-state topics, real actuation) needs actual robot hardware or a
  separate sim bridge — out of scope for what "buildable now" means here,
  flagged as a real gap, not silently assumed away.

**Part B, `go2_arx5` combo case**: stays deferred to Phase 2 as before —
genuinely blocked on the asset merge (`embodiment_asset_composition.md`) and
an ARX5 asset that doesn't exist yet.

## Open questions / risks

- `import_packages`'s exact signature (positional vs keyword `blacklist_pkgs`,
  whether it returns anything callers should check) was read from
  LeggedManip_Lab's usage, not independently verified against
  `isaaclab_tasks`' own source (not installed locally) — low risk since it's
  a one-line call, but worth confirming on the first pod-side run after this
  change lands.
- The naming-contract check now gets built for the Go2-only slice (see
  Testing / verification plan above), so this is no longer an open gap for
  `go2` — only for `go2_arx5` once it's added, at which point extending the
  check's expected-name set is the reminder, not writing the check itself.
- **Go2's own URDF source is not yet chosen.** Research this session found
  a community ROS2 port (`anujjain-dev/unitree-go2-ros2`) as one option;
  Unitree's own official channels may have one too. Needs a concrete choice
  before `common/cerberus_description/`'s `go2.urdf.xacro` can actually be
  authored — flagged for the implementation plan, not resolved here.
- **The Rough checkpoint's ONNX export is a real dependency of Part B's
  "perceptive" mode being loadable at all** — until the short pod export
  happens (see Checkpoint status above), `locomotion_modes.perceptive`'s
  `checkpoint` path in `platforms.yaml` points at a file that doesn't exist
  yet. `blind` mode has no such dependency (Flat's ONNX is already on R2).
- **The `height_scan` flat-stub in perceptive mode is a real, load-bearing
  simplification**, not a minor detail — it means perceptive mode's
  behavior on real hardware is currently untested even for the "legitimate
  architecture" read from `docs/locomotion_architecture.md`, since a
  constant stub isn't the noisy-real-sensor input the checkpoint was
  validated (in sim) against. Worth keeping the startup warning log loud,
  not quiet, so this doesn't get forgotten once the node "just works" in a
  demo.
