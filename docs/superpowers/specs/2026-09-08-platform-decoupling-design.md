# Platform Decoupling Design

Date: 2026-09-08

## Problem

The locomotion codebase currently only supports the Unitree Go2. Nothing about
the CLI, checkpoint naming, R2 sync, or Docker image is Go2-specific (verified
by a codebase survey, see below), but three things are:

1. `policy/locomotion/__init__.py` hand-writes four `gym.register()` calls
   pointing directly at Go2 env cfg classes and Go2's upstream PPO runner cfg.
2. `policy/locomotion/core/push_disturbance_cfg.py` hardcodes Go2-tuned push
   force/torque as module-level constants (`PEAK_FORCE_N`, `PEAK_TORQUE_NM`)
   and hardcodes the root body name (`"base"`) at two call sites.
3. `policy/locomotion/core/go2_rough_env_cfg.py` hardcodes a Go2-leg-length
   terrain rescale factor (`stairs_scale = 0.5`) as a local variable.

None of this is deeply broken — the existing subclass-of-upstream-Isaac-Lab-cfg
pattern is sound and idiomatic. But there is no enforced *contract* for what a
new robot platform must supply, and two of the three items above (`base` body
name, the two peak constants) would fail silently rather than loudly if a
second platform's author forgot to override them.

Goal: restructure so that adding a second robot platform means adding one new
self-contained directory plus one new import line, touching zero existing
Go2 files, with the required per-platform values enforced by a dataclass
rather than discoverable only by reading Go2's source.

Non-goal: this does not attempt to abstract the robot's `ArticulationCfg`
itself (USD, joint names, actuators, reward/termination terms). That still
has to come from Isaac Lab shipping a stock config for the new robot, or
someone authoring one — real engineering work, out of scope here and sized
separately once an actual second platform is chosen.

## Current state (verified)

- CLI scripts, checkpoint path resolution (`agent_cfg.experiment_name`), R2
  sync, `run_remote_job.sh`, and the Docker image contain no Go2-specific
  logic today — confirmed via full-repo grep, no changes needed there.
- `curriculum.py` and `terrain_pinning.py` are fully robot-agnostic already.
- `go2_rough_env_cfg.py` / `go2_flat_env_cfg.py` subclass Isaac Lab's stock
  `UnitreeGo2RoughEnvCfg` / `UnitreeGo2FlatEnvCfg` (from `isaaclab_tasks`,
  not vendored in this repo) and add only: push disturbance, a terrain
  step-height rescale (rough only), and two diagnostic curriculum terms.
- `policy/locomotion/__init__.py` registers `Isaac-Velocity-{Flat,Rough}-
  Unitree-Go2-Cerberus-{,Play}-v0`, each pointing `env_cfg_entry_point` at a
  lazy `"module:ClassName"` string and `rsl_rl_cfg_entry_point` at an actual
  imported class object (`UnitreeGo2FlatPPORunnerCfg`/`UnitreeGo2RoughPPORunnerCfg`)
  — these two kwargs are NOT symmetric today, and the new design must
  preserve that asymmetry rather than force both to strings.

## Design

### New module: `policy/locomotion/core/platforms/registry.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import gymnasium as gym


@dataclass(frozen=True)
class PlatformCfg:
    """Everything platform-specific the training/eval/CLI stack needs from a
    robot. A new platform is a new PlatformCfg instance plus a new env_cfg
    module -- nothing else in the stack should need touching."""

    name: str  # "go2" -- used for --platform, checkpoint prefix, registry key

    # gym task ids
    flat_task_id: str
    flat_play_task_id: str
    rough_task_id: str
    rough_play_task_id: str

    # lazy "module:ClassName" strings (matches today's env_cfg_entry_point convention)
    flat_env_cfg_entry_point: str
    rough_env_cfg_entry_point: str

    # actual imported class objects (matches today's rsl_rl_cfg_entry_point convention --
    # NOT a string, unlike the env_cfg entry points above)
    flat_agent_cfg: Any
    rough_agent_cfg: Any

    # push-disturbance curriculum, mass/geometry-tuned per robot -- see REFERENCES.md
    # for how Go2's values were derived; a new platform must re-derive its own.
    push_peak_force_n: float
    push_peak_torque_nm: float
    root_body_name: str = "base"

    # terrain rescale (rough only), relative to Isaac Lab's stock ANYmal-scale defaults
    stairs_step_height_scale: float = 1.0

    # must match this platform's agent cfg's num_steps_per_env (curriculum cadence --
    # see curriculum.py's NUM_STEPS_PER_ENV comment)
    num_steps_per_env: int = 24

    checkpoint_prefix: str = ""

    def __post_init__(self):
        if not self.checkpoint_prefix:
            object.__setattr__(self, "checkpoint_prefix", f"{self.name}_locomotion")


_PLATFORMS: dict[str, PlatformCfg] = {}


def register_platform(cfg: PlatformCfg) -> None:
    if cfg.name in _PLATFORMS:
        raise ValueError(f"platform '{cfg.name}' already registered")
    _PLATFORMS[cfg.name] = cfg

    for task_id, env_ep, agent_cfg, play_variant in (
        (cfg.flat_task_id, cfg.flat_env_cfg_entry_point, cfg.flat_agent_cfg, False),
        (cfg.flat_play_task_id, cfg.flat_env_cfg_entry_point, cfg.flat_agent_cfg, True),
        (cfg.rough_task_id, cfg.rough_env_cfg_entry_point, cfg.rough_agent_cfg, False),
        (cfg.rough_play_task_id, cfg.rough_env_cfg_entry_point, cfg.rough_agent_cfg, True),
    ):
        cfg_class_name = env_ep if not play_variant else f"{env_ep}_PLAY"
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": cfg_class_name,
                "rsl_rl_cfg_entry_point": agent_cfg,
            },
        )


def get_platform(name: str) -> PlatformCfg:
    try:
        return _PLATFORMS[name]
    except KeyError:
        raise ValueError(f"unknown platform '{name}', available: {sorted(_PLATFORMS)}") from None


def available_platforms() -> list[str]:
    return sorted(_PLATFORMS)
```

Note: the `_PLAY` suffix convention on the env cfg class name (not a separate
entry-point string) matches today's `UnitreeGo2RoughCerberusEnvCfg_PLAY`
naming — kept as-is rather than introducing a fifth dataclass field.

### New package: `policy/locomotion/core/platforms/go2/`

- `env_cfg.py` — moved from `core/go2_rough_env_cfg.py` / `core/go2_flat_env_cfg.py`
  verbatim, except reading push/terrain values off the `GO2` platform instance
  instead of local module constants:

```python
from policy.locomotion.core.platforms.go2.platform import GO2

class UnitreeGo2RoughCerberusEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_push_disturbance(
            self,
            peak_force_n=GO2.push_peak_force_n,
            peak_torque_nm=GO2.push_peak_torque_nm,
            root_body_name=GO2.root_body_name,
        )
        stairs = self.scene.terrain.terrain_generator.sub_terrains
        for key in ("pyramid_stairs", "pyramid_stairs_inv"):
            lo, hi = stairs[key].step_height_range
            stairs[key].step_height_range = (
                lo * GO2.stairs_step_height_scale, hi * GO2.stairs_step_height_scale,
            )
        max_level = self.scene.terrain.terrain_generator.num_rows - 1
        self.curriculum.terrain_level_max = CurrTerm(func=terrain_level_max)
        self.curriculum.terrain_level_frac_at_max = CurrTerm(
            func=terrain_level_frac_at_max, params={"max_level": max_level}
        )
```

- `platform.py` — the `GO2` instance, carrying forward today's exact
  provenance comments (87.5%-recovery-at-120N source, torque-estimate caveat,
  interval-tuning rationale) so nothing gets lost in the move:

```python
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
    UnitreeGo2RoughPPORunnerCfg,
)
from policy.locomotion.core.platforms.registry import PlatformCfg, register_platform

# Peak force anchored to isaaclab-go2-locomotion's validated Go2 number (87.5%
# recovery at 120N vs 0% untrained baseline). Torque has no verified source in
# the Hwangbo/Lee/Miki lineage; it's a reasoned estimate (peak force x ~0.15m
# Go2 body-frame lever arm). See REFERENCES.md.
GO2 = PlatformCfg(
    name="go2",
    flat_task_id="Isaac-Velocity-Flat-Unitree-Go2-Cerberus-v0",
    flat_play_task_id="Isaac-Velocity-Flat-Unitree-Go2-Cerberus-Play-v0",
    rough_task_id="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-v0",
    rough_play_task_id="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0",
    flat_env_cfg_entry_point="policy.locomotion.core.platforms.go2.env_cfg:UnitreeGo2FlatCerberusEnvCfg",
    rough_env_cfg_entry_point="policy.locomotion.core.platforms.go2.env_cfg:UnitreeGo2RoughCerberusEnvCfg",
    flat_agent_cfg=UnitreeGo2FlatPPORunnerCfg,
    rough_agent_cfg=UnitreeGo2RoughPPORunnerCfg,
    push_peak_force_n=120.0,
    push_peak_torque_nm=18.0,
    root_body_name="base",
    stairs_step_height_scale=0.5,
    num_steps_per_env=24,
)

register_platform(GO2)
```

### `push_disturbance_cfg.py` changes

`add_push_disturbance()` gains three required parameters instead of reading
module constants:

```python
def add_push_disturbance(
    env_cfg: ManagerBasedRLEnvCfg,
    *,
    peak_force_n: float,
    peak_torque_nm: float,
    root_body_name: str,
) -> None:
    env_cfg.events.push_disturbance = EventTerm(
        ...,
        params={
            ...,
            "asset_cfg": SceneEntityCfg("robot", body_names=root_body_name),
        },
    )
    env_cfg.events.push_disturbance_clear = EventTerm(
        ...,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=root_body_name)},
    )
    env_cfg.curriculum.push_disturbance_curriculum = CurrTerm(
        func=push_disturbance_curriculum,
        params={
            "event_term_name": "push_disturbance",
            "peak_force": peak_force_n,
            "peak_torque": peak_torque_nm,
            "sparse_interval_range_s": SPARSE_INTERVAL_RANGE_S,
            "dense_interval_range_s": DENSE_INTERVAL_RANGE_S,
            "interval_saturation_k_c": INTERVAL_SATURATION_K_C,
        },
    )
```

`SPARSE_INTERVAL_RANGE_S`, `DENSE_INTERVAL_RANGE_S`, `INTERVAL_SATURATION_K_C`,
`IMPULSE_DURATION_S` stay as module constants — these are training-dynamics
tuning, not robot-geometry-tuned, per the existing research finding.

### `policy/locomotion/__init__.py` shrinks to

```python
"""Cerberus locomotion task registration. Importing this module registers
every known robot platform's gym environments -- add a platform by adding one
import line here, not by editing this file's logic."""

from policy.locomotion.core.platforms.go2 import platform as _go2_platform  # noqa: F401
```

### CLI changes (`train.py`, `play.py`, eval/showcase scripts)

- New `--platform` (default `"go2"`, `choices=available_platforms()`)
- New `--terrain` (`choices=["flat", "rough"]`, default `"rough"` — matches
  today's per-script defaults, each script keeps its own default)
- `--task` stays as a raw-override escape hatch: if passed, used as-is; if
  not, resolved as `platform.rough_task_id if args_cli.terrain == "rough"
  else platform.flat_task_id` (and the `_play_task_id` variants for play/eval
  scripts)
- Checkpoint root resolution keeps using `agent_cfg.experiment_name` (no
  change), but `docs/project_structure.md`'s naming convention gets a
  one-line update noting `platform.checkpoint_prefix` is now the source of
  truth for the prefix rather than an unenforced human convention

## Migration steps (for the implementation plan)

1. Create `core/platforms/registry.py` with `PlatformCfg`/`register_platform`/
   `get_platform`/`available_platforms`.
2. Create `core/platforms/go2/__init__.py` (empty), `platform.py`, and move
   `core/go2_rough_env_cfg.py` + `core/go2_flat_env_cfg.py` into
   `core/platforms/go2/env_cfg.py` (merge both classes into one file, or keep
   as two files in the new directory — implementation plan's call), updating
   them to read from `GO2` instead of local constants.
3. Update `push_disturbance_cfg.py`'s `add_push_disturbance()` signature;
   update its two call sites (flat + rough env cfg) to pass `GO2.*` values.
4. Replace `policy/locomotion/__init__.py`'s four `gym.register()` calls with
   the single platform import.
5. Add `--platform`/`--terrain` to `train.py`, `play.py`, and every eval/
   showcase script under `policy/locomotion/eval/`; resolve task id via
   `get_platform()`.
6. Update `docs/project_structure.md`'s checkpoint-naming section to reference
   `platform.checkpoint_prefix`.
7. Grep the whole repo for any remaining reference to `core.go2_rough_env_cfg`
   / `core.go2_flat_env_cfg` / `PEAK_FORCE_N` / `PEAK_TORQUE_NM` to confirm
   nothing was missed (scripts, docs, `REFERENCES.md` cross-references).
8. Verification: since Isaac Lab isn't installed locally, this can only be
   fully verified on a RunPod pod (import the module, confirm all four gym
   IDs register without error, run a short flat training smoke test). Locally,
   confirm via `python -c "import ast; ..."`-style static checks or a plain
   `python -m py_compile` pass on every touched file, plus a careful read-through
   diff — flag this limitation explicitly rather than claiming full verification
   without pod access.

## Testing / verification plan

- No unit tests exist for this layer today (Isaac Lab env cfgs aren't
  typically unit-tested in isolation). Verification is: (a) static/import-time
  checks locally, (b) a pod-side smoke test importing
  `policy.locomotion` and asserting all four Go2 gym IDs are present in
  `gym.registry.keys()`, (c) a short (~10-20 iteration) training run on both
  flat and rough to confirm no runtime regression versus the pre-refactor
  behavior, before trusting this for a real full retrain.
- This refactor must not change Go2's actual trained behavior — same push
  force/torque/interval/terrain-scale values, just relocated. The pod smoke
  test should confirm reward/curriculum values match the pre-refactor run at
  iteration ~10 (sanity, not a full retrain).

## Open questions / risks

- `isaaclab_tasks` is not installed locally, so this design's assumptions
  about `env_cfg_entry_point` string resolution and `rsl_rl_cfg_entry_point`
  class-vs-string handling are based on reading this repo's existing working
  code, not on independently verifying Isaac Lab's `gym.register()` kwarg
  contract from its own source. Low risk (we're preserving today's exact
  kwarg shapes unchanged), but worth a first pod-side smoke test before
  building further on top.
- Merging `go2_rough_env_cfg.py`/`go2_flat_env_cfg.py` into one `env_cfg.py`
  vs. keeping them as two files in the new directory is left to the
  implementation plan — either is fine, this design doesn't mandate one.
