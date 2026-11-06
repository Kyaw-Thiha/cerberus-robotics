# Platform Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the locomotion codebase so a new robot platform can be added as one new `core/platforms/<robot>/` directory plus one new import line, without touching any existing Go2 file, script, or the registry itself.

**Architecture:** A `PlatformCfg` dataclass (in a new `core/platforms/registry.py`) captures everything platform-specific — gym task ids, env cfg entry points, PPO agent cfg classes, push-disturbance magnitudes, root body name, terrain rescale factor. Go2's values move out of scattered module constants into one `GO2` instance. CLI scripts gain `--platform`/`--terrain` flags that resolve task ids from the registry, with `--task` kept as a raw-override escape hatch.

**Tech Stack:** Python 3.10+, `gymnasium` (via Isaac Lab's `isaaclab` package, not present in this local dev environment), Isaac Lab 2.3.2 / Isaac Sim 5.1.0 (present only in the RunPod remote image, per `docker/Dockerfile.remote`).

## Global Constraints

- Do not change Go2's actual trained behavior — same push force/torque/interval/terrain-scale values, just relocated to new files. No re-tuning as part of this plan.
- Do not commit or push anything — leave all changes in the working tree uncommitted. (Per explicit user instruction this session.)
- Do not start, create, or use any RunPod pod as part of this plan. Pod-side verification is written up as a documented follow-up step for the user to run manually later, not executed now.
- `isaaclab`/`isaaclab_tasks`/`gymnasium` are not installed in this local dev environment — any step that imports them will fail locally by design. Where a step needs `gymnasium` only (no Isaac Lab), install it in a disposable scratch venv (see Task 1) rather than the project's pinned environment.
- Preserve the exact `gym.register()` kwarg shapes already in use: `env_cfg_entry_point` is a lazy `"module:ClassName"` string; `rsl_rl_cfg_entry_point` is an actual imported class object, not a string. Do not make these symmetric.
- Preserve the `_PLAY` suffix convention on env cfg class names (e.g. `UnitreeGo2RoughCerberusEnvCfg_PLAY`) rather than introducing a new entry-point field for play variants.

---

## Task 1: `PlatformCfg` registry

**Files:**
- Create: `policy/locomotion/core/platforms/__init__.py`
- Create: `policy/locomotion/core/platforms/registry.py`
- Test: `policy/locomotion/core/platforms/test_registry.py`

**Interfaces:**
- Produces: `PlatformCfg` (frozen dataclass), `register_platform(cfg: PlatformCfg) -> None`, `get_platform(name: str) -> PlatformCfg`, `available_platforms() -> list[str]`. All later tasks import these from `policy.locomotion.core.platforms.registry`.

- [ ] **Step 1: Create the empty package marker**

`policy/locomotion/core/platforms/__init__.py`:
```python
```
(empty file — this package holds per-platform subpackages; nothing to export at this level)

- [ ] **Step 2: Write the failing tests**

`policy/locomotion/core/platforms/test_registry.py`:
```python
import gymnasium as gym
import pytest

from policy.locomotion.core.platforms.registry import (
    PlatformCfg,
    available_platforms,
    get_platform,
    register_platform,
)


def _make_cfg(name: str) -> PlatformCfg:
    return PlatformCfg(
        name=name,
        flat_task_id=f"Test-Flat-{name}-v0",
        flat_play_task_id=f"Test-Flat-{name}-Play-v0",
        rough_task_id=f"Test-Rough-{name}-v0",
        rough_play_task_id=f"Test-Rough-{name}-Play-v0",
        flat_env_cfg_entry_point=f"some.module.{name}:FlatEnvCfg",
        rough_env_cfg_entry_point=f"some.module.{name}:RoughEnvCfg",
        flat_agent_cfg=object(),
        rough_agent_cfg=object(),
        push_peak_force_n=100.0,
        push_peak_torque_nm=10.0,
        root_body_name="trunk",
        stairs_step_height_scale=0.7,
        num_steps_per_env=24,
    )


def test_checkpoint_prefix_defaults_from_name():
    cfg = _make_cfg("testbot")
    assert cfg.checkpoint_prefix == "testbot_locomotion"


def test_checkpoint_prefix_explicit_override_kept():
    cfg = PlatformCfg(
        name="testbot2",
        flat_task_id="Test-Flat-testbot2-v0",
        flat_play_task_id="Test-Flat-testbot2-Play-v0",
        rough_task_id="Test-Rough-testbot2-v0",
        rough_play_task_id="Test-Rough-testbot2-Play-v0",
        flat_env_cfg_entry_point="some.module:FlatEnvCfg",
        rough_env_cfg_entry_point="some.module:RoughEnvCfg",
        flat_agent_cfg=object(),
        rough_agent_cfg=object(),
        push_peak_force_n=100.0,
        push_peak_torque_nm=10.0,
        checkpoint_prefix="custom_prefix",
    )
    assert cfg.checkpoint_prefix == "custom_prefix"


def test_register_platform_adds_to_available_platforms():
    cfg = _make_cfg("regtest1")
    register_platform(cfg)
    assert "regtest1" in available_platforms()


def test_register_platform_rejects_duplicate_name():
    cfg = _make_cfg("regtest2")
    register_platform(cfg)
    with pytest.raises(ValueError, match="already registered"):
        register_platform(cfg)


def test_register_platform_calls_gym_register_for_all_four_task_ids():
    cfg = _make_cfg("regtest3")
    register_platform(cfg)
    registered_ids = set(gym.registry.keys())
    assert cfg.flat_task_id in registered_ids
    assert cfg.flat_play_task_id in registered_ids
    assert cfg.rough_task_id in registered_ids
    assert cfg.rough_play_task_id in registered_ids


def test_register_platform_play_variant_appends_play_suffix_to_class_name():
    cfg = _make_cfg("regtest4")
    register_platform(cfg)
    spec = gym.registry[cfg.rough_play_task_id]
    assert spec.kwargs["env_cfg_entry_point"] == f"{cfg.rough_env_cfg_entry_point}_PLAY"
    non_play_spec = gym.registry[cfg.rough_task_id]
    assert non_play_spec.kwargs["env_cfg_entry_point"] == cfg.rough_env_cfg_entry_point


def test_register_platform_passes_agent_cfg_as_object_not_string():
    cfg = _make_cfg("regtest5")
    register_platform(cfg)
    spec = gym.registry[cfg.flat_task_id]
    assert spec.kwargs["rsl_rl_cfg_entry_point"] is cfg.flat_agent_cfg


def test_get_platform_returns_registered_cfg():
    cfg = _make_cfg("regtest6")
    register_platform(cfg)
    assert get_platform("regtest6") is cfg


def test_get_platform_unknown_name_raises_with_available_list():
    with pytest.raises(ValueError, match="unknown platform 'nope'"):
        get_platform("nope")
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /home/kevin/Documents/Projects/cerberus-robotics
python3 -m venv /tmp/platform_registry_test_venv
/tmp/platform_registry_test_venv/bin/pip install --quiet gymnasium pytest
/tmp/platform_registry_test_venv/bin/python -m pytest policy/locomotion/core/platforms/test_registry.py -v
```
Expected: FAIL / collection error — `policy.locomotion.core.platforms.registry` does not exist yet.

- [ ] **Step 4: Write the implementation**

`policy/locomotion/core/platforms/registry.py`:
```python
"""Per-robot-platform contract. A new robot platform is a new PlatformCfg
instance (see core/platforms/go2/platform.py for the reference example) plus
a new env_cfg module -- nothing else in the training/eval/CLI stack should
need touching to add one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym


@dataclass(frozen=True)
class PlatformCfg:
    """Everything platform-specific the training/eval/CLI stack needs from a
    robot."""

    name: str  # e.g. "go2" -- used for --platform, checkpoint prefix, registry key

    # gym task ids
    flat_task_id: str
    flat_play_task_id: str
    rough_task_id: str
    rough_play_task_id: str

    # lazy "module:ClassName" strings (Isaac Lab's env_cfg_entry_point convention) --
    # the _PLAY variant's class name is derived by appending "_PLAY", not stored
    # separately.
    flat_env_cfg_entry_point: str
    rough_env_cfg_entry_point: str

    # actual imported class objects (Isaac Lab's rsl_rl_cfg_entry_point convention --
    # NOT a string, unlike the env_cfg entry points above)
    flat_agent_cfg: Any
    rough_agent_cfg: Any

    # push-disturbance curriculum, mass/geometry-tuned per robot -- see
    # policy/locomotion/REFERENCES.md for how Go2's values were derived; a new
    # platform must re-derive its own rather than reuse Go2's.
    push_peak_force_n: float
    push_peak_torque_nm: float
    root_body_name: str = "base"

    # terrain rescale (rough only), relative to Isaac Lab's stock ANYmal-scale defaults
    stairs_step_height_scale: float = 1.0

    # must match this platform's agent cfg's num_steps_per_env (curriculum cadence --
    # see core/curriculum.py's NUM_STEPS_PER_ENV comment)
    num_steps_per_env: int = 24

    checkpoint_prefix: str = ""

    def __post_init__(self) -> None:
        if not self.checkpoint_prefix:
            object.__setattr__(self, "checkpoint_prefix", f"{self.name}_locomotion")


_PLATFORMS: dict[str, PlatformCfg] = {}


def register_platform(cfg: PlatformCfg) -> None:
    """Records `cfg` in the platform registry and registers its four gym
    environments (flat/rough x train/play). Raises if `cfg.name` is already
    registered -- call this exactly once per platform, typically at import
    time of that platform's `platform.py` module.
    """
    if cfg.name in _PLATFORMS:
        raise ValueError(f"platform '{cfg.name}' already registered")
    _PLATFORMS[cfg.name] = cfg

    for task_id, env_cfg_entry_point, agent_cfg, is_play in (
        (cfg.flat_task_id, cfg.flat_env_cfg_entry_point, cfg.flat_agent_cfg, False),
        (cfg.flat_play_task_id, cfg.flat_env_cfg_entry_point, cfg.flat_agent_cfg, True),
        (cfg.rough_task_id, cfg.rough_env_cfg_entry_point, cfg.rough_agent_cfg, False),
        (cfg.rough_play_task_id, cfg.rough_env_cfg_entry_point, cfg.rough_agent_cfg, True),
    ):
        resolved_entry_point = f"{env_cfg_entry_point}_PLAY" if is_play else env_cfg_entry_point
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": resolved_entry_point,
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

- [ ] **Step 5: Run tests to verify they pass**

```bash
/tmp/platform_registry_test_venv/bin/python -m pytest policy/locomotion/core/platforms/test_registry.py -v
```
Expected: PASS (10 tests)

- [ ] **Step 6: Clean up the scratch venv**

```bash
rm -rf /tmp/platform_registry_test_venv
```

(No commit — per this plan's global constraint, leave all changes uncommitted in the working tree.)

---

## Task 2: Go2 platform package

**Files:**
- Create: `policy/locomotion/core/platforms/go2/__init__.py`
- Create: `policy/locomotion/core/platforms/go2/platform.py`
- Create: `policy/locomotion/core/platforms/go2/env_cfg.py`
- Delete: `policy/locomotion/core/go2_flat_env_cfg.py`
- Delete: `policy/locomotion/core/go2_rough_env_cfg.py`

**Interfaces:**
- Consumes: `PlatformCfg`, `register_platform` from Task 1's `policy.locomotion.core.platforms.registry`. `add_push_disturbance` from Task 3 (signature changed there — write this task assuming the new signature `add_push_disturbance(env_cfg, *, peak_force_n, peak_torque_nm, root_body_name)`; Task 3 can be done before or after this one, order doesn't matter since neither imports the other's new symbols until both are complete — do Task 3 first if executing tasks in order, to avoid a transient broken import).
- Produces: `GO2` (a `PlatformCfg` instance), `UnitreeGo2FlatCerberusEnvCfg`, `UnitreeGo2FlatCerberusEnvCfg_PLAY`, `UnitreeGo2RoughCerberusEnvCfg`, `UnitreeGo2RoughCerberusEnvCfg_PLAY` — all under `policy.locomotion.core.platforms.go2.env_cfg` and `...go2.platform`. Task 4 (`__init__.py` simplification) imports `policy.locomotion.core.platforms.go2.platform`.

This task cannot be locally unit-tested (requires `isaaclab`/`isaaclab_tasks`, not installed here) — verify with `python -m py_compile` for syntax correctness locally, and flag full behavioral verification as the pod-side follow-up in Task 11.

- [ ] **Step 1: Create the empty package marker**

`policy/locomotion/core/platforms/go2/__init__.py`:
```python
```
(empty file)

- [ ] **Step 2: Create `platform.py`**

`policy/locomotion/core/platforms/go2/platform.py`:
```python
"""Go2's PlatformCfg instance -- the single source of truth for every
Go2-specific value the training/eval/CLI stack needs. See
core/platforms/registry.py for the field contract every platform must
satisfy.
"""

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
    UnitreeGo2RoughPPORunnerCfg,
)

from policy.locomotion.core.platforms.registry import PlatformCfg, register_platform

# Push peak force anchored to isaaclab-go2-locomotion's validated Go2 number
# (87.5% recovery at 120N peak vs 0% for the untrained baseline). Torque has
# no verified source anywhere in the Hwangbo/Lee/Miki lineage or the
# reference repo; it's a reasoned ESTIMATE (peak force x ~0.15m, an
# approximate Go2 body-frame lever arm), not a verified figure. See
# policy/locomotion/REFERENCES.md.
#
# stairs_step_height_scale: stock UnitreeGo2RoughEnvCfg already rescales
# "boxes"/"random_rough" for Go2's smaller body size, but leaves
# "pyramid_stairs"/"pyramid_stairs_inv" at their stock (ANYmal-scale)
# step_height_range. Rescaled by the same 0.5x factor stock's own boxes
# rescale used -- a reasoned estimate following that established pattern,
# not independently verified. See policy/locomotion/RUN_001.md and
# REFERENCES.md.
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

- [ ] **Step 3: Create `env_cfg.py`** (merges the two existing files, reading push/terrain values off `GO2` instead of local module constants)

`policy/locomotion/core/platforms/go2/env_cfg.py`:
```python
"""Go2 flat and rough locomotion env cfgs, with the Hwangbo-curriculum push
disturbance added on top of Isaac Lab's stock UnitreeGo2FlatEnvCfg /
UnitreeGo2RoughEnvCfg. Rewards, PPO hyperparameters, and network architecture
are untouched -- see policy/locomotion/REFERENCES.md. Values (push
force/torque, root body name, terrain rescale) come from this platform's GO2
PlatformCfg instance (platform.py), not hardcoded here.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

from policy.locomotion.core.curriculum import terrain_level_frac_at_max, terrain_level_max
from policy.locomotion.core.platforms.go2.platform import GO2
from policy.locomotion.core.push_disturbance_cfg import add_push_disturbance


@configclass
class UnitreeGo2FlatCerberusEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_push_disturbance(
            self,
            peak_force_n=GO2.push_peak_force_n,
            peak_torque_nm=GO2.push_peak_torque_nm,
            root_body_name=GO2.root_body_name,
        )


@configclass
class UnitreeGo2FlatCerberusEnvCfg_PLAY(UnitreeGo2FlatCerberusEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing events -- push recovery is evaluated separately
        # (policy/locomotion/eval/), not exercised during a play/visualization run
        self.events.base_external_force_torque = None
        self.events.push_disturbance = None
        # the curriculum term references push_disturbance above; must go too
        self.curriculum.push_disturbance_curriculum = None

        # Follow-cam: the default viewer is a fixed world-space camera that
        # never tracks the robot, so footage drifts out of frame once a push
        # (or just plain walking) moves the robot away from its spawn point --
        # see REFERENCES.md's camera-framing writeup. asset_root origin makes
        # Isaac Lab recompute eye/lookat from the robot's current root pose
        # every step instead of once. Needs a pod-side smoke test to confirm
        # (not verifiable from source alone -- Isaac Lab isn't installed here).
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (-2.5, -2.5, 1.3)
        self.viewer.lookat = (0.0, 0.0, 0.3)


@configclass
class UnitreeGo2RoughCerberusEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_push_disturbance(
            self,
            peak_force_n=GO2.push_peak_force_n,
            peak_torque_nm=GO2.push_peak_torque_nm,
            root_body_name=GO2.root_body_name,
        )

        # Stock UnitreeGo2RoughEnvCfg (super().__post_init__() above) already rescales
        # "boxes"/"random_rough" for Go2's smaller body size, but leaves "pyramid_stairs"/
        # "pyramid_stairs_inv" at their stock (ANYmal-scale) step_height_range -- an
        # absolute-meters parameter, unlike hf_pyramid_slope's scale-invariant slope_range
        # ratio. Found via a real per-sub-terrain-type push-recovery eval (2026-09-04,
        # see RUN_001.md): pyramid_stairs_inv scored just 12% success (vs 91-97% for both
        # slope types and random_rough), pyramid_stairs 49% -- both far below boxes' own
        # already-rescaled 63.5% -- and pyramid_stairs_inv was already bad at the easiest
        # level tested, not a difficulty-driven decline like the other types, the
        # signature of an unscaled absolute-length parameter rather than a genuine skill
        # ceiling. See policy/locomotion/REFERENCES.md.
        stock_step_height_range = (0.05, 0.23)
        scaled_step_height_range = tuple(v * GO2.stairs_step_height_scale for v in stock_step_height_range)
        for stairs_type in ("pyramid_stairs", "pyramid_stairs_inv"):
            self.scene.terrain.terrain_generator.sub_terrains[stairs_type].step_height_range = (
                scaled_step_height_range
            )

        # Diagnostic-only curriculum terms (no effect on training): the stock
        # terrain_levels term above only logs the MEAN across envs, which can't
        # confirm Phase 0's actual "Done when" criterion --
        # envs reaching the curriculum's top row and cycling back down, not
        # just a high average. See curriculum.py's terrain_level_max /
        # terrain_level_frac_at_max docstrings.
        max_level = self.scene.terrain.terrain_generator.num_rows - 1
        self.curriculum.terrain_level_max = CurrTerm(func=terrain_level_max)
        self.curriculum.terrain_level_frac_at_max = CurrTerm(
            func=terrain_level_frac_at_max, params={"max_level": max_level}
        )


@configclass
class UnitreeGo2RoughCerberusEnvCfg_PLAY(UnitreeGo2RoughCerberusEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        # Deliberately NOT shrinking terrain_generator to a smaller grid here --
        # upstream's own Play config does (5x5, curriculum=False) purely to save
        # memory at training's num_envs=4096, but that tradeoff doesn't apply to
        # play/eval's much smaller env counts. Keeping the full ROUGH_TERRAINS_CFG
        # (10 rows x 20 cols, curriculum=True) means: (a) play.py's video and
        # push_recovery_eval.py's --terrain_levels axis see the exact same
        # structured, difficulty-scaled-by-row terrain the policy actually
        # trained against, not a separately-scaled-down grid, and (b) all 6
        # sub-terrain types get proportional column coverage -- a reduced 5-col
        # grid was found to structurally drop one type entirely (see
        # core/terrain_pinning.py's column-assignment math). See REFERENCES.md.

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing events -- push recovery is evaluated separately
        # (policy/locomotion/eval/), not exercised during a play/visualization run
        self.events.base_external_force_torque = None
        self.events.push_disturbance = None
        self.curriculum.push_disturbance_curriculum = None

        # Follow-cam: the default viewer is a fixed world-space camera that
        # never tracks the robot, so footage drifts out of frame once a push
        # (or just plain walking) moves the robot away from its spawn point --
        # and, on Rough, this fixed point sits over one terrain cell regardless
        # of which env/condition is nominally being recorded, which is why
        # showcase clips for different terrain levels/sub-terrain types all
        # looked identical (see REFERENCES.md's camera-framing writeup).
        # asset_root origin makes Isaac Lab recompute eye/lookat from the
        # robot's current root pose every step instead of once. Needs a
        # pod-side smoke test to confirm (not verifiable from source alone --
        # Isaac Lab isn't installed here).
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (-2.5, -2.5, 1.3)
        self.viewer.lookat = (0.0, 0.0, 0.3)
```

- [ ] **Step 4: Delete the old files**

```bash
git rm policy/locomotion/core/go2_flat_env_cfg.py policy/locomotion/core/go2_rough_env_cfg.py
```
(uses `git rm` so the deletion is tracked in the working tree diff; not committing per this plan's global constraint)

- [ ] **Step 5: Verify syntax locally**

```bash
python3 -m py_compile policy/locomotion/core/platforms/go2/platform.py policy/locomotion/core/platforms/go2/env_cfg.py
```
Expected: no output (success). This only checks syntax — `isaaclab_tasks` isn't installed locally, so import-time correctness needs the pod-side check in Task 11.

---

## Task 3: Parameterize push mechanics (`push_disturbance_cfg.py`, `push_recovery_event.py`)

**Files:**
- Modify: `policy/locomotion/core/push_disturbance_cfg.py`
- Modify: `policy/locomotion/eval/push_recovery_event.py`

**Interfaces:**
- Produces: `add_push_disturbance(env_cfg, *, peak_force_n: float, peak_torque_nm: float, root_body_name: str) -> None` (breaking signature change — all 2 call sites are in Task 2's `env_cfg.py`, already written to match). `apply_single_push(env, env_ids, magnitudes, impulse_duration_s, asset_cfg, log_angles=False) -> None` (drops the hardcoded default for `asset_cfg`, making it required — all existing call sites in `policy/locomotion/eval/*.py` already pass it explicitly, so this is a non-breaking cleanup, verified by the grep in Task 11).

- [ ] **Step 1: Modify `push_disturbance_cfg.py`**

Replace the module-level constants and `add_push_disturbance` function. Current content (module docstring + `SPARSE_INTERVAL_RANGE_S`/`DENSE_INTERVAL_RANGE_S`/`INTERVAL_SATURATION_K_C`/`IMPULSE_DURATION_S` and their comments) stays unchanged — only `PEAK_FORCE_N`/`PEAK_TORQUE_NM` and the function signature change:

Remove these two lines (and their preceding comment block, which moves to `go2/platform.py` in Task 2):
```python
PEAK_FORCE_N = 120.0
PEAK_TORQUE_NM = 18.0  # ESTIMATE -- see REFERENCES.md
```

Replace the function:
```python
def add_push_disturbance(
    env_cfg: ManagerBasedRLEnvCfg,
    *,
    peak_force_n: float,
    peak_torque_nm: float,
    root_body_name: str,
) -> None:
    """Adds the push_disturbance EventTerm (+ its clear pair) and curriculum to env_cfg
    in place. `peak_force_n`/`peak_torque_nm`/`root_body_name` are platform-specific --
    see core/platforms/registry.py's PlatformCfg and REFERENCES.md for how Go2's
    values were derived."""
    env_cfg.events.push_disturbance = EventTerm(
        func=apply_push_impulse,
        mode="interval",
        interval_range_s=SPARSE_INTERVAL_RANGE_S,
        params={
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
            "impulse_duration_s": IMPULSE_DURATION_S,
            "asset_cfg": SceneEntityCfg("robot", body_names=root_body_name),
        },
    )
    # Fires every env.step() (degenerate step_dt interval) to zero out any push whose
    # IMPULSE_DURATION_S has elapsed -- see push_impulse_event.py.
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.events.push_disturbance_clear = EventTerm(
        func=clear_expired_push_impulses,
        mode="interval",
        interval_range_s=(step_dt, step_dt),
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

- [ ] **Step 2: Modify `push_recovery_event.py`**

Change the `apply_single_push` signature — remove the hardcoded default:

Old:
```python
def apply_single_push(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    magnitudes: torch.Tensor,
    impulse_duration_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    log_angles: bool = False,
) -> None:
```

New:
```python
def apply_single_push(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    magnitudes: torch.Tensor,
    impulse_duration_s: float,
    asset_cfg: SceneEntityCfg,
    log_angles: bool = False,
) -> None:
```

- [ ] **Step 3: Verify syntax locally**

```bash
python3 -m py_compile policy/locomotion/core/push_disturbance_cfg.py policy/locomotion/eval/push_recovery_event.py
```
Expected: no output (success).

---

## Task 4: Simplify `policy/locomotion/__init__.py`

**Files:**
- Modify: `policy/locomotion/__init__.py`

**Interfaces:**
- Consumes: `policy.locomotion.core.platforms.go2.platform` (Task 2) — importing it triggers `register_platform(GO2)`, which registers all four gym task ids exactly as the four hand-written `gym.register()` calls did before.

- [ ] **Step 1: Replace the file contents**

`policy/locomotion/__init__.py`:
```python
"""Cerberus locomotion task registration. Importing this module registers
every known robot platform's gym environments -- add a platform by adding one
import line here (see core/platforms/registry.py for what a platform must
define), not by editing this file's logic.
"""

from policy.locomotion.core.platforms.go2 import platform as _go2_platform  # noqa: F401
```

- [ ] **Step 2: Verify syntax locally**

```bash
python3 -m py_compile policy/locomotion/__init__.py
```
Expected: no output (success).

---

## Task 5: `--platform`/`--terrain` CLI helper in `cli_common.py`

**Files:**
- Modify: `policy/cli_common.py`

**Interfaces:**
- Produces: `add_platform_args(parser: argparse.ArgumentParser) -> None` (adds `--platform`, default `"go2"`; and `--terrain`, choices `["flat", "rough"]`, no default set here — callers pass their own per-script default via `parser.set_defaults(terrain=...)` after calling this, matching the existing `add_common_args` pattern where each script supplies its own `default_task`). `resolve_task_id(args_cli: argparse.Namespace, *, play: bool) -> str` (returns `args_cli.task` unchanged if the user passed one explicitly and it's not `None`; otherwise looks up the platform via `get_platform(args_cli.platform)` and returns the matching task id for `args_cli.terrain` and `play`). Both are used by Task 6 (`train.py`, `play.py`).
- Note: `get_platform`/`available_platforms` come from `policy.locomotion.core.platforms.registry`, which only needs `gymnasium` (not Isaac Lab) to import — but the registry is only *populated* once something imports a platform's `platform.py` module (e.g. via `policy.locomotion`), which itself needs `isaaclab_tasks` and so must stay deferred until after `AppLauncher` launches, per this file's own existing module docstring constraint. `add_platform_args` (pure argparse, called before `AppLauncher`) is therefore safe to add here; `resolve_task_id` must only be called by callers *after* they've imported `policy.locomotion`, i.e. from the "Rest everything follows" section of `train.py`/`play.py`, not from this module's own top-level code.

- [ ] **Step 1: Add the two new functions to `cli_common.py`**

Insert after `add_common_args` (before `apply_config_preset`):
```python
def add_platform_args(parser: argparse.ArgumentParser) -> None:
    """Adds --platform (default 'go2') and --terrain (choices: flat, rough,
    no default -- caller sets one via parser.set_defaults(terrain=...) after
    calling this, matching add_common_args' per-script default_task pattern).
    """
    parser.add_argument(
        "--platform",
        type=str,
        default="go2",
        help="Robot platform to train/play (see policy/locomotion/core/platforms/).",
    )
    parser.add_argument(
        "--terrain",
        type=str,
        choices=["flat", "rough"],
        help="Which of the platform's env variants to use.",
    )


def resolve_task_id(args_cli: argparse.Namespace, *, play: bool) -> str:
    """Resolves the gym task id to use: args_cli.task if the caller passed
    one explicitly (escape hatch, bypasses --platform/--terrain entirely),
    otherwise looked up from the platform registry via --platform/--terrain.

    Must be called only after `policy.locomotion` (or any other module that
    imports a platform's platform.py) has already been imported -- populating
    the registry needs isaaclab_tasks, which import_common's own callers only
    import after AppLauncher launches. See this function's docstring in
    cli_common.py's own module docstring for why that ordering matters.
    """
    if args_cli.task is not None:
        return args_cli.task

    from policy.locomotion.core.platforms.registry import get_platform

    platform = get_platform(args_cli.platform)
    if args_cli.terrain == "flat":
        return platform.flat_play_task_id if play else platform.flat_task_id
    return platform.rough_play_task_id if play else platform.rough_task_id
```

- [ ] **Step 2: Change `add_common_args`'s `--task` default to `None`**

Old signature and body (the `default_task`/`task_help` parameters and the `--task` line):
```python
def add_common_args(parser: argparse.ArgumentParser, default_task: str, task_help: str, video_help: str) -> None:
    """Adds the CLI args identical across train.py/play.py: --video,
    --video_length, --num_envs, --task, --agent, --seed, --config. Each
    script adds its own additional args after calling this.
    """
    parser.add_argument("--video", action="store_true", default=False, help=video_help)
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=default_task, help=task_help)
```

New:
```python
def add_common_args(parser: argparse.ArgumentParser, task_help: str, video_help: str) -> None:
    """Adds the CLI args identical across train.py/play.py: --video,
    --video_length, --num_envs, --task, --agent, --seed, --config. Each
    script adds its own additional args after calling this.

    --task has no default here (None) -- pass it explicitly to force a
    specific gym task id and bypass --platform/--terrain resolution
    (add_platform_args/resolve_task_id), which is the normal path.
    """
    parser.add_argument("--video", action="store_true", default=False, help=video_help)
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help=task_help)
```

(`default_task` is dropped from the signature entirely — Task 6 updates both call sites in `train.py`/`play.py` to match.)

- [ ] **Step 3: Verify syntax locally**

```bash
python3 -m py_compile policy/cli_common.py
```
Expected: no output (success).

---

## Task 6: Apply `--platform`/`--terrain` to `train.py` and `play.py`

**Files:**
- Modify: `policy/locomotion/train.py`
- Modify: `policy/locomotion/play.py`

**Interfaces:**
- Consumes: `add_common_args` (new signature, Task 5), `add_platform_args`, `resolve_task_id` from `policy.cli_common`.

- [ ] **Step 1: Update `train.py`'s argparse block**

Old:
```python
from policy.cli_common import add_common_args, apply_config_preset, check_gpu_driver_for_rendering  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train the Cerberus Go2 locomotion policy with RSL-RL.")
add_common_args(
    parser,
    default_task="Isaac-Velocity-Flat-Unitree-Go2-Cerberus-v0",
    task_help="Name of the task (override to the -Rough- variant for step 3's terrain curriculum).",
    video_help="Record videos during training.",
)
```

New:
```python
from policy.cli_common import (  # isort: skip
    add_common_args,
    add_platform_args,
    apply_config_preset,
    check_gpu_driver_for_rendering,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train the Cerberus locomotion policy with RSL-RL.")
add_common_args(
    parser,
    task_help="Raw gym task id override -- bypasses --platform/--terrain entirely.",
    video_help="Record videos during training.",
)
add_platform_args(parser)
parser.set_defaults(terrain="flat")
```

- [ ] **Step 2: Resolve `args_cli.task` in `train.py`'s "Rest everything follows" section**

Insert right after `import policy.locomotion  # noqa: F401  -- registers the Cerberus Go2 tasks` and before `from isaaclab_tasks.utils import get_checkpoint_path`:
```python
from policy.cli_common import resolve_task_id  # isort: skip

args_cli.task = resolve_task_id(args_cli, play=False)
```

- [ ] **Step 3: Update `play.py`'s argparse block**

Old:
```python
from policy.cli_common import add_common_args, apply_config_preset, check_gpu_driver_for_rendering  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained Cerberus Go2 locomotion checkpoint with RSL-RL.")
add_common_args(
    parser,
    default_task="Isaac-Velocity-Flat-Unitree-Go2-Cerberus-Play-v0",
    task_help="Name of the task (override to the -Rough- variant to play a rough-terrain checkpoint).",
    video_help="Record videos during play.",
)
```

New:
```python
from policy.cli_common import (  # isort: skip
    add_common_args,
    add_platform_args,
    apply_config_preset,
    check_gpu_driver_for_rendering,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained Cerberus locomotion checkpoint with RSL-RL.")
add_common_args(
    parser,
    task_help="Raw gym task id override -- bypasses --platform/--terrain entirely.",
    video_help="Record videos during play.",
)
add_platform_args(parser)
parser.set_defaults(terrain="flat")
```

- [ ] **Step 4: Resolve `args_cli.task` in `play.py`'s "Rest everything follows" section**

Insert right after `import policy.locomotion  # noqa: F401  -- registers the Cerberus Go2 tasks` and before `from isaaclab_tasks.utils import get_checkpoint_path`:
```python
from policy.cli_common import resolve_task_id  # isort: skip

args_cli.task = resolve_task_id(args_cli, play=True)
```

- [ ] **Step 5: Verify syntax locally**

```bash
python3 -m py_compile policy/locomotion/train.py policy/locomotion/play.py
```
Expected: no output (success). Full behavioral verification (does `@hydra_task_config(args_cli.task, args_cli.agent)` actually pick up the resolved task id correctly) needs the pod-side check in Task 11 — this decorator evaluates at class-definition time, after both inserted resolution blocks above, so ordering is correct, but only a real Isaac Lab environment can confirm the decorator behaves as expected end to end.

---

## Task 7: `--platform` in `push_recovery_eval.py`

**Files:**
- Modify: `policy/locomotion/eval/push_recovery_eval.py`

**Interfaces:**
- Consumes: `add_platform_args`, `resolve_task_id` from `policy.cli_common`; `get_platform` from `policy.locomotion.core.platforms.registry`.

This script builds its own argparse block directly (does not call `add_common_args`), and hardcodes `SceneEntityCfg("robot", body_names="base")` at two call sites (lines 153, 164 per the current file). Both change to use the resolved platform's `root_body_name`.

- [ ] **Step 1: Change the `--task` argument and add `--platform`**

Old:
```python
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-Unitree-Go2-Cerberus-Play-v0",
    help="Base task to build the eval env from (its Play variant's randomization-off settings apply).",
)
```

New:
```python
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Raw gym task id override -- bypasses --platform/--terrain. Its Play variant's "
    "randomization-off settings apply.",
)
```

Add, right after the `--agent` argument:
```python
from policy.cli_common import add_platform_args  # isort: skip

add_platform_args(parser)
parser.set_defaults(terrain="flat")
```
(this needs `add_platform_args` imported near the top with the other `cli_args` import — add `from policy.cli_common import add_platform_args` alongside the existing `import cli_args  # isort: skip` line, not inline mid-file as shown above; the snippet groups them together here for clarity but place the import at the top of the file and the two calls right after the `--agent` argparse block.)

- [ ] **Step 2: Resolve `args_cli.task` before the `@hydra_task_config` decorator**

Insert right after `import policy.locomotion  # noqa: F401  -- registers the Cerberus Go2 tasks` and before `from isaaclab_tasks.utils import get_checkpoint_path`:
```python
from policy.cli_common import resolve_task_id  # isort: skip

args_cli.task = resolve_task_id(args_cli, play=True)
```

- [ ] **Step 3: Resolve the platform and use its `root_body_name`**

Insert at the top of `main()`, right after the `magnitudes = ...` / `trials = ...` / `terrain_levels_list = ...` lines and before `cells = ...`:
```python
from policy.locomotion.core.platforms.registry import get_platform

platform = get_platform(args_cli.platform)
```

Change (line ~153):
```python
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
```
to:
```python
            "asset_cfg": SceneEntityCfg("robot", body_names=platform.root_body_name),
```

Change (line ~164):
```python
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
```
to:
```python
        params={"asset_cfg": SceneEntityCfg("robot", body_names=platform.root_body_name)},
```

- [ ] **Step 4: Verify syntax locally**

```bash
python3 -m py_compile policy/locomotion/eval/push_recovery_eval.py
```
Expected: no output (success).

---

## Task 8: `--platform` in `push_recovery_showcase.py` and `push_recovery_showcase_flat.py`

**Files:**
- Modify: `policy/locomotion/eval/push_recovery_showcase.py`
- Modify: `policy/locomotion/eval/push_recovery_showcase_flat.py`

**Interfaces:**
- Consumes: same as Task 7 (`add_platform_args`, `resolve_task_id`, `get_platform`).

Both files have an identical shape: a `--task` default, a `--agent` argument, then `@hydra_task_config` decorated `main()` with two `SceneEntityCfg("robot", body_names="base")` call sites (an `EventTerm` params dict, and a standalone `asset_cfg = SceneEntityCfg(...)` used for the manual `apply_single_push` call).

- [ ] **Step 1: In `push_recovery_showcase.py`, change `--task` and add `--platform`**

Old:
```python
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0",
    help="Base task to build the showcase env from (must have a terrain-difficulty grid -- Rough only).",
)
```

New:
```python
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Raw gym task id override -- bypasses --platform. Must have a terrain-difficulty grid "
    "(Rough only).",
)
```

Add `from policy.cli_common import add_platform_args` to the existing `cli_common` import line (currently `from policy.cli_common import check_gpu_driver_for_rendering  # isort: skip` — change to `from policy.cli_common import add_platform_args, check_gpu_driver_for_rendering  # isort: skip`), and right after the `--agent` argparse block add:
```python
add_platform_args(parser)
parser.set_defaults(terrain="rough")
```

- [ ] **Step 2: Resolve `args_cli.task` before the `@hydra_task_config` decorator**

Insert right after `import policy.locomotion  # noqa: F401  -- registers the Cerberus Go2 tasks` and before `from isaaclab_tasks.utils import get_checkpoint_path`:
```python
from policy.cli_common import resolve_task_id  # isort: skip

args_cli.task = resolve_task_id(args_cli, play=True)
```

- [ ] **Step 3: Resolve the platform and use its `root_body_name`, both call sites**

Insert at the top of `main()`, right after `magnitudes = [float(m) for m in args_cli.magnitudes.split(",")]`:
```python
from policy.locomotion.core.platforms.registry import get_platform

platform = get_platform(args_cli.platform)
```

Change:
```python
    env_cfg.events.push_showcase_clear = EventTerm(
        func=clear_expired_push_impulses,
        mode="interval",
        interval_range_s=(step_dt, step_dt),
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
```
to:
```python
    env_cfg.events.push_showcase_clear = EventTerm(
        func=clear_expired_push_impulses,
        mode="interval",
        interval_range_s=(step_dt, step_dt),
        params={"asset_cfg": SceneEntityCfg("robot", body_names=platform.root_body_name)},
    )
```

Change:
```python
    asset_cfg = SceneEntityCfg("robot", body_names="base")
```
to:
```python
    asset_cfg = SceneEntityCfg("robot", body_names=platform.root_body_name)
```

- [ ] **Step 4: Repeat steps 1-3 for `push_recovery_showcase_flat.py`**, with two differences:
  - Its `--task` default was `"Isaac-Velocity-Flat-Unitree-Go2-Cerberus-Play-v0"` (Flat, not Rough) — after the change it's still `default=None`, but the help text says "Flat only" and the `parser.set_defaults(terrain=...)` call uses `terrain="flat"` instead of `"rough"`.
  - It has the same two `SceneEntityCfg("robot", body_names="base")` call sites (the `push_showcase_clear` EventTerm and the standalone `asset_cfg = ...`) — apply the identical `platform.root_body_name` substitution to both.

- [ ] **Step 5: Verify syntax locally**

```bash
python3 -m py_compile policy/locomotion/eval/push_recovery_showcase.py policy/locomotion/eval/push_recovery_showcase_flat.py
```
Expected: no output (success).

---

## Task 9: `--platform` in `play_training_progression.py` and `play_terrain_showcase.py`

**Files:**
- Modify: `policy/locomotion/play_training_progression.py`
- Modify: `policy/locomotion/play_terrain_showcase.py`

**Interfaces:**
- Consumes: `add_platform_args`, `resolve_task_id` from `policy.cli_common`.

Neither script hardcodes `body_names="base"` (no push mechanics here — pure walking/terrain showcase), so this task is `--task`/`--platform` resolution only, no `root_body_name` threading needed.

- [ ] **Step 1: In `play_training_progression.py`, change `--task` and add `--platform`**

Old:
```python
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0",
    help="Base task to build the env from.",
)
```

New:
```python
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Raw gym task id override -- bypasses --platform (this script is Rough-only).",
)
```

Change the existing `from policy.cli_common import check_gpu_driver_for_rendering  # isort: skip` line to:
```python
from policy.cli_common import add_platform_args, check_gpu_driver_for_rendering  # isort: skip
```

Add, right after the `--agent` argparse block:
```python
add_platform_args(parser)
parser.set_defaults(terrain="rough")
```

- [ ] **Step 2: Resolve `args_cli.task` before the `@hydra_task_config` decorator**

Insert right after `import policy.locomotion  # noqa: F401  -- registers the Cerberus Go2 tasks` and before `from isaaclab_tasks.utils.hydra import hydra_task_config`:
```python
from policy.cli_common import resolve_task_id  # isort: skip

args_cli.task = resolve_task_id(args_cli, play=True)
```

- [ ] **Step 3: Repeat steps 1-2 for `play_terrain_showcase.py`**, same substitutions (its `--task` default is already `"Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0"`, becomes `None`; `terrain="rough"`; same import-line change; same insertion point after `import policy.locomotion`).

- [ ] **Step 4: Verify syntax locally**

```bash
python3 -m py_compile policy/locomotion/play_training_progression.py policy/locomotion/play_terrain_showcase.py
```
Expected: no output (success).

---

## Task 10: Update `docs/project_structure.md`

**Files:**
- Modify: `docs/project_structure.md`

- [ ] **Step 1: Read the current checkpoint-naming section**

```bash
sed -n '195,220p' docs/project_structure.md
```

- [ ] **Step 2: Add a note that `checkpoint_prefix` is now enforced, not just conventional**

Find the line `## Checkpoint versioning within \`policy/locomotion/checkpoints/\`` and insert immediately after it (before the existing `- go2_locomotion_v1_candidate.pt ...` bullet list):
```markdown

The `go2_` prefix below comes from `GO2.checkpoint_prefix` in
`policy/locomotion/core/platforms/go2/platform.py` (defaults to
`"<platform.name>_locomotion"`, i.e. `go2_locomotion`) -- a new platform gets
its own prefix automatically from its own `PlatformCfg.name`, not a
convention someone has to remember to follow by hand.
```

- [ ] **Step 3: Verify the edit**

```bash
grep -n "checkpoint_prefix" docs/project_structure.md
```
Expected: one match, the new paragraph just added.

---

## Task 11: Repo-wide verification pass

**Files:** none (verification only)

- [ ] **Step 1: Confirm no remaining references to the deleted module paths**

```bash
grep -rn "core\.go2_rough_env_cfg\|core\.go2_flat_env_cfg\|core/go2_rough_env_cfg\|core/go2_flat_env_cfg" --include="*.py" --include="*.md" .
```
Expected: no output. If anything matches (e.g. a stray doc reference), update it to point at `core/platforms/go2/env_cfg.py`.

- [ ] **Step 2: Confirm no remaining references to the removed module constants**

```bash
grep -rn "PEAK_FORCE_N\|PEAK_TORQUE_NM" --include="*.py" policy/
```
Expected: no matches (both are now `GO2.push_peak_force_n`/`GO2.push_peak_torque_nm` fields, referenced only from `platform.py`'s own definition — this grep should show nothing, since the names themselves no longer exist anywhere).

- [ ] **Step 3: Confirm no remaining hardcoded `body_names="base"` outside `platform.py`'s `GO2.root_body_name` field itself**

```bash
grep -rn 'body_names="base"' --include="*.py" policy/
```
Expected: no output — every call site should now read `body_names=platform.root_body_name` or `body_names=GO2.root_body_name` (the latter only inside `push_disturbance_cfg.py`'s callers in `env_cfg.py`, which pass it as a parameter, not a literal string).

- [ ] **Step 4: Byte-compile every touched file**

```bash
python3 -m py_compile \
  policy/locomotion/core/platforms/registry.py \
  policy/locomotion/core/platforms/go2/platform.py \
  policy/locomotion/core/platforms/go2/env_cfg.py \
  policy/locomotion/__init__.py \
  policy/locomotion/core/push_disturbance_cfg.py \
  policy/locomotion/eval/push_recovery_event.py \
  policy/cli_common.py \
  policy/locomotion/train.py \
  policy/locomotion/play.py \
  policy/locomotion/eval/push_recovery_eval.py \
  policy/locomotion/eval/push_recovery_showcase.py \
  policy/locomotion/eval/push_recovery_showcase_flat.py \
  policy/locomotion/play_training_progression.py \
  policy/locomotion/play_terrain_showcase.py
```
Expected: no output (success) — confirms every file is at least syntactically valid Python. This is NOT a substitute for actually running these scripts; it only catches typos/syntax errors, not import-time or runtime issues that need real `isaaclab`/`isaaclab_tasks`.

- [ ] **Step 5: Document the pod-side verification this plan cannot run**

This plan does not start or use any RunPod pod (per this plan's global constraints). Before trusting this refactor for a real training/eval run, the following must be checked on a pod with Isaac Lab installed, and are **not** performed as part of this plan:

1. `python -c "import policy.locomotion; import gymnasium as gym; print([k for k in gym.registry if 'Go2' in k])"` — confirms all four Go2 gym ids still register without error (same four ids as before this refactor).
2. `python policy/locomotion/train.py --terrain rough --max_iterations 10 --logger tensorboard` (no `--task` passed) — confirms `--platform go2 --terrain rough` resolves to the correct task id and a short training run behaves identically to the pre-refactor `Isaac-Velocity-Rough-Unitree-Go2-Cerberus-v0` task (same reward curve shape, same push-disturbance curriculum values in the dumped `params/env.yaml`).
3. A quick run of `push_recovery_eval.py --terrain_levels 0 --trials_per_cell 5` — confirms the `platform.root_body_name` substitution correctly resolves `SceneEntityCfg("robot", body_names="base")` (an error here would surface as a scene-entity resolution failure, not a silent misbehavior, per Isaac Lab's own `SceneEntityCfg` validation).
4. Diff `params/env.yaml` from a pre-refactor run against a post-refactor run at the same iteration count — confirms push force/torque/interval and terrain step-height values are bit-for-bit identical (this refactor must not change Go2's trained behavior, only relocate the values).

Leave this as a checklist for whoever next has pod access, rather than executing it now.

---

## Self-Review Notes

- **Spec coverage:** all 8 migration steps from the design spec (`docs/superpowers/specs/2026-09-08-platform-decoupling-design.md`) are covered: registry (Task 1), Go2 move (Task 2), push_disturbance_cfg signature (Task 3), `__init__.py` (Task 4), CLI flags across train/play/eval/showcase (Tasks 5-9), docs update (Task 10), verification grep (Task 11 steps 1-3) plus the pod-side checklist the spec flagged as unverifiable locally (Task 11 step 5).
- **Extra beyond the spec, found while reading actual files:** `push_recovery_event.py`'s `apply_single_push` had its own independent hardcoded `body_names="base"` default parameter, not mentioned in the original spec (which only called out `push_disturbance_cfg.py`'s two call sites) — added to Task 3. Three eval/showcase scripts (`push_recovery_eval.py`, `push_recovery_showcase.py`, `push_recovery_showcase_flat.py`) also had their own hardcoded `body_names="base"` call sites independent of `add_push_disturbance()` — added as Tasks 7-8.
- **Type/signature consistency:** `add_push_disturbance`'s new keyword-only signature (Task 3) matches its two call sites written in Task 2's `env_cfg.py`. `add_common_args`'s dropped `default_task` parameter (Task 5) is reflected in both its call sites (Task 6). `resolve_task_id(args_cli, *, play: bool)` is called consistently across Tasks 6-9 with `play=False` only in `train.py`, `play=True` everywhere else (all of which build Play-variant envs).
- **No placeholders:** every step shows complete, real code (verified against the actual current file contents read from disk this session, not reconstructed from memory).
