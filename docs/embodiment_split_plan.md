# Embodiment split plan: `UNITREE_GO2_CFG` + `ARX5_CFG` → `GO2_ARX5_CFG`

Status: **plan only** (H1, 2026-11-06). Not implemented, because the code it touches imports `isaaclab`, which cannot run
locally. It is part of the next remote handoff.

## Why

`project_structure.md` ("Embodiment configuration pattern") states the target: separable `UNITREE_GO2_CFG` (legs) and
`ARX5_CFG` (arm) objects composed into `GO2_ARX5_CFG`, so an arm or platform swap edits one sub-config. Today:

- `UNITREE_GO2_CFG` is **imported** from `isaaclab_assets.robots.unitree` (`go2_arx5_cfg.py:12`), not defined here.
- There is **no `ARX5_CFG`**. The arm lives as private module state: `_ARM_ACTUATORS` (6 `ImplicitActuatorCfg`, keys
  `arx5_joint1..6`), `_GRIPPER_ACTUATORS` (one group, key `arx5_gripper`), and `ARX5_STOWED_JOINT_POS` folded into
  `init_state.joint_pos` inside `make_go2_arx5_cfg`.
- `ARX5_EMBODIMENT.articulation_cfg_entry_point` already points at the **fused** cfg, with the comment "the arm is only
  ever spawned fused".

## Target shape

```python
# go2_arx5_cfg.py (after the split)
UNITREE_GO2_CFG = ...            # module-local alias, re-exported; see "what may move" #1
ARX5_CFG = ...                   # new: arm + gripper actuators, stowed init_state, no spawn override
GO2_ARX5_CFG = compose_go2_arx5()           # = make_go2_arx5_cfg(fix_base=False)
GO2_ARX5_FIXED_BASE_CFG = compose_go2_arx5(fix_base=True)
```

`ARX5_CFG` is an `ArticulationCfg` **fragment**, not a spawnable robot: the arm has no standalone USD in this repo (the
only asset is the fused `data/assets/go2_arx5/...usd`), so its `spawn.usd_path` stays empty/`None` and the composition
step supplies the fused USD. `ARX5_CFG.actuators`, `ARX5_CFG.init_state.joint_pos` and the arm's spawn-independent fields
are what the composition reads.

## What moves into `ARX5_CFG`

1. **Actuators**: `_ARM_ACTUATORS` and `_GRIPPER_ACTUATORS` move out of module scope and become `ARX5_CFG.actuators`
   until `ARX5_CFG` is built (they keep the exact keys `arx5_joint1..6`, `arx5_gripper` and the exact
   `ImplicitActuatorCfg` kwargs from `_actuator_kwargs`, including the `effort_limit_sim` / `joint_effort_limit`
   selection in `implicit_actuator_kwargs`). `_actuator_kwargs` and the two dict comprehensions move to
   `go2_arx5_cfg.py` module level next to `ARX5_CFG`, or to a new `arx5_cfg.py`; either way `ARX5_CFG` must be built
   from `ARX5_ACTUATOR_GAINS` / `GRIPPER_ACTUATOR_GAINS`, never from copied numbers.
2. **Init state**: `ARX5_CFG.init_state.joint_pos = ARX5_STOWED_JOINT_POS` (stowed arm, gripper open). The Go2's own
   `init_state.joint_pos` stays with the legs.
3. **Joint *groups*** are already separable in `go2_arx5_spec.py`: `ARX5_EMBODIMENT` (`arm`, `gripper`) and
   `GO2_EMBODIMENT` (`legs`); the split must not change them. Only `ARX5_EMBODIMENT.articulation_cfg_entry_point`
   changes, from fused to `...:go2_arx5_cfg:ARX5_CFG` (with a comment that it is a fragment; see "Open questions").

Nothing else moves. In particular `_actuator_kwargs`' effort-field negotiation, `GO2_ARX5_MOUNT_XYZ/RPY`,
`fused_usd_path()` and the `fix_root_link` branch in `make_go2_arx5_cfg` stay exactly as they are: the mount pose is a
composition-time fact about the *fused* asset, not an arm property.

## What must stay byte-identical in the composed cfg

The composed `GO2_ARX5_CFG` / `GO2_ARX5_FIXED_BASE_CFG` must be field-for-field identical to today's, because
`policy/manipulation/go2_arx5_stack/`, `common/cerberus_worlds/arena_assets/go2_arx5_embodiment.py`, the Mimic demos and
every recorded W2a/W2b number are built on it. Specifically:

- `spawn.usd_path` == `str(fused_usd_path())`; `spawn.fix_root_link` (GA) / `articulation_props.fix_root_link` (2.x)
  unchanged for each `fix_base` value.
- `init_state.joint_pos`: same key set and values (`UNITREE_GO2_CFG.init_state.joint_pos` merged with the 8 stowed arm
  entries), same merged mapping content.
- `actuators`: identical key set and, per key, identical `joint_names_expr`, `stiffness`, `damping`, `armature` and
  effort-limit field/value.
- Leg joints, mass, CoM, inertia and limits unchanged (they come from the arm-free `UNITREE_GO2_CFG` spawn + fused USD).
- `go2_arx5_spec.go2_arx5_urdf` / USD pipeline untouched; no URDF or USD is rebuilt by this refactor.

Equality is checkable in Python because both are frozen dataclasses: compare `repr()` (deterministic field order) or
field-by-field with `dataclasses.fields`, recursively for nested cfgs.

## Acceptance check (remote; needs Isaac Lab, no GPU strictly required)

1. **Static**: `pytest -q common/cerberus_description` (the existing AST-based tests must keep passing:
   `test_cfg_module_defines_top_level_name`, `test_go2_arx5_spec.py:284`). Add `ARX5_CFG` to that name set.
2. **Byte-identical composition** (new test, pod-only, skipped without isaaclab): with the app launched,
   `python -c` builds `GO2_ARX5_CFG` from the refactored module and compares `repr()` against a known-good repr
   captured **before** the refactor (commit the fixture under `common/cerberus_description/test_data/`). Both
   `fix_base=False` and `fix_base=True` must match exactly.
3. **Smoke in sim**: `scripts/spawn_hold_go2_arx5.py` on a Lightwheel kitchen (or the stock table scene) — 16 envs,
   assert 20 DoF, arm drift of the order of the recorded `7.6e-5 rad`, gripper open width 0.088 m, root standing
   (the standing check from `arena_spike_go2.py`). Compare against the W2a numbers recorded on 2026-10-05.
4. **Locomotion non-regression**: `policy/locomotion/play.py` on the fused cfg (Flat, `--fall_height_m 0.05`), compare
   standing pitch and push recovery against the 2026-10-05 row in `cerberus_project_context.md`
   (~1.0 at 40/80/120 N, standing pitch ≈ -1.3°).

## Open questions (decide before implementing)

- `ARX5_EMBODIMENT.articulation_cfg_entry_point`: pointing it at a fragment that cannot spawn could mislead
  `registry.get_backend(...).register_embodiment(...)`, which does assume a spawnable cfg. Safer option: keep the entry
  point on the fused cfg and add `ARX5_EMBODIMENT.articulation_fragment_entry_point` (or a `spawnable=False` field) so
  the world layer can never try to spawn the arm alone. Pick one and document it in `worlds.md`.
- Whether `UNITREE_GO2_CFG` becomes a module-local copy (`UNITREE_GO2_CFG.replace(...)`) or stays a re-export of
  `isaaclab_assets` — the former satisfies "defined here" but forks the upstream asset on any upgrade; the latter keeps
  one source of truth. The handoff's target allows either; a re-export plus a comment is the cheaper, safer choice.
- Gripper gains are still provisional (`GRIPPER_ACTUATOR_GAINS`, stiffness 1e3 vs the 5000 N/m used in W2a tuning);
  the split must not change them, but the mismatch stays on the record.
