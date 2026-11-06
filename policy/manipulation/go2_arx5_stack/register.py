"""Gymnasium registration of the W2a Go2 + ARX5 stack tasks. Pure python: gymnasium is imported lazily.

Mirrors how `isaaclab_mimic/envs/__init__.py` (Isaac Lab `v3.0.0-EA`) registers the Franka ids: string entry points,
so no Isaac module is imported until `gym.make()` resolves them (after the app launched).

Isaac Lab's `record_demos.py`, `annotate_demos.py`, `generate_dataset.py` and `replay_demos.py` accept
`--external_callback <dotted.path>`; call `register_all` through it (repo root on PYTHONPATH):

    --external_callback policy.manipulation.go2_arx5_stack.register.register_all

`register_all` returns None on purpose: those scripts intersect the leftover Hydra args with the callback's return
value when it is not None, which would drop them.
"""

from __future__ import annotations

PACKAGE = "policy.manipulation.go2_arx5_stack"

TASK_ID_IK_REL = "Cerberus-Go2Arx5-Stack-Cube-IK-Rel-v0"
TASK_ID_MIMIC = "Cerberus-Go2Arx5-Stack-Cube-IK-Rel-Mimic-v0"

ENV_CFG_ENTRY_POINT = f"{PACKAGE}.env_cfg:Go2Arx5StackEnvCfg"
MIMIC_ENV_ENTRY_POINT = f"{PACKAGE}.mimic_env:Go2Arx5StackMimicEnv"
MIMIC_ENV_CFG_ENTRY_POINT = f"{PACKAGE}.mimic_env_cfg:Go2Arx5StackMimicEnvCfg"

# (id, entry_point, env_cfg_entry_point)
TASKS: tuple[tuple[str, str, str], ...] = (
    (TASK_ID_IK_REL, "isaaclab.envs:ManagerBasedRLEnv", ENV_CFG_ENTRY_POINT),
    (TASK_ID_MIMIC, MIMIC_ENV_ENTRY_POINT, MIMIC_ENV_CFG_ENTRY_POINT),
)


def task_ids() -> tuple[str, ...]:
    return tuple(task_id for task_id, _, _ in TASKS)


def register_all(gym_module=None) -> None:
    """Register both task ids with gymnasium; safe to call any number of times.

    Args:
        gym_module: gymnasium-like module (needs `register`, and optionally a `registry` mapping of ids). Defaults
            to `import gymnasium`, done here and not at module level.
    """
    if gym_module is None:
        import gymnasium as gym_module

    # gymnasium exposes the id -> spec mapping as `gymnasium.registry` (also `gymnasium.envs.registry`).
    registry = getattr(gym_module, "registry", None)
    if registry is None:
        registry = getattr(getattr(gym_module, "envs", None), "registry", None)
    for task_id, entry_point, cfg_entry_point in TASKS:
        if registry is not None and task_id in registry:
            continue
        gym_module.register(
            id=task_id,
            entry_point=entry_point,
            kwargs={"env_cfg_entry_point": cfg_entry_point},
            disable_env_checker=True,
        )
