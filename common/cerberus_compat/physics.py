"""Pin Isaac Lab's physics backend to PhysX with the legacy actuator path.

Pure python at import time; works on the old (2.x) and the new (3.0) Isaac Lab line.

Why this exists
---------------
On the Lab 3.0 line (verified in ``isaaclab_tasks/core/velocity/config/go2/{flat,rough}_env_cfg.py``
at ``release/3.0.0`` @ dcabdcda) the stock Go2 velocity task sets
``sim.physics.default = newton_mjwarp`` and ``sim.use_newton_actuators = True``. Anything built from it
without an override (``train.py``, ``play.py``, the evals, the SLAM sim) therefore runs on Newton physics
with backend-native actuators, while every Cerberus checkpoint was trained on PhysX with Lab 2.3.2's
actuator path. That mismatch is the prime suspect for the "frozen policy tilts on Lab 3.0" finding.

On Lab 2.x there is no physics preset and PhysX is the only backend, so :func:`pin_physx` is a no-op there.
"""

from __future__ import annotations

from typing import Any

PHYSX_PRESET_NAMES = ("isaacsim_physx", "physx")
"""Preset attributes tried in order. ``isaacsim_physx`` is a concrete ``PhysxCfg``; ``physx`` is the
``PhysxAutoCfg`` wrapper around it (and the OVPhysX variant)."""


def pin_physx(sim_cfg: Any) -> dict[str, Any]:
    """Select PhysX and the legacy actuator execution path on ``sim_cfg`` (an env cfg's ``.sim``).

    Returns what was done: ``{"lab3": bool, "physics_preset": str | None, "use_newton_actuators": bool | None}``.
    Idempotent. Sets ``physics.default`` the same way Lab's own Go2 flat cfg selects Newton, so Hydra preset
    resolution still works and an explicit ``physics=...`` override on the command line still wins.
    """
    physics = getattr(sim_cfg, "physics", None)
    chosen = None
    for name in PHYSX_PRESET_NAMES:
        candidate = getattr(physics, name, None)
        if candidate is not None:
            physics.default = candidate
            chosen = name
            break
    actuators = None
    if hasattr(sim_cfg, "use_newton_actuators"):
        sim_cfg.use_newton_actuators = False
        actuators = False
    return {"lab3": chosen is not None, "physics_preset": chosen, "use_newton_actuators": actuators}


def resolve_physics_preset(sim_cfg: Any) -> dict:
    """Replace a task's physics *preset container* with the preset it selected.

    Lab 3.0 GA's velocity tasks declare physics as a ``PresetCfg`` subclass
    (``RoughPhysicsCfg``) holding ``isaacsim_physx`` / ``newton_mjwarp`` / ... plus ``default``. Hydra
    resolves that container into a concrete backend when it builds a config from the command line.
    Scripts that instantiate a config directly (the stack probe, the checkpoint exporter, the
    adversary trainer, the gate) do not go through Hydra, and ``SimulationContext`` rejects the
    container: "SimulationCfg.physics must be a concrete PhysicsCfg, got RoughPhysicsCfg".

    ``pin_physx`` already chooses the backend by writing ``physics.default``; this applies that
    choice. Idempotent: a concrete config has no ``default`` attribute and is left alone.
    """
    physics = getattr(sim_cfg, "physics", None)
    default = getattr(physics, "default", None)
    if default is None:
        return {"resolved": False, "physics": type(physics).__name__ if physics is not None else None}
    preset_container = type(physics).__name__
    sim_cfg.physics = default
    return {"resolved": True, "preset_container": preset_container, "physics": type(default).__name__}
