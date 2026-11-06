from __future__ import annotations

from types import SimpleNamespace

from common.cerberus_compat.physics import pin_physx


def _lab3_sim():
    physics = SimpleNamespace(isaacsim_physx="PHYSX", physx="AUTO", newton_mjwarp="NEWTON", default="NEWTON")
    return SimpleNamespace(physics=physics, use_newton_actuators=True)


def test_lab3_pins_physx_and_legacy_actuators():
    sim = _lab3_sim()
    info = pin_physx(sim)
    assert sim.physics.default == "PHYSX"
    assert sim.use_newton_actuators is False
    assert info == {"lab3": True, "physics_preset": "isaacsim_physx", "use_newton_actuators": False}


def test_falls_back_to_physx_auto_preset():
    sim = _lab3_sim()
    del sim.physics.isaacsim_physx
    assert pin_physx(sim)["physics_preset"] == "physx"
    assert sim.physics.default == "AUTO"


def test_idempotent():
    sim = _lab3_sim()
    pin_physx(sim)
    assert pin_physx(sim) == {"lab3": True, "physics_preset": "isaacsim_physx", "use_newton_actuators": False}
    assert sim.physics.default == "PHYSX"


def test_lab2_is_a_noop():
    sim = SimpleNamespace(physx=SimpleNamespace())  # no `physics` preset, no use_newton_actuators
    assert pin_physx(sim) == {"lab3": False, "physics_preset": None, "use_newton_actuators": None}
