"""Pure-python tests for scripts/check_isaac_quat_convention.py (no Isaac, no GPU).

The script itself is pod-only, but its judgement logic (which quaternion slot is the scalar part,
is projected gravity standing-upright, does the cfg resolve to PhysX) is plain python and is the
part that must not be wrong -- a broken verdict would let a Newton or axis-swapped run through P0.
"""

import types

import check_isaac_quat_convention as probe


class TestClassifyQuatOrder:
    def test_wxyz_identity(self):
        out = probe.classify_quat_order([1.0, 0.0, 0.0, 0.0])
        assert out["convention"] == "wxyz"
        assert out["argmax_abs"] == 0
        assert out["argmax_is_plus_one"] is True

    def test_xyzw_identity(self):
        out = probe.classify_quat_order([0.0, 0.0, 0.0, 1.0])
        assert out["convention"] == "xyzw"
        assert out["argmax_abs"] == 3

    def test_negative_scalar_still_identifies_order(self):
        # q and -q are the same rotation; sign must not flip the verdict.
        out = probe.classify_quat_order([-0.999, 0.01, 0.0, 0.0])
        assert out["convention"] == "wxyz"
        assert out["argmax_is_plus_one"] is False

    def test_unexpected_axis_dominance_is_ambiguous(self):
        out = probe.classify_quat_order([0.0, 1.0, 0.0, 0.0])
        assert out["convention"] == "ambiguous"

    def test_rejects_wrong_length(self):
        try:
            probe.classify_quat_order([1.0, 0.0, 0.0])
        except ValueError as e:
            assert "4 quaternion components" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


class TestEvaluate:
    def test_upright_robot_passes(self):
        out = probe.evaluate([[0.0, 0.0, -1.0]] * 2, [[1.0, 0.0, 0.0, 0.0]] * 2)
        assert out["passed"] is True
        assert out["quat_convention"] == "wxyz"
        assert out["projected_gravity_max_abs_error"] == 0.0

    def test_axis_swap_fails(self):
        out = probe.evaluate([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], [[1.0, 0.0, 0.0, 0.0]] * 2)
        assert out["passed"] is False
        assert out["projected_gravity_ok"] is False
        assert out["projected_gravity_worst_env"] == 0

    def test_tolerance_boundary(self):
        inside = probe.evaluate([[0.0, 0.0, -1.0 + probe.PROJECTED_GRAVITY_ATOL]], [[1.0, 0.0, 0.0, 0.0]])
        outside = probe.evaluate([[0.0, 0.0, -1.0 + probe.PROJECTED_GRAVITY_ATOL + 1e-6]], [[1.0, 0.0, 0.0, 0.0]])
        assert inside["projected_gravity_ok"] is True
        assert outside["projected_gravity_ok"] is False

    def test_inconsistent_conventions_fail(self):
        out = probe.evaluate(
            [[0.0, 0.0, -1.0]] * 2, [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        assert out["quat_convention"] == "inconsistent"
        assert out["passed"] is False

    def test_rejects_mismatched_row_counts(self):
        try:
            probe.evaluate([[0.0, 0.0, -1.0]], [])
        except ValueError as e:
            assert "differ" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


def _sim(physics_default, **extra):
    return types.SimpleNamespace(physics=types.SimpleNamespace(default=physics_default), **extra)


class _PhysxCfg:
    pass


class _NewtonCfg:
    pass


class TestPhysicsSummary:
    def test_physx_class_is_recognized(self):
        cfg = types.SimpleNamespace(sim=_sim(_PhysxCfg(), dt=0.005, use_newton_actuators=False), decimation=4)
        out = probe.physics_summary(cfg)
        assert out["is_physx"] is True
        assert out["use_newton_actuators"] is False
        assert out["decimation"] == 4

    def test_resolved_physics_cfg_is_read_directly(self):
        # load_env_cfg replaces Lab 3.0's preset container with the selected backend, so there is no
        # `.default` to read any more -- the config's own physics object is the backend.
        cfg = types.SimpleNamespace(sim=_sim(_PhysxCfg(), dt=0.005, use_newton_actuators=False), decimation=4)
        cfg.sim.physics = _PhysxCfg()  # no .default attribute
        out = probe.physics_summary(cfg)
        assert out["is_physx"] is True
        assert out["physics_default_class"] == "_PhysxCfg"

    def test_newton_class_fails_the_gate(self):
        cfg = types.SimpleNamespace(sim=_sim(_NewtonCfg(), dt=0.005, use_newton_actuators=True), decimation=4)
        out = probe.physics_summary(cfg)
        assert out["is_physx"] is False
        assert out["use_newton_actuators"] is True

    def test_no_physics_preset_is_not_physx(self):
        cfg = types.SimpleNamespace(sim=_sim(None), decimation=4)
        out = probe.physics_summary(cfg)
        assert out["is_physx"] is False
        assert out["physics_default_class"] is None
