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
    def test_upright_wxyz_robot_passes(self):
        out = probe.evaluate([[0.0, 0.0, -1.0]] * 2, [[1.0, 0.0, 0.0, 0.0]] * 2)
        assert out["passed"] is True
        assert out["quat_convention"] == "wxyz"
        assert out["projected_gravity_ok"] is True

    def test_pitched_xyzw_robot_identifies_its_order(self):
        # A pitched robot's gravity vector is predicted differently by the two orders, so this is the
        # case that actually answers "which order does root_quat_w use on this stack?".
        import math as _m

        tilt = _m.radians(20.0)
        quat = [0.0, _m.sin(tilt / 2), 0.0, _m.cos(tilt / 2)]  # xyzw, pitched about y
        pg = probe.predicted_projected_gravity(quat, "xyzw")
        out = probe.evaluate([pg], [quat])
        assert out["quat_convention"] == "xyzw"
        assert out["convention_discriminable"] is True
        assert out["passed"] is True

    def test_pitched_wxyz_robot_identifies_its_order(self):
        import math as _m

        tilt = _m.radians(20.0)
        quat = [_m.cos(tilt / 2), 0.0, _m.sin(tilt / 2), 0.0]  # wxyz, pitched about y
        pg = probe.predicted_projected_gravity(quat, "wxyz")
        out = probe.evaluate([pg], [quat])
        assert out["quat_convention"] == "wxyz"
        assert out["passed"] is True

    def test_tilted_but_upright_robot_passes(self):
        # A policy-less robot sags: ~14 degrees of tilt must not be mistaken for a convention error.
        import math as _m

        tilt = _m.radians(14.0)
        pg = [_m.sin(tilt), 0.0, -_m.cos(tilt)]
        quat = [0.0, _m.sin(tilt / 2), 0.0, _m.cos(tilt / 2)]
        out = probe.evaluate([pg], [quat])
        assert out["passed"] is True

    def test_gravity_not_pointing_down_fails(self):
        out = probe.evaluate([[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0]])
        assert out["projected_gravity_ok"] is False
        assert out["passed"] is False

    def test_sideways_gravity_fails(self):
        out = probe.evaluate([[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]])
        assert out["passed"] is False

    def test_reports_the_worst_env(self):
        out = probe.evaluate([[0.0, 0.0, -1.0], [0.0, 0.0, 0.9]], [[1.0, 0.0, 0.0, 0.0]] * 2)
        assert out["projected_gravity_worst_env"] == 1

    def test_mismatched_row_counts_rejected(self):
        try:
            probe.evaluate([[0.0, 0.0, -1.0]], [])
        except ValueError as e:
            assert "differ" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


class TestQuatMath:
    def test_identity_predicts_straight_down(self):
        assert probe.predicted_projected_gravity([1, 0, 0, 0], "wxyz") == [0.0, 0.0, -1.0]

    def test_orders_describe_the_same_rotation(self):
        import math as _m

        tilt = _m.radians(25.0)
        xyzw = [0.0, _m.sin(tilt / 2), 0.0, _m.cos(tilt / 2)]
        wxyz = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
        a = probe.predicted_projected_gravity(xyzw, "xyzw")
        b = probe.predicted_projected_gravity(wxyz, "wxyz")
        assert all(abs(x - y) < 1e-9 for x, y in zip(a, b, strict=True))

    def test_rejects_zero_quaternion(self):
        try:
            probe.quat_to_matrix([0, 0, 0, 0], "xyzw")
        except ValueError as e:
            assert "zero quaternion" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")

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
