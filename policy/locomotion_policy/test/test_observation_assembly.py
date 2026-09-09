import numpy as np
import pytest

from locomotion_policy.observation_assembly import ProprioSample, assemble_observation


def _make_proprio(num_joints: int = 12) -> ProprioSample:
    return ProprioSample(
        base_lin_vel=np.array([0.1, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.array([0.0, 0.0, 0.1], dtype=np.float32),
        projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        velocity_commands=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        joint_pos_rel=np.zeros(num_joints, dtype=np.float32),
        joint_vel_rel=np.zeros(num_joints, dtype=np.float32),
        last_action=np.zeros(num_joints, dtype=np.float32),
    )


def test_blind_shape_has_no_height_scan_and_correct_length():
    proprio = _make_proprio(num_joints=12)
    obs = assemble_observation("proprioception_only", proprio, height_scan=None)
    # 3+3+3+3+12+12+12 = 48
    assert obs.shape == (48,)
    assert obs.dtype == np.float32


def test_blind_shape_term_order():
    proprio = _make_proprio(num_joints=12)
    obs = assemble_observation("proprioception_only", proprio, height_scan=None)
    np.testing.assert_array_equal(obs[0:3], proprio.base_lin_vel)
    np.testing.assert_array_equal(obs[3:6], proprio.base_ang_vel)
    np.testing.assert_array_equal(obs[6:9], proprio.projected_gravity)
    np.testing.assert_array_equal(obs[9:12], proprio.velocity_commands)
    np.testing.assert_array_equal(obs[12:24], proprio.joint_pos_rel)
    np.testing.assert_array_equal(obs[24:36], proprio.joint_vel_rel)
    np.testing.assert_array_equal(obs[36:48], proprio.last_action)


def test_perceptive_shape_appends_height_scan_last():
    proprio = _make_proprio(num_joints=12)
    height_scan = np.full(187, 0.3, dtype=np.float32)  # arbitrary grid size for this test
    obs = assemble_observation("proprioception_plus_height_scan", proprio, height_scan=height_scan)
    assert obs.shape == (48 + 187,)
    np.testing.assert_array_equal(obs[48:], height_scan)


def test_perceptive_shape_without_height_scan_raises():
    proprio = _make_proprio(num_joints=12)
    with pytest.raises(ValueError, match="height_scan"):
        assemble_observation("proprioception_plus_height_scan", proprio, height_scan=None)


def test_unknown_shape_raises():
    proprio = _make_proprio(num_joints=12)
    with pytest.raises(ValueError, match="unknown observation_shape"):
        assemble_observation("not_a_real_shape", proprio, height_scan=None)


def test_apply_action_scale_and_offset():
    from locomotion_policy.observation_assembly import apply_action_postprocessing

    raw_network_output = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    default_joint_pos = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    result = apply_action_postprocessing(raw_network_output, default_joint_pos, scale=0.25)
    expected = raw_network_output * 0.25 + default_joint_pos
    np.testing.assert_allclose(result, expected)


def test_build_permutation_identity():
    from locomotion_policy.observation_assembly import _build_permutation

    names = ["a", "b", "c"]
    perm = _build_permutation(names, names)
    arr = np.array([10, 20, 30])
    np.testing.assert_array_equal(arr[perm], arr)


def test_build_permutation_real_permutation():
    from locomotion_policy.observation_assembly import _build_permutation

    from_names = ["a", "b", "c"]
    to_names = ["c", "a", "b"]
    perm = _build_permutation(from_names, to_names)
    arr = np.array([10, 20, 30])  # a=10, b=20, c=30
    np.testing.assert_array_equal(arr[perm], np.array([30, 10, 20]))

    # And the inverse should round-trip back to the original order.
    inverse_perm = _build_permutation(to_names, from_names)
    np.testing.assert_array_equal(arr[perm][inverse_perm], arr)


def test_build_permutation_mismatched_names_raises():
    from locomotion_policy.observation_assembly import _build_permutation

    with pytest.raises(RuntimeError, match="joint name mismatch"):
        _build_permutation(["a", "b", "c"], ["a", "b", "d"])
