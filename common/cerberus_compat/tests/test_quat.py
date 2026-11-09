import numpy as np
import pytest

from common.cerberus_compat import quat

WXYZ = np.array([[0.1, 0.2, 0.3, 0.4], [1.0, 0.0, 0.0, 0.0]])
XYZW = np.array([[0.2, 0.3, 0.4, 0.1], [0.0, 0.0, 0.0, 1.0]])


def test_conversions_numpy():
    assert np.array_equal(quat.to_xyzw(WXYZ, "wxyz"), XYZW)
    assert np.array_equal(quat.to_wxyz(XYZW, "xyzw"), WXYZ)
    assert quat.to_wxyz(WXYZ, "wxyz") is WXYZ
    assert quat.to_xyzw(XYZW, "xyzw") is XYZW


def test_round_trip():
    assert np.array_equal(quat.to_wxyz(quat.to_xyzw(WXYZ, "wxyz"), "xyzw"), WXYZ)


def test_tuple_input():
    assert quat.to_xyzw((1.0, 2.0, 3.0, 4.0), "wxyz") == (2.0, 3.0, 4.0, 1.0)
    assert quat.to_wxyz([2.0, 3.0, 4.0, 1.0], "xyzw") == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(ValueError):
        quat.to_wxyz((1, 2, 3), "xyzw")


def test_torch():
    torch = pytest.importorskip("torch")
    t = torch.tensor(WXYZ)
    assert torch.equal(quat.to_xyzw(t, "wxyz"), torch.tensor(XYZW))
    assert torch.equal(quat.to_wxyz(torch.tensor(XYZW), "xyzw"), t)


def test_bad_shape_and_convention():
    with pytest.raises(ValueError):
        quat.to_wxyz(np.zeros((2, 3)), "xyzw")
    with pytest.raises(ValueError):
        quat.to_wxyz(WXYZ, "abcd")


@pytest.mark.parametrize("conv", ["wxyz", "xyzw"])
def test_from_to_isaac_with_override(monkeypatch, conv):
    monkeypatch.setenv(quat.ENV_OVERRIDE, conv)
    assert quat.isaac_quat_convention() == conv
    native = WXYZ if conv == "wxyz" else XYZW
    assert np.array_equal(quat.from_isaac(native), WXYZ)
    assert np.array_equal(quat.to_isaac(WXYZ), native)
    assert np.array_equal(quat.to_isaac(quat.from_isaac(native)), native)


def test_invalid_override(monkeypatch):
    monkeypatch.setenv(quat.ENV_OVERRIDE, "zzzz")
    with pytest.raises(ValueError):
        quat.isaac_quat_convention()


@pytest.mark.parametrize(
    "version,expected",
    [("2.3.2", "wxyz"), ("3.0.0", "xyzw"), ("3.0.0rc1", "xyzw"), ("garbage", "wxyz")],
)
def test_version_detection(monkeypatch, version, expected):
    monkeypatch.delenv(quat.ENV_OVERRIDE, raising=False)
    monkeypatch.setattr(quat.metadata, "version", lambda name: version)
    assert quat.isaac_quat_convention() == expected


def test_not_installed(monkeypatch):
    monkeypatch.delenv(quat.ENV_OVERRIDE, raising=False)

    def missing(name):
        raise quat.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(quat.metadata, "version", missing)
    assert quat.isaac_quat_convention() == "wxyz"


def test_projected_gravity_from_quat_matches_the_ga_measurement():
    # Measured on the GA stack (docs/h2_report.md §5): a standing Go2 with spawn yaw 0.
    q_xyzw = [0.0645, -0.0453, 0.0007, 0.9969]
    expected = [-0.090, -0.129, -0.988]
    got = quat.projected_gravity_from_quat(q_xyzw)
    assert np.allclose(got, expected, atol=2e-3), got
    # numpy input and the identity quaternion
    assert np.allclose(quat.projected_gravity_from_quat(np.array(q_xyzw)), expected, atol=2e-3)
    assert quat.projected_gravity_from_quat((0.0, 0.0, 0.0, 1.0)) == (0.0, -0.0, -1.0)
    # a unit vector: gravity is never re-scaled by a normalised input
    assert np.isclose(np.linalg.norm(got), 1.0, atol=1e-6)
    with pytest.raises(ValueError, match="4 components"):
        quat.projected_gravity_from_quat([0.0, 0.0, 1.0])
