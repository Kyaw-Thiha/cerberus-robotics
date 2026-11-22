import pytest

from common.cerberus_policies.base import (
    Boundary,
    CommandShapeError,
    PolicyBackend,
    Role,
    check_command,
    make_loco_command,
)


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        PolicyBackend()


def test_must_implement_load_and_act():
    class Partial(PolicyBackend):
        boundary = Boundary.B1_LOCO
        rate_hz = 10.0

        def load(self, checkpoint_path, embodiment_cfg):
            pass

    with pytest.raises(TypeError):
        Partial()


def test_concrete_subclass_instantiates_and_defaults():
    class Ok(PolicyBackend):
        boundary = Boundary.B2_ARM
        rate_hz = 10

        def load(self, checkpoint_path, embodiment_cfg):
            pass

        def act(self, observation):
            return {}

    assert Ok().role is Role.CONSUMER
    assert list(Ok.required_sensors) == []


@pytest.mark.parametrize(
    "attrs",
    [
        {"rate_hz": 10.0},  # missing boundary
        {"boundary": "B1_LOCO", "rate_hz": 10.0},  # not a Boundary
        {"boundary": Boundary.B1_LOCO},  # missing rate
        {"boundary": Boundary.B1_LOCO, "rate_hz": 0},
        {"boundary": Boundary.B1_LOCO, "rate_hz": 10, "required_sensors": "lidar"},
    ],
)
def test_bad_class_attributes_fail_at_definition(attrs):
    with pytest.raises(TypeError):

        class Bad(PolicyBackend):
            def load(self, checkpoint_path, embodiment_cfg):
                pass

            def act(self, observation):
                return {}

            locals().update(attrs)


def test_fused_backend_may_read_raw_streams():
    class Fused(PolicyBackend):
        boundary = Boundary.B1_LOCO
        rate_hz = 50
        required_sensors = ("lidar", "imu")

        def load(self, checkpoint_path, embodiment_cfg):
            pass

        def act(self, observation):
            return {"n_raw_points": len(observation["sensors"]["lidar"])}

    assert Fused().act({"sensors": {"lidar": [1, 2, 3]}}) == {"n_raw_points": 3}


def test_loco_command_flags_instead_of_nan():
    cmd = make_loco_command(vx=0.5, body_pitch=0.1)
    assert cmd["has_body_pitch"] is True and cmd["has_body_height"] is False
    assert cmd["body_height"] == 0.0
    check_command(Boundary.B1_LOCO, cmd)


def test_check_command_rejects_wrong_shapes():
    with pytest.raises(CommandShapeError):
        check_command(Boundary.B1_LOCO, {"vx": 0.0})
    bad = make_loco_command()
    bad["has_body_pitch"] = 1
    with pytest.raises(CommandShapeError):
        check_command(Boundary.B1_LOCO, bad)


def test_iit_11d_command_is_a_one_step_chunk():
    body = {
        "loco": make_loco_command(vx=0.2, vy=0.0, wz=0.1, body_pitch=0.05),  # twist 3 + pitch 1
        "arm": {
            "mode": 2,  # JOINT_POSITION
            "ee_poses": [],
            "joint_positions": [0.0] * 6,  # 6 arm joints
            "num_joints": 6,
            "gripper": [0.5],  # gripper 1
            "dt": 0.1,
        },
        "dt": 0.1,
    }
    check_command(Boundary.B3_BODY, body)
    assert 3 + 1 + len(body["arm"]["joint_positions"]) + len(body["arm"]["gripper"]) == 11
    body["arm"]["joint_positions"] = [0.0] * 5
    with pytest.raises(CommandShapeError):
        check_command(Boundary.B3_BODY, body)
