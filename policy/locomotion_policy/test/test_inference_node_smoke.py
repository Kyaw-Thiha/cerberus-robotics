"""Import check for the rclpy wrapper, and an in-process ROS smoke test when the graph is available.

The smoke test needs rclpy AND the built ``cerberus_msgs`` package (``source install/setup.bash`` or a
colcon install prefix on PYTHONPATH), plus the gitignored GA ONNX candidate. It skips with a clear
reason otherwise; ``test_wrapper_imports_under_ros`` runs anywhere rclpy does.
"""

import importlib
import os
import time

import pytest

rclpy = pytest.importorskip("rclpy")

REPO_ROOT = None
for _p in __import__("pathlib").Path(__file__).resolve().parents:
    if (_p / "pixi.toml").exists():
        REPO_ROOT = _p
        break
CANDIDATE = REPO_ROOT / "policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate.onnx"


def test_wrapper_imports_under_ros():
    import locomotion_policy.inference_node as node_module

    assert callable(node_module.main)
    assert node_module.InferenceNode.__mro__[1].__name__ == "Node"


def _messages_available() -> bool:
    try:
        importlib.import_module("cerberus_msgs.msg")
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _messages_available(), reason="cerberus_msgs not built/on PYTHONPATH (source install/setup.bash)")
@pytest.mark.skipif(not (CANDIDATE.is_file() and __import__("pathlib").Path(str(CANDIDATE) + ".data").is_file()),
                    reason="GA candidate ONNX (+ .onnx.data) is gitignored and not present locally")
def test_rclpy_round_trip_publishes_name_keyed_joint_targets():
    import numpy as np
    from cerberus_msgs.msg import HeightScan, LocoCommand
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import Imu, JointState

    from locomotion_policy.inference_node import InferenceNode

    os.environ.setdefault("ROS_DOMAIN_ID", "93")
    rclpy.init()
    names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    defaults = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]
    overrides = [
        Parameter("checkpoint_path", value=str(CANDIDATE)),
        Parameter("backend", value="heightscan_ppo"),
        Parameter("repo_root", value=str(REPO_ROOT)),
        Parameter("observation_shape", value="proprioception_plus_height_scan"),
        Parameter("joint_names", value=names),
        Parameter("default_joint_pos", value=defaults),
        Parameter("num_joints", value=12),
        Parameter("height_scan_size", value=187),
        Parameter("action_scale", value=0.25),
        Parameter("command_limiter_config", value=str(REPO_ROOT / "common/cerberus_policies/config/command_limiter.yaml")),
        Parameter("enable_command_limiter", value=True),
    ]

    received = []

    class Driver(Node):
        def __init__(self):
            super().__init__("h3_smoke_driver")
            # scrambled JointState order on purpose: the node must map by name
            self._joint_order = [names[i] for i in (5, 0, 11, 7, 2, 9, 4, 1, 10, 8, 3, 6)]
            self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
            self.imu_pub = self.create_publisher(Imu, "imu", 10)
            self.scan_pub = self.create_publisher(HeightScan, "height_scan", 10)
            self.cmd_pub = self.create_publisher(LocoCommand, "loco_command", 10)
            self.create_subscription(JointState, "joint_position_command", received.append, 10)

        def drive(self):
            js = JointState()
            js.name = list(self._joint_order)
            index = {n: i for i, n in enumerate(names)}
            js.position = [defaults[index[n]] for n in self._joint_order]
            js.velocity = [0.0] * 12
            self.joint_pub.publish(js)

            imu = Imu()
            imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w = 0.0, 0.0, 0.0, 1.0
            imu.angular_velocity.x = 0.01
            self.imu_pub.publish(imu)

            scan = HeightScan()
            scan.rows, scan.cols, scan.resolution = 17, 11, 0.1
            scan.data = np.zeros(17 * 11, dtype=np.float32).tolist()
            self.scan_pub.publish(scan)

            cmd = LocoCommand()
            cmd.vx, cmd.vy, cmd.wz = 0.3, 0.0, 0.0
            self.cmd_pub.publish(cmd)

    node = InferenceNode(parameter_overrides=overrides)
    driver = Driver()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(driver)
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not received:
            driver.drive()
            executor.spin_once(timeout_sec=0.05)
        assert received, "no joint target message within 20 s (discovery/typing failed?)"
        msg = received[0]
        assert list(msg.name) == names
        assert len(msg.position) == 12
        assert all(np.isfinite(v) for v in msg.position)
    finally:
        executor.remove_node(node)
        executor.remove_node(driver)
        executor.shutdown()
        node.destroy_node()
        driver.destroy_node()
        rclpy.shutdown()
