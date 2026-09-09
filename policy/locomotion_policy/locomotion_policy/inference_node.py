"""ROS2 inference node: loads an exported ONNX locomotion policy, assembles
observations from live topics, publishes joint position commands. Which
checkpoint/observation_shape/joint_names/default_joint_pos/num_joints/
height_scan_size/action_scale to use comes from launch-time parameters,
resolved by cerberus_bringup's launch file from platforms.yaml -- this node
never reads platforms.yaml itself. `default_joint_pos` in particular comes
from the `default_joint_pos` parameter (platforms.yaml's per-platform
`default_joint_pos` field, currently a zero-vector placeholder -- see that
file's comment).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

from locomotion_policy.observation_assembly import (
    SHAPE_PERCEPTIVE,
    ProprioSample,
    _build_permutation,
    apply_action_postprocessing,
    assemble_observation,
)


class InferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("locomotion_policy")

        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("observation_shape", "")
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("joint_names", [""])
        self.declare_parameter("default_joint_pos", [0.0])
        self.declare_parameter("num_joints", 12)
        self.declare_parameter("height_scan_size", 187)
        self.declare_parameter("action_scale", 0.25)
        self.declare_parameter("joint_state_timeout_s", 0.5)

        checkpoint_path = self.get_parameter("checkpoint_path").value
        self.observation_shape = self.get_parameter("observation_shape").value
        if not checkpoint_path or not self.observation_shape:
            raise RuntimeError("checkpoint_path and observation_shape parameters are required")

        self.joint_names = list(self.get_parameter("joint_names").value)
        self.num_joints = int(self.get_parameter("num_joints").value)
        self.height_scan_size = int(self.get_parameter("height_scan_size").value)
        self.action_scale = float(self.get_parameter("action_scale").value)
        self.joint_state_timeout_s = float(self.get_parameter("joint_state_timeout_s").value)

        if len(self.joint_names) != self.num_joints:
            raise RuntimeError(
                f"joint_names parameter has {len(self.joint_names)} entries but "
                f"num_joints parameter is {self.num_joints} -- these must match"
            )

        default_joint_pos = list(self.get_parameter("default_joint_pos").value)
        if len(default_joint_pos) != self.num_joints:
            raise RuntimeError(
                f"default_joint_pos parameter has {len(default_joint_pos)} entries but "
                f"num_joints parameter is {self.num_joints} -- these must match"
            )
        self._default_joint_pos = np.array(default_joint_pos, dtype=np.float32)

        if self.observation_shape == SHAPE_PERCEPTIVE:
            self.get_logger().warn(
                "height_scan stubbed flat -- perceptive mode is running in bring-up/testing "
                "configuration, not a validated real-sensor deployment. See "
                "docs/locomotion_architecture.md."
            )

        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.is_file():
            raise RuntimeError(
                f"ONNX checkpoint not found at {checkpoint_file.resolve()!s}. Checkpoints under "
                "policy/locomotion/checkpoints/ are gitignored and synced separately via R2 "
                "(see docs/artifact_storage.md) -- a fresh clone will not have them until synced "
                "down."
            )
        self.session = ort.InferenceSession(checkpoint_path)
        self._input_name = self.session.get_inputs()[0].name

        self._latest_joint_state: JointState | None = None
        self._latest_joint_state_stamp = None
        self._latest_cmd_vel = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(self.num_joints, dtype=np.float32)

        # Cached permutation from the most recently seen JointState.name order
        # into self.joint_names order, and its inverse -- built once the
        # first JointState message arrives (see _on_joint_state).
        self._joint_state_to_policy_perm: np.ndarray | None = None
        self._policy_to_joint_state_perm: np.ndarray | None = None
        self._last_joint_state_names: list[str] | None = None

        self.create_subscription(JointState, "joint_states", self._on_joint_state, 10)
        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, 10)
        self._action_pub = self.create_publisher(Float32MultiArray, "joint_position_command", 10)

        control_rate = self.get_parameter("control_rate_hz").value
        self.create_timer(1.0 / control_rate, self._step)

    def _on_joint_state(self, msg: JointState) -> None:
        names = list(msg.name)
        if names != self._last_joint_state_names:
            # Names changed (or this is the first message) -- (re)build the
            # cached permutation mapping msg.name's order to self.joint_names
            # order, and its inverse for publishing actions back out.
            self._joint_state_to_policy_perm = _build_permutation(names, self.joint_names)
            self._policy_to_joint_state_perm = _build_permutation(self.joint_names, names)
            self._last_joint_state_names = names

        self._latest_joint_state = msg
        self._latest_joint_state_stamp = self.get_clock().now()

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._latest_cmd_vel = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32)

    def _height_scan_stub(self) -> np.ndarray:
        # Constant flat-terrain value -- see the startup warning above and
        # docs/locomotion_architecture.md for why this is a bring-up stub,
        # not a real sensor reading.
        return np.zeros(self.height_scan_size, dtype=np.float32)

    def _step(self) -> None:
        if self._latest_joint_state is None:
            return  # no joint data yet, skip this tick

        age_s = (self.get_clock().now() - self._latest_joint_state_stamp).nanoseconds / 1e9
        if age_s > self.joint_state_timeout_s:
            self.get_logger().warn(
                f"/joint_states is stale ({age_s:.2f}s old, timeout {self.joint_state_timeout_s}s) "
                "-- skipping this control step instead of publishing a command derived from a "
                "frozen observation.",
                throttle_duration_sec=2.0,
            )
            return

        raw_joint_pos = np.array(self._latest_joint_state.position, dtype=np.float32)
        raw_joint_vel = np.array(self._latest_joint_state.velocity, dtype=np.float32)

        perm = self._joint_state_to_policy_perm
        joint_pos = raw_joint_pos[perm]
        joint_vel = raw_joint_vel[perm]

        proprio = ProprioSample(
            base_lin_vel=np.zeros(3, dtype=np.float32),  # TODO(hardware): wire to a real IMU/odometry topic
            base_ang_vel=np.zeros(3, dtype=np.float32),  # TODO(hardware): wire to a real IMU topic
            projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),  # TODO(hardware): wire to IMU orientation
            velocity_commands=self._latest_cmd_vel,
            joint_pos_rel=joint_pos - self._default_joint_pos,
            joint_vel_rel=joint_vel,
            last_action=self._last_action,
        )
        height_scan = self._height_scan_stub() if self.observation_shape == SHAPE_PERCEPTIVE else None
        obs = assemble_observation(self.observation_shape, proprio, height_scan)

        raw_output = self.session.run(None, {self._input_name: obs.reshape(1, -1)})[0].reshape(-1)
        self._last_action = raw_output
        joint_targets = apply_action_postprocessing(raw_output, self._default_joint_pos, scale=self.action_scale)

        # joint_targets is in self.joint_names order -- apply the inverse
        # permutation so the published array matches the most recently
        # received JointState.name order, the same order a downstream
        # consumer subscribing to /joint_states would expect.
        published_targets = joint_targets[self._policy_to_joint_state_perm]

        msg = Float32MultiArray()
        msg.data = published_targets.tolist()
        self._action_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
