"""ROS2 inference node: thin rclpy wrapper around ``locomotion_policy.core``.

Inputs: ``LocoCommand`` (B1, primary), ``sensor_msgs/Imu``, ``sensor_msgs/JointState`` and
``cerberus_msgs/HeightScan``. Output: ``sensor_msgs/JointState`` with joint targets keyed by name.
A deprecated ``geometry_msgs/Twist`` adapter exists behind ``enable_cmd_vel_adapter`` (default off)
for the pre-H3 scripted missions.

Joint order always comes from ``joint_names`` (launch-time YAML / the observation contract), never
from array position: incoming ``JointState`` values are remapped by name. A ``CommandRateLimiter``
sits between the command input and the policy when ``enable_command_limiter`` is true; its numbers
come from ``command_limiter_config`` (the shipped YAML is a placeholder, not a tuning).

The policy itself is resolved through ``common.cerberus_policies`` (registry / ONNX backend), so
``repo_root`` must point at the repo checkout. That import happens inside ``__init__``, after
``repo_root`` is on ``sys.path``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState

from locomotion_policy.core import LocomotionCore, PolicySpec, build_limiter, zero_loco_command
from locomotion_policy.observation_assembly import _build_permutation


class InferenceNode(Node):
    def __init__(self, **node_kwargs) -> None:
        # **node_kwargs is for embedding/tests (e.g. parameter_overrides); normal launches pass none.
        super().__init__("locomotion_policy", **node_kwargs)

        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("backend", "")  # "" = legacy inline ONNX path (blind), else registry name
        self.declare_parameter("repo_root", "")
        self.declare_parameter("contract_path", "")
        self.declare_parameter("observation_shape", "")
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("joint_names", [""])
        self.declare_parameter("default_joint_pos", [0.0])
        self.declare_parameter("num_joints", 12)
        self.declare_parameter("height_scan_size", 187)
        self.declare_parameter("action_scale", 0.25)
        self.declare_parameter("joint_state_timeout_s", 0.5)
        self.declare_parameter("imu_timeout_s", 0.5)
        self.declare_parameter("height_scan_timeout_s", 0.5)
        self.declare_parameter("enable_command_limiter", True)
        self.declare_parameter("command_limiter_config", "")
        self.declare_parameter("enable_cmd_vel_adapter", False)
        self.declare_parameter("loco_command_topic", "loco_command")
        self.declare_parameter("cmd_vel_topic", "cmd_vel")
        self.declare_parameter("imu_topic", "imu")
        self.declare_parameter("joint_state_topic", "joint_states")
        self.declare_parameter("height_scan_topic", "height_scan")
        self.declare_parameter("output_topic", "joint_position_command")

        checkpoint_path = self.get_parameter("checkpoint_path").value
        self.observation_shape = self.get_parameter("observation_shape").value
        if not checkpoint_path or not self.observation_shape:
            raise RuntimeError("checkpoint_path and observation_shape parameters are required")

        self.backend_name = self.get_parameter("backend").value
        self.joint_names = list(self.get_parameter("joint_names").value)
        self.num_joints = int(self.get_parameter("num_joints").value)
        self.height_scan_size = int(self.get_parameter("height_scan_size").value)
        self.action_scale = float(self.get_parameter("action_scale").value)
        self.joint_state_timeout_s = float(self.get_parameter("joint_state_timeout_s").value)
        self.imu_timeout_s = float(self.get_parameter("imu_timeout_s").value)
        self.height_scan_timeout_s = float(self.get_parameter("height_scan_timeout_s").value)
        self.enable_cmd_vel_adapter = bool(self.get_parameter("enable_cmd_vel_adapter").value)
        self.enable_command_limiter = bool(self.get_parameter("enable_command_limiter").value)

        if len(self.joint_names) != self.num_joints:
            raise RuntimeError(
                f"joint_names parameter has {len(self.joint_names)} entries but "
                f"num_joints parameter is {self.num_joints} -- these must match"
            )
        default_joint_pos = [float(v) for v in self.get_parameter("default_joint_pos").value]
        if len(default_joint_pos) != self.num_joints:
            raise RuntimeError(
                f"default_joint_pos parameter has {len(default_joint_pos)} entries but "
                f"num_joints parameter is {self.num_joints} -- these must match"
            )

        policy_fn = self._make_policy_fn(checkpoint_path)
        # common.* is importable only after _repo_root() has run inside _make_policy_fn
        from common.cerberus_compat.quat import projected_gravity_from_quat

        self._projected_gravity = projected_gravity_from_quat
        limiter = self._make_limiter()
        spec = PolicySpec(
            joint_names=tuple(self.joint_names),
            default_joint_pos=tuple(default_joint_pos),
            action_scale=self.action_scale,
            observation_shape=self.observation_shape,
            height_scan_size=self.height_scan_size,
        )
        self.core = LocomotionCore(spec, policy_fn, limiter=limiter)

        self._latest_joint_state: JointState | None = None
        self._latest_imu: Imu | None = None
        self._latest_height_scan: np.ndarray | None = None
        self._joint_state_to_policy_perm: np.ndarray | None = None
        self._last_joint_state_names: list[str] | None = None
        self._command = zero_loco_command()
        self._got_loco_command = False
        self._last_step_stamp = None
        self._latest_joint_state_stamp = None
        self._latest_imu_stamp = None
        self._latest_height_scan_stamp = None

        self.create_subscription(JointState, self.get_parameter("joint_state_topic").value, self._on_joint_state, 10)
        self.create_subscription(Imu, self.get_parameter("imu_topic").value, self._on_imu, 10)
        self.create_subscription(
            self._height_scan_type(), self.get_parameter("height_scan_topic").value, self._on_height_scan, 10
        )
        self.create_subscription(
            self._loco_command_type(), self.get_parameter("loco_command_topic").value, self._on_loco_command, 10
        )
        if self.enable_cmd_vel_adapter:
            self.get_logger().warn(
                "cmd_vel adapter enabled: Twist is converted into a LocoCommand and IGNORED once a real "
                "LocoCommand arrives. Deprecated -- use the LocoCommand topic."
            )
            self.create_subscription(Twist, self.get_parameter("cmd_vel_topic").value, self._on_cmd_vel, 10)
        self._target_pub = self.create_publisher(JointState, self.get_parameter("output_topic").value, 10)

        control_rate = self.get_parameter("control_rate_hz").value
        self.create_timer(1.0 / control_rate, self._step)

        if self.observation_shape == "proprioception_plus_height_scan":
            self.get_logger().info(
                "waiting for a HeightScan; the height-scan convention is unverified against the live "
                "Lab term (docs/interfaces.md, heightscan_ppo_v1 contract)"
            )

    # -- wiring helpers ------------------------------------------------------------------------
    @staticmethod
    def _height_scan_type():
        from cerberus_msgs.msg import HeightScan

        return HeightScan

    @staticmethod
    def _loco_command_type():
        from cerberus_msgs.msg import LocoCommand

        return LocoCommand

    def _repo_root(self) -> Path:
        configured = self.get_parameter("repo_root").value
        if not configured:
            configured = os.environ.get("CERBERUS_ROOT", "")
        if not configured:
            raise RuntimeError(
                "the registry-based policy backend needs repo_root (or CERBERUS_ROOT) to import "
                "common.cerberus_policies; the bringup launch passes repo_root"
            )
        if configured not in sys.path:
            sys.path.insert(0, configured)
        return Path(configured)

    def _make_policy_fn(self, checkpoint_path: str):
        graph = Path(checkpoint_path)
        if not graph.is_file():
            raise RuntimeError(
                f"ONNX checkpoint not found at {graph.resolve()!s}. Checkpoints under "
                "policy/locomotion/checkpoints/ are gitignored and synced separately via R2 "
                "(see docs/artifact_storage.md) -- a fresh clone will not have them until synced down."
            )
        if self.backend_name:
            self._repo_root()
            from common.cerberus_policies import registry

            contract_path = self.get_parameter("contract_path").value
            config = {"contract_path": contract_path} if contract_path else None
            backend = registry.get_backend("loco", self.backend_name, config=config)
            backend.load(str(graph), None)
            self.get_logger().info(f"policy backend: {self.backend_name} ({type(backend).__name__})")
            return lambda obs: backend.act(obs)["actions"]

        # Legacy path: an export without a registry backend (the blind/flat checkpoint). Kept for
        # parity with the pre-H3 node; new checkpoints should ship a contract + backend instead.
        import onnxruntime as ort

        session = ort.InferenceSession(str(graph))
        input_name = session.get_inputs()[0].name
        self.get_logger().warn(
            "no backend parameter set: using the deprecated inline ONNX path (no contract checks, "
            "no external .onnx.data guard)"
        )
        return lambda obs: session.run(None, {input_name: obs.reshape(1, -1)})[0].reshape(-1)

    def _make_limiter(self):
        if not self.enable_command_limiter:
            self.get_logger().warn("command rate limiter DISABLED by parameter")
            return None
        config_path = self.get_parameter("command_limiter_config").value
        if not config_path:
            raise RuntimeError(
                "enable_command_limiter is true but command_limiter_config is empty: pass the YAML "
                "path (common/cerberus_policies/config/command_limiter.yaml); the values are "
                "untested placeholders"
            )
        self._repo_root()
        limiter = build_limiter(config_path)
        self.get_logger().info(
            f"command limiter: max_rate={dict(limiter.max_rate)} max_abs={dict(limiter.max_abs)} "
            "(placeholder values, effect untested)"
        )
        return limiter

    # -- callbacks -----------------------------------------------------------------------------
    def _on_joint_state(self, msg: JointState) -> None:
        names = list(msg.name)
        if names != self._last_joint_state_names:
            self._joint_state_to_policy_perm = _build_permutation(names, self.joint_names)
            self._last_joint_state_names = names
        self._latest_joint_state = msg
        self._latest_joint_state_stamp = self.get_clock().now()

    def _on_imu(self, msg: Imu) -> None:
        self._latest_imu = msg
        self._latest_imu_stamp = self.get_clock().now()

    def _on_height_scan(self, msg) -> None:
        data = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        expected = int(msg.rows) * int(msg.cols)
        if expected != data.size:
            self.get_logger().warn(
                f"HeightScan says {msg.rows}x{msg.cols}={expected} cells but carries {data.size}; dropping it",
                throttle_duration_sec=2.0,
            )
            return
        self._latest_height_scan = data
        self._latest_height_scan_stamp = self.get_clock().now()

    def _on_loco_command(self, msg) -> None:
        self._command = {
            "vx": float(msg.vx),
            "vy": float(msg.vy),
            "wz": float(msg.wz),
            "body_pitch": float(msg.body_pitch),
            "body_height": float(msg.body_height),
            "has_body_pitch": bool(msg.has_body_pitch),
            "has_body_height": bool(msg.has_body_height),
        }
        self._got_loco_command = True

    def _on_cmd_vel(self, msg: Twist) -> None:
        if self._got_loco_command:
            return  # a real LocoCommand source is present; the adapter stays silent
        self._command = {**zero_loco_command(), "vx": float(msg.linear.x), "vy": float(msg.linear.y), "wz": float(msg.angular.z)}

    # -- control -------------------------------------------------------------------------------
    def _fresh(self, stamp, timeout: float) -> bool:
        if stamp is None:
            return False
        return (self.get_clock().now() - stamp).nanoseconds / 1e9 <= timeout

    def _step(self) -> None:
        if self._latest_joint_state is None:
            return
        if not self._fresh(self._latest_joint_state_stamp, self.joint_state_timeout_s):
            self.get_logger().warn(
                "joint_states is stale -- skipping this control step instead of publishing a command "
                "derived from a frozen observation.",
                throttle_duration_sec=2.0,
            )
            return
        if not self._fresh(self._latest_imu_stamp, self.imu_timeout_s):
            self.get_logger().warn(
                "no fresh IMU message -- waiting for orientation/angular velocity (the policy's "
                "projected gravity and base angular velocity come from it).",
                throttle_duration_sec=2.0,
            )
            return
        scan = self._latest_height_scan
        if self.observation_shape == "proprioception_plus_height_scan" and not self._fresh(
            self._latest_height_scan_stamp, self.height_scan_timeout_s
        ):
            self.get_logger().warn(
                "no fresh HeightScan -- skipping this control step.", throttle_duration_sec=2.0
            )
            return

        perm = self._joint_state_to_policy_perm
        raw_joint_pos = np.asarray(self._latest_joint_state.position, dtype=np.float32)[perm]
        raw_joint_vel = np.asarray(self._latest_joint_state.velocity, dtype=np.float32)[perm]
        joint_positions = dict(zip(self.joint_names, raw_joint_pos))
        joint_velocities = dict(zip(self.joint_names, raw_joint_vel))

        now = self.get_clock().now()
        if self._last_step_stamp is None:
            dt = 1.0 / float(self.get_parameter("control_rate_hz").value)
        else:
            dt = max((now - self._last_step_stamp).nanoseconds / 1e9, 1e-6)
        self._last_step_stamp = now

        command = self.core.command(self._command, dt)
        imu = self._latest_imu
        q = imu.orientation
        gravity = self._projected_gravity([q.x, q.y, q.z, q.w])
        # TODO(hardware): base linear velocity has no IMU source; feed it from odometry / a state
        # estimator on the real robot (zeros here means the policy never sees its own speed).
        base_lin_vel = np.zeros(3, dtype=np.float32)
        out = self.core.step(
            base_lin_vel=base_lin_vel,
            base_ang_vel=np.array([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z], dtype=np.float32),
            projected_gravity=gravity,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            height_scan=scan,
            command=command,
        )

        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = list(out.joint_targets)
        msg.position = list(out.joint_targets.values())
        self._target_pub.publish(msg)


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
