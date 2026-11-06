"""Computes REAL stereo-matched depth for one camera position (front or rear)
of the `multicam_slam` rig, from the two rectified RGB images that position's
stereo pair (`camera_configs.FRONT_CAMERA_LEFT_CFG`/`_RIGHT_CFG`, or the REAR
equivalents) produces. Deliberately a SEPARATE node from
`common_adapter.SlamBackendAdapter` -- see that module's docstring: "the
backend adapter is thin ... does NOT encode backend-specific logic beyond
remapping". Turning a stereo pair into a depth image via block matching is
real, non-trivial per-frame computation (not a topic rename), so it lives
here instead, launched alongside (not inside) `adapter_launch.py`.

Why real depth instead of Isaac Lab's privileged sim ground-truth depth:
docs/perception.md's project-wide, citation-backed decision -- a 2026
quadruped SLAM benchmark found stereo outperforms RGB-D on legged platforms,
and global-shutter stereo avoids the motion-blur rolling-shutter RGB-D showed
under quadruped locomotion. Privileged depth also doesn't exist on real
hardware, so anything trained/tuned against it wouldn't transfer. This node
is how Cerberus produces depth that DOES transfer: pure OpenCV stereo block
matching over the two RGB images, the same input a real stereo rig would
give the algorithm.

Output format is dictated by multicam_slam's ROS2 wrapper (confirmed
separately, see adapter_launch.py's docstring for the sourcing) and by
ORB-SLAM2's own RGBDDataset convention: 16-bit millimeters (`16UC1`),
`DepthMapFactor ~1000.0` (i.e. pixel value / 1000.0 = depth in meters, same
as a real RealSense/Astra RGB-D sensor's native depth encoding -- the
algorithm has zero opinion on how that 16UC1 buffer was produced, per the
task brief this module was built from).
"""

from __future__ import annotations

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

# --- Camera geometry, sourced from (not imported from -- that module imports
# isaaclab, which this ROS2-side node's container does not have installed;
# see sensor_configs.py's own module docstring for the same constraint)
# common/cerberus_description/sensors/sensor_configs.py. Keep these two
# modules' numbers in sync by hand, the same way
# system/cerberus_bringup/config/sensor_profiles.yaml already does for the
# suite wiring. ---
#
# _FRONT_CAM_SPAWN = sim_utils.PinholeCameraCfg(focal_length=24.0, ...,
# horizontal_aperture=20.955, ...); FRONT_CAMERA_LEFT_CFG/RIGHT_CFG and
# REAR_CAMERA_LEFT_CFG/RIGHT_CFG all reuse this spawn and are both
# height=480, width=640.
SENSOR_CONFIGS_FOCAL_LENGTH_MM = 24.0
SENSOR_CONFIGS_HORIZONTAL_APERTURE_MM = 20.955
SENSOR_CONFIGS_IMAGE_WIDTH_PX = 640
# front_cam_left/right offsets: pos=(0.32, -0.06, 0.05) / (0.32, 0.06, 0.05);
# rear_cam_left/right: pos=(-0.32, -0.06, 0.05) / (-0.32, 0.06, 0.05). Both
# pairs share the same +/-0.06m y-offset -> 0.06 - (-0.06) = 0.12m baseline.
SENSOR_CONFIGS_STEREO_BASELINE_M = 0.12

_ENCODING_CHANNELS = {"mono8": 1, "rgb8": 3, "bgr8": 3}


def pinhole_focal_length_px(focal_length_mm: float, horizontal_aperture_mm: float, width_px: int) -> float:
    """Converts an Isaac Lab `PinholeCameraCfg`'s physical focal length (mm)
    into the pixel-space focal length (fx) the disparity-to-depth formula
    needs: fx_px = focal_length_mm * width_px / horizontal_aperture_mm --
    standard pinhole-camera unit conversion (aperture and focal length share
    units, so the ratio is dimensionless and scales directly by pixel
    width)."""
    if horizontal_aperture_mm <= 0:
        raise ValueError(f"horizontal_aperture_mm must be positive, got {horizontal_aperture_mm}")
    return focal_length_mm * width_px / horizontal_aperture_mm


def decode_image_message(msg: Image) -> np.ndarray:
    """Manual sensor_msgs/Image -> numpy decode (no cv_bridge dependency --
    keeps this node's dependency footprint to rclpy/cv2/numpy, already
    required elsewhere). Handles row padding via `step` rather than assuming
    step == width * channels."""
    channels = _ENCODING_CHANNELS.get(msg.encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}, expected one of {sorted(_ENCODING_CHANNELS)}")
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    row_stride = msg.step
    rows = buf.reshape(msg.height, row_stride)[:, : msg.width * channels]
    if channels == 1:
        return rows.reshape(msg.height, msg.width)
    return rows.reshape(msg.height, msg.width, channels)


def to_grayscale(image: np.ndarray, encoding: str) -> np.ndarray:
    """Converts a decoded image array to single-channel grayscale, the input
    `cv2.StereoSGBM_create` expects. `mono8` passes through unchanged."""
    if encoding == "mono8":
        return image
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if encoding == "bgr8":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"unsupported image encoding {encoding!r}, expected one of {sorted(_ENCODING_CHANNELS)}")


def disparity_to_depth_mm(
    disparity_px: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    min_valid_disparity_px: float = 1.0,
    max_depth_mm: float = 65535.0,
) -> np.ndarray:
    """Standard stereo triangulation: depth = (focal_length_px * baseline_m) /
    disparity_px, converted to millimeters and quantized to uint16 (the
    16UC1/DepthMapFactor=1000.0 convention this whole node exists to
    produce). Pixels with disparity below `min_valid_disparity_px` (no match
    found, or a match so far away it's numerically unstable -- near-zero
    disparity blows up to near-infinite depth) are marked invalid with 0,
    the same "no data" convention a real RealSense/Astra sensor uses and
    ORB-SLAM2's RGBDDataset reader already treats as invalid (depth <= 0
    skipped). Depth beyond `max_depth_mm` (uint16's ~65.5m range) is clipped,
    not wrapped -- silently wrapping a far point back into a near, plausible-
    looking depth would be a worse failure than clamping it to the sensor's
    max range."""
    if disparity_px.dtype != np.float32 and disparity_px.dtype != np.float64:
        disparity_px = disparity_px.astype(np.float32)
    valid = disparity_px >= min_valid_disparity_px
    depth_mm = np.zeros(disparity_px.shape, dtype=np.float64)
    depth_mm[valid] = (focal_length_px * baseline_m * 1000.0) / disparity_px[valid]
    depth_mm = np.clip(depth_mm, 0.0, max_depth_mm)
    depth_mm[~valid] = 0.0
    return depth_mm.astype(np.uint16)


def encode_depth_image_message(depth_mm: np.ndarray, header: Header) -> Image:
    """numpy uint16 depth (millimeters) -> sensor_msgs/Image, `16UC1`
    little-endian -- the exact format multicam_slam's ROS2 wrapper expects
    (see this module's docstring)."""
    if depth_mm.dtype != np.uint16:
        raise ValueError(f"depth_mm must be uint16, got {depth_mm.dtype}")
    msg = Image()
    msg.header = header
    msg.height, msg.width = depth_mm.shape
    msg.encoding = "16UC1"
    msg.is_bigendian = 0
    msg.step = msg.width * 2
    msg.data = depth_mm.astype("<u2").tobytes()
    return msg


def compute_disparity(left_gray: np.ndarray, right_gray: np.ndarray, matcher: cv2.StereoSGBM) -> np.ndarray:
    """Runs the matcher and converts OpenCV's fixed-point disparity output
    (int16, scaled by 16) into real disparity-in-pixels (float32). See
    https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html --
    `compute()`'s stated output format."""
    raw = matcher.compute(left_gray, right_gray)
    return raw.astype(np.float32) / 16.0


def make_stereo_sgbm(
    num_disparities: int = 128,
    block_size: int = 7,
    min_disparity: int = 0,
    uniqueness_ratio: int = 10,
    speckle_window_size: int = 100,
    speckle_range: int = 32,
    disp12_max_diff: int = 1,
) -> cv2.StereoSGBM:
    """`cv2.StereoSGBM_create` -- the modern, recommended semi-global block
    matcher (per the task brief: NOT the older/deprecated `StereoBM`, which
    trades away SGBM's smoothness penalty for speed multicam_slam's offline-
    ish SLAM pipeline doesn't need). P1/P2 (small/large disparity-change
    smoothness penalties) use OpenCV's own documented formula, scaled by
    channel count (1, since inputs are grayscale) and block_size^2."""
    if num_disparities % 16 != 0:
        raise ValueError(f"num_disparities must be a multiple of 16 (OpenCV requirement), got {num_disparities}")
    channels = 1
    return cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * channels * block_size**2,
        P2=32 * channels * block_size**2,
        disp12MaxDiff=disp12_max_diff,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


class StereoToDepthNode(Node):
    """One instance handles ONE camera position (front or rear) -- launch two
    instances (see stereo_to_depth_launch.py) rather than one node juggling
    both positions' synchronizers, so a slow/stuck rear pair can never stall
    front depth publishing."""

    def __init__(self, **node_kwargs) -> None:
        # **node_kwargs forwards straight to rclpy.node.Node (e.g.
        # parameter_overrides=[...]) -- lets tests construct a node with a
        # parameter already overridden without needing a live launch/CLI
        # argument source, the same mechanism `ros2 launch`'s Node(parameters=[...])
        # uses under the hood.
        super().__init__("stereo_to_depth_node", **node_kwargs)

        self.declare_parameter("left_image_topic", "/camera_01/rgb/image_raw")
        self.declare_parameter("right_image_topic", "/camera_01/right/image_raw")
        self.declare_parameter("depth_image_topic", "/camera_01/depth/image_raw")
        self.declare_parameter("input_encoding", "rgb8")
        self.declare_parameter("sync_slop_sec", 0.02)

        self.declare_parameter("focal_length_mm", SENSOR_CONFIGS_FOCAL_LENGTH_MM)
        self.declare_parameter("horizontal_aperture_mm", SENSOR_CONFIGS_HORIZONTAL_APERTURE_MM)
        self.declare_parameter("image_width_px", SENSOR_CONFIGS_IMAGE_WIDTH_PX)
        self.declare_parameter("baseline_m", SENSOR_CONFIGS_STEREO_BASELINE_M)
        # Overrides the focal_length_mm/horizontal_aperture_mm/image_width_px
        # derivation below when >0 -- an escape hatch for a real stereo rig
        # (calibrated intrinsics, not a PinholeCameraCfg) without touching sim config.
        self.declare_parameter("focal_length_px_override", -1.0)

        self.declare_parameter("num_disparities", 128)
        self.declare_parameter("block_size", 7)
        self.declare_parameter("min_disparity", 0)
        self.declare_parameter("uniqueness_ratio", 10)
        self.declare_parameter("speckle_window_size", 100)
        self.declare_parameter("speckle_range", 32)
        self.declare_parameter("disp12_max_diff", 1)
        self.declare_parameter("min_valid_disparity_px", 1.0)

        focal_length_px_override = self.get_parameter("focal_length_px_override").value
        if focal_length_px_override > 0:
            self._focal_length_px = float(focal_length_px_override)
        else:
            self._focal_length_px = pinhole_focal_length_px(
                self.get_parameter("focal_length_mm").value,
                self.get_parameter("horizontal_aperture_mm").value,
                self.get_parameter("image_width_px").value,
            )
        self._baseline_m = float(self.get_parameter("baseline_m").value)
        self._input_encoding = self.get_parameter("input_encoding").value
        self._min_valid_disparity_px = float(self.get_parameter("min_valid_disparity_px").value)

        self._matcher = make_stereo_sgbm(
            num_disparities=self.get_parameter("num_disparities").value,
            block_size=self.get_parameter("block_size").value,
            min_disparity=self.get_parameter("min_disparity").value,
            uniqueness_ratio=self.get_parameter("uniqueness_ratio").value,
            speckle_window_size=self.get_parameter("speckle_window_size").value,
            speckle_range=self.get_parameter("speckle_range").value,
            disp12_max_diff=self.get_parameter("disp12_max_diff").value,
        )

        depth_topic = self.get_parameter("depth_image_topic").value
        self._depth_pub = self.create_publisher(Image, depth_topic, 10)

        left_topic = self.get_parameter("left_image_topic").value
        right_topic = self.get_parameter("right_image_topic").value
        self._left_sub = message_filters.Subscriber(self, Image, left_topic)
        self._right_sub = message_filters.Subscriber(self, Image, right_topic)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._left_sub, self._right_sub],
            queue_size=10,
            slop=self.get_parameter("sync_slop_sec").value,
        )
        self._sync.registerCallback(self._on_stereo_pair)

        self.get_logger().info(
            f"stereo_to_depth_node: {left_topic} + {right_topic} -> {depth_topic} "
            f"(fx={self._focal_length_px:.2f}px, baseline={self._baseline_m:.3f}m)"
        )

    def _on_stereo_pair(self, left_msg: Image, right_msg: Image) -> None:
        left_gray = to_grayscale(decode_image_message(left_msg), self._input_encoding)
        right_gray = to_grayscale(decode_image_message(right_msg), self._input_encoding)
        disparity_px = compute_disparity(left_gray, right_gray, self._matcher)
        depth_mm = disparity_to_depth_mm(
            disparity_px,
            focal_length_px=self._focal_length_px,
            baseline_m=self._baseline_m,
            min_valid_disparity_px=self._min_valid_disparity_px,
        )
        self._depth_pub.publish(encode_depth_image_message(depth_mm, left_msg.header))


def main() -> None:
    rclpy.init()
    node = StereoToDepthNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
