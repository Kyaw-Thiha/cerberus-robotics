import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from slam.slam.backends.multicam_slam.stereo_to_depth_node import (
    StereoToDepthNode,
    compute_disparity,
    decode_image_message,
    disparity_to_depth_mm,
    encode_depth_image_message,
    make_stereo_sgbm,
    pinhole_focal_length_px,
    to_grayscale,
)

# --- pinhole_focal_length_px -------------------------------------------------

def test_pinhole_focal_length_px_matches_sensor_configs_front_camera():
    # 24.0mm focal length, 20.955mm horizontal aperture, 640px width -- the
    # exact PinholeCameraCfg/CameraCfg numbers in
    # common/cerberus_description/sensors/sensor_configs.py's _FRONT_CAM_SPAWN.
    fx_px = pinhole_focal_length_px(focal_length_mm=24.0, horizontal_aperture_mm=20.955, width_px=640)
    assert fx_px == pytest.approx(24.0 * 640 / 20.955, rel=1e-9)
    assert fx_px == pytest.approx(733.0, abs=0.01)


def test_pinhole_focal_length_px_rejects_non_positive_aperture():
    with pytest.raises(ValueError, match="horizontal_aperture_mm must be positive"):
        pinhole_focal_length_px(focal_length_mm=24.0, horizontal_aperture_mm=0.0, width_px=640)


# --- disparity_to_depth_mm: the actual stereo-to-depth math -----------------

def test_disparity_to_depth_mm_known_disparity_matches_triangulation_formula():
    # depth = fx * baseline / disparity. fx=732.52px, baseline=0.12m,
    # disparity=36.626px -> depth = 732.52 * 0.12 / 36.626 = 2.4m = 2400mm.
    fx_px = 732.52
    baseline_m = 0.12
    disparity_px = np.full((4, 4), 36.626, dtype=np.float32)
    depth_mm = disparity_to_depth_mm(disparity_px, focal_length_px=fx_px, baseline_m=baseline_m)
    assert depth_mm.dtype == np.uint16
    assert np.all(np.abs(depth_mm.astype(np.int32) - 2400) <= 1)  # uint16 rounding tolerance


def test_disparity_to_depth_mm_smaller_disparity_means_farther_depth():
    fx_px = 700.0
    baseline_m = 0.12
    near = disparity_to_depth_mm(np.array([[50.0]], dtype=np.float32), fx_px, baseline_m)
    far = disparity_to_depth_mm(np.array([[10.0]], dtype=np.float32), fx_px, baseline_m)
    assert far[0, 0] > near[0, 0]


def test_disparity_to_depth_mm_zero_or_negative_disparity_is_invalid_zero():
    # No stereo match (or a degenerate/negative disparity) must never be
    # silently reported as "very far away" -- it's marked 0 (no data), the
    # same convention a real depth sensor and ORB-SLAM2's reader both use.
    disparity_px = np.array([[0.0, -5.0, 0.4]], dtype=np.float32)
    depth_mm = disparity_to_depth_mm(disparity_px, focal_length_px=700.0, baseline_m=0.12, min_valid_disparity_px=1.0)
    assert np.all(depth_mm == 0)


def test_disparity_to_depth_mm_clips_rather_than_wraps_beyond_max_range():
    # A tiny nonzero disparity would triangulate to a depth far beyond
    # uint16's ~65.5m range -- must clip to max_depth_mm, not silently wrap
    # around into a small, plausible-looking (and wrong) value.
    disparity_px = np.array([[0.001]], dtype=np.float32)
    depth_mm = disparity_to_depth_mm(
        disparity_px, focal_length_px=700.0, baseline_m=0.12, min_valid_disparity_px=0.0, max_depth_mm=65535.0
    )
    assert depth_mm[0, 0] == 65535


# --- image message encode/decode --------------------------------------------

def test_decode_image_message_roundtrips_rgb8():
    arr = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    msg = Image()
    msg.height, msg.width = 4, 6
    msg.encoding = "rgb8"
    msg.step = 6 * 3
    msg.data = arr.tobytes()
    decoded = decode_image_message(msg)
    assert decoded.shape == (4, 6, 3)
    assert np.array_equal(decoded, arr)


def test_decode_image_message_handles_row_padding_via_step():
    # step wider than width * channels (padding bytes) -- must be sliced
    # off, not folded into the image.
    height, width = 3, 5
    row_stride = 8  # 3 bytes of padding per row
    rows = []
    for r in range(height):
        row = np.arange(width, dtype=np.uint8) + r * 10
        rows.append(np.concatenate([row, np.zeros(row_stride - width, dtype=np.uint8)]))
    buf = np.stack(rows)
    msg = Image()
    msg.height, msg.width = height, width
    msg.encoding = "mono8"
    msg.step = row_stride
    msg.data = buf.tobytes()
    decoded = decode_image_message(msg)
    assert decoded.shape == (height, width)
    for r in range(height):
        assert np.array_equal(decoded[r], np.arange(width, dtype=np.uint8) + r * 10)


def test_decode_image_message_rejects_unsupported_encoding():
    msg = Image()
    msg.height, msg.width = 2, 2
    msg.encoding = "32FC1"
    msg.step = 8
    msg.data = bytes(16)
    with pytest.raises(ValueError, match="unsupported image encoding"):
        decode_image_message(msg)


def test_to_grayscale_mono8_passthrough():
    arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    assert np.array_equal(to_grayscale(arr, "mono8"), arr)


def test_to_grayscale_rgb8_produces_single_channel():
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    arr[..., 0] = 255  # pure red
    gray = to_grayscale(arr, "rgb8")
    assert gray.shape == (2, 2)
    assert gray.dtype == np.uint8


def test_encode_depth_image_message_produces_16uc1():
    depth_mm = np.array([[0, 1000], [2400, 65535]], dtype=np.uint16)
    header = Header()
    header.frame_id = "front_cam_left"
    msg = encode_depth_image_message(depth_mm, header)
    assert msg.encoding == "16UC1"
    assert msg.height == 2 and msg.width == 2
    assert msg.step == 2 * 2
    assert msg.header.frame_id == "front_cam_left"
    roundtrip = np.frombuffer(bytes(msg.data), dtype="<u2").reshape(2, 2)
    assert np.array_equal(roundtrip, depth_mm)


def test_encode_depth_image_message_rejects_non_uint16():
    with pytest.raises(ValueError, match="must be uint16"):
        encode_depth_image_message(np.zeros((2, 2), dtype=np.float32), Header())


# --- make_stereo_sgbm / compute_disparity: end-to-end pipeline check --------

def test_make_stereo_sgbm_rejects_non_multiple_of_16_num_disparities():
    with pytest.raises(ValueError, match="multiple of 16"):
        make_stereo_sgbm(num_disparities=100)


def test_compute_disparity_end_to_end_recovers_known_constant_shift():
    # Synthetic textured scene: random noise so SGBM has something to match
    # on (a flat image has no unique correspondences). Right image is the
    # left image shifted left by a known number of pixels -- simulating a
    # single fronto-parallel plane at a fixed depth, disparity = shift_px
    # everywhere the shifted window stays in-frame.
    rng = np.random.default_rng(0)
    height, width = 200, 300
    shift_px = 12
    base = rng.integers(0, 256, size=(height, width + shift_px), dtype=np.uint8)
    # OpenCV disparity convention: disparity = x_left - x_right (positive for
    # near objects) -- so the RIGHT image's column x corresponds to the LEFT
    # image's column x + shift_px, i.e. right is the base image and left is
    # base shifted so its content appears shift_px columns further right.
    left = base[:, :width]
    right = base[:, shift_px:]

    matcher = make_stereo_sgbm(num_disparities=32, block_size=5)
    disparity_px = compute_disparity(left, right, matcher)

    # Interior region only -- SGBM leaves the left numDisparities columns
    # and other border pixels unmatched (disparity < 0 there by design).
    interior = disparity_px[20:-20, 40:-40]
    valid = interior[interior >= 0]
    assert valid.size > 0
    assert np.median(valid) == pytest.approx(shift_px, abs=1.0)


def test_disparity_to_depth_mm_end_to_end_matches_expected_depth_within_tolerance():
    rng = np.random.default_rng(1)
    height, width = 200, 300
    shift_px = 20
    fx_px = pinhole_focal_length_px(24.0, 20.955, 640)
    baseline_m = 0.12
    expected_depth_mm = fx_px * baseline_m * 1000.0 / shift_px

    base = rng.integers(0, 256, size=(height, width + shift_px), dtype=np.uint8)
    left = base[:, :width]
    right = base[:, shift_px:]

    matcher = make_stereo_sgbm(num_disparities=32, block_size=5)
    disparity_px = compute_disparity(left, right, matcher)
    depth_mm = disparity_to_depth_mm(disparity_px, focal_length_px=fx_px, baseline_m=baseline_m)

    interior = depth_mm[20:-20, 40:-40]
    valid = interior[interior > 0]
    assert valid.size > 0
    # Block-matching over noise is imprecise at the pixel level (rounds to
    # integer disparity) -- a generous but real tolerance, not a rubber stamp.
    assert np.median(valid.astype(np.float64)) == pytest.approx(expected_depth_mm, rel=0.15)


# --- StereoToDepthNode: parameter wiring and callback ------------------------

def test_node_derives_focal_length_px_from_sensor_config_defaults():
    rclpy.init()
    try:
        node = StereoToDepthNode()
        try:
            assert node._focal_length_px == pytest.approx(24.0 * 640 / 20.955, rel=1e-9)
            assert node._baseline_m == pytest.approx(0.12)
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


def test_node_focal_length_px_override_wins_over_derived_value():
    from rclpy.parameter import Parameter

    rclpy.init()
    try:
        node = StereoToDepthNode(
            parameter_overrides=[Parameter("focal_length_px_override", value=999.0)],
        )
        try:
            assert node._focal_length_px == pytest.approx(999.0)
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


def test_node_publishes_depth_image_from_synced_stereo_pair():
    rclpy.init()
    try:
        node = StereoToDepthNode()
        try:
            published = []
            node._depth_pub.publish = lambda msg: published.append(msg)

            rng = np.random.default_rng(2)
            # Wide enough that the node's default num_disparities=128 search
            # window fits comfortably (StereoSGBM needs width clearly larger
            # than numDisparities, or OpenCV's internal buffer sizing breaks).
            height, width, shift_px = 200, 400, 20
            base = rng.integers(0, 256, size=(height, width + shift_px), dtype=np.uint8)
            left_gray = base[:, :width]
            right_gray = base[:, shift_px:]

            def _make_mono_msg(arr: np.ndarray) -> Image:
                msg = Image()
                msg.height, msg.width = arr.shape
                msg.encoding = "mono8"
                msg.step = arr.shape[1]
                msg.data = arr.tobytes()
                msg.header.frame_id = "front_cam_left"
                return msg

            node._input_encoding = "mono8"  # avoid needing a real rgb8 test image
            node._on_stereo_pair(_make_mono_msg(left_gray), _make_mono_msg(right_gray))

            assert len(published) == 1
            out = published[0]
            assert out.encoding == "16UC1"
            assert out.height == height and out.width == width
            assert out.header.frame_id == "front_cam_left"
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
