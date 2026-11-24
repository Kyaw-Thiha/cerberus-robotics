"""Camera geometry: a pose log must prove rigidity, the chase camera must orbit, and a forward vector must point
where the name says. No Isaac, no GPU.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.cerberus_eval import camera_rig as cr


def test_camera_axes_are_right_handed_and_look_forward():
    R = cr.rotation_from_forward(np.array([1.0, 0.0, 0.0]))
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.allclose(R @ np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]), atol=1e-9)
    assert np.allclose(R @ np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), atol=1e-9)


def test_pitched_camera_points_down():
    f = cr.forward_from_yaw_pitch(0.0, math.radians(20.0))
    assert f[0] > 0.9 and f[2] < -0.3
    R = cr.rotation_from_forward(f)
    assert np.allclose(R @ np.array([0.0, 0.0, -1.0]), f, atol=1e-9)


def test_quaternion_round_trip_matches_the_matrix():
    R = cr.rotation_from_forward(cr.forward_from_yaw_pitch(1.1, 0.3))
    q = cr.quat_from_matrix(R)
    w, x, y, z = q
    Rq = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    assert np.allclose(Rq, R, atol=1e-9)
    assert abs(math.sqrt(sum(v * v for v in q)) - 1.0) < 1e-9


def test_chase_eye_sits_behind_the_base_and_above_it():
    eye, R = cr.chase_pose(np.array([2.0, 3.0, 0.4]), 0.0)
    assert eye[0] < 2.0          # behind along +X yaw
    assert abs(eye[2] - (0.4 + 0.9)) < 1e-9
    f = R @ np.array([0.0, 0.0, -1.0])
    assert f[0] > 0 and f[2] < 0  # looking forward and down


def test_pose_log_schema_is_what_clip_qa_reads(tmp_path):
    log = cr.PoseLog(camera="ego", width=1920, height=1080, hfov_deg=90.0)
    q = cr.quat_from_matrix(np.eye(3))
    for i in range(3):
        log.append(t=i * 0.1, base_pos=(i, 0, 0.4), base_quat_wxyz=q,
                   cam_pos=(i + 0.3, 0, 0.48), cam_quat_wxyz=q)
    path = log.write(tmp_path / "poses.json")
    import json
    data = json.loads(Path(path).read_text())
    assert data["schema"] == cr.POSE_SCHEMA
    assert data["camera"] == "ego" and len(data["frames"]) == 3
    frame = data["frames"][0]
    assert set(frame) == {"t", "base_pos", "base_quat_xyzw", "cam_pos", "cam_quat_xyzw"}
    # xyzw order: identity is (0, 0, 0, 1)
    assert frame["cam_quat_xyzw"][3] == 1.0


def test_focal_length_matches_the_horizontal_fov():
    # focal = (aperture/2)/tan(hfov/2); 90 degrees on a 20.955 mm aperture is 10.4775 mm
    assert abs(cr.focal_length_for(90.0) - 10.4775) < 1e-6


def test_rgb_from_annotator_data_accepts_both_shapes():
    """Replicator 1.13 returns the array; older releases wrap it in {'data': array}."""
    import numpy as np

    from common.cerberus_eval.camera_rig import rgb_from_annotator_data

    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    assert rgb_from_annotator_data(rgb).shape == (4, 5, 3)
    assert rgb_from_annotator_data({"data": rgb}).shape == (4, 5, 3)

    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    assert rgb_from_annotator_data(rgba).shape == (2, 3, 3)
    assert rgb_from_annotator_data(rgba).dtype == np.uint8


def test_rgb_from_annotator_data_handles_odd_layouts():
    import numpy as np

    from common.cerberus_eval.camera_rig import rgb_from_annotator_data

    chw = np.zeros((3, 16, 20), dtype=np.uint8)
    assert rgb_from_annotator_data(chw).shape == (16, 20, 3)
    gray = np.zeros((8, 9), dtype=np.uint8)
    assert rgb_from_annotator_data(gray).shape == (8, 9, 3)
    one = np.zeros((8, 9, 1), dtype=np.uint8)
    assert rgb_from_annotator_data(one).shape == (8, 9, 3)
    floats = np.full((4, 5, 3), 0.5, dtype=np.float32)
    out = rgb_from_annotator_data(floats)
    assert out.dtype == np.uint8 and out.max() in (127, 128)


def test_update_uses_the_passed_pose_and_logs_a_world_ego_pose():
    """The chase camera must follow the pose the caller measured, not the (stale) USD xform."""
    import numpy as np

    from common.cerberus_eval.camera_rig import CameraRig, CameraSpec, rotate_vector

    rig = CameraRig.__new__(CameraRig)  # no Kit: exercise only the arithmetic
    rig.specs = {
        "chase": CameraSpec("chase", "chase", (640, 480), hfov_deg=70.0, distance=1.5, height=1.2,
                            pitch_down_deg=15.0),
        "ego": CameraSpec("ego", "ego", (640, 480), hfov_deg=90.0, offset=(0.38, 0.0, 0.30)),
    }
    rig._chase_yaw = None
    rig._authored = {}
    written = {}
    rig._write_xform = lambda name, pos, quat: written.__setitem__(name, np.asarray(pos, dtype=float))
    rig.update(base_pos=np.array([2.0, 1.0, 0.4]), base_quat_wxyz=(1.0, 0.0, 0.0, 0.0))

    chase = written["chase"]
    assert np.linalg.norm(chase[:2] - np.array([2.0, 1.0])) == pytest.approx(1.5, abs=1e-6)
    ego_pos, _ = rig.world_pose("ego")
    assert ego_pos == pytest.approx([2.38, 1.0, 0.70])

    # a quarter turn about z moves the ego offset to +y in world coordinates
    rotated = rotate_vector((0.38, 0.0, 0.3), (0.7071067811865476, 0.0, 0.0, 0.7071067811865476))
    assert rotated == pytest.approx([0.0, 0.38, 0.3], abs=1e-6)
