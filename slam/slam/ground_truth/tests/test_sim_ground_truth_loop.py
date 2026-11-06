import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from common.cerberus_worlds import admission
from common.cerberus_worlds.admission import AdmissionRow, SpawnPose
from slam.slam.ground_truth import sim_ground_truth_loop as loop

METRIC_USDA = "#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = \"Z\"\n)\n"  # a real (empty) metre-scale stage: scene USDs are opened for unit checks


def _args(tmp_path, **overrides):
    checkpoint = tmp_path / "policy.onnx"
    checkpoint.write_text("")
    argv = ["--mission", "household", "--sensor-rig", "lidar_camera", "--checkpoint", str(checkpoint)]
    args, _ = loop.build_parser().parse_known_args(argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_prepare_run_stock_defaults(tmp_path):
    plan = loop.prepare_run(_args(tmp_path))
    assert plan.task_id == "Cerberus-Go2-SlamEval-LidarCamera-v0"
    assert plan.checkpoint_kind == "onnx" and plan.scene.scene_id == "stock"
    assert plan.duration_s == pytest.approx(21.0)  # 19 s household schedule + 2 s settle
    assert plan.scene.spawn is None


def test_prepare_run_pt_checkpoint_and_settle(tmp_path):
    args = _args(tmp_path, settle_seconds=0.0)
    pt = tmp_path / "model_1500.pt"
    pt.write_text("")
    args.checkpoint = str(pt)
    plan = loop.prepare_run(args)
    assert plan.checkpoint_kind == "pt" and plan.duration_s == pytest.approx(19.0)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"mission": "nope"}, "unknown mission"),
        ({"checkpoint": "model.bin"}, "--checkpoint must end in"),
        ({"checkpoint": "/does/not/exist.pt"}, "is not a file"),
        ({"settle_seconds": -1.0}, "settle"),
        ({"spawn": "0,0,0,0"}, "only apply to a scene"),
    ],
)
def test_prepare_run_rejects_bad_args(tmp_path, overrides, message):
    with pytest.raises(loop.RunArgError, match=message):
        loop.prepare_run(_args(tmp_path, **overrides))


def test_prepare_run_scene_metadata_carries_scene_and_spawn(tmp_path, monkeypatch):
    monkeypatch.setenv("CERBERUS_SCENES_DIR", str(tmp_path))
    monkeypatch.setenv("MDL_SYSTEM_PATH", "")
    scene = tmp_path / "grscenes" / "home" / "scenes" / "h_usd"
    scene.mkdir(parents=True)
    (scene / "start_result_navigation.usd").write_text(METRIC_USDA)
    admission.write_manifest(
        "grscenes",
        [AdmissionRow(scene_id="grscenes:h_usd", source="grscenes", status="pass", spawn_poses=(SpawnPose(1.0, 2.0, 0.3, 0.0),))],
        root=tmp_path,
    )
    args = _args(tmp_path, scene="grscenes:h_usd", backend="fast_lio2", run_timestamp="t0")
    meta = loop.prepare_run(args).metadata(args)
    assert meta["scene_id"] == "grscenes:h_usd"
    assert meta["spawn"] == {"x": 1.0, "y": 2.0, "yaw": 0.3, "floor_z": 0.0}
    assert meta["spawn_source"] == "manifest:grscenes[0]"
    assert meta["usd_path"].endswith("start_result_navigation.usd")
    assert meta["report"].endswith("results/fast_lio2/t0/sim_loop_report.json")


def test_main_dry_run_prints_plan_without_isaac(tmp_path, capsys):
    checkpoint = tmp_path / "p.onnx"
    checkpoint.write_text("")
    code = loop.main(["--mission", "household", "--sensor-rig", "camera_only", "--checkpoint", str(checkpoint), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    plan = json.loads(out.split("run plan:\n", 1)[1])
    assert plan["task_id"].endswith("CameraOnly-v0") and plan["scene_id"] == "stock"


def test_main_dry_run_bad_scene_exits_2(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("CERBERUS_SCENES_DIR", str(tmp_path))
    checkpoint = tmp_path / "p.pt"
    checkpoint.write_text("")
    argv = ["--mission", "household", "--sensor-rig", "camera_only", "--checkpoint", str(checkpoint), "--dry-run"]
    assert loop.main([*argv, "--scene", "grscenes:missing_usd"]) == 2
    assert "not found" in capsys.readouterr().err


def test_lidar_points_are_finite_and_in_the_yaw_frame():
    hits_w = np.array(
        [[11.0, 5.0, 0.0], [np.inf, np.inf, np.inf], [10.0, 7.0, 0.5], [np.nan, 0.0, 0.0]],
        dtype=np.float32,
    )
    sensor = np.array([10.0, 5.0, 0.4], dtype=np.float32)
    points, keep = loop.lidar_points_in_sensor_frame(hits_w, sensor, math.pi / 2)
    assert keep.tolist() == [True, False, True, False]
    assert points.dtype == np.float32 and points.shape == (2, 3)
    # robot yawed +90 deg: a hit 1 m along world +x is 1 m to the robot's right (-y); 2 m along world +y is dead ahead
    np.testing.assert_allclose(points, [[0.0, -1.0, -0.4], [2.0, 0.0, 0.1]], atol=1e-6)


def test_lidar_ring_mask_stays_aligned_with_points():
    hits_w = np.array([[1.0, 0, 0], [np.inf, 0, 0], [2.0, 0, 0], [3.0, 0, 0]], dtype=np.float32)
    points, keep = loop.lidar_points_in_sensor_frame(hits_w, np.zeros(3, dtype=np.float32), 0.0)
    ring = (np.arange(4) // 2)[keep]  # 2 channels x 2 azimuth samples
    assert len(ring) == len(points) == 3


def test_yaw_from_quat():
    assert loop._yaw_from_quat_wxyz([math.cos(0.4), 0.0, 0.0, math.sin(0.4)]) == pytest.approx(0.8)


def test_check_spawn_accepts_the_spawn_and_rejects_a_stray_robot():
    env_cfg = SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(init_state=SimpleNamespace(pos=(2.0, 3.0, 0.45)))))
    spawn = SpawnPose(2.0, 3.0, 0.0, 0.05)
    loop._check_spawn([2.01, 3.0, 0.44], spawn, env_cfg)
    with pytest.raises(RuntimeError, match="wrong env origin"):
        loop._check_spawn([0.0, 0.0, 0.44], spawn, env_cfg)
    with pytest.raises(RuntimeError, match="dropped through the floor"):
        loop._check_spawn([2.0, 3.0, -0.5], spawn, env_cfg)


def test_tilt_from_quat():
    assert loop.tilt_deg_from_quat_wxyz([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert loop.tilt_deg_from_quat_wxyz([math.cos(0.5), 0.0, 0.0, math.sin(0.5)]) == pytest.approx(0.0, abs=1e-6)  # pure yaw
    roll = math.radians(40.0)
    assert loop.tilt_deg_from_quat_wxyz([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]) == pytest.approx(40.0)
    assert loop.tilt_deg_from_quat_wxyz([0.0, 1.0, 0.0, 0.0]) == pytest.approx(180.0)


class _Scene(dict):
    """A ``raw_env.scene`` stand-in: ``scene.sensors`` and ``scene["lidar"]`` both work."""

    def __getattr__(self, name):
        return self[name]


def _env_with(**sensors):
    return SimpleNamespace(scene=_Scene(sensors=sensors))


def _contact_sensor(body_names, forces):
    torch = pytest.importorskip("torch")
    return SimpleNamespace(body_names=body_names, data=SimpleNamespace(net_forces_w=torch.tensor([forces], dtype=torch.float32)))


def test_contact_offenders_reports_non_foot_bodies_only():
    torch = pytest.importorskip("torch")
    env = _env_with(contact_forces=_contact_sensor(["base", "FL_foot", "FL_thigh"], [[50.0, 0.0, 0.0], [0.0, 900.0, 0.0], [6.0, 0.0, 0.0]]))
    assert loop._contact_offenders(env) == ["base", "FL_thigh"]
    assert loop._contact_offenders(env) is not None  # a clean-ish robot still answers (the verdict, not None)
    assert torch.isfinite(torch.tensor(0.0))


def test_contact_offenders_is_unknown_rather_than_clean_without_a_usable_sensor():
    assert loop._contact_offenders(_env_with()) is None  # no contact_forces at all: no torch needed to answer
    mismatch = _env_with(contact_forces=_contact_sensor(["base", "head"], [[1.0, 0.0, 0.0]]))
    assert loop._contact_offenders(mismatch) is None  # forces cannot be attributed to bodies
    empty = _env_with(contact_forces=SimpleNamespace(body_names=["base"], data=SimpleNamespace(net_forces_w=None)))
    assert loop._contact_offenders(empty) is None  # sensor exists but never updated


def test_lidar_clearance_gates_on_the_sensor_s_own_update_time():
    torch = pytest.importorskip("torch")
    hits = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [2.0, 0.0, 0.2]]], dtype=torch.float32)
    lidar = SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits), _timestamp_last_update=SimpleNamespace(numpy=lambda: np.array([1.0])))
    env = _env_with(lidar=lidar)
    loop._lidar_clearance._last_update_time = None
    assert loop._lidar_clearance(env, np.array([0.0, 0.0]), 0.0) == pytest.approx(1.0)
    assert loop._lidar_clearance(env, np.array([0.0, 0.0]), 0.0) is None  # same scan: no new sample
    lidar._timestamp_last_update = SimpleNamespace(numpy=lambda: np.array([2.0]))
    assert loop._lidar_clearance(env, np.array([0.0, 0.0]), 0.0) == pytest.approx(1.0)


def test_lidar_clearance_is_none_for_a_rig_without_a_lidar():
    assert loop._lidar_clearance(_env_with(), np.array([0.0, 0.0]), 0.0) is None

