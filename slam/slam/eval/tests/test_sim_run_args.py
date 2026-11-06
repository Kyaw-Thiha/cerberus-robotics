import os

import pytest

from common.cerberus_worlds import admission
from common.cerberus_worlds.admission import AdmissionRow, SpawnPose
from slam.slam.eval import sim_run_args as sra

SCENE = "grscenes:house_a_usd"

METRIC_USDA = "#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = \"Z\"\n)\n"  # a real (empty) metre-scale stage: scene USDs are opened for unit checks


@pytest.fixture(autouse=True)
def _restore_mdl_system_path(monkeypatch):
    """resolve_scene sets MDL_SYSTEM_PATH (a real process-env side effect); setenv registers the undo."""
    monkeypatch.setenv("MDL_SYSTEM_PATH", "")


@pytest.fixture
def scenes_dir(tmp_path, monkeypatch):
    """data/scenes layout with one scene on disk, a Materials dir, and an admission manifest."""
    monkeypatch.setenv("CERBERUS_SCENES_DIR", str(tmp_path))
    home = tmp_path / "grscenes" / "home"
    (home / "Materials").mkdir(parents=True)
    scene = home / "scenes" / "house_a_usd"
    scene.mkdir(parents=True)
    (scene / "start_result_navigation.usd").write_text(METRIC_USDA)
    (home / "scenes" / "house_b_usd").mkdir()
    (home / "scenes" / "house_b_usd" / "start_result_navigation.usd").write_text(METRIC_USDA)
    rows = [
        AdmissionRow(
            scene_id=SCENE, source="grscenes", category="home", backend="plain_usd", status="pass",
            spawn_poses=(SpawnPose(1.0, 2.0, 0.5, 0.01), SpawnPose(3.0, 4.0, -1.0, 0.02)),
        ),
        AdmissionRow(scene_id="grscenes:house_b_usd", source="grscenes", backend="plain_usd", status="fail", reasons=("fell",)),
    ]
    admission.write_manifest("grscenes", rows, root=tmp_path)
    return tmp_path


def test_parse_spawn_override():
    assert sra.parse_spawn_override("1.5, -2,0.25,0.01") == SpawnPose(1.5, -2.0, 0.25, 0.01)
    for bad in ("1,2,3", "1,2,3,4,5", "a,b,c,d", "1,2,nan,4"):
        with pytest.raises(sra.RunArgError):
            sra.parse_spawn_override(bad)


def test_checkpoint_kind_dispatches_on_extension():
    assert sra.checkpoint_kind("a/b/model_1500.pt") == "pt"
    assert sra.checkpoint_kind("policy/locomotion/checkpoints/go2_locomotion_flat_v1_candidate.onnx") == "onnx"
    assert sra.checkpoint_kind("MODEL.ONNX") == "onnx"
    for bad in ("model", "model.pth", "model.onnx.bak"):
        with pytest.raises(sra.RunArgError, match="--checkpoint"):
            sra.checkpoint_kind(bad)


def test_task_ids_per_rig():
    assert sra.task_id_for_sensor_rig("lidar_camera") == sra.TASK_ID_LIDAR_CAMERA
    assert sra.task_id_for_sensor_rig("camera_only") == sra.TASK_ID_CAMERA_ONLY
    assert sra.task_id_for_sensor_rig("camera_stereo") == sra.TASK_ID_CAMERA_STEREO
    assert len({sra.TASK_ID_LIDAR_CAMERA, sra.TASK_ID_CAMERA_ONLY, sra.TASK_ID_CAMERA_STEREO}) == 3


def test_stock_scene_needs_no_spawn():
    resolved = sra.resolve_scene("stock", None, None)
    assert (resolved.scene_id, resolved.scene, resolved.spawn, resolved.spawn_source) == ("stock", None, None, "stock")
    for kwargs in ({"spawn_index": 0, "spawn_override": None}, {"spawn_index": None, "spawn_override": "0,0,0,0"}):
        with pytest.raises(sra.RunArgError, match="only apply to a scene"):
            sra.resolve_scene("stock", **kwargs)


def test_scene_spawn_comes_from_the_manifest(scenes_dir):
    first = sra.resolve_scene(SCENE, None, None)
    assert first.spawn == SpawnPose(1.0, 2.0, 0.5, 0.01) and first.spawn_source == "manifest:grscenes[0]"
    assert first.scene["usd_path"].endswith("house_a_usd/start_result_navigation.usd")
    second = sra.resolve_scene(SCENE, 1, None)
    assert second.spawn == SpawnPose(3.0, 4.0, -1.0, 0.02) and second.spawn_source == "manifest:grscenes[1]"
    assert second.to_metadata()["spawn"] == {"x": 3.0, "y": 4.0, "yaw": -1.0, "floor_z": 0.02}


def test_scene_sets_mdl_system_path_before_app_launch(scenes_dir):
    sra.resolve_scene(SCENE, None, "0,0,0,0")
    assert os.environ["MDL_SYSTEM_PATH"].endswith("grscenes/home/Materials")


def test_spawn_override_replaces_the_manifest(scenes_dir):
    resolved = sra.resolve_scene(SCENE, None, "9,8,0.1,0.0")
    assert resolved.spawn == SpawnPose(9.0, 8.0, 0.1, 0.0) and resolved.spawn_source == "override"
    with pytest.raises(sra.RunArgError, match="mutually exclusive"):
        sra.resolve_scene(SCENE, 0, "9,8,0.1,0.0")


def test_spawn_override_works_without_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("CERBERUS_SCENES_DIR", str(tmp_path))
    scene = tmp_path / "grscenes" / "home" / "scenes" / "x_usd"
    scene.mkdir(parents=True)
    (scene / "start_result_raw.usd").write_text(METRIC_USDA)
    assert sra.resolve_scene("grscenes:x_usd", None, "1,2,0,0").spawn.x == 1.0


def test_scene_never_falls_back_to_origin(scenes_dir, tmp_path):
    with pytest.raises(sra.RunArgError, match="not admitted"):
        sra.resolve_scene("grscenes:house_b_usd", None, None)
    with pytest.raises(sra.RunArgError, match="no spawn pose at --spawn-index 2"):
        sra.resolve_scene(SCENE, 2, None)
    with pytest.raises(sra.RunArgError, match=">= 0"):
        sra.resolve_scene(SCENE, -1, None)
    (scenes_dir / "grscenes" / "admission.json").unlink()
    with pytest.raises(sra.RunArgError, match="no admission manifest"):
        sra.resolve_scene(SCENE, None, None)


def test_manifest_without_the_scene_is_an_error(scenes_dir):
    extra = scenes_dir / "grscenes" / "home" / "scenes" / "house_c_usd"
    extra.mkdir()
    (extra / "start_result_navigation.usd").write_text(METRIC_USDA)
    with pytest.raises(sra.RunArgError, match="house_c_usd"):
        sra.resolve_scene("grscenes:house_c_usd", None, None)


def test_unknown_or_foreign_scene_refs(scenes_dir):
    with pytest.raises(sra.RunArgError, match="not found"):
        sra.resolve_scene("grscenes:nope_usd", None, "0,0,0,0")
    with pytest.raises(sra.RunArgError, match="Lightwheel"):
        sra.resolve_scene("lightwheel:1:1", None, "0,0,0,0")
    with pytest.raises(sra.RunArgError, match="Unknown scene"):
        sra.resolve_scene("house", None, "0,0,0,0")


def test_report_path_prefers_override_then_run_dir():
    assert sra.report_path(None, None, "/tmp/r.json").as_posix() == "/tmp/r.json"
    run = sra.report_path("fast_lio2", "2026-10-06_10-00-00", None)
    assert run.as_posix().endswith("slam/slam/eval/results/fast_lio2/2026-10-06_10-00-00/sim_loop_report.json")
    assert sra.report_path("fast_lio2", None, None).as_posix().endswith("data/reports/slam_sim_loop.json")


def test_write_report_round_trips_and_reports_failure(tmp_path):
    path = tmp_path / "a" / "r.json"
    assert sra.write_report(path, {"passed": True, "p": tmp_path}) is True
    assert '"passed": true' in path.read_text()
    blocker = tmp_path / "file"
    blocker.write_text("")
    assert sra.write_report(blocker / "r.json", {}) is False


def test_negative_spawn_is_accepted_with_the_equals_form():
    """run_slam_eval_remote.sh passes --spawn=VALUE; with a space, argparse reads a leading '-' as an option."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--spawn")
    assert parser.parse_args(["--spawn=-4.157,1.327,0,0"]).spawn == "-4.157,1.327,0,0"
    with pytest.raises(SystemExit):
        parser.parse_args(["--spawn", "-4.157,1.327,0,0"])
