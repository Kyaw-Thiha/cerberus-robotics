from __future__ import annotations

import pytest

from common.cerberus_worlds import admission as adm


def _row(scene_id="grscenes:a", status=adm.STATUS_PASS, category="home", **kw):
    poses = (adm.SpawnPose(1.0, 2.0, 0.5, 0.01),) if status == adm.STATUS_PASS else ()
    return adm.AdmissionRow(
        scene_id=scene_id, source="grscenes", category=category, backend="plain_usd",
        status=status, spawn_poses=poses, reasons=kw.pop("reasons", ()), **kw,
    )


def test_round_trip(tmp_path):
    rows = [_row("grscenes:b"), _row("grscenes:a", adm.STATUS_FAIL, reasons=("build timeout",))]
    path = adm.write_manifest("grscenes", rows, tool_versions={"x": 1}, generated_utc="t", root=tmp_path)
    assert path == tmp_path / "grscenes" / "admission.json"
    loaded = adm.load_manifest("grscenes", tmp_path)
    assert [r.scene_id for r in loaded] == ["grscenes:a", "grscenes:b"]  # sorted
    assert loaded[0].status == adm.STATUS_FAIL and loaded[0].reasons == ("build timeout",)
    assert loaded[1].spawn_poses[0] == adm.SpawnPose(1.0, 2.0, 0.5, 0.01)


def test_pass_row_requires_spawn_pose():
    with pytest.raises(ValueError, match="spawn pose"):
        adm.AdmissionRow(scene_id="s", source="x", status=adm.STATUS_PASS)


def test_bad_status_and_nonfinite_pose():
    with pytest.raises(ValueError):
        adm.AdmissionRow(scene_id="s", source="x", status="maybe")
    with pytest.raises(ValueError):
        adm.SpawnPose(float("nan"), 0.0, 0.0, 0.0)


def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        adm.write_manifest("grscenes", [_row(), _row()], root=tmp_path)


def test_admitted_ids_and_spawn_pose(tmp_path):
    rows = [_row("grscenes:a"), _row("grscenes:b", adm.STATUS_FAIL), _row("grscenes:c", category="commercial")]
    adm.write_manifest("grscenes", rows, root=tmp_path)
    assert adm.admitted_scene_ids("grscenes", root=tmp_path) == ["grscenes:a", "grscenes:c"]
    assert adm.admitted_scene_ids("grscenes", "home", root=tmp_path) == ["grscenes:a"]
    assert adm.spawn_pose("grscenes", "grscenes:a", root=tmp_path).floor_z == 0.01
    with pytest.raises(ValueError, match="not admitted"):
        adm.spawn_pose("grscenes", "grscenes:b", root=tmp_path)
    with pytest.raises(KeyError):
        adm.spawn_pose("grscenes", "grscenes:zzz", root=tmp_path)


def test_schema_version_checked(tmp_path):
    p = adm.manifest_path("grscenes", tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text('{"schema_version": 99, "scenes": []}')
    with pytest.raises(ValueError, match="schema_version"):
        adm.load_manifest("grscenes", tmp_path)


def test_write_scene_ids_file_returns_count_and_is_empty_when_empty(tmp_path):
    # The upload step must branch on the count: an empty list has to be a 0-byte file, because a lone newline
    # passes a size/`-s` test and is then read as an empty (misused) id list by the consumer.
    path = tmp_path / "admitted.txt"
    assert adm.write_scene_ids_file(path, ["a", "b"]) == 2
    assert path.read_text() == "a\nb\n"
    assert adm.write_scene_ids_file(path, []) == 0
    assert path.read_text() == ""
    assert path.stat().st_size == 0
