import json
import sys

import pytest
import record_scripted_demos_go2_arx5 as rs


def parse(*argv):
    return rs.build_parser(with_app_args=False).parse_args(list(argv))


def test_module_import_is_isaac_free():
    assert "isaaclab" not in sys.modules
    assert "omni" not in sys.modules


def test_parser_defaults_and_flags():
    args = parse()
    assert args.task == "Cerberus-Go2Arx5-Stack-Cube-IK-Rel-v0"
    assert args.num_demos == 10 and args.seed == 1
    assert args.output == rs.DEFAULT_OUTPUT and args.report == rs.DEFAULT_REPORT
    assert args.headless is False and args.overwrite is False
    assert args.grasp_pitch_deg == pytest.approx(90.0)
    args = parse(
        "--num_demos",
        "3",
        "--output",
        "x/y.hdf5",
        "--seed",
        "7",
        "--headless",
        "--report",
        "r.json",
    )
    assert (args.num_demos, args.output, args.seed, args.headless, args.report) == (
        3,
        "x/y.hdf5",
        7,
        True,
        "r.json",
    )


def test_validate_args():
    assert rs.validate_args(parse()) == []
    assert rs.validate_args(parse("--num_demos", "0"))
    assert rs.validate_args(parse("--output", "demos.txt"))
    assert rs.validate_args(parse("--hover_height", "0"))
    assert rs.validate_args(parse("--close_pause_steps", "0"))
    assert rs.validate_args(parse("--max_attempts", "-1"))


def test_attempt_budget():
    assert rs.attempt_budget(parse("--num_demos", "4")) == 17
    assert rs.attempt_budget(parse("--num_demos", "4", "--max_attempts", "6")) == 6


def test_pick_place_config_follows_args():
    cfg = rs.pick_place_config(
        parse(
            "--hover_height",
            "0.12",
            "--grasp_z_offset",
            "0.01",
            "--open_pause_steps",
            "20",
        )
    )
    assert (
        cfg.hover_height == 0.12
        and cfg.grasp_z_offset == 0.01
        and cfg.open_pause_steps == 20
    )
    assert (
        cfg.cube_size == rs.layout.CUBE_SIZE
        and cfg.ik_scale == rs.layout.IK_ACTION_SCALE
    )


@pytest.mark.parametrize(
    "requested, recorded, hdf5, error, ok, needle",
    [
        (3, 3, {"num_demos": 3, "success": [True] * 3}, None, True, None),
        (3, 2, {"num_demos": 2, "success": [True] * 2}, None, False, "recorded 2 of 3"),
        (
            3,
            3,
            {"num_demos": 2, "success": [True] * 2},
            None,
            False,
            "output file has 2 demos",
        ),
        (3, 3, {"error": "OSError()"}, None, False, "could not read the output file"),
        (3, 3, None, None, False, "could not read the output file"),
        (
            3,
            3,
            {"num_demos": 3, "success": [True, False, True]},
            None,
            False,
            "not flagged successful",
        ),
        (3, 0, None, "boom", False, "exception: boom"),
    ],
)
def test_evaluate_run(requested, recorded, hdf5, error, ok, needle):
    passed, reasons = rs.evaluate_run(requested, recorded, hdf5, error)
    assert passed is ok
    if needle:
        assert any(needle in r for r in reasons)
    else:
        assert reasons == []


def test_report_structure_and_json_roundtrip():
    attempts = [
        {"index": 0, "status": "success", "steps": 200},
        {
            "index": 1,
            "status": "script_failed",
            "steps": 50,
            "failure_reason": "timeout in descend_pick",
        },
        {"index": 2, "status": "success", "steps": 190},
    ]
    report = rs.build_report(
        {"num_demos": 2, "output": "o.hdf5"},
        attempts,
        2,
        {"num_demos": 2, "success": [True, True]},
        {"torch": "2"},
    )
    assert report["passed"] is True
    assert report["attempt_status_counts"] == {"success": 2, "script_failed": 1}
    assert report["num_demos_recorded"] == 2 and report["output"] == "o.hdf5"
    assert json.loads(json.dumps(report)) == report
    failed = rs.build_report({"num_demos": 5}, attempts, 2, None, {}, error="x")
    assert failed["passed"] is False and failed["error"] == "x"


def test_summarize_hdf5_reads_back_a_dataset(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "d.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(
            {"env_name": "Cerberus-Go2Arx5-Stack-Cube-IK-Rel-v0"}
        )
        for i, n in enumerate((120, 130)):
            demo = data.create_group(f"demo_{i}")
            demo.attrs["num_samples"] = n
            demo.attrs["success"] = True
    info = rs.summarize_hdf5(path)
    assert (
        info["num_demos"] == 2
        and info["num_samples"] == [120, 130]
        and info["success"] == [True, True]
    )
    assert info["demo_keys"] == ["demo_0", "demo_1"]
    assert info["env_args"]["env_name"].startswith("Cerberus-Go2Arx5")
    assert "error" in rs.summarize_hdf5(tmp_path / "missing.hdf5")
