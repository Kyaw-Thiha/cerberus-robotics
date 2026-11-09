"""Tests for the first-step assertions (mocked joint lists and synthetic observations)."""

import pytest

from common.cerberus_eval.scene_assertions import (
    AssertionReport,
    assert_all,
    assert_joint_order,
    build_report,
    check_command_slice,
    check_finite,
    check_joint_order,
    check_observation_dim,
    check_term_layout,
    contract_default_joint_pos,
    contract_joint_order,
    contract_terms,
    height_scan_report,
    joint_order_problems,
    load_contract,
    platform_joint_order,
    term_offsets,
)

GOOD = contract_joint_order()


def test_contract_and_platforms_agree_on_joint_order():
    # The two files are separate sources of truth; the comment in platforms.yaml promises a test keeps them in sync.
    assert platform_joint_order() == GOOD
    assert len(GOOD) == 12


def test_contract_default_positions_cover_every_joint():
    assert set(contract_default_joint_pos()) == set(GOOD)


def test_correct_joint_order_has_no_problems():
    assert check_joint_order(GOOD) == []


def test_swapped_joint_order_is_reported_at_the_first_difference():
    swapped = list(GOOD)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    problems = check_joint_order(swapped)
    assert problems
    assert "joint 0" in problems[0]


def test_missing_joint_is_reported():
    problems = check_joint_order(GOOD[:-1])
    assert any("11 joints vs 12" in p for p in problems)
    assert any("missing joints" in p for p in problems)


def test_extra_joint_is_reported():
    problems = joint_order_problems([*GOOD, "tail_joint"], GOOD, "contract")
    assert any("unexpected joints" in p for p in problems)


def test_assert_joint_order_raises_loudly():
    swapped = list(GOOD)
    swapped[2], swapped[5] = swapped[5], swapped[2]
    with pytest.raises(AssertionError, match="joint order does not match"):
        assert_joint_order(swapped)


def test_observation_dim_check():
    assert check_observation_dim(235) == []
    assert check_observation_dim(33)


def test_term_layout_is_contract_order():
    terms = contract_terms()
    assert check_term_layout(terms) == []
    assert [t["name"] for t in terms] == [
        "base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
        "joint_pos", "joint_vel", "actions", "height_scan",
    ]


def test_term_layout_detects_reordering():
    terms = list(contract_terms())
    terms[3], terms[4] = terms[4], terms[3]
    assert check_term_layout(terms)


def test_term_offsets_place_the_command_at_9_to_12():
    offsets = term_offsets(contract_terms())
    assert offsets["velocity_commands"] == (9, 12)
    assert offsets["height_scan"] == (48, 235)


def obs_with_command(vx=0.3, vy=-0.2, wz=0.1):
    row = [0.0] * 235
    row[9:12] = [vx, vy, wz]
    return row


def test_command_slice_matches_the_injected_command():
    assert check_command_slice(obs_with_command(), {"vx": 0.3, "vy": -0.2, "wz": 0.1}) == []


def test_command_slice_catches_a_reordered_observation():
    problems = check_command_slice(obs_with_command(), {"vx": -0.2, "vy": 0.3, "wz": 0.1})
    assert problems and "injected command" in problems[0]


def test_height_scan_on_flat_ground_reports_the_constant_and_the_matching_convention():
    scan = [-0.15] * 187
    report = height_scan_report(scan, base_height_m=0.35, offset_m=0.5)
    assert report["constant"] is True
    assert report["mean"] == pytest.approx(-0.15)
    assert report["at_clip_low"] == 0 and report["at_clip_high"] == 0
    assert "base_minus_offset" in report["matches"]
    assert report["convention_status"] == "unverified"


def test_height_scan_records_clip_saturation():
    report = height_scan_report([1.0] * 187, base_height_m=0.35)
    assert report["at_clip_high"] == 187


def test_height_scan_without_a_base_height_still_reports_measurements():
    report = height_scan_report([0.2] * 187)
    assert report["mean"] == pytest.approx(0.2)
    assert report["matches"] == []


def test_height_scan_rejects_an_empty_scan():
    with pytest.raises(ValueError):
        height_scan_report([])


def test_check_finite_finds_nan_and_inf_rows():
    assert check_finite([[0.0, 1.0], [2.0, 3.0]]) == []
    problems = check_finite([[0.0, float("nan")], [float("inf"), 1.0]])
    assert len(problems) == 2
    assert "row 0" in problems[0]


def test_build_report_passes_with_a_consistent_mock():
    report = build_report(
        joint_names=GOOD,
        obs_dim=235,
        obs_row=obs_with_command(),
        command={"vx": 0.3, "vy": -0.2, "wz": 0.1},
        scan=[-0.15] * 187,
        base_height_m=0.35,
        stand_rows=[[0.0] * 235] * 200,
    )
    assert report.passed, report.failures()
    assert report.soft["height_scan_flat_ground"]["constant"] is True
    assert set(report.soft) == {"height_scan_flat_ground"}


def test_build_report_skips_checks_it_has_no_data_for():
    report = build_report(joint_names=GOOD, obs_dim=235)
    assert report.passed
    assert "command_slice" in report.soft and "stand_finite" in report.soft


def test_build_report_fails_on_a_swapped_joint_order():
    swapped = list(GOOD)
    swapped[0], swapped[3] = swapped[3], swapped[0]
    report = build_report(joint_names=swapped, obs_dim=235)
    assert not report.passed
    with pytest.raises(AssertionError, match="joint_order"):
        assert_all(report)


def test_report_round_trips_to_json(tmp_path):
    report = build_report(joint_names=GOOD, obs_dim=235)
    path = report.write(tmp_path / "assertions.json")
    assert '"passed": true' in path.read_text()
    assert AssertionReport().passed  # an empty report has nothing failing


def test_load_contract_reads_the_checked_in_file():
    contract = load_contract()
    assert contract["observation"]["total_dim"] == 235
    assert contract["height_scan"]["convention_status"] == "unverified"
