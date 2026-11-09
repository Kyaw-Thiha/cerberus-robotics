"""Tests for the navigation metrics (pure Python)."""

import json

import pytest

from common.cerberus_eval.nav_metrics import (
    StepState,
    aggregate,
    aggregate_by_scene,
    evaluate_episode,
    planner_failure_result,
    standing_ok,
    write_aggregate_csv,
    write_run_json,
)


def trace(points, *, fallen_at=None, nonfinite_at=None, contact_at=(), base_z=None, dt=0.1):
    return [
        StepState(
            t=i * dt,
            x=x,
            y=y,
            yaw=0.0,
            fallen=(fallen_at is not None and i >= fallen_at),
            nonfinite=(nonfinite_at is not None and i == nonfinite_at),
            contact=i in contact_at,
            base_minus_foot_z=base_z,
        )
        for i, (x, y) in enumerate(points)
    ]


def straight(n=20, step=0.05):
    return [(i * step, 0.0) for i in range(n)]


def test_success_within_tolerance():
    result = evaluate_episode(trace(straight()), (1.0, 0.0), tolerance_m=0.5, shortest_path_m=1.0)
    assert result["outcome"] == "success"
    assert result["reached_goal"] is True
    assert result["steps"] == 11  # x=0.50 is the first step inside 0.5 m
    assert result["path_length_m"] == pytest.approx(0.5)
    assert result["spl"] == pytest.approx(1.0)  # capped at 1: cannot beat the shortest path


def test_spl_is_shortest_over_travelled():
    result = evaluate_episode(trace(straight(n=40, step=0.05)), (1.0, 0.0), shortest_path_m=1.0)
    assert result["outcome"] == "success"
    assert result["path_length_m"] == pytest.approx(0.5)
    detour = evaluate_episode(trace([(0.0, 0.0), (1.0, 0.0)]), (1.0, 0.0), shortest_path_m=1.0)
    assert detour["spl"] == pytest.approx(1.0)


def test_path_length_is_truncated_at_the_deciding_step():
    points = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (1.0, 0.0), (5.0, 0.0)]
    result = evaluate_episode(trace(points), (0.4, 0.0), tolerance_m=0.1, shortest_path_m=0.4)
    assert result["outcome"] == "success"
    assert result["path_length_m"] == pytest.approx(0.4)  # not the 5 m the trace continues to
    assert result["steps"] == 3


def test_success_then_fall_is_a_fall_and_is_flagged():
    result = evaluate_episode(trace(straight(n=20), fallen_at=15), (1.0, 0.0), shortest_path_m=1.0)
    assert result["outcome"] == "fall"
    assert result["reached_goal"] is True
    assert result["fell_after_success"] is True
    assert result["spl"] == 0.0


def test_fall_before_success_is_a_fall():
    result = evaluate_episode(trace(straight(n=20), fallen_at=3), (1.0, 0.0), shortest_path_m=1.0)
    assert result["outcome"] == "fall"
    assert result["reached_goal"] is False
    assert result["fell_after_success"] is False


def test_nonfinite_pose_is_not_a_success_even_at_the_goal():
    result = evaluate_episode(trace(straight(n=20), nonfinite_at=9), (1.0, 0.0), shortest_path_m=1.0)
    assert result["outcome"] == "nan"
    assert result["nonfinite"] is True
    assert result["success"] is False


def test_timeout_reports_final_distance():
    result = evaluate_episode(trace(straight(n=5)), (5.0, 0.0), shortest_path_m=5.0)
    assert result["outcome"] == "timeout"
    assert result["final_distance_m"] == pytest.approx(4.8)
    assert result["spl"] == 0.0


def test_contacts_are_counted_only_up_to_the_deciding_step():
    result = evaluate_episode(trace(straight(n=20), contact_at={1, 2, 15}), (0.4, 0.0), tolerance_m=0.1)
    assert result["contacts"] == 2


def test_planner_failure_record():
    result = planner_failure_result(notes=["unreachable"])
    assert result["outcome"] == "planner_failure"
    assert result["notes"] == ["unreachable"]


def test_empty_trace_is_rejected():
    with pytest.raises(ValueError):
        evaluate_episode([], (0.0, 0.0))


def test_standing_check_threshold():
    assert standing_ok(0.30)
    assert not standing_ok(0.15)
    assert not standing_ok(0.05)


def test_aggregate_rates_and_mean_spl():
    rows = [
        {"outcome": "success", "spl": 1.0, "time_s": 10.0, "contacts": 0},
        {"outcome": "success", "spl": 0.5, "time_s": 20.0, "contacts": 2},
        {"outcome": "fall", "spl": 0.0, "time_s": 5.0, "contacts": 1, "fell_after_success": False},
        {"outcome": "timeout", "spl": 0.0, "time_s": 60.0, "contacts": 0},
        planner_failure_result(),
    ]
    agg = aggregate(rows)
    assert agg["episodes"] == 5
    assert agg["success_rate"] == pytest.approx(0.4)
    assert agg["spl_mean"] == pytest.approx((1.0 + 0.5 + 0.0 + 0.0) / 5)
    assert agg["fall_rate"] == pytest.approx(0.2)
    assert agg["planner_failure_rate"] == pytest.approx(0.2)
    assert agg["total_static_contacts"] == 3
    assert agg["episodes_with_contacts"] == 2
    # planner failures have no episode time and must not drag the mean
    assert agg["mean_time_s"] == pytest.approx((10 + 20 + 5 + 60) / 4)


def test_aggregate_of_nothing_is_empty():
    assert aggregate([]) == {"episodes": 0}


def test_aggregate_by_scene_adds_an_all_entry():
    rows = [
        {"scene": "a", "outcome": "success", "spl": 1.0, "time_s": 1.0, "contacts": 0},
        {"scene": "b", "outcome": "fall", "spl": 0.0, "time_s": 1.0, "contacts": 0},
    ]
    agg = aggregate_by_scene(rows)
    assert set(agg) == {"a", "b", "all"}
    assert agg["a"]["success_rate"] == 1.0
    assert agg["all"]["episodes"] == 2


def test_writers_round_trip(tmp_path):
    rows = [{"scene": "a", "seed": 1, "pair_index": 0, "outcome": "success", "spl": 1.0, "time_s": 2.0,
             "path_length_m": 1.0, "planned_length_m": 1.0, "contacts": 0, "final_distance_m": 0.1,
             "fell": False, "fell_after_success": False, "nonfinite": False}]
    run_path = write_run_json(tmp_path / "run.json", rows[0])
    assert json.loads(run_path.read_text())["outcome"] == "success"
    csv_path = write_aggregate_csv(tmp_path / "agg.csv", rows)
    header = csv_path.read_text().splitlines()[0]
    assert header.startswith("scene,seed,pair_index,outcome,success,spl")
    assert "True" in csv_path.read_text()
