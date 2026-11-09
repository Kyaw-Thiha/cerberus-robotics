"""Tests for clip selection and naming (pure Python; no Isaac, no imageio)."""

from pathlib import Path

import pytest

from common.cerberus_eval.video_clips import (
    ClipRecorder,
    clip_and_metrics_paths,
    clip_filename,
    metrics_filename,
    outcome_counts,
    plan_clip_requests,
    plan_clips,
    safe_name,
    summarise_clips,
)


def test_keeps_the_first_success_and_the_first_failure():
    # the first failure here is the opening timeout, not the later fall: the rule is "first of each", not
    # "a success and a fall".
    outcomes = ["timeout", "success", "fall", "success", "timeout"]
    assert plan_clips(outcomes, max_clips=2) == [0, 1]


def test_keeps_the_first_failure_after_the_first_success():
    outcomes = ["success", "timeout", "fall", "success"]
    assert plan_clips(outcomes, max_clips=2) == [0, 1]


def test_fills_to_the_requested_count_in_episode_order():
    outcomes = ["fall", "timeout", "success", "fall", "success", "timeout"]
    assert plan_clips(outcomes, max_clips=4) == [0, 1, 2, 3]


def test_all_successes_still_produce_clips():
    assert plan_clips(["success"] * 5, max_clips=3) == [0, 1, 2]


def test_all_failures_keep_the_first_failure():
    assert plan_clips(["timeout", "fall", "nan"], max_clips=2) == [0, 1]


def test_zero_clips_disables_recording():
    assert plan_clips(["success", "fall"], max_clips=0) == []


def test_one_clip_prefers_the_first_success():
    assert plan_clips(["fall", "success", "timeout"], max_clips=1) == [1]


def test_one_clip_with_only_failures_takes_the_first():
    assert plan_clips(["fall", "timeout"], max_clips=1) == [0]


def test_more_clips_than_episodes_is_harmless():
    assert plan_clips(["success", "fall"], max_clips=10) == [0, 1]


def test_clip_filename_follows_the_handoff_scheme():
    assert clip_filename("grscenes:MV7J6NIKTKJZ2AABAAAAADQ8_usd", 3, "success") == \
        "MV7J6NIKTKJZ2AABAAAAADQ8_usd_3_success.mp4"
    assert metrics_filename("MV7_usd_3_success.mp4") == "MV7_usd_3_success.json"


def test_safe_name_strips_the_prefix_and_unsafe_characters():
    assert safe_name("grscenes:home/a b") == "home_a_b"
    assert safe_name("///") == "scene"


def test_plan_clip_requests_carries_the_outcome():
    requests = plan_clip_requests("stock", 11, ["timeout", "success"], max_clips=2)
    assert [r.episode_index for r in requests] == [0, 1]
    assert [r.outcome for r in requests] == ["timeout", "success"]
    assert requests[1].filename == "stock_11_success.mp4"


def test_clip_and_metrics_paths_share_a_stem(tmp_path):
    request = plan_clip_requests("stock", 2, ["fall"], max_clips=1)[0]
    clip, metrics = clip_and_metrics_paths(tmp_path, request)
    assert clip == Path(tmp_path) / "stock_2_fall.mp4"
    assert metrics == Path(tmp_path) / "stock_2_fall.json"


def test_summarise_clips_reports_missing_files():
    requests = plan_clip_requests("stock", 1, ["success", "fall"], max_clips=2)
    summary = summarise_clips(requests, [requests[0].filename])
    assert summary["count"] == 1
    assert summary["missing"] == [requests[1].filename]


def test_outcome_counts():
    assert outcome_counts(["success", "fall", "fall"]) == {"success": 1, "fall": 2}


def test_recorder_disabled_is_a_no_op(tmp_path):
    with ClipRecorder(tmp_path, enabled=False) as recorder:
        assert recorder.begin("x.mp4") is None
        assert recorder.end() is None
    assert list(tmp_path.iterdir()) == []


def test_recorder_rejects_a_bad_downscale(tmp_path):
    with pytest.raises(ValueError):
        ClipRecorder(tmp_path, downscale=0)
