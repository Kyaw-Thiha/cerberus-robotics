"""Tests for the replay selection (the Kit log capture tests live in scripts/host/tests/test_kit_logs.py)."""

from nav_eval import select_replay_indices


def test_replay_selection_by_outcome():
    records = [{"outcome": "fall"}, {"outcome": "success"}, {"outcome": "timeout"}, {"outcome": "success"}]
    assert select_replay_indices(records, "", "success,failure") == [1, 0]
    assert select_replay_indices(records, "3,1", "success,failure") == [3, 1]
    assert select_replay_indices(records, "", "failure") == [0]
