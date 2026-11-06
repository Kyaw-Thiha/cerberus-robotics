import pytest

from slam.slam.eval.scripted_motion import MotionSegment, canned_schedule, command_at


def test_canned_schedule_household_is_forward_turn_forward_revisit():
    schedule = canned_schedule("household")
    assert [s.label for s in schedule] == ["forward", "turn", "forward", "revisit"]
    assert all(isinstance(s, MotionSegment) and s.duration_seconds > 0 for s in schedule)


def test_canned_schedule_is_identical_across_calls():
    assert canned_schedule("household") == canned_schedule("household")


def test_canned_schedule_unknown_mission_raises():
    with pytest.raises(ValueError, match="unknown mission 'nope'"):
        canned_schedule("nope")


def test_command_at_returns_first_segments_command_at_time_zero():
    schedule = canned_schedule("household")
    assert command_at(schedule, elapsed_seconds=0.0) == schedule[0].command


def test_command_at_advances_to_next_segment_after_duration_elapses():
    schedule = canned_schedule("household")
    assert command_at(schedule, elapsed_seconds=schedule[0].duration_seconds + 0.01) == schedule[1].command


def test_command_at_past_schedule_end_returns_zero_command():
    schedule = canned_schedule("household")
    total = sum(s.duration_seconds for s in schedule)
    assert command_at(schedule, elapsed_seconds=total + 1.0) == (0.0, 0.0, 0.0)
