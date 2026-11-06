"""Tests for the SLAM eval mission's collision/clearance bookkeeping (pure numpy, no Isaac)."""

import numpy as np
import pytest

from slam.slam.ground_truth import collision_monitor as cm


class TestOffendingBodies:
    def test_feet_are_ignored_whatever_their_case(self):
        names = ["base", "FL_foot", "RL_FOOT", "FL_thigh"]
        forces = [0.0, 500.0, 500.0, 0.0]
        assert cm.offending_bodies(names, forces) == []

    def test_threshold_is_exclusive_and_strictly_above_it_offends(self):
        names = ["base", "head"]
        assert cm.offending_bodies(names, [5.0, 5.0]) == []
        assert cm.offending_bodies(names, [5.1, 0.0]) == ["base"]
        assert cm.offending_bodies(names, [0.0, 5.1]) == ["head"]

    def test_nan_counts_as_offending_and_mismatched_lengths_raise(self):
        assert cm.offending_bodies(["base"], [float("nan")]) == ["base"]
        with pytest.raises(ValueError, match="2 body names but 1 force values"):
            cm.offending_bodies(["base", "head"], [1.0])


class TestBandClearance:
    BASE_XY = (0.0, 0.0)

    def test_nearest_in_band_hit_wins_and_the_floor_is_excluded(self):
        hits = np.array(
            [
                [0.05, 0.0, 0.0],  # on the floor: below the band, must not become a 5 cm "obstacle"
                [1.0, 0.0, 0.20],  # in the band
                [0.4, 0.3, 0.30],  # in the band, nearest
                [3.0, 0.0, 5.0],  # above the band (a ceiling/overhead)
            ]
        )
        assert cm.band_clearance_m(hits, self.BASE_XY, floor_z=0.0) == pytest.approx(0.5)

    def test_misses_and_out_of_band_hits_give_none(self):
        assert cm.band_clearance_m(np.full((3, 3), np.inf), self.BASE_XY, floor_z=0.0) is None
        assert cm.band_clearance_m(np.empty((0, 3)), self.BASE_XY, floor_z=0.0) is None
        below = np.array([[1.0, 0.0, 0.01]])
        assert cm.band_clearance_m(below, self.BASE_XY, floor_z=0.0) is None

    def test_band_is_measured_from_the_scene_floor_and_survives_nan(self):
        hits = np.array([[np.nan, 0.0, 0.2], [2.0, 0.0, 1.21]])
        assert cm.band_clearance_m(hits, self.BASE_XY, floor_z=1.0) == pytest.approx(2.0)


class TestCollisionMonitorSummary:
    def test_no_contact_sensor_is_reported_as_unknown_not_as_clean(self):
        monitor = cm.CollisionMonitor()
        monitor.update(0.0, None)
        summary = monitor.summary()
        assert summary["contact_sensor"] is False
        assert summary["collision_events"] == 0
        assert summary["valid_for_scoring"] is False
        assert summary["invalid_reasons"] == ["no_contact_sensor"]

    def test_clean_run_is_valid_and_records_clearance(self):
        monitor = cm.CollisionMonitor()
        for step in range(10):
            monitor.update(step * 0.02, [], clearance_m=1.0 - 0.01 * step)
        summary = monitor.summary()
        assert summary["valid_for_scoring"] is True and summary["invalid_reasons"] == []
        assert summary["collision_steps"] == 0 and summary["collision_events"] == 0
        assert summary["bodies_in_collision"] == []
        assert summary["min_clearance_m"] == pytest.approx(0.91)
        assert summary["clearance_samples"] == 10

    def test_one_continuous_contact_is_one_event_with_its_bodies_and_time(self):
        monitor = cm.CollisionMonitor()
        monitor.update(0.00, [])
        monitor.update(0.02, ["base"])  # event starts
        monitor.update(0.04, ["base", "FL_thigh"])
        monitor.update(0.06, [])  # contact ends
        monitor.update(0.08, ["base", "FL_thigh"])  # a second, separate event
        summary = monitor.summary()
        assert summary["collision_steps"] == 3
        assert summary["collision_events"] == 2
        assert summary["first_collision_t_s"] == pytest.approx(0.02)
        assert summary["bodies_in_collision"] == ["FL_thigh", "base"]  # sorted
        assert summary["valid_for_scoring"] is False and summary["invalid_reasons"] == ["collision"]

    def test_a_missing_clearance_measurement_is_not_counted_as_a_sample(self):
        monitor = cm.CollisionMonitor()
        monitor.update(0.0, ["base"], clearance_m=None)
        summary = monitor.summary()
        assert summary["min_clearance_m"] is None and summary["clearance_samples"] == 0
        assert summary["contact_sensor"] is True  # the sensor answered; only the clearance rig was missing
        assert summary["invalid_reasons"] == ["collision"]
