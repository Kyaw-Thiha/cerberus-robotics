import json
import threading
import time

import pytest

from slam.slam.eval import smoke_check as sc
from slam.slam.eval.record_mission_bag import (
    RAW_SENSOR_TOPICS_CAMERA_STEREO,
    RAW_SENSOR_TOPICS_LIDAR_CAMERA,
)

DURATION = 21.0


def _stats(counts, stop_reason="sim_exited"):
    return {
        "stop_reason": stop_reason,
        "duration_s": DURATION,
        "topics": {t: {"count": c, "first_stamp_s": 0.0 if c else None, "last_stamp_s": 20.9 if c else None} for t, c in counts.items()},
    }


def _good_counts(raw=RAW_SENSOR_TOPICS_LIDAR_CAMERA):
    return {t: int(sc.topic_rate_hz(t) * DURATION) for t in sc.smoke_topics(raw)}


def _sim_report(**overrides):
    report = {
        "passed": True,
        "steps_completed": 1050,
        "metadata": {
            "scene_id": "grscenes:x_usd", "physics_cfg": "PhysxCfg", "use_newton_actuators": False, "total_steps": 1050,
            "spawn": {"x": 0.0, "y": 0.0, "yaw": 0.0, "floor_z": 0.0}, "checkpoint_kind": "onnx",
        },
        "final_state": {"base_height_above_floor_m": 0.27, "tilt_deg": 3.0, "max_tilt_deg": 12.0},
        "collision": {
            "contact_sensor": True, "collision_steps": 0, "collision_events": 0, "bodies_in_collision": [],
            "min_clearance_m": 0.8, "valid_for_scoring": True, "invalid_reasons": [],
        },
    }
    report.update(overrides)
    return report


def test_smoke_topics_cover_ground_truth_and_every_raw_sensor_of_the_rig():
    assert sc.smoke_topics(RAW_SENSOR_TOPICS_LIDAR_CAMERA) == [
        "/ground_truth/odom", "/tf", "/lidar/points", "/imu/data", "/front_camera/rgb", "/front_camera/depth",
    ]
    assert "/front_camera_right/rgb" in sc.smoke_topics(RAW_SENSOR_TOPICS_CAMERA_STEREO)


def test_rates():
    assert [sc.topic_rate_hz(t) for t in ("/lidar/points", "/imu/data", "/tf", "/front_camera/rgb")] == [10.0, 50.0, 50.0, 30.0]


def test_full_rate_passes_and_exactly_half_passes():
    results = sc.evaluate_topics(_stats(_good_counts()), DURATION, {t: 5 for t in _good_counts()})
    assert all(r["passed"] for r in results.values())
    half = {t: sc.math.ceil(0.5 * sc.topic_rate_hz(t) * DURATION) for t in _good_counts()}
    assert all(r["passed"] for r in sc.evaluate_topics(_stats(half), DURATION, {t: 1 for t in half}).values())


def test_zero_messages_and_low_counts_fail_with_a_reason():
    counts = _good_counts()
    counts["/lidar/points"] = 0
    counts["/imu/data"] = 10
    results = sc.evaluate_topics(_stats(counts), DURATION, {t: 5 for t in counts})
    assert results["/lidar/points"]["passed"] is False and "ZERO messages" in results["/lidar/points"]["reasons"][0]
    assert "only 10 messages" in results["/imu/data"]["reasons"][0]
    assert results["/tf"]["passed"] is True


def test_bag_must_have_recorded_each_topic():
    counts = _good_counts()
    bag = {t: 5 for t in counts}
    bag["/front_camera/depth"] = 0
    results = sc.evaluate_topics(_stats(counts), DURATION, bag)
    assert results["/front_camera/depth"]["reasons"] == ["nothing recorded in the bag"]
    assert sc.evaluate_topics(_stats(counts), DURATION, None)["/tf"]["reasons"] == ["bag metadata unreadable"]


def test_read_bag_counts(tmp_path):
    (tmp_path / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n  topics_with_message_count:\n"
        "  - topic_metadata: {name: /tf, type: tf2_msgs/msg/TFMessage}\n    message_count: 12\n"
        "  - topic_metadata: {name: /lidar/points, type: x}\n    message_count: 3\n"
    )
    assert sc.read_bag_counts(tmp_path) == {"/tf": 12, "/lidar/points": 3}
    assert sc.read_bag_counts(tmp_path / "missing") is None


def test_evaluate_sim_accepts_a_good_run():
    assert sc.evaluate_sim(_sim_report(), "grscenes:x_usd") == []


@pytest.mark.parametrize(
    "mutate, text",
    [
        (lambda r: r.update(passed=False, failure_reasons=["boom"]), "sim loop failed"),
        (lambda r: r["metadata"].update(scene_id="stock"), "!= requested"),
        (lambda r: r["metadata"].update(physics_cfg="NewtonCfg"), "not PhysX"),
        (lambda r: r["metadata"].update(use_newton_actuators=True), "use_newton_actuators"),
        (lambda r: r.update(steps_completed=300), "steps_completed"),
        (lambda r: r["final_state"].update(base_height_above_floor_m=-0.4), "base height"),
        (lambda r: r["final_state"].update(tilt_deg=80.0), "final tilt"),
        (lambda r: r["final_state"].update(max_tilt_deg=100.0), "max tilt"),
        (lambda r: r.pop("final_state"), "no final_state"),
        (lambda r: r.pop("collision"), "no collision block"),
        (lambda r: r["collision"].update(contact_sensor=False), "no contact sensor"),
    ],
)
def test_evaluate_sim_flags_each_problem(mutate, text):
    report = _sim_report()
    mutate(report)
    assert any(text in f for f in sc.evaluate_sim(report, "grscenes:x_usd"))


def test_evaluate_sim_without_a_report():
    assert "never wrote its report" in sc.evaluate_sim(None, "stock")[0]


def test_collisions_are_reported_but_do_not_fail_the_plumbing_smoke(capsys):
    # The canned open-loop schedule is generic, so a collision is not a plumbing failure; scoring is refused
    # separately (run_slam_eval_remote.sh reads valid_for_scoring), and the smoke must still say it happened.
    report = _sim_report()
    report["collision"].update(
        collision_events=2, collision_steps=40, bodies_in_collision=["base"],
        valid_for_scoring=False, invalid_reasons=["collision"],
    )
    assert sc.evaluate_sim(report, "grscenes:x_usd") == []
    sc.print_report({"topics": {}, "stop_reason": "sim_exited", "sim_failures": [], "passed": True, "sim": {"collision": report["collision"]}})
    out = capsys.readouterr().out
    assert "collision check: INVALID (collision)" in out and "2 event(s) over 40 step(s)" in out and "['base']" in out



def _write_run(tmp_path, counts, sim_report, stop_reason="sim_exited"):
    (tmp_path / sc.STATS_NAME).write_text(json.dumps(_stats(counts, stop_reason)))
    bag_lines = "".join(
        f"  - topic_metadata: {{name: {t}, type: x}}\n    message_count: {c}\n" for t, c in counts.items()
    )
    (tmp_path / "bag").mkdir()
    (tmp_path / "bag" / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n  topics_with_message_count:\n" + bag_lines
    )
    if sim_report is not None:
        (tmp_path / sc.SIM_REPORT_NAME).write_text(json.dumps(sim_report))


def test_finalize_pass_writes_report_and_exits_zero(tmp_path, capsys):
    _write_run(tmp_path, _good_counts(), _sim_report())
    assert sc.main(["finalize", "--run-dir", str(tmp_path), "--expect-scene", "grscenes:x_usd"]) == 0
    out = capsys.readouterr().out
    assert "[smoke] PASS" in out
    saved = json.loads((tmp_path / sc.REPORT_NAME).read_text())
    assert saved["passed"] is True and saved["sim"]["checkpoint_kind"] == "onnx" and saved["sim"]["spawn"]["floor_z"] == 0.0
    assert saved["sim"]["collision"]["valid_for_scoring"] is True  # carried into smoke_check.json for the record


def test_finalize_dead_sensor_exits_nonzero_with_a_clear_line(tmp_path, capsys):
    counts = _good_counts()
    counts["/front_camera/rgb"] = 0
    _write_run(tmp_path, counts, _sim_report())
    assert sc.main(["finalize", "--run-dir", str(tmp_path), "--expect-scene", "grscenes:x_usd"]) == 1
    out = capsys.readouterr().out
    assert "[smoke] FAIL /front_camera/rgb: 0 msgs" in out and "ZERO messages" in out and "\n[smoke] FAIL\n" in out


def test_finalize_fails_when_the_collector_did_not_see_the_sim_exit(tmp_path, capsys):
    _write_run(tmp_path, _good_counts(), _sim_report(), stop_reason="boot_timeout")
    assert sc.main(["finalize", "--run-dir", str(tmp_path), "--expect-scene", "grscenes:x_usd"]) == 1
    assert "boot_timeout" in capsys.readouterr().out


def test_finalize_without_stats_exits_nonzero(tmp_path, capsys):
    assert sc.main(["finalize", "--run-dir", str(tmp_path), "--expect-scene", "stock"]) == 1
    assert "cannot finalize" in capsys.readouterr().err


def test_collect_topic_stats_counts_real_messages_and_stops_when_the_sim_exits(tmp_path):
    """Real rclpy publishers (local DDS, same precedent as test_record_mission_bag.py)."""
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    from tf2_msgs.msg import TFMessage

    done = tmp_path / ".sim_exited"

    def publish() -> None:
        ctx = rclpy.Context()
        ctx.init()
        node = rclpy.create_node("regtest_smoke_publisher", context=ctx)
        try:
            pubs = {"/ground_truth/odom": (node.create_publisher(Odometry, "/ground_truth/odom", 10), Odometry),
                    "/imu/data": (node.create_publisher(Imu, "/imu/data", 10), Imu)}
            tf_pub = node.create_publisher(TFMessage, "/tf", 10)
            time.sleep(1.0)  # discovery
            for i in range(30):
                for pub, cls in pubs.values():
                    msg = cls()
                    msg.header.stamp.sec, msg.header.stamp.nanosec = 1, i * 20_000_000
                    pub.publish(msg)
                tf = TFMessage()
                tf.transforms.append(TransformStamped())
                tf.transforms[0].header.stamp.sec = 2
                tf_pub.publish(tf)
                time.sleep(0.02)
            done.write_text("")
        finally:
            node.destroy_node()
            ctx.shutdown()

    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    stats = sc.collect_topic_stats(
        {"/ground_truth/odom": Odometry, "/imu/data": Imu, "/tf": TFMessage, "/never": Imu},
        done, boot_timeout_s=30.0, run_timeout_s=60.0, drain_s=5.0,
    )
    thread.join(timeout=5.0)
    assert stats["stop_reason"] == "sim_exited"
    topics = stats["topics"]
    assert topics["/ground_truth/odom"]["count"] > 0 and topics["/imu/data"]["count"] > 0
    assert topics["/tf"]["first_stamp_s"] == 2.0  # a TFMessage has no header of its own
    assert topics["/imu/data"]["first_stamp_s"] == pytest.approx(1.0)
    assert topics["/never"]["count"] == 0 and topics["/never"]["first_stamp_s"] is None


def test_collect_topic_stats_reports_a_sim_that_never_published(tmp_path):
    from nav_msgs.msg import Odometry

    stats = sc.collect_topic_stats({"/ground_truth/odom": Odometry}, tmp_path / "never", boot_timeout_s=0.5, run_timeout_s=5.0)
    assert stats["stop_reason"] == "boot_timeout" and stats["topics"]["/ground_truth/odom"]["count"] == 0
