"""Backend-free SLAM plumbing smoke check: did every sensor topic the rig should publish actually arrive, and did the
sim side run as intended (scene, spawn, PhysX + legacy actuators, policy type, robot upright at the end).

Used by ``scripts/run_slam_eval_remote.sh --smoke``, which runs the sim loop + the rclpy relay + this module's
recorder (``record_mission_bag.py --smoke``) with NO SLAM backend and NO scoring. Two steps:

1. ``collect_topic_stats`` (under the sim-ros2 env, concurrent with the sim): subscribes to every topic and keeps
   per-topic message counts and first/last header stamps (sim time). It stops when the sim has exited (a sentinel
   file the shell script creates), not on a wall-clock timer: the sim usually runs slower than real time.
2. ``python3 slam/slam/eval/smoke_check.py finalize --run-dir <dir> ...`` merges those stats, the rosbag2 metadata
   (counts actually recorded) and the sim loop's ``sim_loop_report.json`` into ``smoke_check.json``, prints one line
   per check and exits non-zero if anything failed.

Thresholds are provisional (recorded in the JSON): a topic passes with at least ``MIN_FRACTION`` of
``rate * mission_duration`` messages, because the relay publishes every env step and a busy subscriber may drop some.
``rclpy`` is imported only inside ``collect_topic_stats``; everything else is plain python.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STATS_NAME = "smoke_topics.json"
REPORT_NAME = "smoke_check.json"
SIM_REPORT_NAME = "sim_loop_report.json"

MIN_FRACTION = 0.5
LIDAR_RATE_HZ = 10.0  # LIDAR_CFG update_period 1/10; published only when the sensor really updated
STEP_RATE_HZ = 50.0  # sim 0.005 s x decimation 4: pose, tf and IMU are published on every env step
CAMERA_RATE_HZ = 30.0  # CameraCfg update_period 1/30 (the loop publishes every step, so this is a floor)

# Provisional sim-side bounds (m above the scene floor, degrees): the frozen policy crouches to 0.11-0.17 m while moving.
BASE_HEIGHT_RANGE_M = (0.08, 0.6)
MAX_FINAL_TILT_DEG = 30.0
MAX_ANY_TILT_DEG = 60.0

GT_ODOM_TOPIC = "/ground_truth/odom"
TF_TOPIC = "/tf"


def topic_rate_hz(topic: str) -> float:
    if topic == "/lidar/points":
        return LIDAR_RATE_HZ
    if topic in (GT_ODOM_TOPIC, TF_TOPIC, "/imu/data"):
        return STEP_RATE_HZ
    return CAMERA_RATE_HZ


def smoke_topics(raw_sensor_topics: list[str]) -> list[str]:
    """Every topic a backend-free run must see: the ground truth the relay publishes plus the rig's raw sensors."""
    return [GT_ODOM_TOPIC, TF_TOPIC, *raw_sensor_topics]


@dataclass
class TopicStats:
    count: int = 0
    first_stamp_s: float | None = None
    last_stamp_s: float | None = None

    def add(self, stamp_s: float) -> None:
        self.count += 1
        if self.first_stamp_s is None:
            self.first_stamp_s = stamp_s
        self.last_stamp_s = stamp_s


def stamp_seconds(msg: Any) -> float:
    """Header stamp of a message in seconds; a TFMessage has none of its own, so use its first transform's."""
    header = getattr(msg, "header", None)
    if header is None:
        header = msg.transforms[0].header
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def collect_topic_stats(
    topic_types: dict[str, type],
    done_file: Path,
    boot_timeout_s: float,
    run_timeout_s: float,
    drain_s: float = 10.0,
) -> dict[str, Any]:
    """Count messages per topic until the sim is done. Needs rclpy (sim-ros2 env).

    Stops, with ``stop_reason``: ``boot_timeout`` (no ground truth within ``boot_timeout_s``), ``run_timeout``, or
    ``sim_exited`` (``done_file`` exists and the ground-truth stream has been quiet for 2 s, or ``drain_s`` passed).
    """
    import rclpy

    stats = {topic: TopicStats() for topic in topic_types}
    rclpy.init()
    node = rclpy.create_node("smoke_topic_counter")
    try:
        for topic, msg_type in topic_types.items():
            node.create_subscription(msg_type, topic, lambda msg, topic=topic: stats[topic].add(stamp_seconds(msg)), 100)
        start = time.monotonic()
        last_gt_wall = start
        gt_count = 0
        done_seen = None
        stop_reason = "run_timeout"
        while True:
            rclpy.spin_once(node, timeout_sec=0.2)
            now = time.monotonic()
            if stats[GT_ODOM_TOPIC].count != gt_count:
                gt_count = stats[GT_ODOM_TOPIC].count
                last_gt_wall = now
            if gt_count == 0 and now - start > boot_timeout_s:
                stop_reason = "boot_timeout"
                break
            if now - start > run_timeout_s:
                break
            if done_file.exists():
                done_seen = now if done_seen is None else done_seen
                if now - done_seen >= drain_s or (now - last_gt_wall > 2.0 and now - done_seen >= 1.0):
                    stop_reason = "sim_exited"
                    break
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return {
        "stop_reason": stop_reason,
        "topics": {
            topic: {"count": s.count, "first_stamp_s": s.first_stamp_s, "last_stamp_s": s.last_stamp_s}
            for topic, s in stats.items()
        },
    }


def read_bag_counts(bag_dir: Path) -> dict[str, int] | None:
    """Per-topic message counts rosbag2 wrote into ``<bag>/metadata.yaml``; None if unreadable."""
    try:
        import yaml

        info = yaml.safe_load((bag_dir / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
        return {t["topic_metadata"]["name"]: int(t["message_count"]) for t in info["topics_with_message_count"]}
    except (OSError, ImportError, KeyError, TypeError, ValueError):
        return None


def evaluate_topics(
    stats: dict[str, Any], duration_s: float, bag_counts: dict[str, int] | None
) -> dict[str, dict[str, Any]]:
    results = {}
    for topic, s in stats["topics"].items():
        expected_min = math.ceil(MIN_FRACTION * topic_rate_hz(topic) * duration_s)
        bag = None if bag_counts is None else bag_counts.get(topic, 0)
        reasons = []
        if s["count"] == 0:
            reasons.append("ZERO messages received")
        elif s["count"] < expected_min:
            reasons.append(f"only {s['count']} messages, need >= {expected_min}")
        if bag is None:
            reasons.append("bag metadata unreadable")
        elif bag == 0:
            reasons.append("nothing recorded in the bag")
        results[topic] = {
            **s,
            "bag_count": bag,
            "expected_rate_hz": topic_rate_hz(topic),
            "expected_min_count": expected_min,
            "passed": not reasons,
            "reasons": reasons,
        }
    return results


def evaluate_sim(sim_report: dict[str, Any] | None, expected_scene: str) -> list[str]:
    """Failure reasons from the sim loop's own report (empty list = the sim side did what the run asked for)."""
    if sim_report is None:
        return [f"no {SIM_REPORT_NAME}: the sim loop never wrote its report"]
    failures = []
    meta = sim_report.get("metadata", {})
    if not sim_report.get("passed"):
        failures.append(f"sim loop failed: {sim_report.get('failure_reasons')}")
    if meta.get("scene_id") != expected_scene:
        failures.append(f"scene_id {meta.get('scene_id')!r} != requested {expected_scene!r}")
    if "physx" not in str(meta.get("physics_cfg", "")).lower():
        failures.append(f"physics is {meta.get('physics_cfg')!r}, not PhysX")
    if meta.get("use_newton_actuators") is not False:
        failures.append(f"use_newton_actuators is {meta.get('use_newton_actuators')!r}, not False")
    if sim_report.get("steps_completed") != meta.get("total_steps"):
        failures.append(f"steps_completed {sim_report.get('steps_completed')} != total_steps {meta.get('total_steps')}")
    collision = sim_report.get("collision")
    if collision is None:
        failures.append("no collision block in the sim report (the contact/clearance check did not run)")
    elif not collision.get("contact_sensor"):
        failures.append("no contact sensor in the env: the collision check could not run, so this run is unscoreable")
    final = sim_report.get("final_state")
    if final is None:
        failures.append("no final_state in the sim report (mission did not finish)")
    else:
        low, high = BASE_HEIGHT_RANGE_M
        if not low <= final["base_height_above_floor_m"] <= high:
            failures.append(f"final base height {final['base_height_above_floor_m']:.3f} m outside [{low}, {high}]")
        if final["tilt_deg"] > MAX_FINAL_TILT_DEG:
            failures.append(f"final tilt {final['tilt_deg']:.1f} deg > {MAX_FINAL_TILT_DEG}")
        if final["max_tilt_deg"] > MAX_ANY_TILT_DEG:
            failures.append(f"max tilt {final['max_tilt_deg']:.1f} deg > {MAX_ANY_TILT_DEG}")
    return failures


def finalize(run_dir: Path, expected_scene: str) -> tuple[dict[str, Any], bool]:
    stats = json.loads((run_dir / STATS_NAME).read_text())
    topics = evaluate_topics(stats, stats["duration_s"], read_bag_counts(run_dir / "bag"))
    sim_path = run_dir / SIM_REPORT_NAME
    sim_report = json.loads(sim_path.read_text()) if sim_path.is_file() else None
    sim_failures = evaluate_sim(sim_report, expected_scene)
    meta = (sim_report or {}).get("metadata", {})
    report = {
        "passed": all(t["passed"] for t in topics.values()) and not sim_failures and stats["stop_reason"] == "sim_exited",
        "stop_reason": stats["stop_reason"],
        "duration_s": stats["duration_s"],
        "thresholds": {
            "min_fraction": MIN_FRACTION,
            "base_height_range_m": BASE_HEIGHT_RANGE_M,
            "max_final_tilt_deg": MAX_FINAL_TILT_DEG,
            "max_any_tilt_deg": MAX_ANY_TILT_DEG,
        },
        "topics": topics,
        "sim_failures": sim_failures,
        "sim": {
            "scene_id": meta.get("scene_id"),
            "usd_path": meta.get("usd_path"),
            "spawn": meta.get("spawn"),
            "spawn_source": meta.get("spawn_source"),
            "physics_cfg": meta.get("physics_cfg"),
            "physics_manager": meta.get("physics_manager"),
            "use_newton_actuators": meta.get("use_newton_actuators"),
            "checkpoint": meta.get("checkpoint"),
            "checkpoint_kind": meta.get("checkpoint_kind"),
            "steps_completed": (sim_report or {}).get("steps_completed"),
            "final_state": (sim_report or {}).get("final_state"),
            "collision": (sim_report or {}).get("collision"),
        },
    }
    return report, report["passed"]


def print_report(report: dict[str, Any]) -> None:
    for topic, t in report["topics"].items():
        verdict = "ok  " if t["passed"] else "FAIL"
        span = f"stamps {t['first_stamp_s']}..{t['last_stamp_s']}" if t["count"] else "no stamps"
        why = f" -- {'; '.join(t['reasons'])}" if t["reasons"] else ""
        print(f"[smoke] {verdict} {topic}: {t['count']} msgs (need >= {t['expected_min_count']}), bag {t['bag_count']}, {span}{why}")
    if report["stop_reason"] != "sim_exited":
        print(f"[smoke] FAIL collector stopped on {report['stop_reason']!r}, not because the sim exited")
    for failure in report["sim_failures"]:
        print(f"[smoke] FAIL sim: {failure}")
    collision = report["sim"].get("collision")
    if collision is not None:
        # Not a smoke failure by itself: the canned open-loop schedule is generic, and only the non-smoke path
        # refuses to score an invalid run (see run_slam_eval_remote.sh). Reported so it cannot go unnoticed.
        verdict = "ok" if collision["valid_for_scoring"] else f"INVALID ({','.join(collision['invalid_reasons'])})"
        print(
            f"[smoke] collision check: {verdict}; {collision['collision_events']} event(s) over "
            f"{collision['collision_steps']} step(s), bodies {collision['bodies_in_collision']}, "
            f"min clearance {collision['min_clearance_m']} m"
        )
    print(f"[smoke] sim: {json.dumps(report['sim'], sort_keys=True, default=str)}")
    print(f"[smoke] {'PASS' if report['passed'] else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    final = sub.add_parser("finalize", help="merge topic stats + bag metadata + the sim report; exit 1 on any failed check")
    final.add_argument("--run-dir", required=True, type=Path)
    final.add_argument("--expect-scene", required=True, help="scene id the run was asked for ('stock' or grscenes:<id>)")
    args = parser.parse_args(argv)
    try:
        report, ok = finalize(args.run_dir, args.expect_scene)
    except (OSError, ValueError, KeyError) as e:
        print(f"[smoke] FAIL cannot finalize: {type(e).__name__}: {e} (did the recorder write {STATS_NAME}?)", file=sys.stderr)
        return 1
    (args.run_dir / REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print_report(report)
    print(f"[smoke] report: {args.run_dir / REPORT_NAME}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
