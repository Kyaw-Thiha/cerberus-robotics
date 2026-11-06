#!/usr/bin/env python3
"""Records one SLAM eval mission as a rosbag2 bag: every backend's contract
topics (/slam/odom, /tf, /slam/map), the RAW sensor feeds, /ground_truth/odom,
and the showcase camera. It does NOT drive the robot: sim_ground_truth_loop.py
applies the canned schedule itself (its process has no rclpy). It records until
`--sim-done-file` appears -- the file scripts/run_slam_eval_remote.sh creates once the
sim loop has exited -- plus a short drain, NOT for a wall-clock mission length: the
sim usually runs slower than real time, so a timer truncates the bag. `--run-timeout-s`
is the hard cap.

With --smoke (scripts/run_slam_eval_remote.sh --smoke) no SLAM backend runs: it records and counts only the
relay's own topics, see slam/slam/eval/smoke_check.py.

Output path is derived entirely from --run-timestamp (single source of
truth, so Task 10's R2 sync always finds what this script wrote):
    slam/slam/eval/results/<backend>/<run_timestamp>/bag/
"""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# Needed when this file is invoked by its own relative path (e.g. `python3
# slam/slam/eval/record_mission_bag.py`, as scripts/run_slam_eval_remote.sh
# does) -- Python puts the SCRIPT's own directory on sys.path[0], not the
# repo root, so the `from slam.slam...` absolute import below fails with
# "ModuleNotFoundError: No module named 'slam'" otherwise. Confirmed by
# hitting exactly that against a real pod run. Same fix
# sim_ground_truth_loop.py's own header already carries, for the same
# reason.

from nav_msgs.msg import Odometry

from slam.slam.eval import smoke_check
from slam.slam.eval.scripted_motion import canned_schedule

CONTRACT_TOPICS = ["/slam/odom", "/tf", "/slam/map", "/ground_truth/odom", "/showcase_camera/rgb"]
# PLACEHOLDER -- match this repo's existing sensor-mount naming convention, but NOT independently confirmed against a real
# Isaac-Lab-to-ROS2 publishing layer, which doesn't exist yet (see this
# plan's Explicitly-out-of-scope section: Dockerfile.remote has no ROS2).
# Confirm/correct once that publishing layer exists, same as Task 6c's
# per-backend topic-name placeholders.
RAW_SENSOR_TOPICS_LIDAR_CAMERA = ["/lidar/points", "/imu/data", "/front_camera/rgb", "/front_camera/depth"]
RAW_SENSOR_TOPICS_CAMERA_ONLY = ["/front_camera/rgb", "/front_camera/depth", "/rear_camera/rgb", "/rear_camera/depth"]
# multicam_slam's rig -- CAMERA_STEREO_SUITE's genuine stereo pairs. Depth
# topics NOT included: multicam_slam consumes real stereo-computed depth
# from stereo_to_depth_node.py (a separate node's output, not a raw sensor
# topic this bridge publishes), never Isaac Lab's privileged sim depth --
# see slam/slam/backends/multicam_slam/stereo_to_depth_node.py's docstring.
RAW_SENSOR_TOPICS_CAMERA_STEREO = [
    "/front_camera/rgb", "/front_camera_right/rgb",
    "/rear_camera/rgb", "/rear_camera_right/rgb",
]

SETTLE_SECONDS = 2.0  # the sim loop's default --settle-seconds (smoke mission length = schedule + this)
DRAIN_SECONDS = 5.0   # after the sim has exited: let the relay's last messages reach the recorder before stopping it


def run_dir_for(backend: str, run_timestamp: str) -> Path:
    return Path("slam/slam/eval/results") / backend / run_timestamp


def record_until_sim_done(record_cmd: list[str], done_file: Path, run_timeout_s: float) -> bool:
    """Run `ros2 bag record` until `done_file` exists (+ DRAIN_SECONDS) or the hard cap; True if the sim finished."""
    record_proc = subprocess.Popen(record_cmd)
    start = time.monotonic()
    finished = False
    try:
        while time.monotonic() - start < run_timeout_s:
            if done_file.exists():
                finished = True
                time.sleep(DRAIN_SECONDS)
                break
            time.sleep(1.0)
    finally:
        record_proc.send_signal(signal.SIGINT)  # let rosbag2 flush cleanly
        record_proc.wait(timeout=30)
    return finished


def run_smoke(args: argparse.Namespace, raw_topics: list[str], total_duration: float) -> int:
    """Backend-free smoke: record the relay's topics and count them per topic until `--sim-done-file` appears.

    Always returns 0 once the stats are written; smoke_check.py finalize judges them. The mission is NOT driven from
    here (sim_ground_truth_loop.py applies its own schedule) and nothing is synced to R2.
    """
    from sensor_msgs.msg import Image, Imu, PointCloud2
    from tf2_msgs.msg import TFMessage

    topics = smoke_check.smoke_topics(raw_topics)
    known_types = {"/lidar/points": PointCloud2, "/imu/data": Imu, smoke_check.GT_ODOM_TOPIC: Odometry, smoke_check.TF_TOPIC: TFMessage}
    topic_types = {topic: known_types.get(topic, Image) for topic in topics}  # every other raw topic is an image
    run_dir = run_dir_for(args.backend, args.run_timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    record_cmd = ["ros2", "bag", "record", "-o", str(run_dir / "bag"), *topics]
    print(f"[record_mission_bag] smoke: {' '.join(record_cmd)}")
    record_proc = subprocess.Popen(record_cmd)
    try:
        stats = smoke_check.collect_topic_stats(
            topic_types, Path(args.sim_done_file), boot_timeout_s=args.run_timeout_s, run_timeout_s=args.run_timeout_s
        )
    finally:
        record_proc.send_signal(signal.SIGINT)  # let rosbag2 flush cleanly (metadata.yaml is read afterwards)
        record_proc.wait(timeout=30)
    stats["duration_s"] = total_duration + SETTLE_SECONDS  # the loop's default --settle-seconds
    (run_dir / smoke_check.STATS_NAME).write_text(json.dumps(stats, indent=2))
    print(f"[record_mission_bag] smoke: collector stopped on {stats['stop_reason']}; stats in {run_dir / smoke_check.STATS_NAME}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--mission", required=True)
    parser.add_argument("--run-timestamp", required=True, help="e.g. 2026-09-20_10-00-00 -- determines the bag's output path, see module docstring")
    parser.add_argument("--sensor-rig", choices=["lidar_camera", "camera_only", "camera_stereo"], required=True)
    parser.add_argument("--sim-done-file", required=True, help="a file the caller creates once the sim loop has exited; recording stops after it appears")
    parser.add_argument("--run-timeout-s", type=float, default=3600.0, help="hard cap on the recording")
    parser.add_argument("--smoke", action="store_true", help="backend-free smoke mode")
    args = parser.parse_args()

    raw_topics = {
        "lidar_camera": RAW_SENSOR_TOPICS_LIDAR_CAMERA,
        "camera_only": RAW_SENSOR_TOPICS_CAMERA_ONLY,
        "camera_stereo": RAW_SENSOR_TOPICS_CAMERA_STEREO,
    }[args.sensor_rig]
    all_topics = CONTRACT_TOPICS + raw_topics

    schedule = canned_schedule(args.mission)  # validate the mission name early, before spawning anything
    total_duration = sum(segment.duration_seconds for segment in schedule)
    if args.smoke:
        return run_smoke(args, raw_topics, total_duration)

    run_dir = run_dir_for(args.backend, args.run_timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    record_cmd = ["ros2", "bag", "record", "-o", str(run_dir / "bag"), *all_topics]
    print(f"[record_mission_bag] backend={args.backend} mission={args.mission} rig={args.sensor_rig} duration={total_duration}s")
    print(f"[record_mission_bag] {' '.join(record_cmd)}")
    finished = record_until_sim_done(record_cmd, Path(args.sim_done_file), args.run_timeout_s)

    sync_cmd = ["python3", "scripts/sync_to_r2.py", "sync-run", "--module", "slam", "--run-dir", str(run_dir)]
    print(f"[record_mission_bag] syncing to R2: {' '.join(sync_cmd)}")
    subprocess.run(sync_cmd, check=False)  # per run_remote_job.sh's own pattern: warn, don't crash the run, on a sync failure

    if not finished:
        print(f"[record_mission_bag] ERROR: the sim did not finish within {args.run_timeout_s:.0f}s", file=sys.stderr)
        return 1
    recorded = smoke_check.read_bag_counts(run_dir / "bag")
    if not recorded or recorded.get(smoke_check.GT_ODOM_TOPIC, 0) == 0:
        print(f"[record_mission_bag] ERROR: no {smoke_check.GT_ODOM_TOPIC} message in the bag -- the sim never published (crashed, or a topic-name mismatch)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
