"""Isaac Lab video capture for SLAM eval runs, mirroring
policy/locomotion/core/video_capture.py's "write frames directly via imageio"
convention -- but sourcing frames from the showcase camera's CameraCfg sensor
output (Task 5's SHOWCASE_CAMERA_CFG) each physics step, not from
viewport_camera_controller (that mechanism is Phase-0-specific: it watches
one fixed env's viewport for a run's whole lifetime, which doesn't apply
here -- the showcase camera is a real moving sensor prim, per §8's soft-follow
chase cam, not the shared viewport).

Actual per-step frame-writing loop (env.step() -> read the showcase camera's
rgb tensor -> imageio.append_data(...)) needs a live Isaac Lab env and is a
pod-side integration step, out of scope here -- same constraint as
ground_truth/pose_export.py and showcase_camera_follow.py. This module only
contains the locally-testable path-naming logic that keeps output aligned
with scripts/sync_to_r2.py's expected run-directory shape.
"""

from __future__ import annotations

from pathlib import Path


def build_frame_writer_path(backend_name: str, mission: str, run_timestamp: str) -> Path:
    """Mirrors docs/artifact_storage.md's local-layout-mirrors-R2 rule:
    slam/slam/eval/results/<backend_name>/<run_timestamp>/videos/<mission>_showcase.mp4
    so `sync_to_r2.py sync-run --module slam --run-dir
    slam/slam/eval/results/<backend_name>/<run_timestamp>` uploads it under
    videos/ automatically, same shape as policy/locomotion's own runs."""
    return Path("slam/slam/eval/results") / backend_name / run_timestamp / "videos" / f"{mission}_showcase.mp4"
