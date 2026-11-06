from pathlib import Path

from slam.slam.eval.showcase_camera_video import build_frame_writer_path


def test_build_frame_writer_path_matches_r2_sync_run_dir_shape():
    # Must land under a path scripts/sync_to_r2.py's sync-run can walk as one
    # run directory (module=slam), same shape as
    # policy/locomotion/checkpoints/<experiment>/<run_timestamp>/videos/ --
    # per docs/artifact_storage.md's mirrored-layout requirement.
    path = build_frame_writer_path(backend_name="fast_lio2", mission="household", run_timestamp="2026-09-20_10-00-00")
    assert path == Path("slam/slam/eval/results/fast_lio2/2026-09-20_10-00-00/videos/household_showcase.mp4")


def test_build_frame_writer_path_differs_per_backend_and_run():
    a = build_frame_writer_path(backend_name="fast_lio2", mission="household", run_timestamp="2026-09-20_10-00-00")
    b = build_frame_writer_path(backend_name="orb_slam3", mission="household", run_timestamp="2026-09-20_10-00-00")
    assert a != b
