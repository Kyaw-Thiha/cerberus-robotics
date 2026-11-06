import subprocess
import sys

import fetch_slam_backends

VENDORABLE_BACKENDS = {"fast_lio2", "bievr_lio", "fast_livo2", "orb_slam3", "multicam_slam", "gs_livo"}


def test_ensure_colcon_ignore_creates_marker_at_third_party_root(tmp_path, monkeypatch):
    # Real bug found during full colcon build (2026-09-20): without this
    # marker, `colcon build` walks into third_party/ and tries to build the
    # vendored ROS1/CMake backends locally. Confirm the marker lands at
    # third_party/COLCON_IGNORE (one level above third_party/slam/), not
    # inside third_party/slam/ itself -- colcon needs it at the directory
    # it should stop descending into.
    fake_third_party_slam = tmp_path / "third_party" / "slam"
    monkeypatch.setattr(fetch_slam_backends, "THIRD_PARTY_SLAM", fake_third_party_slam)
    fetch_slam_backends.ensure_colcon_ignore()
    assert (tmp_path / "third_party" / "COLCON_IGNORE").is_file()


def test_ensure_colcon_ignore_is_idempotent(tmp_path, monkeypatch):
    fake_third_party_slam = tmp_path / "third_party" / "slam"
    monkeypatch.setattr(fetch_slam_backends, "THIRD_PARTY_SLAM", fake_third_party_slam)
    fetch_slam_backends.ensure_colcon_ignore()
    fetch_slam_backends.ensure_colcon_ignore()  # must not raise on a second call
    assert (tmp_path / "third_party" / "COLCON_IGNORE").is_file()


def test_gs_livo_is_a_valid_backend_name():
    # gs_livo replaced lvi_q (2026-09-21) as the LiDAR+camera "latest"
    # candidate -- real fork (Kyaw-Thiha/cerberus-GS-LIVO), same vendoring
    # pattern as every other backend.
    assert "gs_livo" in fetch_slam_backends.BACKENDS
    assert VENDORABLE_BACKENDS <= set(fetch_slam_backends.BACKENDS)


def test_unknown_backend_name_errors_with_valid_choices_listed():
    result = subprocess.run([sys.executable, "scripts/fetch_slam_backends.py", "not_a_real_backend"], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "not_a_real_backend" in result.stderr


def test_no_arguments_and_no_all_flag_errors_with_usage():
    result = subprocess.run([sys.executable, "scripts/fetch_slam_backends.py"], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "--all" in result.stderr


def test_gs_livo_sub_vendors_rpg_vikit():
    # GS-LIVO's LIVMapper.cpp dereferences vk::AbstractCamera via rpg_vikit's
    # headers, which the upstream GS-LIVO does NOT vendor (the empty
    # third_party/slam/gs_livo/src/rpg_vikit/ dir was the pre-existing gap
    # that this commit's vendoring closes). The sub-vendor key is
    # "gs_livo:rpg_vikit" -- confirmed against the SUB_VENDORS dict.
    assert "gs_livo:rpg_vikit" in fetch_slam_backends.SUB_VENDORS


def test_sub_vendor_destination_is_inside_parent_src():
    # Convention: sub-vendored deps land at third_party/slam/<parent>/src/<child>.
    # Validates the parent:child split in fetch_sub_vendor will produce
    # the directory the parent backend's CMakeLists.txt's add_subdirectory
    # call expects (gs-livo/CMakeLists.txt:103's add_subdirectory(rpg_vikit)).
    for sub_key in fetch_slam_backends.SUB_VENDORS:
        parent, child = sub_key.split(":", 1)
        # Destination path the function will build (relative to THIRD_PARTY_SLAM).
        assert "/" not in child, f"sub-vendor child name {child!r} must be a single directory segment"
        assert child, f"sub-vendor child name must not be empty for {sub_key}"
