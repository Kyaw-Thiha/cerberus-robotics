from pathlib import Path

import yaml

import slam.slam.core.algorithms as slam_algorithms

SLAM_BACKENDS_YAML = Path(__file__).resolve().parents[1] / "config" / "slam_backends.yaml"


def _load() -> dict:
    with open(SLAM_BACKENDS_YAML) as f:
        return yaml.safe_load(f)


# The six real candidates -- checked explicitly rather
# than trusting available_algorithms() wholesale, since that registry is a
# shared, process-global dict that slam/slam/core/test_algorithms.py's own
# closure tests deliberately pollute with throwaway names (regtest1, etc.) --
# the same convention every other module's test_algorithms.py uses. Iterating
# available_algorithms() blindly here would fail whenever this test happens to
# run in the same pytest process after those tests already registered their
# scratch entries -- a real cross-test-file ordering bug, not a config gap.
REAL_SLAM_CANDIDATES = {"fast_lio2", "bievr_lio", "fast_livo2", "gs_livo", "orb_slam3", "multicam_slam"}


def test_every_registered_slam_algorithm_has_a_backends_yaml_entry():
    backends = _load()
    for name in REAL_SLAM_CANDIDATES & set(slam_algorithms.available_algorithms()):
        assert name in backends


def test_every_backends_yaml_entry_is_a_registered_slam_algorithm():
    backends = _load()
    known = slam_algorithms.available_algorithms()
    for name in backends:
        assert name in known


def test_every_backend_has_a_real_image_entry():
    # Unlike lvi_q (no public repo, replaced 2026-09-21 by gs_livo), every
    # currently-registered candidate now has a real upstream fork to vendor
    # and build an image from -- no more null-image exception exists.
    backends = _load()
    for name, entry in backends.items():
        assert entry["image"] is not None, f"{name} has no image entry -- was a null-image exception intended here?"
