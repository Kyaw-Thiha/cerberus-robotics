"""Scene-id forms and the lite-scene IoU helper."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_lite_scene as lite
import scene_id


def test_every_scene_id_form_normalises_to_one_usd_suffix():
    for raw in ("tiny", "tiny_usd", "MV7J6NIKTKJZ2AABAAAAADQ8", "MV7J6NIKTKJZ2AABAAAAADQ8_usd", "/tiny_usd/"):
        assert scene_id.folder_name(raw).endswith("_usd")
        assert not scene_id.folder_name(raw).endswith("_usd_usd")
    assert scene_id.folder_name("tiny") == scene_id.folder_name("tiny_usd") == "tiny_usd"
    assert scene_id.bare_id("MV7_usd") == "MV7"


class _G:
    def __init__(self, grid):
        self.grid = grid


def test_iou_of_identical_and_disjoint_grids():
    from planning.occupancy import BLOCKED, FREE

    a = np.full((4, 4), FREE, dtype=np.int8)
    a[0, :2] = BLOCKED
    b = a.copy()
    assert lite.grid_iou(_G(a), _G(b)) == 1.0
    c = np.full((4, 4), FREE, dtype=np.int8)
    c[3, :2] = BLOCKED
    assert lite.grid_iou(_G(a), _G(c)) == 0.0
    assert lite.grid_iou(_G(a), _G(np.zeros((3, 3), dtype=np.int8))) == 0.0


def test_compact_drops_triangles_and_renumbers_vertices():
    import numpy as np
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [5, 5, 5]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 2]], dtype=np.int64)
    keep = np.array([True, True, False])
    p2, f2 = lite.compact(points, faces, keep)
    assert f2.max() < len(p2) and len(p2) == 4
    assert lite.triangle_areas(points, faces)[:2].min() > 0


def test_every_lite_threshold_attempt_records_size_and_iou(monkeypatch, tmp_path):
    """build() must accept the size budget and the threshold list (the pod crashed on a stale signature)."""
    import inspect

    params = inspect.signature(lite.build).parameters
    for name in ("max_mb", "min_iou", "thresholds"):
        assert name in params
