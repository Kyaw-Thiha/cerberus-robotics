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
