"""num_target_bodies: how many bodies a resolved SceneEntityCfg.body_ids selects.

Loaded by file path because importing `policy.locomotion` runs `policy/locomotion/__init__.py`,
which needs `isaaclab_tasks` (pod-only). The helper itself is dependency-free.
"""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "body_ids.py"
_spec = importlib.util.spec_from_file_location("_body_ids_under_test", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
num_target_bodies = _mod.num_target_bodies

NUM_BODIES = 13


class _FakeTensor:
    """Stands in for a torch tensor: only `numel()` is used by the helper."""

    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n


def test_list_counts_selected_bodies():
    assert num_target_bodies([0], NUM_BODIES) == 1
    assert num_target_bodies([0, 3, 5], NUM_BODIES) == 3


def test_tuple_counts_selected_bodies():
    assert num_target_bodies((2, 4), NUM_BODIES) == 2


def test_tensor_counts_selected_bodies_not_all_bodies():
    # The latent bug: a tensor body_ids (Lab 3.x finalizes manager-term body_ids to a tensor) used to fall
    # through to asset.num_bodies, sizing forces for every body while selecting only one.
    assert num_target_bodies(_FakeTensor(1), NUM_BODIES) == 1
    assert num_target_bodies(_FakeTensor(4), NUM_BODIES) == 4


def test_slice_none_means_all_bodies():
    assert num_target_bodies(slice(None), NUM_BODIES) == NUM_BODIES


def test_partial_slice_counts_its_range():
    assert num_target_bodies(slice(2, 6), NUM_BODIES) == 4
    assert num_target_bodies(slice(0, 100), NUM_BODIES) == NUM_BODIES  # clipped to the asset


def test_unknown_type_falls_back_to_all_bodies():
    assert num_target_bodies(None, NUM_BODIES) == NUM_BODIES
