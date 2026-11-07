"""Tests for observation-group selection (pure python)."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "policy_io.py"
_spec = importlib.util.spec_from_file_location("_policy_io_under_test", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class TestSelectPolicyObs:
    def test_mapping_with_the_named_group(self):
        tensor = object()
        assert _mod.select_policy_obs({"policy": tensor, "critic": object()}) is tensor

    def test_single_group_mapping_is_unambiguous(self):
        tensor = object()
        assert _mod.select_policy_obs({"whatever": tensor}) is tensor

    def test_bare_tensor_passes_through(self):
        tensor = object()
        assert _mod.select_policy_obs(tensor) is tensor

    def test_ambiguous_mapping_names_the_groups(self):
        with pytest.raises(KeyError, match="groups"):
            _mod.select_policy_obs({"a": object(), "b": object()})

    def test_nested_mappings_are_unwrapped(self):
        tensor = object()
        assert _mod.select_policy_obs({"outer": {"policy": tensor}}) is tensor

    def test_custom_group_name(self):
        tensor = object()
        assert _mod.select_policy_obs({"teacher": tensor}, group="teacher") is tensor
