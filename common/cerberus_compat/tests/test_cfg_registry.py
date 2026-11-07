"""Tests for the gym task config entry-point resolver (pure python, no Isaac)."""

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "cfg_registry.py"
_spec = importlib.util.spec_from_file_location("_cfg_registry_under_test", _PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)


class TestResolveEntryPoint:
    def test_instantiates_module_attr(self):
        out = _mod.resolve_entry_point("collections:OrderedDict")
        assert isinstance(out, dict)

    def test_resolves_nested_attribute(self, monkeypatch):
        # "pkg.mod:Outer.Inner" -- dotted attrs resolve left to right, as the registry's
        # entry-point strings allow.
        import types

        module = types.ModuleType("_cfg_registry_fake_mod")
        module.Outer = types.SimpleNamespace(Inner=dict)
        monkeypatch.setitem(sys.modules, "_cfg_registry_fake_mod", module)
        assert isinstance(_mod.resolve_entry_point("_cfg_registry_fake_mod:Outer.Inner"), dict)

    def test_rejects_malformed_entry_point(self):
        with pytest.raises(ValueError, match="module:attr"):
            _mod.resolve_entry_point("collections.OrderedDict")

    def test_reports_missing_module_clearly(self):
        with pytest.raises(ModuleNotFoundError):
            _mod.resolve_entry_point("definitely_not_a_module:Thing")

    def test_rejects_non_callable_target(self):
        with pytest.raises(TypeError, match="non-callable"):
            _mod.resolve_entry_point("collections:__name__")


class _FakeSpec:
    def __init__(self, kwargs):
        self.kwargs = kwargs


class _FakeGym:
    def __init__(self, specs):
        self._specs = {name: _FakeSpec(kw) for name, kw in specs.items()}

    def spec(self, name):
        if name not in self._specs:
            raise KeyError(name)
        return self._specs[name]


@pytest.fixture
def fake_gym(monkeypatch):
    specs = {
        "Task-Play-v0": {"env_cfg_entry_point": "collections:OrderedDict"},
        "Task-Obj-v0": {"env_cfg_entry_point": lambda: {"from": "object"}, "rsl_rl_cfg_entry_point": dict},
    }
    monkeypatch.setitem(sys.modules, "gymnasium", _FakeGym(specs))
    return specs


class TestLoadCfg:
    def test_loads_registered_string_entry_point(self, fake_gym):
        assert isinstance(_mod.load_env_cfg("Task-Play-v0"), dict)

    def test_accepts_a_registered_callable(self, fake_gym):
        assert _mod.load_env_cfg("Task-Obj-v0") == {"from": "object"}

    def test_loads_agent_cfg_object(self, fake_gym):
        assert _mod.load_agent_cfg("Task-Obj-v0") is not None

    def test_missing_key_names_the_key(self, fake_gym):
        with pytest.raises(KeyError, match="rsl_rl_cfg_entry_point"):
            _mod.load_cfg("Task-Play-v0", "rsl_rl_cfg_entry_point")

    def test_unknown_task_propagates(self, fake_gym):
        with pytest.raises(KeyError):
            _mod.load_env_cfg("Nope-v0")
