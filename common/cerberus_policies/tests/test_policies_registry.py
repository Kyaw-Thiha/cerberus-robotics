import sys

import pytest

from common.cerberus_policies import registry
from common.cerberus_policies.base import Boundary, PolicyBackend


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    registry._reset_for_tests()
    for env in registry.SLOT_ENV.values():
        monkeypatch.delenv(env, raising=False)
    yield
    registry._reset_for_tests()


def test_no_silent_default():
    with pytest.raises(ValueError, match="LOCO_BACKEND"):
        registry.get_backend("loco")


def test_unknown_backend_and_slot_errors_list_available():
    with pytest.raises(ValueError, match="null_loco"):
        registry.get_backend("loco", "nope")
    with pytest.raises(ValueError, match="Unknown policy slot"):
        registry.get_backend("legs", "null_loco")
    with pytest.raises(ValueError, match="Unknown policy slot"):
        registry.available_backends("legs")


def test_env_var_selects_backend_and_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv("LOCO_BACKEND", "null_loco")
    assert type(registry.get_backend("loco")).__name__ == "NullLocoBackend"
    assert type(registry.get_backend("loco", "scripted_loco")).__name__ == "ScriptedLocoBackend"


def test_empty_slots_have_no_backends_and_no_default(monkeypatch):
    assert registry.available_backends("arm") == []
    monkeypatch.setenv("ARM_BACKEND", "null_loco")  # a loco backend is not an arm backend
    with pytest.raises(ValueError, match="Unknown arm"):
        registry.get_backend("arm")


def test_lazy_import_and_instance_caching():
    for m in [m for m in sys.modules if m.startswith("common.cerberus_policies.backends")]:
        del sys.modules[m]
    assert registry.available_backends("loco") == ["null_loco", "scripted_loco"]
    assert "common.cerberus_policies.backends.null_loco" not in sys.modules  # listing does not import
    a = registry.get_backend("loco", "null_loco")
    assert "common.cerberus_policies.backends.null_loco" in sys.modules
    assert "common.cerberus_policies.backends.scripted_loco" not in sys.modules
    assert registry.get_backend("loco", "null_loco") is a


def test_config_kwargs_give_fresh_uncached_instance():
    b = registry.get_backend("loco", "scripted_loco", config={"vx": 0.4})
    b.load(None, None)
    assert b.act({})["vx"] == 0.4
    assert registry.get_backend("loco", "scripted_loco") is not b


def test_register_custom_and_duplicate():
    class Stub(PolicyBackend):
        boundary = Boundary.B2_ARM
        rate_hz = 10

        def load(self, checkpoint_path, embodiment_cfg):
            pass

        def act(self, observation):
            return {}

    registry.register_backend("arm", "stub", Stub)
    assert isinstance(registry.get_backend("arm", "stub"), Stub)
    registry.register_backend("arm", "lazy", "common.cerberus_policies.backends.null_loco:NullLocoBackend")
    with pytest.raises(ValueError, match="already registered"):
        registry.register_backend("arm", "stub", Stub)
    with pytest.raises(ValueError, match="Unknown policy slot"):
        registry.register_backend("legs", "x", Stub)
