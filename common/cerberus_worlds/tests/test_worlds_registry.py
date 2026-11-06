import pytest

from common.cerberus_worlds import registry
from common.cerberus_worlds.backends.arena_backend import ArenaBackend
from common.cerberus_worlds.base import WorldBackend


class _Dummy(WorldBackend):
    def list_scenes(self):
        return []

    def load_scene(self, scene_id):
        return scene_id

    def register_embodiment(self, cfg):
        return None

    def build_env(self, scene, embodiment, task, num_envs):
        return None


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.delenv(registry.WORLD_BACKEND_ENV, raising=False)
    registry._reset_for_tests()
    yield
    registry._reset_for_tests()


def test_builtins_listed():
    assert registry.available_backends() == ["arena", "plain_usd"]


def test_duplicate_raises():
    with pytest.raises(ValueError):
        registry.register_backend("arena", _Dummy)


def test_unknown_lists_available():
    with pytest.raises(ValueError, match="plain_usd"):
        registry.get_backend("nope")


def test_neither_set_raises():
    with pytest.raises(ValueError, match="WORLD_BACKEND"):
        registry.get_backend()


def test_env_var(monkeypatch):
    registry.register_backend("tests.dummy", _Dummy)
    monkeypatch.setenv("WORLD_BACKEND", "tests.dummy")
    assert isinstance(registry.get_backend(), _Dummy)


def test_explicit_beats_env(monkeypatch):
    registry.register_backend("tests.dummy", _Dummy)
    monkeypatch.setenv("WORLD_BACKEND", "arena")
    assert isinstance(registry.get_backend("tests.dummy"), _Dummy)


def test_string_factory():
    registry.register_backend(
        "tests.str", "common.cerberus_worlds.backends.arena_backend:ArenaBackend"
    )
    assert isinstance(registry.get_backend("tests.str"), ArenaBackend)
    assert isinstance(registry.get_backend("arena"), ArenaBackend)


def test_instance_cached():
    assert registry.get_backend("arena") is registry.get_backend("arena")


def test_reset_restores_builtins():
    registry.register_backend("tests.dummy", _Dummy)
    registry._reset_for_tests()
    assert registry.available_backends() == ["arena", "plain_usd"]
