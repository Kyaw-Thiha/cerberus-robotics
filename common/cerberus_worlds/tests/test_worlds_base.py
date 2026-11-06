import pytest

from common.cerberus_worlds.backends.arena_backend import ArenaBackend
from common.cerberus_worlds.backends.plain_usd_backend import PlainUsdBackend
from common.cerberus_worlds.base import SceneInfo, WorldBackend


def test_scene_info_defaults():
    s = SceneInfo("kitchen")
    assert (s.category, s.source) == ("", "")
    assert dict(s.metadata) == {}


def test_abc_requires_all_methods():
    class Partial(WorldBackend):
        def list_scenes(self):
            return []

        def load_scene(self, scene_id):
            return None

        def register_embodiment(self, cfg):
            return None

    with pytest.raises(TypeError):
        Partial()


@pytest.mark.parametrize("cls", [ArenaBackend, PlainUsdBackend])
def test_backends_implement_world_backend(cls):
    # W1: the stubs were replaced by real implementations; they must still satisfy the ABC and
    # be constructible without isaaclab installed (all sim imports are lazy).
    b = cls()
    assert isinstance(b, WorldBackend)
    for name in ("list_scenes", "load_scene", "register_embodiment", "build_env"):
        assert callable(getattr(b, name))
