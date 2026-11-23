"""`sim_usd_path` honours CERBERUS_SCENE_USD_OVERRIDE (the render-unblock preview wrapper) before any cache."""

from pathlib import Path

import pytest

from common.cerberus_worlds.arena_assets import (
    SCENE_USD_OVERRIDE_ENV,
    SceneFiles,
    sim_usd_path,
)


def _files(folder: Path) -> SceneFiles:
    usd = folder / "start_result_navigation.usd"
    usd.write_text("#usda 1.0\n")
    return SceneFiles(scene_id="grscenes:TEST", source="grscenes", category="home", folder=folder,
                      usd_path=usd, variant="navigation", mdl_path=None)


def test_override_is_used_when_the_file_exists(tmp_path, monkeypatch):
    wrapper = tmp_path / "scene_preview_metric.usd"
    wrapper.write_text("#usda 1.0\n")
    monkeypatch.setenv(SCENE_USD_OVERRIDE_ENV, str(wrapper))
    assert sim_usd_path(_files(tmp_path)) == wrapper


def test_missing_override_raises_instead_of_falling_back(tmp_path, monkeypatch):
    monkeypatch.setenv(SCENE_USD_OVERRIDE_ENV, str(tmp_path / "nope.usd"))
    with pytest.raises(RuntimeError, match=SCENE_USD_OVERRIDE_ENV):
        sim_usd_path(_files(tmp_path))


def test_no_override_keeps_the_cached_path(tmp_path, monkeypatch):
    monkeypatch.delenv(SCENE_USD_OVERRIDE_ENV, raising=False)
    files = _files(tmp_path)
    result = sim_usd_path(files)
    # the normal route: the cached metre-scale wrapper beside the scene, never the override
    assert result.name in {"scene_metric.usd", files.usd_path.name}
    assert result != Path(SCENE_USD_OVERRIDE_ENV)
