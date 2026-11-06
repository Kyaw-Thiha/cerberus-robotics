import argparse
import sys
import types

import pytest

from common.cerberus_compat import launcher
from common.cerberus_compat.launcher import (
    add_launcher_args,
    headless_requested,
    local_rank,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HEADLESS", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)


def ns(**kwargs):
    return argparse.Namespace(**kwargs)


# ---- headless_requested: old arg shape (Lab 2.3.2: --headless, no --viz) -------------------------------


def test_old_shape_headless_flag_true():
    assert headless_requested(ns(headless=True)) is True


def test_old_shape_headless_flag_false():
    assert headless_requested(ns(headless=False)) is False


def test_old_shape_headless_defaults_to_gui():
    assert headless_requested(ns(video=False)) is False


# ---- new arg shape (Lab 3.0: no --headless; --viz/--visualizer dest 'visualizer') ----------------------


@pytest.mark.parametrize(
    "viz", [None, "none", "NONE", [], ["none"], "rerun", ["viser", "newton_gl"], ""]
)
def test_new_shape_no_kit_visualizer_is_headless(viz):
    assert headless_requested(ns(visualizer=viz)) is True
    assert headless_requested(ns(viz=viz)) is True


@pytest.mark.parametrize("viz", ["kit", ["kit"], "kit,rerun", ["rerun", "Kit"]])
def test_new_shape_kit_visualizer_opens_window(viz):
    assert headless_requested(ns(visualizer=viz)) is False
    assert headless_requested(ns(viz=viz)) is False


def test_new_shape_headless_flag_still_wins_over_kit_viz():
    assert headless_requested(ns(headless=True, visualizer=["kit"])) is True


def test_viz_attribute_takes_precedence_order_viz_first():
    assert headless_requested(ns(viz="kit", visualizer=None)) is False


def test_livestream_forces_headless():
    assert headless_requested(ns(livestream=2, visualizer=["kit"])) is True
    assert headless_requested(ns(livestream=-1, visualizer=["kit"])) is False
    assert headless_requested(ns(livestream=0)) is False


# ---- environment variable ------------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on"])
def test_env_var_truthy(monkeypatch, value):
    monkeypatch.setenv("HEADLESS", value)
    assert headless_requested(ns(headless=False)) is True
    assert headless_requested(ns(visualizer=["kit"])) is True
    assert headless_requested(ns()) is True


@pytest.mark.parametrize("value", ["0", "", "false", "no"])
def test_env_var_falsy(monkeypatch, value):
    monkeypatch.setenv("HEADLESS", value)
    assert headless_requested(ns(headless=False)) is False


def test_explicit_environ_mapping_overrides_process_env(monkeypatch):
    monkeypatch.setenv("HEADLESS", "1")
    assert headless_requested(ns(headless=False), environ={}) is False
    assert headless_requested(ns(headless=False), environ={"HEADLESS": "1"}) is True


def test_accepts_plain_dict():
    assert headless_requested({"headless": True}) is True
    assert headless_requested({"visualizer": ["kit"]}) is False
    assert headless_requested({}) is False


# ---- add_launcher_args ---------------------------------------------------------------------------------


def _install_fake_app_launcher(monkeypatch, add_headless: bool):
    calls = []

    class FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            calls.append(parser)
            parser.add_argument("--device", default="cuda:0")
            if add_headless:
                parser.add_argument("--headless", action="store_true", default=False)

    isaaclab = types.ModuleType("isaaclab")
    app = types.ModuleType("isaaclab.app")
    app.AppLauncher = FakeAppLauncher
    isaaclab.app = app
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app)
    return calls


def test_add_launcher_args_delegates_and_keeps_existing_headless(monkeypatch):
    calls = _install_fake_app_launcher(monkeypatch, add_headless=True)
    parser = argparse.ArgumentParser()
    add_launcher_args(parser)  # must not raise a conflicting-option error
    assert calls == [parser]
    args = parser.parse_args(["--headless"])
    assert args.headless is True and args.device == "cuda:0"


def test_add_launcher_args_adds_headless_when_lab_dropped_it(monkeypatch):
    _install_fake_app_launcher(monkeypatch, add_headless=False)
    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    assert parser.parse_args([]).headless is False
    assert parser.parse_args(["--headless"]).headless is True
    assert headless_requested(parser.parse_args(["--headless"])) is True


# ---- local_rank ----------------------------------------------------------------------------------------


def test_local_rank_uses_attribute_when_present():
    assert local_rank(types.SimpleNamespace(local_rank=3)) == 3
    assert local_rank(types.SimpleNamespace(local_rank=0)) == 0


def test_local_rank_falls_back_to_env(monkeypatch):
    assert local_rank(object()) == 0
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert local_rank(object()) == 2


def test_module_is_pure_python():
    assert "isaaclab" not in vars(launcher)
