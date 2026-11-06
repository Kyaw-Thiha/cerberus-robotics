import math
import os

import pytest

from common.cerberus_description.embodiment import EmbodimentCfg, compose_embodiments


def _legs():
    return EmbodimentCfg("go2", "math:pi", {"legs": ("a", "b"), "hip": ("c",)})


def _arm():
    return EmbodimentCfg("arx5", "math:pi", {"arm": ("j1", "j2")})


def test_checkpoint_prefix_defaults_to_name():
    assert _legs().checkpoint_prefix == "go2"
    assert EmbodimentCfg("x", "math:pi", {}, checkpoint_prefix="y").checkpoint_prefix == "y"


def test_joint_names_group_order():
    assert _legs().joint_names == ("a", "b", "c")


def test_compose_merges():
    c = compose_embodiments("go2_arx5", _legs(), _arm(), "os.path:sep")
    assert c.joint_groups == {"legs": ("a", "b"), "hip": ("c",), "arm": ("j1", "j2")}
    assert c.checkpoint_prefix == "go2_arx5"
    assert c.joint_names == ("a", "b", "c", "j1", "j2")


def test_compose_collision_raises():
    arm = EmbodimentCfg("arm", "math:pi", {"arm": ("b",)})
    with pytest.raises(ValueError):
        compose_embodiments("x", _legs(), arm, "math:pi")


def test_resolve_lazy_import():
    assert _legs().resolve() == math.pi
    assert EmbodimentCfg("s", "os.path:sep", {}).resolve() == os.path.sep
    with pytest.raises(ValueError):
        EmbodimentCfg("s", "nocolon", {}).resolve()
