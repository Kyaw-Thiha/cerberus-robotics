import dataclasses
import subprocess
import sys
import types

import pytest

from common.cerberus_compat import urdf_converter as uc


def test_filter_cfg_kwargs_keeps_only_existing_fields():
    @dataclasses.dataclass
    class FakeCfg:
        asset_path: str = ""
        fix_base: bool = False

    kept, dropped = uc.filter_cfg_kwargs(
        FakeCfg,
        {
            "asset_path": "a.urdf",
            "fix_base": False,
            "make_instanceable": False,
            "joint_drive": None,
        },
    )
    assert kept == {"asset_path": "a.urdf", "fix_base": False}
    assert dropped == ["joint_drive", "make_instanceable"]


def test_import_urdf_converter_falls_back_and_reports_failure(monkeypatch):
    good = types.ModuleType("fake_good")
    good.UrdfConverter = object
    good.UrdfConverterCfg = dataclasses.make_dataclass("Cfg", [("asset_path", str, "")])
    monkeypatch.setitem(sys.modules, "fake_good", good)
    conv, cfg, name = uc.import_urdf_converter(("fake_does_not_exist", "fake_good"))
    assert (conv, name) == (object, "fake_good") and cfg is good.UrdfConverterCfg
    with pytest.raises(RuntimeError, match="UrdfConverter"):
        uc.import_urdf_converter(("fake_does_not_exist",))


def test_framework_versions_has_keys_without_isaac():
    assert set(uc.framework_versions()) == {"isaaclab", "isaacsim"}


def test_default_modules():
    assert uc.URDF_CONVERTER_MODULES == ("isaaclab.sim.converters", "isaaclab.sim")


def test_import_does_not_pull_in_isaac():
    code = (
        "import sys\n"
        "import common.cerberus_compat.urdf_converter\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('isaaclab', 'omni', 'isaacsim')]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
