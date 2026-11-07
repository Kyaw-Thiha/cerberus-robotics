"""common.cerberus_policies must import with no Isaac, torch, or ROS available."""

import subprocess
import sys
from pathlib import Path

FORBIDDEN = ("torch", "isaaclab", "isaacsim", "omni", "rclpy", "cerberus_msgs", "gymnasium")

SCRIPT = f"""
import importlib, sys
for m in ["base", "registry", "backend_card", "validate_config"]:
    importlib.import_module("common.cerberus_policies." + m)
from common.cerberus_policies import registry
for slot in ("loco", "arm", "body"):
    for name in registry.available_backends(slot):
        registry.get_backend(slot, name)
bad = [m for m in {FORBIDDEN!r} if m in sys.modules]
print("BAD:" + ",".join(bad))
"""


def test_imports_without_heavy_deps():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pixi.toml").exists())
    out = subprocess.run([sys.executable, "-c", SCRIPT], cwd=root, env={"PYTHONPATH": str(root), "PATH": ""}, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("BAD:"), out.stdout
