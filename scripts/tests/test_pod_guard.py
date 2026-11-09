"""Pod-only guard: semantics, the error message, and every guarded entry point.

The subprocess checks prove the CLI refuses *before* rclone or pxr is reached (no network, no scene I/O).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pod_guard import POD_ENV_VAR, is_pod, require_pod

SCRIPTS_DIR = Path(__file__).resolve().parents[1]


def test_is_pod_accepts_exactly_one():
    assert is_pod({}) is False
    assert is_pod({POD_ENV_VAR: "0"}) is False
    assert is_pod({POD_ENV_VAR: "true"}) is False
    assert is_pod({POD_ENV_VAR: "1"}) is True


def test_require_pod_raises_an_actionable_message():
    with pytest.raises(SystemExit) as exc:
        require_pod("thing.py", reason="because scenes are large")
    message = str(exc.value)
    assert "thing.py" in message
    assert "pod-only" in message
    assert POD_ENV_VAR in message
    assert "because scenes are large" in message
    assert "docs/worlds.md" in message


def test_require_pod_is_silent_on_the_pod(monkeypatch):
    monkeypatch.setenv(POD_ENV_VAR, "1")
    require_pod("thing.py")


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != POD_ENV_VAR}
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script), *args],
        env=env, capture_output=True, text=True, timeout=120,
    )


@pytest.mark.parametrize("script", ["fetch_grscenes_from_r2.py", "fetch_grscenes_to_r2.py",
                                    "build_lidar_proxy.py", "validate_scenes.py"])
def test_entry_guard_refuses_before_any_work(script):
    result = _run(script)
    assert result.returncode == 1, result.stderr
    assert "pod-only" in result.stderr
    assert POD_ENV_VAR in result.stderr


def test_in_main_guards_refuse(monkeypatch):
    monkeypatch.delenv(POD_ENV_VAR, raising=False)
    import build_occupancy
    import build_scan_proxy

    for call in (
        lambda: build_scan_proxy.main(["--scene", "grscenes:x", "--dry-run"]),
        lambda: build_occupancy.main(["--scene", "grscenes:x"]),
    ):
        with pytest.raises(SystemExit) as exc:
            call()
        assert "pod-only" in str(exc.value)


def test_help_still_works_for_in_main_guarded_scripts():
    import build_occupancy
    import build_scan_proxy

    for module in (build_scan_proxy, build_occupancy):
        with pytest.raises(SystemExit) as exc:
            module.main(["--help"])
        assert exc.value.code == 0
