"""Guard key selection, provider stop routing, and redaction of provider response bodies."""

import os
import subprocess
import sys
from pathlib import Path

import redact
import session_guard


def test_runpod_guard_needs_only_the_runpod_key():
    name, key = session_guard.select_api_key("runpod", {"RUNPOD_API_KEY": "r-key"})
    assert (name, key) == ("RUNPOD_API_KEY", "r-key")
    # a VAST-only environment must not satisfy a RunPod guard
    name, key = session_guard.select_api_key("runpod", {"VAST_API_KEY": "v-key"})
    assert (name, key) == ("RUNPOD_API_KEY", "")
    assert session_guard.select_api_key("vast", {"RUNPOD_API_KEY": "r"})[1] == ""


def test_runpod_stop_never_uses_the_vast_path(monkeypatch):
    calls = {}
    import runpod_provider
    import vast_provider

    monkeypatch.setattr(runpod_provider, "stop_pod",
                        lambda key, iid: (calls.__setitem__("runpod", (key, iid)) or (200, "{}")))
    monkeypatch.setattr(vast_provider, "stop_instance",
                        lambda key, iid: (calls.__setitem__("vast", (key, iid)) or (200, "{}")))
    status, _ = session_guard.stop_target("runpod", "r-key", "pod1")
    assert status == 200
    assert calls == {"runpod": ("r-key", "pod1")}


def test_a_runpod_stop_body_with_env_is_masked():
    body = ('{"id":"pod1","env":{"R2_SECRET_ACCESS_KEY":"abcdef0123456789abcdef",'
            '"RUNPOD_API_KEY":"rpa_1234567890abcdef"},"apiKey":"rpa_1234567890abcdef"}')
    masked = redact.mask_text(body)
    assert "abcdef0123456789abcdef" not in masked
    assert "rpa_1234567890abcdef" not in masked
    assert '"R2_SECRET_ACCESS_KEY":"***"' in masked


def test_watch_job_runs_as_a_plain_script_without_pythonpath():
    repo = Path(__file__).resolve().parents[2]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = subprocess.run([sys.executable, str(repo / "scripts" / "watch_job.py"), "--help"],
                         capture_output=True, text=True, env=env, cwd="/tmp")
    assert out.returncode == 0, out.stderr
    assert "--pod-id" in out.stdout
