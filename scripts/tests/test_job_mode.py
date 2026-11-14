"""Job mode: restricted env, args/entrypoint payload, and the laptop watchdog decisions."""

import pytest

import create_vast_instance as cvi
import watch_job


def test_job_env_is_the_allowlist_only(monkeypatch):
    for key in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.setenv(key, f"v-{key}")
    monkeypatch.setenv("VAST_API_KEY", "must-not-leak")
    monkeypatch.setenv("WANDB_API_KEY", "must-not-leak")
    monkeypatch.setenv("RUNPOD_API_KEY", "must-not-leak")
    env = cvi.job_env("j1", "migration/x/code/b.bundle", 85, "img:v1")
    assert set(env) == {"R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET",
                        "CERBERUS_CODE_BUNDLE", "CERBERUS_JOB_ID", "CERBERUS_DEADLINE_MINUTES",
                        "CERBERUS_IMAGE_TAG"}
    assert "must-not-leak" not in ",".join(env.values())


def test_job_env_requires_the_r2_variables(monkeypatch):
    for key in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(SystemExit, match="missing R2"):
        cvi.job_env("j", "b", 85, "img")


def test_payload_onstart_is_the_default_hook():
    payload = cvi.build_job_payload({"dph_total": 0.18}, image="img:v1", disk_gb=80, label="l",
                                    env={"A": "1"}, deadline_minutes=85)
    assert payload["runtype"] == "args" and payload["args"] == []
    assert "scripts/job_entrypoint.sh" in payload["onstart"] and "rclone copyto" in payload["onstart"]
    assert payload["env"] == {"A": "1"} and payload["disk"] == 80.0


def test_payload_can_use_the_args_hook():
    payload = cvi.build_job_payload({"dph_total": 0.18}, image="img:v1", disk_gb=80, label="l",
                                    env={"A": "1"}, deadline_minutes=85, hook="args")
    assert payload["args"][0] == "bash" and payload["onstart"] == ""


def test_bootstrap_does_not_embed_secrets():
    script = cvi.job_bootstrap()
    assert "$R2_SECRET_ACCESS_KEY" in script
    assert "must-not-leak" not in script


def test_decide_done_and_wall_and_stale():
    now = 1000.0
    assert watch_job.decide({"stage": "done"}, now=now, last_ok=now - 10, started=now - 100) == "done"
    assert watch_job.decide({"stage": "stand"}, now=now, last_ok=now - 301, started=now - 400) == "stop"
    assert watch_job.decide({"stage": "stand"}, now=now, last_ok=now - 10, started=now - 5401) == "stop"
    assert watch_job.decide({"stage": "stand"}, now=now, last_ok=now - 10, started=now - 100) == "wait"


def test_decide_stop_failed_beats_a_fresh_heartbeat():
    now = 1000.0
    assert watch_job.decide({"stage": "finishing", "detail": "STOP_FAILED"}, now=now,
                            last_ok=now, started=now - 10) == "stop"


def test_missing_status_file_is_not_a_heartbeat():
    # rclone returns 0 for an absent source; the monitor must not treat that as a heartbeat
    now = 1000.0
    # but an instance still booting gets the startup grace before the staleness rule applies
    assert watch_job.decide(None, now=now, last_ok=None, started=now - 10) == "wait"
    assert watch_job.decide(None, now=now, last_ok=None, started=now - 901) == "stop"


def test_manifest_diff():
    assert watch_job.manifest_diff(["a", "b"], ["a"]) == {"missing": ["b"], "extra": []}
    assert watch_job.manifest_diff(["a"], ["a", "c"]) == {"missing": [], "extra": ["c"]}
