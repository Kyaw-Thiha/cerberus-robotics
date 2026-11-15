"""Guard tick liveness: stale ticks mean stop the pod; null-runtime pods are deleted."""

import time

import guard_health
import runpod_provider


def test_tick_age_is_none_without_a_tick_and_capped_at_zero():
    now = 1000.0
    assert guard_health.tick_age(now, None) is None
    assert guard_health.tick_age(now, 950.0) == 50.0
    assert guard_health.tick_age(now, 1200.0) == 0.0


def test_stale_tick_stops_the_pod(tmp_path, monkeypatch, capsys):
    tick = tmp_path / "guard_tick.json"
    tick.write_text('{"epoch": %f, "status": "waiting"}' % (time.time() - 300))
    calls = {}
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr(runpod_provider, "stop_pod", lambda key, pod: (calls.__setitem__(pod, key) or (200, "{}")))
    rc = guard_health.main(["--max-gap", "120", "--pod-id", "p1", "--tick", str(tick)])
    assert rc == 1 and calls == {"p1": "k"}
    err = capsys.readouterr().err
    assert "GUARD TICK STALE" in err and "stopping the pod now" in err


def test_fresh_tick_is_healthy(tmp_path):
    tick = tmp_path / "guard_tick.json"
    tick.write_text('{"epoch": %f, "status": "waiting"}' % time.time())
    assert guard_health.main(["--max-gap", "120", "--tick", str(tick)]) == 0


def test_delete_pod_uses_delete():
    calls = []

    def transport(method, url, key, body=None):
        calls.append((method, url))
        return 204, ""

    status, _ = runpod_provider.delete_pod("k", "p1", transport=transport)
    assert status == 204 and calls == [("DELETE", "https://rest.runpod.io/v1/pods/p1")]
