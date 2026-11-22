"""``driver_gate.main`` verdicts and the provider-neutral reject hook (nvidia-smi and the CSV are mocked)."""

import driver_gate
import pytest
from driver_gate import HostInfo


def _run(monkeypatch, tmp_path, driver, argv=()):
    monkeypatch.setattr(driver_gate, "CSV_PATH", tmp_path / "launch_probe.csv")
    monkeypatch.setattr(driver_gate, "collect_host_info",
                        lambda image=None: HostInfo(driver, "13.0", "NVIDIA L4", "h1", "unknown", "img"))
    return driver_gate.main(list(argv))


def test_allowed_driver_exits_zero_and_never_runs_the_hook(monkeypatch, tmp_path):
    marker = tmp_path / "hook-ran"
    rc = _run(monkeypatch, tmp_path, "580.95.05", ["--on-reject-cmd", f"touch {marker}"])
    assert rc == 0 and not marker.exists()


def test_rejected_driver_exits_two_and_runs_the_hook(monkeypatch, tmp_path, capsys):
    marker = tmp_path / "hook-ran"
    rc = _run(monkeypatch, tmp_path, "595.91.07", ["--on-reject-cmd", f"touch {marker}"])
    assert rc == driver_gate.EXIT_REJECT and marker.exists()
    out = capsys.readouterr().out
    assert "REJECT" in out and "on-reject command ok" in out
    assert "595.91.07" in (tmp_path / "launch_probe.csv").read_text()


def test_a_failing_hook_does_not_change_the_gate_result(monkeypatch, tmp_path, capsys):
    rc = _run(monkeypatch, tmp_path, "570.1.1", ["--on-reject-cmd", "exit 7"])
    assert rc == driver_gate.EXIT_REJECT
    assert "failed (exit 7)" in capsys.readouterr().out


def test_reject_without_a_hook_says_so(monkeypatch, tmp_path, capsys):
    assert _run(monkeypatch, tmp_path, "unknown") == driver_gate.EXIT_REJECT
    assert "no --on-reject-cmd" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["RUNPOD_API_KEY", "RUNPOD_POD_ID"])
def test_provider_variables_are_not_consulted(monkeypatch, tmp_path, name):
    monkeypatch.setenv(name, "x")
    assert _run(monkeypatch, tmp_path, "595.91.07") == driver_gate.EXIT_REJECT
