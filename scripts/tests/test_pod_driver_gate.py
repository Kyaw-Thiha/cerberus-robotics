"""Tests for the tiered driver gate, the Kit retry wrapper and the probe classification (no pod, no Isaac)."""

import csv

import pytest

import kit_retry
from pod_driver_gate import (
    CSV_COLUMNS,
    TIER_A,
    TIER_B,
    TIER_REJECT,
    HostInfo,
    append_probe_row,
    classify_driver,
    parse_cuda_version,
)
from pod_launch_probe import classify, non_ok_hosts

import json
from pathlib import Path

ALLOWLIST = json.loads((Path(__file__).resolve().parents[1] / "pod_driver_allowlist.json").read_text())


@pytest.mark.parametrize("driver", ["580.95.05", "580.159.04", "595.58.03"])
def test_tier_a(driver):
    assert classify_driver(driver, ALLOWLIST)[0] == TIER_A


@pytest.mark.parametrize("driver", ["595.91.07", "595.71.05", "595.79", "590.10.01"])
def test_tier_b_needs_the_probe(driver):
    tier, reason = classify_driver(driver, ALLOWLIST)
    assert tier == TIER_B and reason


@pytest.mark.parametrize("driver", ["580.65.05", "570.195.03", "615.78.08", "", "unknown"])
def test_rejected_drivers(driver):
    assert classify_driver(driver, ALLOWLIST)[0] == TIER_REJECT


def test_the_recorded_595_91_07_is_tier_b_not_rejected():
    # the GA training, gate and eval runs completed on this driver; the startup crash is flaky
    tier, reason = classify_driver("595.91.07", ALLOWLIST)
    assert tier == TIER_B and "flaky" in reason


def test_cuda_version_is_read_from_the_header():
    header = "| NVIDIA-SMI 595.91.07   Driver Version: 595.91.07   CUDA Version: 13.0 |"
    assert parse_cuda_version(header) == "13.0"
    assert parse_cuda_version("no header here") == "unknown"


def test_probe_rows_carry_the_tier(tmp_path):
    path = tmp_path / "launch_probe.csv"
    info = HostInfo("595.91.07", "13.0", "NVIDIA A6000", "pod-1", "US-GA-1", "v6-ga")
    append_probe_row(path, info, stage="gate", verdict="PROBE_REQUIRED", note="tier B", tier=TIER_B)
    rows = list(csv.DictReader(path.open()))
    assert set(rows[0]) == set(CSV_COLUMNS)
    assert rows[0]["tier"] == TIER_B


# --- kit_retry ---------------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_runner(sequence, clock):
    seq = list(sequence)

    def runner(cmd):
        rc, elapsed = seq.pop(0)
        clock.t += elapsed
        return rc

    return runner


def test_retries_a_fast_139_once_then_returns_success():
    clock = FakeClock()
    log = []
    rc = kit_retry.run(["cmd"], runner=make_runner([(139, 10), (0, 12)], clock), clock=clock,
                       log=lambda *a: log.append(a))
    assert rc == 0 and len(log) == 2
    assert log[0][3] is False and log[1][3] is True  # first is a retry, second final


def test_three_launches_then_gives_up():
    clock = FakeClock()
    rc = kit_retry.run(["cmd"], runner=make_runner([(139, 5)] * 3, clock), clock=clock, log=lambda *a: None)
    assert rc == 139


def test_other_exit_codes_pass_through_immediately():
    clock = FakeClock()
    rc = kit_retry.run(["cmd"], runner=make_runner([(1, 5)], clock), clock=clock, log=lambda *a: None)
    assert rc == 1


def test_a_late_crash_is_not_retried():
    clock = FakeClock()
    rc = kit_retry.run(["cmd"], runner=make_runner([(139, 200)], clock), clock=clock, log=lambda *a: None)
    assert rc == 139


def test_should_retry_boundary():
    assert kit_retry.should_retry(139, 179.9, 180.0)
    assert not kit_retry.should_retry(139, 180.0, 180.0)
    assert not kit_retry.should_retry(1, 10.0, 180.0)


# --- probe -------------------------------------------------------------------------------------

@pytest.mark.parametrize("rc,output,expected", [
    (0, "PROBE_OK task=...", "OK"),
    (0, "no marker", "OTHER"),
    (139, "", "SEGFAULT"),
    (-11, "", "SEGFAULT"),
    (124, "", "TIMEOUT"),
    (1, "boom", "OTHER"),
])
def test_probe_classification(rc, output, expected):
    assert classify(rc, output) == expected


def test_two_non_ok_hosts_stop_everything(tmp_path):
    path = tmp_path / "launch_probe.csv"
    a = HostInfo("595.91.07", "13.0", "A6000", "host-a", "unknown", "v6-ga")
    b = HostInfo("595.91.07", "13.0", "A6000", "host-b", "unknown", "v6-ga")
    append_probe_row(path, a, stage="probe", verdict="SEGFAULT")
    assert non_ok_hosts(path) == {"host-a"}
    append_probe_row(path, b, stage="probe", verdict="SEGFAULT")
    assert non_ok_hosts(path) == {"host-a", "host-b"}
