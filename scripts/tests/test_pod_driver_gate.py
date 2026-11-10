"""Tests for the host-driver gate and the launch-probe classification (no pod, no Isaac)."""

import csv

import pytest

from pod_driver_gate import (
    CSV_COLUMNS,
    HostInfo,
    append_probe_row,
    driver_allowed,
    parse_cuda_version,
)
from pod_launch_probe import classify, non_ok_hosts

ALLOWLIST = {
    "rules": [
        {"branch": "580", "min_version": "580.95.05", "why": "580 validated"},
        {"branch": "595", "exact_version": "595.58.03", "why": "595 validated build"},
    ]
}


@pytest.mark.parametrize("driver", ["580.95.05", "580.159.04", "595.58.03"])
def test_allowed_drivers(driver):
    allowed, _ = driver_allowed(driver, ALLOWLIST)
    assert allowed


@pytest.mark.parametrize("driver", [
    "580.95.04", "580.65.06", "595.58.04", "595.71.05", "595.79", "595.84", "595.91.07",
    "590.10.01", "570.195.03", "615.78.08", "", "unknown",
])
def test_rejected_drivers(driver):
    allowed, reason = driver_allowed(driver, ALLOWLIST)
    assert not allowed and reason


def test_cuda_version_is_read_from_the_header():
    header = "| NVIDIA-SMI 595.91.07   Driver Version: 595.91.07   CUDA Version: 13.0 |"
    assert parse_cuda_version(header) == "13.0"
    assert parse_cuda_version("no header here") == "unknown"


def test_probe_row_has_every_column(tmp_path):
    path = tmp_path / "launch_probe.csv"
    info = HostInfo(driver="595.91.07", cuda="13.0", gpu="NVIDIA A40", host="pod-1",
                    datacenter="US-GA-1", image="v6-ga")
    append_probe_row(path, info, stage="gate", verdict="REJECT", note="branch 595 not allowlisted")
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 1
    assert set(rows[0]) == set(CSV_COLUMNS)
    assert rows[0]["driver"] == "595.91.07" and rows[0]["verdict"] == "REJECT"


def test_append_keeps_one_header(tmp_path):
    path = tmp_path / "launch_probe.csv"
    info = HostInfo("595.58.03", "13.0", "A40", "pod-1", "unknown", "v6-ga")
    append_probe_row(path, info, stage="gate", verdict="ALLOW")
    append_probe_row(path, info, stage="probe", verdict="OK")
    assert path.read_text().count("ts,stage,driver") == 1
    assert len(list(csv.DictReader(path.open()))) == 2


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
    a = HostInfo("595.91.07", "13.0", "A40", "host-a", "unknown", "v6-ga")
    b = HostInfo("595.91.07", "13.0", "4090", "host-b", "unknown", "v6-ga")
    append_probe_row(path, a, stage="probe", verdict="SEGFAULT")
    assert non_ok_hosts(path) == {"host-a"}
    append_probe_row(path, b, stage="probe", verdict="SEGFAULT")
    assert non_ok_hosts(path) == {"host-a", "host-b"}


def test_ok_rows_do_not_count_as_failed_hosts(tmp_path):
    path = tmp_path / "launch_probe.csv"
    info = HostInfo("595.58.03", "13.0", "A40", "host-a", "unknown", "v6-ga")
    append_probe_row(path, info, stage="probe", verdict="OK")
    append_probe_row(path, info, stage="gate", verdict="ALLOW")
    assert non_ok_hosts(path) == set()
