"""Tiered driver gate and per-host launch counting (no pod)."""

import csv
import json
from pathlib import Path

import pytest

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
from pod_launch_probe import launch_counts

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
    tier, reason = classify_driver("595.91.07", ALLOWLIST)
    assert tier == TIER_B and "flaky" in reason


def test_cuda_version_is_read_from_the_header():
    header = "| NVIDIA-SMI 595.91.07   Driver Version: 595.91.07   CUDA Version: 13.2 |"
    assert parse_cuda_version(header) == "13.2"
    assert parse_cuda_version("no header here") == "unknown"


def test_probe_rows_carry_the_tier_and_cache(tmp_path):
    path = tmp_path / "launch_probe.csv"
    info = HostInfo("595.91.07", "13.2", "NVIDIA A40", "pod-1", "US-GA-1", "v6-ga")
    append_probe_row(path, info, stage="gate", verdict="PROBE_REQUIRED", note="tier B", tier=TIER_B, cache="false")
    row = next(iter(csv.DictReader(path.open())))
    assert set(row) == set(CSV_COLUMNS)
    assert row["tier"] == TIER_B and row["cache"] == "false"


def test_launch_counts_per_host(tmp_path):
    path = tmp_path / "launch_probe.csv"
    rows = [
        {"ts": "", "stage": "launch", "driver": "1", "cuda": "", "gpu": "", "host": "a", "datacenter": "",
         "image": "", "verdict": "LAUNCH_FAIL", "tier": "", "cache": "", "note": ""},
        {"ts": "", "stage": "launch", "driver": "1", "cuda": "", "gpu": "", "host": "a", "datacenter": "",
         "image": "", "verdict": "STARTED", "tier": "", "cache": "", "note": ""},
        {"ts": "", "stage": "launch", "driver": "1", "cuda": "", "gpu": "", "host": "b", "datacenter": "",
         "image": "", "verdict": "LAUNCH_FAIL", "tier": "", "cache": "", "note": ""},
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert launch_counts(path, "a") == (1, 2)
    assert launch_counts(path, "b") == (0, 1)
    assert launch_counts(path, "c") == (0, 0)
