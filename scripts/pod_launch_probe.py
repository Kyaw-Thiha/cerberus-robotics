#!/usr/bin/env python3
"""Launch probe: does Kit start on this host at all? Run only after ``pod_driver_gate.sh`` passes.

A 5-minute hard timeout around the smallest headless launch (ground-plane Go2, 1 env, 5 steps). The verdict is
appended to ``outputs/launch_probe.csv``:

* ``OK``       exit 0 and ``PROBE_OK`` in the output
* ``SEGFAULT`` exit 139 (the libcarb.tasking crash that motivated this)
* ``TIMEOUT``  killed at 300 s
* ``OTHER``    any other non-zero exit

A non-OK verdict stops the session. **Two non-OK verdicts on different hosts stop everything** -- no more retries:
the hosts are the variable, and the next one is unlikely to be better.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "outputs" / "launch_probe.csv"
TIMEOUT_S = 300
EXIT_NON_OK = 4
EXIT_TWO_HOSTS = 3

NON_OK = ("SEGFAULT", "TIMEOUT", "OTHER")


def classify(returncode: int, output: str) -> str:
    if returncode == 0 and "PROBE_OK" in output:
        return "OK"
    if returncode == 139 or returncode == -11:
        return "SEGFAULT"
    if returncode == 124:
        return "TIMEOUT"
    return "OTHER"


def non_ok_hosts(csv_path: Path) -> set[str]:
    """Hosts that already produced a non-OK probe verdict."""
    import csv

    if not csv_path.is_file():
        return set()
    hosts = set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("stage") == "probe" and row.get("verdict") in NON_OK:
                hosts.add(row.get("host") or "unknown")
    return hosts


def main(argv: list[str] | None = None) -> int:
    from pod_driver_gate import HostInfo, append_probe_row, collect_host_info
    from pod_guard import require_pod

    require_pod("pod_launch_probe.sh")
    lab = os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab").rstrip("/") + "/isaaclab.sh"
    if not Path(lab).is_file():
        print(f"LAUNCH PROBE: OTHER isaaclab.sh not found at {lab}")
        info = collect_host_info()
        append_probe_row(CSV_PATH, info, stage="probe", verdict="OTHER", note="isaaclab.sh missing")
        return EXIT_NON_OK

    env = os.environ.copy()
    env.setdefault("CERBERUS_ALLOW_BAD_DRIVER_RENDER", "1")
    cmd = [lab, "-p", str(REPO_ROOT / "scripts" / "pod_launch_probe_env.py"), "--headless"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, env=env)
        rc, output = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc = 124
        output = ((e.stdout or "") if isinstance(e.stdout, str) else "") + "\nTIMEOUT"
    except OSError as e:
        rc, output = 1, f"{type(e).__name__}: {e}"

    verdict = classify(rc, output)
    info: HostInfo = collect_host_info()
    note = "" if verdict == "OK" else output.strip().splitlines()[-1][:300] if output.strip() else ""
    print(f"LAUNCH PROBE: {verdict} driver={info.driver} gpu={info.gpu} host={info.host} rc={rc}")
    append_probe_row(CSV_PATH, info, stage="probe", verdict=verdict, note=note)
    if verdict == "OK":
        return 0

    hosts = non_ok_hosts(CSV_PATH)
    if len(hosts) >= 2:
        print(f"LAUNCH PROBE: two hosts have now failed ({', '.join(sorted(hosts))}); stopping everything")
        return EXIT_TWO_HOSTS
    return EXIT_NON_OK


if __name__ == "__main__":
    sys.exit(main())
