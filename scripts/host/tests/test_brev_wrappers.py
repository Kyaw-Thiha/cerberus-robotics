"""The Brev wrappers, against stub `brev`, `systemd-run` and `systemctl` binaries (no account, no spend)."""

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HOST = REPO / "scripts" / "host"

BREV_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$CALLS"
case "$1" in
  ls) [[ -n "${BREV_LS_JSON:-}" ]] || exit 1; printf '%s\\n' "$BREV_LS_JSON" ;;
  create) exit "${BREV_CREATE_RC:-0}" ;;
  stop|delete|exec|copy) exit "${BREV_CMD_RC:-0}" ;;
  *) exit 0 ;;
esac
"""

SYSTEMD_RUN_STUB = """#!/usr/bin/env bash
printf 'systemd-run %s\\n' "$*" >> "$CALLS"
exit "${SYSTEMD_RUN_RC:-0}"
"""

SYSTEMCTL_STUB = """#!/usr/bin/env bash
printf 'systemctl %s\\n' "$*" >> "$CALLS"
case " $* " in
  *list-timers*) [[ "${SYSTEMCTL_LIST_RC:-0}" == "0" ]] && echo "cerberus-kill-demo.timer" ;;
esac
exit 0
"""


@pytest.fixture
def env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("brev", BREV_STUB), ("systemd-run", SYSTEMD_RUN_STUB), ("systemctl", SYSTEMCTL_STUB)):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    calls = tmp_path / "calls"
    calls.write_text("")
    e = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", CALLS=str(calls),
             CERBERUS_BREV_DIR=str(tmp_path / "brev"),
             BREV_LS_JSON=json.dumps({"instances": [{"name": "other", "state": "STOPPED"}]}))
    return e


def run(script, *args, env, timeout=60):
    return subprocess.run(["bash", str(HOST / script), *args], capture_output=True, text=True,
                          env=env, timeout=timeout, check=False)


def calls(env):
    return Path(env["CALLS"]).read_text()


def up(env, *extra, price="0.828", hours="3.5"):
    return run("brev_up.sh", "--name", "demo", "--type", "gpu_1x_rtx6000", "--price", price,
               "--hours", hours, *extra, env=env)


def seed_ledger(env, n=1, price=1.0, hours_ago=0):
    path = Path(env["CERBERUS_BREV_DIR"]) / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    from datetime import UTC, datetime, timedelta
    ts = (datetime.now(UTC) - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [{"ts": ts, "action": "create", "name": f"old{i}", "price": price} for i in range(n)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_up_creates_arms_the_timer_and_logs(env):
    r = up(env)
    assert r.returncode == 0, r.stderr
    text = calls(env)
    assert "systemd-run" in text and "cerberus-kill-demo" in text
    assert "create demo --type gpu_1x_rtx6000 --detached --min-disk 200" in text
    assert text.index("systemd-run") < text.index("create demo")  # kill switch armed first
    ledger = (Path(env["CERBERUS_BREV_DIR"]) / "ledger.jsonl").read_text()
    assert '"action":"create"' in ledger and '"price":0.828' in ledger


def test_up_refuses_above_the_price_cap(env):
    r = up(env, price="1.75")
    assert r.returncode == 3 and "exceeds the cap" in r.stderr
    assert "create demo" not in calls(env)


def test_up_refuses_more_than_the_creation_cap(env):
    seed_ledger(env, n=4)
    r = up(env)
    assert r.returncode == 4 and "cap 4" in r.stderr
    assert "create demo" not in calls(env)


def test_up_refuses_more_than_the_hours_cap(env):
    r = up(env, hours="4")
    assert r.returncode == 3 and "per-instance cap" in r.stderr
    assert calls(env) == ""


def test_up_refuses_while_another_instance_is_live(env):
    env["BREV_LS_JSON"] = json.dumps({"instances": [{"name": "other", "state": "RUNNING"}]})
    r = up(env)
    assert r.returncode == 6 and "one live instance" in r.stderr
    assert "create demo" not in calls(env)


def test_up_refuses_when_not_logged_in(env):
    env["BREV_LS_JSON"] = ""
    r = up(env)
    assert r.returncode == 2 and "not logged in" in r.stderr
    assert "create demo" not in calls(env)


def test_up_reads_the_current_workspaces_schema(env):
    env["BREV_LS_JSON"] = json.dumps({"workspaces": None})
    assert "0" in subprocess.run(["bash", "-c", f"source {HOST}/brev_lib.sh; brev_live_count"], env=env,
                                 capture_output=True, text=True, check=False).stdout
    env["BREV_LS_JSON"] = json.dumps({"workspaces": [{"name": "x", "status": "RUNNING"}]})
    assert "1" in subprocess.run(["bash", "-c", f"source {HOST}/brev_lib.sh; brev_live_count"], env=env,
                                 capture_output=True, text=True, check=False).stdout


def test_a_deleting_instance_is_not_live(env):
    env["BREV_LS_JSON"] = json.dumps({"workspaces": [{"name": "x", "status": "DELETING"}]})
    out = subprocess.run(["bash", "-c", f"source {HOST}/brev_lib.sh; brev_live_count"], env=env,
                         capture_output=True, text=True, check=False).stdout
    assert out.strip() == "0"


def test_up_refuses_when_the_live_count_is_unreadable(env):
    env["BREV_LS_JSON"] = "not json"
    r = up(env)
    assert r.returncode == 5 and "unknown live count" in r.stderr
    assert "create demo" not in calls(env)


def test_up_creates_nothing_if_the_timer_cannot_be_armed(env):
    env["SYSTEMD_RUN_RC"] = "1"
    r = up(env)
    assert r.returncode == 7 and "could not arm" in r.stderr
    assert "create demo" not in calls(env)


def test_up_cancels_and_refuses_if_the_timer_does_not_appear(env):
    env["SYSTEMCTL_LIST_RC"] = "1"
    r = up(env)
    assert r.returncode == 7 and "did not appear" in r.stderr
    assert "create demo" not in calls(env)


def test_up_cancels_the_timer_when_create_fails(env):
    env["BREV_CREATE_RC"] = "1"
    r = up(env)
    assert r.returncode == 8 and "cancelling the kill timer" in r.stderr
    assert "systemctl --user stop cerberus-kill-demo.timer" in calls(env)


def test_up_dry_run_touches_nothing(env):
    r = up(env, "--dry-run")
    assert r.returncode == 0 and "would arm" in r.stdout
    text = calls(env)
    assert "create demo" not in text and "systemd-run" not in text  # the guards are read-only


def test_down_refuses_an_instance_this_run_did_not_create(env):
    r = run("brev_down.sh", "someone-elses", env=env)
    assert r.returncode == 4 and "not created by this run" in r.stderr
    assert "stop someone-elses" not in calls(env)


def test_down_stops_deletes_and_cancels(env):
    assert up(env).returncode == 0
    Path(env["CALLS"]).write_text("")
    r = run("brev_down.sh", "demo", env=env)
    assert r.returncode == 0, r.stderr
    text = calls(env)
    assert "stop demo" in text and "delete demo" in text and "stop cerberus-kill-demo.timer" in text
    ledger = (Path(env["CERBERUS_BREV_DIR"]) / "ledger.jsonl").read_text()
    assert '"action":"delete"' in ledger


def test_down_reports_a_failed_delete_and_keeps_the_backstop(env):
    assert up(env).returncode == 0
    env["BREV_CMD_RC"] = "1"
    r = run("brev_down.sh", "demo", env=env)
    assert r.returncode == 1 and "kill timer stays armed" in r.stderr


def test_exec_and_copy_refuse_unmanaged_names(env):
    assert run("brev_exec.sh", "nope", "nvidia-smi", env=env).returncode == 4
    assert run("brev_copy.sh", "nope:/x", "/tmp/y", env=env).returncode == 4


def test_exec_and_copy_pass_through_for_managed_names(env):
    assert up(env).returncode == 0
    Path(env["CALLS"]).write_text("")
    assert run("brev_exec.sh", "demo", "nvidia-smi", env=env).returncode == 0
    assert "exec demo nvidia-smi" in calls(env)
    assert run("brev_copy.sh", "./local", "demo:/remote", env=env).returncode == 0
    assert "copy ./local demo:/remote" in calls(env)


def test_status_prints_the_spend_estimate(env):
    seed_ledger(env, n=1, price=1.0, hours_ago=2)
    r = run("brev_status.sh", env=env)
    assert r.returncode == 0
    assert "spend estimate: $2.0" in r.stdout
    assert "1 of 4" in r.stdout


def test_ledger_maths_counts_only_open_instances(env):
    seed_ledger(env, n=2, price=1.0, hours_ago=1)
    r = run("brev_status.sh", env=env)
    assert "2 of 4" in r.stdout
    assert "$2.0" in r.stdout
