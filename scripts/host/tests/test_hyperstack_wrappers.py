"""The Hyperstack wrappers against a fake API server and stub systemd/ssh binaries (no account, no spend)."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HOST = REPO / "scripts" / "host"
API_KEY = "test-key-do-not-print-0123456789"

SYSTEMD_RUN = """#!/usr/bin/env bash
printf 'systemd-run %s\\n' "$*" >> "$CALLS"
exit 0
"""
SYSTEMCTL = """#!/usr/bin/env bash
printf 'systemctl %s\\n' "$*" >> "$CALLS"
case " $* " in
  *list-timers*)
    for a in "$@"; do case "$a" in *.timer) echo "$a";; esac; done ;;
esac
exit 0
"""
SSH = """#!/usr/bin/env bash
printf 'ssh %s\\n' "$*" >> "$CALLS"
exit 0
"""


class FakeHyperstack:
    def __init__(self):
        self.vms = []
        self.next_id = 101
        self.last_create = None
        self.keypair_body = None
        self.deleted = []

    def handle(self, method, path, body):
        base = path.split("?", 1)[0]
        if base == "/core/environments" and method == "GET":
            return 200, {"environments": [{"id": 1, "name": "default-CANADA-1", "region": "CANADA-1"}]}
        if base == "/core/environments" and method == "POST":
            return 200, {"environment": body}
        if base == "/core/keypairs" and method == "GET":
            return 200, {"keypairs": []}
        if base == "/core/keypairs" and method == "POST":
            self.keypair_body = body
            return 200, {"keypair": {"id": 7, "name": body.get("name")}}
        if base == "/core/flavors":
            return 200, {"data": [{"region_name": "CANADA-1", "flavors": [{
                "id": 9, "name": "n3-L40x1-bigroot", "region_name": "CANADA-1", "gpu": "L40",
                "gpu_count": 1, "cpu": 28, "ram": 58.0, "disk": 825, "stock_available": True}]}]}
        if base == "/core/images":
            return 200, {"images": [{"region_name": "CANADA-1", "images": [
                {"id": 175, "name": "Ubuntu Server 22.04 LTS R535 CUDA 12.2"},
                {"id": 1624, "name": "Ubuntu Server 24.04 LTS R595 CUDA 13.2 with Docker"}]}]}
        if base == "/pricebook":
            return 200, [
                {"name": "L40", "value": "1.000000000"},
                {"name": "Cloud-SSD", "value": "0.000096774"},
                {"name": "PublicIP", "value": "0.006720430"},
                {"name": "RTX-A6000", "value": "0.500000000"},
            ]
        if base == "/core/virtual-machines" and method == "GET":
            return 200, {"instances": self.vms}
        if base == "/core/virtual-machines" and method == "POST":
            self.last_create = body
            vm = {
                "id": self.next_id, "name": body.get("name"), "status": "ACTIVE",
                "floating_ip": "203.0.113.9", "flavor": {"name": body.get("flavor_name"), "gpu": "L40"},
                "image": {"name": body.get("image_name")},
            }
            self.next_id += 1
            self.vms.append(vm)
            return 200, {"status": True, "instances": [vm]}
        if base.startswith("/core/virtual-machines/"):
            vm_id = int(base.rsplit("/", 1)[1])
            vm = next((v for v in self.vms if v["id"] == vm_id), None)
            if vm is None:
                return 404, {"status": False, "message": "not found"}
            if method == "DELETE":
                self.deleted.append(vm_id)
                self.vms.remove(vm)
                return 200, {"status": True}
            return 200, {"instance": vm}
        return 404, {"status": False, "message": f"no route {method} {base}"}


@pytest.fixture
def hs(tmp_path):
    state = FakeHyperstack()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _do(self, method):
            length = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(length)) if length else None
            if self.headers.get("api_key") != API_KEY:
                code, payload = 401, {"status": False, "message": "bad key"}
            else:
                code, payload = state.handle(method, self.path, body)
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self._do("GET")

        def do_POST(self):
            self._do("POST")

        def do_DELETE(self):
            self._do("DELETE")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("systemd-run", SYSTEMD_RUN), ("systemctl", SYSTEMCTL), ("ssh", SSH)):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    calls = tmp_path / "calls"
    calls.write_text("")
    key = tmp_path / "key"
    key.write_text("PRIVATE-KEY-MATERIAL")
    (tmp_path / "key.pub").write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITEST cerberus-hyperstack\n")
    env_file = tmp_path / ".env"
    env_file.write_text(f"HYPERSTACK_API_KEY={API_KEY}\nR2_BUCKET=dummy\n")

    env = dict(os.environ,
               PATH=f"{bin_dir}:{os.environ['PATH']}",
               CALLS=str(calls),
               CERBERUS_HYPERSTACK_DIR=str(tmp_path / "hs"),
               HYPERSTACK_ENV_FILE=str(env_file),
               HYPERSTACK_API_BASE=f"http://127.0.0.1:{server.server_port}",
               HYPERSTACK_SSH_KEY=str(key),
               HYPERSTACK_PUBLIC_IP="203.0.113.7")
    for var in ("HYPERSTACK_MAX_PRICE", "HYPERSTACK_MAX_HOURS", "HYPERSTACK_MAX_CREATIONS",
                "HYPERSTACK_TOTAL_SPEND", "HYPERSTACK_TOTAL_HOURS"):
        env.pop(var, None)
    yield env, state, tmp_path
    server.shutdown()
    server.server_close()


def run(script, *args, env, timeout=120):
    return subprocess.run(["bash", str(HOST / script), *args], capture_output=True, text=True,
                          env=env, timeout=timeout, check=False)


def calls(env):
    return Path(env["CALLS"]).read_text()


def up(env, *extra, flavor="n3-L40x1-bigroot", name="cerberus-canary"):
    return run("hyperstack_up.sh", "--name", name, "--flavor", flavor, *extra, env=env)


def seed_ledger(env, rows):
    path = Path(env["CERBERUS_HYPERSTACK_DIR"]) / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_up_arms_both_timers_and_posts_the_restricted_rule(hs):
    env, state, tmp_path = hs
    r = up(env)
    assert r.returncode == 0, r.stderr
    text = calls(env)
    assert "cerberus-kill-name-cerberus-canary" in text
    assert "cerberus-kill-101" in text
    assert text.index("cerberus-kill-name") < text.index("cerberus-kill-101")
    assert state.last_create["security_rules"][0]["remote_ip_prefix"] == "203.0.113.7/32"
    assert state.last_create["security_rules"][0]["port_range_min"] == 22
    assert state.last_create["assign_floating_ip"] is True
    assert state.last_create["enable_port_randomization"] is False
    assert state.last_create["key_name"] == "cerberus-hyperstack"
    assert state.last_create["image_name"].endswith("R595 CUDA 13.2 with Docker")
    assert state.keypair_body["public_key"].startswith("ssh-ed25519")
    assert "PRIVATE-KEY-MATERIAL" not in json.dumps(state.keypair_body)
    ledger = (Path(env["CERBERUS_HYPERSTACK_DIR"]) / "ledger.jsonl").read_text()
    assert '"action":"create"' in ledger and '"id":101' in ledger and '"price":1.0865' in ledger
    assert API_KEY not in r.stdout + r.stderr
    assert "203.0.113.9" in r.stdout


def test_up_refuses_above_the_price_cap(hs):
    env, state, _ = hs
    env["HYPERSTACK_MAX_PRICE"] = "0.5"
    r = up(env)
    assert r.returncode == 3 and "exceeds the cap" in r.stderr
    assert state.last_create is None
    assert "systemd-run" not in calls(env)


def test_up_refuses_more_than_the_creation_cap(hs):
    env, state, _ = hs
    seed_ledger(env, [{"action": "create", "name": f"old{i}", "price": 1.0} for i in range(3)])
    r = up(env)
    assert r.returncode == 4 and "cap 3" in r.stderr
    assert state.last_create is None


def test_up_refuses_while_a_vm_is_live(hs):
    env, state, _ = hs
    state.vms.append({"id": 5, "name": "owner-box", "status": "ACTIVE"})
    r = up(env)
    assert r.returncode == 6 and "one live" in r.stderr
    assert state.last_create is None


def test_up_dry_run_creates_nothing(hs):
    env, state, _ = hs
    r = up(env, "--dry-run")
    assert r.returncode == 0
    assert state.last_create is None
    assert "systemd-run" not in calls(env)


def test_kill_check_is_a_get_then_delete_verifies(hs):
    env, state, _ = hs
    state.vms.append({"id": 77, "name": "cerberus-x", "status": "ACTIVE", "floating_ip": "203.0.113.1"})
    r = run("hyperstack_kill.sh", "--check", "77", env=env)
    assert r.returncode == 0 and "status=ACTIVE" in r.stderr + r.stdout
    assert state.deleted == []
    r = run("hyperstack_kill.sh", "77", env=env)
    assert r.returncode == 0, r.stderr
    assert state.deleted == [77] and state.vms == []
    assert "systemctl --user stop cerberus-kill-77.timer" in calls(env)


def test_kill_refuses_a_vm_this_run_did_not_create(hs):
    env, state, _ = hs
    state.vms.append({"id": 88, "name": "owner-box", "status": "ACTIVE"})
    r = run("hyperstack_kill.sh", "88", env=env)
    assert r.returncode == 3 and "refusing" in r.stderr
    assert state.deleted == []


def test_ssh_uses_the_dedicated_key_and_resolves_the_ip(hs):
    env, state, tmp_path = hs
    up(env)
    r = run("hyperstack_ssh.sh", "cerberus-canary", "nvidia-smi", env=env)
    assert r.returncode == 0, r.stderr
    text = calls(env)
    assert "-i" in text and "IdentitiesOnly=yes" in text and "StrictHostKeyChecking=accept-new" in text
    assert "ubuntu@203.0.113.9" in text and "nvidia-smi" in text


def test_status_reports_spend_without_secrets(hs):
    env, state, _ = hs
    state.vms.append({"id": 77, "name": "cerberus-x", "status": "ACTIVE", "floating_ip": "203.0.113.1"})
    seed_ledger(env, [{"ts": "2026-11-22T10:00:00Z", "action": "create", "name": "cerberus-x",
                       "id": 77, "price": 1.0}])
    r = run("hyperstack_status.sh", env=env)
    assert r.returncode == 0, r.stderr
    assert "cerberus-x" in r.stdout and "spend ~$" in r.stdout
    assert API_KEY not in r.stdout + r.stderr
