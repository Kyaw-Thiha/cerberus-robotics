#!/usr/bin/env python3
"""R2 command channel, laptop side: run commands on a host that has no SSH, over the bucket.

The instance runs ``scripts/host/r2_channel.sh`` (installed by the startup script this tool bundles). It lists
``jobs/<job>/cmd/`` every 5 s, runs each command with bash and a timeout, and uploads stdout/stderr/exit status to
``jobs/<job>/out/``; a heartbeat lands every 30 s. This client writes a command file and polls for the result.

Commands::

    r2_channel.py bundle --job J [--out /tmp/J-startup.sh]     # startup script to pass to brev create
    r2_channel.py exec   --job J --cmd 'uname -a; df -h' [--timeout 900]
    r2_channel.py wait   --job J --timeout 600                 # until the channel's first heartbeat
    r2_channel.py put    --job J --src ./local --key input/scene.tar
    r2_channel.py get    --job J --key out/0001.out --dst ./x
    r2_channel.py tail   --job J --key out/0001.out --lines 40
    r2_channel.py heartbeat --job J

Credentials come from the environment or ``--env-file`` (default ``.env``) and are placed only in the environment of
the rclone subprocess; nothing prints them. Only the job prefix is read or written.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from redact import mask_text

JOB_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
REQUIRED = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
CHANNEL = HERE / "r2_channel.sh"


def load_env(env_file: str | None) -> dict[str, str]:
    values = {k: os.environ[k] for k in REQUIRED if k in os.environ}
    path = Path(env_file) if env_file else REPO_ROOT / ".env"
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.removeprefix("export ").partition("=")
            key = key.strip()
            if key in REQUIRED:
                values[key] = value.strip().strip('"').strip("'")
    missing = [k for k in REQUIRED if not values.get(k)]
    if missing:
        raise SystemExit(f"missing credentials: {', '.join(missing)} (names only)")
    return values


def rclone_env(creds: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    for key in REQUIRED:
        env.pop(key, None)
    prefix = "RCLONE_CONFIG_CERBERUS_R2_"
    env[prefix + "TYPE"] = "s3"
    env[prefix + "PROVIDER"] = "Cloudflare"
    env[prefix + "ACCESS_KEY_ID"] = creds["R2_ACCESS_KEY_ID"]
    env[prefix + "SECRET_ACCESS_KEY"] = creds["R2_SECRET_ACCESS_KEY"]
    env[prefix + "ENDPOINT"] = f"https://{creds['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    env[prefix + "ACL"] = "private"
    env[prefix + "NO_CHECK_BUCKET"] = "true"
    env["RCLONE_S3_NO_CHECK_BUCKET"] = "true"
    env["R2_BUCKET"] = creds["R2_BUCKET"]
    return env


class Channel:
    def __init__(self, job: str, creds: dict[str, str]):
        if not JOB_RE.match(job):
            raise SystemExit("job id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
        self.job = job
        self.env = rclone_env(creds)
        self.base = f"jobs/{job}"

    # -- primitives -------------------------------------------------------------------------------------------
    def _run(self, args: list[str], timeout: float = 120) -> subprocess.CompletedProcess:
        return subprocess.run(["rclone", "--s3-no-check-bucket", "--no-traverse", *args],
                              env=self.env, capture_output=True, text=True, timeout=timeout, check=False)

    def remote(self, key: str) -> str:
        return f"cerberus_r2:{self.env['R2_BUCKET']}/{self.base}/{key.lstrip('/')}"

    def copyto(self, src: Path, key: str) -> None:
        proc = self._run(["copyto", str(src), self.remote(key)])
        if proc.returncode != 0:
            raise SystemExit(f"rclone copyto failed: {mask_text(proc.stderr.strip()[:300])}")

    def cat(self, key: str, timeout: float = 60) -> str | None:
        proc = self._run(["cat", self.remote(key)], timeout=timeout)
        return proc.stdout if proc.returncode == 0 else None

    def lsf(self, prefix: str) -> list[str]:
        proc = self._run(["lsf", self.remote(prefix), "--files-only", "--max-depth", "1"])
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()] if proc.returncode == 0 else []

    # -- protocol ---------------------------------------------------------------------------------------------
    def next_seq(self) -> int:
        seqs = [int(m.group(1)) for name in self.lsf("cmd/") if (m := re.match(r"^(\d{4})_\d+\.cmd$", name))]
        return (max(seqs) + 1) if seqs else 1

    def exec(self, command: str, timeout: int = 1800, poll: float = 2.0, wait: bool = True,
             log=print) -> int:
        seq = self.next_seq()
        key_base = f"{seq:04d}"
        with tempfile.NamedTemporaryFile("w", suffix=".cmd", delete=False) as fh:
            fh.write(command if command.endswith("\n") else command + "\n")
            tmp = Path(fh.name)
        try:
            self.copyto(tmp, f"cmd/{key_base}_{int(timeout)}.cmd")
        finally:
            tmp.unlink(missing_ok=True)
        log(f"[r2] queued cmd {key_base} (timeout {timeout}s)")
        if not wait:
            return 0
        deadline = time.monotonic() + timeout + 120
        rc_text = None
        while time.monotonic() < deadline:
            rc_text = self.cat(f"out/{key_base}.rc")
            if rc_text is not None:
                break
            time.sleep(poll)
        if rc_text is None:
            log(f"[r2] no result within {timeout + 120}s")
            return 124
        out = self.cat(f"out/{key_base}.out") or ""
        err = self.cat(f"out/{key_base}.err") or ""
        if out:
            print(out, end="" if out.endswith("\n") else "\n")
        if err:
            print(err, end="" if err.endswith("\n") else "\n", file=sys.stderr)
        try:
            return int(rc_text.strip())
        except ValueError:
            return 1

    def heartbeat(self) -> dict | None:
        text = self.cat("heartbeat.json")
        if not text:
            return None
        try:
            return json.loads(text)
        except ValueError:
            return None

    def state(self) -> dict | None:
        text = self.cat("state/channel.json")
        try:
            return json.loads(text) if text else None
        except ValueError:
            return None

    def wait_ready(self, timeout: float = 600, poll: float = 3.0) -> dict | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            hb = self.heartbeat()
            if hb:
                return hb
            time.sleep(poll)
        return None

    def put(self, src: Path, key: str) -> None:
        self.copyto(src, key)

    def get(self, key: str, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        proc = self._run(["copyto", self.remote(key), str(dst)])
        if proc.returncode != 0:
            raise SystemExit(f"get failed: {mask_text(proc.stderr.strip()[:300])}")

    def pull(self, prefix: str, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        proc = self._run(["copy", self.remote(prefix), str(dst), "--min-size", "1B"])
        if proc.returncode != 0:
            raise SystemExit(f"pull failed: {mask_text(proc.stderr.strip()[:300])}")


STARTUP_TEMPLATE = """#!/bin/bash
# Brev startup script generated by scripts/host/r2_channel.py for job {job}.
# Small on purpose: install rclone, write a 0600 credentials file, fetch the channel from the bucket, start it.
JOB_ID={job_q}
R2_BUCKET={bucket_q}
LOG=/var/log/cerberus-startup.log
[ -w /var/log ] || LOG=/tmp/cerberus-startup.log
exec >>"$LOG" 2>&1
echo "=== $(date -u +%FT%TZ) startup job=$JOB_ID arch=$(uname -m) uid=$(id -u)"
case "$(uname -m)" in aarch64|arm64) RARCH=arm64 ;; *) RARCH=amd64 ;; esac
if ! command -v rclone >/dev/null 2>&1; then
  for attempt in $(seq 1 10); do
    echo "rclone install attempt $attempt"
    if [ "$(id -u)" = "0" ]; then TARGET=/usr/local/bin; else TARGET="$HOME/.local/bin"; fi
    mkdir -p "$TARGET" /tmp/rclone-install
    URL="https://downloads.rclone.org/rclone-current-linux-$RARCH.zip"
    if command -v curl >/dev/null 2>&1; then curl -fsSL -o /tmp/rclone-install/rclone.zip "$URL"
    elif command -v wget >/dev/null 2>&1; then wget -q -O /tmp/rclone-install/rclone.zip "$URL"
    else python3 -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$URL" /tmp/rclone-install/rclone.zip
    fi || echo "download failed"
    python3 - <<'ZIPEXTRACT' || echo "extract failed"
import glob, zipfile
z = glob.glob('/tmp/rclone-install/rclone*.zip')
if not z:
    raise SystemExit(1)
with zipfile.ZipFile(z[0]) as fh:
    fh.extractall('/tmp/rclone-install')
ZIPEXTRACT
    if install -m755 "$(find /tmp/rclone-install -type f -name rclone | head -1)" "$TARGET/rclone" 2>/dev/null; then
        export PATH="$TARGET:$PATH"; break
    fi
    sleep 20
  done
fi
echo "rclone: $(command -v rclone || echo MISSING)"
if [ "$(id -u)" = "0" ]; then RUN_DIR=/opt/cerberus; ENVF=/etc/cerberus-r2.env; else RUN_DIR="$HOME/cerberus-channel"; ENVF="$HOME/.cerberus-r2.env"; fi
mkdir -p "$RUN_DIR"
umask 077
cat > "$ENVF" <<'R2ENV'
{env_lines}
R2ENV
chmod 600 "$ENVF"
set -a
# shellcheck disable=SC1090
. "$ENVF"
set +a
for attempt in $(seq 1 10); do
    if rclone --s3-no-check-bucket copyto "cerberus_r2:$R2_BUCKET/jobs/$JOB_ID/input/r2_channel.sh" "$RUN_DIR/r2_channel.sh" 2>/dev/null; then
        echo "channel fetched on attempt $attempt"; break
    fi
    echo "channel fetch attempt $attempt failed"; sleep 20
done
chmod 755 "$RUN_DIR/r2_channel.sh" 2>/dev/null
if [ "$(id -u)" = "0" ] && systemctl is-system-running >/dev/null 2>&1; then
    cat > /etc/systemd/system/cerberus-channel.service <<UNIT
[Unit]
Description=Cerberus R2 command channel ({job})
After=network-online.target
Wants=network-online.target
[Service]
Environment=JOB_ID={job}
Environment=R2_ENV=$ENVF
ExecStart=/bin/bash $RUN_DIR/r2_channel.sh
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
UNIT
    systemctl daemon-reload
    systemctl enable --now cerberus-channel.service || echo "systemctl start failed"
fi
# Always ensure one process is running, whether or not systemd exists or worked.
if ! pgrep -f "$RUN_DIR/r2_channel.sh" >/dev/null 2>&1; then
    setsid nohup env JOB_ID="$JOB_ID" R2_ENV="$ENVF" bash "$RUN_DIR/r2_channel.sh" >> "$RUN_DIR/channel.log" 2>&1 < /dev/null &
    sleep 2
fi
echo "channel pids: $(pgrep -f "$RUN_DIR/r2_channel.sh" | tr '\n' ' ')"
rclone --s3-no-check-bucket copyto "$LOG" "cerberus_r2:$R2_BUCKET/jobs/$JOB_ID/state/startup.log" || echo "log upload failed"
date -u +%FT%TZ > "$RUN_DIR/started_at"
"""


def bundle(job: str, creds: dict[str, str], out: Path) -> Path:
    if not JOB_RE.match(job):
        raise SystemExit("job id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
    env_lines = "\n".join([
        f"export R2_BUCKET={shlex.quote(creds['R2_BUCKET'])}",
        "export RCLONE_CONFIG_CERBERUS_R2_TYPE=s3",
        "export RCLONE_CONFIG_CERBERUS_R2_PROVIDER=Cloudflare",
        f"export RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID={shlex.quote(creds['R2_ACCESS_KEY_ID'])}",
        f"export RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY={shlex.quote(creds['R2_SECRET_ACCESS_KEY'])}",
        f"export RCLONE_CONFIG_CERBERUS_R2_ENDPOINT={shlex.quote('https://' + creds['R2_ACCOUNT_ID'] + '.r2.cloudflarestorage.com')}",
        "export RCLONE_CONFIG_CERBERUS_R2_ACL=private",
        "export RCLONE_CONFIG_CERBERUS_R2_NO_CHECK_BUCKET=true",
        "export RCLONE_S3_NO_CHECK_BUCKET=true",
    ])
    script = STARTUP_TEMPLATE.format(job=job, job_q=shlex.quote(job), bucket_q=shlex.quote(creds["R2_BUCKET"]),
                                     env_lines=env_lines, channel=CHANNEL.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script, encoding="utf-8")
    out.chmod(0o600)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", default=None, help="credentials file (default: repo .env)")
    sub = parser.add_subparsers(dest="action", required=True)

    b = sub.add_parser("bundle")
    b.add_argument("--job", required=True)
    b.add_argument("--out", required=True)

    e = sub.add_parser("exec")
    e.add_argument("--job", required=True)
    e.add_argument("--cmd", required=True)
    e.add_argument("--timeout", type=int, default=1800)
    e.add_argument("--poll", type=float, default=2.0)
    e.add_argument("--no-wait", action="store_true")

    w = sub.add_parser("wait")
    w.add_argument("--job", required=True)
    w.add_argument("--timeout", type=float, default=600)

    p = sub.add_parser("put")
    p.add_argument("--job", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--key", required=True)

    g = sub.add_parser("get")
    g.add_argument("--job", required=True)
    g.add_argument("--key", required=True)
    g.add_argument("--dst", required=True)

    t = sub.add_parser("tail")
    t.add_argument("--job", required=True)
    t.add_argument("--key", required=True)
    t.add_argument("--lines", type=int, default=40)

    hb = sub.add_parser("heartbeat")
    hb.add_argument("--job", required=True)

    args = parser.parse_args(argv)
    creds = load_env(args.env_file)
    if args.action == "bundle":
        path = bundle(args.job, creds, Path(args.out))
        print(f"{path} (mode 600; pass with --startup-script @{path})")
        return 0
    channel = Channel(args.job, creds)
    if args.action == "exec":
        return channel.exec(args.cmd, timeout=args.timeout, poll=args.poll, wait=not args.no_wait)
    if args.action == "wait":
        hb = channel.wait_ready(args.timeout)
        if hb is None:
            print("[r2] no heartbeat within timeout", file=sys.stderr)
            return 1
        print(json.dumps(hb, indent=2))
        return 0
    if args.action == "put":
        channel.put(Path(args.src), args.key)
        print(f"uploaded {args.key}")
        return 0
    if args.action == "get":
        channel.get(args.key, Path(args.dst))
        print(f"downloaded {args.key} -> {args.dst}")
        return 0
    if args.action == "tail":
        text = channel.cat(args.key) or ""
        print("\n".join(text.splitlines()[-args.lines:]))
        return 0
    if args.action == "heartbeat":
        hb = channel.heartbeat()
        print(json.dumps(hb, indent=2) if hb else "no heartbeat yet")
        return 0 if hb else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
