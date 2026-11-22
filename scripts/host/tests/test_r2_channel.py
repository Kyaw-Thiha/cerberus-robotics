"""The R2 command channel: bundling, round trip, timeout, ordering, heartbeat and file transfer.

The instance loop (r2_channel.sh) runs for real against a local fake bucket; only `rclone` is stubbed.
"""

import os
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest
import r2_channel
from r2_channel import Channel

REPO = Path(__file__).resolve().parents[3]
HOST = REPO / "scripts" / "host"
FAKE = HOST / "tests" / "fake_rclone.py"

CREDS = {"R2_ACCOUNT_ID": "acct", "R2_ACCESS_KEY_ID": "AKIAFAKE", "R2_SECRET_ACCESS_KEY": "fake" * 10,
         "R2_BUCKET": "testbucket"}


@pytest.fixture
def bucket(tmp_path):
    root = tmp_path / "bucket"
    root.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "rclone"
    wrapper.write_text(f'#!/bin/bash\nexec python3 {FAKE} "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", FAKE_BUCKET=str(root))
    return root, env


@pytest.fixture
def channel(bucket, tmp_path):
    root, env = bucket
    env = dict(env, JOB_ID="testjob", R2_BUCKET="testbucket", POLL_S="0.2", HB_S="1",
               STATE_DIR=str(tmp_path / "state"), WORK_DIR=str(tmp_path / "work"),
               RCLONE_CONFIG_CERBERUS_R2_TYPE="s3",
               RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID="x", RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY="y",
               RCLONE_CONFIG_CERBERUS_R2_NO_CHECK_BUCKET="true")
    proc = subprocess.Popen(["bash", str(HOST / "r2_channel.sh")], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ch = Channel("testjob", CREDS)
    ch.env = dict(env)  # same fake PATH/bucket for the client
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and ch.state() is None:
        time.sleep(0.1)
    assert ch.state() is not None, proc.stdout.read() if proc.poll() is not None else "channel did not start"
    yield ch
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_bundle_is_private_and_embeds_the_channel(tmp_path, capsys):
    out = r2_channel.bundle("testjob", CREDS, tmp_path / "startup.sh")
    text = out.read_text()
    assert out.stat().st_mode & 0o777 == 0o600
    assert "r2_channel.sh" in text and "JOB_ID=testjob" in text
    assert "RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID" in text
    assert "systemctl enable --now cerberus-channel" in text  # root path
    assert "setsid nohup" in text                             # non-root path
    printed = capsys.readouterr().out
    assert CREDS["R2_SECRET_ACCESS_KEY"] not in printed


def test_state_and_heartbeat(channel):
    state = channel.state()
    assert state["job"] == "testjob" and state["arch"]
    hb = channel.heartbeat()
    assert hb["job"] == "testjob" and "ts" in hb and "disk" in hb


def test_exec_round_trip_and_exit_status(channel, capsys):
    rc = channel.exec("echo hello-from-instance; echo oops >&2; exit 3", timeout=30)
    captured = capsys.readouterr()
    assert rc == 3
    assert "hello-from-instance" in captured.out
    assert "oops" in captured.err


def test_exec_timeout_is_enforced_instance_side(channel):
    started = time.monotonic()
    rc = channel.exec("sleep 30", timeout=1)
    assert rc == 124
    assert time.monotonic() - started < 20  # not the full sleep


def test_commands_run_in_order_with_increasing_sequence(channel):
    assert channel.exec("echo one", timeout=20) == 0
    assert channel.exec("echo two", timeout=20) == 0
    names = channel.lsf("out/")
    assert "0001.rc" in names and "0002.rc" in names
    assert channel.next_seq() == 3


def test_put_and_get_round_trip(tmp_path, channel):
    src = tmp_path / "payload.bin"
    src.write_bytes(b"\x00canary\xff" * 100)
    channel.put(src, "input/payload.bin")
    dst = tmp_path / "back.bin"
    channel.get("input/payload.bin", dst)
    assert dst.read_bytes() == src.read_bytes()


def test_pull_gets_a_prefix(tmp_path, channel):
    assert channel.exec("echo a > a.txt; echo b > b.txt", timeout=20) == 0
    channel.pull("out/", tmp_path / "outs")
    assert (tmp_path / "outs" / "0001.rc").read_text().strip() == "0"


def test_only_the_job_prefix_is_used():
    ch = Channel("job1", CREDS)
    assert ch.remote("out/x").startswith("cerberus_r2:testbucket/jobs/job1/")
    with pytest.raises(SystemExit):
        Channel("../escape", CREDS)


def test_next_seq_starts_at_one(channel):
    assert channel.next_seq() == 1
