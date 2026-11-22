"""The provider-neutral host scripts: the library helpers, bootstrap, and the session plan (stubbed commands)."""

import os
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HOST = REPO / "scripts" / "host"


def bash(script, env=None, extra_env=None, check=False):
    e = dict(os.environ, PYTHONPATH=str(REPO), **(env or {}), **(extra_env or {}))
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=e, timeout=120, check=check)


def stub(tmp_path, name, body):
    path = tmp_path / "bin" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_lib_syntax_and_helpers():
    for script in [HOST / "session_lib.sh", HOST / "run_session.sh", HOST / "bootstrap_host.sh",
                   HOST / "run_stack_check.sh", HOST / "run_scene_eval.sh", HOST / "run_scene_admission.sh",
                   HOST / "run_slam_eval.sh"]:
        r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
        assert r.returncode == 0, f"{script}: {r.stderr}"


def test_run_isaac_retries_a_crash_then_succeeds(tmp_path):
    counter = tmp_path / "n"
    r = bash(f'''
source {HOST}/session_lib.sh
REPORT_DIR={tmp_path}/report; mkdir -p "$REPORT_DIR"
fail_twice() {{
  n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}
  if [ "$n" -lt 3 ]; then echo "A crash has occurred"; exit 139; fi
  echo "Simulation App Startup Complete"; exit 0
}}
run_isaac stage 5 3 -- fail_twice
echo "rc=$?"
''')
    assert "rc=0" in r.stdout, r.stdout + r.stderr
    assert "attempt 3/3" in r.stdout + r.stderr
    assert (tmp_path / "report" / "stage.log").is_file()


def test_run_isaac_gives_up_after_the_attempts(tmp_path):
    r = bash(f'''
source {HOST}/session_lib.sh
REPORT_DIR={tmp_path}/report; mkdir -p "$REPORT_DIR"
run_isaac stage 5 2 -- bash -c 'echo "A crash has occurred"; exit 139'
echo "rc=$?"
''')
    assert "rc=1" in r.stdout


def test_r2_helper_refuses_without_credentials_and_prints_no_values(tmp_path):
    r = bash(f'''
source {HOST}/session_lib.sh
unset R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET
cerberus_r2 lsf cerberus_r2:bucket
echo "rc=$?"
''')
    assert "rc=1" in r.stdout and "missing" in r.stderr
    # Variable names may appear in the hint; the value must not (there is none here, and none is echoed).
    assert "R2_ACCESS_KEY_ID" in r.stderr and "s3://" not in r.stdout + r.stderr


def test_bootstrap_is_idempotent_for_an_image_already_present(tmp_path):
    calls = tmp_path / "calls"
    stub(tmp_path, "docker", f'printf "%s\\n" "$*" >> {calls}; [[ "$1 $2" == "image inspect" ]] && exit 0; exit 0\n')
    r = bash(f'{HOST}/bootstrap_host.sh --data-dir {tmp_path}/data --skip scene ',
             extra_env={"PATH": f"{tmp_path}/bin:" + os.environ["PATH"]})
    assert "image PASS" in r.stdout, r.stdout + r.stderr
    assert "pull" not in calls.read_text()


def test_bootstrap_pulls_when_the_image_is_missing(tmp_path):
    calls = tmp_path / "calls"
    stub(tmp_path, "docker", f'printf "%s\\n" "$*" >> {calls}; [[ "$1 $2" == "image inspect" ]] && exit 1; exit 0\n')
    r = bash(f'{HOST}/bootstrap_host.sh --data-dir {tmp_path}/data --skip scene ',
             extra_env={"PATH": f"{tmp_path}/bin:" + os.environ["PATH"]})
    assert "image PASS" in r.stdout
    assert calls.read_text().count("pull") == 1


def test_bootstrap_fails_a_tiny_disk_before_pulling(tmp_path):
    calls = tmp_path / "calls"
    stub(tmp_path, "docker", f'printf "%s\\n" "$*" >> {calls}; exit 0\n')
    r = bash(f'{HOST}/bootstrap_host.sh --data-dir {tmp_path}/data --skip scene --min-disk-gb 999999 ',
             extra_env={"PATH": f"{tmp_path}/bin:" + os.environ["PATH"]})
    assert "disk FAIL" in r.stdout
    assert "pull" not in calls.read_text()


def test_bootstrap_scene_needs_credentials_or_says_so(tmp_path):
    r = bash(f'{HOST}/bootstrap_host.sh --data-dir {tmp_path}/data --env-file {tmp_path}/none --skip image ',
             extra_env={"PATH": f"{tmp_path}/bin:" + os.environ["PATH"]})
    assert "scene FAIL" in r.stdout and "no credentials file" in r.stdout


def test_run_session_dry_run_lists_only_the_requested_stages(tmp_path):
    r = bash(f'{HOST}/run_session.sh --dry-run --skip-gate --stages stand,nav --out-dir {tmp_path}/out')
    assert r.returncode == 0, r.stderr
    assert "stage stand" in r.stdout and "stage nav" in r.stdout
    assert "stage demo" not in r.stdout


def test_run_session_refuses_to_shut_down_a_non_vm(tmp_path):
    stub(tmp_path, "systemd-detect-virt", "exit 1\n")
    r = bash(f'{HOST}/run_session.sh --dry-run --skip-gate --stages stand --out-dir {tmp_path}/out --shutdown-after',
             extra_env={"PATH": f"{tmp_path}/bin:" + os.environ["PATH"]})
    assert "refusing --shutdown-after" in r.stdout
    assert "shutting this machine down" not in r.stdout


def test_run_session_would_shut_down_a_vm(tmp_path):
    stub(tmp_path, "systemd-detect-virt", "exit 0\n")
    r = bash(f'{HOST}/run_session.sh --dry-run --skip-gate --stages stand --out-dir {tmp_path}/out --shutdown-after',
             extra_env={"PATH": f"{tmp_path}/bin:" + os.environ["PATH"]})
    assert "shutting this machine down" in r.stdout
    assert r.returncode == 0


def test_every_archived_entry_point_has_a_provider_neutral_replacement():
    for name in ("run_stack_check.sh", "run_scene_admission.sh", "run_scene_eval.sh", "run_slam_eval.sh"):
        text = (HOST / name).read_text()
        assert "pod_deadman" not in text and "runpod" not in text.lower() and "vast" not in text.lower()
        assert "session_lib.sh" in text
