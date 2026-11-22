"""host_doctor checks, from mocked system output only (no GPU, Docker daemon or network needed)."""

import json

import host_doctor as hd
import pytest

ALLOWLIST = json.loads((hd.ALLOWLIST_PATH).read_text())
UBUNTU_2404 = (
    'NAME="Ubuntu"\nVERSION="24.04.1 LTS (Noble Numbat)"\nID=ubuntu\nVERSION_ID="24.04"\n'
    'PRETTY_NAME="Ubuntu 24.04.1 LTS"\n'
)


def gpu(name="NVIDIA RTX A6000", mem=49140, driver="580.95.05"):
    return [hd.Gpu(name=name, memory_mib=mem, driver=driver)]


def ok(rc=0, out="", err=""):
    return hd.CmdResult(rc, out, err)


# --- os --------------------------------------------------------------------------------------------------------

def test_supported_os_passes():
    assert hd.check_os(UBUNTU_2404).status == hd.PASS
    assert hd.check_os('ID=ubuntu\nVERSION_ID="22.04"').status == hd.PASS


@pytest.mark.parametrize("text", ['ID=ubuntu\nVERSION_ID="20.04"', 'ID=fedora\nVERSION_ID="40"', 'ID=debian\n'])
def test_other_os_fails_with_a_fix(text):
    result = hd.check_os(text)
    assert result.status == hd.FAIL and "Ubuntu 22.04" in result.fix


def test_missing_os_release_fails():
    assert hd.check_os(None).status == hd.FAIL


def test_parse_os_release_handles_quotes_and_comments():
    parsed = hd.parse_os_release('# comment\nID=ubuntu\nVERSION_ID="24.04"\nEMPTY=\n')
    assert parsed["ID"] == "ubuntu" and parsed["VERSION_ID"] == "24.04" and parsed["EMPTY"] == ""


# --- gpu / rt cores / vram -------------------------------------------------------------------------------------

def test_no_gpu_fails_and_dependents_skip():
    assert hd.check_gpu([], "nvidia-smi: command not found").status == hd.FAIL
    assert hd.check_rt_cores([]).status == hd.SKIP
    assert hd.check_vram([]).status == hd.SKIP
    assert hd.check_driver([], ALLOWLIST).status == hd.SKIP


@pytest.mark.parametrize("name", ["NVIDIA A100-SXM4-80GB", "NVIDIA H100 PCIe", "Tesla V100-SXM2", "NVIDIA B200"])
def test_non_rt_core_gpus_fail(name):
    result = hd.check_rt_cores(gpu(name, 81920))
    assert result.status == hd.FAIL and "RT cores" in result.detail


@pytest.mark.parametrize("name", ["NVIDIA RTX 4090", "NVIDIA L4", "NVIDIA L40S", "NVIDIA RTX A6000",
                                  "NVIDIA A10G", "NVIDIA A40", "NVIDIA RTX 6000 Ada Generation"])
def test_rt_core_gpus_pass(name):
    assert hd.check_rt_cores(gpu(name)).status == hd.PASS


def test_unknown_gpu_model_warns():
    assert hd.check_rt_cores(gpu("NVIDIA Mystery Card")).status == hd.WARN


def test_vram_threshold():
    assert hd.check_vram(gpu(mem=16376)).status == hd.PASS
    assert hd.check_vram(gpu(mem=12288)).status == hd.FAIL
    assert hd.check_vram(gpu(mem=4096)).status == hd.FAIL


def test_parse_gpus_skips_junk_and_reads_units():
    text = "NVIDIA L4, 23034 MiB, 580.95.05\nnot a gpu line\nNVIDIA RTX 4090, 24564, 595.58.03\n"
    parsed = hd.parse_gpus(text)
    assert [(g.name, g.memory_mib, g.driver) for g in parsed] == [
        ("NVIDIA L4", 23034, "580.95.05"), ("NVIDIA RTX 4090", 24564, "595.58.03")]


def test_parse_gpus_empty():
    assert hd.parse_gpus("") == []


# --- driver ----------------------------------------------------------------------------------------------------

def test_validated_drivers_pass():
    assert hd.check_driver(gpu(driver="580.95.05"), ALLOWLIST).status == hd.PASS
    assert hd.check_driver(gpu(driver="595.58.03"), ALLOWLIST).status == hd.PASS


def test_bad_driver_fails_and_names_the_valid_set():
    result = hd.check_driver(gpu(driver="595.91.07"), ALLOWLIST)
    assert result.status == hd.FAIL and "595.91.07" in result.detail and "580.95.05" in result.fix


# --- ram -------------------------------------------------------------------------------------------------------

def test_ram_threshold():
    assert hd.check_ram("MemTotal:       32700000 kB\nMemFree: 1 kB\n").status == hd.PASS
    assert hd.check_ram("MemTotal:       15700000 kB\n").status == hd.FAIL
    assert hd.check_ram(None).status == hd.FAIL
    assert hd.check_ram("garbage").status == hd.FAIL


# --- docker / container ----------------------------------------------------------------------------------------

def test_docker_missing_fails_with_install_hint():
    result = hd.check_docker(ok(127, err="command not found"), ok(1))
    assert result.status == hd.FAIL and "docs.docker.com" in result.fix


def test_docker_daemon_down_fails():
    result = hd.check_docker(ok(0, "Docker version 27.0, build x"), ok(1, err="Cannot connect to the Docker daemon"))
    assert result.status == hd.FAIL and "daemon" in result.detail


def test_docker_without_nvidia_runtime_warns():
    result = hd.check_docker(ok(0, "Docker version 27.0"), ok(0, '{"runc":{}}'))
    assert result.status == hd.WARN and "nvidia-container-toolkit" in result.fix


def test_docker_with_nvidia_runtime_passes():
    assert hd.check_docker(ok(0, "Docker version 27.0"), ok(0, '{"runc":{},"nvidia":{}}')).status == hd.PASS


def test_container_gpu_skip_fail_pass():
    assert hd.check_container_gpu(None).status == hd.SKIP
    assert hd.check_container_gpu(ok(1, err="could not select device driver")).status == hd.FAIL
    assert hd.check_container_gpu(ok(0, "GPU 0: NVIDIA L4 (UUID: GPU-1)")).status == hd.PASS


# --- disk ------------------------------------------------------------------------------------------------------

def test_disk_thresholds():
    assert hd.check_disk("/d", 250e9).status == hd.PASS
    assert hd.check_disk("/d", 150e9).status == hd.WARN
    assert hd.check_disk("/d", 58e9).status == hd.FAIL
    assert hd.check_disk("/d", None).status == hd.FAIL


# --- bucket ----------------------------------------------------------------------------------------------------

GOOD_ENV = {"R2_ACCOUNT_ID": "acct", "R2_ACCESS_KEY_ID": "AKIAEXAMPLE", "R2_SECRET_ACCESS_KEY": "s" * 40,
            "R2_BUCKET": "cerberus-robotics"}


def test_bucket_missing_vars_names_them_without_values():
    result = hd.check_bucket_vars({"R2_ACCOUNT_ID": "x"})
    assert result.status == hd.FAIL
    assert "R2_ACCESS_KEY_ID" in result.detail and "x" not in result.detail


def test_bucket_reachable_passes_with_entries():
    result = hd.check_bucket_reachable(GOOD_ENV, ok(0, "rclone v1.67.0"), ok(0, "home/\nadmission.json\n"))
    assert result.status == hd.PASS and "2 entries" in result.detail


def test_bucket_denied_is_reported_and_secrets_masked():
    secret = "abcdef0123456789abcdef0123456789"
    env = dict(GOOD_ENV, R2_SECRET_ACCESS_KEY=secret)
    result = hd.check_bucket_reachable(env, ok(0, "rclone v1.67.0"),
                                       ok(1, err=f"AccessDenied key={secret}"))
    assert result.status == hd.FAIL and "AccessDenied" in result.detail
    assert secret not in result.detail


def test_bucket_requires_rclone_and_a_recent_one():
    assert hd.check_bucket_reachable(GOOD_ENV, ok(127), None).status == hd.FAIL
    old = hd.check_bucket_reachable(GOOD_ENV, ok(0, "rclone v1.58.1"), ok(0, ""))
    assert old.status == hd.FAIL and "1.59" in old.detail


def test_bucket_vars_take_precedence_over_reachability():
    assert hd.check_bucket_reachable({}, ok(0, "rclone v1.67"), ok(0, "")).status == hd.FAIL


# --- orchestration ----------------------------------------------------------------------------------------------

def _runner_factory(bad_driver=False, no_docker=False):
    def runner(cmd, env=None):
        joined = " ".join(cmd)
        if cmd[0] == "nvidia-smi":
            driver = "595.91.07" if bad_driver else "580.95.05"
            return ok(0, f"NVIDIA RTX A6000, 49140, {driver}")
        if joined.startswith("docker --version"):
            return ok(127, err="command not found") if no_docker else ok(0, "Docker version 27.0, build x")
        if joined.startswith("docker info"):
            return ok(0, '{"nvidia":{}}')
        if joined.startswith("docker run"):
            return ok(0, "GPU 0: NVIDIA RTX A6000 (UUID: GPU-1)")
        if joined.startswith("rclone version"):
            return ok(0, "rclone v1.67.0")
        if joined.startswith("rclone lsf"):
            assert "RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY" in env  # subprocess env only
            return ok(0, "home/\n")
        raise AssertionError(f"unexpected command {joined}")
    return runner


def _reader(path):
    return UBUNTU_2404 if path.endswith("os-release") else "MemTotal: 65000000 kB\n"


def test_run_checks_all_pass():
    results = hd.run_checks(skip=set(), data_dir=hd.Path("/d"), env=GOOD_ENV, runner=_runner_factory(),
                            reader=_reader, free_bytes=lambda p: 250e9, allowlist=ALLOWLIST)
    assert [r.name for r in results] == list(hd.ALL_CHECKS)
    assert [r.name for r in results if r.status != hd.PASS] == []


def test_run_checks_failures_are_collected_not_raised():
    results = hd.run_checks(skip=set(), data_dir=hd.Path("/d"), env={}, runner=_runner_factory(bad_driver=True),
                            reader=_reader, free_bytes=lambda p: 10e9, allowlist=ALLOWLIST)
    failed = {r.name for r in results if r.status == hd.FAIL}
    assert failed == {"driver", "disk", "bucket"}


def test_run_checks_no_docker_skips_container_probe():
    results = hd.run_checks(skip=set(), data_dir=hd.Path("/d"), env=GOOD_ENV,
                            runner=_runner_factory(no_docker=True), reader=_reader, free_bytes=lambda p: 250e9,
                            allowlist=ALLOWLIST)
    by_name = {r.name: r for r in results}
    assert by_name["docker"].status == hd.FAIL
    assert by_name["container_gpu"].status == hd.SKIP


def test_skip_marks_checks_and_does_not_run_them():
    def runner(cmd, env=None):
        raise AssertionError("skipped checks must not touch the host")

    results = hd.run_checks(skip=set(hd.ALL_CHECKS), data_dir=hd.Path("/d"), env={}, runner=runner,
                            reader=_reader, free_bytes=lambda p: 1, allowlist=ALLOWLIST)
    assert all(r.status == hd.SKIP for r in results)


def test_report_and_format_agree_on_the_verdict():
    passing = [hd.CheckResult("os", hd.PASS, "ok")]
    failing = [hd.CheckResult("os", hd.FAIL, "bad", "fix it")]
    assert hd.build_report(passing)["overall"] == hd.PASS
    assert hd.build_report(failing) == {**hd.build_report(failing), "failed": ["os"], "overall": hd.FAIL}
    assert "HOST DOCTOR: PASS" in hd.format_results(passing)
    assert "fix: fix it" in hd.format_results(failing)


def test_main_writes_json_and_returns_one_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(hd, "run_checks", lambda **kw: [hd.CheckResult("os", hd.FAIL, "bad")])
    out = tmp_path / "r.json"
    assert hd.main(["--json", str(out)]) == 1
    assert json.loads(out.read_text())["failed"] == ["os"]


def test_main_returns_zero_when_nothing_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(hd, "run_checks", lambda **kw: [hd.CheckResult("os", hd.PASS, "ok"),
                                                        hd.CheckResult("disk", hd.WARN, "tight")])
    assert hd.main(["--json", str(tmp_path / "r.json")]) == 0


def test_main_rejects_an_unknown_skip(capsys):
    assert hd.main(["--skip", "nonsense"]) == 2
    assert "nonsense" in capsys.readouterr().err


def test_main_reads_credentials_from_env_file(tmp_path, monkeypatch):
    seen = {}

    def fake_run_checks(**kw):
        seen.update(kw["env"])
        return [hd.CheckResult("bucket", hd.PASS, "ok")]

    monkeypatch.setattr(hd, "run_checks", fake_run_checks)
    env_file = tmp_path / ".env"
    env_file.write_text('R2_ACCOUNT_ID="acct"\nexport R2_BUCKET=b\nOTHER=x\n')
    assert hd.main(["--env-file", str(env_file)]) == 0
    assert seen == {"R2_ACCOUNT_ID": "acct", "R2_BUCKET": "b"}


def test_env_file_values_are_never_printed(tmp_path, monkeypatch, capsys):
    secret = "s" * 40
    monkeypatch.setattr(hd, "run_checks", lambda **kw: [hd.CheckResult("bucket", hd.FAIL, "AccessDenied")])
    env_file = tmp_path / ".env"
    env_file.write_text(f"R2_SECRET_ACCESS_KEY={secret}\n")
    hd.main(["--env-file", str(env_file)])
    assert secret not in capsys.readouterr().out


def test_free_bytes_walks_up_to_an_existing_parent(tmp_path):
    assert hd.free_bytes_at(tmp_path / "not" / "yet") > 0


def test_load_env_file_ignores_junk(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# c\n\nBROKEN\nA=1\n")
    assert hd.load_env_file(path) == {"A": "1"}
