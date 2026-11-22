#!/usr/bin/env python3
"""Host doctor: is this machine able to run the Cerberus Isaac Sim stack? PASS/FAIL per item, with the exact fix.

Run it on any host (own workstation, cluster node, cloud VM) before spending time on downloads::

    python3 scripts/host/host_doctor.py                       # all checks, human-readable
    python3 scripts/host/host_doctor.py --json report.json    # also write the machine-readable report
    python3 scripts/host/host_doctor.py --env-file .env       # bucket credentials from a file (never printed)
    python3 scripts/host/host_doctor.py --skip container,bucket   # skip checks that need Docker or the network

Exit status: 0 when no check FAILed (WARN is allowed), 1 when any check FAILed, 2 on bad arguments.

Every check is a pure function of text a collector read from the host (``nvidia-smi``, ``/etc/os-release``,
``/proc/meminfo`` ...), so the unit tests feed it canned output. The requirements and their sources are in
``docs/host_contract.md``; the driver rules live in ``scripts/host/driver_allowlist.json``.

Secrets: the bucket check reads ``R2_*`` from the environment or ``--env-file`` into the environment of ONE rclone
subprocess. Values are never printed or written to the report; any rclone output goes through ``redact.mask_text``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALLOWLIST_PATH = HERE / "driver_allowlist.json"
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from driver_gate import TIER_A, TIER_B, classify_driver
from redact import mask_text

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"

SUPPORTED_OS = {"ubuntu": ("22.04", "24.04")}
MIN_VRAM_MIB = 16000  # "16 GB" as cards report it; a 16 GB card shows 16376-16384 MiB, a 12 GB card 12288
MIN_RAM_GIB = 30.0  # "32 GB" as the kernel reports it: a 32 GiB machine shows ~31.3 GiB after reservations
MIN_DISK_GB = 100.0
RECOMMENDED_DISK_GB = 200.0
REQUIRED_BUCKET_VARS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
BUCKET_PROBE_PREFIX = "worlds/grscenes"  # the token is bucket-scoped: the bucket root cannot be listed by design
CONTAINER_PROBE_IMAGE = "ubuntu:22.04"  # the NVIDIA toolkit injects nvidia-smi into any image

# GPUs without RT cores (or unsupported by NVIDIA for Isaac Sim RTX rendering). Matched first.
NO_RT_CORES = re.compile(r"\b(A100|A800|A30|H100|H800|H200|H20|B100|B200|B300|GB200|GH200|V100|P100|P40)\b")
# GPU families that carry RT cores.
HAS_RT_CORES = re.compile(r"\bRTX\b|\bL4\b|\bL40S?\b|\bA10G?\b|\bA16\b|\bA40\b|\bA2\b|\bT4\b")


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    fix: str = ""


@dataclass
class Gpu:
    name: str
    memory_mib: int
    driver: str


@dataclass
class CmdResult:
    rc: int
    out: str
    err: str = ""


Runner = Callable[[list[str], Mapping[str, str] | None], CmdResult]


def run_cmd(cmd: list[str], env: Mapping[str, str] | None = None, timeout: float = 120) -> CmdResult:
    """Run ``cmd``; a missing binary or a timeout is a non-zero result, never an exception."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False,
                              env=None if env is None else dict(env))
    except FileNotFoundError:
        return CmdResult(127, "", f"{cmd[0]}: command not found")
    except (OSError, subprocess.SubprocessError) as e:
        return CmdResult(1, "", f"{type(e).__name__}: {e}")
    return CmdResult(proc.returncode, proc.stdout.strip(), proc.stderr.strip())


# --- parsers (pure) ---------------------------------------------------------------------------------------------


def parse_os_release(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def parse_gpus(csv_text: str) -> list[Gpu]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits``."""
    gpus = []
    for line in csv_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        mem = re.search(r"\d+", parts[1])
        if mem:
            gpus.append(Gpu(name=parts[0], memory_mib=int(mem.group()), driver=parts[2]))
    return gpus


def parse_meminfo_gib(text: str) -> float | None:
    found = re.search(r"^MemTotal:\s+(\d+)\s*kB", text, re.MULTILINE)
    return int(found.group(1)) / (1024 * 1024) if found else None


def load_env_file(path: Path) -> dict[str, str]:
    """``KEY=VALUE`` lines (optionally ``export``-prefixed, optionally quoted). Values are returned, never printed."""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


# --- checks (pure functions of collected text) ------------------------------------------------------------------


def check_os(os_release_text: str | None) -> CheckResult:
    if not os_release_text:
        return CheckResult("os", FAIL, "cannot read /etc/os-release (not Linux?)",
                           "Use Ubuntu 22.04 or 24.04 (NVIDIA's validated platforms for Isaac Sim 6).")
    info = parse_os_release(os_release_text)
    distro, version = info.get("ID", "").lower(), info.get("VERSION_ID", "")
    label = f"{info.get('PRETTY_NAME') or distro + ' ' + version}"
    if version in SUPPORTED_OS.get(distro, ()):
        return CheckResult("os", PASS, label)
    return CheckResult("os", FAIL, f"{label} is not Ubuntu 22.04/24.04",
                       "Create the machine from an Ubuntu 22.04 or 24.04 image. Other distributions may work but "
                       "are not validated by NVIDIA and have not been run by us.")


def check_gpu(gpus: list[Gpu], smi_error: str = "") -> CheckResult:
    if not gpus:
        return CheckResult("gpu", FAIL, f"nvidia-smi reported no GPU ({smi_error or 'no output'})",
                           "Install the NVIDIA driver (see the driver check) and reboot; on a cloud VM pick a GPU "
                           "instance type.")
    names = ", ".join(f"{g.name} ({g.memory_mib} MiB)" for g in gpus)
    return CheckResult("gpu", PASS, names)


def check_rt_cores(gpus: list[Gpu]) -> CheckResult:
    if not gpus:
        return CheckResult("rt_cores", SKIP, "no GPU to inspect")
    verdicts = []
    for g in gpus:
        if NO_RT_CORES.search(g.name):
            verdicts.append((g.name, False))
        elif HAS_RT_CORES.search(g.name):
            verdicts.append((g.name, True))
        else:
            verdicts.append((g.name, None))
    if any(v is False for _, v in verdicts):
        bad = ", ".join(n for n, v in verdicts if v is False)
        return CheckResult("rt_cores", FAIL, f"{bad}: no RT cores (NVIDIA lists A100/H100 as unsupported for Isaac Sim)",
                           "Use an RT-core GPU: RTX 4080/4090/5080, RTX A4000-A6000, RTX 6000 Ada, L4, L40S, A10G.")
    if any(v is None for _, v in verdicts):
        unknown = ", ".join(n for n, v in verdicts if v is None)
        return CheckResult("rt_cores", WARN, f"cannot tell whether {unknown} has RT cores",
                           "Look the model up in NVIDIA's product page; Isaac Sim RTX rendering needs RT cores.")
    return CheckResult("rt_cores", PASS, "RT-core GPU family")


def check_vram(gpus: list[Gpu]) -> CheckResult:
    if not gpus:
        return CheckResult("vram", SKIP, "no GPU to inspect")
    best = max(g.memory_mib for g in gpus)
    if best >= MIN_VRAM_MIB:
        return CheckResult("vram", PASS, f"{best} MiB on the largest GPU (>= 16 GB)")
    return CheckResult("vram", FAIL, f"{best} MiB on the largest GPU, below 16 GB",
                       "Use a GPU with at least 16 GB (24 GB or more is more comfortable with RTX rendering).")


def check_driver(gpus: list[Gpu], allowlist: dict) -> CheckResult:
    if not gpus:
        return CheckResult("driver", SKIP, "no GPU, so no driver to inspect")
    drivers = sorted({g.driver for g in gpus})
    results = [(d, *classify_driver(d, allowlist)) for d in drivers]
    bad = [(d, reason) for d, tier, reason in results if tier not in (TIER_A, TIER_B)]
    if bad:
        d, reason = bad[0]
        return CheckResult(
            "driver", FAIL, f"driver {d} is outside the validated set ({reason})",
            "Install the 580 production branch at 580.95.05 or newer (for example "
            "`sudo apt-get install -y cuda-drivers-580` from NVIDIA's CUDA apt repository, or the .run installer "
            "from nvidia.com/drivers), or exactly 595.58.03, then reboot. 595.71.05, 595.79, 595.84 and 595.91.07 "
            "crash Isaac Sim 6 at startup (exit 139).")
    return CheckResult("driver", PASS, ", ".join(drivers))


def check_ram(meminfo_text: str | None) -> CheckResult:
    gib = parse_meminfo_gib(meminfo_text or "")
    if gib is None:
        return CheckResult("ram", FAIL, "cannot read MemTotal from /proc/meminfo", "Run the doctor on the Linux host.")
    if gib >= MIN_RAM_GIB:
        return CheckResult("ram", PASS, f"{gib:.1f} GiB (>= 32 GB nominal)")
    return CheckResult("ram", FAIL, f"{gib:.1f} GiB, below 32 GB",
                       "Use a machine with at least 32 GB of RAM (64 GB recommended for full homes).")


def check_docker(version: CmdResult, info: CmdResult) -> CheckResult:
    if version.rc == 127:
        return CheckResult("docker", FAIL, "docker is not installed",
                           "Install Docker Engine: https://docs.docker.com/engine/install/ubuntu/ and add your user "
                           "to the docker group (`sudo usermod -aG docker $USER`, then log in again).")
    if version.rc != 0 or info.rc != 0:
        reason = (info.err or version.err or "docker info failed").splitlines()[0][:160]
        return CheckResult("docker", FAIL, f"docker is installed but the daemon is not usable: {reason}",
                           "Start it with `sudo systemctl enable --now docker`; if the error is a permission "
                           "problem, add your user to the docker group and log in again.")
    runtimes = info.out
    if "nvidia" not in runtimes.lower():
        return CheckResult("docker", WARN, f"{version.out.splitlines()[0] if version.out else 'docker'}; the nvidia "
                           "runtime is not registered",
                           "Install the NVIDIA Container Toolkit: `sudo apt-get install -y nvidia-container-toolkit; "
                           "sudo nvidia-ctk runtime configure --runtime=docker; sudo systemctl restart docker`.")
    return CheckResult("docker", PASS, f"{version.out.splitlines()[0] if version.out else 'docker'}; nvidia runtime "
                       "registered")


def check_container_gpu(result: CmdResult | None, image: str = CONTAINER_PROBE_IMAGE) -> CheckResult:
    if result is None:
        return CheckResult("container_gpu", SKIP, "docker is not usable, so the container probe was not run")
    if result.rc == 0 and "GPU" in result.out:
        first = result.out.splitlines()[0]
        return CheckResult("container_gpu", PASS, f"nvidia-smi inside {image}: {first}")
    reason = (result.err or result.out or "no output").splitlines()[-1][:200]
    return CheckResult("container_gpu", FAIL, f"`docker run --gpus all {image} nvidia-smi -L` failed: {reason}",
                       "Install and configure the NVIDIA Container Toolkit (nvidia-ctk runtime configure "
                       "--runtime=docker, restart docker). If the image cannot be pulled, check network access.")


def check_disk(path: str, free_bytes: int | None, min_gb: float = MIN_DISK_GB,
               recommended_gb: float = RECOMMENDED_DISK_GB) -> CheckResult:
    if free_bytes is None:
        return CheckResult("disk", FAIL, f"cannot read free space at {path}", "Create the directory or pass --data-dir.")
    free_gb = free_bytes / 1e9
    if free_gb < min_gb:
        return CheckResult("disk", FAIL, f"{free_gb:.0f} GB free at {path}, below {min_gb:.0f} GB",
                           f"Free space or attach a larger volume and pass --data-dir. Budget: the image, the Kit and "
                           f"shader caches, one GRScenes home (~25 GB) and results. {recommended_gb:.0f} GB is "
                           "recommended.")
    if free_gb < recommended_gb:
        return CheckResult("disk", WARN, f"{free_gb:.0f} GB free at {path} (enough, {recommended_gb:.0f} GB "
                           "recommended)", "")
    return CheckResult("disk", PASS, f"{free_gb:.0f} GB free at {path}")


def check_bucket_vars(env: Mapping[str, str]) -> CheckResult:
    missing = [v for v in REQUIRED_BUCKET_VARS if not env.get(v)]
    if missing:
        return CheckResult("bucket", FAIL, f"missing variables: {', '.join(missing)} (names only; values never shown)",
                           "Put R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and R2_BUCKET in .env and run "
                           "with --env-file .env, or export them. Any S3-compatible bucket with the same layout "
                           "(docs/storage.md) works if you adapt the endpoint.")
    return CheckResult("bucket", PASS, "credential variables present")


def check_bucket_reachable(env: Mapping[str, str], rclone_version: CmdResult,
                           listing: CmdResult | None) -> CheckResult:
    base = check_bucket_vars(env)
    if base.status != PASS:
        return base
    if rclone_version.rc != 0:
        return CheckResult("bucket", FAIL, "rclone is not installed", "Install rclone >= 1.59: "
                           "`curl -fsSL https://rclone.org/install.sh | sudo bash` (older builds reject the "
                           "Cloudflare S3 provider).")
    found = re.search(r"rclone v(\d+)\.(\d+)", rclone_version.out)
    if found and (int(found.group(1)), int(found.group(2))) < (1, 59):
        return CheckResult("bucket", FAIL, f"rclone {found.group(0)} is older than 1.59",
                           "Upgrade rclone: `curl -fsSL https://rclone.org/install.sh | sudo bash`.")
    if listing is None:
        return CheckResult("bucket", SKIP, "reachability not probed")
    if listing.rc == 0:
        entries = len([ln for ln in listing.out.splitlines() if ln.strip()])
        return CheckResult("bucket", PASS, f"authenticated; {entries} entries under {BUCKET_PROBE_PREFIX}/")
    reason = mask_text((listing.err or listing.out or "no output").splitlines()[0])[:200]
    return CheckResult("bucket", FAIL, f"listing {BUCKET_PROBE_PREFIX}/ failed: {reason}",
                       "403/AccessDenied: the token is not scoped to this bucket, or you probed the bucket root (a "
                       "bucket-scoped token cannot list it; list a prefix). Other errors: check the account id, "
                       "keys, clock and network.")


# --- collection and orchestration -------------------------------------------------------------------------------

ALL_CHECKS = ("os", "gpu", "rt_cores", "vram", "driver", "ram", "docker", "container_gpu", "disk", "bucket")
GPU_CHECKS = ("gpu", "rt_cores", "vram", "driver")
SKIP_ALIASES = {"container": "container_gpu", "gpus": "gpu"}


def rclone_env(env: Mapping[str, str]) -> dict[str, str]:
    """An env-defined rclone remote (no config file), like scripts/sync_to_r2.py. Contains secrets: subprocess only."""
    out = dict(os.environ)
    prefix = "RCLONE_CONFIG_CERBERUS_R2_"
    out[prefix + "TYPE"] = "s3"
    out[prefix + "PROVIDER"] = "Cloudflare"
    out[prefix + "ACCESS_KEY_ID"] = env["R2_ACCESS_KEY_ID"]
    out[prefix + "SECRET_ACCESS_KEY"] = env["R2_SECRET_ACCESS_KEY"]
    out[prefix + "ENDPOINT"] = f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    out[prefix + "ACL"] = "private"
    out[prefix + "NO_CHECK_BUCKET"] = "true"
    return out


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def free_bytes_at(path: Path) -> int | None:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def run_checks(*, skip: set[str], data_dir: Path, env: Mapping[str, str], min_disk_gb: float = MIN_DISK_GB,
               container_image: str = CONTAINER_PROBE_IMAGE, runner: Runner = run_cmd,
               reader: Callable[[str], str | None] = read_text,
               free_bytes: Callable[[Path], int | None] = free_bytes_at,
               allowlist: dict | None = None) -> list[CheckResult]:
    """Run every non-skipped check. ``runner``/``reader``/``free_bytes`` are the seams the tests replace."""
    allowlist = allowlist or json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    results: list[CheckResult] = []

    def want(name: str) -> bool:
        if name in skip:
            results.append(CheckResult(name, SKIP, "skipped by --skip"))
            return False
        return True

    if want("os"):
        results.append(check_os(reader("/etc/os-release")))

    gpus: list[Gpu] = []
    smi_error = ""
    if any(name not in skip for name in GPU_CHECKS):  # never touch the GPU stack when all of these are skipped
        smi = runner(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                     None)
        gpus = parse_gpus(smi.out) if smi.rc == 0 else []
        smi_error = (smi.err or smi.out).splitlines()[0] if (smi.err or smi.out) else ""
    if want("gpu"):
        results.append(check_gpu(gpus, smi_error))
    if want("rt_cores"):
        results.append(check_rt_cores(gpus))
    if want("vram"):
        results.append(check_vram(gpus))
    if want("driver"):
        results.append(check_driver(gpus, allowlist))
    if want("ram"):
        results.append(check_ram(reader("/proc/meminfo")))

    docker_ok = False
    if "docker" in skip:
        results.append(CheckResult("docker", SKIP, "skipped by --skip"))
    else:
        version = runner(["docker", "--version"], None)
        info = runner(["docker", "info", "--format", "{{json .Runtimes}}"], None) if version.rc == 0 else CmdResult(1, "")
        results.append(check_docker(version, info))
        docker_ok = version.rc == 0 and info.rc == 0
    if "container_gpu" in skip:
        results.append(CheckResult("container_gpu", SKIP, "skipped by --skip"))
    elif not docker_ok:
        results.append(check_container_gpu(None, container_image))
    else:
        probe = runner(["docker", "run", "--rm", "--gpus", "all", container_image, "nvidia-smi", "-L"], None)
        results.append(check_container_gpu(probe, container_image))

    if want("disk"):
        results.append(check_disk(str(data_dir), free_bytes(data_dir), min_disk_gb))

    if "bucket" in skip:
        results.append(CheckResult("bucket", SKIP, "skipped by --skip"))
    elif check_bucket_vars(env).status != PASS:
        results.append(check_bucket_vars(env))
    else:
        rc_version = runner(["rclone", "version"], None)
        listing = None
        if rc_version.rc == 0:
            listing = runner(["rclone", "lsf", f"cerberus_r2:{env['R2_BUCKET']}/{BUCKET_PROBE_PREFIX}",
                              "--max-depth", "1"], rclone_env(env))
        results.append(check_bucket_reachable(env, rc_version, listing))
    return results


def build_report(results: list[CheckResult]) -> dict:
    failed = [r.name for r in results if r.status == FAIL]
    return {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node() or "unknown",
        "overall": FAIL if failed else PASS,
        "failed": failed,
        "checks": [asdict(r) for r in results],
    }


def format_results(results: list[CheckResult]) -> str:
    lines = []
    for r in results:
        lines.append(f"[{r.status:4}] {r.name:14} {r.detail}")
        if r.fix and r.status in (FAIL, WARN):
            lines.append(f"       fix: {r.fix}")
    failed = [r.name for r in results if r.status == FAIL]
    lines.append("")
    lines.append(f"HOST DOCTOR: {'FAIL (' + ', '.join(failed) + ')' if failed else 'PASS'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", metavar="PATH", help="write the JSON report here")
    parser.add_argument("--env-file", metavar="PATH", help="read R2_* credentials from this file (never printed)")
    parser.add_argument("--data-dir", default=os.environ.get("CERBERUS_DATA_DIR", str(Path.home() / "cerberus-data")),
                        help="directory whose free space is checked (image cache, scenes, results)")
    parser.add_argument("--min-disk-gb", type=float, default=MIN_DISK_GB)
    parser.add_argument("--container-image", default=CONTAINER_PROBE_IMAGE)
    parser.add_argument("--skip", default="", help=f"comma-separated checks to skip: {', '.join(ALL_CHECKS)}")
    args = parser.parse_args(argv)
    skip = {SKIP_ALIASES.get(s.strip(), s.strip()) for s in args.skip.split(",") if s.strip()}
    unknown = skip - set(ALL_CHECKS)
    if unknown:
        print(f"host_doctor: unknown check(s) in --skip: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    env: dict[str, str] = {k: os.environ[k] for k in REQUIRED_BUCKET_VARS if k in os.environ}
    if args.env_file:
        try:
            env.update({k: v for k, v in load_env_file(Path(args.env_file)).items() if k in REQUIRED_BUCKET_VARS})
        except OSError as e:
            print(f"host_doctor: cannot read --env-file: {type(e).__name__}", file=sys.stderr)
            return 2

    results = run_checks(skip=skip, data_dir=Path(args.data_dir), env=env, min_disk_gb=args.min_disk_gb,
                         container_image=args.container_image)
    print(format_results(results))
    report = build_report(results)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
