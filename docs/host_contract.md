# Host contract

What a machine must provide to run this stack, where each requirement comes from, and which parts our own runs have
actually confirmed. `scripts/host/host_doctor.py` checks this list item by item; `docs/dev_machine.md` shows three
ways to get such a machine.

Status of the sources: the NVIDIA pages below were read on **2026-11-22**. Versions move; re-check them before buying
hardware. "Verified by us" means a run recorded in this repository, not a reading of documentation.

## Requirements

| Item | Requirement | Source | Verified by us |
|---|---|---|---|
| OS | Ubuntu 22.04 or 24.04, x86_64 (Isaac Sim's container is Linux-only). NVIDIA marks **22.04 deprecated** and recommends 24.04. | [Isaac Sim system requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) | No: our runs used the `cerberus-remote-v6` image on Ubuntu-based cloud hosts, whose exact base release was not recorded. |
| GPU | An **RT-core** GPU. "GPUs without RT Cores (A100, H100) are not supported." The documented minimum is a GeForce RTX 4080 (16 GB); the recommended tier is an RTX PRO 6000 Blackwell (48 GB). | same page | Partly: the H1-H4 sessions ran on RTX A5000/A6000, RTX 3090/4090 and L4-class hosts (RT cores present). No A100/H100 run was attempted, because NVIDIA excludes them. |
| VRAM | **16 GB minimum** per GPU; below 16 GB "may be insufficient" for a complex scene rendering more than 16 MP per frame. | same page | Yes, negatively: a 4 GB laptop GPU fails `host_doctor` and has never run the stack. Our scene runs used 24-48 GB cards with several GB of headroom for cameras. |
| System RAM | **32 GB minimum**, 64 GB recommended. | same page | Yes: the image plus Kit, PhysX and a scene build has been run on 32-64 GB instances. A 15 GiB laptop fails the doctor. |
| Driver | NVIDIA's page lists **Linux 595.58.03** as the tested driver for this release. Our own rule (below) is stricter: **580.95.05 or newer on the 580 branch, or exactly 595.58.03**. | NVIDIA page; our rule: `scripts/host/driver_allowlist.json` + the H4 evidence in `docs/h4_report.md` | Yes, for the rejects: driver 595.91.07 crashed Isaac Sim 6 at startup (SIGSEGV in `libcarb.tasking`, exit 139) on three hosts in the H4 session. 580.95.05 and 595.58.03 passed the gate and the launch probe. |
| Docker | Docker Engine, plus the **NVIDIA Container Toolkit** and its `nvidia` runtime registered with Docker. | [Isaac Sim container installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html) | Yes: every run in these reports used the container; a container `nvidia-smi` is how we confirm the toolkit. |
| Disk | NVIDIA's minimum is 50 GB (500 GB recommended). For **this** repo budget **200 GB**: the image is ~35 GB, the Kit and shader caches grow by several GB, one GRScenes home is ~25 GB, and results (videos, bags) are kept. | NVIDIA page for the base; our figure is measured from the image and scene sizes in `docs/storage.md`, `docs/arena_integration_log.md` | Partly: the 35.4 GB image and ~25 GB home are measured; the 200 GB total is a budget, not a measurement. |
| Network | Outbound HTTPS; NVIDIA's online asset root is `omniverse-content-production.s3-us-west-2.amazonaws.com`. | NVIDIA page | Yes: without it Kit stalls fetching assets. |
| Object storage | **S3-compatible bucket** with the Cerberus layout (`docs/storage.md`), read for scenes and write for results. Not an NVIDIA requirement; it is how this repo moves scenes and clips. | this repo | Yes: R2 has been used for every scene and clip in these reports. Any S3-compatible service works for the layout, but the fetch scripts assume the R2 endpoint today. |
| CPU | 4 cores minimum, 8-16 recommended. | NVIDIA page | Yes: 8-12 vCPU instances were used. |
| Virtualisation | A **VM with sudo** if the driver must be changed; a container-only offering cannot install a driver. | this repo | Yes: the H4 session found hosts whose driver could not be changed from inside the container and had to be replaced. |

## The driver rule, and why it is stricter than NVIDIA's page

NVIDIA's page names one tested Linux driver (595.58.03). The 595 line, however, shipped several bad builds that crash
Isaac Sim 6 at start-up. We accept:

- **580 branch, 580.95.05 or newer** — the branch the previous Isaac Lab 3.0 documentation named, and the branch our
  successful sessions ran on; or
- **exactly 595.58.03** — NVIDIA's tested build, and the one our launch probe passed on.

Everything else is rejected before any download, including 595.71.05, 595.79, 595.84 and 595.91.07. The rules and the
evidence beside them are in `scripts/host/driver_allowlist.json`; `scripts/host/driver_gate.py` applies them and
`host_doctor` reports the same verdict.

## What a compliant machine is not

- **Not a laptop with a small GPU.** 4-8 GB of VRAM is below the documented minimum; the doctor will say so.
- **Not a CPU-only node.** Kit and PhysX need the GPU.
- **Not an A100/H100 rental.** Plenty of VRAM, no RT cores, explicitly unsupported. `host_doctor` fails these by name.
- **Not a container offering without a compliant driver.** You cannot install a kernel driver from inside it.

## Checks in one command

```bash
python3 scripts/host/host_doctor.py --env-file .env --json doctor.json
```

PASS/FAIL per item, an exact fix hint for each failure, and the JSON report. Exit status 1 means at least one FAIL.
