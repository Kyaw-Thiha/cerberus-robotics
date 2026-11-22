# Getting a machine that meets the host contract

Three routes, cheapest first. Every price below carries the date it was read and its source; a price with no source
is marked **unverified**. The requirements themselves are in `docs/host_contract.md`; check any new machine with:

```bash
python3 scripts/host/host_doctor.py --env-file .env
```

## 1. Your own workstation or a cluster node

The cheapest option if you already have an RT-core GPU and 32 GB of RAM.

```bash
# Ubuntu 22.04/24.04, a compliant driver, Docker + the NVIDIA Container Toolkit
python3 scripts/host/host_doctor.py                 # fix anything that FAILs
scripts/host/bootstrap_host.sh --data-dir ~/cerberus-data --env-file .env
scripts/host/run_session.sh --scene MV7J6NIKTKJZ2AABAAAAADQ8_usd --skip-gate
```

On a cluster, run the same three commands inside the allocation (usually a container with `--gpus all` and a bind mount
for `--data-dir`). If the cluster's driver is older than 580.95.05 the doctor fails; ask the cluster admins or use
route 2. **Do not** upgrade a shared machine's driver yourself.

Measured on this project's workstation (EndeavourOS, RTX 3050 Laptop, 15 GiB RAM, driver 615.78.08): the doctor fails
on OS, VRAM, driver, RAM and disk. That is the intended answer, not a bug: this laptop cannot run the stack.

## 2. A cloud VM (AWS worked example)

Any provider with an RT-core GPU works; the commands below are the AWS shape. **Prices are unverified**: AWS's pricing
page is rendered dynamically, so check it on the day. GPU instance families that satisfy the contract:

- `g6.2xlarge` (NVIDIA L4, 24 GB, 32 GB RAM) — the usual choice;
- `g5.2xlarge` (NVIDIA A10G, 24 GB, 32 GB RAM);
- `g6e.xlarge` (L40S, 48 GB) for larger scenes or more cameras.

```bash
# 1. An Ubuntu 24.04 AMI with a GPU (the Deep Learning AMI ships a recent driver; verify it!)
aws ec2 run-instances \
  --image-id <ubuntu-24.04-gpu-ami> --instance-type g6.2xlarge \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=250,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cerberus}]' \
  --key-name <your-key>

# 2. Open SSH to your address only, then connect
aws ec2 authorize-security-group-ingress --group-id <sg> --protocol tcp --port 22 --cidr <your-ip>/32
ssh -i <your-key>.pem ubuntu@<public-ip>

# 3. On the instance: driver first, then the repo
nvidia-smi                                  # driver version must pass the allowlist
sudo apt-get update && sudo apt-get install -y docker.io
sudo usermod -aG docker "$USER" && newgrp docker
git clone https://github.com/Kyaw-Thiha/cerberus-robotics.git && cd cerberus-robotics
python3 scripts/host/host_doctor.py         # must be all PASS (bucket needs your .env)

# 4. Work, then STOP the instance (stopped instances usually keep the disk but not the GPU billing)
aws ec2 stop-instances --instance-ids <id>
```

The Deep Learning AMI's driver is not guaranteed to be in the allowlist — run the doctor and, if it fails, either pick
a newer AMI or install 580.95.05+ from NVIDIA's `.run` installer (a full VM with sudo can do this; a container cannot).

## 3. Brev

Brev is the route this repository's cloud sessions used. The CLI is `brev` (v0.6.337 at the time of writing).

### What the catalog offered on 2026-11-22

Command (works without being logged in; it reads the public catalog):

```bash
brev search gpu --min-vram 24 --min-ram 32 --min-disk 200 --json
```

127 types matched the filters. After removing GPUs without RT cores (A100/H100/B200 and similar), requiring at least
24 GB VRAM, 32 GB RAM and a 200 GB disk, and capping the price at 1.50 USD/h, six remained, cheapest first. **Prices are
the catalog's `price_per_hour` on 2026-11-22**, not a quote; Brev's catalog changes daily.

| USD/h | type | GPU | VRAM | RAM | disk | boot | stoppable |
|---|---|---|---|---|---|---|---|
| 0.828 | `gpu_1x_rtx6000` | RTX 6000 | 24 GB | 46 GB | 512 GB | 600 s | no |
| 1.056 | `massedcompute_L40` | L40 | 48 GB | 72 GB | 625 GB | 330 s | no |
| 1.056 | `massedcompute_L40S` | L40S | 48 GB | 72 GB | 625 GB | 390 s | no |
| 1.140 | `scaleway_L4` | L4 | 24 GB | 48 GB | 500 GB | 390 s | no |
| 1.164 | `massedcompute_RTX6000Ada` | RTX 6000 Ada | 48 GB | 80 GB | 350 GB | 390 s | no |
| 1.200 | `hyperstack_L40` | L40 | 48 GB | 58 GB | 825 GB | 330 s | no |

Ranking: price first, then boot time. The catalog does **not** report the driver version, so the first two steps on any
new instance are `nvidia-smi` and `host_doctor`; a non-compliant driver means replacing the instance rather than
fighting it (the H4 session lost hours to exactly that).

`brev create --dry-run` was verified against the top choice:

```bash
brev create cerberus-t0-dry --type gpu_1x_rtx6000 --min-disk 200 --dry-run
# -> gpu_1x_rtx6000
```

### The wrappers in this repo (recommended)

`scripts/host/brev_up.sh` enforces the caps before anything is created: at most **1.50 USD/h**, at most **3.5 h** per
instance, at most **4 creations** per run, and **one live instance** at a time. It arms a systemd user timer that
stops and deletes the instance even if the agent dies, and refuses to create anything if the timer cannot be armed.

```bash
scripts/host/brev_up.sh --name cerberus-t0 --type gpu_1x_rtx6000 --price 0.828
scripts/host/brev_exec.sh cerberus-t0 'nvidia-smi'
scripts/host/brev_copy.sh ./scripts cerberus-t0:/workspace/cerberus/scripts
scripts/host/brev_status.sh                # state, spend estimate, armed timers
scripts/host/brev_down.sh cerberus-t0      # stop, delete, cancel the timer
```

Only instances this run created can be touched: `brev_exec`, `brev_copy` and `brev_down` refuse any other name, so the
owner's other workspaces are safe.

### Raw CLI, if you prefer it

```bash
brev login                                  # interactive; the one step that cannot be automated here
brev create cerberus-t0 --type gpu_1x_rtx6000 --detached --min-disk 200
brev ls                                     # pipes to instance names when not a tty
brev exec cerberus-t0 'nvidia-smi'
brev copy ./local cerberus-t0:/workspace/cerberus/
brev stop cerberus-t0 && brev delete cerberus-t0
```

**Unverified, because no account was logged in during this run:** `brev login`, `brev create` (without `--dry-run`),
`brev start`, `brev stop`, `brev delete`, `brev exec` and `brev copy` against a real instance. Their flag names come from
`brev <cmd> --help` (v0.6.337) and their behaviour is covered by mock tests
(`scripts/host/tests/test_brev_wrappers.py`), but no live call was made. `brev search gpu --json` and
`brev create --dry-run` were run and worked.

### If SSH to an instance times out at "banner exchange"

Symptom: `brev exec`, `brev shell` and plain `ssh <name>` all print `Connection timed out during banner exchange`; `brev ls`
shows the instance `RUNNING` with shell `READY` and health `HEALTHY`. The relay accepts TCP (ssh's `-vvv` says
`Connection established`) and never sends a banner.

What this machine showed on 2026-11-22 (CLI v0.6.337, OpenSSH 10.6, cloudflared 2024.10.0): the generated
`~/.brev/ssh_config` entry had **no `ProxyCommand`**, and no cloudflared process or brev service was running.
`brev register` installs the tunnel but requires sudo (`register.sudoWriteFile` in the CLI); after running it the config
was unchanged and no tunnel appeared. `brev refresh`/`brev enable-ssh`, clearing `ControlMaster`, `brev exec --host`,
`brev port-forward`, and a different instance/provider all failed the same way.

So: check `grep ProxyCommand ~/.brev/ssh_config` and `pgrep -a cloudflared` first. If both are empty, this is the
client-side tunnel, not the instance — do not spend money creating another instance. Full evidence, timeline and a
ready-to-send support message: `docs/reports/brev_ssh_issue.md`.

## On the instance

```bash
git clone https://github.com/Kyaw-Thiha/cerberus-robotics.git cerberus && cd cerberus
python3 scripts/host/host_doctor.py --env-file .env --json doctor.json
scripts/host/bootstrap_host.sh --data-dir /workspace/cerberus-data --env-file .env
scripts/host/run_session.sh --scene MV7J6NIKTKJZ2AABAAAAADQ8_usd --out-dir /workspace/cerberus-results/s1
```

Put the credentials file on the instance over `brev copy`, `chmod 600` it, and delete it before deleting the instance
(see the report for the rotation advice that follows any token copy).

## Cost control

- The wrappers refuse to exceed 1.50 USD/h, 3.5 h, 4 creations and one live instance.
- The kill timer is armed **before** the instance exists.
- `brev_status.sh` prints a spend estimate from the local ledger (`~/cerberus-results/brev/ledger.jsonl`), computed as
  price x hours alive for every create that has no matching delete.
- Check `brev ls` before you go to sleep; a stopped instance can still bill for its disk on some providers.
