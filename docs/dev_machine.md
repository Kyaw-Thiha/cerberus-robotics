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

## 3. Brev (legacy relay)

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

## 4. Hyperstack (plain SSH, no relay)

Hyperstack gives a normal `ubuntu@<floating-ip>` shell, which is why the cloud session moved here after the Brev relay
failed (see `docs/reports/first_scene_session.md`). The API base is `https://infrahub-api.nexgencloud.com/v1` and
authentication is the `api_key` header; the wrappers read `HYPERSTACK_API_KEY` from `.env` inside a subshell for each
call and never print it.

Prices below are the live `GET /pricebook` values read on **2026-11-22** (GPU + Cloud-SSD disk + public IP), not a
quote:

| flavor | GPU | VRAM | root disk | all-in USD/h |
|---|---|---|---|---|
| `n3-RTX-A6000x1` | RTX A6000 | 48 GB | 100 GB | 0.516 |
| `n3-L40x1-bigroot` | L40 | 48 GB | 825 GB | 1.087 |
| `n3-RTX-A4000x1` (NORWAY-1, no stock) | RTX A4000 | 16 GB | 100 GB | 0.166 |

`pick-image` chooses the Ubuntu image with the highest driver build in the region; on CANADA-1 that is
`Ubuntu Server 24.04 LTS R595 CUDA 13.2 with Docker`. The driver rule still applies: check `nvidia-smi` and run
`host_doctor`; if the image is on a rejected 595 build, install 580.95.05+ from NVIDIA's runfile (a full VM has sudo).

```bash
# one-time key (public half is uploaded; the private half never leaves the laptop)
ssh-keygen -t ed25519 -f ~/.ssh/cerberus_hyperstack -N '' -C cerberus-hyperstack

scripts/host/hyperstack_up.sh --name cerberus-canary --flavor n3-RTX-A6000x1 --hours 1
scripts/host/hyperstack_status.sh                  # state, spend, caps, armed timers
scripts/host/hyperstack_ssh.sh cerberus-canary 'nvidia-smi; docker info'
scripts/host/hyperstack_down.sh cerberus-canary    # DELETE, verify gone, cancel timer
```

The wrappers enforce the run's caps before creation: **1.20 USD/h**, **3 h per VM**, **3 creations**, **one live VM**,
**6.00 USD** and **5 h** total. A `systemd-run --user` timer is armed for creation + 3 h **before** the POST (keyed by
name) and re-armed on the real VM id straight after it, so a lost response cannot leave an untracked VM billing.
`hyperstack_kill.sh <id>` is the switch itself: DELETE, poll until gone, cancel the timers. `--check` is a harmless
GET. Down/kill/ssh refuse a VM this run did not create (ledger plus the `cerberus-` name prefix).

The SSH ingress rule is created with the VM: TCP 22, `remote_ip_prefix` set to the current public IP as `/32`
(`curl -s ifconfig.me`, or `--cidr`). On carrier-grade NAT the address can change mid-session; if SSH stops answering,
re-read the IP and re-create the rule. Use `--dry-run` to see the payload without spending anything.

## On the instance

```bash
git clone https://github.com/Kyaw-Thiha/cerberus-robotics.git cerberus && cd cerberus
python3 scripts/host/host_doctor.py --env-file .env --json doctor.json
scripts/host/bootstrap_host.sh --data-dir /workspace/cerberus-data --env-file .env
scripts/host/run_session.sh --scene MV7J6NIKTKJZ2AABAAAAADQ8_usd --out-dir /workspace/cerberus-results/s1
```

Put the credentials file on the instance over `brev copy`, `chmod 600` it, and delete it before deleting the instance
(see the report for the rotation advice that follows any token copy).

## Running without SSH: the R2 command channel

When a host gives no shell (see the Brev SSH issue), `scripts/host/r2_channel.*` turns a bucket prefix into a job
runner. The instance side is started by the Brev startup script; the laptop side writes commands and reads results.

```bash
# 1. put the channel where the bootstrap will fetch it, then build a startup script (mode 600, contains credentials)
python3 scripts/host/r2_channel.py put --job scene1 --src scripts/host/r2_channel.sh --key input/r2_channel.sh
python3 scripts/host/r2_channel.py bundle --job scene1 --out /tmp/scene1-start.sh

# 2. create the instance with it
brev create NAME --type TYPE --startup-script @/tmp/scene1-start.sh --detached

# 3. wait for the channel, then run commands; large files move R2 <-> instance, never through the laptop
python3 scripts/host/r2_channel.py wait --job scene1 --timeout 600
python3 scripts/host/r2_channel.py exec --job scene1 --cmd 'nvidia-smi; df -h' --timeout 120
python3 scripts/host/r2_channel.py put --job scene1 --src ./scene.tar --key input/scene.tar
python3 scripts/host/r2_channel.py get --job scene1 --key out/0007.out --dst ./out.txt
```

Layout: `jobs/<job>/cmd/<seq>_<timeout>.cmd`, `out/<seq>.{out,err,rc}`, `heartbeat.json` every 30 s,
`state/channel.json`, `state/startup.log`. The loop runs only commands under its own job prefix, with bash, in
`/tmp/cerberus/work/<job>`, under `timeout`; a long job should `nohup` itself and log to a file for a later command to
upload. Only the bucket-scoped token is used, passed to rclone as environment variables inside a 0600 file.

**Measured limitation (2026-11-22):** on this Brev account the startup script did not start the channel on three
attempts (two CPU, one GPU), although a minimal startup-script canary had run in an earlier session. No heartbeat and
no `state/startup.log` appeared, which points at the script running before the network is usable or not running at all
on those instances. The bootstrap now retries the rclone install and the channel fetch for up to ~3 minutes each and
uploads its log, but that has not yet been confirmed on a live instance. Treat the channel as implemented and tested
locally, not yet proven on Brev.

## Cost control

- The wrappers refuse to exceed 1.50 USD/h, 3.5 h, 4 creations and one live instance.
- The kill timer is armed **before** the instance exists.
- `brev_status.sh` prints a spend estimate from the local ledger (`~/cerberus-results/brev/ledger.jsonl`), computed as
  price x hours alive for every create that has no matching delete.
- Check `brev ls` before you go to sleep; a stopped instance can still bill for its disk on some providers.

## Known issues (2026-11-24)

- **Isaac Lab 3.0 pose terms.** `robot.data.root_quat_w` / `root_pos_w` report shape `(num_envs,)` (ProxyArray)
  although one entry holds 4/3 values. Slice per env (`[i]`) and stack; `data.root_quat_w[:, :]` raises
  "quaternion last dim must be 4".
- **Height scan source.** The locomotion height scanner must ray-cast `scan_proxy.usd`, one clean triangle mesh with
  ceilings/roofs removed. Against the raw house it hits the ceiling and every cell reads the same.
- **LiDAR source.** JEPLO's depth sensor uses the same proxy (`/World/scan_proxy`); it contains walls, floor and
  furniture, which is what the policy must see.
- **Arm locomotion checkpoint does not translate in the scene (2026-11-24).** Commands reach the observation and
  the height scan is sane, but `go2_arx5_locomotion_ga_v1_candidate.pt` produces large actions (mean |a| 1.87) with
  base velocity 0.015 m/s under a 0.7 m/s command. The legs-only `go2_locomotion_ga_v1_candidate.pt` tracks 0.41 m/s
  in the same scene and env. Do not tune the policy to hide this; use the Go2 baseline and report it.
- **Pure-pursuit singularity.** `wz = 2 v sin(alpha) / d` is zero at `alpha = pi`; without a turn-in-place branch
  the robot drives away in a large loop from a target directly behind (measured: 14 m away from the next goal).
- **SimulationApp.close() exits with code 0 and eats the traceback.** Wrap the run in try/except and
  `os._exit(3)` after printing, or failures look like clean runs.
- **Random libcarb crash at Kit startup.** Roughly one launch in five dies within ~10 s (dump file 0 bytes).
  Retry the same command; judge success by output files, not the exit code.
- **Driver install on the provider image.** The runfile aborts while Ubuntu's packaged driver is present, while
  `/usr/lib/nvidia/alternate-install-present` exists, or while the nvidia modules are loaded. Purge the packaged
  driver, remove that marker and the 570 leftovers, `rmmod` the modules, then install; keep `libnvidia-container`
  (removing it breaks Docker's GPU runtime). `scripts/host/install_driver.sh` does this.
- **R2 from the CLI.** Writes need the env-defined remote (`RCLONE_CONFIG_<name>_*` from `.env`) plus
  `NO_CHECK_BUCKET=true` and bucket-qualified paths. Bucket-level operations (versioning, ListBuckets) answer 403;
  that is expected and does not block object writes.
