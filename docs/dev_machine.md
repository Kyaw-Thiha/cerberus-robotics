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

## 3. Hyperstack (plain SSH, no relay)

Hyperstack gives a normal `ubuntu@<floating-ip>` shell, which is why the cloud session moved here after the earlier relay approach
failed (the session-2/3 evidence is in the archive: tag `archive/session4-2026-11-24`, R2
`archive/2026-11-24/repo-tidy/`). The API base is `https://infrahub-api.nexgencloud.com/v1` and
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

Put the credentials file on the instance over `scp`, `chmod 600` it, and delete it before deleting the instance
(see the report for the rotation advice that follows any token copy).

## Cost control

- The Hyperstack wrappers refuse to exceed the run's caps (session 4: 1.50 USD/h, 11 h, 2 creations, one live VM).
- The kill timer is armed **before** the instance exists.
- `hyperstack_status.sh` prints a spend estimate from the local ledger, price x hours alive for every create without
  a delete. The kill timer is armed before the VM exists; the VM-side dead-man deletes it if the heartbeat goes stale.

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
