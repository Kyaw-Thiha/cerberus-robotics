# Brev SSH: banner-exchange timeout (issue report, unsent)

Status: **Brev SSH does not work from this machine.** Every path (CLI and plain `ssh`) reaches Brev's relay and then
times out waiting for the SSH banner. Instances are healthy and report READY. The tunnel client the CLI expects is not
running on this laptop, and `brev register` (the command that installs it) needs sudo and, having been run, left no
tunnel process or config change.

Reported by: unattended/attended sessions of this repo, 2026-11-22. Do not send the message below without reading the
timeline.

## Environment

| item | value |
|---|---|
| CLI | `brev` v0.6.337 (`/home/kevin/.local/bin/brev`) |
| Org | `k.thiha10.mail-0f8b9b-h2xd` |
| SSH client | OpenSSH_10.6p1, OpenSSL 3.6.5 |
| cloudflared | `/usr/bin/cloudflared` and `~/.brev/cloudflared`, both version 2024.10.0 |
| Network | Wi-Fi `wlan0` on CGNAT (`100.102.58.252`), Tailscale `tailscale0` up; external HTTPS 200 |
| OS | EndeavourOS (Arch), kernel 6.18 |

## Timeline (local time, EST)

| time | event |
|---|---|
| 11:44:05 | `cerberus-ssh1` created (`m8g.medium`, AWS EC2, $0.0385/h); 40-minute delete timer armed first and verified |
| 11:48:33 | `RUNNING`, build `COMPLETED`, shell `READY`, health `HEALTHY` |
| 11:48-11:52 | `brev exec` -> `Connection timed out during banner exchange`; `ssh -vvv` connects to `99.83.144.240:27355`, prints `Connection established`, then times out |
| 11:53 | direct host `13.206.202.190` ports 22 and 27355: both closed |
| 11:54 | `cloudflared access ssh --hostname global.prd.ga.run.brev.nvidia.com` -> `websocket: bad handshake` |
| 11:55 | `brev port-forward ... -p 18888:8888` -> no listener, curl `000` |
| 11:56 | binary inspection: `ProxyCommand {{ .ProxyCommand }}`, `pkg/store/cloudflared.go`, `register.sudoWriteFile` |
| 12:05 | owner ran `brev register --name kevin-workstation --org k.thiha10.mail-0f8b9b-h2xd`; `~/.brev/ssh_config` and `credentials.json` rewritten at 12:05 |
| 12:06 | `brev refresh`; config still has **no ProxyCommand**; entry unchanged (`global.prd...:27355`, user `ubuntu`) |
| 12:07-12:12 | `brev exec`, `ssh` direct, `brev shell`: same banner timeout |
| 12:15 | no `cloudflared` process, no brev systemd unit, no tunnel |
| 12:16 | `brev port-forward` again: same timeout; `brev shell` spawns no cloudflared |
| 12:17 | instance deleted; timer cancelled |

Earlier the same day, four GPU instances on three providers (massedcompute L40, scaleway L4, hyperstack L40, plus an
RTX 6000) showed the identical failure; see `docs/reports/first_scene_session.md`.

## Exact outputs (trimmed to the decisive lines)

```
$ brev exec cerberus-ssh1 'echo ok'
Connection timed out during banner exchange

$ ssh -vvv -o ControlMaster=no -o ConnectTimeout=30 cerberus-ssh1 true
debug1: Connecting to global.prd.ga.run.brev.nvidia.com [99.83.144.240] port 27355.
debug1: Connection established.
debug1: Local version string SSH-2.0-OpenSSH_10.6
Connection timed out during banner exchange

$ cloudflared access ssh --hostname global.prd.ga.run.brev.nvidia.com
websocket: bad handshake

$ brev healthcheck
Healthy!
```

`~/.brev/ssh_config` for the instance (unchanged after register):

```
Host cerberus-ssh1
  Hostname global.prd.ga.run.brev.nvidia.com
  IdentityFile "/home/kevin/.brev/brev.pem"
  User ubuntu
  Port 27355
  ...
```

No `ProxyCommand` line. `brev.pem` is a valid RSA key (`ssh-keygen -y` reads it), though `ssh -vvv` logged
`no pubkey loaded from /home/kevin/.brev/brev.pem` — secondary, since the banner never arrives.

## What was tried and did not help

- `brev refresh` (repeated), `brev enable-ssh`, clearing `ControlMaster`/`ControlPath`, `ssh -o ControlMaster=no`.
- Waiting several minutes after `READY` (the relay port accepts TCP immediately, so this is not a slow start).
- `brev exec --host` (host route also closed/timeout).
- `brev shell`, `brev port-forward` — both use the same failing SSH path.
- `brev register` (requires sudo; run by the owner at 12:05) — no ProxyCommand, no tunnel process, no service.
- A different instance/provider (earlier run: three providers, same result).

## Working hypothesis

The CLI generates an SSH config that points at Brev's relay, but nothing on this laptop establishes the client half of
the Brev tunnel (the binary contains a `ProxyCommand` template and a cloudflared store, yet this version neither writes
the line nor starts cloudflared on demand). The relay accepts TCP and has no live backend for this org/device, so no
banner is ever sent. It is not the instance (fresh CPU instance, healthy, same failure), not the credentials file
(private key valid), and not general connectivity (HTTPS fine, `brev healthcheck` healthy).

## Support message (ready to send to brev-support@nvidia.com)

```text
Subject: SSH to any Brev instance times out at "banner exchange" (org k.thiha10.mail-0f8b9b-h2xd, CLI v0.6.337)

Hello,

From one Linux workstation (EndeavourOS, OpenSSH 10.6p1, brev v0.6.337) every SSH session to a Brev instance fails
after TCP connects:

  ssh -vvv <instance>
  debug1: Connecting to global.prd.ga.run.brev.nvidia.com [99.83.144.240] port 27355.
  debug1: Connection established.
  Connection timed out during banner exchange

brev exec / brev shell / brev port-forward fail the same way; brev ls reports the instance RUNNING with build COMPLETED,
shell READY, health HEALTHY; brev healthcheck reports Healthy.

The generated ~/.brev/ssh_config entry has no ProxyCommand line. cloudflared 2024.10.0 is present at /usr/bin and
~/.brev. I ran `brev register --name kevin-workstation --org k.thiha10.mail-0f8b9b-h2xd` (it required sudo and
completed); afterwards the config is unchanged and no cloudflared process or brev systemd unit is running. `brev shell`
does not spawn cloudflared either. `cloudflared access ssh --hostname global.prd.ga.run.brev.nvidia.com` returns
"websocket: bad handshake".

Reproduced on four instances across three providers (AWS m8g.medium at 2026-11-22 11:44-12:17 EST; and L40/L4-class
instances on massedcompute, scaleway and hyperstack earlier the same day).

Questions:
1. For this org, should the SSH config contain a cloudflared ProxyCommand (Brev tunnel), and what starts that tunnel?
2. Does the org need a device registration or tunnel enabled server-side before instances accept SSH?
3. Is there a supported non-sudo setup for a workstation that cannot run `brev register` as root?

Happy to provide full logs and timestamps.

Thanks,
Kevin
```

## If SSH starts working

The run is ready: `scripts/host/host_doctor.py`, `scripts/host/bootstrap_host.sh --data-dir /workspace/cerberus-data`,
`scripts/host/run_session.sh --scene MV7J6NIKTKJZ2AABAAAAADQ8_usd`. Nothing in the repo assumes a particular Brev
transport; only `scripts/host/brev_*.sh` is Brev-specific, and those call `brev exec`/`brev copy`.
