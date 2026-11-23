# First scene session: tier reached, the cloud attempt and why it stopped

Date: 2026-11-21 22:00 to 2026-11-22 (local, unattended), with a second cloud pass after the owner signed in to Brev.
Scope: the overnight handoff "Host Contract, Brev Setup and First Real Scene Render".

## Tier reached: **T0** (local setup done); T1-T3 not reached

| tier | meaning | reached |
|---|---|---|
| T0 | provider-neutral local setup, tested, committed, pushed | **yes** |
| T1 | a compliant cloud machine, Kit starts, stand test passes | **no** — no instance ever became reachable |
| T2 | RTX clip of the Go2 following a scripted path in a real GRScenes home | **no** |
| T3 | navigation eval (N=10) with success and failure clips | **no** |

The blocker is **not** the driver, the image, the scene or our code: it is that no Brev instance exposed a working SSH
shell. Four instances on three providers, one stuck before boot and three fully "READY", none of them answered an SSH
session.

## The cloud attempt, in order

| # | type / provider | USD/h | created | outcome | wall |
|---|---|---|---|---|---|
| 1 | `gpu_1x_rtx6000` (RTX 6000, 24 GB) | 0.828 | 08:02Z | stuck `DEPLOYING`/`BUILDING` for 38 min; `brev reset` unsupported ("Not Found"); deleted | 37m |
| 2 | `massedcompute_L40` (L40, 48 GB) | 1.056 | 08:47Z | `RUNNING`, build `COMPLETED`, shell `READY`, health `HEALTHY`; SSH never answered (banner timeout); deleted | 17m |
| 3 | `scaleway_L4` (L4, 24 GB) | 1.140 | 09:05Z | same as #2 after 26 min of retries; deleted | 26m |
| 4 | `hyperstack_L40` (L40, 48 GB) | 1.200 | 09:31Z | same as #2; deleted | 9m |

All four stayed under the 1.50 USD/h cap, each had a kill timer armed **before** creation, and every instance was
deleted within the 3.5 h window. Creation 4 of 4 was the last one the rules allow.

### The exact SSH diagnosis

- `brev ls --json` for #2-#4: `status RUNNING`, `build_status COMPLETED`, `shell_status READY`, `health_status HEALTHY`.
- `brev exec` and `brev shell`: `Connection timed out during banner exchange`.
- Raw TCP to Brev's relay (`global.prd.ga.run.brev.nvidia.com` -> 75.2.122.140 / 99.83.144.240, port 12649) **connects**;
  `ssh -v` reaches "Connection established" and then waits for a banner that never comes. So the relay accepts the
  connection and has no live shell behind it.
- Direct host ports are closed (`163.172.163.237:22` and `:12649` timed out for #3).
- Tried and failed: `brev refresh` (several times), `brev enable-ssh`, clearing `ControlMaster` sockets and
  `ssh -o ControlMaster=no -o ControlPath=none`, waiting 5-10 minutes after "READY", `brev exec --host`.
- `brev healthcheck` reports `Healthy!`; local HTTPS egress works (200 from an external site), so neither our network
  nor Brev's control plane is down: the data path from the relay to each instance's shell is.

This is the same error class on three different providers (#2, #3, #4), which is the handoff's stop condition for cloud
work. It was reached at #4, so there was no retry left to spend.

## Hosts tried, launches, assertions, metrics

No Kit launch was attempted anywhere, because no shell was reachable: launches ok/total, the stand assertions, the
navigation metrics and the clips are all absent, and no physics number in this repository changed. The H4 numbers
remain the latest, and the kinematic upper bound (20/20, SPL 1.0, synthetic flat) is still only an upper bound.

## What was built (the T0 deliverable)

| piece | what it does | evidence |
|---|---|---|
| `docs/host_contract.md` | NVIDIA's requirements with sources, marked verified-by-us or documentation-only; the driver rule and its evidence | sources read 2026-11-22 |
| `scripts/host/host_doctor.py` | one PASS/FAIL per contract item with an exact fix, plus a JSON report; secret values never printed | 48 unit tests; run on this laptop (fails 6 of 10) and against the real bucket (passes) |
| `scripts/host/{bootstrap_host,run_session}.sh` | idempotent image/scene/cache setup; the stand/demo/smoke/nav/replay plan judged by artifacts, `--shutdown-after` refused outside a VM | dry runs; 12 shell tests with stubbed binaries |
| `scripts/host/run_{stack_check,scene_admission,scene_eval,slam_eval}.sh` | the four archived entry points, re-created provider-neutral | dry runs; a test asserts no provider-specific code remains |
| `scripts/host/brev_{up,status,exec,copy,down}.sh` | price/hours/creation/live caps, kill timer before create, ownership guard, spend ledger | 20 mock tests plus live use on all four instances |
| `scripts/host/` move | driver gate, launch probe, `kit_retry`, `kit_logs`, `launch_gate`, `job_attempt_log`, `run_resilient`, hygiene, allowlist | 1371 tests pass (baseline 1285); lint clean |
| `docs/dev_machine.md` | workstation, AWS worked example, Brev with prices dated 2026-11-22 | catalog JSON; unverified items marked |

## Bugs found and fixed

1. `host_doctor` probed the GPU when every GPU check was skipped (found by a test with a runner that raises).
2. `brev_live_count` never read its input: it piped JSON into `python3 - <<'PY'`, where the heredoc replaces stdin, so
   every creation would have been refused with "unknown live count". The JSON now passes as an argument.
3. `date -d '+3.5 hours'` is not portable, so the kill deadline could be empty; hours are converted to seconds and an
   empty deadline refuses creation.
4. `run_session.sh` did not parse `--internal-stage`, so every stage child would have exited 2.
5. `run_slam_eval.sh`'s preflight read `$2`/`$3` from a function called with one argument (unbound under `set -u`).
6. The driver gate hard-coded RunPod termination and a pod-only guard; it now returns a verdict plus an optional
   `--on-reject-cmd` hook.
7. The live-instance guard only knew `{"instances": ...}`, but the real API answers `{"workspaces": ...}`, and counted
   `DELETING` as live. Both fixed before the cloud pass, which is why the creations were allowed at all.
8. Process bug: a doc edit script wrote README text into `docs/project_structure.md`; caught by the diff, restored from
   HEAD, redone. No commit contains the damaged file.

## What still fails, and the exact next step

**What fails:** Brev's SSH data path on this account/network — the relay accepts TCP but never carries a shell, on
three providers, while the API reports the instances healthy and ready. Until a shell answers, no tier above T0 can be
attempted, on any GPU.

**Next step, in order:**

1. From the same machine, check Brev's own reachability outside this repo: `brev create test --type scaleway_L4` then
   `brev shell test`. If that also times out, it is account- or network-level: open a Brev support ticket with the
   timestamps above and the `ssh -v` line "Connection established" followed by "banner exchange" timeout, and ask
   whether the org's instances register with `global.prd.ga.run.brev.nvidia.com`.
2. Try the same command from a different network (a phone hotspot is enough) to tell a local-egress problem from a
   Brev-side one.
3. With a working shell, the run is ready: `host_doctor.py`, `bootstrap_host.sh --data-dir /workspace/cerberus-data`,
   `run_session.sh --scene MV7J6NIKTKJZ2AABAAAAADQ8_usd`. The first stage writes `assertions.json`; if it fails for a
   code reason the log names it and the 60-minute debug box applies.
4. If Brev cannot be fixed and a non-Brev VM is allowed later, the same three commands work on any host that passes
   the contract; only `brev_*.sh` is Brev-specific.

## Decisions from `outputs/overnight_log.md` that mattered

- Skip the cloud phases while Brev was logged out (handoff rule); do not run `brev login` unattended.
- After the owner signed in: create with the cheapest compliant offering, keep the kill timer armed before creation,
  and replace rather than debug a bad host.
- #1: replace after 38 min of no boot (`brev reset` unsupported) instead of waiting out the 3.5 h.
- #2-#4: treat "READY but no SSH" as a host-level failure after 3 retries plus `refresh`/`enable-ssh`, and move to a
  provider with a different network path rather than re-running the same one.
- Stop at #4: three consecutive same-class failures on different instances is the handoff's stop condition.
- Do not exceed the 1.50 USD/h, 3.5 h, 4-creation or one-live limits at any point; all four were honoured.

## 2026-11-22 14:00-15:00: R2 command channel (no SSH) - still no T1

Because Brev's SSH relay is broken, a bucket-based command channel was built: `scripts/host/r2_channel.sh` (instance
loop: polls `jobs/<job>/cmd/`, runs each command with bash under a timeout, uploads stdout/stderr/rc, heartbeat every
30 s) and `scripts/host/r2_channel.py` (bundle the startup script, `exec`, `put`/`get`, `wait`). Nine tests drive the
real loop against a fake bucket, plus an end-to-end local bootstrap test (startup script -> heartbeat -> exec with a
non-zero exit status).

Three live attempts, all with the channel bundled into the startup script:

| # | type | USD/h | outcome |
|---|---|---|---|
| 1 | m8g.medium | 0.0385 | RUNNING for 84 min, no heartbeat, no objects; deleted |
| 2 | m8g.medium | 0.0385 | same after fixing the script's `unzip` dependency and its systemd-only start; deleted |
| 3 | massedcompute L40 | 1.056 | same with the minimal bootstrap (fetches the channel from R2, nohup fallback); deleted |

No `state/startup.log` was ever uploaded, so the startup script either did not run on these instances or ran before
the network was usable. An earlier minimal canary on the same account did run and reached R2 in 33 s, so the mechanism
exists but is not reliable here. The bootstrap now retries the rclone install and the channel fetch (~3 min each) and
logs every step; that change is tested locally but has not been confirmed on a live instance. `brev reset` is not
supported for these types, so a failed startup script cannot be retried on the same instance.

Tier remains **T0**. The GPU instance (L40, 24 GB, RT cores, driver unknown) never became usable.

Spend for this attempt: about 0.23 USD (70 min + 13 min CPU at 0.0385/h, 15 min L40 at 1.056/h); 3 of 3 creations used.

## Spend

| item | amount |
|---|---|
| instances created | 4 of 4 |
| instance wall clock | 90 min (37 + 17 + 26 + 9) |
| spend estimate (price x time, from the ledger) | **~1.50 USD of 12.00** |
| kill timers armed / fired | 4 armed, 0 needed (all deleted normally); 0 timers left |
| R2 objects written | 0 |

The final billing could differ slightly if the provider bills boot time differently; the ledger estimate is the upper
bound of what the four instances could have cost at their listed prices.

## 2026-11-23: Hyperstack plain-SSH session (T1 reached)

The Brev relay is gone; this pass used a plain `ubuntu@<floating-ip>` VM on Hyperstack.

| item | value |
|---|---|
| host | Hyperstack CANADA-1, `cerberus-scene1`, id 1176254, L40 48 GB, 825 GB root |
| price | 1.0866 USD/h all-in (GPU + disk + public IP), read from `GET /pricebook` |
| driver | 580.196.07, then 580.95.05, then 595.58.03; only 595.58.03 survives camera runs |
| canary | A6000 unavailable (stock race), L40 24 GB: SSH yes, driver 570.195.03, Docker + nvidia runtime OK, cost 0.06 USD |
| image | `Ubuntu Server 24.04 LTS R570 CUDA 12.8 with Docker` (the R595 image is B300-only) |
| bootstrap | 35 GB image, 25 GB GRScenes home, GA checkpoints, admission manifest; 22 min wall |
| host_doctor | PASS on the host (OS, GPU, driver, RAM, Docker, container GPU, disk) |

### Stand test (T1): PASS

`run_session.sh --stages stand` on scene `MV7J6NIKTKJZ2AABAAAAADQ8_usd`, 1 env, 200 zero-command steps.
`assertions.json`: **8/8 PASS** — joint order, 235-D term layout, command slice, finite stand, stand length,
no fall, robot stands (min base-minus-foot 0.30 m).

### Bugs found and fixed (all in our code, none tuned)

1. `scan_proxy.usd` was built on Kit's thread after `AppLauncher`; the 13 s `pxr` pass blocked the app loop and
   Kit exited 0 at ~50 s. It is now built in a child interpreter **before** Kit.
2. `CERBERUS_SCENES_DIR` pointed at the fetch destination, not the discovery root; admission was never found.
   `session_lib` now exports `<data>/scenes` and the manifest is fetched to `<root>/grscenes/admission.json`.
3. `env.get_observations()` is a TensorDict; the code recorded `{policy: [...]}` and `obs_dim=1`. The policy
   tensor is now taken from `obs["policy"]`.
4. The contract's joint order was per-leg; Isaac Lab preserves the asset order (hips, thighs, calves). Confirmed
   live on this run and corrected in the contract and `platforms.yaml`.
5. The observation was the cached reset value after a command write; it is recomputed via
   `observation_manager.compute()`. `command_slice` now passes.
6. Kit's native layer exits the process when the pure-Python assertion report runs on the app thread. The report is
   built in a child process from `assert_inputs.json`.
7. `kit_retry` treated the missing `Simulation App Startup Complete` marker as a failed launch and waited 300 s on
   each loader crash. It now accepts a configurable start marker (`[scene_env] build timings`) and fails fast on
   `libcarb.tasking`/`make_fcontext`.

### Host stability

Kit starts about 1 launch in 10 on this L40 (probe: 6/63). With the build marker and 30 retries a scene run gets
through; camera runs are the fragile part (`carb.windowing-glfw` fails headless). The demo clip was produced with the
RTX viewport at 960x540 (downscale 2); a 1080p re-run is in progress.

Cost so far: about 5 USD of 18. The VM kill timer is armed at creation + 10 h 15 m.

### Demo clip and navigation: what was actually produced

| artifact | result |
|---|---|
| `demo_scene1.mp4` | **1920x1080, 50 fps, 90.0 s**, 6.9 MB, RTX viewport, real scene, follow camera, scripted path through the slew limiter. Measured with `ffprobe` (h264). Local `cerberus-results/2026-11-23/`, R2 `migration/first_scene/20261123/` |
| `demo_scene1.json` | clip properties (steps 4500, 90 s, 1080p50, seed 0) and the kinematic expected end pose; `measured: false` — Kit exited natively before the sim could write the pose/fall record |
| stand `assertions.json` | 8/8 PASS (kept locally and on R2) |
| smoke / nav | **blocked**: every attempt that reached the environment died at the first pure-Python block after the sim loop (stand report, `read_poses`, nav sampling). The nav run sampled **5 pairs from 12 attempts** (2 unreachable, 5 outside 3-10 m) and wrote `build_timings.json`, but `summary.json` was never reached. No navigation success number exists for this session. |
| replay clips | not attempted after the camera launches failed 30/30; a 960x540 demo had rendered earlier, so the 1080p clip is the second valid render |

The host's failure mode is not our scene code: `nvidia-smi` is clean, `host_doctor` passes, and a 70 s pump loop kept
Kit alive, but the native layer calls `exit(0)` at the first Python boundary after a long sim block and the loader
crashes about 9 times in 10. The seven code fixes above are real and tested; the remaining blocker is host/GL
stability.

### Spend and wrap-up

- VM L40 825 GB at 1.0866 USD/h from 21:25Z to about 03:30Z: ~5.6 USD plus the canary 0.06 USD, about **6 USD of 18**.
- The R2 token file and the Hyperstack key were removed from the VM; the VM was deleted through the API and verified
  gone; the 10 h timer and the 3 h timer were cancelled; the sleep inhibitor was stopped.
- **Rotate the R2 token and the Hyperstack API key**: both were present on the VM for the run.

### Navigation: real-grid planner result (kinematic, not physics)

Because the physics navigation run died before its first episode, the A* + pure pursuit stack was run on the real
scene-1 grid (`migration/h4_nav/20261110T0857Z/occupancy/MV7J…ADQ8_usd/occupancy.npz`) with the kinematic backend,
N=10, seed 0:

| metric | value |
|---|---|
| pairs | 10 from 18 attempts (8 outside the 3–10 m window) |
| success | **10/10** |
| SPL | 1.0 (capped; episodes stop at the 0.5 m tolerance) |
| collisions | 0 |
| mean path / time | 5.62 m / 8.66 s |

This is the **ceiling**, not a policy result: the unicycle has no dynamics, no slip and no fall. It confirms the
planner and controller still work on the real grid with the current code; the physics number is still absent.
Artifacts: `~/cerberus-results/2026-11-23/kinematic_n10/` (per-pair JSON, PNG, CSV, `summary.json`).

### Second host (third and last creation): same boundary

A fresh L40 (`cerberus-scene1b`, id 1176656, 595.58.03, image + scene + checkpoints) reproduced the failure exactly:

- `run_session --stages smoke`: stand completed, `assert_inputs.json` written, exit 0 at 214 s before the scripted block.
- `--scripted-only` (build → scripted loop → metrics, no stand): exit 0 after the policy load, before the first
  `reset_envs_to`; no `clips/*.json`.

So the native exit is not one bad node. Every remaining stage needs a Python call after a long sim block, and Kit
terminates there. Smoke metrics and replay are therefore **BLOCKED on this image/host class**, not on our code.
The in-loop metrics writer is implemented and unit-tested; it will produce the JSON on a host that survives the
scripted block.

Final spend: about **5.95 USD of 18**. No VMs, no timers, no secret files, no sleep inhibitor.
