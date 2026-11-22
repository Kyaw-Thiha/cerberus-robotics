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
