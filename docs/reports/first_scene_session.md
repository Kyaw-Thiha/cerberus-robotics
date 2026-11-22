# First scene session: tier reached, and why the cloud half did not run

Date: 2026-11-21 22:00 to 2026-11-22 (local, unattended). Scope: the overnight handoff "Host Contract, Brev Setup and
First Real Scene Render".

## Tier reached: **T0** (local setup done), and only T0

| tier | meaning | reached |
|---|---|---|
| T0 | provider-neutral local setup, tested, committed | **yes** |
| T1 | a compliant cloud machine, Kit starts, stand test passes | **no** — no instance was ever created |
| T2 | RTX clip of the Go2 following a scripted path in a real GRScenes home | **no** |
| T3 | navigation eval (N=10) with success and failure clips | **no** |

**Why no instance: Brev was logged out and could not be logged in without a human.** `brev --version` works
(v0.6.337), but every authenticated call fails:

```
$ brev ls
WARN: malformed refresh token, logging out
You are currently logged out, would you like to log in? [Y/n]:
```

`~/.brev/credentials.json` holds a refresh token that the client rejects (`~/.brev/.invalid-refresh-token` is written
on each attempt), no `BREV_*` token is present in the environment, and `brev login` requires an interactive browser
flow. The handoff's own rule for this case is "if not logged in and no token in the environment, skip the cloud
phases", so Phases 2-4 were skipped, no instance was created, and nothing was spent.

## Hosts tried

None. Zero creations, zero live instances, **spend 0.00 USD of the 12 USD cap**. No kill timer was armed because no
instance existed; the timer mechanism itself is implemented and mock-tested in `scripts/host/brev_up.sh`
(`systemd-run --user --unit=cerberus-kill-<name> --on-calendar=... --timer-property=Persistent=yes`), including the
refusal to create anything when the timer cannot be armed or does not appear in `systemctl --user list-timers`.

Part of Phase 2 did run, because it needs no account: `brev search gpu --json` reads the public catalog. Filtered to
RT-core GPUs, >= 24 GB VRAM, >= 32 GB RAM, >= 200 GB disk and <= 1.50 USD/h, six types remained; the cheapest is
`gpu_1x_rtx6000` at 0.828 USD/h (24 GB, 46 GB RAM, 512 GB disk, 600 s boot). `brev create --dry-run` against it
resolved the type. The table with the date and every exclusion rule is in `docs/dev_machine.md`.

## Launches, assertions, metrics

None: no Kit launch was attempted, so launches ok/total, stand assertions, navigation metrics, clip locations and
the physics-versus-kinematic comparison are all empty. The last physics numbers remain the H4 report's, and nothing
here supersedes them. The kinematic upper bound is unchanged (20/20, SPL 1.0, synthetic flat), and it is still an
upper bound, not a scene result.

## What was built (the T0 deliverable)

| piece | what it does | evidence |
|---|---|---|
| `docs/host_contract.md` | NVIDIA's requirements with sources, marked verified-by-us or documentation-only; the driver rule and its evidence | sources read 2026-11-22 |
| `scripts/host/host_doctor.py` | one PASS/FAIL per contract item with an exact fix, plus a JSON report; secret values never printed | 48 unit tests on mocked output; run on this laptop, which fails 6 of 10 checks (4 GB VRAM, 15 GiB RAM, driver 615.x, non-Ubuntu, 58 GB disk), and against the real bucket, which passes |
| `scripts/host/{bootstrap_host,run_session}.sh` | idempotent image/scene/cache setup; the stand/demo/smoke/nav/replay plan judged by artifacts, with `--shutdown-after` refused outside a VM | dry runs; 12 shell tests with stubbed binaries |
| `scripts/host/run_{stack_check,scene_admission,scene_eval,slam_eval}.sh` | the four archived entry points, re-created provider-neutral (no pod guard, no dead-man switch, no self-terminate; paths from the environment) | dry runs; a test asserts no provider-specific code remains |
| `scripts/host/brev_{up,status,exec,copy,down}.sh` | price/hours/creation/live caps, kill timer before create, ownership guard, spend ledger | 18 mock tests, including every refusal path |
| `scripts/host/` move | driver gate, launch probe, `kit_retry`, `kit_logs`, `launch_gate`, `job_attempt_log`, `run_resilient`, hygiene, allowlist | 1369 tests pass (baseline 1285; the new host tests add 84); lint clean |
| `docs/dev_machine.md` | workstation, AWS worked example, Brev with prices dated 2026-11-22 | prices from the catalog; unverified items marked |

## Bugs found and fixed

1. **`host_doctor` probed the GPU when every GPU check was skipped.** A test with a runner that raises caught it;
   `nvidia-smi` is now only run when at least one of gpu/rt_cores/vram/driver is wanted.
2. **`brev_live_count` never read its input.** It piped JSON into `python3 - <<'PY'`, where the heredoc replaces
   stdin, so every creation would have been refused with "unknown live count". The JSON now passes as an argument.
3. **`date -d '+3.5 hours'` is not portable**, so the kill deadline could be empty. Hours are converted to seconds
   first, and an empty deadline now refuses creation.
4. **`run_session.sh` passed `--internal-stage` to itself without parsing it**, so every stage child would have exited
   2 and the artifact check would have reported a false failure. The flag is handled before the stage loop.
5. **`run_slam_eval.sh`'s preflight read `$2`/`$3` from a function called with one argument**, an unbound variable
   under `set -u`. Fixed and made dry-run aware.
6. **The driver gate hard-coded RunPod termination and a pod-only guard.** It now returns a verdict plus an optional
   `--on-reject-cmd` hook, and the tests assert no provider variable is consulted.
7. Process bug: while editing docs, one script wrote the README text into `docs/project_structure.md`. Caught by the
   diff (355 lines to 121), restored from HEAD, both edits redone separately. No commit contains the damaged file.

## What still fails, and the exact next step

**What fails:** every tier above T0, for one reason: no authenticated Brev session.

**Next step, in order:**

1. Log in: `brev login` (a human, once). Confirm with `brev ls`.
2. `scripts/host/brev_up.sh --name cerberus-s1 --type gpu_1x_rtx6000 --price 0.828` — this arms the 3.5 h kill timer,
   checks the caps, and creates the instance. If the driver is outside the allowlist, replace the instance with the
   next row of the table (L40 or L4) instead of fighting the driver.
3. On the instance: `host_doctor.py`, then `bootstrap_host.sh --data-dir /workspace/cerberus-data`.
4. `run_session.sh --scene MV7J6NIKTKJZ2AABAAAAADQ8_usd` — stand first; if the assertions fail for a code reason, the
   log names the artifact (`assertions.json`) and the 60-minute debug box applies.
5. `brev_down.sh cerberus-s1` when the results are pulled. Delete the credential file first, and rotate the token:
   it is designed to be copied to a rented host.

## Decisions from `outputs/overnight_log.md` that mattered

- Skip Phases 2-4 on the logged-out Brev account (handoff rule), and keep the local phase broad instead of idling.
- Do not run `brev login`: it needs a browser, and asking the owner is not available.
- Name the moved tools provider-neutrally (`driver_gate`, `launch_probe`) rather than keeping `pod_*` names.
- Re-create the four archived entry points with a shared library instead of copying them, so the pod-only paths and
  provider calls cannot come back by accident.
- Mark every Brev command that was not run against a live account as **unverified** in the docs, rather than implying
  a tested path.

## Spend

| item | amount |
|---|---|
| instances created | 0 of 4 |
| instance hours | 0 of 3.5 |
| spend | **0.00 USD of 12.00** |
| R2 objects written | 0 (no results to sync) |
