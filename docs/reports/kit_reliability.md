# Kit launch reliability on this host class

Numbers measured on Hyperstack L40 VMs with the `cerberus-remote-v6` image (Isaac Sim 6.2 / Isaac Lab 3.0 GA) and
driver 595.58.03. "Start" means our code ran: the launcher printed the build-timing marker (scene runs) or the
training log directory line (training runs). "Complete" means the unit of work left its artifacts.

## What was measured

| observation | value |
|---|---|
| launch attempts, 2026-11-23 session, all stages | 163 |
| starts | 4 (2.5 %) |
| launch attempts, 2026-11-22 session | 63 |
| starts | 6 (9.5 %) |
| dominant failure | loader crash at 14-15 s, `A crash has occurred`, exit -9, dump in `/isaac-sim/kit/data/Kit/IsaacLab/3.0/` |
| second failure | runs that start, then die at 36-80 s with a Python traceback (our code) |
| masked failure | runs that leave no artifacts and exit 0 at the first Python block after a long sim block |
| recovery | relaunch; the 1500-iteration arm retrain started on the 24th attempt of its wrapper |

## A/B matrix

The matrix (V0 baseline / V1 `taskset -c 0-7` / V2 `PXR_WORK_THREAD_LIMIT=1 OMP_NUM_THREADS=1` / V3 official 6.2.0
container / V4 cameras off) is implemented as `scripts/host/kit_ab_probe.sh` with a minimal launcher
(`scripts/host/kit_min_probe.py`), 8-15 launches per variant.

Only V0 was attempted, and its first run used a probe whose `--device` default (`None`) made the app fail at 3 s, so
its 8 failures measured the probe, not the host; the corrected run was cancelled to stop competing with the retrain.
**The matrix is therefore not measured.** V3 could not run at all: no `NGC_API_KEY` is configured for this
environment.

## What is in place instead

| tool | job |
|---|---|
| `scripts/host/kit_retry.py` | relaunch until a start marker; fail fast on the crash signature; never waits out the 5-minute window on a known crash |
| `scripts/host/run_until_complete.py` | relaunch until the declared outputs exist; a clean exit without its marker counts as a crash; resume from the newest checkpoint for training |
| `scripts/host/train_arm_retrain.sh` | the training shape of the same rule (launch, resume, relaunch, stop when `model_<iters>.pt` exists) |
| `scripts/host/kit_crash_classify.py` | classify each failure as masked crash or clean exit from `[Fatal]`, crash-reporter lines, dump files, `coredumpctl` and dmesg `Xid` lines |

## Recommended default

Keep the relaunch-until-artifacts rule as the default for every Kit job on this host class, and keep driver 595.58.03
(the only one on which camera runs survived here). Read `docs/reports/kit_crash_report.md` for the evidence; the
report is prepared but not filed.
