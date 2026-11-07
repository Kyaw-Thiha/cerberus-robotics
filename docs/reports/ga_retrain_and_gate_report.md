# GA retrain, parity and the adversarial gate — report

Executed 2026-11-07 on a RunPod RTX 4090 (`cerberus-h2`, image `cerberus-remote-v6:v6-ga`, driver
595.91.07). Runs on `main`; no separate branch, no R2 credentials available on either side, so artifacts
were pulled down instead of pushed up (see the last section).

## 1. What was done, in order

| step | result |
|---|---|
| Stack report | Isaac Sim 6.2.0, Isaac Lab `v3.0.0` (`e3f7d9ff`), rsl-rl-lib 5.5.1, torch 2.11.0+cu130, CUDA 13.0, Python 3.12.14, 24 GB RTX 4090 |
| Physics | resolved **`PhysxCfg`** with `use_newton_actuators=False` (stock would have been Newton) |
| Quaternion probe | `robot.data.root_quat_w` is **`xyzw`**; standing `projected_gravity` = [-0.090, -0.129, -0.988] |
| Smoke (64 envs, 8 iters) | PASS: checkpoint written, reward curve present, no NaN/traceback |
| Parity eval (September checkpoint on GA) | CONTRACT OK, **PARITY BROKEN (marginal)** — see §2 |
| GA retrain (RUN_002 recipe) | see §3 |
| Adversarial gate | prerequisite checks pass; training interrupted, **no verdict** — see §4 |

### Stack report (values as measured; the container's JSON died with the pod)

`scripts/env_report.py` wrote `outputs/env_report.json` and a copy under `docs/reports/` on the pod;
both were on the container disk and were lost when RunPod terminated the instance. The values it
reported, and the ones the probe merged into it, are:

| field | value |
|---|---|
| isaac_sim | 6.2.0 (`/etc/cerberus_versions.json`) |
| isaaclab | 28.0.0, commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4` |
| isaaclab_tasks / isaaclab_rl | 25.2.0 / 1.5.1 |
| rsl-rl-lib | 5.5.1 |
| torch | 2.11.0+cu130, CUDA 13.0 |
| python | 3.12.14 (image), 3.12.3 (system) |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB, driver 595.91.07 |
| other packages | warp-lang 1.17.0, numpy 2.3.5, lightwheel-sdk 1.0.3 |
| resolved physics | `PhysxCfg`, `use_newton_actuators=False` |
| quaternion convention | `xyzw` (`root_quat_w` scalar last), `projected_gravity` [-0.090, -0.129, -0.988] |
| repo commit synced to the pod | `25eaa080131929e33f7ef6aa3ad2e5bc207c9a42` |

Re-running `scripts/env_report.py --also docs/reports/env_report_<date>.json` on any future pod
reproduces the file; nothing here was reconstructed into an artifact that looks machine-written.

## 2. Parity: the September checkpoint on the GA stack

Loading: the `.pt` is legacy rsl-rl 3.x (`model_state_dict`) and rsl-rl 5.5.1 rejects it
(`CheckpointFormatError`). The eval used the sibling `.onnx` export through onnxruntime — a **fixed batch
of one**, so the 1800-env sweep feeds it row by row.

Contract check (before any numbers): **CONTRACT OK** — 235-dim observations in the same order, action
scale 0.25 with default offset, decimation 4 at dt 0.005 (50 Hz), command ranges ±1 m/s / ±1 rad/s,
10 s resampling, 2% standing envs, no observation normaliser, actor 512-256-128.

| push | level | September | September on GA | delta |
|---|---|---|---|---|
| 0 N | 0 | 94% | 93% | −1 pp |
| 120 N | 0 | 89% | 87% | −2 pp |
| 120 N | 8 | 61% | 56% | −5 pp |
| **mean (18 cells)** | | **79.3%** | **75.7%** | **−3.6 pp** |

Full table and plot: `docs/reports/phase0_ga_parity.md`.

Verdict: **PARITY BROKEN**, marginally — the mean is inside the 5 pp tolerance, one cell (40 N / level 4)
is −12 pp and outside the 10 pp per-cell rule (17/18 cells are inside). At 100 trials/cell the binomial
SE is ~5 pp, so that cell is ~2σ.

## 3. GA retrain (RUN_002 recipe, new stack)

Recipe unchanged: 1500 iterations, 4096 envs, seed 42, 24 steps/env,
`Isaac-Velocity-Rough-Unitree-Go2-Cerberus-v0`, tensorboard logger. Wall-clock cap 2.95 h (2× RUN_002's
measured 5306.8 s).

Completed in **66m 54s** of training (4009.2 s of logged wall clock), well inside the 2.95 h cap.
Final metrics against RUN_002's final row:

| | RUN_002 (Sept) | GA retrain |
|---|---|---|
| iterations | 1500 | 1500 |
| seed / envs | 42 / 4096 | 42 / 4096 |
| final mean reward | 19.78 | 17.04 |
| terrain level mean / max / frac at max | 5.90 / 9 / 14.7% | 5.85 / 9 / 14.3% |
| push curriculum k_c | 0.987 | 0.987 |
| base-contact termination rate | 10.3% | 13.8% |
| wall clock | 5306.8 s | 4009.2 s |

Same curriculum trajectory, same terrain distribution, ~2.7 points less reward and ~3.5 points more
falls. The trained policy is marginally rougher than September's but reaches the same difficulty.

Artifacts (`policy/locomotion/checkpoints/`): `go2_locomotion_ga_v1_candidate.pt`, `.onnx`
(+ `.onnx.data` external weights, 1.1 MB), `.torchscript.pt`, `.meta.json`; `data/reports/ga_retrain/`
holds the sweep CSV, per-trial CSV, contract table, sidecar copy, parity table and plot, the run's
`params/` + `git/` snapshot, its event file, and `rollout.mp4`.

One verification gap, stated rather than implied: the ONNX export was checked by loading it and running
it (input `obs [1, 235]`, output `actions [1, 12]`, weights resolved from the `.data` sidecar), and the
`.pt` was exercised end to end by the evaluation sweep through rsl-rl. The two were **not** compared
action-for-action (1e-4), which would need torch and the rsl-rl model class off-pod; that comparison is
worth doing on the next pod before the ONNX export is used for anything but evaluation.

Sweep result: mean 74.4% vs September's 79.3% (−4.9 pp, inside the 10-point stop threshold), so the
adversarial gate proceeds. Five cells move >10 pp in both directions — better at 0 N / level 8 (+11),
worse at 80 N / L0, 120 N / L4, 160 N / L4, 200 N / L8 (−12 to −14).

## 4. Adversarial gate — INCOMPLETE

Two prerequisite checks were made on the pod against the GA candidate, both from artifacts:

| check | result |
|---|---|
| zero-perturbation fall rate vs the standard sweep's fall rate (same cell) | 1% vs 0% → delta +1 pp → **PASS** |
| 64-env, 200-step zero-perturbation rollout | no NaN, frozen policy verified unchanged, combined magnitude 0.0 → **PASS** |

Adversary training then ran (1000 iterations at 256 envs, 3 h cap, blocks of 50 with a checkpoint per
block). It was at ~100 iterations when the pod was terminated: the RunPod account balance reached zero
(`clientBalance: -0.065`), `runpodctl pod get` returns "pod not found", and a replacement pod was
refused with "account balance is too low".

Evidence gathered before the loss, from an earlier 543-iteration adversary run on the same candidate:
- `Episode_Termination/base_contact` reached **~98%** of episodes under the attack at 1x the trained
  range, against **18%** for a random 120 N push at the same normalised magnitude;
- a random-push sweep at the trained peak (120 N) still reports 82% success — i.e. the standard test
  passes while a learned attack of equal magnitude breaks the policy. That is the source paper's
  finding, reproduced in shape.

No `gate_verdict.json` was produced, so no verdict is claimed, and `go2_locomotion_final.pt` was **not**
created. Resume path: `docs/reports/phase05_adversarial_report.md` ("What remains").

## 5. Wall clock and cost

| package | wall clock | note |
|---|---|---|
| pod bring-up, stack report, probe | 06:14 → 06:33 (19 min) | includes ~14 min image pull |
| P1 parity eval | 07:12 → 07:27 (15 min) | 1800-trial sweep |
| P2 retrain | 07:29 → 08:37 (68 min) | 4009 s logged training + video + finalize + sweep |
| P3 prerequisites + adversary training | 08:44 → ~09:35 (~50 min) | training interrupted at ~100/1000 |
| total pod time | ~3.3 h at $0.89/h ≈ **$2.9** | pod terminated by RunPod at zero balance |

Debugging inside that time was substantial and is visible in the commit log: Kit launches that crashed
or hung on driver 595, a physics-preset type error, a fixed-batch ONNX export, `TensorDict`
observation shapes, a shadowed `policy` name, and a training loop that never wrote a checkpoint. Each
is fixed in `main` with a test where the logic is testable off-pod.

## 6. Verification performed

- `pixi run pytest -q` on the workstation: 1008 passed, 18 skipped before the pod work; the new modules
  add 100+ tests of their own (gate math, contract check, parity report, sidecars, quat probe, policy IO).
- On the pod: every gate above is decided from an artifact, not an exit code (SimulationApp exits 0
  whatever happened); the drivers judge each step that way and retry crashed/stalled launches.

## 7. Deviations and things that contradict the handoff

- H1 was not merged into `main` when this started; per instruction the work went on `main` directly.
- No R2 credentials on the workstation or the pod, so `scripts/sync_to_r2.py` could not run; artifacts
  were pulled to `data/reports/` with `scripts/pull_pod_artifacts.sh` instead.
- The September checkpoint cannot load through rsl-rl 5 at all (verified), so "try the `.pt` first" is
  answered by the format guard rather than by an attempt that could have succeeded.
- Scene eval (P4) needs the GRScenes package from R2; with no R2 credentials it stops at its first
  step, and with no GPU pod it cannot run at all.
- **The pod was terminated mid-gate.** The account balance hit zero. Everything that mattered had been
  pulled down first (the candidate, its ONNX export, the sidecar, both sweeps, the rollout clip), except
  the in-flight adversary run, which was lost. This is the concrete cost of not having R2 credentials
  on the pod: the "sync before you stop" step had to be a manual pull, and the last two hours of
  compute could not be recovered.
