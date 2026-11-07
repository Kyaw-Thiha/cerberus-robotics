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
| Adversarial gate | see §4 |

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

Sweep result: mean 74.4% vs September's 79.3% (−4.9 pp, inside the 10-point stop threshold), so the
adversarial gate proceeds. Five cells move >10 pp in both directions — better at 0 N / level 8 (+11),
worse at 80 N / L0, 120 N / L4, 160 N / L4, 200 N / L8 (−12 to −14).

## 4. Adversarial gate

<!-- FILLED IN WHEN THE GATE RUNS -->

## 5. Verification performed

- `pixi run pytest -q` on the workstation: 1008 passed, 18 skipped before the pod work; the new modules
  add 100+ tests of their own (gate math, contract check, parity report, sidecars, quat probe, policy IO).
- On the pod: every gate above is decided from an artifact, not an exit code (SimulationApp exits 0
  whatever happened); the drivers judge each step that way and retry crashed/stalled launches.

## 6. Deviations and things that contradict the handoff

- H1 was not merged into `main` when this started; per instruction the work went on `main` directly.
- No R2 credentials on the workstation or the pod, so `scripts/sync_to_r2.py` could not run; artifacts
  were pulled to `data/reports/` with `scripts/pull_pod_artifacts.sh` instead.
- The September checkpoint cannot load through rsl-rl 5 at all (verified), so "try the `.pt` first" is
  answered by the format guard rather than by an attempt that could have succeeded.
- Scene eval (P4) needs the GRScenes package from R2; with no R2 credentials it stops at its first step.
