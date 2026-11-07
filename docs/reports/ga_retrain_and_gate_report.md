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

<!-- FILLED IN WHEN THE RUN FINISHES -->

Wall clock so far: training started 07:29:24 UTC; at 733/1500 iterations the projected total is ~65 min
(2.6 s/iteration), inside the 2.95 h cap with room to spare. Early-run comparison with RUN_002 at the
same iteration (~500): reward 15.9 vs 18.0, terrain level 4.5 vs 3.2 (the curriculum advances faster),
but fall rate 22.7% vs 7.0% — the GA run falls more often at equal curriculum progress, which is the
difference the evaluation has to settle.

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
