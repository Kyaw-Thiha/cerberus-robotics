# Phase 0 parity: September checkpoint on the Isaac Lab 3.0 GA stack

Status: **measured** (2026-11-07, pod `rbbfy5mf5rd4az`, RTX 4090, driver 595.91.07). The GA retrain row
is added by the same table once that run finishes.

## What is being compared

| | September (source of truth) | GA (measured now) |
|---|---|---|
| checkpoint | `go2_locomotion_v1_candidate.pt` (2026-09-08, RUN_002) | the same file, same weights |
| stack | Isaac Sim 5.x / Isaac Lab 2.3.2 / rsl-rl 3.1.2 | Isaac Sim 6.2 / Isaac Lab 3.0 GA (`e3f7d9ff`) / rsl-rl 5.5.1 |
| control path | legacy `ActorCritic` | ONNX export, fed row by row |
| physics | PhysX | **PhysX** (`PhysxCfg`, `use_newton_actuators=False`) |

## Loading path: the `.pt` is rejected, the ONNX export is what ran

`check_checkpoint_format` on the September `.pt`, with the GA stack's rsl-rl 5.5.1:

```
CheckpointFormatError: Checkpoint '.../go2_locomotion_v1_candidate.pt' is in the LEGACY rsl-rl 3.x
format (single 'model_state_dict' key), but rsl-rl-lib 5.5.1 expects the SPLIT format
('actor_state_dict' + 'critic_state_dict', ...).
```

So the documented fallback is not optional here: the evaluation ran
`go2_locomotion_v1_candidate.onnx` through onnxruntime. Two properties of that export mattered:

- **input shape `[1, 235]`** — a fixed batch of one, so the 1800-env sweep is fed row by row (the
  wrapper reads the declared batch and falls back automatically);
- **no observation normaliser** — `actor_obs_normalization: false` in the reference, confirmed by the
  contract check, so raw observations go straight in.

## Contract check: CONTRACT OK

Produced by `push_recovery_eval.py --contract_reference` against the September run's own dumped config,
before any numbers:

| field | September | GA | status |
|---|---|---|---|
| observation terms | `base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions, height_scan` | same, same order | ok |
| observation dim | 235 | 235 | ok |
| action dim / scale / offset | 12 / 0.25 / 0.0 (default-offset) | same | ok |
| decimation, sim dt, step dt | 4 / 0.005 s / 0.02 s | same | ok |
| command ranges | ±1.0 m/s x/y, ±1.0 rad/s yaw, heading ±π | same | ok |
| command resample / standing envs | 10 s / 2% | same | ok |
| observation normaliser | none | none | ok |
| actor hidden dims | 512-256-128 | 512-256-128 | ok |
| actor class | `ActorCritic` (rsl-rl 3.x) | `MLPModel` (rsl-rl 5.x) | renamed, same network |

Two translation issues had to be fixed before the check could say anything (both were false alarms, not
drift): rsl-rl 5 reports unset fields as `dataclasses.MISSING` rather than `null`, and it renamed the
actor-critic class. Anything genuinely different (a different architecture, an added normaliser) still
fails the check.

Also recorded: **`robot.data.root_quat_w` is `xyzw` on this stack** (scalar component last), and a standing
Go2's `projected_gravity` reads `[-0.090, -0.129, -0.988]` — down, as it must be. That is the H3 input.

## Side by side (100 trials/cell, 1800 trials total)

| push | level | September | September on GA | delta |
|---|---|---|---|---|
| 0 N | 0 | 94% | 93% | −1 pp |
| 0 N | 4 | 86% | 82% | −4 pp |
| 0 N | 8 | 65% | 63% | −2 pp |
| 40 N | 0 | 93% | 90% | −3 pp |
| 40 N | 4 | 88% | 76% | **−12 pp** |
| 40 N | 8 | 71% | 68% | −3 pp |
| 80 N | 0 | 93% | 88% | −5 pp |
| 80 N | 4 | 88% | 84% | −4 pp |
| 80 N | 8 | 66% | 62% | −4 pp |
| 120 N | 0 | 89% | 87% | −2 pp |
| 120 N | 4 | 86% | 81% | −5 pp |
| 120 N | 8 | 61% | 56% | −5 pp |
| 160 N | 0 | 90% | 88% | −2 pp |
| 160 N | 4 | 84% | 75% | −9 pp |
| 160 N | 8 | 64% | 58% | −6 pp |
| 200 N | 0 | 74% | 77% | +3 pp |
| 200 N | 4 | 74% | 74% | 0 pp |
| 200 N | 8 | 62% | 61% | −1 pp |
| **mean** | | **79.3%** | **75.7%** | **−3.6 pp** |

![parity](parity_september_on_ga.png)

## Verdict: PARITY BROKEN (marginally)

Criteria from the handoff: every cell within 10 percentage points, and the mean within 5.

- mean: **−3.6 pp** → inside the 5 pp mean tolerance;
- per-cell: **one** cell outside 10 pp — 40 N / level 4 at −12 pp; 160 N / level 4 is −9 pp.

Interpretation, stated plainly: the checkpoint **transfers** — the contract is identical, the response
curve keeps its shape, and 17 of 18 cells are within 10 pp — but it is not bit-for-bit equivalent, and
one cell trips the per-cell rule. With 100 trials per cell the binomial standard error is ~5 pp, so a
single −12 pp cell is roughly 2σ; that is suggestive rather than conclusive, and the honest description
is "small but real degradation, concentrated in one cell".

What it does **not** mean: the checkpoint is unusable, or the retrain is answering a different question.
The handoff makes the GA-trained checkpoint the Phase 0.5 candidate regardless of this verdict.

Most likely causes, in order: the actuator/contact path differs between Lab 2.3.2 and 3.0 GA even with
PhysX pinned, and the ONNX export is the actor alone (no recurrent state, no normaliser — neither
applies here, but it is a different execution path from the runner).

## Reproduce

```bash
$ISAACLAB_PATH/isaaclab.sh -p policy/locomotion/eval/push_recovery_eval.py \
    --platform go2 --terrain rough --headless \
    --checkpoint policy/locomotion/checkpoints/go2_locomotion_v1_candidate.pt \
    --policy_onnx policy/locomotion/checkpoints/go2_locomotion_v1_candidate.onnx \
    --contract_reference policy/locomotion/runs/2026-09-08_19-28-32_analysis \
    --magnitudes 0,40,80,120,160,200 --terrain_levels 0,4,8 --trials_per_cell 100 \
    --output <report>/eval_rough_september_ckpt_on_ga.csv

python3 policy/locomotion/eval/make_parity_report.py \
    --reference policy/locomotion/runs/2026-09-08_19-28-32_analysis/eval/eval_rough_full_sweep.csv \
    --candidate <report>/eval_rough_september_ckpt_on_ga.csv \
    --contract <report>/eval_rough_september_ckpt_on_ga_contract.txt \
    --reference-label September --candidate-label "September on GA" \
    --out-md <report>/parity_september_on_ga.md --out-plot <report>/parity_september_on_ga.png
```

Artifacts: `data/reports/ga_stack/` (summary CSV, per-trial CSV, contract table, provenance JSON, plot,
this table's generator output).
