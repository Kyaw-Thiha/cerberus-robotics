# H2 report — GA retrain, parity, and the Phase 0.5 adversarial gate

Platform: legs-only Unitree Go2. Branch: `main` (H1 was never merged when this started; the work went
onto `main` directly, per instruction). Remote GPU pods: `cerberus-remote-v6:v6-ga` (Isaac Sim 6.2 /
Isaac Lab 3.0 GA `e3f7d9ff` / rsl-rl 5.5.1 / torch 2.11.0+cu130), RTX 4090.

## 1. Commits per work package

| package | commits (subject lines) |
|---|---|
| P0 stack report + probe | `Add environment report and Isaac quat/physics probe scripts`, `compat: resolve the Lab 3.0 physics preset for non-Hydra config construction`, `scripts: read the image's version manifest and run the report inside Isaac's python`, `Launch Kit with cameras enabled in every entry point` |
| P1 parity | `locomotion eval: contract check, ONNX policy fallback, parity report`, `locomotion eval: feed fixed-batch ONNX exports row by row`, `locomotion eval: read the GA actor/critic config layout in the contract check`, `locomotion eval: unwrap rsl-rl's TensorDict observations`, `docs: record the September checkpoint's parity on the GA stack` |
| P2 retrain + artifacts | `locomotion: checkpoint sidecars and GA checkpoint finalize script`, `locomotion: read final reward and terrain stats from the run's event file`, `finalize: launch Kit before exporting ONNX`, `finalize: export ONNX into place and keep its external weights`, `verify the ONNX export against the checkpoint's actor weights` |
| P3 gate | `locomotion: adversarial robustness gate for the frozen candidate`, `attack_env: hand the frozen policy a group-keyed observation`, `adversary: derive the runner config from the task, not from a hand-built dict`, `gate: read the adversary's runner config only from its manifest`, `gate: make scale above 1x actually reach above the trained range`, `docs: record the gate verdict — FAIL at 0.568x the trained push range` |
| tooling | `scripts: one-shot pod drivers …`, `scripts: treat a silent Isaac launch as dead`, `Add a local pre-flight, and correct the claim that R2 was unavailable` |
| P4 scene eval | see `docs/reports/scene_eval_phase0.md` (status recorded there) |

## 2. Gate results

| gate | result | key numbers |
|---|---|---|
| **P0** smoke / PhysX / env report | **PASS** | 64-env smoke wrote a checkpoint, no NaN, reward curve present; physics resolved to `PhysxCfg` (`use_newton_actuators=False`) |
| **P1** parity report exists | **PASS — verdict PARITY BROKEN (marginal)** | mean 79.3% → 75.7% (−3.6 pp, inside 5 pp); one cell (40 N/L4) at −12 pp, outside 10 pp; 17/18 cells inside. Contract: CONTRACT OK |
| **P2** candidate + ONNX + sidecar + sweep | **PASS** | 1500 iters, seed 42, 4096 envs, 4009 s, reward 17.04, terrain 5.85/9/14.3%, falls 13.8%; sweep mean 74.4% (−4.9 pp vs September, inside the 10-point stop rule); ONNX matches the checkpoint actor to 3.8e-06 |
| **P3** adversarial gate | **FAIL** | gate scale **0.568×** of the trained push range, 51% falls, where the standard sweep still reports 82% success. Characterised afterwards: the command channel pins at its clamp (±0.803 m/s) and flips sign ~3.4×/s (≈1.7 Hz per axis) while the vector almost never reverses; applied force is only ~3 N RMS / ~10 N peak against a 68 N allowance; every failure is base contact (zero time-outs) |
| **P4** scene eval | see `docs/reports/scene_eval_phase0.md` | non-blocking: GRScenes fetch + metre preprocess + load verified; the scripted-velocity path is blocked by the backend's navigation observation contract (33 dims, no velocity command vs the policy's 235) |

Reference cells, September → September-on-GA → GA retrain:

| cell | September | Sept on GA | GA retrain |
|---|---|---|---|
| 0 N / level 0 | 94% | 93% | 88% |
| 120 N / level 0 | 89% | 87% | 82% |
| 120 N / level 8 | 61% | 56% | **66%** |

`go2_locomotion_final.pt` was **not** created: the gate failed, so nothing was promoted.

## 3. GPU wall clock per package vs caps

| package | wall clock | cap | notes |
|---|---|---|---|
| P0 | 101 s | smoke only | probe re-run after the physics-preset fix |
| P1 | 103 s | — | 18 cells × 100 trials in one pass |
| P2 | 73.5 min total (training 66.8 min) | 2.95 h | 38% of the cap |
| P3 | 25.3 min training + 6.5 min gate eval | 3 h training | 23% of the cap |
| P3 wasted | ~80 min | — | interface mismatches on the first attempt (see §6) |
| P4 | see scene report | 1 h | |
| **total** | **≈5.5 h ≈ $4.9** | | across three pod instances |

## 4. Everything synced to R2

Bucket layout under `r2://$R2_BUCKET/` (manifest printed by `scripts/sync_to_r2.py` at the end of the
run; the full manifest is pasted in the terminal log and summarised here):

| prefix | contents |
|---|---|
| `migration/adversary_20261108T051752Z/` | trained adversary (1000 iterations, all blocks), manifest |
| `migration/gate_eval_20261108T055128Z/` | `gate_verdict.json` |
| `migration/adversarial_gate_20261108T051752Z/` | driver logs, zero-attack check, dry run |
| `migration/ga_retrain_<stamp>/` | sweep CSV, sidecar, parity table/plot, rollout clip |
| `migration/ga_stack_<stamp>/` | September-on-GA sweep and contract table |

## 5. Findings for H3 (ROS inference node)

1. **Quaternion convention: `xyzw`.** Measured on the GA stack from a standing Go2 with spawn yaw
   pinned to zero: `robot.data.root_quat_w` = `[0.0645, -0.0453, 0.0007, 0.9969]` (scalar last), and
   `projected_gravity` = `[-0.090, -0.129, -0.988]` — down, as it must be. `common/cerberus_compat/quat.py`
   already assumes `xyzw` for Isaac Lab >= 3.0 via the `isaaclab` major version; that assumption is now
   measured, not inferred. **The ROS node must convert to wxyz before publishing.**
2. **No observation or ordering mismatches found.** The contract check compared the live stack against
   the September run's own dumped config: observation terms in the same order, dim 235 (3+3+3+3+12+12+12
   +187 height cells), action dim 12 / scale 0.25 / default offset, decimation 4 at dt 0.005 (50 Hz),
   command ranges ±1 m/s and ±1 rad/s, 10 s resampling, 2% standing envs, no observation normaliser.
   Two *representation* differences had to be translated, not fixed: rsl-rl 5 reports unset fields as
   `dataclasses.MISSING` instead of `null`, and it renamed `ActorCritic` to `MLPModel`.
3. **rsl-rl resume behaviour.** September checkpoints are legacy rsl-rl 3.x (`model_state_dict`) and are
   **rejected** by rsl-rl 5.5.1 (`CheckpointFormatError`) — verified by running the loader, not inferred.
   No conversion path was written. Fine-tuning must start from a GA-produced checkpoint; the GA
   candidate resumes cleanly through `--resume --checkpoint` (used by the robustify driver).
4. **ONNX export details that bite.** rsl-rl 5 exports with a **fixed batch of 1** and with **external
   weights** (`<name>.onnx.data`), so a consumer must feed one row at a time (or re-export with dynamic
   axes) and must ship the `.data` file with the graph.
5. **Actuator/physics path.** Stock Lab 3.0 Go2 selects Newton; the repo pins PhysX in the env cfg, and
   any *new* entry point must build through the Cerberus cfg or it silently changes the actuator model.

## 6. What could not be done, and why

| item | reason |
|---|---|
| Robustification (P3e) | **decided against by the project owner** after the FAIL; no round was run and no new adversary was trained |
| P4 scene eval | see the scene report: status and the exact stopping point are recorded there |
| ONNX action-for-action check on the pod | done locally instead (CPU torch, no rsl-rl needed): 3.8e-06 |
| Continuous R2 syncing during the first pod | R2 credentials were present in `.env` and the driver supports `--upload`; an operator key scan missed them, so no upload was attempted until the last run. Corrected in the run report, and `scripts/preflight_local.sh` now authenticates against the bucket before a pod is started. |

Two pods were reclaimed by the host mid-run; nothing that had been pulled to the workstation was lost,
and the last run uploaded the adversary to R2 as soon as training finished.

## 7. What this phase concludes

The GA stack changes the checkpoint's behaviour only slightly (parity mean −3.6 pp, one cell outside
10 pp), and a RUN_002-recipe retrain on GA lands within 4.9 pp of September while reaching the same
curriculum and terrain distribution. But the retrained candidate **fails** the adversarial gate: a
learned attack breaks it with 68 N + 0.80 m/s, a little over half the magnitude at which the standard
push sweep certifies 82% success. Passing the random-push evaluation is not evidence of robustness, and
the candidate is therefore not frozen for Phase 1.
