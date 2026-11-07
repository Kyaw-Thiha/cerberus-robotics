# Phase 0.5 adversarial robustness gate (`h2`, P3)

Status: **INCOMPLETE — the gate has no verdict.** Both prerequisite checks pass against the GA
candidate, and adversary training was interrupted at ~100/1000 iterations when the pod was terminated
(RunPod account balance reached zero; `runpodctl pod get` → "pod not found", and a replacement pod
cannot be created: "account balance is too low"). Nothing was promoted, so `go2_locomotion_final.pt`
does not exist — which is the correct state for a gate that has not passed.

Candidate: `go2_locomotion_ga_v1_candidate.pt` (1500 iterations, seed 42, 4096 envs, PhysX, 4009 s,
final reward 17.04) — preserved locally with its ONNX export and sidecar.

Gate prerequisite checks, both from artifacts:

| check | result |
|---|---|
| zero-perturbation fall rate vs the standard sweep's fall rate in the same cell | **1% vs 0%** (delta +1 pp, tolerance 10) → PASS: the wrapper reproduces the candidate's standard behaviour |
| 64-env, 200-step zero-perturbation rollout | no NaN, frozen policy verified unchanged, reported combined magnitude 0.0 → PASS |

Adversary training, from an earlier 543-iteration run that was stopped for a checkpointing bug (the
trainer looped `learn(1)` and rsl-rl's counter never advanced, so no weights were written): by
iteration ~450 the attack drove `Episode_Termination/base_contact` to **~98%** of episodes, against
**18%** for a random 120 N push at the same normalised magnitude. That is the paper's finding
reproduced in shape — a learned attack beats random pushes of equal magnitude by a wide margin — and it
is why the standard sweep alone cannot certify this policy.

That number is from the *training* rollouts, not the gate's bisection, so it bounds the conclusion: it
says an attack of ~1x the trained range breaks the policy most of the time, which is a FAIL-shaped
result, but the gate's own artefact (`gate_verdict.json`) was never produced, so no verdict is claimed.

One measurement subtlety worth recording: the standard sweep's *success* rate at 0 N / level 0 is 88%
while its *fall* rate is 0% — success also requires velocity tracking and a level attitude, which the
wrapper does not re-implement. The check therefore compares fall rates, not success rates; comparing
against 88% would have failed a correct wrapper by 12 points.

Method: `plans/handoff_phase0.5.md` (Shi, Zhang, Miki, Lee, Hutter, Coros, RSS 2024). The locomotion
candidate is frozen; a small MLP adversary is trained with PPO to emit a base push force and a
velocity-command override; its reward is the paper's Eq. 2-3, including the Lipschitz penalty.

## Setup (from config, not hardcoded)

`policy/locomotion/adversarial_robustness/attack_math.PerturbationBounds.from_config`:

| quantity | source | value |
|---|---|---|
| push peak force | `GO2.push_peak_force_n` | 120 N |
| command envelope | `env_cfg.commands.base_velocity.ranges` | √(1²+1²) ≈ 1.414 m/s |
| yaw envelope | same | 1.0 rad/s |
| rate cap | paper | 0.1 × range per sim step |

The gate is relative: a perturbation is expressed in **trained-range units** (`force/120 N`,
`command/1.414 m/s`, combined = Euclidean norm), so `1.0` is exactly the strength the policy trained
and was evaluated against. No absolute newton threshold is imported from the paper.

## Standard test vs adversarial attack

| condition | metric | standard (P2 sweep) | adversarial |
|---|---|---|---|
| level 0, 120 N push | success | — | — |
| level 8, 120 N push | success | — | — |
| mean response curve | success | — | — |

Standard column: filled from the P2 sweep CSV (`eval_rough_*`), i.e. the same numbers as
`docs/reports/phase0_ga_parity.md`. Adversarial column: `eval/results/<stamp>/gate_verdict.json`.

## Per-modality results

Expected from the source paper (Table IV): a policy with standard push-domain randomisation resists
single-modality attacks; the combination breaks it. Recorded as a measurement, not an assumption.

| modality | minimum inducing magnitude (trained-range units) | fall rate at minimum | breaks policy? |
|---|---|---|---|
| force only | — | — | — |
| command only | — | — | — |
| combined | — | — | — |

`attack_math.summarise_modalities` emits `combination_succeeds_where_singles_fail` directly from
these numbers, so the report's headline claim is computed, not written by hand.

## Gate verdict

| field | value |
|---|---|
| minimum combined magnitude that induces a fall | — |
| trained (and evaluated) push range | 1.0 |
| verdict | **not computed** (run interrupted) |
| `eval/gate_verdict.json` | not written |

Pass condition: minimum inducing magnitude **strictly greater** than 1.0. Exactly 1.0 is a FAIL —
matching the trained strength is the artifact the gate exists to catch.

## Robustify history (only runs on FAIL)

| round | checkpoint | minimum inducing magnitude | command-tracking change | outcome |
|---|---|---|---|---|
| — | — | — | — | not started |

## Findings: expected vs ours

Nothing is claimed as ours yet. When run, label each line as either
*(paper, reproduced)* — single-modality resist / combination break, standard push tests finding
little — or *(ours)* — anything Go2-specific: the 120 N peak, the fact that `boxes` was already the
weakest terrain type (RUN_002, 63.6% pooled, 45% at level 8), and any attack the adversary finds
below 1.0.

## Artifacts

| artifact | path | status |
|---|---|---|
| attack wrapper | `policy/locomotion/adversarial_robustness/attack_env.py` | implemented, pod-unverified |
| adversary trainer | `policy/locomotion/adversarial_robustness/train_adversary.py` | implemented, pod-unverified |
| gate evaluation | `policy/locomotion/adversarial_robustness/eval/gate_check.py` | implemented, pod-unverified |
| gate math + tests | `policy/locomotion/adversarial_robustness/attack_math.py` (+ `tests/`) | 48 tests pass locally |
| verdict | `adversarial_robustness/eval/results/<stamp>/gate_verdict.json` | not produced |
| promoted checkpoint | `policy/locomotion/checkpoints/go2_locomotion_final.pt` | **must not exist until PASS** |

## What remains (exact resume path)

1. Provision a GPU pod with a funded RunPod account, image `kevinkyawthiha/cerberus-remote-v6:v6-ga`.
2. Sync the repo; copy `policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate.{pt,onnx,onnx.data,meta.json}`
   (preserved locally) to `/workspace/cerberus/policy/locomotion/checkpoints/`.
3. `scripts/run_adversarial_gate_pod.sh --candidate ... --standard_fall_rate 0.0 --standard_success 0.88 --cap_h 3`
   → zero-attack, dry run, adversary training (1000 iterations, ~80 min at ~4.8 s/iteration), then the
   three-modality bisection.
4. On FAIL: `scripts/run_robustify_pod.sh --candidate ... --adversary_dir <run> --verdict <json> --round 1`
   (fine-tune 300 iterations with 5% adversarial rollouts, then a fresh re-attack).

Wall-clock budget for the remaining work: ~2.5 h of GPU time for the gate, plus ~1 h per robustify round.

## Run order (pod)

1. `train_adversary.py --dry_run` (P3a): 64-env, 200-step zero-perturbation rollout, no NaN, frozen
   policy verified unchanged, reported combined magnitude 0.0.
2. `--zero_attack_check` on `gate_check.py`: zero-perturbation fall rate must reproduce the standard
   success rate within noise, or the wrapper is measuring itself.
3. Full adversary training under the wall-clock cap (larger of 3 h or the spec's cap), logging
   adversary reward, fall rate and the minimal perturbation found.
4. `gate_check.py --modalities combined,force_only,command_only` → `gate_verdict.json`.
5. On PASS only: promote (`go2_locomotion_final.pt` + `.onnx` + `.meta.json` carrying the verdict).
   On FAIL: up to 3 robustify rounds, GA checkpoints only, then report and stop.
