# Cerberus — Phase 0.5 handoff: Adversarial robustness gate

Read `docs/cerberus_project_context.md` and `docs/project_structure.md` first
for full context, plus the Phase 0 handoff for what this phase consumes as
input. This doc is the actionable spec for this phase only.

## Objective

Stress-test the Phase 0 candidate locomotion checkpoint
(`go2_locomotion_v1_candidate.pt`) with a purpose-built adversarial attack,
not just the standard random-push tests already run in Phase 0. Gate whether
the checkpoint is allowed to become `go2_locomotion_final.pt` — the version
Phase 1 is permitted to freeze and build a high-level navigation policy on
top of.

**Why this exists, concretely**: Shi, Zhang, Miki, Lee, Hutter, Coros (2024,
RSS), "Rethinking Robustness Assessment: Adversarial Attacks on
Learning-based Quadrupedal Locomotion Controllers," ran four standard
push-test variants (fixed constant push, random push at 0.5Hz, max-force
random-direction push, random impact) against a well-trained, push-randomized
policy — 1000 trials each, zero failures found. A learned adversarial attack
policy then broke the same controller with 100% success. Passing Phase 0's
random-push eval is therefore not sufficient evidence of robustness; this
phase exists to check for the gap that finding demonstrates, before the
checkpoint becomes expensive to change (once Phase 1 composes a high-level
policy on top of it, retraining the locomotion policy invalidates that
composed work too — see project context doc for the full argument).

**No public code exists for this method** (confirmed via search). Everything
below is specified directly from the paper — there is no reference repo to
fall back to for missing details. If something below is ambiguous, resolve it
by re-reading the paper's Section IV, not by improvising.

## Architecture

This is fundamentally different from a normal Isaac Lab training run: the
policy being trained (the adversary) is not the locomotion policy — the
locomotion policy is **frozen** and sits inside the environment as a fixed
component.

Build `policy/locomotion/adversarial_robustness/attack_env.py`: a custom
Isaac Lab environment where, per simulation step:
1. The environment observes robot state (proprioception — same fields as the
   locomotion policy's own input; see Table I of the paper for the exact
   dimensions used there, adapt field names to whatever your Go2 observation
   space actually uses).
2. The **adversary policy** (the thing being trained via `rsl_rl`/PPO) takes
   this observation and outputs an attack action: a perturbation force
   (applied to the base, x/y plane) and a command override (replacing the
   velocity command the locomotion policy would otherwise track).
3. The environment applies the adversary's force and command override, then
   runs the **frozen** locomotion policy forward on the (attacked)
   observation to get its joint-target action, then steps physics.
4. The environment computes the adversary's reward from the resulting robot
   state (formula below) and returns it as the training signal for the
   adversary — the locomotion policy is never updated in this loop.

`train_adversary.py`: standard `rsl_rl` PPO training script, but pointed at
`attack_env.py` instead of a normal locomotion task. Reuse the existing
`rsl_rl` dependency already pinned in `Dockerfile.remote` — no new library.

## Adversary specification (from the paper — implement exactly)

**Network**: MLP, 2–3 hidden layers (paper uses 2 for the simpler "didactic"
case, 3 for the more robust "Miki" policy case — use 3, since Go2's push-
randomized policy is closer to the latter case than the former).

**Reward** (paper Eq. 2–3):
```
adversary_reward = -1_alive + r_aux - λ * Σ_i ||θ_i||_∞

r_aux = c_orient * g_z
      + c_shake  * (ω_x² + ω_y²)
      + c_torque * Σ_j ReLU(|τ_j| / τ_lim,j - 1)
```
Where:
- `1_alive`: constant penalty applied every step the robot has *not* failed
  (this is what drives the adversary toward causing a fall — it's being
  penalized for the robot staying alive).
- `g_z`: z-component of normalized projected gravity (−1 = standing upright,
  1 = lying down) — rewards the adversary for pushing the robot toward bad
  orientation.
- `ω_x, ω_y`: base angular velocity in x/y (roll/pitch rate) — rewards
  inducing a shaking/unstable base.
- `τ_j`, `τ_lim,j`: joint torque and its soft limit for joint j — rewards
  pushing joints toward their torque limits.
- `λ * Σ_i ||θ_i||_∞`: Lipschitz regularization — penalizes the infinity norm
  of each layer's weights in the adversary network. This is what keeps
  learned attacks smooth and physically realistic instead of a degenerate
  bang-bang output. Do not skip this term — the paper's own ablation (their
  Table V) shows removing pieces of this reward structure measurably
  degrades or breaks convergence; the zero-sum formulation (adversary reward
  = negative of locomotion reward) specifically failed to find any attack at
  all in their experiments, so don't substitute that simpler-seeming
  alternative.

**Fall/failure definition**: reuse Phase 0's own training termination
criteria (base contact, roll/pitch limits, whatever was actually configured
there) for what counts as "not alive" in `1_alive` above. Do not define a
separate failure criterion for this phase — the whole point of the gate is
comparing against the policy's actual trained failure boundary.

**Attack modality scope**: implement **perturbation-force + command-space**
attacks only. The paper's own ablation (Table IV) shows single-modality
attacks fail against a policy that already has standard push-domain-
randomization (which the Phase 0 candidate does) — only combinations of two
or more modalities succeed. Force + command is the cheapest pair to
implement (force reuses Phase 0's existing push-injection mechanism; command
override reuses the existing velocity-command input channel). **Do not
implement the observation/state-estimator attack modality in this pass** — it
requires a state estimator with realistic noise characteristics already in
the loop, which is a larger prerequisite not yet built. Leave it as a
documented stretch goal (`docs/cerberus_project_context.md`'s open decisions
list already notes this).

**Rate limiting**: cap the attack's rate of change at 0.1× its max range per
simulation timestep (paper's value) — this, combined with the Lipschitz
regularization above, is what keeps the found attacks realistic rather than
exploiting an unrealistically fast/discontinuous perturbation.

## Gate criterion

Do not use an absolute force/command magnitude threshold copied from the
paper's own numbers (their 15N / ±0.5 m/s / ±3° figures were chosen for a
~30-50kg ANYmal and justified as "realistic" by the paper's authors for that
robot — not a universal constant, and not directly transferable to a ~15kg
Go2).

Instead: **the minimum combined force+command magnitude the trained adversary
needs to induce a fall must exceed the push magnitude range the Phase 0
candidate was actually trained and evaluated against** (the push-recovery
sweep's tested range from the Phase 0 handoff).

- To find this minimum: after the adversary converges, run the reduce-until-
  survives procedure from the paper (Section VI-B, their Fig. 10 methodology)
  — gradually reduce the adversary's available perturbation range until the
  locomotion policy survives, and record that boundary.
- **Gate passes** if this boundary exceeds the Phase 0 trained push range.
- **Gate fails** if the adversary can break the policy at or below that
  range — this means Phase 0's random-push eval passing was an artifact of
  testing the exact magnitude trained on, not evidence of real robustness at
  that magnitude in combination with other realistic factors.

## Robustify-and-reattack loop (only if gate fails)

Per the paper's Section IV-C and V-A.3 (attack → finetune → reattack):
1. Finetune the locomotion policy (warm-started from the candidate
   checkpoint's actor/critic weights) with a mix of rollouts: some continuing
   under standard domain randomization (as in Phase 0), some subjected to the
   adversarial attack found in the failed gate check. The paper uses roughly
   a 5% adversarial-rollout mixing ratio during finetuning — start there,
   adjust if convergence is unstable.
2. Save the finetuned result as `go2_locomotion_v2_candidate.pt` (increment
   the version, don't overwrite `v1`).
3. Train a **new** adversary against this new candidate (a stale adversary
   trained against the old checkpoint isn't a valid retest) and rerun the
   gate.
4. Repeat until the gate passes. Watch for the performance-robustness
   trade-off the paper flags — over-weighting adversarial rollouts during
   finetuning can produce overly conservative locomotion behavior; if
   command-tracking accuracy degrades noticeably (compare against Phase 0's
   own tracking metrics), reduce the adversarial mixing ratio rather than
   pushing through.

## Checkpoint naming (see `docs/project_structure.md` for the full scheme)

```
go2_locomotion_v1_candidate.pt      # from Phase 0, input to this phase
go2_locomotion_v1_robustified.pt    # only if gate failed and robustify ran once
go2_locomotion_v2_candidate.pt      # equivalent to _v1_robustified — pick one naming, don't use both; recommend _v{n}_candidate throughout, reserving "robustified" only for the final passing one if a single pass suffices
go2_locomotion_final.pt             # the checkpoint that passed the gate — only this is used by Phase 1
```
Keep every intermediate checkpoint (don't overwrite) — the before/after
comparison is part of the required deliverable below.

## Repo scaffolding

```
policy/locomotion/adversarial_robustness/
├── attack_env.py           # wrapper env: frozen locomotion policy in the loop
├── train_adversary.py      # rsl_rl PPO training script for the adversary
└── eval/
    ├── standard_baseline.py   # reruns Phase 0's 4 standard-test variants as the control condition
    ├── gate_check.py           # reduce-until-survives, gate pass/fail logic
    └── report.md (or .ipynb)  # the deliverable, see below
```

## Deliverable

A short adversarial vulnerability report under
`policy/locomotion/adversarial_robustness/eval/`, containing:
- Standard-test baseline results (the four ST variants, N=1000 trials each,
  as a control condition — expect these to mostly find nothing, per the
  source paper, which is itself part of the point being demonstrated).
- Adversarial attack result: minimum force+command magnitude to induce
  failure, compared explicitly against Phase 0's trained push range.
- Gate outcome (pass/fail) and reasoning.
- If robustification was needed: before/after comparison — minimum
  adversarial magnitude to fail (should increase after finetuning), plus a
  command-tracking accuracy check (should not meaningfully regress).

## Explicit non-goals for this phase
- No observation/state-estimator attack modality (stretch goal, deferred).
- No cross-robot / generalization testing — single Go2 checkpoint only.
- Don't invent a new absolute-magnitude threshold — the gate is defined
  relative to Phase 0's own trained range, not a copied number from the
  source paper.
- This is not the point where the safety filter gets involved — that reuse
  happens later, in Phase 3, against the policy+filter system. This phase
  only ever tests the bare locomotion policy.

## Done when
- [ ] `attack_env.py` correctly wraps the frozen Phase 0 candidate — verify
      with a smoke test that the frozen policy's weights are never updated
      during adversary training.
- [ ] Adversary trained (force + command modalities, reward formula above,
      Lipschitz regularization included, rate-capped at 0.1×/timestep).
- [ ] Standard-test baseline rerun as the control condition.
- [ ] Gate check run: minimum adversarial magnitude vs. Phase 0's trained
      push range.
- [ ] If gate failed: robustify/reattack loop completed, gate re-checked,
      before/after comparison recorded.
- [ ] `go2_locomotion_final.pt` produced and saved to
      `policy/locomotion/checkpoints/`, synced to object storage.
- [ ] Vulnerability report written under
      `policy/locomotion/adversarial_robustness/eval/`.

**Handing off to Phase 1 next** — Phase 1 is only permitted to freeze and
compose `go2_locomotion_final.pt`, never a `_candidate` or intermediate
`_robustified` checkpoint.
