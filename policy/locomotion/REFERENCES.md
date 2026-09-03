# Phase 0 locomotion — reference papers and what we borrow from each

Per the project's synthesis-over-invention ethos (`docs/cerberus_project_context.md`):
every technique here is borrowed from a published source and credited honestly,
not re-derived from scratch. This doc records which paper backs which piece of
`policy/locomotion/`, and — just as importantly — what we deliberately did *not*
take from each one.

## The base recipe: Rudin et al. 2022

**Rudin, Hoeller, Reist, Hutter — "Learning to Walk in Minutes Using Massively
Parallel Deep Reinforcement Learning"** (CoRL 2022). Code: `legged_gym`
(`leggedrobotics/legged_gym`).

What we use: the whole PPO-via-`rsl_rl`, thousands-of-parallel-environments
training recipe this phase follows is Rudin's. Isaac Lab's built-in
`Isaac-Velocity-Flat/Rough-Unitree-Go2` tasks are themselves an implementation
of this recipe — we start from them unmodified per the Phase 0 handoff.

What we deliberately did **not** use: Rudin's own push-perturbation method
(`_push_robots` in `legged_robot.py` — sets base velocity directly, an
instantaneous "kick"). Verified directly against `legged_gym`'s source. We
use a different, force-based mechanism instead — see below for why.

## Where the push-disturbance mechanism actually comes from

**Hwangbo, Lee, Dosovitskiy, Bellicoso, Tsounis, Koltun, Hutter — "Learning
agile and dynamic motor skills for legged robots"** (Science Robotics, 2019).

This is the origin of two things we use directly:

1. **The curriculum factor formula**, verified against the paper's own arXiv
   LaTeX source (`arxiv.org/abs/1901.08652`):
   ```
   k_{c, j+1} = (k_{c, j}) ^ k_d
   ```
   `k_c` is the curriculum factor (multiplies disturbance magnitude and
   several cost terms), `j` is the RL iteration index, `k_d ∈ (0,1)` is the
   advance rate. Tuned constants from the paper: **k₀ = 0.3, k_d = 0.997**.
   Monotonically increasing, asymptotically approaching 1 — fast ramp early,
   flattening late, no hard cutoff iteration. This is the exact formula our
   `curriculum.py` implements.
2. **Force-based (not velocity-based) perturbation** as the disturbance
   mechanism — "external pushes to the main body" per the paper's own
   description of its qualitative results.

**Lee, Hwangbo, Wellhausen, Koltun, Hutter — "Learning quadrupedal locomotion
over challenging terrain"** (Science Robotics, 2020).

Not directly implemented here — its contributions (teacher-student
distillation, TCN-based student policy, particle-filter adaptive terrain
curriculum) are a different, heavier architecture than what Phase 0 needs.
Isaac Lab's own built-in adaptive terrain-level curriculum (used when we move
to the Rough task) is a native Isaac Lab implementation of the same general
idea, not a port of this paper's code. Included here for lineage: it's the
paper Miki et al. 2022 (next) directly builds on, and the one real published
number we found for this family — **50N applied laterally for 5s** as a
diagnostic robustness *evaluation* test (not training) on ANYmal (~32kg) —
comes from this paper's arXiv source. Not used as our training range (wrong
robot mass, and it's an eval number, not a training range), but was a useful
sanity-check data point when we couldn't verify Hwangbo/Miki's own training
magnitudes (see below).

**Miki, Lee, Hwangbo, Wellhausen, Koltun, Hutter — "Learning robust perceptive
locomotion for quadrupedal robots in the wild"** (Science Robotics, 2022).

Not directly implemented in Phase 0 — its core contribution (exteroceptive
perception fused via a recurrent belief-state encoder) is explicitly Phase 1's
job in this project, architected as a *separate* high-level policy composed
on top of Phase 0's frozen low-level policy, not folded into the same network
the way this paper does it (see "Why we don't add perception now" below).
Confirmed directly against its arXiv LaTeX source that it reuses Hwangbo's
exact curriculum formula (cites it rather than restating), and its own
domain-randomization section confirms force+torque perturbation, matching
Hwangbo. Included here because it's the direct predecessor to the next paper,
and because it's the specific policy Shi et al. 2024 (below) attacks.

## Why this lineage matters for Phase 0.5

**Shi, Zhang, Miki, Lee, Hutter, Coros — "Rethinking Robustness Assessment:
Adversarial Attacks on Learning-based Quadrupedal Locomotion Controllers"**
(RSS 2024) — **this project's own Phase 0.5 reference**, already named in
`docs/cerberus_project_context.md` before this doc existed.

Not implemented in Phase 0 (it's the next phase's whole job). The reason it's
listed here: Shi et al.'s target policy *is* Miki et al.'s policy, and their
adversarial-attack claim explicitly assumes the target "has already been
through standard push-based domain randomization" — for this lab's lineage,
that means force-based, curriculum-ramped disturbance (Hwangbo's method), not
Rudin's velocity-kick. Training Phase 0's checkpoint with the same disturbance
family Shi et al.'s own targets were trained with makes Phase 0.5's gate test
something representative of what it's actually meant to test, rather than a
different, easier baseline.

## The concrete numbers: `isaaclab-go2-locomotion`

Hwangbo/Lee/Miki's own papers give the curriculum *formula* but never publish
an exact training-time force/torque magnitude in Newtons — verified by
reading their arXiv LaTeX source directly (not just the rendered PDF) for all
three papers, plus attempting Miki 2022's official Science Robotics
supplementary material (blocked, paywalled) and the authors' own merged PDF
(extracted successfully via `pdftotext`, but contains no such table either).
This appears to be an implementation detail the papers never made public, not
something we simply couldn't reach.

So the concrete magnitude/timing numbers we anchor to instead come from
**`isaaclab-go2-locomotion`** (github.com/BrandoUlissi/isaaclab-go2-locomotion,
already the project's reference repo for the base recipe) — the *only* source
with numbers actually validated on this exact robot (Unitree Go2) in Isaac
Lab, with a real reported result (87.5% recovery at 120N peak force, vs. 0%
for the untrained baseline):

| Disturbance | Duration | Trigger interval | Magnitude (their linear ramp) |
|---|---|---|---|
| Impulse | 0.15-0.25 s | every 6-10 s | 30 N → 120 N over iter 0-800 |
| Sustained load | 8-12 s | every 25-40 s | 10 N → 40 N over iter 200-1000 |

## Push direction: symmetric on all three axes, including vertical

`curriculum.py` sets `force_range = (-peak, peak)`, applied identically to x,
y, *and* z. An earlier version used `(0, peak)`, which was a real bug, not a
deliberate choice: `apply_external_force_torque` samples each axis
independently from the same range, so `(0, peak)` meant every push pointed
into the same fixed positive-x/y/z octant every time, never a genuinely random
direction.

The fix keeps z symmetric too rather than zeroing it out, because the
literature doesn't support horizontal-only pushes as the standard case. Shi et
al. 2024 (arXiv:2405.12424) describes base perturbation as simulating "pushes
**and payload**" (Sec. III-B) — payload is inherently a vertical force, not a
horizontal one — and no paper in the Hwangbo/Lee/Miki lineage explicitly
restricts base perturbation to the horizontal plane. (One narrow counter-
example exists in Shi et al.'s own LaTeX source — a deliberately-biased
illustrative experiment confining forces to "45° from the x-axis in the
sagittal plane" to demonstrate lateral-push vulnerability — but it's a special
one-off case for making a specific point about adversarial fine-tuning, not
their general domain-randomization recipe.)

We reuse these as the *endpoints* our curriculum ramps toward, via Hwangbo's
asymptotic formula instead of their linear one. **Torque** has no equivalent
verified number anywhere in this lineage (the reference repo is force-only);
our torque range is a reasoned estimate (force × an approximate Go2 body-frame
lever arm), flagged explicitly as such rather than a verified figure.

## Why we don't add perception into Phase 0's policy

Miki et al. 2022's architecture bakes perception directly into the same
network as the locomotion policy. This project's plan
(`docs/cerberus_project_context.md`, `docs/project_structure.md`) uses a
different, hierarchical pattern: Phase 0's policy stays blind/proprioceptive
permanently, frozen after the Phase 0.5 gate, and Phase 1 composes a
*separate* high-level policy on top of it that consumes perception and issues
velocity commands — Isaac Lab's native hierarchical composition support,
not teacher-student distillation into the same network. Retrofitting
perception into Phase 0's policy later is possible in principle but would mean
discarding the frozen checkpoint and retraining from scratch with a different
architecture, not extending what exists.
