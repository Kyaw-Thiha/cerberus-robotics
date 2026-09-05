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

## Curriculum update cadence: once per training iteration, not once per call

`curriculum.py`'s `push_disturbance_curriculum` is invoked by Isaac Lab's
`CurriculumManager.compute()`, which is called from
`ManagerBasedRLEnv._reset_idx()` — **not** once per training iteration. With
`num_envs=4096` running in parallel, some environment resets on nearly every
single simulation step, so an unthrottled `k_c = k_c**KD` update was firing
roughly `num_steps_per_env` (24) times per training iteration instead of once.

This is a real bug, found via a real training run: `k_c` was empirically
observed reaching ~1.0 within the first ~8-10 iterations of *any* run,
regardless of total training budget (confirmed against `isaaclab-go2-locomotion`'s
own table above — their linear ramp spans iterations 0-800, a genuinely
gradual multi-hundred-iteration schedule, which an ~24x-faster update rate
does not remotely resemble). Hwangbo's `k0=0.3, kd=0.997` constants are only
meaningful at the update cadence the paper itself used ("once per training
session's iteration," their own convention) — applying them ~24x more often
collapses the intended gradual ramp into the first 1-2% of any run, meaning
the curriculum was spending nearly the entire training run at full difficulty.

Fixed by throttling: track `env.common_step_counter // NUM_STEPS_PER_ENV` as
an iteration index, and only apply the `k_c**KD` update when that index
actually changes, regardless of how many times `_reset_idx()` fires within
that window. `NUM_STEPS_PER_ENV=24` is hardcoded to match
`RslRlOnPolicyRunnerCfg.num_steps_per_env` (identical for both Go2 Flat and
Rough runner cfgs) — not derivable from the env cfg itself, since the
curriculum term has no visibility into the RSL-RL runner config.

## True impulse, not a sustained force

Found investigating a real result: a fixed-curriculum 1500-iteration Flat run's
own training metrics showed `Episode_Termination/base_contact` pinned near
100% (virtually no episode surviving to its natural time_out) essentially
unchanged from iteration ~850 through 1500, and the resulting checkpoint's
push-recovery eval sweep showed a hard cliff -- 96% recovery at 0N, 86% at
40N, **10% at 80N, 0% at 120N** (the exact peak trained magnitude) and above.
`isaaclab-go2-locomotion` reports 87.5% recovery at 120N, so a 0% result at
the same magnitude, on a checkpoint whose training metrics otherwise looked
healthy (velocity tracking error was *improving* over the run, not
degrading), meant something structural, not undertraining.

Traced to Isaac Lab's `Articulation.write_data_to_sim`/`reset` source
directly: `set_forces_and_torques` (which both curriculum.py's training push
and push_recovery_event.py's eval push used, via
`asset.permanent_wrench_composer`) writes into the *permanent* wrench
composer -- distinct from the *instantaneous* wrench composer, which resets
every physics step. The permanent composer only resets at **episode reset**,
never between an interval EventTerm's triggers. So what both callers called
a "push" was actually a continuous, unending force:

- **In eval**, `apply_single_push` fired once at `push_time_s` and was never
  cleared again -- the robot experienced a constant force at the full
  sampled magnitude for the rest of the episode (~4s), not a shove. At 120N
  (close to the Go2's own ~147N body weight) applied horizontally,
  non-stop, no legged robot recovers from that.
- **In training**, the curriculum's interval-mode trigger had the identical
  issue: each trigger's sampled force persisted for the *entire* interval
  (5-12s early, 5-7s dense) until the next trigger replaced it -- explaining
  the persistent ~100% `base_contact` failure rate throughout the run.

This also revealed a category mismatch in where `PEAK_FORCE_N=120.0` came
from: it was anchored to isaaclab-go2-locomotion's **Impulse** row (30N to
120N, but lasting only **0.15-0.25s**), while our implementation actually
behaved like their **Sustained load** row (continuous, multi-second) --
whose own magnitude ceiling is just 10N to 40N. We had combined the high
magnitude of one disturbance category with the duration semantics of the
other, without ever implementing either one's actual duration correctly.

**Fix** (`core/push_impulse_event.py`, shared by both training and eval):
apply the sampled force as before, but explicitly schedule it to clear back
to zero after `IMPULSE_DURATION_S=0.2` (the midpoint of Hwangbo/isaaclab-go2-
locomotion's 0.15-0.25s Impulse duration) via a second EventTerm
(`clear_expired_push_impulses`) registered at a degenerate `(step_dt,
step_dt)` interval, so it fires every `env.step()` and zeroes any push whose
duration has elapsed -- tracked via a `env.push_clear_step` buffer (same
"stash state directly on env" pattern as `env.push_curriculum_k_c` /
`env.push_angle_log`, for the same underlying reason: Isaac Lab deep-copies
EventTermCfg params, so a tensor passed that way can't be written back to by
the term function). `curriculum.py`'s own force/torque-range and interval
scaling logic is unchanged -- only the underlying applied force is now a
true short impulse instead of persisting until the next trigger.

## Interval curriculum: raised the dense floor, and saturates early

Two related fixes made alongside the cadence fix, both prompted by the same
investigation (a real 600-iteration Flat run showing poor recovery even at
0 push force — i.e. the problem wasn't push-recovery specifically, it was
basic walking under the disturbance regime):

- **`DENSE_INTERVAL_RANGE_S` raised from `(3.0, 5.0)` to `(5.0, 7.0)`.**
  Pushing every 3-5s at full curriculum difficulty was structurally squeezing
  out the training time needed to consolidate ordinary walking, not just
  push-recovery. Our own design choice (this interval curriculum was never
  literature-derived in the first place — Miki et al.'s exact interval, if
  published at all, is in supplementary material we couldn't verify), not a
  paper-verified figure.
- **`INTERVAL_SATURATION_K_C = 0.5`**: the interval curriculum reaches its
  dense floor once `k_c` crosses this fraction of its own range, instead of
  continuing to interpolate all the way to `k_c=1.0` like force/torque do.
  Force still ramps the full literature-grounded Hwangbo curve to 1.0 across
  the whole run; only the interval's approach to its (now less extreme)
  floor saturates early, so the back half of training happens at a stable
  push frequency rather than one still slowly changing throughout.

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

## Stair sub-terrains rescaled for Go2's body size (pyramid_stairs/pyramid_stairs_inv)

Found investigating why Rough's terrain-difficulty curriculum plateaued at
~level 5.4 of ~9 (RUN_001.md) even after an ablation ruled out push
disturbance as the cause. Isaac Lab's stock `UnitreeGo2RoughEnvCfg` already
rescales two of the six sub-terrain types for Go2's smaller body:
`boxes.grid_height_range` and `random_rough.noise_range`/`noise_step`, both
roughly halved from `ROUGH_TERRAINS_CFG`'s ANYmal-scale defaults. It does
**not** touch `pyramid_stairs`/`pyramid_stairs_inv`'s `step_height_range`
(stock `(0.05, 0.23)` m) at all — an oversight in the upstream Go2 adaptation,
not a deliberate choice we could find any reasoning for.

A per-sub-terrain-type push-recovery eval (2026-09-04, `sub_terrain_type`
column added to `push_recovery_eval.py`'s trial CSV for this) confirmed it's
the cause: at terrain levels 4-8, `random_rough` and both slope types
(`hf_pyramid_slope`, `hf_pyramid_slope_inv`) scored 91-97% success;
already-rescaled `boxes` scored 63.5%; unscaled `pyramid_stairs` scored 49%;
unscaled **`pyramid_stairs_inv` scored just 12%** — and was already bad at
level 4 (8%), the *easiest* level tested, not a gradually-worsening decline
like every other type. That's the signature of an unscaled absolute-length
parameter overwhelming the policy at every difficulty, not a genuine skill
ceiling.

**Why stairs need rescaling but slopes don't**: `step_height_range` and
`grid_height_range`/`noise_range` are all absolute lengths in meters —
scale-blind, so the same 0.23m step is trivial for an ANYmal-sized robot and
a serious obstacle for a much smaller Go2. `slope_range` (used by both
`hf_pyramid_slope*` types) is a dimensionless ratio (rise/run) — scale-
invariant by construction, which is exactly why the slope types never showed
this problem despite also never being explicitly rescaled for Go2.

**Why `pyramid_stairs_inv` (12%) is so much worse than `pyramid_stairs` (49%)
despite sharing the identical `step_height_range`**: a legged-locomotion
asymmetry, not a scaling bug. `pyramid_stairs` is ascending — a leg reaching
for a higher surface makes contact as expected or slightly early, a forgiving
failure mode. `pyramid_stairs_inv` means stepping *down* into a stairwell —
without vision (this policy is proprioceptive-only), a leg reaching for a
step lower than expected free-falls further before contact, producing a
bigger, less-recoverable impact. This compounds with the same oversized step
height rather than being a separate cause.

**Fix** (`core/go2_rough_env_cfg.py`): rescaled both stair types'
`step_height_range` by the same 0.5x factor stock's own `boxes` rescale used
(`(0.05, 0.2) -> (0.025, 0.1)`), giving `(0.025, 0.115)` from the stock
`(0.05, 0.23)`. A reasoned estimate following that established pattern, not
independently verified — flagged the same way `PEAK_TORQUE_NM` is elsewhere
in this doc. Needs a real Rough retrain to confirm it actually lets
`terrain_levels` climb past the ~5.4 plateau; not yet run.

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
