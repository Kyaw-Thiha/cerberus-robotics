# Adversarial robustness gate: method

Why the gate exists, what the adversary optimises, and how a pass or fail is decided. The results are in
`docs/reports/phase05_adversarial_report.md`; the implementation is `policy/locomotion/adversarial_robustness/`.

## Why

A well-trained quadruped policy that passes ordinary push randomisation can still be broken easily. Shi, Zhang, Miki,
Lee, Hutter and Coros (RSS 2024), "Rethinking Robustness Assessment: Adversarial Attacks on Learning-based Quadrupedal
Locomotion Controllers", ran four standard push-test variants against a push-randomised policy, 1000 trials each, and
found no failures; a learned adversarial policy broke the same controller every time. Passing a random-push evaluation
is therefore evidence about the magnitude that was tested, not about robustness.

The gate exists because the locomotion checkpoint is expensive to change once a navigation policy is composed on top of
it. There is no public code for the method, so everything below is specified from the paper.

## Architecture

The locomotion policy is frozen and sits inside the environment as a component; the policy being trained is the
adversary. Per simulation step:

1. the environment observes robot state (proprioception, the same fields the locomotion policy sees);
2. the adversary, trained with PPO through `rsl_rl`, outputs an attack: a base perturbation force in the x-y plane and a
   velocity-command override;
3. the environment applies both, runs the frozen locomotion policy on the attacked observation, and steps physics;
4. the adversary's reward is computed from the resulting state. The locomotion policy is never updated.

`attack_env.py` is the wrapper environment, `train_adversary.py` the PPO script pointed at it, `eval/gate_check.py` the
reduce-until-survives procedure and `eval/standard_baseline.py` the control condition.

## Adversary specification

**Network:** MLP, 3 hidden layers (the paper's 2-layer case is didactic; 3 suits a push-randomised policy).

**Reward** (paper equations 2-3):

```
adversary_reward = -1_alive + r_aux - lambda * sum_i ||theta_i||_inf

r_aux = c_orient * g_z
      + c_shake  * (w_x^2 + w_y^2)
      + c_torque * sum_j ReLU(|tau_j| / tau_lim,j - 1)
```

- `1_alive` is a constant penalty for every step the robot has *not* failed: it drives the adversary to cause a fall.
- `g_z` is the z component of normalised projected gravity: upright is -1, lying down is +1.
- `w_x`, `w_y` are the base roll and pitch rates.
- `tau_j` is joint torque and `tau_lim,j` its soft limit.
- The Lipschitz term penalises the infinity norm of each layer's weights. It is what keeps the attack smooth and
  physical rather than bang-bang, and the paper's ablation shows removing parts of this reward degrades or breaks
  convergence. The zero-sum alternative (adversary reward equals negative locomotion reward) found no attack at all and
  is not a substitute.

**Failure definition:** Phase 0's own training termination criteria (base contact, tilt limits as configured), never a
separate criterion. The gate compares against the policy's actual trained failure boundary.

**Modalities:** perturbation force plus command override. Single-modality attacks fail against a policy that already has
push randomisation (the paper's ablation), and force reuses the existing push path while the command override reuses the
velocity input. Observation and state-estimator attacks are a documented stretch goal: they need a state estimator with
realistic noise in the loop, which does not exist yet. If the retraining target ever becomes perception-conditioned, that
modality becomes the most relevant one and the gate scope has to be re-read before implementation.

**Rate limiting:** cap the attack's rate of change at 0.1 of its maximum range per simulation step. With the Lipschitz
term, this is what keeps a found attack realisable.

## Gate criterion

No absolute magnitude is copied from the paper: its 15 N, +-0.5 m/s and +-3 degree figures were chosen for a 30-50 kg
ANYmal and are not transferable to a 15 kg Go2.

The criterion is relative. After the adversary converges, reduce the available perturbation range until the locomotion
policy survives (the paper's reduce-until-survives procedure), and record that boundary:

- **pass** if the minimum combined force and command magnitude needed to induce a fall exceeds the push range the
  candidate was trained and evaluated against;
- **fail** if the adversary can break the policy at or below that range, which means the random-push evaluation passed
  because it tested exactly the trained magnitude, not because the policy is robust at that magnitude in combination
  with other realistic factors.

## Robustify and re-attack (only after a fail)

1. Finetune the locomotion policy, warm-started from the candidate's actor and critic, on a mix of rollouts: some under
   the standard domain randomisation, some under the attack that broke it. The paper mixes roughly 5% adversarial
   rollouts; start there and adjust if convergence is unstable.
2. Save it as a new version, never overwriting the candidate.
3. Train a **new** adversary against the new checkpoint (an adversary trained against the old one is not a valid retest)
   and rerun the gate.
4. Repeat until it passes, watching the paper's performance-robustness trade-off: too much adversarial mixing produces
   conservative locomotion, so compare command tracking against Phase 0's metrics and lower the mixing ratio if it
   degrades.

## Checkpoint naming

```
go2_locomotion_v1_candidate.pt      # from Phase 0, the gate's input
go2_locomotion_v1_robustified.pt    # only if the gate failed and one robustify pass ran
go2_locomotion_v2_candidate.pt      # the next version; always _v{n}_candidate, one scheme, not both
go2_locomotion_final.pt             # the checkpoint that passed; the only one navigation may use
```

Intermediate checkpoints are kept, not overwritten: the before-and-after comparison is part of the deliverable.
