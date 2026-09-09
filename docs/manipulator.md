# Cerberus — Manipulation Policy: VLM/VLA Notes

> **Status: provisional.** Everything in this doc is a first pass — findings
> from a research-assistant literature scan done before Phase 2 was reached,
> not a locked architecture decision. Per the project's own ethos ("research
> and lock in a technique just before that phase starts, not wholesale up
> front"), this doc exists to seed that Phase 2 research, not replace it.
> **When Phase 2 research actually happens, it supersedes everything below.**
> Update or delete sections here as better information replaces them — don't
> treat this as authoritative just because it's written down.

## The question this doc is scoping

The task orchestrator (see `cerberus_project_context.md`, Planning and
orchestration) currently plans to decompose a mission into navigate/pick/place
steps. Two open sub-questions:

1. Should that decomposition be driven by a vision-language model (VLM) doing
   high-level planning on top of a separately-trained IL/RL execution policy,
   or by a vision-language-**action** model (VLA) that more directly maps
   perception + language to control?
2. Should a VLA also be one of the compared algorithms in Phase 2's IL
   comparison (alongside BC / Diffusion Policy / ACT), not just a planning
   layer?

## What "VLA on a quadruped" actually looks like today

The honest finding from this pass: no VLA found runs as the tight-loop,
joint-level whole-body controller. Every quadruped-relevant VLA operates as a
slower, higher-level layer that outputs coarse commands (velocities, gait
parameters, waypoint targets), sitting on top of a separately-trained,
faster RL/IL policy that handles actual joint control. This matches the
project's own planned architecture (frozen locomotion policy + IL manipulation
policy, with the orchestrator on top) — a VLA is a plausible drop-in for the
orchestrator's brain, not a replacement for the frozen low-level policies.

**Concrete latency data point:** MobileVLA-R1 (8B-parameter backbone) reports
~10-15 seconds per decision step on an H20 GPU. That's fine for a slow-moving
orchestration layer issuing occasional coarse commands, and nowhere near fast
enough for 50Hz joint control.

## Papers / models found, roughly most to least directly relevant

- **ODYSSEY** (Wang et al. 2025, *Open-World Quadrupeds Exploration and
  Manipulation for Long-Horizon Tasks*). Go2 + ARX5 — same hardware combo as
  this project. Two-model stack: GPT-4.1 decomposes a language instruction
  into navigate/pick/place/push/pull/drag actions over an instance-level
  semantic map (built from onboard RGB+LiDAR); Qwen2.5-VL-72B grounds each
  atomic action to a contact point + end-effector pose. No public code found
  (project page only) — would be implemented from the paper's spec, same
  pattern as Phase 0.5's Shi et al. work.
- **QUART / QUARD** (Ding et al., ECCV 2024). A VLA and benchmark built
  specifically for quadruped locomotion + manipulation. Worth checking for
  public code/weights when Phase 2 research starts.
- **NaVILA** (Cheng et al., NVIDIA/UCSD/USC, RSS 2025). Legged-robot VLA for
  language-guided navigation. Widely cited; the base model MobileVLA-R1
  builds on.
- **MobileVLA-R1** (Peking University, Nov 2025). Built on NaVILA; two-stage
  SFT + GRPO training; real-world deployed on a Unitree Go2. **Code is
  public**: `github.com/AIGeeksGroup/MobileVLA-R1`. This is the most directly
  reusable reference implementation found — quadruped-native, and code
  actually exists, unlike ODYSSEY.
- **CFAM** (Singh et al., Sept 2026, *Continual Field-Adaptive Models for
  Post-Deployment Physical AI*). Benchmarks three general-purpose VLA
  backbones (π0, CogACT, SpatialVLA) against a Go2 among five embodiments,
  for mission-critical intervention tasks. Useful mainly as evidence that
  general-purpose (non-quadruped-native) VLA backbones are a live comparison
  point, and for its framing of post-deployment continual adaptation, which
  overlaps with this project's own Phase 4.
- **WildLMa** (Qiu et al., UCSD/MIT, 2024). Not a VLA in the language-conditioned-
  action-model sense — a VLM/LLM planner over an imitation-learned skill
  library (B1 + Z1, not Go2). Relevant for its skill-library pattern and its
  household-adjacent task set (trash collection, articulated-object opening,
  shelf rearrangement) more than for direct code reuse.

## Current best guess (to be revisited in Phase 2)

- **Orchestrator layer:** replace ODYSSEY's two-model stack (separate LLM
  planner + VLM grounder) with a single quadruped-native VLA — MobileVLA-R1
  is the leading candidate specifically because its code is public and it's
  already validated on the same base platform (Go2). This is a genuine
  upgrade under the project's "borrow the more load-bearing published
  technique" rule, not a random swap.
- **Phase 2 IL comparison:** consider a small or LoRA-fine-tuned VLA as an
  optional fourth comparison point (BC / Diffusion Policy / ACT / VLA) on the
  project's own Isaac Lab Mimic demos, specifically because this ties directly
  to the stated PhD interest in foundation models for embodied AI — not
  included by default, since it adds real compute cost (see below).
- **Compute honesty:** MobileVLA-R1's SFT stage used 4×H20 (96GB) GPUs — too
  heavy for this project's rented-GPU budget as a from-scratch full
  fine-tune. If pursued, scope down to a smaller backbone (Octo-small) or a
  4-bit-quantized LoRA fine-tune of a 7B-class VLA, and run it only for the
  high-level decision layer (low frequency), never as the per-timestep
  controller.

## Explicitly not decided

- Whether a VLA appears at all in the final Phase 2 comparison, or whether
  the orchestrator upgrade (VLM/VLA-driven decomposition) ships without a
  VLA-as-IL-algorithm variant.
- Which specific backbone, if any (this list will likely be outdated by the
  time Phase 2 starts — VLA research is moving fast; re-search before
  committing).
- How the VLA/VLM layer's output interfaces with the task orchestrator's
  fault-recovery/retry logic (not addressed by any source found so far).
