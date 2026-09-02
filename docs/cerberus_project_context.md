# Cerberus: end to end quadruped robot learning stack

## Motivation

The founder (3rd year undergrad, robotics focused) has broad existing experience across SLAM, sensor fusion and tracking, motion prediction, generative models for synthetic data, and ROS2 robotics engineering, but zero hands on experience in robot learning specifically: no IL, no RL applied to manipulation, no learned policy execution pipeline. This is also the exact center of the stated PhD interest: IL for data efficient policy learning, MPC and safety filters for reliable execution, continual adaptation for deployment, and world models as a possible unifying substrate. The project exists to close that gap with real implementation experience rather than only paper reading, and to serve as a resume and PhD application asset that is not a copy of existing club work (the founder is already in a self driving car club, so this deliberately targets a different robotics domain).

Secondary goal: general systems engineering experience relevant to a future AI or CV adjacent robotics startup, specifically the failure modes that only show up when you try to make a learned system reliable (sim to real gap, reward hacking, demo collection bottlenecks, policy brittleness at deployment), not just the training itself.

Constraints: no real hardware access, fully simulation based. Compute will be paid for out of pocket on a remote rented GPU (Lambda, RunPod, vast.ai style provider), so cost awareness matters, particularly around hyperparameter search scope.

## Project ethos: synthesis over invention

Cerberus is explicitly not an attempt to contribute a new algorithm to any single subfield (locomotion RL, imitation learning, safety filtering, continual learning, adversarial robustness). Every individual technique used in this project is deliberately borrowed from an existing, published source, credited honestly, and used close to as-published. The actual work, and the actual claimed contribution, is combining multiple published methodologies from different papers and different subfields into one coherent, evaluated, end to end stack on one robot platform — something that (per the novelty framing section below) does not currently exist publicly, precisely because it requires integration effort that academia does not reward and industry does not open source.

This has a concrete implication for how every phase gets scoped: the default move, whenever a phase needs a technique, is to go find the paper that already solved that specific sub-problem well, understand its method well enough to implement it faithfully, and fold it into the stack at the point where it's actually load-bearing — rather than inventing a bespoke approach, or padding a phase with a technique that sounds related but doesn't add a genuinely different capability. The adversarial-attack-based robustness gate (Phase 0.5, see below) is the first concrete instance of this pattern: it exists because a specific paper's own results showed that the more obvious approach (random push testing) misses failures that a purpose-built method catches, and because that method extends the stack's honest claim about robustness rather than merely restating the same test with more knobs.

Two things this ethos explicitly does NOT mean: (1) it is not license to keep adding "one more paper's technique" to any given phase indefinitely — see the scoping discussions per phase for where a technique was considered and deliberately deferred or excluded (e.g. cross-morphology generalization work, vertical/dynamic-terrain perturbation) because it didn't add a load-bearing capability relative to its cost; and (2) it does not mean re-deriving published algorithms from first principles — where a method's original authors published code, that code is the first reference, not a re-implementation from the paper text alone (see the coding-agent handoff docs for which techniques have public reference implementations and which do not).

## Domain and platform decision, with reasoning

Considered platforms: quadruped loco-manipulation, humanoid loco-manipulation, wheeled mobile manipulator, aerial manipulation and drones.

Quadruped was selected. Reasoning:
- Best supported platform in Isaac Lab. Isaac Lab natively covers 11 robot morphologies including Go1/2, ANYmal, Spot, with documented sim to real recipes (e.g. RAI Institute's Spot pipeline) and an existing loco-manipulation demo (navigate, pick, place with one policy stack).
- Less crowded and lower tooling risk than humanoid, while still being a current industry relevant platform (Boston Dynamics, Unitree, RAI Institute are all actively building on this exact stack).
- More technically differentiated than a wheeled mobile manipulator, which is the most common and least novel student project template.

Drones and aerial manipulation were explicitly considered and rejected for now, not because they are uninteresting, but because of a genuine tooling mismatch:
- Isaac Lab and Isaac Sim have no first class drone support. Flight is bolted on via Pegasus Simulator, a third party community extension, multirotor only, with hard version coupling between Isaac Sim version, Pegasus version, and PX4 firmware version. Confirmed via a real project's dev log that explicitly retired their Isaac Sim plus Pegasus stack mid project due to this exact issue.
- The actual industry standard toolchain for drones is different: PX4 or ArduPilot SITL plus Gazebo (Gazebo is PX4's official core team supported default simulator), with AirSim as the alternative for photorealistic Unreal Engine rendering. None of this overlaps with Isaac Lab.
- Conclusion: if a drone project happens later, it should be built on PX4 SITL plus Gazebo plus ROS2 specifically, as its own separate project, not forced into the Isaac Lab stack now.

Platform choice for the quadruped itself: Unitree Go2. Best documented option in Isaac Lab, a public reference implementation exists (isaaclab-go2-locomotion, explicitly described by its author as a portfolio project validating the pipeline end to end, not research) confirming the pipeline runs even on modest consumer GPUs (RTX 4050 6GB was reported sufficient for the locomotion baseline specifically). Also the most affordable real platform if a sim to real story is ever pursued later.

## Novelty framing

Verified via web search: no open source project combines locomotion RL, vision based navigation, SLAM, IL manipulation, a safety filter, and continual learning or world model refinement on one quadruped platform, end to end. What exists publicly is always a slice:
- Multiple public Isaac Lab locomotion only baselines (Go2, ANYmal PPO variants, an RL plus MPC quadruped repo). Confirms Phase 0 is correctly scoped as infrastructure validation, not the novel contribution.
- LOTUS (UT Austin, continual imitation learning via unsupervised skill discovery) is continual IL for a static arm, not a legged mobile platform.
- Peng et al.'s motion_imitation is a quadruped imitating animal reference motions for locomotion style, a different use of "imitation learning" entirely (gait style, not manipulation).
- Various quadruped plus ROS2 stacks exist (SLAM, nav2, teleop) but none touch RL locomotion, IL manipulation, safety filters, and continual learning together.

Why this gap exists despite many robotics companies existing: companies that actually build integrated stacks like this keep them proprietary, since integration quality is the commercial moat, not something they open source. Academia rewards a single novel algorithmic contribution per paper, not cross module integration, so individual components get published and open sourced but the tedious glue work does not. And the skill combination (RL, IL, classical safety and control theory, systems engineering) usually sits across different specialists inside a company, rarely one person doing it purely for portfolio value.

Framing for the README and any public description: be explicit about what is borrowed versus novel. Locomotion is built on Isaac Lab's standard PPO recipe (Rudin et al. 2022, Learning to Walk in Minutes Using Massively Parallel Deep RL). The contribution is the integration itself (this exact combination does not appear to exist elsewhere), plus the evaluation artifacts: a push recovery robustness curve, an adversarial robustness gate on the locomotion checkpoint, a demo efficiency plot across IL algorithms, a safety filter ablation table, and a forgetting curve across sequential tasks. These evaluation artifacts, not the individual trained policies, are the actual deliverables that carry weight.

Important caveat to state honestly wherever this project is described: since there is no real hardware, "real world demo" means a well instrumented simulation demo, not a claim of physical deployment.

## Stack architecture

Six sequential stages, wrapped in a ROS2 middleware integration layer, with a training loop feeding back into the policy layer.

**Perception.** Terrain and obstacle perception from onboard depth and RGB, for navigation. Separately, object level perception (detection plus 6D pose estimation) for manipulation targets, since the manipulation policy needs to know what and where to grasp, not just that something is nearby.

**SLAM.** Persistent mapping and localization, built and maintained during navigation. Specific SLAM approach and tooling not yet confirmed, to be decided later.

**Planning and orchestration.** Global path planning (A*, D*, or RRT family) over the SLAM built map, plus a lightweight task orchestrator, likely a simple state machine, sequencing multi step behavior such as navigate to A, pick up X, navigate to B, place X. Should include basic fault recovery or retry logic (for example, retry grasp on failure, replan on navigation stall), not just a hardcoded happy path script.

**Policy layer.** The learned core. RL for locomotion, IL for manipulation. This is where the hierarchical composition happens: a frozen low level locomotion policy driven by a high level policy that outputs velocity commands conditioned on perception input and a goal, following Isaac Lab's native support for this composition pattern.

**Safety filter.** Sits between policy output and actuation. Two candidate approaches: a QP based control barrier function (CBF) that projects an unsafe action onto the nearest safe action when a constraint such as roll and pitch limits, joint torque limits, or obstacle distance is about to be violated, or a lighter weight shielding approach (Alshiekh et al., Safe Reinforcement Learning via Shielding) where a fallback controller takes over when predicted state crosses a threshold. CBF is more rigorous and more publishable shaped. Shielding is faster to implement under time pressure. Pick one to start, CBF preferred if time allows.

**Robot actuation.** Low level joint control, handled largely by Isaac Lab's existing actuator abstractions.

**Training loop (continual learning plus world model).** Continual learning: introduce new terrains or manipulation objects sequentially, compare naive sequential fine tuning (expected to show forgetting) against a mitigation such as LoRA or adapter based task specific fine tuning (see TAIL, Zhao et al.), structured similarly to LIBERO's lifelong learning benchmark split (Liu et al., LIBERO, Benchmarking Knowledge Transfer for Lifelong Robot Learning). World model: train a diffusion based world model on collected manipulation trajectory data, then refine the IL policy entirely within that learned model without collecting new real or sim demos, following the approach in World4RL (Jiang et al., Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation). This reuses data already collected in Phase 2, which is what makes it feel like a genuine unifying layer rather than a bolted on extra.

**ROS2 integration layer.** Wrap the pipeline as actual communicating ROS2 nodes rather than one monolithic Python script. This is standard practice for real deployed systems, reuses existing ROS2 experience from a prior club project, and is what would make a future port to real hardware plausible rather than a full rewrite.

**Non stack item worth budgeting time for near the end:** a live visualization dashboard showing the SLAM map building in real time, robot state, and current task step. Not a robotics module, but disproportionately important for how convincing a demo video looks to someone who did not build the system.

## Phases

Phase numbers below describe rough sequencing, not fixed contracts. Per the project ethos, each phase's exact technique selection is researched and locked in just before that phase starts, not decided wholesale up front. **Phase 0.5 is deliberately not a numbered phase in the usual sense** — it is a quality gate that attaches to the locomotion checkpoint at a specific point in the pipeline (after training, before that checkpoint is frozen and composed under a higher-level policy), and it stays there regardless of how later phases end up getting resequenced or renumbered.

**Phase 0, locomotion (RL).** Platform: Unitree Go2. Algorithm: PPO via rsl_rl, Isaac Lab's default RL library, on the built in Isaac-Velocity-Flat/Rough-Unitree-Go2 tasks, following Rudin et al. 2022's massively parallel training recipe, terrain curriculum from flat to rough. Depth add beyond the baseline tutorial: random push perturbations during training, with a quantified push recovery success rate reported across terrain difficulty levels. This is infrastructure validation, expect to lean on the public isaaclab-go2-locomotion reference implementation for the base recipe. Deliberately kept simple (standard Rudin-recipe push testing, not adversarial) since this phase's job is proving the pipeline runs end to end, not producing the project's robustness contribution.

**Phase 0.5, adversarial robustness gate (attaches immediately after Phase 0, before the locomotion checkpoint is frozen for composition).** Purpose: standard random-push testing (Phase 0's eval) is known, per published results (Shi et al. 2024, RSS, "Rethinking Robustness Assessment: Adversarial Attacks on Learning-based Quadrupedal Locomotion Controllers"), to miss failure modes that a purpose-built adversarial search finds — in their experiments, four different standard push-test variants found zero failures in 1000 trials each against a policy that a learned adversarial attacker broke with 100% success. This phase exists specifically to close that gap before the Phase 0 checkpoint becomes a load-bearing dependency for Phase 1's composed navigation policy (retraining after composition is expensive; retraining before composition is cheap — this is the entire reason this gate is placed here rather than deferred).

Method: train a small adversarial policy (MLP, 2-3 hidden layers, PPO via the same rsl_rl library already in use) whose job is to find the minimal, realistic perturbation sequence that breaks the frozen-candidate locomotion policy, following Shi et al.'s method. Per that paper's own ablation, single-modality attacks (perturbation-force only, or command only, or observation only) fail against a policy that has already been through standard push-based domain randomization — only combinations of two or more modalities succeed. Scope for Cerberus: **perturbation-force + command-space attack** (both cheap to wire up, reuse existing interfaces — push injection and the velocity-command channel respectively). Observation/state-estimator attack is an explicit stretch goal, deferred by default, since it requires a state estimator with realistic noise characteristics already in the loop, which is a larger prerequisite than the other two.

Gate: the minimum combined force+command magnitude needed for the adversary to induce a fall must exceed the push magnitude range the policy was actually trained and evaluated against in Phase 0. If the adversary can break the policy using perturbations within its own trained range, that's a fail — it reveals that passing Phase 0's random-push eval reflected the specific magnitude tested, not genuine robustness to that magnitude in combination with other realistic factors. If the gate fails, robustify by finetuning the policy against the discovered adversarial scenarios (per Shi et al.'s attack-finetune-reattack loop) and repeat until it passes, before freezing.

Deliverable: a short adversarial vulnerability report (standard-test results vs. adversarial-attack results, side by side), plus a before/after comparison if robustification was needed. This is a first-class artifact in its own right, not just an internal gate — it demonstrates the gap between "passes standard tests" and "is actually robust," which most portfolio-scale locomotion projects don't touch.

Note: no public code exists for Shi et al.'s method (confirmed via search as of this doc). This phase is implemented from the paper's specification directly — see the Phase 0.5 coding-agent handoff doc for the exact reward formula, architecture, and hyperparameter ranges pulled from the paper.

**Phase 1, navigation and perception.** Add SLAM for persistent mapping and localization. Add a global planner over the built map. Compose the Phase 0(.5) locomotion policy as a frozen low level controller under a new high level policy that outputs velocity commands conditioned on onboard depth or RGB input and a goal, using Isaac Lab's native hierarchical policy composition support. Deliverable: robot navigates a cluttered scene from point A to point B using onboard perception and the SLAM map, with a quantified obstacle avoidance success rate.

**Phase 2, manipulation (IL).** Mount a gripper arm (reference the existing Isaac Lab loco-manipulation demo that combines navigation and pick and place on one system). Add object level perception for grasp target detection and pose estimation. Collect demonstrations via Isaac Lab Mimic (teleoperate a small number of demos, let Mimic's subtask randomization generate a larger dataset automatically, which is the practical answer to data efficient IL rather than just a theoretical goal). Train and compare at least two IL algorithms on the same task and demo counts, for example plain behavior cloning as a baseline against Diffusion Policy (Chi et al.) as the strong comparison point, with ACT (Zhao et al., ALOHA) as an optional third point if compute allows. Report success rate versus number of demonstrations for each algorithm, this plot is the actual deliverable, not just a working demo video. Deliverable: navigate to object, pick, navigate to target, place, chained end to end through the task orchestrator.

**Phase 3, safety filter.** Add the CBF or shielding filter (see architecture section above) wrapping the RL and or IL policy outputs. Deliverable: an ablation table, failure rate (falls, collisions, or other constraint violations) with the filter active versus disabled, across a fixed set of evaluation scenarios. The adversarial-attack infrastructure built for Phase 0.5 is directly reusable here — same adversary-training method, retargeted at the policy+filter system instead of the bare policy, filter on vs. off — which upgrades this ablation from "random scenarios" to "adversarially discovered worst-case scenarios," a meaningfully stronger and more publishable-shaped claim. This table is likely the single most convincing artifact in the whole project for demonstrating an understanding of deployment reliability rather than just training a policy that works most of the time.

**Phase 4, continual learning and world model.** Structure the continual learning study similarly to LIBERO's lifelong learning split: introduce new terrains or objects sequentially, compare naive fine tuning against an adapter or LoRA based mitigation, report a forgetting curve (task 1 performance over time as new tasks are added). Separately, train a diffusion based world model on Phase 2's collected trajectory data and use it to refine the manipulation policy in imagination, following World4RL's approach, without collecting new demonstrations. Deliverable: forgetting curve plot, plus before and after policy refinement success rates from the world model step.

Phases 0 through 2 alone already constitute a complete, demoable, resume ready project. Phases 3 and 4 are upside, not a requirement to have something real to show.

## Compute and cost notes

All compute will be paid for out of pocket on a rented GPU. Cost is concentrated almost entirely in Phase 0 and Phase 1 (locomotion and navigation RL), which requires thousands of parallel simulated environments run for millions of steps, and benefits from a workstation class GPU (24GB plus VRAM ideal, though the public Go2 baseline ran acceptably on a 6GB RTX 4050, so it is not a hard requirement). Phase 0.5's adversary training is comparatively cheap — a small MLP policy trained against a fixed frozen environment, converging in a small number of iterations per the source paper's own ablations, not a new large-scale parallel training run. Phase 2 (IL) is comparatively cheap, on the order of hours on a single GPU per algorithm, since demo datasets are small relative to RL environment step counts. Phase 3 and 4 evaluations (safety filter ablation, continual learning runs) are also comparatively cheap, a handful of runs each rather than large searches.

Guidance to avoid overspending on the Phase 2 IL comparison specifically:
- Run a small, cheap hyperparameter sweep (learning rate, batch size) once, on the fastest algorithm (plain BC), to lock in a sane training recipe, then reuse that recipe across the other algorithms with only algorithm specific knobs tuned. Do not re-search from scratch per algorithm.
- If budget is tight, drop to two algorithms (BC versus Diffusion Policy) rather than three. This is still the most informative single comparison (naive baseline versus current state of the art).
- The actual valuable sweep axis is number of demonstrations (for example 10, 25, 50, 100), not hyperparameters, once a recipe is locked in.
- Use a pretrained visual encoder (for example R3M) rather than training vision from scratch, standard current practice, meaningfully cuts training time.

## Reference repos and papers to consult during implementation

- isaac-sim/IsaacLab (official framework repo)
- isaaclab-go2-locomotion (public Go2 locomotion baseline, reference for Phase 0 recipe and expected compute footprint)
- Isaac Lab's documented loco-manipulation demo (navigation plus pick and place on one humanoid system, adapt the composition pattern for the quadruped plus arm case)
- RoboMimic and Isaac Lab Mimic documentation (demo collection and IL training integration)
- Rudin et al. 2022, Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning (locomotion RL recipe; code publicly available as legged_gym, used only as a fallback reference behind Isaac Lab's native task and isaaclab-go2-locomotion)
- Shi, Zhang, Miki, Lee, Hutter, Coros, 2024 (RSS), Rethinking Robustness Assessment: Adversarial Attacks on Learning-based Quadrupedal Locomotion Controllers (Phase 0.5 adversarial robustness gate method; no public code, implemented from paper specification)
- Chi et al., Diffusion Policy (IL algorithm)
- Zhao et al., ALOHA / ACT (IL algorithm, optional third comparison point)
- Alshiekh et al., Safe Reinforcement Learning via Shielding (safety filter, lighter weight option)
- Liu et al., LIBERO, Benchmarking Knowledge Transfer for Lifelong Robot Learning (continual learning benchmark structure)
- Zhao et al. (TAIL), Task specific Adapters for Imitation Learning with Large Pretrained Models (continual learning mitigation)
- Jiang et al., World4RL, Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation (world model refinement approach)
- Wan et al., LOTUS, Continual Imitation Learning for Robot Manipulation Through Unsupervised Skill Discovery (related work, different platform, useful for framing the continual learning section)

## Open decisions

Project name is confirmed as Cerberus, not open anymore.

Most other module level implementation specifics are intentionally not locked in yet. The architecture and phase structure above are settled, but the exact algorithm, library, or technique within each module will be researched and decided when that phase is actually reached, the same way the SLAM approach is being left open for now. Known open items so far:

- SLAM approach and tooling for Phase 1, not yet confirmed.
- Safety filter approach, CBF versus shielding, pick one to start.
- Number of IL algorithms to compare in Phase 2 (two versus three), depends on compute budget as it becomes concrete.
- Whether to build the visualization dashboard as a late Phase 2 or Phase 4 item, has not been scheduled yet.
- Global path planning method (A*, D*, or RRT family), not yet decided.
- Continual learning mitigation method beyond the general LoRA/adapter direction, not yet decided.
- Object level perception approach for grasp target detection and pose estimation, not yet decided.
- Phase 0.5's observation/state-estimator attack modality (stretch goal, deferred by default) — revisit only if the two-modality gate is inconclusive or if time allows after the core gate is implemented.
