SESSION 2 MASTER HANDOFF (v5). Supersedes every earlier handoff and addendum,
including the previous session's.

1. AUTONOMY CONTRACT
The owner is away for about 10 hours with the terminal and laptop open and gives
NO input. Never ask a question, never use any ask-the-user tool, never wait for
approval, confirmation or a reply. When a decision is needed, choose the option
that (1) cannot exceed the limits in section 3, (2) loses no data, (3) keeps the
session moving; log it as one line with a timestamp in outputs/session_log.md.
Output hygiene: never cat large files or logs (use tail, head, grep -c, wc); keep
chat updates to one line.
FIX LADDER for every problem, inside the item's time box: (1) DIAGNOSE: read the
logs, state the cause in one line; (2) FIX: minimal fix in OUR code or config,
update or add a test when code changes, rerun; up to 3 attempts per problem,
each testing a different hypothesis; (3) ALTERNATIVE: the documented fallback in
section 13 or another route to the same goal; (4) SKIP AND CONTINUE: record
exactly what happened, mark the item BLOCKED in the ledger, START THE NEXT ITEM
IMMEDIATELY. Never idle, never loop on the same failure.
SECOND PASS: after the first pass, with at least 30 percent of the budget left
and VM age under 7.5 hours, return to BLOCKED items by importance (P3 mission,
P2 retrain, P0 look, P1) and try again with a DIFFERENT approach in a fresh
45-minute box.
NO EARLY END: the previous session ended with most of the queue never attempted.
Keep outputs/queue_ledger.md (item, status DONE / BLOCKED / RUNNING /
NOT ATTEMPTED, evidence, time and cost) and update it after every item. You may end
the session ONLY when every ledger item is DONE, or BLOCKED with evidence of two
different fix attempts plus one alternative, and the second pass is done, or when
a hard limit is reached. Failure of Kit is NOT a reason to stop: do the items that
do not need Kit. Before wrap-up print the ledger; every NOT ATTEMPTED item must be
attempted first. Never pretend: report every failure, every re-run and every label
('legs-only policy', 'magic grasp', 'kinematic') plainly.

2. CONTEXT AND SCOPE
Last session reached T1: a plain-SSH Hyperstack L40 ran the real GRScenes home;
Kit started; the scene-1 stand test passed 8/8 contract assertions; a 90 s 1080p
RTX clip of the GA policy exists but is REJECTED (the whole scene renders as one
flat red-orange, so materials are broken, and the camera was fixed, not following
the robot); never use that clip in the showcase. Kit started only about 1 launch
in 10, and runs 'exited 0' after the sim loop: the working hypothesis is that these
are one bug, a random Kit SIGSEGV (libcarb.tasking / make_fcontext, known on Isaac
Sim 6.0.1 and recommended to try on driver 595.58.03), reported as exit 0 by
Isaac Lab's AppLauncher. The camera runs only stayed alive on driver 595.58.03.
Eight bugs in our code were fixed; scripts/host/hyperstack_*.sh exist with caps.
Robot: from P2 on, the robot in EVERY run is the Go2 WITH the ARX5 arm
(GO2_ARX5_CFG); the legs-only run (P1) is the control baseline. Dropped by the
owner: the adversarial gate, rough-terrain and push-recovery re-renders (Phase 0
already has them; never delete or overwrite them), the Newton lane, any
imitation-learning work before the mission is done.

3. HARD LIMITS (never exceed; the only terminators)
Total spend this session: 14 USD (estimate hours x listed hourly price incl.
public IP and disk; log at every stage). VM at most 1.50 USD/hour; at most 9
hours from creation; at VM age 8 hours start packaging (P6) regardless of the
queue; at most 3 VM creations, one live at a time. If the remaining budget is
under 25 percent of the cap, stop starting items and go to P6 and the wrap-up.
Touch no cloud resource you did not create. No tuning of policies, planners or
SLAM parameters (fixing OUR bugs is not tuning). The ONLY training allowed: the
arm locomotion retrain (P2) and, if P5 is reached, behaviour cloning. Do not
delete R2 objects (SLAM rosbags, Phase 0 clips and checkpoints stay). Do not
rewrite git history. Videos stay out of git. Never commit third_party/arx-model
(no license file) or generated embodiment assets.

4. SECRETS
All keys in .env were rotated by the owner. Read HYPERSTACK_API_KEY, R2_* (and
NGC_API_KEY if present) per command in a subshell (bash -c 'set -a; . ./.env;
set +a; ...'); verify non-empty WITHOUT printing (grep -c '^NAME=.' .env); never
print .env; never run env/printenv; never use curl -v or rclone -vv/--dump with
secrets; pass any output that could echo config through scripts/redact.py. The
restricted R2 token and the Hyperstack key may be on the VM only as chmod 600
root-owned files used by the dead-man and sync loops, and are removed at wrap-up.
docker login to nvcr.io (user $oauthtoken) only per command, never printed. The
final report tells the owner to rotate all keys again.

5. SAFETY NETS (never zero kill switches while the VM exists)
a. Laptop-side systemd user timer at VM creation + 9 h 15 min running
   scripts/host/hyperstack_kill.sh <vm_id>; verify with systemctl --user
   list-timers; when replacing a timer arm the NEW one first, then cancel the old.
b. VM-side dead-man (systemd service + timer, every 5 minutes): if
   /var/lib/cerberus/heartbeat is older than 45 minutes, or the clock is past
   creation + 9 h 30 min: rclone-copy the results and training directories to R2
   (10-minute timeout), then delete THIS VM through the Hyperstack API using its
   own id. A laptop-side service (systemd-run --user --unit=cerberus-heartbeat)
   touches that file over SSH every 60 seconds. Test the dead-man in a
   logging-only mode before relying on it.
c. Keep the laptop awake: background 'systemd-inhibit --what=sleep:idle sleep
   infinity'; stop it at the end.
d. Every 10 minutes while a long job runs, sync its output directory to R2 so
   deletion never loses checkpoints or clips.
e. SSH is restricted to the laptop's current public IP /32 in the Hyperstack
   firewall; the home connection is behind carrier-grade NAT, so if SSH fails
   check curl -s ifconfig.me and update the rule; this is not a failure.
f. Run every long job on the VM inside tmux (or nohup).

6. STATE, RESUME, PUSH
After EACH item write outputs/session_state.md (VM id, public IP, firewall rule
id, timers and services, items DONE / BLOCKED (with reason and what was tried) /
RUNNING (tmux names) / NEXT, spend so far) and append results to
docs/reports/first_scene_session.md; commit docs and push main (tests and lint
green; fast-forward; per-command gh helper over HTTPS: git -c credential.helper=
-c 'credential.helper=!gh auth git-credential' push
https://github.com/Kyaw-Thiha/cerberus-robotics.git main:main; never force; if a
push or the tests fail, fix or skip the push, continue, retry at the end). A fresh
agent must be able to continue from this handoff, the repo, the state file and R2.

7. STEP 0 (PREFLIGHT AND VM; nothing here may stop the session except an unusable
VM, which you replace)
Check non-empty keys (no printing), gh auth status, laptop disk, systemctl --user
list-timers, and whether a VM named cerberus-scene* exists (use
scripts/host/hyperstack_status); the previous VMs were deleted. Create a VM within
the limits (L40 or A6000 48 GB, at least 200 GB disk, an image with the newest
NVIDIA driver, SSH key pair and /32 firewall rule via the scripts/host wrappers),
install driver 595.58.03 from NVIDIA's runfile (the driver on which the camera
runs stayed alive; it is in the allowlist and on NVIDIA's Isaac Sim 6.2
requirements page), reboot, host_doctor must PASS, then bootstrap (public image
from docker/remote_v6, scene cache, GA checkpoint set from docs/storage.md; run the
image pull and the scene fetch in parallel; reuse R2 caches). Arm all safety nets.
While the VM bootstraps, START TRACK N on the laptop (no GPU needed).

8. PRIORITY ORDER AND CONCURRENCY
Order of importance when time is short: P0 look and camera QA; K3 harness; P2 retrain
started early; P3 mission; N1 kinematic mission video; N3 JEPLO; N2 SLAM; P1 rest;
K2/K4; P5. Suggested flow: Step 0 -> K3 (30 min) -> P0 stills and K1 -> P2 retrain in
tmux while N-track and P3 code proceed -> K2 matrix when the GPU is otherwise idle
(never measure crash rates while other Kit jobs run) -> P1 rest and P3 runs with the
retrained policy -> P6. Run independent items concurrently when GPU and CPU allow
(L40 48 GB; pause SLAM/JEPLO if the retrain slows by more than 20 percent) and say
so in chat.

9. TRACK K: KIT RELIABILITY (60-minute box; K3 first)
K3 CRASH-TOLERANT HARNESS (30 min, do first): success is judged by completion
 markers and output files, NEVER by exit code. Every unit of work (trial, pair,
 clip, still) writes its output atomically when it finishes; the wrapper relaunches
 until all units exist (up to 60 launches per job); a run that finishes 'cleanly'
 without its marker counts as a crash; launch rule: kit_retry plus the liveness
 rule (hung = no Kit log growth for 4 minutes AND CPU under 5% AND GPU idle; wait
 up to 5 minutes; continue as soon as ONE launch reaches Startup Complete); the
 training wrapper saves a checkpoint every 50 iterations and relaunches with resume
 from the latest on a crash. Test with a fake job that crashes randomly.
K1 CLASSIFY: for the last failures and for new ones, check the Kit log for
 '[Fatal] [carb.crashreporter' lines, dump files (find ~/.nvidia-omniverse/crashes,
 /tmp and the working dir for *.dmp), coredumpctl and dmesg; label each 'exit 0' as
 masked crash or clean exit, with counts. If it applies cleanly to the Isaac Lab ref
 pinned in the image, apply the AppLauncher exit-status fix from upstream Isaac Lab
 PR #6636 in the image only (truthful exit codes); never commit it here.
K2 A/B MATRIX: a minimal launcher (stand loop for 300 steps with a camera on, then
 60 s of rendering), 15 launches per variant, alone on the GPU: V0 baseline on
 595.58.03; V1 under 'taskset -c 0-7'; V2 with PXR_WORK_THREAD_LIMIT=1 and
 OMP_NUM_THREADS=1; V3 (only if NGC_API_KEY is set) NVIDIA's official container
 nvcr.io/nvidia/isaac-sim:6.2.0 running an equivalent minimal SimulationApp loop;
 V4 cameras off versus on for metric runs. Record starts ok/15, completes ok/15,
 median seconds to crash. Pick the most reliable variant and use it for everything
 after; write docs/reports/kit_reliability.md (numbers) and the recommended default
 into docs/dev_machine.md.
K4 BUG REPORT: docs/reports/kit_crash_report.md, ready to file on isaac-sim/IsaacSim
 but NOT filed: environment (GPU, driver, OS, install, versions, image tag), minimal
 reproducer, the A/B numbers, representative [Fatal] lines and stack frames, evidence
 for 'masked crash reported as exit 0', mitigations that helped, link to the NVIDIA
 forum thread on random SIGSEGV in libcarb.tasking on Isaac Sim 6.0.1; redacted, at
 most two pages.

10. P0 LOOK AND CAMERA QA (box 90 minutes; DO BEFORE ANY CLIP; work on single stills,
about 20 s each)
A. LOOK. (1) Grep the Kit log of the last scene run for 'Unable to resolve', 'MDL',
 'MDLC', 'texture', 'Failed to' and count them. (2) docs/worlds.md: GRScenes needs
 MDL_SYSTEM_PATH; set it to the root containing the scene cache's Materials folder
 and make sure the render process, its children and tmux inherit it. (3) With pxr on
 the VM, inspect the scene USD that the run loads: count Material prims; for several
 print the bound shader inputs (texture/MDL asset paths and colour constants) and
 whether those files exist on disk; compare the preprocessed navigation variant with
 the original or the interaction (-i) variant if present, and choose the variant that
 renders with textures for the VISUAL layer (physics may keep the navigation
 variant). (4) Render stills from 5 viewpoints (living area, a corridor, the room
 with the washing machine, two others); add a dome light and, if needed, a few
 rect/distant lights and exposure until the checks pass; keep a fixed small set of
 light settings and report them. (5) STILL ACCEPTANCE: mean luminance 70-170 (0-255),
 at most 5 percent of pixels clipped at 255, at least 3 hue bins (30 degrees) each
 holding at least 5 percent of the pixels (a single-colour scene FAILS), no
 unresolved-material warnings for visible materials. (6) Open every accepted still
 and write one sentence per image in outputs/session_log.md on the floor, walls and
 furniture colours you actually see.
B. CAMERAS (all moving cameras rigidly attached to the robot): EGO camera on the
 base link at the front-head position (about 0.30 m forward, 0.08 m up, 15-20 degrees
 down), horizontal FOV about 90 degrees, 1920x1080 (1280x720 if needed), 30 fps,
 recorded from the sensor; CHASE camera parented to a yaw-only frame of the base, eye
 about 1.6 m behind and 0.9 m above, 10-15 degree downward pitch, low-pass on yaw;
 OVERVIEW camera fixed and wide, labelled 'fixed overview' wherever it appears. Log
 the world pose of each camera and the base for EVERY frame to JSON/NPZ next to each
 video. The hero clip cuts EGO, CHASE, OVERVIEW (about 20 s each).
C. CLIP QA GATE (every video, before it counts as DONE; script
 scripts/host/clip_qa.py with unit tests on synthetic frames): ffprobe; extract 5
 frames at 0, 25, 50, 75 and 100 percent; apply the still checks to each frame; for
 EGO and CHASE verify from the logged poses that the pose changes over time and the
 camera-to-base transform is constant within 2 cm and 2 degrees (yaw-only for CHASE);
 for OVERVIEW verify the pose is constant; verify the robot is visible in CHASE frames;
 look at the 5 frames yourself and describe them in outputs/session_log.md. A clip that
 fails any check is not DONE: fix the cause and re-render; report the first render and
 the re-render honestly.
D. Re-render the legs-only scene demo (90 s, EGO/CHASE/OVERVIEW cuts) and pass the gate
 before anything that depends on the look. If materials cannot be fixed within 60
 minutes, render the best variant with the dome light and say so in the showcase
 ('scene shown without its original materials'), and carry on.
E. Record in the ledger: root cause found (material resolution, variant, light,
 exposure) with evidence and the settings used.

11. TRACK N: NO-KIT ITEMS (start immediately; must not wait for Kit)
N1 MISSION CODE AND KINEMATIC MISSION (laptop, no GPU; box 2 hours): scripted IK pick
 and place skills (differential-IK end-effector waypoints: pre-grasp, descend, close,
 lift, stow; place: approach, lower, open, retract) replacing the stub PickSkill and
 PlaceSkill in planning/task_orchestrator, with a mocked robot and unit tests;
 failure detection (cube not lifted, cube dropped, timeout) and retries via the
 orchestrator's retry logic (failure injection OFF); a prop table (box about 0.6 x
 0.6 x 0.5 m) with a 4 cm cube on top in free space in scene 1 and a second prop
 surface several metres away as the place target, both added (inflated) to the
 planner's occupancy map, with an approach waypoint about 0.6 m from the table edge
 facing the cube; the chain navigate -> pick -> navigate -> place in the kinematic
 simulation on the real scene-1 grid, N=10 seeded trials; a top-down animation video
 of one full mission (matplotlib + ffmpeg) labelled 'kinematic simulation, scripted
 skills'.
N2 SLAM (35-minute box per backend, 2.5 hours effort in total): read
 docs/slam_design.md, the slam package docs and docs/storage.md (do NOT delete the
 rosbags); existing rosbags and the docker/slam setups; order FAST-LIO2 (LiDAR
 only), ORB-SLAM3 camera-only, then other backends whose images build inside their box;
 against ground truth: APE and RPE, a trajectory-vs-ground-truth plot, and a 30-40 s
 map-building animation per backend (top-down map accumulating, estimated trajectory
 in colour, ground truth dashed). If the bags do not cover a configuration and P3 is
 done, record one LiDAR + IMU (+ stereo) bag from the arm-robot mission run (60-minute
 box). A backend that fails to build or diverges is a valid documented result.
N3 JEPLO (MUST; 75-minute box; license GPLv3): clone ASIG-X/JEPLO into a directory
 OUTSIDE the repository (for example ~/jeplo on the VM), never copy its code into
 this repo, never add its dependencies to pixi.toml. Run its released pretrained
 checkpoint in its MuJoCo sim-to-sim demo headless (EGL) in its own venv with its
 own pins; record commands, timings and a 20-30 s clip (1080p if possible; apply the
 clip QA gate). Write docs/reports/jeplo_spike.md: does it run yes/no and why not; its
 input (LiDAR model and scan pattern, how the 60x25 spherical depth image is built),
 command interface versus our LocoCommand, robot asset and joint order versus our Go2
 contract, Isaac Lab and torch pins versus our Lab 3.0 stack, checkpoint size,
 inference rate; and a one-page estimate of the adapter work (sensor simulation,
 observation assembly, license isolation). Do NOT change the card's support level;
 state that this is the authors' demo, not our pipeline.
N4 PACKAGING SKELETON: start docs/showcase.md now and fill it as items finish.

12. TRACK P: KIT ITEMS (only through the K3 harness and the K2 winner)
P1 LEGS-ONLY CONTROL REMAINING (box 90 minutes): scripted smoke; the navigation eval
 (N=10, reduce to 5 if needed) with success and failure clips (EGO/CHASE/OVERVIEW,
 clip QA gate); replay clips; these numbers are the baseline. The stand assertions
 already passed.
P2 ARM ROBOT AND MANDATORY RETRAIN (box 2.5 hours; start the retrain as early as
 possible; it is the longest GPU job):
 a. Make the scene env and the stack take the embodiment as a config (legs-only stays
    available). The policy observes and acts on the 12 LEG joints by name with the
    SAME 235-D contract so the ONNX backend still applies; arm joints are held by
    position targets at the stowed pose, plus the differential-IK controller for
    manipulation. Assert joint names and order against go2_arx5_spec.py, base height,
    masses, no NaN. Regenerate the fused assets on the VM from the committed xacro
    and a private copy of third_party/arx-model if missing; transfer only through the
    PRIVATE R2 bucket under embodiment/, never git.
 b. Compatibility check (20 min): the legs-only GA policy on go2_arx5, arm stowed, 64
    envs, 0.5 m/s, 20 s, flat ground: fall rate and tracking error next to the
    baseline, plus a short clip.
 c. MANDATORY RETRAIN: locomotion on go2_arx5 with the SAME recipe and hyperparameters
    as the GA retrain (Cerberus rough task, 4096 envs, 1500 iterations, seed 42, same
    terrain curriculum and push events), arm stowed with randomised arm joint offsets
    and payload mass per docs/cerberus_project_context.md. Check s/iteration at 10
    minutes; if the projection exceeds 100 minutes reduce the iteration count to what
    fits and say so. Run it in tmux under the resume wrapper; use the legs-only policy
    on the arm robot for debugging while it runs. Checkpoint name
    go2_arx5_locomotion_ga_v1_candidate with ONNX, .onnx.data and meta JSON (git hash,
    config hash, iterations, seed, wall clock, final reward, terrain stats); sync to R2
    immediately; never overwrite the legs-only checkpoints. Evaluate with the existing
    eval tooling (small N): fall rate and tracking error next to the baseline, with a
    clip.
 d. With the retrained policy and the same seeds as P1: navigation eval (N=10) with
    success and failure clips; numbers next to the P1 baseline and the kinematic upper
    bound. Kinematic.
P3 HEADLINE: HOUSEHOLD MISSION IN THE REAL SCENE (box 3 hours; use N1's code and
 props; finish with the retrained policy):
 a. Physical grasp first (45-minute box); if it cannot be made reliable, kinematic
    attach of the cube to the gripper on close and release on open, LABELLED 'magic
    grasp' in clips, report and showcase.
 b. The chain through the orchestrator: navigate to the table (A* + pure pursuit through
    the retrained policy), pick, navigate to the second surface, place. Success = cube
    resting on the place surface within 5 cm of the target XY, robot upright, within
    240 s. N=10 seeded trials with randomised prop poses; report mission success,
    per-step success, retries, time and failure modes (with clips).
 c. Showcase renders (RTX, 1080p, H.264, each at most 60 s and 50 MB, all through the
    clip QA gate): (i) the full mission with EGO, CHASE and a close-up of the grasp;
    (ii) split-screen: left the RTX view, right a top-down 2D map with the planned path,
    robot pose, cube state and mission step, synchronised by simulation time
    (matplotlib frames composed with ffmpeg); (iii) one failure clip with an explanation;
    (iv) a 10 s teaser GIF of at most 8 MB.
N1/N2/N3/N4 as in section 11.
P5 OPTIONAL, only if everything above is done and at least 2 hours and 4 USD remain
 (90-minute box): Isaac Lab Mimic on the arm robot: 1-2 scripted demos in HDF5 format
 (fixed base is fine), annotate subtasks, generate 50 demos; report yes/no and exactly
 where it fails (EE actions, FK, RigidObject objects, observation terms). If it works:
 behaviour cloning with Isaac Lab's robomimic scripts at 10, 25 and 50 demos (same
 recipe, fixed seeds, short training), 20 evaluation episodes each, plot success versus
 demonstrations, labelled 'first BC baseline'.
P6 PACKAGING (reserve 30 minutes and 1.5 USD no matter what; start at VM age 8 hours or
 when the budget is under 25 percent): docs/showcase.md: for each clip what it shows and
 what is simulated, ground truth or scripted; metrics tables (navigation baseline vs arm
 robot vs kinematic upper bound; mission results; SLAM; JEPLO status; BC if P5 ran);
 failures stated plainly. Credit GRScenes-100 (CC BY-NC-SA 4.0, non-commercial use,
 link the dataset) and Isaac Sim / Isaac Lab; label JEPLO as the authors' demo with
 their released checkpoint. Check 'gh repo view --json isPrivate' and create a DRAFT
 GitHub release with all clips that passed the QA gate (including the re-rendered demo
 and the kinematic mission animation) and the teaser as assets (gh release create
 showcase-<date> --draft); do NOT publish it. Add a Showcase section to README.md;
 update docs/reports/first_scene_session.md and docs/reports/arm_bringup.md. Tests and
 lint green; push main.

13. FIX PLAYBOOK AND FALLBACKS (inside the ladder)
- SSH fails: compare curl -s ifconfig.me with the firewall rule and update the /32; retry
  with backoff up to 15 minutes; check VM status through the API; if ACTIVE but SSH stays
  dead, reboot it once via the API; after 30 more minutes treat the VM as lost: create a
  replacement (counts as a creation) and resume from R2 and the repository.
- Image pull or scene fetch slow or failing: retry with resume, run both in parallel; if
  not complete 90 minutes after creation, replace the VM (another region/flavor); do the
  laptop-side work meanwhile.
- Kit crash or hang: the K3 harness and liveness rule; after 10 consecutive failures
  check nvidia-smi and the driver, reinstall 595.58.03 once and reboot; if it persists,
  record the Kit logs, run everything that does not need Kit and retry in the second pass.
- Scene path fails for code reasons: fix OUR code (60-minute box per problem); fallback:
  ground-plane scene with the same props, labelled 'no scene'.
- Flat or wrong-looking render: clip_qa catches it; follow P0 (materials, variant, light).
- Disk full: docker system prune of unused images and builder cache and scratch caches
  (never the scene cache, checkpoints or results); check df before every big step.
- GPU out of memory: fewer envs, or cameras off for metric runs.
- Retrain NaN or collapse before iteration 300: rerun once with seed 43, then with half
  the envs; if all fail, continue with the legs-only GA policy on the arm robot (arm
  stowed) and label every result 'legs-only policy'.
- IK or arm control unstable: lower the gains of OUR wrapper, slower waypoints, lock the
  wrist, then a fixed arm pose plus the magic grasp.
- Navigation fails in the scene: log the cause (planner, controller or policy), shorten to
  the nearest reachable pair, report the kinematic upper bound next to it.
- SLAM backend fails to build or diverges: document, move on, retry in the second pass.
- JEPLO does not run: document the exact point of failure; second pass: another route
  from its README or an older pin.
- Hyperstack API errors or rate limits: exponential backoff, at most 10 minutes per call.
- R2 errors: retry; continue locally and re-sync later.
- Tests or lint fail on your changes: fix them; never push broken tests.

14. AFTER EACH ITEM AND HONESTY
Sync to R2, pull videos to ~/cerberus-results/<date>/, run clip_qa on every video, one-line
chat update with the files to watch first, update the ledger and the state file, commit
and push docs. Every clip is a real render of a real run. Label ground truth (pose,
occupancy map), simulator-provided data (height scan) and scripted parts (velocity
commands, pick and place, 'magic grasp' if used). Report the FIRST valid run per clip
and say if you re-ran after a fix.

15. WRAP-UP (always; after the queue, the second pass, or at the limits)
Sync everything, remove the secret files from the VM, stop the dead-man, delete the VM
and verify it is gone through the API, cancel every timer and the heartbeat service, stop
the sleep inhibitor, push the docs, then post a final chat message: the ledger, spend vs
the 14 USD cap, the files to watch first, what failed with evidence and the exact next
step, and a reminder that the owner must rotate the R2 token, the Hyperstack API key and
the NGC key if used.

ADDENDUM 8: ARM FIRST (owner decision; overrides sections 2, 8, 10D and 12 where
they conflict).
1. From the very first render the robot is the Go2 WITH the ARX5 arm
   (GO2_ARX5_CFG). No legs-only clip is rendered for the showcase. The legs-only
   physics runs (the rest of P1) are demoted to the LAST optional item (a legs-only
   navigation eval N=5 with the same seeds, only if time and budget remain after
   P3), used only as the baseline for 'what the arm costs'; otherwise the baseline
   is the existing legs-only stand results and the kinematic upper bound.
2. NEW ORDER after Step 0 and K3: (a) P2a embodiment bring-up immediately: the arm
   robot spawns in scene 1; assertions on joint names/order, base height, masses, no
   NaN; the arm held at the stowed pose by position targets. (b) As soon as P2a
   passes, START THE P2c RETRAIN in tmux under the resume wrapper (target: running
   within 2 hours of VM creation; if it is not running by then, make it the top GPU
   priority and pause other new GPU work until it is). (c) While it trains: P0 look
   and camera QA on the ARM robot (stills first, then clips) using the existing
   legs-only GA policy with the arm stowed, labelled 'policy not yet retrained for
   the arm mass'; run the P2b compatibility check (fall rate and tracking error)
   and, if the robot stays up, render the first arm clips: EGO, CHASE and OVERVIEW
   cuts through the clip QA gate. (d) ARM SHOWCASE CLIP (about 20 s, quick, no
   navigation): the robot stands in a real room and the arm does a scripted reach
   forward, wrist turn, gripper open and close, filmed from CHASE and from a
   close-up camera, through the QA gate. (e) When the retrain finishes: re-render
   the hero clip and run the navigation eval and the mission with the retrained
   policy. Keep both versions of the hero clip (old policy and retrained policy) if
   both pass the gate, clearly labelled.
3. The EGO camera must not be blocked by the stowed arm: if the arm or gripper
   appears in the EGO view at the stowed pose, adjust the camera offset and say so.
   The CHASE and close-up cameras must show the arm clearly; the QA gate must verify
   that the arm links are visible in the CHASE frames.
4. If the early arm clips show the robot falling because the old policy cannot carry
   the arm mass, keep one such clip as evidence ('fall with the legs-only policy')
   and wait for the retrained policy for the hero clips; do not tune anything.
5. Update outputs/queue_ledger.md to this order.
