# Session 2 queue ledger

Order per ADDENDUM 8 (arm first). Status values: NOT_ATTEMPTED / RUNNING / DONE / BLOCKED.
Spend is logged per item in outputs/session_log.md; VM ledger in ~/cerberus-results/hyperstack-s2/ledger.jsonl.

| # | item | status | evidence | time/cost |
|---|---|---|---|---|
| S0 | preflight + VM + driver 595.58.03 + host_doctor + bootstrap + safety nets | DONE | L40 1177066, 595.58.03 after nouveau blacklist; pull+scene done; dead-man+heartbeat tested | 07:28-08:00Z, ~0.9 USD |
| K3 | crash-tolerant harness (markers, relaunch, liveness, fake-job test) | DONE | run_until_complete.py + atomic_io + 6 tests incl. random-crash fake job; used on the VM | 08:10Z |
| P2a | arm embodiment bring-up in scene 1, assertions, stowed arm | DONE | platform registers; action term resolves the 12 contract joints in contract order (live log); 20-DoF articulation accepted by the assertions; eyebrow: scene stand with the arm body blocked by the host's masked exits | 09:43Z |
| P2c | arm locomotion retrain started in tmux (target: <2 h after VM) | DONE | model_1499.pt, 1500 iters, 4096 envs, seed 42, 3109 s; R2 migration/session2/checkpoints/go2_arx5_ga_v1; arm randomisation events off (PhysX assert) | 09:43Z, ~45 min GPU |
| P0A | look: material root cause, MDL_SYSTEM_PATH, 5 stills + acceptance | BLOCKED | root cause proven (3,282 MDL nodes, 0 missing files, MDL_SYSTEM_PATH empty); with the path set the scene build dies at ~149 s (exit 0, no stills); without it, flat colour. No still passed the acceptance checks | 10:05Z |
| K1 | classify exit-0 failures (masked crash vs clean), PR #6636 | DONE (evidence) | classifier tool + dumps/Xid evidence in docs/reports/kit_crash_report.md; PR not applied to the image | 09:30Z |
| P2b | compatibility check: legs-only policy on arm robot (fall rate, tracking) | BLOCKED | needs a working camera-less physics run in the scene; runs that reached the scene build exited 0 without artifacts (same masked-exit failure), so no fall-rate number exists | 10:10Z |
| P0B | cameras EGO/CHASE/OVERVIEW + pose logs + clip|QA gate | NOT_ATTEMPTED | | |
| P0C | first arm clips (EGO, CHASE, OVERVIEW) + close-up, QA gate | BLOCKED | same as P0B; `scene_clip.py` never reached its camera rig | 10:10Z |
| P0D | arm showcase clip: reach, wrist turn, gripper, CHASE + close-up | BLOCKED | same as P0B | 10:10Z |
| P2d | retrain eval + hero clip re-render with retrained policy | BLOCKED | checkpoint produced and published (R2 + laptop); the clip/eval run needs the RTX path that does not start here | 10:10Z |
| P2e | navigation eval N=10 with retrained policy, clips | BLOCKED | same; no physics launch completed with artifacts | 10:10Z |
| P3 | household mission in real scene (grasp, chain, N=10, showcase renders) | BLOCKED | code exists (planning/*, scripts/scene_arm_grasp.py) but the physics mission cannot render or run on this host; the kinematic chain is the upper bound reported in N1 | 10:10Z |
| N1 | IK pick/place skills, props, kinematic N=10 mission, top-down animation | DONE | planning/{skills,props,task_orchestrator,kinematic_mission}.py + 9 tests; N=10 on the real grid: 4/10; 1080p30 animation | 09:55Z |
| N3 | JEPLO spike (own venv, MuJoCo EGL, clip, report) | BLOCKED | does not build: the authors' unitree_mujoco needs both `user_images_new_` (MuJoCo 3.3.x) and `Simulate::render_callback` (absent in 3.3.6); 3.1.6/3.2.7/3.3.6 tried. docs/reports/jeplo_spike.md has the exact point, the 60x25 depth format, the command/sensor delta and the adapter estimate. Licence left isolated in ~/jeplo | 09:25Z |
| N2 | SLAM: FAST-LIO2, ORB-SLAM3, APE/RPE, map animations | NOT_ATTEMPTED | no time box left after the arm-first items; rosbags untouched, tooling in docker/slam and scripts/fetch_slam_backends.py unchanged | 10:10Z |
| P1 | legs-only nav eval N=5 (optional baseline, only after P3) | BLOCKED | P3 did not run; the nav path needs the same physics launches that masked-exit here | 10:10Z |
| K2 | A/B reliability matrix V0-V4 | BLOCKED | tooling + probe written; V0's first run measured a probe bug, corrected run cancelled to protect the retrain; V3 needs NGC_API_KEY (absent). docs/reports/kit_reliability.md | 09:35Z |
| K4 | Kit crash report (not filed) | DONE | docs/reports/kit_crash_report.md | 09:40Z |
| P5 | optional: Isaac Lab Mimic + BC | NOT_ATTEMPTED | optional, gated on everything above; not reached | 10:10Z |
| N4 | docs/showcase.md | DONE | written, filled with what exists; clips to be added | 09:30Z |
| P6 | packaging: release draft, README, tests, lint, push | DONE | draft release `showcase-2026-11-23` (kinematic animation, gated); README Showcase section; docs/showcase.md, arm_bringup.md, kit reports, jeplo report; tests 1436 passed (1 pre-existing ROS-msg failure, 1 pre-existing slam import error); lint clean; main pushed to 0f259e1 | 10:20Z |
