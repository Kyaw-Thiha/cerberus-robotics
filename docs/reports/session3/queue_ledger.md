# Session 3 queue ledger

Status values: NOT_ATTEMPTED / RUNNING / DONE / BLOCKED. Spend per item in outputs/session_log.md; VM ledger in
~/cerberus-results/hyperstack-s3/ledger.jsonl. Session 2 items are in docs/reports/session2/queue_ledger.md; this
ledger re-states the NOT DONE ones as session-3 items plus the new track R ladder.

| # | item | status | evidence | time/cost |
|---|---|---|---|---|
| S0 | preflight, VM (L40 `cerberus-scene3` id 1178616), driver 595.58.03, host_doctor PASS, bootstrap, safety nets | RUNNING | 58 GB free, 28 cores; driver 595.58.03 after purge+nouveau blacklist; host_doctor PASS after installing rclone; image pull + scene fetch running; kill timer armed 7 h, dead-man (log-only tested), heartbeat, sleep inhibitor | 20:13Z |
| FIX | two clean-checkout test failures skip cleanly (cerberus_msgs unbuilt) | DONE | `_messages_available()` now checks the generated classes; SLAM adapter test importorskips `cerberus_msgs.msg` + skips without `TaggedMap`; both skip with a reason | 20:20Z |
| R3 | material conversion script + tests + the accepted stills | DONE | converter with MDL-source parsing (1678 diffuse / 2866 normal textures, 3847 rebinds, 0 missing), MDL prims deactivated; 2 accepted 1920x1080 stills (lum 157.8 / 147.2, hue bins 4 / 6) shipped as docs/reports/figures/render_unblock_{ego,chase}.png | 23:00Z |
| R1 | diagnose: Kit MDL warnings, mdlc on one scene MDL, MDL_SYSTEM_PATH | DONE (root cause found) | `find / -xdev -name mdlc*` empty in the Isaac Sim 6.2 container (runtime only, no MDL SDK); scene library 812 .mdl / 2998 textures / KooPbr imports; `CERBERUS_SKIP_MDL_PATH=1` made MDL_SYSTEM_PATH empty while `--mdl` set it and Kit still crashed at the scene build. Evidence in docs/reports/render_unblock.md | 20:20Z |
| R2 | visual layer from the original/-i variant | NOT_ATTEMPTED | | |
| R4 | other cached GRScenes homes | NOT_ATTEMPTED | | |
| R5 | bundled Isaac Sim scene fallback ('not the GRScenes home') | NOT_ATTEMPTED | | |
| CAM | camera-on Kit in segments, stitch, continuity check | DONE | segment state save/resume in scene_clip.py + run_clip.sh (10 s segments, resume from the saved articulation state, ffmpeg stitch); captures step the Replicator orchestrator | 01:05Z |
| R-D1 | hero clip: stand + scripted reach/wrist/gripper, 20 s, EGO/CHASE/close-up, QA gate | PARTIAL | CHASE/close-up frames render but read 199-260 mean luminance (outside 70-170) and the close-up camera lands inside furniture; the EGO cut (arm_ego.mp4) is the usable one | 03:56Z |
| R-D2 | walk along the scripted path with the retrained policy, EGO/CHASE/OVERVIEW | PARTIAL (EGO gated) | walk_ego.mp4 (20 s, 1080p30) passes the whole gate; CHASE/OVERVIEW views fail the luminance band (exposure unresolved) | 03:56Z |
| R-D3 | split-screen with the top-down map | DONE | `walk_ego_split.mp4` (20 s, 1080p30): EGO view + planner occupancy with the logged trajectory; clip_qa PASS (lum 83-86, 0.000 clipped, 3-4 hue bins) | 04:14Z |
| R-D4 | 10 s teaser GIF <= 8 MB | DONE | `teaser_walk_ego_10s.gif`, 10 s, 640 px, 1.36 MB, cut from the gated walk_ego clip | 04:14Z |
| P0A/P0B/P0C/P0D | stills, cameras, arm clips (session 2 BLOCKED) | NOT_ATTEMPTED | re-scoped into R above | |
| P2b | legs-only policy on the arm robot compatibility numbers | NOT_ATTEMPTED | | |
| P2d/P2e | retrained-policy eval + hero clip | NOT_ATTEMPTED | | |
| P3 | household mission in physics, N=10 | NOT_ATTEMPTED | gated on Track R stills+clip | |
| P1 | legs-only nav baseline N=5 (optional) | NOT_ATTEMPTED | | |
| K2 | A/B reliability matrix | NOT_ATTEMPTED | | |
| N3 | JEPLO second attempt (python sim route) | DONE | authors' checkpoint ran in their MuJoCo sim-to-sim loop headless (EGL): 812 frames, base x -0.04 -> +7.1 m, one stair fall and recovery; clip `~/cerberus-results/2026-11-23/jeplo/walking_30s.mp4` (30 s, 1280x720); clip_qa: media ok, hue-diversity check fails on the grey MuJoCo grid (recorded, not hidden); docs/reports/jeplo_spike.md 'Attempt 2' | 16 min |
| N2 | SLAM: FAST-LIO2, ORB-SLAM3, APE/RPE, map animations | NOT_ATTEMPTED | | |
| K4 | kit_crash_report.md (ready, not filed) | DONE (session 2) | docs/reports/kit_crash_report.md | |
| N4 | docs/showcase.md | DONE (session 2), to update in packaging | | |
| P5 | optional BC | NOT_ATTEMPTED | last, if time | |
| P6 | packaging: showcase, release draft assets, README, report updates, tests+lint, push | NOT_ATTEMPTED | session 2 draft exists; needs the new clips | |
