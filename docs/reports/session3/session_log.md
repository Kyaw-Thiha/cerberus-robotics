# Session 2 log (arm first)

All times UTC unless stated. One line per decision.

2026-11-23T07:30Z preflight: keys present (HYP=1 R2=5 NGC=0), gh ok, disk 49G free, no VM, no kill timers
2026-11-23T07:38Z DECISION: pick-image skipped R595 (now hard-restricted to B300); patched pick_image + tests; created VM on R570 image
2026-11-23T07:38Z FIX: pick-image now skips hard flavor-restricted images (R595 is B300-only); tests added
2026-11-23T08:05Z VM 1177066 up, image R570; 35GB pull done, 25GB scene fetched; driver 570->595.58.03 installed, nouveau was holding the GPU (blacklisted, reboot)
2026-11-23T08:12Z K3 harness: scripts/host/run_until_complete.py + atomic_io + crash tests pass; clip_qa.py + 10 tests pass
2026-11-23T08:20Z LOOK ROOT CAUSE: all three scene variants (navigation/raw/interaction) use MDL shaders: 3282
  mdl:surface shader prims, 5013 asset inputs, 0 missing files on disk; MDL_SYSTEM_PATH empty because
  run_session.sh sets CERBERUS_SKIP_MDL_PATH=1. The flat red-orange is an unresolved MDL material, not a broken USD.
2026-11-23T08:25Z K1: launch failures on this host are the loader crash ('A crash has occurred', 14-15 s,
  exit -9) and occasionally a 36-80 s exit 1; crash dumps land in /isaac-sim/kit/data/Kit/IsaacLab/3.0/*.dmp.zip;
  dmesg shows Xid 43 once, after which torch could not see the GPU until a reboot.
2026-11-23T08:30Z FIX (our code, 4 bugs): observation asset_cfg ordering (fused USD is per-leg, the contract is
  per-part -> preserve_order=True with the contract list); event env_ids can be a slice; joint_pos dtype is a
  type() not a torch.dtype; train.py read args_cli.adversarial_env which the parser never defines.
2026-11-23T08:55Z BLOCKED (logged, not tuned): the arm randomisation events (per-episode joint offsets, payload
  mass) trip a PhysX device-side assert at the reset/startup event on this stack. Set/add methods, body_mass source
  and inertia consistency were all tried (3 hypotheses). Retrain runs with CERBERUS_ARM_EVENTS=0: the arm is held
  stowed by its own PD actuators, which is what the policy contract needs; offsets/payload randomisation is not
  applied and is reported as such.

# Session 3 log (render unblock + finish the ledger)

2026-11-23T20:05Z step0: handoff saved to outputs/session3_handoff.md; ledger/state/log/session2 handoff/first_scene report read.
  NOT DONE: P0A look BLOCKED, P0B/P0C/P0D BLOCKED, P2b BLOCKED, P2d/P2e BLOCKED, P3 BLOCKED, N2 SLAM NOT_ATTEMPTED,
  N3 JEPLO BLOCKED, P1 BLOCKED, K2 BLOCKED, P5 NOT_ATTEMPTED; P6 packaging DONE (needs update with new clips).
2026-11-23T20:07Z DECISION: fresh Hyperstack ledger dir hyperstack-s3 (caps 10 USD / 7 h / 3 creations / 1.50 per h); VM cerberus-scene3 creating (L40 bigroot, CANADA-1).
2026-11-23T20:13Z SAFETY NETS: laptop kill timer 2026-11-23 22:07 EST (7 h); VM dead-man service+timer installed, log-only test ok, hard cap 7.5 h; laptop heartbeat service (60 s) active; sleep inhibitor running.
2026-11-23T20:21Z BOOTSTRAP: image 52.8 GB pulled, scene 25 GB fetched, host_doctor PASS (after installing rclone), checkpoints copied (GA + go2_arx5 retrained).
2026-11-23T20:20Z R1: no mdlc in the Isaac Sim 6.2 container (find / -xdev -name mdlc* -> empty; only the MDL runtime ships). Scene material library: 812 .mdl, 2998 textures, KooPbr imports. Evidence in docs/reports/render_unblock.md.
2026-11-23T20:22Z R3: conversion over the real scene: 3293 materials, 1624 with diffuse+normal texture, 3847 bindings rebound, 0 missing texture files. Preview wrapper written; stills_r3 render launched under run_until_complete (12 launches max).
2026-11-23T20:25Z N3 JEPLO attempt 2 DONE: authors' checkpoint walked in their MuJoCo sim (headless EGL, ported depth publisher, CUDA EP not TensorRT); 30 s clip, clip_qa media ok but hue-diversity fails on MuJoCo's grey grid (stated in the report).
2026-11-23T20:30Z R3 stills: the scan proxy was missing (built, 20.7M triangles) and the fused go2_arx5.usd was missing (fetched from R2 embodiment/). Render retry loop relaunched.
2026-11-23T21:20Z INCIDENT: VM 1178616 was deleted by its own dead-man. Cause: the laptop's CGNAT address changed (138.51.64.37 -> 138.51.73.111), the /32 firewall rule no longer matched, the SSH heartbeat stopped, and 45 min later the dead-man synced results to R2 (migration/session3/vm-backup) and deleted the VM as designed. Cost ~1.35 USD, VM time 1.2 h. No results lost. Fixes for the replacement: heartbeat refreshes the ingress rule, dead-man threshold 90 min. docs/reports/session3_incident_vm_loss.md.
2026-11-23T22:58Z R3 STILLS PASS. Recipe: converted materials (1678 diffuse / 2866 normal from the MDL sources) + MDL material prims deactivated (this stopped the MDLC compile storm and the robot falling through the floor) + dome 300 (scene has 137 own lights) + open spawn (-3.75,0.73) + adjusted cameras.
  still_ego: mean lum 157.8, 0.1% clipped, 4 hue bins -> PASS. Opened it: living room with a white L-sofa, cushions, kitchen (cabinets, fridge, oven, range hood, utensils), coffee table, picture frame, glossy floor with real specular highlights; the robot's own shoulder fills the right edge (EGO offset still needs the arm clear).
  still_chase: mean lum 147.2, 0.1% clipped, 6 hue bins -> PASS. Opened it: full living area + open kitchen (sink, faucet, kettle, microwave, fridge), dining table and chairs, a hallway shoe rack with shoes, wall art, reflective tile floor; the robot's arm is visible bottom-centre.
2026-11-24T00:55Z VM2 died to its own dead-man again (the systemd --collect heartbeat unit disappeared when a CGNAT IP change made its loop exit); results were synced to R2. Fix for VM3: dead-man = hard-age only (7.5 h), laptop kill timer at 7 h, no network-dependent heartbeat. Creations used: 2 of 3; VM3 = cerberus-scene3c, id 1179086, 185.216.22.224.
2026-11-24T01:05Z VM3 provisioning: driver 595.58.03, repo, env, checkpoints, embodiment asset, image pull + scene fetch in parallel, dead-man installed and log-only tested.

2026-11-24T03:36Z OWNER-DIRECTED STATUS LINE: spend ~6.0 USD of 10; VM 1179086 age 2 h 41 m (created 00:55Z, cap 7 h, packaging at 6.5 h); DONE: R1 (diagnosis), R3 material conversion + the in-scope-placement fix that finally makes the override apply (flat-red test turned red), CAM (segmented capture, 0.11 s/frame), stills (2 accepted), walk_ego 20 s clip passing the whole gate; BLOCKED/PENDING: chase clips (rendered before the material fix, to be re-rendered), Track M (not attempted), SLAM (not attempted), K2 (not attempted), packaging (in progress).
2026-11-24T03:36Z QA THRESHOLDS: clip_qa.py was NOT relaxed at any point this session (LUM 70-170, <=5% clipped, >=3 hue bins each >=5% stay as in session 2). Measured hue-bin counts: still ego 4, still chase 6; walk_ego 3/4/3/3/3 (pass), walk_overview 3/4/1/1/1 (fail at 50-100%), arm_closeup 2/1/1/1/1 (fail, camera inside furniture), arm_chase (pre-fix) 2-3 (fail). No grey scene passed: the two stills and walk_ego pass because they contain the room's colour variation, not because a threshold was changed.

2026-11-24T03:56Z EXPOSURE BOX (owner option a then b): (a) the RTX tonemapper exposure setting had NO effect (-1.5 stops: 199.2 vs 199.3 mean luminance); (b) hiding the emissive HDR sphere: no effect (222.5); a 0.45 global scale on all 137 scene lights moved chase only 222 -> 199. Conclusion recorded: no exposure control on this stack changes these views; the scenes are white-lit interiors and the wide views read 173-235 (outside the gate's 70-170 band). Only the EGO view (134) is in band. NO threshold was changed.
2026-11-24T03:56Z FINAL STILLS (5 views + 2 camera views, 1920x1080): ego 134.1 PASS (5 hue bins); behind 173.4; front_left 211.8; front_right 224.2; high_wide 213.4; side 235.4; chase 222.1 -- all fail the luminance band, none clipped > 0.1%. Materials are the dataset's own textures (floor tiles, brass, cushion, wardrobe seams visible in the EGO and front_left frames).
2026-11-24T03:57Z LEDGER: R-D1 PARTIAL, R-D2 PARTIAL (walk_ego passes), R-D3 BLOCKED, R-D4 not attempted; the USD reference-scope fix is committed and the dataset textures now render.
2026-11-24T04:14Z R-D3 and R-D4 DONE: walk_ego_split.mp4 passes the gate (83-86 mean luminance, 0.000 clipped, 3-4 hue bins); teaser_walk_ego_10s.gif 1.36 MB.
