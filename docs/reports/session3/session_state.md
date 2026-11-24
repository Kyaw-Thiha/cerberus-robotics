# Session 3 state — FINAL (2026-11-24 04:20Z)

## Host
- No VM live. Session used three L40s, all deleted or deleted by their dead-man:
  - `cerberus-scene3` id 1178616 (20:07-21:20Z) — killed by its dead-man after a CGNAT move broke the SSH heartbeat
  - `cerberus-scene3b` id 1178770 (21:27-23:05Z) — same failure mode; the heartbeat systemd unit had exited
  - `cerberus-scene3c` id 1179086 (00:55-04:04Z) — the productive one; deleted at wrap-up, verified gone via the API
- Driver 595.58.03 (runfile + DKMS, nouveau blacklisted), image `kevinkyawthiha/cerberus-remote-v6:v6-ga`,
  scene `MV7J6NIKTKJZ2AABAAAAADQ8_usd`, GA + go2_arx5 checkpoints, fused `go2_arx5.usd`, scan proxy.

## Spend
- 6.50 USD of the 10 USD cap (5.93 VM-hours; ~2.6 h of that was lost to the two dead-man deletions).

## The render unblock (docs/reports/render_unblock.md)
1. Materials: MDL -> UsdPreviewSurface from the KooPbr `.mdl` sources (1,678 diffuse / 2,866 normal maps);
   MDL prims deactivated; **materials must be authored inside the scene's own root** or USD drops the binding
   targets as out of reference scope (that was the flat grey).
2. Capture: `sim.render()` + `app.update()` = 0.11 s/frame (vs 15.0 s for `orchestrator.step`).
3. Cameras: driven and logged from the Fabric pose; `root_quat_w` is XYZW on Lab 3.0.
4. Exposure: NOT solved. Tonemapper exposure, hiding the HDR sphere and a 0.45 light scale change these views by
   <10 %; wide views read 173-235, the EGO view 134.

## Gated artifacts (~/cerberus-results/2026-11-23/clips_final/)
- `walk_ego.mp4` PASS, `walk_ego_split.mp4` PASS, `teaser_walk_ego_10s.gif` (1.36 MB),
  `still_ego.png` PASS (figures in docs/reports/figures/), `arm_ego.mp4` 4/5 frames.
- Draft release `showcase-2026-11-23` (NOT published) carries walk_ego, walk_ego_split, arm_ego, the teaser and the
  kinematic mission animation.

## Not done
- Track M (physics mission N=10), SLAM, K2 matrix, the CHASE/close-up hero clip and the 60 s three-camera cut
  (rendered but out of the luminance band).
