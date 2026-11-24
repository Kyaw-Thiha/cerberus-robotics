SESSION 3 HANDOFF (v1): RENDER UNBLOCK AND FINISH THE LEDGER. FIRST ACTION: save this
message verbatim to outputs/session3_handoff.md. Then READ outputs/session2_handoff.md,
outputs/queue_ledger.md, outputs/session_state.md, outputs/session_log.md and
docs/reports/first_scene_session.md. Every rule in session2_handoff.md still applies
(autonomy contract, fix ladder, NO EARLY END with the ledger, secrets, safety nets with
the laptop timer + VM dead-man + heartbeat + sleep inhibitor, state/resume/push,
playbook, honesty, clip QA gate) EXCEPT where this message changes it; this message
wins on conflicts. The owner is away for up to 10 hours and gives NO input; never ask.
All keys in .env were rotated by the owner.

NEW LIMITS (fresh counters): total spend 10 USD; VM at most 1.50 USD/hour; VM at most 7
hours from creation; at VM age 6.5 hours start packaging; at most 3 VM creations, one
live at a time; if the remaining budget is under 25 percent stop starting items and
package. No tuning of policies, planners or SLAM parameters.

CONTEXT: session 2 delivered the arm-robot locomotion retrain
(go2_arx5_locomotion_ga_v1_candidate, on R2) and the kinematic mission animation. RTX
rendering in the real scene is BLOCKED: the scene materials fail (the earlier frame was
one flat red-orange) and camera launches do not stay up. A high-fidelity RTX video of
the ARM robot in a real scene is a HARD REQUIREMENT of the owner. The previous agent's
next step was to compile one scene MDL with mdlc to find the failing import.

STEP 0 (30 min): read the ledger; list every item NOT DONE with its status. VM as in
session 2 section 7 (L40/A6000 48 GB, at least 200 GB disk, driver 595.58.03 from the
runfile, host_doctor PASS, bootstrap with R2 caches reused, all safety nets). Start the
no-Kit leftovers on the laptop immediately (see TRACK F).

TRACK R: RTX HERO CLIP OF THE ARM ROBOT (box 4 hours; the top priority; run first). Work
on single STILLS first (20 s each) and apply the still acceptance checks from session
2 section 10 (mean luminance 70-170, at most 5 percent clipped, at least 3 hue bins each
at least 5 percent, no unresolved-material warnings for visible materials); open and
describe every accepted still in outputs/session_log.md. Climb this ladder, 45 minutes per
rung, stop at the first rung that passes, keep notes on every rung in
docs/reports/render_unblock.md with the evidence:
 R1 DIAGNOSE (mdlc and logs): grep the Kit log for MDL/texture warnings and count them;
    compile ONE failing scene MDL with mdlc (the MDL compiler in the Isaac Sim install)
    to find the failing import or missing module; verify MDL_SYSTEM_PATH (and
    MDL_USER_PATH) point to the roots that contain the scene's MDL and texture folders
    and that the render process, its children and tmux inherit them; fix the path or
    supply the missing MDL modules.
 R2 VARIANT: render the VISUAL layer from the original or the interaction (-i)
    preprocess variant instead of the navigation variant (physics may keep the
    navigation variant).
 R3 MATERIAL CONVERSION (no MDL compiler needed): with pxr, walk the stage; for every bound
    MDL material read the diffuse/albedo texture path and colour constants (and roughness,
    normal if present) from its shader inputs or from the MDL file; create an equivalent
    UsdPreviewSurface material (UsdShade) with UsdUVTexture nodes pointing at the same
    texture files and rebind; verify the texture files exist on disk. Visual layer only.
    Write it as scripts/scene_materials_to_preview.py with unit tests on a small synthetic
    stage.
 R4 OTHER SCENE: the other cached GRScenes homes (MVUCSQAKTKJ5EAABAAAAAAI8_usd,
    MVUHLWYKTKJ5EAABAAAAACY8_usd, and the second scene id from the earlier work); repeat R1-R3.
 R5 BUNDLED SCENE: an Isaac Sim bundled environment that renders with its own working
    materials, labelled in every clip and in the showcase as 'not the GRScenes home'.
 Whatever rung works, set up the lighting (dome light plus a few rect or distant lights
 and exposure) and keep a fixed small set of light settings; record them.
 CAMERA STABILITY (do in parallel with R1-R3; use the session 2 K2 winner and the K3
 harness): camera-on Kit launches must be crash-tolerant. Render in SEGMENTS: each segment
 about 10-15 s of simulated time in its own launch, resuming from a saved robot state
 (root pose, joint positions, velocities, arm targets, command) written at the end of the
 previous segment; a segment counts only if its frames and its end-state file exist (never
 judge by exit code); stitch with ffmpeg; the stitched clip must be continuous (verify the
 base pose jump between segments is under 5 mm and 0.5 degrees). Try the other variants
 from session 2 K2 again if the numbers say they help (taskset -c 0-7, thread limits, the
 official NVIDIA container if NGC_API_KEY is set).
 CAMERAS (from session 2 section 10 B): EGO on the base link at the front-head position
 (adjust if the stowed arm blocks it), CHASE on a yaw-only frame behind and above (the arm
 links must be visible), a close-up camera for the arm, and a fixed OVERVIEW labelled
 'fixed overview'; log camera and base poses per frame.
 DELIVERABLES OF TRACK R, each through scripts/host/clip_qa.py (ffprobe; 5 frames at 0,
 25, 50, 75, 100 percent each passing the still checks; camera-pose checks; the arm
 visible in CHASE frames; look at the frames yourself and describe them): (1) arm robot
 standing in a real room, scripted reach, wrist turn, gripper open and close (20 s,
 CHASE and close-up); (2) the arm robot walking along the planned path with the
 retrained policy, EGO/CHASE/OVERVIEW cuts (60 s); (3) the split-screen with the top-down
 map (60 s); (4) a 10 s teaser GIF at most 8 MB. If no rung of the ladder passes, produce
 the best clips on R5, say so plainly, and record the exact blockers.

TRACK M: MISSION IN PHYSICS (only after Track R has at least one passing still and one
passing 20 s clip; box 3 hours): the household mission in the scene with the arm robot,
retrained policy, N=10 seeded trials, physical grasp first (45 minutes) with the 'magic
grasp' fallback LABELLED; clips of one success and one failure; numbers next to the
kinematic upper bound. If Kit stays unstable, run the trials through the segment harness.

TRACK F: LEFTOVERS (laptop or VM, no Kit; start at once; one item at a time as the ledger
requires): everything NOT DONE in the ledger that does not need Kit, in this order:
JEPLO (75-minute box, GPLv3, outside the repo, authors' released checkpoint in its MuJoCo
demo, clip, docs/reports/jeplo_spike.md), SLAM (35 minutes per backend, existing rosbags,
APE/RPE, trajectory plot, map-building animation), kit_reliability.md and
kit_crash_report.md (ready to file, NOT filed), docs/showcase.md.
SMALL FIX (30 minutes, laptop): the two tests that fail on a clean checkout (one imports
cerberus_msgs.msg.HeightScan which exists only after a colcon build; one imports a
vendored SLAM backend) must skip cleanly with a reason when those artefacts are absent
(pytest.importorskip or a marker); tests and lint green on a clean checkout; push.

PACKAGING (reserve 30 minutes and 1.5 USD): update docs/showcase.md with every clip that
passed the QA gate (including the kinematic mission animation and the retrained-policy
results), metrics tables, failures stated plainly, credits (GRScenes-100 CC BY-NC-SA 4.0,
non-commercial; Isaac Sim / Isaac Lab; JEPLO as the authors' demo); update the draft
GitHub release (gh release create showcase-<date> --draft, or upload to the existing draft
with gh release upload) with the new clips; never publish it; README Showcase section;
update docs/reports/first_scene_session.md and docs/reports/render_unblock.md; tests and
lint green; push main (fast-forward, per-command gh helper over HTTPS, never force).

SECOND PASS and WRAP-UP as in session 2 (sections 1 and 15): second pass for BLOCKED items
with different approaches; before wrap-up print the ledger and attempt every NOT
ATTEMPTED item; wrap-up deletes the VM and verifies it is gone, cancels every timer and the
heartbeat service, stops the sleep inhibitor, removes the secret files, pushes docs, then
the final chat message: the ledger, spend vs the 10 USD cap, the files to watch first, what
failed with evidence and the exact next step, and the reminder to rotate all keys.
