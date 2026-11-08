# Scene eval (Phase 0, non-blocking)

Policy under test: **`go2_locomotion_ga_v1_candidate.pt`** — the GA retrain. `go2_locomotion_final.pt`
does not exist, because the Phase 0.5 gate failed, so the candidate is what any scene check can use.

Purpose: sanity-check the frozen policy outside procedural terrain, and exercise `plain_usd` with a
walking robot. This report is **non-blocking**: nothing here changes the Phase 0.5 verdict or the
decision not to freeze a checkpoint.

## What was run

Scenes: three admitted GRScenes homes from the private R2 mirror
(`worlds/grscenes/home`, status `pass` in `worlds/grscenes/admission.json`):

- `MV7J6NIKTKJZ2AABAAAAADQ8_usd`
- `MV7J6NIKTKJZ2AABAAAAADY8_usd`
- `MVUCSQAKTKJ5EAABAAAAAAI8_usd`

Per scene, in order: fetch from R2 → `plain_usd` preprocess (GRScenes are centimetre-authored and Isaac
does not convert units, so `sim_usd_path` generates a cached metre-scale wrapper beside the raw USD) →
load + stand check (`arena_spike_go2.py --backend plain_usd`, 16 envs, 200 zero-action steps).

<!-- FILLED IN FROM THE RUN -->

## Why the scripted velocity path could not be driven (the actual finding)

The `plain_usd` backend builds a **navigation** env, not the locomotion policy's env. The two contracts
differ in exactly the places that matter, so the trained policy cannot be stepped in these scenes
without writing a new task package:

| | plain_usd navigation env (`make_go2_mdp_cfgs`) | candidate policy (measured contract) |
|---|---|---|
| observation group | `joint_pos, joint_vel, base_lin_vel, base_ang_vel, projected_gravity` → **33 dims** | 235 dims: the same six plus `velocity_commands` (3), `last_action` (12) and `height_scan` (187) |
| observation shape | `concatenate_terms = False` → a **dict of groups** | one concatenated tensor, fixed order |
| velocity command term | **absent** — there is no `base_velocity` command manager | required; the policy tracks ±1 m/s |
| action | `JointPositionAction`, scale `ACTION_SCALE`, default offset | `JointPositionAction`, scale **0.25**, default offset |
| terrain/scene | USD scene or ground plane | generated rough terrain + a height scanner over it |

So a scripted velocity path would require adding a velocity-command term, a `last_action` term and a
height scanner to the navigation cfg, and matching the training action scale — i.e. composing the
locomotion task into a USD scene. That is Phase 1 work (the velocity-command composition), not a
sanity check, and it is out of this handoff's scope. **The scene runner therefore stops before
driving and records that boundary rather than faking a path**: `drive SKIP` in the run summary.

What *was* verified is the part that was actually in question: that GRScenes homes fetch, preprocess to
metres, load through `cerberus_worlds`, and host a Go2 standing on GPU physics. That is the same check
the W1 spike made, repeated here on the GA stack with the GA-era code.

## Cost

GPU time for the whole package is capped at **1 h**; the runner stops starting new scenes past
`SCENE_EVAL_CAP_S` (default 3600 s) and records `SKIP cap` for the rest. Actual time is recorded in the
run summary and in `docs/h2_report.md`.

## Reproduction

```bash
scripts/run_scene_eval_pod.sh                       # default: the three scenes above
scripts/run_scene_eval_pod.sh --scenes <id>,<id> --num_envs 16
```

Artifacts: `data/reports/scene_eval_<stamp>/` (per-scene fetch log, preprocess JSON, spike JSON and log,
summary), uploaded to R2 under `migration/scene_eval_<stamp>/`.
