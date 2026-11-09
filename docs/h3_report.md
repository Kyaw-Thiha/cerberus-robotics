# H3 report — adopting the interfaces (ONNX backend, B1 limiter, ROS node, docs truth-up)

Local only (no pod, no GPU, no Isaac imports), on `main`, starting from `5d18f9f`. Date: 2026-11-09.
Handoff: `plans/handoff_h3_adopt_interfaces.md`.

## 1. Commits per work package

| package | commit | contents |
|---|---|---|
| A | `79ae1a9` | `pyyaml` declared; `blind_ppo` → `heightscan_ppo`; `blind_ppo_flat` at L0 (then L3 in C); cumulative L2/L3→L1 evidence rule + failing-card test; `modular_stack_options.md` note; README phase table/claims/Git workflow; `h1_report.md` rename note; table regenerated (9 backends) |
| B | `2da0328` | `command_limiter.py` + placeholder YAML; optional card fields `max_command_rate` / `command_limiter_applied` (schema + validator); 13 limiter tests; `interfaces.md` rate-limiting section |
| C | `e8c4eb4` | `contracts/heightscan_ppo_v1.yaml`; `backends/heightscan_ppo_onnx.py`; `projected_gravity_from_quat` + GA test vector; `heightscan_ppo` card → L3; registry entry; `height_scan` as an upstream module output in `validate_config`; contract tests incl. a 100-observation smoke run on the real candidate |
| D | `e814722` | `HeightScan.msg`; `locomotion_policy/core.py` (pure) + thin wrapper; LocoCommand/IMU/JointState/HeightScan wiring with the limiter; deprecated Twist adapter (off); JointState output keyed by name; bringup params; `platforms.yaml` joint order/defaults aligned to the contract; IMU declared required; rclpy round-trip smoke test |
| E | this commit | `interfaces.md`, `project_structure.md`, this report, table regenerated |

**Push:** every package's `git push` failed — the workstation has no SSH identity for
`git@github.com:Kyaw-Thiha/cerberus-robotics.git` (`Permission denied (publickey)`, `ssh-add -l`: "The
agent has no identities"). `git fetch` failed the same way before any edit. Work was done locally from
`5d18f9f`; **CI has not run and cannot be confirmed green until the remote is reachable.** This
violates the handoff's "push after each package" instruction only in the sense that it was impossible;
nothing was force-pushed or rewritten.

## 2. Gate results

All runs are `pixi run …` from the repo root, on the local machine.

| gate | pytest | lint | table freshness | package-specific |
|---|---|---|---|---|
| baseline (before edits) | 1056 passed, 19 skipped | clean | fresh | — |
| **A** | 1057 passed, 19 skipped | clean | fresh (`gen_backend_table.py --check` exit 0) | `grep -rn blind_ppo` → only `blind_ppo_flat`, its card table row, the test expectation, and history/handoff references; README Phase 0.5 no longer "Not started" |
| **B** | 1072 passed, 19 skipped | clean | fresh | limiter imports with no torch/ROS/Isaac/numpy (`sys.modules` check); adversarial-robustness suite unchanged: 54 passed, 1 skipped; no diff under `policy/locomotion/adversarial_robustness/` |
| **C** | 1086 passed, 19 skipped | clean | fresh | backend module import leaves torch/rclpy/onnxruntime/numpy out of `sys.modules`; card validates at L3; real-candidate smoke ran (not skipped) |
| **D** | 1097 passed, 20 skipped | clean | fresh | `colcon build --packages-up-to locomotion_policy cerberus_bringup` → 4 packages finished, 0 failed; rclpy round-trip test 2 passed when the built `cerberus_msgs` is sourced |
| **E** | 1097 passed, 20 skipped | clean | fresh | — |

The 20th skip is the rclpy round-trip test when `cerberus_msgs` is not on the path (CI does not build
it). It was run locally against a `/tmp` install and passed; the skip reason says how to enable it.

Limiter semantics were checked by full-module runs (`common/cerberus_policies/tests/`,
`policy/locomotion_policy/test/`), not single tests, and every rename/field change was grepped across
the repo (`blind_ppo`, `available_backends("loco")`, `sensor_requirements`, `joint_position_command`).

## 3. Design choices where the handoff was silent

1. **`CommandRateLimiter` adopts its first command.** After `reset()` there is no previous command to
   slew from, so the next `apply(dt)` returns the clamped request rather than ramping from zero.
   Ramping from zero would be a different (start-up) behaviour and would delay every engage; pinned by
   a test.
2. **The limiter is independent of `RateLimitedChannel`.** The attack-side channel bounds the
   adversary per sim step in 2-D (force + command) inside the frozen gate; it is untouched here. The
   deployment limiter is 3-D + optional body fields, with physical units per second and an explicit
   `dt`. A comment in `command_limiter.py` records the relationship; sharing code would couple
   deployment to the adversarial package and its sim-step semantics.
3. **Missing `.onnx.data` detection is a byte scan.** ONNX protobuf does not spell out the field name
   `external_data`; the GA export's bytes contain `location` and the literal data filename (verified
   on the real file). `_declares_external_data` checks for those, so a self-contained graph still
   loads while an external-weight graph without its sibling fails with a clear `FileNotFoundError`.
4. **`height_scan` is an upstream module output, not a rig sensor.** `validate_config` adds
   `MODULE_OUTPUTS = {"height_scan"}` rather than inventing a sixth key in every
   `sensor_profiles.yaml` rig (that file mirrors `sensor_configs.py`, and its exact-keys test says so).
   Any other unknown id still fails the rig check.
5. **`heightscan_ppo` sat at L0 in package A and became L3 in package C.** The A2 instruction says to
   set the level the evidence rules allow; with the cumulative rule from A4, good numbers without an
   adapter are L0. C adds the adapter and contract test, so L3 (trained + evaluated + reported) is
   allowed with the gate FAIL recorded in `results`, not hidden.
6. **Card `license` / `checkpoint_url` at L3.** `unknown` is forbidden above L0, so the card records
   `license: internal (weights trained by us; Isaac Lab BSD-3 tooling)` and the R2 prefix as printed
   in `docs/h2_report.md` (`r2://$R2_BUCKET/locomotion/unitree_go2_rough/2026-11-07_07-29-40/`). The
   bucket name itself was not read (it lives in `.env`).
7. **Node backend selection: `backend` parameter.** Empty = the deprecated inline ONNX path (no
   contract checks), used by the blind/flat mode because no flat contract/backend exists yet;
   `heightscan_ppo` = the registry backend. The launch selects per mode, so `blind` keeps working and
   the new path is only used where its contract exists.
8. **Output message changed from `Float32MultiArray` to `sensor_msgs/JointState`.** "Joint targets
   keyed by name" needs names; no in-repo consumer existed, and the topic name
   (`joint_position_command`) is unchanged.
9. **The node skips a control step without a fresh IMU (and HeightScan).** The policy's projected
   gravity and base angular velocity now come from `sensor_msgs/Imu`; publishing targets from a
   stale/missing orientation would be worse than not publishing.
10. **`platforms.yaml` joint order and defaults were wrong, and are fixed.** The file carried the
    URDF order (FR first) with a "PLACEHOLDER" comment, while training used FL → FR → RL → RR with
    real default angles. Both are now the contract's values, and a test pins the two files together.
    The blind/legacy path benefits too (the node still remaps by name).
11. **IMU declared required in `platforms.yaml`.** An IMU-less rig (`camera_only`) cannot run this
    node now; failing at launch beats a node that waits forever. `height_scan` stays outside the rig
    requirement, per that file's own design.

## 4. Contradictions and corrections found

- **The H2 report's "the ROS node must convert to wxyz before publishing" was inferred, and the node
  had no orientation code at all.** `inference_node.py` hardcoded `projected_gravity = [0, 0, -1]`
  with a TODO; nothing assumed w-first. The new wiring feeds `Imu.orientation` (xyzw) straight into
  `projected_gravity_from_quat`, so no reordering exists between ROS and Isaac Lab 3.0. A regression
  test pins the GA measurement.
- **`platforms.yaml` joint order contradicted the training order** (see choice 10) — a real
  observation-scrambling bug for the perceptive policy, not just a placeholder.
- **`default_joint_pos` was a zero vector**, which silently shifted every joint target
  (`default + 0.25*action`). Replaced with the training defaults.
- **`test_platforms_yaml.py` asserted `sensor_requirements == {}`**; the fact changed (IMU now
  required), so the assertion changed with it rather than being worked around.
- **`pixi.lock` did not change** when `pyyaml` was declared: it was already resolved at 6.0.3, so
  `pixi lock` reported "already up-to-date". The manifest now declares it explicitly (CI uses
  `locked: true`).
- **`docs/h1_report.md` referred to the `blind_ppo` card**; annotated instead of rewritten, since
  reports are history.
- **`docs/status_report_2026-11-06.md` cites `inference_node.py:153-155`** for the IMU TODOs; the line
  numbers moved. The report is a dated audit and was left as written; the remaining TODO is
  `base_lin_vel` only.

## 5. Deferred to remote (pod / live Lab)

| item | why it is deferred | how to close it |
|---|---|---|
| **Height-scan convention** (`sensor_z - hit_z - offset`, offset default 0.5, clip ±1, yaw alignment) | transcribed from the vendored Lab source; the effective offset is inferred from a default the saved params do not override, and it was not checked against the live term (`contract: convention_status: unverified`) | on a pod, read the live `height_scan` term values on flat ground and one slope; compare sign and offset with `HeightScan.msg` |
| **Eval of the policy behind the limiter** | the limiter's defaults are untested placeholders; nothing here claims the gate would pass | rerun the Phase 0.5 adversary against policy+limiter (the adversarial module is unchanged and reusable) and report the minimum inducing magnitude |
| **Map-corruption eval of the height scan** | the policy trusts a ground-truth scan; robustness to a degraded/estimated map is unmeasured | corrupt the scan (noise, dropout, lag, biased offset) and sweep fall rate; feeds the planned estimated-map ablation |
| **Live joint order** | the contract's order comes from the saved params (`joint_ordering: null`), not from reading `env.scene['robot'].joint_names` | print the articulation joint list on the pod and diff against `heightscan_ppo_v1.yaml` |
| **Blind/flat backend + contract** | only the rough policy has a contract/backend; blind mode still uses the deprecated inline ONNX path | add a flat contract and reuse the ONNX backend, then drop the inline path |
| **Real-hardware IMU paths** | `base_lin_vel` has no IMU source; odometry/state estimation is a hardware concern | wire a state estimator and remove the TODO |
| **rclpy round-trip in CI** | CI does not colcon-build `cerberus_msgs`, so the test skips | build the msgs in CI (or a separate job) and source the install prefix |
| **Push + CI** | no SSH identity on this workstation | push from a machine with the key; confirm `lint-and-test` green on `main` |

## 6. What this phase concludes

The interfaces are now real rather than documented: a real policy backend loads the GA export
(including its external weights), the observation contract is in the repo with the term order pulled
from the saved params, `projected_gravity` is pinned to a measured vector, a B1 command limiter exists
with honest placeholder defaults, and the ROS node consumes `LocoCommand` + IMU + joint states +
`HeightScan` and publishes name-keyed joint targets through the shared core. The policy is still the
one that failed the gate at 0.568×, and the limiter's effect is unknown — both facts are recorded in
the card, the README and this report rather than smoothed over.
