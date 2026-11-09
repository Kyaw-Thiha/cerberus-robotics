# H1 report (v2): doc reconciliation, housekeeping, interface freeze

Branch `h1-docs-interfaces`, local only (no Isaac, no GPU, no pod). Date: 2026-11-06.

## 1. Commits

- `4bfb20b` **A**: `pytest.importorskip` for the 3 collection errors, `colcon_defaults.yaml`, `ruff.toml`, `.github/workflows/ci.yml`, handoff saved to `plans/handoff_h1_docs_and_interfaces.md`.
- `2beb7e2` **B**: README rewrite (79 lines), `project_structure.md` tree reconciled, `worlds.md`, `cerberus_project_context.md`, `isaac_lab_workflow.md`, `artifact_storage.md`, `manipulator.md` superseded banner, `modular_stack_options.md` stack/lineup notes, superpowers duplicate banners, arena-log standing-check note, `status_report_2026-11-06.md` committed.
- `d36b8a0` **C1**: `LocoCommand.msg`, `ArmCommand.msg`, `BodyCommand.msg`, `CMakeLists.txt`, `package.xml`; colcon-built and import-checked.
- `fb85a6d` **C2-C4**: `common/cerberus_policies/` (ABC, registry, backend cards + schema, config validation, `null_loco`, `scripted_loco`, 8 cards, 7 test modules), `docs/interfaces.md`, `scripts/gen_backend_table.py`, `pixi run backend-table`, `docs/backend_support.md`, freshness test, README link, `project_structure.md` Rule 8.
- `0d51e2d` **D**: `docs/embodiment_split_plan.md`.

## 2. Gate results

**Gate A** (0 collection errors, lint passes):

```
$ pixi run pytest -q
899 passed, 18 skipped, 2 warnings in 27.43s
$ pixi run lint
All checks passed!
```

Before: `839 passed, 15 skipped, 2 warnings, 3 errors` and `Found 54 errors` from ruff. The 3 errors are now skips with reasons (`pytest.importorskip`); none of the three files was otherwise touched.

**Gate B** (stale hits, `grep -rnE 'OmniGibson|BEHAVIOR-1K|MobileVLA-R1|not yet written|Nothing is committed|approval' README.md docs --include=*.md`, archive excluded):

Every remaining hit is history, a negation, a banner, or the status report quoting its own findings:
`arena_integration_log.md` (W0 history, and line 253 = the `gated: false` source of truth), `cerberus_project_context.md:49` ("evaluated and dropped"), `:92` (explicit "QUART and MobileVLA-R1 are no longer leading candidates"), `manipulator.md` (banner + body kept as history), `worlds.md:5,9,156` (dropped/inapplicable), `worlds.md:55,170` ("no approval step"), `project_structure.md:307` and `runpod.md:19` (OmniGibson removal history), `modular_stack_options.md:173` (why T4 is a non-goal), `status_report_2026-11-06.md` (the audit itself, kept verbatim). No unflagged hit remains.

**Gate C** (new tests, no-Isaac import, docs exist, freshness):

```
$ pixi run python -c "<import + instantiate every registered backend>"
heavy modules loaded: []
$ pixi run python scripts/gen_backend_table.py --check   # exit 0
table-fresh
```

`docs/interfaces.md` (65 lines) and `docs/backend_support.md` (25 lines, generated) exist. 60 new tests; the freshness test (`scripts/tests/test_gen_backend_table.py`) fails when a card changes without regeneration.

**C1 build** (this env does carry ROS Jazzy, contrary to the handoff's assumption):

```
$ pixi run colcon build --symlink-install --packages-select cerberus_msgs --build-base /tmp/cb/build --install-base /tmp/cb/install
Finished <<< cerberus_msgs [9.31s]
$ python -c "from cerberus_msgs.msg import BodyCommand, ArmCommand, LocoCommand; ..."
0 1 2 6 6
```

The generated Python package includes all three new messages and the `ArmCommand` constants. The deprecation warning about `cmake_minimum_required` was already there.

## 3. Design choices where the spec was silent

1. **`Role` (producer/consumer) is explicit** on `PolicyBackend`. Read literally, the spec makes `null_loco`/`scripted_loco` ("returns zero velocity", "constant command from config") backend implementations, but a locomotion backend is a B1 *consumer* in `modular_stack_options.md`. Resolution: they are `Role.PRODUCER` B1 command sources, still in the `loco` slot (the slot is keyed by boundary). `validate_config` therefore enforces exactly one producer and one consumer per boundary in use, and takes `external_producers` / `external_consumers` for non-policy endpoints (a ROS planner, the actuation layer).
2. **Card `slot` matches the registry slot** (`loco`/`arm`/`body`), and a card's `boundary` must agree with it.
3. **`blind_ppo` is `architecture: modular`** — no perception encoder is trained inside the policy. (H3 renamed the card `heightscan_ppo`; the rough policy conditions on a simulator-provided height scan, and the truly blind flat policy is `blind_ppo_flat`.)
4. **`joint_encoders`** is treated as an always-available sensor id (ADAS `ALWAYS_AVAILABLE` in `validate_config.py`) alongside the rig ids from `sensor_profiles.yaml`.
5. **Sensor check is standalone**, not `cerberus_bringup.validate_sensor_suite` (that module is only importable after colcon builds it). A `TODO(unify)` is in `validate_config.py`.
6. **Evidence is typed** (`file`/`report`/`commit`), so the level rules are mechanical: L1+ needs non-empty evidence, L2/L3 need a `results` entry plus a `report` evidence entry, L3 needs `trained_by_us: true`, and `file` refs must exist (checked by the all-cards test). `unknown` is allowed only at L0, only on `required_sensors`, `rate_hz`, `license`, `checkpoint_url`, `code_url`. `target_support_level >= support_level`.
7. **L0 cards use `rate_hz: 50`** (the doc states all locomotion paradigms emit leg targets at ~50 Hz) and cite `docs/modular_stack_options.md` as their only evidence; everything the doc does not state is `unknown`.
8. **Ruff config is a baseline, not a policy** (`ruff.toml`, 18 ignored rule families with hit counts inline). The command needs to pass without a mass reformat; `I001` alone is 17 hits. This hides 60 real findings until someone burns the list down.
9. **`colcon_defaults.yaml` uses `base-paths: [common, policy, slam, system]`** — the four dirs that actually hold a `package.xml`; this keeps colcon out of `third_party/` and the `docker/slam/gs_livo/patches/rpg_vikit` patch tree.
10. **`docs/backend_support.md` evidence links are relative to `docs/`** (the generator strips the `docs/` prefix), so they resolve on GitHub.

## 4. Not done, and deferred

Not done (with reason):

- **Full `pixi run build`**: only `cerberus_msgs` was built, to a `/tmp` prefix (the repo's `build/` and `install/` hold stale colcon output for other packages; rebuilding them was not needed for this handoff and is not verifiable here).
- **ROS round-trip test of the new messages** (publish/subscribe a `BodyCommand` through an rclpy node): needs a built workspace and a ROS graph, which is a pod/laptop-ROS concern.
- **`pyproject.toml`**: none exists; `ruff.toml` was chosen so no packaging metadata is invented.
- **`pyyaml` is not declared in `pixi.toml`** — it comes in transitively via the ROS Jazzy env. `backend_card.py` and `gen_backend_table.py` depend on it; CI (locked env) is fine today, but a ROS-free env would break. Adding it would force a `pixi.lock` re-solve, which needs network, so it is left as a finding.

**Deferred to remote** (needs Isaac/GPU):

- Package D implementation (the `ARX5_CFG` split): `docs/embodiment_split_plan.md` has the exact refactor, the byte-identical acceptance check and the sim smoke list.
- Any L2/L3 evidence for a backend: no checkpoint-loading backend was written (explicitly out of scope); all six external cards stay L0.
- Arena W1 rerun on a scene where the robot can stand; the `stock`-scene standing check still fails (`robot_z_min_max [-0.73,-0.48]`).
- Phase 0.5 (`plans/handoff_phase0.5.md`), unchanged by this handoff.

**Housekeeping finding (A5):** `.env` is mode `644` with live-looking `WANDB_API_KEY`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_TOKEN_VALUE`, `RUNPOD_API_KEY`. It is gitignored (`.gitignore:46`) and was never read or committed. Recommend `chmod 600 .env`.

## 5. Contradictions / surprises found

1. **A1's diagnosis was incomplete.** `policy/locomotion/core/platforms/tests/test_registry.py` fails on `gymnasium`, but even with gymnasium installed it would fail next on `isaaclab_tasks` (imported by `policy/locomotion/__init__.py`). Both are now `importorskip`ped; on a pod it still runs as-is.
2. **"No ROS available" was wrong for this env.** The pixi default env has `ros-jazzy-ros-base` + `colcon-common-extensions`, so `cerberus_msgs` actually built. `cerberus_bringup` / `cerberus_msgs` Python packages are still not on `sys.path` by default (not installed into the env), which is why the skip guards remain correct.
3. **The handoff's "14 doc/code mismatches" list and the status report disagree in one place**: the report calls `perception/terrain_perception` and `planning/global_planner` "ROS2 pkg" without `package.xml`; the doc now says "planned ROS2 pkg" for those. `planning/local_planner` exists and was undocumented in the tree; it was added.
4. **`worlds.md` said the `stock` Arena scene "passed" on the pinned Sim 6.1 stack with no fall-through check recorded** — the log itself flags it as unverified, so it is now described as such rather than as evidence.
5. **The `arena_spike_go2.json` `passed: true` is not what the current script would write.** `scripts/arena_spike_go2.py:480` (`report["passed"] = not reasons`) already consults the standing check on this branch; the JSON predates it. Reported in `arena_integration_log.md`, spike code untouched, as instructed.
6. **`modular_stack_options.md` still says "our Isaac Lab 2.3 / Sim 5.1 pin"**, which the 2026-10-25 GA pin supersedes. A note was added; the doc was otherwise left alone, as instructed.
7. **`docs/project_structure.md` claimed `UNITREE_GO2_CFG` and `ARX5_CFG` are defined in `common/cerberus_description`** — only the composition exists; the status is now stated inline and the refactor is in `docs/embodiment_split_plan.md`.
8. **`scripts/arena_spike_go2.py` still pairs a `--scene stock` default with `--backend plain_usd` in the smoke script**, so the Arena path is exercised only when passed explicitly.
