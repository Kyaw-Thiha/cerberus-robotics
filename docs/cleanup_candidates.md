# Cleanup candidates — audit, 2026-11-08 (Phase A, nothing applied)

Scope: the **merged** tree (`main` @ `5d18f9f`, after H1 was merged in), branch `cleanup-2026-11-08`.
386 tracked files. Nothing has been deleted, moved, or edited yet — this file is the audit.

## Method

- **References**: for every tracked file, its basename and stem were grepped across all tracked text
  files (code, tests, `pixi.toml`, Dockerfiles, CI, scripts, docs). A file is "unreferenced" only if no
  other tracked file mentions it.
- **Duplicates**: sha256 over every tracked file; groups of identical non-empty content reported.
- **Empty files**: every tracked file of size 0, with its role checked.
- **Dead code**: `ruff check` (the H1 baseline, `ruff.toml`) plus targeted greps for definitions with no
  production callers.
- **Stale docs**: every Markdown link `](target)` in every tracked `.md` resolved against disk —
  **0 broken links**.
- **Committed build output**: tracked `__pycache__/`, `*.pyc`, `*.log`, `*.DS_Store`, editor backups —
  **0 files**.
- **Session debris**: untracked leftovers from earlier branch switches (see the last section).

## Candidates

| path | type | why it is a candidate | evidence | class |
|---|---|---|---|---|
| `policy/locomotion/TRAINING_MECHANICS_FAQ.md` | doc | Phase-0 mechanics FAQ (training time, reward shape, eval setup, video capture); content now lives in `docs/isaac_lab_workflow.md` and `policy/locomotion/runs/RUN_00{1,2}.md` | 0 references in any tracked file; last commit 2026-09-05; headings overlap `isaac_lab_workflow.md` (video/monitor/tensorboard/checkpoint) | **ARCHIVE** |
| `docs/reports/status_report_2026-11-06.md` | doc | Byte-identical duplicate of `docs/status_report_2026-11-06.md` | `sha256 69b55f9b…` on both; both paths are in the PROTECTED list | **NEEDS APPROVAL** |
| `docs/reports/gate_verdict_20261108T055128Z.json` | doc (data) | Duplicate of the canonical `policy/locomotion/adversarial_robustness/eval/results/gate_eval_20261108T055128Z/gate_verdict.json` | `sha256 0e22a6c8…` on both; the sidecar and `phase05_adversarial_report.md` point at the eval/results copy; this one exists so the verdict is findable in `docs/reports/` | **NEEDS APPROVAL** |
| `policy/locomotion/adversarial_robustness/attack_math.py` → `RateLimitedChannel` (class) | code | Superseded within its own module: the attack and robustify wrappers rate-limit per env through `rate_limited_step()`, so the class has no production caller | `grep RateLimitedChannel`: 0 hits outside its definition except `tests/test_adversarial_attack_math.py` | **NEEDS APPROVAL** |
| `policy/locomotion/checkpoints/.gitkeep` | config | Empty placeholder whose only job is to keep the directory in a clone | size 0; 0 references; deleting removes the git-tracked directory that every doc calls the checkpoint home | **NEEDS APPROVAL** |
| `scripts/pull_pod_artifacts.sh` | script | One-off pull-down helper written when R2 upload was believed unavailable (it was available); superseded by `scripts/sync_to_r2.py` / `rclone` | referenced only by `scripts/preflight_local.sh`'s shellcheck list, no doc; added 2026-11-08 | **NEEDS APPROVAL** |
| `scripts/run_ga_smoke_pod.sh` vs `scripts/run_ga_stack_check_pod.sh` | script | Overlapping pod smoke drivers: the first (H1) does train-smoke → play+video → Arena spike; the second (H2) does env report → quat/physics probe → 64-env smoke → parity sweep | `run_ga_smoke_pod.sh` referenced by `docs/runpod.md`; `run_ga_stack_check_pod.sh` by `scripts/preflight_local.sh` | **NEEDS APPROVAL** |
| `policy/locomotion/eval/push_recovery_showcase.py`, `…_flat.py` | script | Two video helpers with overlapping machinery for different terrains (rough vs flat) | rough referenced by RUN_001/RUN_002 + the FAQ (ARCHIVE row); flat by RUN_001 + `arena_integration_log.md`; 246 vs 169 lines, not duplicates | **NEEDS APPROVAL** |
| `plans/handoff_h2_ga_retrain_and_phase05.md` | doc | Exists only on branch `h2-phase0-ga-and-phase05` (`0101435`); not on `main` | `git diff --name-status main h2-phase0-ga-and-phase05` shows this as the branch's only file absent from main | **NEEDS APPROVAL** |

## SAFE rows: none

No tracked file passed all the SAFE conditions on this tree. What was checked and rejected:

- **Duplicates**: the two duplicate groups are both under PROTECTED paths (`docs/reports/*`,
  `docs/status_report_2026-11-06.md`), so they cannot be SAFE-deleted by this cleanup.
- **Empty files**: 23 tracked files are 0 bytes, and every one is a Python package marker
  (`__init__.py`), an ament resource index (`policy/locomotion_policy/resource/…`, `slam/resource/slam`),
  `policy/locomotion_policy/conftest.py` (deliberately empty — it puts the package on `sys.path`), or
  `.gitkeep`. Deleting any of them removes a directory or breaks an import; none is a placeholder in the
  SAFE sense.
- **Committed generated output**: none (0 `__pycache__`, `.pyc`, `.log`, editor backups).
- **Unreferenced scripts**: none — every `scripts/*.py` and `scripts/*.sh` is referenced outside its own
  test file.
- **Broken doc links**: none, so there is no dead-reference cleanup.
- **This session's one-shot drivers**: `run_ga_stack_check_pod.sh`, `run_ga_retrain_pod.sh`,
  `run_adversarial_gate_pod.sh`, `run_robustify_pod.sh`, `run_scene_eval_pod.sh`, `pod_isaac_lib.sh` are
  all pointed at by either `scripts/preflight_local.sh` or a report — the handoff's condition ("that no
  doc or handoff still points to") is not met, so none qualifies.

## Session debris (untracked, already out of the tree)

Moved to `/tmp/untracked-backup-2026-11-08/` before the merge, as instructed:

| item | what it was | reconciliation |
|---|---|---|
| `docs/reports/h1_report.md` | untracked copy of H1's report | **byte-identical** to the merged `docs/h1_report.md` (`sha256 81829db8…`); backup copy removed |
| `common/cerberus_policies/` | leftover `__pycache__` only — 14 `.pyc` files, zero source | not comparable to the merged tree (bytecode caches of H1's sources, which the merge brought in); backup kept in `/tmp` |

## ruff: ignored families that are auto-fixable and safe to enable (report only)

`ruff.toml` ignores 18 families to keep the 2026-11-06 baseline green. Hits and auto-fixability if each
were enabled today:

| rule | hits | auto-fixable | verdict |
|---|---|---|---|
| `RUF100` unused-noqa | 98 | 98 | **safe**: every hit is a `# noqa: PLC0415`-style marker whose rule is not enabled; deleting them changes no behaviour |
| `I001` unsorted-imports | 21 | 21 | **safe**: import order only, no semantics |
| `UP035` deprecated-import | 1 | 1 | **safe**: mechanical replacement |
| `PLR0402` manual-from-import | 1 | 1 | **safe**: mechanical |
| `C408` unnecessary-collection-call | 2 | 0 | safe to enable, needs a hand edit (`dict()` → literal) |
| `SIM118` in-dict-keys | 1 | 0 | safe, hand edit |
| `SIM102` collapsible-if | 1 | 0 | safe, hand edit (fold nested `if`) |
| `RUF015` unnecessary-iterable-allocation | 3 | 0 | safe, hand edit (`next(iter(x))`) |
| `PLW1510` subprocess without `check` | 4 | 0 | safe **only** as explicit `check=False` — must preserve current behaviour per call |
| `EXE001` shebang-not-executable | 3 | 0 | safe: `chmod +x` |
| `DTZ005` naive `datetime.now()` | 1 | 0 | **not** behaviour-free: adds a tz, which changes generated timestamp strings |
| `F841` unused-variable | 2 | 0 | review first: usually a real bug |
| `B008` call-in-default-argument | 6 | 0 | review first: may be a deliberate dataclass default |
| `RUF012` mutable-class-default | 3 | 0 | review first: dataclass field semantics |
| `RUF059` unused-unpacked-variable | 4 | 0 | cosmetic, but renames are structural here |
| `S110` try/except-pass | 1 | 0 | deliberate in that location — keep ignored |
| `TRY004` type-check-without-TypeError | 1 | 0 | deliberate — keep ignored |

Recommended burn-down, cheapest first: `RUF100` + `I001` (119 mechanical fixes, zero behaviour change),
then `UP035`/`PLR0402`/`C408`/`SIM118`/`SIM102`/`RUF015`/`EXE001` (10 hand edits), then decide the
`PLW1510`/`DTZ005`/`F841`/`B008`/`RUF012` group individually. Enabling them is **not** part of this
cleanup; it is reported here for a separate change.

## Proposed Phase B actions (awaiting approval)

1. **Archive** `policy/locomotion/TRAINING_MECHANICS_FAQ.md` → `docs/archive/` with a banner pointing at
   `docs/isaac_lab_workflow.md` and `runs/RUN_001.md`/`RUN_002.md`; add it to the archive index if one is
   created.
2. **Approved deletions only** — none proposed yet: every remaining row is a decision for the owner
   (duplicate under protected paths, a superseded class, a placeholder, two overlapping driver pairs).
3. **Doc-reference fixes** after any move; regenerate `docs/backend_support.md` if a card changes (none
   do today: `pixi run backend-table -- --check` is green).
4. `docs/project_structure.md` tree update for anything moved.
