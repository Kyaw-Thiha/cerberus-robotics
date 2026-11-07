# Handoff H1 (v2): Doc Reconciliation, Housekeeping, Interface Freeze

Runs **locally, no Isaac, no GPU**. Branch: `h1-docs-interfaces`. One commit per work package (A-D), each passing `pixi run pytest -q` and `pixi run lint` before moving on.

## Why this exists

The 2026-11-06 status report found: Phase 0 trained, Phase 0.5 not started, 14 doc/code mismatches, 3 pytest collection errors, a 2-line README, and no interface layer (`LocoCommand` / `ArmCommand` / `BodyCommand` / `PolicyBackend`). Later work (Phase 0.5, JEPLO, whole-body) plugs into those interfaces, and agents read the docs as ground truth, so we fix both first.

## Project philosophy

Cerberus is a **platform**: built-in support for several locomotion, manipulation, whole-body and sensor-rig paradigms, each integrated through a swappable backend. We do not train everything. Only a few backends get trained by us; others are integrated and evaluated with released weights or no training (e.g. SLAM); others are integrated and contract-tested only. Every backend carries a **support level**:

| Level | Meaning | Evidence required |
|---|---|---|
| L0 | Documented | Backend card only, no code |
| L1 | Integrated | Adapter exists and passes contract tests (loads, `act()` returns correct shapes, sensor needs validate against a rig) |
| L2 | Evaluated | Run on our scenes or shared evals with released weights or without training; numbers recorded |
| L3 | Trained by us | Trained and evaluated by us; result recorded in a report |

Backends may be **modular** (consume a map or other upstream module output) or **fused** (consume raw sensor streams, perception encoder trained jointly inside the policy, e.g. JEPLO, DAWN/WMP, Omni-Perception). Both plug in at the same boundary (B1 for locomotion). Fused backends declare raw sensor streams in `required_sensors`.

## Ground rules

1. No Isaac Sim/Lab imports in anything run. Isaac-only changes go under "Deferred to remote".
2. Docs describe reality. Missing things are marked `(planned, not yet created)`; no empty dirs.
3. Do not delete any doc. Superseded content gets a banner pointing to its replacement.
4. Do not touch `policy/locomotion/`, `slam/`, or `common/cerberus_worlds/` behaviour (docs, tests, new files only where stated).
5. Never print or commit `.env`.
6. Do not research external repos; write "unverified on the current stack".

## Work packages

- **A** Housekeeping: importorskip for 3 collection errors; `colcon_defaults.yaml`; ruff config; `.github/workflows/ci.yml`; report `.env` mode.
- **B** Doc reconciliation, items 1-10 (README, project_structure, worlds, project_context, isaac_lab_workflow, artifact_storage, manipulator banner, modular_stack_options notes, superpowers duplicate banners, arena log note). Gate B: re-grep stale terms.
- **C** Interface freeze: C1 `.msg` files (LocoCommand, ArmCommand, BodyCommand); C2 `common/cerberus_policies/` (base, registry, backend_card + schema, validate_config, null_loco, scripted_loco, cards, tests); C3 `docs/interfaces.md`; C4 `scripts/gen_backend_table.py` + `docs/backend_support.md` + freshness test + pixi task.
- **D** (optional) `docs/embodiment_split_plan.md` for the `UNITREE_GO2_CFG` / `ARX5_CFG` split. Do not change the Isaac-dependent import path.

## Final report

Save as `docs/h1_report.md`: commits per package; gate results A/B/C; design choices where spec was silent; things not done + "Deferred to remote"; anything contradicting this handoff.

## Out of scope

Phase 0.5 code, any policy backend that loads a checkpoint, SLAM changes, scene fetching, Dockerfile changes, Arena W1 rerun, VLA/manipulation work, verifying external backends.

(Condensed copy of the original handoff instruction as received 2026-11-06; the work-package tables in the original are the authority for per-item detail.)
