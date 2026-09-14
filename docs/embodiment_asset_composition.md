# Cerberus — Embodiment Asset Composition (Go2 + ARX5)

> **Status: provisional.** Research pass for Phase 2 (arm integration). Nothing is built yet — `common/cerberus_description/` doesn't exist, no arm asset has been sourced. Update or delete sections as better info arrives.

## Scope

There's no factory-made "Go2 with an ARX5 arm attached" 3D model. Before Phase 2 can train anything with the arm in the loop, Go2 and ARX5's separate USDs need to be physically combined into one model. This doc covers how that combining should happen, and how to structure the result so future platform swaps stay cheap.

This is separate from `project_structure.md`'s "Embodiment configuration pattern" section, which covers where the resulting config lives and how it's consumed — that's the code-organization side, this is the asset-authoring side.

## Robot Assembler solves this

Isaac Sim ships **Robot Assembler** (`isaacsim.robot_setup.assembler`) for exactly this. Pick a "Base Robot" (Go2), an "Attach Robot" (ARX5), a mount frame, and it connects them with a fixed joint. The "single robot" option merges both robots' joints under one controllable articulation rooted at the base — exactly what's needed.

It has a Python API too, not just a GUI: `RobotAssembler.assemble_articulations()` does the same thing in code. Wrap it once as `assemble(legs_usd, arm_usd, mount_point) -> merged_usd`, and a future arm or leg-platform swap is just rerunning the script with different inputs.

## Reference project didn't script this either

Quick check of LeggedManip_Lab (this project's cited reference, github.com/zzzJie-Robot/LeggedManip_Lab): each of its 7 combo USDs is committed as a finished binary paired with a hand-written `ArticulationCfg`. No authoring script visible. Scripting this for Cerberus is deliberately ahead of that precedent, not an oversight.

## Recommended approach

1. Source separate Go2 and ARX5 USDs first (Go2 from Isaac Lab's asset library; ARX5 from ARX's release channels or community Isaac Sim ports).
2. Write a small composition script using `RobotAssembler.assemble_articulations()` — inputs are a legs USD, an arm USD, and a mount transform; output is one merged USD with "single robot" enabled. Keep it generic over inputs, not hardcoded to Go2/ARX5 specifically — that's what delivers the future-swap benefit.
3. Run it once for Go2+ARX5, commit `go2_arx5.usd` under `common/cerberus_description/`.
4. Write one short `ArticulationCfg` pointing at the merged file — matching LeggedManip_Lab's convention (one self-contained config per combo, not sub-configs glued together at import).
5. A future swap (different arm, different legs like ANYmal) is just rerunning the script with new inputs and writing one new cfg. Go2's USD and Python files stay untouched.

## Open items for when Phase 2 starts

- Confirm Go2's attach point (likely the top plate/mounting rail — check Unitree's mounting spec) and the matching mount frame on ARX5's base.
- Confirm ARX5's USD is portable to Isaac Sim, or needs URDF→USD conversion (Isaac Sim has a URDF importer).
- Check joint-namespace collisions between Go2 and ARX5 — Robot Assembler may or may not auto-prefix on merge.
- Decide whether the composition script lives in `common/cerberus_description/` itself (checked in for reproducibility) or in `scripts/` per the existing convention.
