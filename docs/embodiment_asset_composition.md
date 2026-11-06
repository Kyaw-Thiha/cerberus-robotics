# Cerberus — Embodiment Asset Composition (Go2 + ARX5)

> **Status (updated 2026-10-05): the URDF merge is the only workflow and the default.** One description (xacro `sim` variant + patched arm URDF + fixed mount joint) is merged offline and converted with Isaac Lab `UrdfConverter`. It works on Lab 2.3.2 (old stack) and the Lab 3.0 line (EA) and needs no Isaac assembler. The default fused USD `data/assets/go2_arx5/go2_arx5.usd` is built this way (Lab v3.0.0-EA, floating base, 20 DoF, 18.472 kg, flattened, self-contained). Mount z is the spec value 0.07 m (see Mount decision). W2a was passed on the earlier assembler-built asset (effective mount z 0) and was NOT re-verified on the new default; structure, hold and gripper tests were verified on the 0.07 m converted USD. Evidence: `data/reports/overnight/UrdfPath/`.

## Workflow

Steps 1, 2 and 4 are local, pure python, no Isaac; step 3 needs a pod with Isaac Lab.

1. **Fetch the arm.** `pixi run python scripts/fetch_arx5_assets.py` clones the official ARX_Model X5A into `third_party/arx-model` (gitignored).
2. **Merge.** `pixi run python scripts/build_go2_arx5_urdf.py [--out PATH] [--force]` expands the Go2 xacro with `sim:=true` (`common/cerberus_description/xacro_expand.py`, no ROS needed), patches the X5A URDF (`urdf_patch.patch_arm_urdf`: absolute meshes, `eef_link`, spec limits) and joins them with a fixed `arm_mount_joint` (Go2 `base` to arm `base_link`, offset `GO2_ARX5_MOUNT_XYZ`) into `go2_arx5`. Default out: `data/assets/go2_arx5/urdf/go2_arx5.urdf` (plus `<out>.build.json`). Result: 27 links, 26 joints, 20 actuated, 18.472 kg. Exit 1 validation failed, 2 bad args/refusing to overwrite.
   - `sim:=true` gives root link `base`, no rotor/sensor links, no gazebo/transmission output, foot mass 0.04 kg; Go2 total 15.017 kg vs stock Isaac 15.019 kg (Head_upper/Head_lower, 0.001 kg each, omitted). The default (`sim:=false`) ROS expansion is guarded by a golden test.
3. **Convert (pod, Isaac Lab).** `python scripts/convert_go2_arx5_urdf_to_usd.py [--urdf PATH] [--out PATH] [--force] [--collider convex_hull] [--dry-run]`. Defaults: in `data/assets/go2_arx5/urdf/go2_arx5.urdf`, out `go2_arx5_spec.default_fused_usd_path()` = `data/assets/go2_arx5/go2_arx5.usd` (plus `go2_arx5.usd.convert.json`). The converter writes payload/instance references; the script flattens the output into one self-contained USD. `--merge-fixed-joints` is off so `eef_link` survives as a body. It checks the result (one articulation root, joint set, mass, base and TCP prim paths).
4. **Compare (offline, optional).** `pixi run python scripts/compare_go2_urdf_to_usd.py [--urdf PATH] [--usd PATH] [--report PATH]` compares mass, CoM, inertia, joint limits and joint names. The golden reference is the legacy assembler-built USD `data/assets/go2_arx5/legacy/go2_arx5_assembler.usd`, kept only for this comparison (gitignored).

## Conversion results (pods `cerb-urdf-ea` 4090 and `cerb-urdf-old`, both deleted)

- `UrdfConverter` works on BOTH Lab 2.3.2 and the Lab 3.0 line EA: 1 articulation root, the 20 expected joints, no world root joint (floating base works), 18.472 kg.
- Converted vs legacy assembler-built USD comparison: PASS on both stacks.
- Hold test (EA, 16 envs, 2000 steps, includes gripper close/open), legacy (mount z 0) vs converted (z 0.07): free-base pitch -15.51 vs -15.07 deg, roll -0.59 vs -1.14, height 0.234 vs 0.216 m; fixed base identical (0.400 m, tilt 0); mass 18.474 vs 18.472 kg. All pass.
- Known benign differences: foot inertia (URDF 6.4e-6 vs legacy USD 9.6e-6; the legacy asset kept the 0.06 kg foot inertia), fixed foot joints named `*_foot_fixed` vs `*_foot_joint`, calf effort/velocity (URDF 35.55/20.06 vs legacy 45.43/15.7; the `ArticulationCfg` actuators override).
- Bug found by the offline comparison and fixed: the vendored Go2 xacro (and upstream unitree_ros) used the FRONT thigh limits (-1.5708..3.4907 rad) for all four legs; Unitree's `go2_description.urdf` and stock Isaac Go2 use -0.5236..4.5379 for the rear legs. Fixed in `leg.xacro`/`const.xacro` (the real-robot ROS description was affected too); the golden test was updated deliberately (only RL/RR thigh limits differ).

## Prim paths

The converted hierarchy has default prim `/go2_arx5`, root body `Geometry/base`, and the TCP at `Geometry/base/base_link/link1/.../link6/eef_link`. The legacy USD had `base` and `arx5_arm/eef_link`. Body and joint names are unchanged (`base`, `FL_hip`, ..., `joint1`..`joint8`, `eef_link`), so name-based code is unaffected; only literal prim paths matter.

The single source of truth is `common/cerberus_description/go2_arx5_spec.py`: `GO2_ARX5_BASE_PRIM_REL_PATH = "Geometry/base"` and `GO2_ARX5_TCP_PRIM_REL_PATH = "Geometry/base/base_link/link1/link2/link3/link4/link5/link6/eef_link"`, both relative to the robot's default prim (so under `{ENV_REGEX_NS}/Robot/` in a scene). Layout and env code derive their paths from these constants. They are guarded two ways: the conversion script's checks record and verify the base and TCP prim paths of the converted USD, and the constants test in `policy/manipulation/go2_arx5_stack/test_config_constants.py` pins the layout paths to the spec. Changing the hierarchy means changing the constants and passing both checks.

## Mount decision

`GO2_ARX5_MOUNT_XYZ = (0, 0, 0.07)` in `go2_arx5_spec.py`; the URDF applies it and the arm base_link starts at world z 0.07 in the default USD. History: the earlier assembler-built USD had every arm body exactly 0.07 m lower, i.e. effectively mounted at the Go2 `base` origin (z 0), an assembler artefact. Evidence for 0.07: with z=0 the arm base sits fully inside the Go2 body shell; LeggedManip_Lab uses 0.07 and 0.06. UNVERIFIED: the real mounting-plate thickness was not checked in hardware docs (Unitree's mounting spec was never consulted). `layout.py` (`ARM_BASE_Z = ROBOT_BASE_Z + GO2_ARX5_MOUNT_XYZ[2]`) uses the same constant. A mount z=0 variant (`mountz0`) reproduced the legacy behaviour (height 0.231 vs 0.234 m, pitch -15.56 vs -15.51 deg).

To change it: edit the one constant in `go2_arx5_spec.py`, rebuild the URDF, reconvert on a pod, then re-run the hold test and the W2a reach check (W2a and the W2 reach scans depend on the arm base height).

## Policy status (unverified items labelled)

The frozen-policy eval on EA (`eval_frozen_policy_on_fused.py`, deleted 2026-11-06 because W4 retrains from scratch; the numbers below are its recorded results) is NOT a valid URDF-vs-USD discriminator: even the legacy assembler-built USD gives verdict `fails` (fused standing tilt max 59.7 deg; stock Go2 22.4 deg standing vs 2.94 on the old stack), so the Lab 2.3.2-trained checkpoint behaves worse on Lab 3.0 EA even for the stock Go2. On the SAME stack, converted vs legacy are equivalent (standing tilt max 59.6 vs 59.7, push recovery 1.0 at 40/80/120 N both, vx 0.5/1.0 fall rate 0.02 vs 0). The `mountz0` variant: standing tilt 38.2. UNVERIFIED: the old-stack policy eval on the converted USD was not completed (pod killed at the cost cap), so "policy transfers on the converted USD on the old stack" is not shown. The earlier old-stack result for the legacy USD ("transfers": standing pitch -1.07 stock vs -1.33 fused, push recovery 1.0) stands. What to do about the frozen policy on Lab 3.0 is open.

## Adding a new arm or new legs

1. Write the description as a xacro macro (legs: a `sim`-capable xacro like `go2.urdf.xacro` with root link `base`, no world/floating joint, no sensor/rotor links, absolute mesh paths; arm: fetch the URDF and add a patch step like `urdf_patch.patch_arm_urdf`). Check joint limits against the vendor's own URDF per leg (the rear-thigh bug shows copy-paste limits slip through).
2. Add the spec constants (links, joints, mount xyz/rpy, limits, TCP link name, prim-path constants) next to `go2_arx5_spec.py` and the matching `ArticulationCfg`.
3. Rerun, in order: `build_go2_arx5_urdf.py`, `convert_go2_arx5_urdf_to_usd.py` on a pod, `compare_go2_urdf_to_usd.py`, then a hold test. UNVERIFIED: only Go2+ARX5 has been run; the scripts are Go2/ARX5-specific today and need parametrizing for another combo.
4. Look up bodies by name. Keep the stock Go2 and arm sources untouched. Per the project pattern, one self-contained `ArticulationCfg` per combo (as in LeggedManip_Lab), not sub-configs glued together at import.

## Licensing

The ARX_Model repo has no license file: its meshes and the fetched arm URDF are not redistributed and stay gitignored (`third_party/arx-model`). The merged URDF (it references those meshes by absolute path) and the converted USD (it embeds them) are gitignored build outputs under `data/`; never commit them. The xacro, scripts and specs are ours and are committed.

## Arm source and fused asset facts

- Arm: `ARXroboticsX/ARX_Model` X5A (two prismatic fingers), cloned to `third_party/arx-model`.
- Fused robot: 20 DoF (12 leg + 6 arm + 2 fingers), total mass 18.47 kg, arm 3.455 kg. The free-base lean is Go2 PD sag at stock gains (stock legs-only Go2: pitch -11.9 deg on the old stack), not the arm.
- LeggedManip_Lab commits each of its 7 combo USDs as a finished binary with a hand-written `ArticulationCfg`; it has no authoring script. Scripting the build here is deliberately ahead of that precedent.

## History

The first build path used Isaac Sim's Robot Assembler (`isaacsim.robot_setup.assembler`): a one-shot `RobotAssembler.assemble_rigid_bodies` call on Sim 5.1, plus a Sim 6.1 workflow adapter. It was removed from the repo (script, its test and the adapter) in favour of the URDF merge. Findings kept for reference: on Sim 6.1 the public begin/assemble/finish workflow failed four times on pods (null or expired layer errors in `finish_assemble`, missing `Layer.SaveAs`, a session sublayer that was never removed); NVIDIA made `assemble_rigid_bodies` private in extension changelog 3.3.2; the 6.1 workflow targets variant-structured `Robots_Multiphysics` assets with IsaacRobotAPI relationships, and our flat stock Go2 likely lacks the relationships `finish_assemble` writes (hypothesis for the failures, unproven). Alternatives searched: Isaac Lab 3.0 has no robot-attach tool; Arena composes at environment level only; Lab issue #2280 has no maintainer answer. The assembler path also produced the z=0 mount artefact described under Mount decision.
