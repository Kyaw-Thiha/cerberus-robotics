# Render unblock: getting a textured RTX still of the ARM robot in a GRScenes home

Session 3 (2026-11-23). Host: Hyperstack L40 48 GB, `cerberus-scene3` id 1178616, driver 595.58.03, the
`cerberus-remote-v6:v6-ga` container (Isaac Sim 6.2.0, Isaac Lab 3.0 GA). Scene: `MV7J6NIKTKJZ2AABAAAAADQ8_usd`.

The prior state (session 2): the scene rendered as one flat red-orange and camera launches died at the scene build.
The question this document answers: which rung of the ladder (R1 diagnose, R2 variant, R3 material conversion,
R4 other scene, R5 bundled scene) produces a still that passes the look checks.

## R1: diagnosis (mdlc and logs)

**There is no `mdlc` in the Isaac Sim 6.2.0 container.** `find / -xdev -name 'mdlc*'` returns nothing; what the image
ships is the MDL *runtime* (`/isaac-sim/kit/mdl/{core,mtlx,rtx}` with the source `.mdl` modules and
`/isaac-sim/extscache/omni.mdl.neuraylib-*`), not the standalone MDL SDK compiler. Compiling one scene MDL with
`mdlc`, as the previous agent planned, is therefore not possible on this image without fetching NVIDIA's MDL SDK
separately; the rung is answered with the runtime's own evidence instead.

What the scene's material library actually is (measured on the VM):

| quantity | value |
|---|---|
| `Materials/*.mdl` | 812 |
| `Materials/Textures/*` | 2,998 (4.7 GB) |
| top imports in those modules | `::state::normal` (774), `::anno::description` (774), `::KooPbr::KooMtl` (690), `::base::texture_return` (212), `::tex::gamma_mode` (188) |

So the materials are KooPbr-based MDL. Session 2's probe (no MDL path, `CERBERUS_SKIP_MDL_PATH=1`) produced the
flat colour; with the path set, Kit crashed during the scene build. The runtime errors from that run
(`MdlModule is NOT in the DB`, `USD_MDL ... is Invalid`) are consistent with the module search path not being
inherited by the render process (Kit reads `MDL_SYSTEM_PATH` at startup).

`MDL_SYSTEM_PATH` inheritance: `scripts/host/run_session.sh` exports `CERBERUS_SKIP_MDL_PATH=1`, and
`common/cerberus_worlds/arena_assets.ensure_mdl_system_path` then refuses to set the variable, which is why the
render process never saw a path. `scene_stills.py --mdl` sets `CERBERUS_SKIP_MDL_PATH=0` before `AppLauncher`, and
the backend then points `MDL_SYSTEM_PATH` at `<scenes>/grscenes/home/Materials` and copies the UE4 helper modules in.
The process, its children and tmux inherit it because it is set in the parent python before the app is created.

## R3: material conversion (the route that was built)

`scripts/scene_materials_to_preview.py` (new, unit-tested on a synthetic stage) walks the stage, reads every material's
MDL shader inputs (texture asset paths, base-colour constants, normal map), writes an equivalent `UsdPreviewSurface`
material with `UsdUVTexture` nodes pointing at the same texture files, re-binds the visual geometry to those
materials, and writes a metre-scale wrapper (`scene_preview_metric.usd`) whose `subLayers` put the override layer
above the original scene. The dataset itself is opened read-only.

`CERBERUS_SCENE_USD_OVERRIDE` makes the sim load that wrapper; it is honoured inside
`common/cerberus_worlds/arena_assets.sim_usd_path` (tested).

_(results of the still runs are appended below as they finish)_
