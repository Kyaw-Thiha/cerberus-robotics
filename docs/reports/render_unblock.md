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

### Accepted stills (the first that pass every check)

| still | mean luminance | clipped | hue bins >= 5% | verdict |
|---|---|---|---|---|
| `figures/render_unblock_ego.png` (EGO, 1920x1080) | 157.8 | 0.1% | 4 | **PASS** |
| `figures/render_unblock_chase.png` (CHASE, 1920x1080) | 147.2 | 0.1% | 6 | **PASS** |

What they show (opened and described, not inferred): the EGO still looks along an L-shaped white sofa with two
cushions into an open kitchen (cabinets, range hood, oven, fridge), with a coffee table, a framed picture and a
glossy floor carrying the scene's own rect-light speculars; the robot's own shoulder still fills the right edge of
the EGO frame, so that offset is not final. The CHASE still is the wide look: the full living area and kitchen
(sink, faucet, kettle, microwave, fridge), a dining table with four chairs, a hallway shoe rack with shoes, wall
art, and the robot's arm at the bottom centre. Neither is "flat colour" and neither is the scene's original MDL
look: the textures are the dataset's own files (the MDL sources name them) rendered through UsdPreviewSurface.

Settings recorded for every render: dome intensity 300, scene lights on, spawn `-3.75,0.73,0.0`,
EGO offset `0.38,0,0.30` pitch 14, CHASE distance 1.3 / height 1.35 / pitch 22 for stills and
1.15 / 1.05 / 15 for clips.

## The host state after a crash storm (measured 2026-11-23)

Repeated failed launches leave the L40 unusable for CUDA even though `nvidia-smi` reports an idle card:

```
[omni.physx.tensors.plugin] CUDA error: CUDA-capable device(s) is/are busy or unavailable
[omni.physx.tensors.plugin] Failed to create primary CUDA context
```

and, before that, `torch.cuda.init()` raising `RuntimeError: No CUDA GPUs are available` inside a container whose
`nvidia-smi -L` lists the GPU. No Xid appears in `dmesg`, no process holds `/dev/nvidia*`, and a fresh container
still fails. `sudo shutdown -r now` clears it; the next launch then runs normally. Practical rule adopted here: a burst
of three identical GPU-context failures means reboot the VM, do not retry.

## R3 result: a textured still that passes the look checks

The route that works, in order:

1. `scripts/scene_materials_to_preview.py` converts every material to `UsdPreviewSurface` and **reads the KooPbr
   instance MDL sources** for the maps and the `reflect` colour (`texture_2d("./textures/...")`), because the USD
   shader inputs are empty for those materials. On this scene: 3,293 materials, **1,678 with a diffuse texture**,
   **2,866 with a normal map**, 3,847 bindings rebound, 0 missing files.
2. The same script **deactivates every MDL material prim** after the re-bind. This is the step that made rendering
   work at all: leaving them active made the renderer compile all 3,293 modules (`[MDLC:COMPILER] comp error`,
   `MdlModule is NOT in the DB`), which stalled the scene build for minutes per launch. With them inactive the log
   has zero MDLC lines, the scene builds, and the robot stands on the floor (base z = 0.40 m) instead of falling
   through it (z = -6.66 m in the broken configuration).
3. The visual layer is loaded through `CERBERUS_SCENE_USD_OVERRIDE=.../scene_preview_metric.usd`, so physics keeps
   the navigation variant and only the visual layer changes.
4. Lighting: the scene carries **137 of its own lights** (112 `RectLight`, 24 `SphereLight`, 1 `DistantLight`); the
   locomotion env's dome light (2000 by default) blows the room out. Fixed setting used for every render here:
   **dome intensity 300** (`--dome-intensity 300`), the scene's own lights on.
5. The default robot spawn sits against a wardrobe and the EGO camera sits inside the robot's own shoulder. Spawn,
   chase geometry and ego offset are now options; the accepted stills use the occupancy grid's most open cell
   (`--spawn -3.75,0.73,0.0`) with `--ego-offset 0.38,0.0,0.30 --ego-pitch 14` and
   `--chase-distance 1.3 --chase-height 1.35 --chase-pitch 22`.

Frame rate: one 1920x1080 still costs ~40-60 s of GPU (scene build once, then ~1 s per captured frame through the
Replicator orchestrator). `sim.render()` alone leaves the RGB annotator empty on this stack, so capture steps the
orchestrator.

## Why the render stayed grey with every material converted (root cause, fixed)

Converting all 3,293 materials to `UsdPreviewSurface` changed nothing on screen. The decisive test the owner asked
for: bind one flat red material to every material prim and re-render the identical view.

| test | result |
|---|---|
| red layer with the preview scope at `/PreviewMaterials` (stage root) | **grey** (the override was ignored) |
| same, with the preview materials at `<scene root>/__PreviewMaterials` | **red** (the override applies) |

The USD warnings from the first case name the mechanism exactly:

```
Warning: ... The relationship target </PreviewMaterials/m_Root_Meshes_...> from
</Root/Meshes/.../Component_3.material:binding> in layer @.../materials_preview.usda@ refers to a path outside the
scope of the reference from </Scene>.  Ignoring.
```

The preview wrapper references the scene into `/Scene`. A relationship target that is an **absolute path outside that
reference scope** is dropped by USD, so every surface fell back to the default material -- hence one flat grey. The
fix is to author the preview materials *inside* the scene's own root (`/Root/__PreviewMaterials/...`, the scene's
default prim), so every binding target is in scope. `Umbrella` note: this is not a renderer quirk; it is USD
composition, and it is why re-binding "worked" in the layer but not in the picture.

Instanceable roots: `prim.IsInstanceable()` is **false for every prim** in this scene (`instances_deinstanced: 0`),
so per-instance overrides were never the obstacle -- the reference scope was.

With the in-scope placement, the dataset's textures appear: marble/stone floor tiles, brass candle legs, a dark
cushion on the stool, seams and sheen on the white wardrobe doors, the kitchen table and chairs beyond.

Exposure: the `--dome-intensity` flag measurably does nothing on this stage (a 300 and a 120 run produce identical
mean luminance, 222.4 and 222.4), so the brightness is the scene's own 137 lights. The EGO view reads 134
(passes the gate's 70-170 band); the chase views read 211-224 and fail it. No threshold was changed to make them
pass.

QA honesty: `scripts/host/clip_qa.py` thresholds were untouched all session (luminance 70-170, clipping <= 5 %,
>= 3 hue bins at >= 5 %). Measured hue bins: still ego 5, still chase 2; walk_ego clip 3/4/3/3/3; walk_overview
3/4/1/1/1; arm_closeup 2/1/1/1/1; arm_chase (pre-fix materials) 2-3.

## What shipped, and what is still open

Gated this session (clip QA unchanged):

| artifact | verdict |
|---|---|
| `walk_ego.mp4` (1080p30, 20 s, EGO, retrained policy, real textures) | **PASS** - luminance 120-133, 0.000-0.001 clipped, 4-5 hue bins, camera-pose checks ok |
| `walk_ego_split.mp4` (same walk + top-down planner map) | **PASS** - luminance 83-86, 0.000 clipped, 3-4 hue bins |
| `teaser_walk_ego_10s.gif` (10 s, 640 px) | 1.36 MB, within the 8 MB cap |
| `still_ego.png` 1920x1080 | **PASS** - luminance 134, 5 hue bins |
| `arm_ego.mp4` (arm demo, 20 s) | 4/5 frames pass; frame 0 reads 173.8 (limit 170) |
| `arm_chase.mp4`, `arm_closeup.mp4`, `walk_chase.mp4`, `walk_overview.mp4` | rendered, kept as evidence; fail the luminance band (199-260) and, for the close-up, the camera lands inside furniture |

Open, with the measurement that closes it for now: **exposure**. The RTX tonemapper exposure (-1.5 stops: 199.2 vs
199.3), hiding the emissive HDR sphere (222.5) and a 0.45 scale on all 137 scene lights (222 -> 199) do not bring this
white-lit home into 70-170. A next attempt should drive the render product's own auto-exposure histogram rather than
guessing setting keys, or move the camera to darker rooms.

Track M (physics mission), SLAM and the K2 matrix were not attempted: the session's remaining time went to the
render unblock, and the two earlier VMs were lost to their dead-man on CGNAT address changes.
