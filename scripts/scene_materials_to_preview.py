#!/usr/bin/env python3
"""Rewrite a GRScenes home's MDL materials as ``UsdPreviewSurface`` so the RTX renderer can texture it.

Why: every GRScenes material binds an MDL shader (``mdl:surface`` -> ``Materials/DayMaterial.mdl``), and MDL
resolves only when ``MDL_SYSTEM_PATH`` points at the dataset's 810-file ``Materials`` folder. Pointing Kit at that
folder crashes it at startup on this image (measured 2026-11-22), and with the path unset every material fails to
compile and the scene renders as one flat red-orange. The dataset's textures ARE all present on disk (0 missing
files), so this script bypasses MDL entirely: it builds an equivalent ``UsdPreviewSurface`` material per MDL
material, points ``UsdUVTexture`` nodes at the same texture files, and re-binds the visual geometry to the new
materials. Physics is untouched (collision geometry does not use material bindings).

It also writes a metre-scale wrapper for the previewed scene and records the path, because GRScenes houses are
authored in centimetres and the simulator spawns them 100x too big otherwise (see
``common/cerberus_worlds/scene_geometry.py``). Point ``CERBERUS_SCENE_USD_OVERRIDE`` at that wrapper.

Nothing is edited in place: the original stage is opened read-only, the new material layer sublayers it, and every
material binding is overridden in the new layer.

Usage (pxr only; no Kit, no GPU)::

    pixi run python scripts/scene_materials_to_preview.py --usd /path/scene/start_result_navigation.usd \\
        --out-dir /path/scene --json preview.json

Acceptance is in the report: how many materials were converted, how many carry a diffuse texture, and the list of
texture files that do NOT exist (which is what would render flat).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_TEXTURE_2D = re.compile(r'texture_2d\(\s*"([^"]+)"')
_COLOR_CALL = re.compile(r"reflect\s*:\s*color\(([^)]*)\)")
_DIFFUSE_CALL = re.compile(r"diffuse\s*:\s*color\(([^)]*)\)")


def mdl_texture_slots(text: str) -> dict[str, str]:
    """Texture paths a material instance MDL names, classified by filename: normal / roughness / metallic / diffuse.

    GRScenes material instances are `KooPbr::KooMtl(...)` calls that name their maps inside the MDL source
    (`texture_2d("./textures/....jpg", ::tex::gamma_srgb, "")`), not as USD shader inputs, so the USD alone shows no
    textures. Measured 2026-11-23: 523 `texture_2d` references across the 812 modules.
    """
    slots: dict[str, str] = {}
    for path in _TEXTURE_2D.findall(text):
        name = path.lower()
        if any(tag in name for tag in ("_n.", "_normal", "normal_", "_bump")):
            slot = "normal"
        elif any(tag in name for tag in ("_m.", "_mask", "rough", "metal", "_orm", "_rma")):
            slot = "masks"
        else:
            slot = "diffuse"
        slots.setdefault(slot, path)
    return slots


def mdl_colour(text: str) -> tuple[float, float, float] | None:
    """The constant colour a KooPbr instance names for `reflect` (UE4's base colour), if it names one."""
    match = _COLOR_CALL.search(text) or _DIFFUSE_CALL.search(text)
    if not match:
        return None
    parts = [p.strip().rstrip("fF") for p in match.group(1).split(",")]
    try:
        values = [float(v) for v in parts[:3]]
    except ValueError:
        return None
    return tuple(values) if len(values) == 3 else None

PREVIEW_ROOT = "/PreviewMaterials"
LAYER_NAME = "materials_preview.usda"
WRAPPER_NAME = "scene_preview_metric.usd"

_TEXTURE_NAME_HINTS = ("texture", "tex", "file", "albedo", "diffuse", "basecolor", "base_color", "normal",
                       "rough", "metal", "emissive")
_COLOR_NAME_HINTS = ("diffuse", "basecolor", "base_color", "albedo", "color")


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def classify_asset_input(input_name: str) -> str:
    """Map a shader input name to a material slot: 'normal', 'roughness', 'metallic', 'emissive' or 'diffuse'."""
    name = input_name.lower()
    if "normal" in name:
        return "normal"
    if "rough" in name:
        return "roughness"
    if "metal" in name:
        return "metallic"
    if "emis" in name:
        return "emissive"
    return "diffuse"


def classify_color_input(input_name: str) -> str | None:
    """Map a shader input name to a colour slot, or None when it is not a base-colour constant."""
    name = input_name.lower()
    if "emiss" in name or "normal" in name:
        return "emissive" if "emiss" in name else None
    if any(hint in name for hint in _COLOR_NAME_HINTS):
        return "diffuse"
    return None


def _texture_candidates(shader_inputs: dict, usd_dir: Path) -> dict[str, dict]:
    """Pick one texture per slot from ``{input_name: value}``; asset values may be ``Sdf.AssetPath`` already."""
    found: dict[str, dict] = {}
    for name, value in shader_inputs.items():
        asset = None
        if isinstance(value, dict) and "asset" in value:
            asset = value
        if asset is None:
            continue
        path = asset.get("resolved") or asset.get("asset")
        if not path or str(path).endswith(".mdl"):
            continue
        if not any(hint in name.lower() for hint in _TEXTURE_NAME_HINTS):
            continue
        slot = classify_asset_input(name)
        candidate = Path(str(path))
        if not candidate.is_absolute():
            candidate = usd_dir / candidate
        entry = {"input": name, "path": str(candidate), "exists": candidate.exists()}
        # prefer a file that exists over one that does not
        if slot not in found or (entry["exists"] and not found[slot]["exists"]):
            found[slot] = entry
    return found


def _colour_constants(shader_inputs: dict) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for name, value in shader_inputs.items():
        slot = classify_color_input(name)
        if slot is None or isinstance(value, dict):
            continue
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                rgb = tuple(float(v) for v in value[:3])
            except (TypeError, ValueError):
                continue
            out.setdefault(slot, rgb)
    return out


def _shader_inputs(shader) -> dict:
    from pxr import Sdf

    values: dict = {}
    for inp in shader.GetInputs():
        try:
            value = inp.Get()
        except Exception:  # noqa: BLE001, S112 - a broken input must not stop the conversion
            continue
        if value is None:
            continue
        base = inp.GetBaseName()
        if isinstance(value, Sdf.AssetPath):
            values[base] = {"asset": value.path, "resolved": value.resolvedPath or None}
        elif isinstance(value, (str, int, float, bool)):
            values[base] = value
        elif hasattr(value, "__len__") and not isinstance(value, str):
            try:
                values[base] = [float(v) for v in value]
            except (TypeError, ValueError):
                values[base] = str(value)
        if isinstance(value, Sdf.AssetPath):
            values[base]["resolved"] = value.resolvedPath or None
    return values


def _mdl_attributes(shader) -> dict:
    from pxr import Sdf

    found: dict = {}
    for attr in shader.GetAttributes():
        if "mdl" not in attr.GetName().lower():
            continue
        value = attr.Get()
        if isinstance(value, Sdf.AssetPath):
            found[attr.GetName()] = {"asset": value.path, "resolved": value.resolvedPath or None}
        elif value is not None:
            found[attr.GetName()] = str(value)
    return found


def material_description(material_prim, base_dir: Path = Path('.')) -> dict:
    """Everything the converter needs from one MDL material: textures, colours, roughness/metallic, source module."""
    from pxr import UsdShade

    material = UsdShade.Material(material_prim)
    shaders: list[dict] = []
    stack = [material_prim]
    while stack:
        prim = stack.pop()
        children = prim.GetChildren()
        stack.extend(children)
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            shaders.append({"path": str(prim.GetPath()), "shader_id": str(shader.GetShaderId() or ""),
                            "inputs": {**_shader_inputs(shader), **_mdl_attributes(prim)}})

    merged: dict = {}
    for shader in shaders:
        merged.update(shader["inputs"])
    surfaces = [s for s in shaders if s["shader_id"] == "mdl:surface"] or shaders

    described = {
        "prim_path": str(material_prim.GetPath()),
        "shaders": shaders,
        "surface_sources": [s["path"] for s in surfaces],
        "textures": _texture_candidates(merged, base_dir),
        "colours": _colour_constants(merged),
        "mdl_source": next((v["asset"] for k, v in merged.items()
                            if isinstance(v, dict) and "mdl" in k.lower() and v.get("asset")), None),
    }
    _fill_from_mdl(described, base_dir)
    return described


def _fill_from_mdl(item: dict, base_dir: Path) -> None:
    """Fill missing textures/colours from the material's own MDL source (KooPbr instances name them there)."""
    source = item.get("mdl_source")
    if not source:
        return
    path = Path(str(source))
    if not path.is_absolute():
        path = base_dir / path
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        item["mdl_read"] = "missing"
        return
    item["mdl_read"] = "ok"
    slots = mdl_texture_slots(text)
    for slot, rel in slots.items():
        if slot in item["textures"]:
            continue
        candidate = (path.parent / rel).resolve()
        item["textures"][slot] = {"input": "mdl", "path": str(candidate), "exists": candidate.exists()}
    if "diffuse" not in item["colours"]:
        colour = mdl_colour(text)
        if colour is not None:
            item["colours"]["diffuse"] = colour


def collect(stage, base_dir: Path = Path(".")) -> tuple[list[dict], dict[str, str]]:
    """All MDL-backed materials plus (prim_path -> bound material path) for every binding relationship."""
    from pxr import UsdShade

    materials = [UsdShade.Material(prim) for prim in stage.Traverse() if prim.IsA(UsdShade.Material)]
    described = [material_description(m.GetPrim(), base_dir) for m in materials]
    bindings: dict[str, str] = {}
    for prim in stage.Traverse():
        for rel in prim.GetRelationships():
            if rel.GetName() != "material:binding":
                continue
            targets = rel.GetTargets()
            if targets:
                bindings[str(prim.GetPath())] = str(targets[0])
    return described, bindings


def uv_sets_by_material(stage) -> dict[str, str]:
    """material path -> the UV primvar name used by the meshes bound to it.

    GRScenes meshes carry ``st`` on some models and ``st1``/``st2`` on others (measured 2026-11-23: 215 of 401
    sampled meshes have ``st``, 160 have ``st1``). A texture whose ``st`` primvar does not exist samples its
    fallback -- white -- which is why a textured material can still render flat.
    """

    found: dict[str, str] = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() not in {"Mesh", "GeomMesh"}:
            continue
        names = {a.GetName()[len("primvars:"):] for a in prim.GetAttributes()
                 if a.GetName().startswith("primvars:") and not a.GetName().endswith(":indices")}
        uv = next((n for n in ("st", "st1", "st2", "st3") if n in names), None)
        if uv is None:
            continue
        for rel in prim.GetRelationships():
            if rel.GetName() != "material:binding":
                continue
            for target in rel.GetTargets():
                found.setdefault(str(target), uv)
    return found


def _primvar_name(stage) -> str:
    """The mesh UV primvar name the scene actually authors ('st' for UE4 exports, but check)."""
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if name.startswith("primvars:") and name.endswith(":indices"):
                continue
            if name.startswith("primvars:st"):
                return name.split(":")[1] if name.count(":") == 1 else "st"
    return "st"


def build_layer(usd_path: Path, out_layer: Path, *, rebind: bool = True) -> dict:
    """Write `out_layer`: UsdPreviewSurface materials + (optionally) re-bound geometry. Returns the report."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    usd_path = Path(usd_path).resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"cannot open {usd_path}")
    usd_dir = usd_path.parent
    described, bindings = collect(stage, usd_dir)
    uv_primvar = _primvar_name(stage)
    uv_by_material = uv_sets_by_material(stage)
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))

    out = Usd.Stage.CreateInMemory()
    out.GetRootLayer().subLayerPaths.append(str(usd_path))
    out.GetRootLayer().metersPerUnit = mpu
    scope = UsdGeom.Scope.Define(out, PREVIEW_ROOT)

    converted: dict[str, str] = {}  # original material path -> new material path
    report_materials = []
    for index, item in enumerate(described):
        new_path = f"{PREVIEW_ROOT}/m_{_sanitize(item['prim_path'].strip('/').replace('/', '_'))}"
        if new_path in converted.values():  # several materials can sanitise to the same name
            new_path = f"{new_path}_{index}"
        mat = UsdShade.Material.Define(out, new_path)
        surface = UsdShade.Shader.Define(out, f"{new_path}/PreviewSurface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        surface.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)

        diffuse = item["colours"].get("diffuse")
        if diffuse:
            surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse)

        uv_name = uv_by_material.get(item["prim_path"], uv_primvar)
        textures = item["textures"]
        tex_summary = {}
        if "diffuse" in textures and textures["diffuse"]["exists"]:
            tex = textures["diffuse"]
            reader = UsdShade.Shader.Define(out, f"{new_path}/stReader")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(uv_name)
            uv_tex = UsdShade.Shader.Define(out, f"{new_path}/diffuseTex")
            uv_tex.CreateIdAttr("UsdUVTexture")
            uv_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex["path"]))
            uv_tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            uv_tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            uv_tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            uv_tex.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set((1.0, 1.0, 1.0, 1.0))
            uv_tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),
                                                                               "result")
            surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                uv_tex.ConnectableAPI(), "rgb")
            tex_summary["diffuse"] = tex
        if "normal" in textures and textures["normal"]["exists"]:
            tex = textures["normal"]
            reader = UsdShade.Shader.Define(out, f"{new_path}/normalStReader")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(uv_name)
            ntex = UsdShade.Shader.Define(out, f"{new_path}/normalTex")
            ntex.CreateIdAttr("UsdUVTexture")
            ntex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex["path"]))
            ntex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            ntex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            ntex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2.0, 2.0, 2.0, 1.0))
            ntex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1.0, -1.0, -1.0, 0.0))
            surface.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(ntex.ConnectableAPI(), "rgb")
            tex_summary["normal"] = tex

        mat.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        converted[item["prim_path"]] = new_path
        report_materials.append({
            "source": item["prim_path"], "preview": new_path,
            "mdl_source": item["mdl_source"],
            "textures": {k: v["path"] for k, v in tex_summary.items()},  # only files that exist are bound
            "missing_textures": [v["path"] for v in textures.values() if not v["exists"]],
            "colours": {k: list(v) for k, v in item["colours"].items()},
        })

    deactivated = 0
    if rebind:
        # Deactivate the MDL material prims themselves. Re-binding alone leaves them in the stage, and the renderer
        # then tries to compile every one of them (measured 2026-11-23: an `[MDLC:COMPILER] comp error` per material
        # and minutes of stall before the first frame). With the geometry on UsdPreviewSurface they are dead weight.
        for item in described:
            prim = out.OverridePrim(item["prim_path"])
            try:
                prim.SetActive(False)
                deactivated += 1
            except Exception:  # noqa: BLE001, S112 - an older pxr without SetActive must not stop the conversion
                continue

    rebound, unbound = 0, 0
    if rebind:
        for prim_path, material_path in bindings.items():
            target = converted.get(material_path)
            prim = out.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                unbound += 1
                continue
            rel = prim.GetRelationship("material:binding")
            if target:
                rel.SetTargets([target])
                rebound += 1
            else:
                unbound += 1

    out_layer = Path(out_layer)
    out_layer.parent.mkdir(parents=True, exist_ok=True)
    out.GetRootLayer().Export(str(out_layer))

    missing = sorted({p for m in report_materials for p in m["missing_textures"]})
    return {
        "usd": str(usd_path),
        "layer": str(out_layer),
        "materials": len(report_materials),
        "materials_with_diffuse_texture": sum(1 for m in report_materials if "diffuse" in m["textures"]),
        "materials_with_normal_texture": sum(1 for m in report_materials if "normal" in m["textures"]),
        "bindings": len(bindings),
        "rebound": rebound,
        "mdl_materials_deactivated": deactivated,
        "unbound": unbound,
        "uv_primvar": uv_primvar,
        "uv_primvars_used": {k: v for k, v in list(uv_by_material.items())[:10]},
        "missing_texture_count": len(missing),
        "missing_textures": missing[:20],
        "samples": report_materials[:5],
    }


def write_preview_wrapper(scene_usd: Path, layer: Path) -> Path:
    """A metre-scale wrapper that puts `layer` above `scene_usd` (so GRScenes' centimetre data spawns at 1 m/unit)."""
    from pxr import Gf, Usd, UsdGeom

    scene_usd = Path(scene_usd).resolve()
    layer = Path(layer).resolve()
    stage = Usd.Stage.Open(str(scene_usd))
    if stage is None:
        raise RuntimeError(f"cannot open {scene_usd}")
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    up_axis = UsdGeom.GetStageUpAxis(stage)

    preview = scene_usd.with_name("scene_preview.usda")
    preview.write_text(
        "#usda 1.0\n"
        "(\n"
        f"    subLayers = [@{layer.name}@, @{scene_usd.name}@]\n"
        ")\n"
    )
    out_path = scene_usd.with_name(WRAPPER_NAME)
    if out_path.exists():
        out_path.unlink()
    wrapper = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageMetersPerUnit(wrapper, 1.0)
    UsdGeom.SetStageUpAxis(wrapper, up_axis)
    xform = UsdGeom.Xform.Define(wrapper, "/Scene")
    if abs(mpu - 1.0) > 1e-9:
        xform.AddScaleOp().Set(Gf.Vec3d(mpu, mpu, mpu))
    # Reference the scene's own default prim by name: the preview layer is a bare subLayers list and has no
    # defaultPrim of its own, so `@file@<defaultPrim>` would not resolve.
    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        xform.GetPrim().GetReferences().AddReference(f"./{preview.name}", default_prim.GetPath())
    else:
        xform.GetPrim().GetReferences().AddReference(f"./{preview.name}")
    wrapper.GetRootLayer().defaultPrim = "Scene"
    wrapper.Save()
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--usd", required=True, help="the scene USD to convert (visual layer)")
    parser.add_argument("--out-dir", default=None, help="where the layer and wrapper go (default: beside --usd)")
    parser.add_argument("--json", default=None)
    parser.add_argument("--no-rebind", action="store_true", help="write the materials but leave bindings alone")
    args = parser.parse_args(argv)

    usd = Path(args.usd)
    out_dir = Path(args.out_dir) if args.out_dir else usd.parent
    report = build_layer(usd, out_dir / LAYER_NAME, rebind=not args.no_rebind)
    wrapper = write_preview_wrapper(usd, out_dir / LAYER_NAME)
    report["wrapper"] = str(wrapper)
    report["override_env"] = f"CERBERUS_SCENE_USD_OVERRIDE={wrapper}"
    text = json.dumps(report, indent=2)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    print(f"\n[CERBERUS_SCENE_USD_OVERRIDE={wrapper}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
