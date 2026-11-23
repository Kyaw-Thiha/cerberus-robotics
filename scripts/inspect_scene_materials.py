#!/usr/bin/env python3
"""Inspect a GRScenes USD's materials: how many, what shaders they bind, and whether the referenced files exist.

Runs with plain ``pxr`` (the Isaac Sim python has it; no Kit app, no GPU, no rendering). This is the first step of
the look work: the scene rendered as one flat red-orange, which is what an unresolved material looks like, and the
question is whether the bound shaders are MDL modules (need ``MDL_SYSTEM_PATH``) or texture-backed
``UsdPreviewSurface`` (need the texture files, which the fetch may not have brought).

Usage::

    /isaac-sim/python.sh scripts/inspect_scene_materials.py --usd scene.usd [--json out.json] [--sample 6]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pxr import Sdf, Usd, UsdShade


def mdl_attributes(prim: Usd.Prim) -> dict[str, object]:
    """``mdl:*`` attributes on a shader prim (``info:mdl:sourceAsset`` and friends hold the MDL module path)."""
    found: dict[str, object] = {}
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if "mdl" not in name.lower():
            continue
        value = attr.Get()
        if isinstance(value, Sdf.AssetPath):
            found[name] = {"asset": value.path, "resolved": value.resolvedPath or None}
        elif value is not None:
            found[name] = str(value)
    return found


def shader_inputs(shader: UsdShade.Shader) -> dict[str, object]:
    out: dict[str, object] = {}
    for inp in shader.GetInputs():
        name = inp.GetBaseName()
        if not any(key in name.lower() for key in ("texture", "file", "asset", "color", "diffuse", "base",
                                                    "reflect", "mdl")):
            continue
        value = inp.Get()
        if value is None:
            continue
        if isinstance(value, Sdf.AssetPath):
            out[name] = {"asset": value.path, "resolved": value.resolvedPath or None}
        elif isinstance(value, (str, int, float, bool)):
            out[name] = value
        elif hasattr(value, "__len__"):
            try:
                out[name] = [float(v) for v in value]
            except (TypeError, ValueError):
                out[name] = str(value)
        else:
            out[name] = str(value)
    return out


def asset_exists(entry: object, usd_dir: Path) -> bool | None:
    if not isinstance(entry, dict):
        return None
    path = entry.get("resolved") or entry.get("asset")
    if not path:
        return None
    candidate = Path(str(path))
    if not candidate.is_absolute():
        candidate = usd_dir / candidate
    return candidate.exists()


def inspect(usd_path: Path, sample: int = 6) -> dict:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise SystemExit(f"could not open {usd_path}")
    usd_dir = usd_path.parent
    materials, mdl_refs, texture_refs, missing = [], 0, 0, []
    shader_kinds: dict[str, int] = {}
    prim_count = 0
    for prim in stage.Traverse():
        prim_count += 1
        if not prim.IsA(UsdShade.Material):
            continue
        material = UsdShade.Material(prim)
        shaders = []
        # walk the material's own subtree for shaders
        stack = list(material.GetPrim().GetChildren())
        while stack:
            child = stack.pop()
            stack.extend(child.GetChildren())
            if not child.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(child)
            shader_id = str(shader.GetShaderId() or "unknown")
            shader_kinds[shader_id] = shader_kinds.get(shader_id, 0) + 1
            inputs = {**shader_inputs(shader), **mdl_attributes(child)}
            if "mdl" in shader_id.lower() or "mdl" in json.dumps(inputs).lower():
                mdl_refs += 1
            for value in inputs.values():
                if isinstance(value, dict) and "asset" in value:
                    texture_refs += 1
                    exists = asset_exists(value, usd_dir)
                    if exists is False:
                        missing.append(value.get("asset"))
            shaders.append({"path": str(child.GetPath()), "shader_id": shader_id, "inputs": inputs})
        sources = []
        for out in material.GetOutputs():
            try:
                connected = out.GetConnectedSources()
            except Exception:  # noqa: BLE001 - the pxr API differs by version; the inputs below are the real evidence
                connected = []
            for entry in connected:
                for src in (entry if isinstance(entry, list) else [entry]):
                    target = getattr(src, "source", None)
                    getter = getattr(target, "GetPath", None)
                    sources.append(f"{out.GetBaseName()}<-{getter() if getter else target}")
        materials.append({"path": str(prim.GetPath()), "surface_sources": sources, "shaders": shaders})
    report = {
        "usd": str(usd_path),
        "prim_count": prim_count,
        "material_count": len(materials),
        "mdl_shader_nodes": mdl_refs,
        "asset_inputs": texture_refs,
        "missing_assets": sorted(set(missing))[:20],
        "missing_asset_count": len(set(missing)),
        "shader_id_histogram": dict(sorted(shader_kinds.items(), key=lambda kv: -kv[1])),
        "samples": materials[:sample],
    }
    report["mdl_system_path"] = os.environ.get("MDL_SYSTEM_PATH", "")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--usd", required=True)
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument("--sample", type=int, default=6)
    args = parser.parse_args(argv)
    report = inspect(Path(args.usd), args.sample)
    text = json.dumps(report, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    for material in report["samples"]:
        print(f"\n-- {material['path']} sources={material['surface_sources']}")
        for shader in material["shaders"][:3]:
            print(f"   {shader['shader_id']} {shader['path']}")
            for name, value in list(shader["inputs"].items())[:6]:
                print(f"     {name} = {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
