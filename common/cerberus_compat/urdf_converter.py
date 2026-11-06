"""Isaac Lab UrdfConverter shims (pure python; isaaclab is imported lazily and only by `import_urdf_converter`)."""

from __future__ import annotations

import dataclasses
import importlib

# Modules that may export UrdfConverter/UrdfConverterCfg, tried in order (Lab 2.3.2 first).
URDF_CONVERTER_MODULES = ("isaaclab.sim.converters", "isaaclab.sim")


def framework_versions() -> dict:
    """Installed isaaclab / isaacsim versions (importlib.metadata; None when the distribution is not installed)."""
    from importlib import metadata

    out: dict = {}
    for pkg in ("isaaclab", "isaacsim"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def filter_cfg_kwargs(cfg_cls: type, candidates: dict) -> tuple[dict, list[str]]:
    """Keep only `candidates` that are dataclass fields of `cfg_cls`. Returns (kept, dropped names)."""
    names = {f.name for f in dataclasses.fields(cfg_cls)}
    kept = {k: v for k, v in candidates.items() if k in names}
    return kept, sorted(set(candidates) - set(kept))


def import_urdf_converter(module_names: tuple[str, ...] = URDF_CONVERTER_MODULES):
    """(UrdfConverter, UrdfConverterCfg, module name) from the first module exporting both; else RuntimeError."""
    errors = []
    for name in module_names:
        try:
            module = importlib.import_module(name)
            return module.UrdfConverter, module.UrdfConverterCfg, name
        except (ImportError, AttributeError) as e:
            errors.append(f"{name}: {e!r}")
    raise RuntimeError(
        f"could not import UrdfConverter/UrdfConverterCfg from any of {list(module_names)}: {errors}. STOP and "
        "report this to the orchestrator: do not patch Isaac Lab."
    )
