"""The authors' spherical-slice ray pattern, as a first-class Isaac Lab pattern config.

Their `RaySphericalSlicePatternCfg` (third_party/jeplo, GPL-3) is 25 vertical rays over -4.5..45.5 deg and 60
horizontal rays over -60..60 deg, with directions

    x = cos(v) cos(h),  y = cos(v) sin(h),  z = sin(v)

in the sensor frame, plus the 180-degree x offset rotation on the sensor. Isaac Lab's stock `LidarPatternCfg`
does not produce the same directions, which left our depth image empty; this module reproduces theirs (our own
implementation, same math) so the policy sees the scan it was trained on. Imported only on the pod.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

try:  # pod only
    from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
    from isaaclab.utils import configclass
except ImportError:  # local test collection: no Isaac Lab
    PatternBaseCfg = object

    def configclass(cls):  # type: ignore[misc]
        return cls


def ray_spherical_slice_pattern(cfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    v = torch.deg2rad(torch.linspace(cfg.vertical_fov_deg[0], cfg.vertical_fov_deg[1], cfg.vertical_num_rays))
    h = torch.deg2rad(torch.linspace(cfg.horizontal_fov_deg[0], cfg.horizontal_fov_deg[1], cfg.horizontal_num_rays))
    vert, horiz = torch.meshgrid(v, h, indexing="ij")
    x = torch.cos(vert) * torch.cos(horiz)
    y = torch.cos(vert) * torch.sin(horiz)
    z = torch.sin(vert)
    directions = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)
    return torch.zeros_like(directions), directions


@configclass
class RaySphericalSlicePatternCfg(PatternBaseCfg):  # type: ignore[misc, valid-type]
    """25 x 60 spherical slice matching the authors' Mid-360 pattern."""

    func: Callable = ray_spherical_slice_pattern
    vertical_fov_deg: tuple[float, float] = (-4.5, 45.5)
    vertical_num_rays: int = 25
    horizontal_fov_deg: tuple[float, float] = (-60.0, 60.0)
    horizontal_num_rays: int = 60
