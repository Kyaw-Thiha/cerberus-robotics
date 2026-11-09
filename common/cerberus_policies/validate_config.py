"""Config validation for a selected set of policy backends.

Rules (docs/modular_stack_options.md, "Design principle: three command boundaries"):
1. every boundary in use has exactly one producer and exactly one consumer;
2. every backend's required sensors are a subset of the active sensor suite.

A boundary is "in use" when any selected backend, or any declared external endpoint, touches it.
Endpoints that are not policy backends (a ROS planner node producing B1, the actuation layer
consuming B2, ...) are declared via ``external_producers`` / ``external_consumers``.

TODO(unify): the sensor check duplicates the intent of
system/cerberus_bringup/cerberus_bringup/validate_sensor_suite.validate_sensor_requirements. That
module is only importable once cerberus_bringup is colcon-built/installed, so this package stays
standalone; fold the two together when the bringup validator becomes importable without ROS.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml

from common.cerberus_policies.base import Boundary, PolicyBackend, Role

#: proprioception every Go2 build has; not part of the sensor rig profiles
ALWAYS_AVAILABLE = frozenset({"joint_encoders"})
#: upstream module outputs a modular backend may consume. Not physical rig sensors: the height scan
#: comes from the simulator's ray-caster in training and from terrain perception at deployment, so the
#: rig profile is the wrong gate for it (the producing module has its own registry/validation).
MODULE_OUTPUTS = frozenset({"height_scan"})
SENSOR_PROFILES_YAML = Path(__file__).resolve().parents[2] / "system/cerberus_bringup/config/sensor_profiles.yaml"


class ConfigValidationError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def sensor_suite_from_profile(profile: str, path: str | Path = SENSOR_PROFILES_YAML) -> dict[str, bool]:
    """The `sensors` block of a named profile in sensor_profiles.yaml (e.g. 'lidar_camera')."""
    profiles = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if profile not in profiles:
        raise ValueError(f"Unknown sensor profile {profile!r}. Available: {sorted(profiles)}")
    return dict(profiles[profile]["sensors"])


def validate_config(
    backends: Iterable[PolicyBackend | type[PolicyBackend]],
    sensor_suite: Mapping[str, bool],
    external_producers: Mapping[Boundary, str] | None = None,
    external_consumers: Mapping[Boundary, str] | None = None,
) -> None:
    """Raise ConfigValidationError listing every violation; return None when the config is valid."""
    problems: list[str] = []
    producers: dict[Boundary, list[str]] = {}
    consumers: dict[Boundary, list[str]] = {}
    for b, who in (external_producers or {}).items():
        producers.setdefault(Boundary(b), []).append(who)
    for b, who in (external_consumers or {}).items():
        consumers.setdefault(Boundary(b), []).append(who)

    active = {s for s, on in sensor_suite.items() if on} | ALWAYS_AVAILABLE | MODULE_OUTPUTS
    for backend in backends:
        cls = backend if isinstance(backend, type) else type(backend)
        label = cls.__name__
        side = producers if cls.role is Role.PRODUCER else consumers
        side.setdefault(cls.boundary, []).append(label)
        missing = sorted(set(cls.required_sensors) - active)
        if missing:
            problems.append(f"{label} requires sensors not in the sensor suite: {missing}")

    in_use = set(producers) | set(consumers)
    for boundary in sorted(in_use, key=lambda b: b.value):
        for kind, table in (("producer", producers), ("consumer", consumers)):
            who = table.get(boundary, [])
            if not who:
                problems.append(f"{boundary.value} has no {kind}")
            elif len(who) > 1:
                problems.append(f"{boundary.value} has {len(who)} {kind}s: {who}")
    if problems:
        raise ConfigValidationError(problems)
