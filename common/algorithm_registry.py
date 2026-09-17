"""Generic algorithm-selection registry, shared by every perception/SLAM/
planning module. Each module calls `make_algorithm_registry()` once to get
its OWN independent registry -- modules never share a dict, only this
definition. Pure data + closures, no domain content, no dependency on
anything else -- matches common/'s "dependency-only, no standalone
behavior" role the same way cerberus_msgs (shared message *types*) does;
this is a shared *shape*, not a runnable component.

See e.g. slam/slam/core/algorithms.py for a module using it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AlgorithmCfg:
    """Everything the sensor-suite-compatibility check needs to know about
    one candidate. `sensor_requirements` keys match
    `SensorSuiteCfg.active_sensors()`'s keys (lidar/front_camera/rear_camera/
    wrist_camera); values are "required" or "optional" -- an "optional"
    sensor may change the algorithm's internal behavior but its absence
    never blocks launch.
    """

    name: str
    sensor_requirements: dict[str, str] = field(default_factory=dict)
    description: str = ""  # one-line pointer to the paper/repo, see docs/perception.md
    implemented: bool = False  # real code exists, not just this declaration -- see validate_sensor_suite.py


def make_algorithm_registry(
    domain: str,
) -> tuple[Callable[[AlgorithmCfg], None], Callable[[str], AlgorithmCfg], Callable[[], list[str]]]:
    """Returns (register_algorithm, get_algorithm, available_algorithms)
    closed over one private dict. Call this once per module -- each call
    creates a fresh, independent dict, so two modules calling this never
    see each other's registrations. `domain` is used only in error messages
    (e.g. "SLAM", "global planner").
    """
    algorithms: dict[str, AlgorithmCfg] = {}

    def register_algorithm(cfg: AlgorithmCfg) -> None:
        if cfg.name in algorithms:
            raise ValueError(f"{domain} algorithm '{cfg.name}' already registered")
        algorithms[cfg.name] = cfg

    def get_algorithm(name: str) -> AlgorithmCfg:
        try:
            return algorithms[name]
        except KeyError:
            raise ValueError(f"unknown {domain} algorithm '{name}', available: {sorted(algorithms)}") from None

    def available_algorithms() -> list[str]:
        return sorted(algorithms)

    return register_algorithm, get_algorithm, available_algorithms
