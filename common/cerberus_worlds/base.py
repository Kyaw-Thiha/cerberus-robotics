"""World backend interface: scenes + embodiment + task -> env.

Backends (Isaac Lab Arena, plain USD) implement this; everything else goes
through ``registry.get_backend`` and never imports a backend directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from common.cerberus_description.embodiment import EmbodimentCfg


@dataclass(frozen=True)
class SceneInfo:
    """Backend-neutral description of an available scene."""

    scene_id: str
    category: str = ""
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


class WorldBackend(ABC):
    """Contract every world backend must fulfil."""

    @abstractmethod
    def list_scenes(self) -> list[SceneInfo]: ...

    @abstractmethod
    def load_scene(self, scene_id: str) -> Any: ...

    @abstractmethod
    def register_embodiment(self, cfg: EmbodimentCfg) -> None: ...

    @abstractmethod
    def build_env(self, scene: Any, embodiment: EmbodimentCfg, task: Any, num_envs: int) -> Any: ...
