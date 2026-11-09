"""ONNX-backed rough-terrain locomotion policy (B1 consumer): heightscan_ppo_v1.

Consumes a B1 ``LocoCommand`` (plus IMU/joint proprioception and a height scan), emits 12 raw
actions. Contract: ``common/cerberus_policies/contracts/heightscan_ppo_v1.yaml`` (term order, dims,
joint order by name, action scale, control rate).

Two deployment quirks, both measured in H2 (``docs/h2_report.md`` §5):
* the export has a **fixed batch of 1**, so observations are fed one row at a time;
* the weights live in an external ``<name>.onnx.data`` file that must ship next to the graph.

onnxruntime and numpy are imported lazily, inside ``load``/``act``: importing this module pulls in
neither torch nor ROS.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from common.cerberus_policies.base import Boundary, PolicyBackend

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "contracts" / "heightscan_ppo_v1.yaml"
#: name of the contract term holding the height scan; the rest are proprioception/command terms
HEIGHT_SCAN_TERM = "height_scan"


class HeightScanPpoOnnxBackend(PolicyBackend):
    """PolicyBackend for the rough policy exported from the GA candidate checkpoint."""

    boundary = Boundary.B1_LOCO
    required_sensors: ClassVar[Sequence[str]] = ("height_scan", "imu", "joint_encoders")
    rate_hz = 50.0

    def __init__(self, contract_path: str | Path = CONTRACT_PATH) -> None:
        self.contract_path = Path(contract_path)
        self._session = None
        self._input_name: str | None = None
        self._contract: dict[str, Any] | None = None
        self._obs_dim = 0
        self._action_dim = 0

    # -- loading ------------------------------------------------------------------------------
    def load(self, checkpoint_path: str | None, embodiment_cfg: Any = None) -> None:
        """Load the ONNX graph, the sibling ``.onnx.data`` weights and the observation contract."""
        if checkpoint_path is None:
            raise ValueError("heightscan_ppo_onnx needs a .onnx checkpoint path")
        graph = Path(checkpoint_path)
        if not graph.is_file():
            raise FileNotFoundError(f"ONNX graph not found: {graph}")
        external = Path(str(graph) + ".data")
        if not external.is_file() and _declares_external_data(graph, external.name):
            raise FileNotFoundError(
                f"missing external weights file {external.name!r} next to {graph.name}: the export keeps its "
                "weights outside the graph (rsl-rl fixed-batch export), so the .onnx.data file must be copied "
                "alongside the .onnx file"
            )

        import onnxruntime as ort  # lazy: keeps this module importable without onnxruntime

        self._session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._contract = _load_contract(self.contract_path)
        self._obs_dim = int(self._contract["observation"]["total_dim"])
        self._action_dim = int(self._contract["action"]["dim"])
        shape = self._session.get_outputs()[0].shape
        if shape and isinstance(shape[-1], int) and shape[-1] not in (-1, self._action_dim):
            raise ValueError(f"{graph.name} outputs {shape[-1]} actions, contract says {self._action_dim}")

    # -- inference ----------------------------------------------------------------------------
    def act(self, observation: Mapping[str, Any] | Any) -> dict[str, Any]:
        """One control step: named terms or a pre-assembled 235-vector -> 12 raw actions.

        Returns ``{"actions": <12 float32>, "joint_targets": {name: target}}``; the joint targets are
        ``default_joint_pos + scale * action`` by name. ``joint_targets`` needs the contract's joint
        order, so it is always present when the contract is loaded.
        """
        if self._session is None or self._contract is None:
            raise RuntimeError("heightscan_ppo_onnx: call load() before act()")
        import numpy as np  # lazy

        vector = self._assemble(observation)
        if vector.shape != (self._obs_dim,):
            raise ValueError(f"observation must have shape ({self._obs_dim},), got {vector.shape}")
        if vector.dtype != np.float32:
            raise ValueError(f"observation dtype must be float32, got {vector.dtype}")
        # fixed-batch export: one row at a time
        raw = self._session.run(None, {self._input_name: vector.reshape(1, self._obs_dim)})[0]
        actions = np.asarray(raw, dtype=np.float32).reshape(-1)
        if actions.shape != (self._action_dim,):
            raise ValueError(f"backend returned shape {raw.shape}, expected ({self._action_dim},)")
        return {"actions": actions, "joint_targets": self.actions_to_joint_targets(actions)}

    def actions_to_joint_targets(self, actions: Sequence[float] | Any) -> dict[str, float]:
        """``default_joint_pos + scale * action``, keyed by joint name from the contract."""
        if self._contract is None:
            raise RuntimeError("heightscan_ppo_onnx: call load() before actions_to_joint_targets()")
        import numpy as np  # lazy

        a = np.asarray(actions, dtype=np.float32).reshape(-1)
        scale = float(self._contract["action"]["scale"])
        joints = self._contract["joints"]
        order = list(joints["order"])
        defaults = joints["default_joint_pos_rad"]
        if a.size != len(order):
            raise ValueError(f"expected {len(order)} actions, got {a.size}")
        return {name: float(defaults[name]) + scale * float(value) for name, value in zip(order, a)}

    def _assemble(self, observation: Mapping[str, Any] | Any) -> Any:
        import numpy as np  # lazy

        terms = self._contract["observation"]["terms"]  # type: ignore[index]
        if not isinstance(observation, Mapping):
            # keep the dtype of array-likes so act() can reject a silent float64 cast; plain
            # Python sequences carry no dtype and are treated as float32 values
            if hasattr(observation, "dtype"):
                return np.asarray(observation).reshape(-1)
            return np.asarray(observation, dtype=np.float32).reshape(-1)
        missing = [t["name"] for t in terms if t["name"] not in observation]
        if missing:
            raise ValueError(
                f"observation is missing terms {missing}; pass all {len(terms)} named terms or a "
                f"pre-assembled {self._obs_dim}-vector"
            )
        parts = []
        for term in terms:
            value = np.asarray(observation[term["name"]], dtype=np.float32).reshape(-1)
            if value.size != int(term["dim"]):
                raise ValueError(f"term {term['name']!r} must have {term['dim']} values, got {value.size}")
            parts.append(value)
        return np.concatenate(parts)


def _declares_external_data(graph: Path, external_name: str) -> bool:
    """Whether the ONNX graph references weights stored outside itself.

    The protobuf does not spell out the ``external_data`` field name; it carries the ``location``
    key and the data file's name as strings (verified on the GA export, whose bytes contain
    ``go2_locomotion_ga_v1_candidate.onnx.data`` and ``location``). A byte scan is cheap on a graph
    that holds no weights, and avoids depending on the ``onnx`` package at runtime.
    """
    blob = graph.read_bytes()
    return external_name.encode() in blob or b"location" in blob


def _load_contract(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        raise FileNotFoundError(f"observation contract not found: {path}")
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    total = sum(int(t["dim"]) for t in contract["observation"]["terms"])
    if total != int(contract["observation"]["total_dim"]):
        raise ValueError(f"contract {path.name}: term dims sum to {total}, total_dim says {contract['observation']['total_dim']}")
    order = list(contract["joints"]["order"])
    defaults = contract["joints"]["default_joint_pos_rad"]
    if set(order) != set(defaults):
        raise ValueError(f"contract {path.name}: joint order and default_joint_pos_rad name different joints")
    if len(order) != int(contract["action"]["dim"]):
        raise ValueError(f"contract {path.name}: {len(order)} joint names vs action dim {contract['action']['dim']}")
    return contract
