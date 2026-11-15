"""Read the observation/action contract file used by the transfer check.

Pure python (yaml only), so it can be imported before the simulation starts.
"""

from __future__ import annotations

from pathlib import Path

CONTRACT_PATH = Path("common/cerberus_policies/contracts/heightscan_ppo_v1.yaml")


def _load(path: Path | None = None) -> dict:
    import yaml

    target = Path(path) if path else CONTRACT_PATH
    if not target.is_file():
        target = Path(__file__).resolve().parents[2] / CONTRACT_PATH
    return yaml.safe_load(target.read_text())


def contract_joint_order(path: Path | None = None) -> list[str]:
    """The 12 joint names in the order the PhysX-trained checkpoint was trained on."""
    return list(_load(path)["joints"]["order"])


def contract_observation(path: Path | None = None) -> dict:
    obs = _load(path)["observation"]
    terms = obs["terms"]
    return {
        "total_dim": int(obs["total_dim"]),
        "names": [t["name"] for t in terms],
        "dims": [int(t["dim"]) for t in terms],
    }


def contract_summary(path: Path | None = None) -> dict:
    data = _load(path)
    return {
        "policy": data.get("policy"),
        "contract": data.get("contract"),
        "observation": contract_observation(path),
        "action": data.get("action"),
        "control": data.get("control"),
        "commands": data.get("commands"),
        "height_scan": data.get("height_scan"),
        "joints": contract_joint_order(path),
    }
