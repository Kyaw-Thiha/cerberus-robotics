"""Backend cards: one YAML per policy backend, validated so support-level claims stay tied to evidence.

Fields (see docs/backend_card.schema.yaml, kept in sync by a test):
name, slot, boundary, paradigm, architecture, required_sensors, rate_hz, license, checkpoint_url,
code_url, support_level, target_support_level, evidence, results.

Support levels: L0 documented (card only) < L1 integrated (adapter + contract tests) < L2 evaluated
(numbers recorded) < L3 trained by us (result in a report). ``target_support_level`` is ambition,
``support_level`` is what the evidence proves today.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from common.cerberus_policies.registry import SLOT_BOUNDARY

SLOTS = tuple(SLOT_BOUNDARY)
BOUNDARIES = tuple(b.value for b in SLOT_BOUNDARY.values())
ARCHITECTURES = ("modular", "fused")
SUPPORT_LEVELS = ("L0", "L1", "L2", "L3")
EVIDENCE_KINDS = ("file", "report", "commit")
UNKNOWN = "unknown"
#: fields that may hold the literal "unknown", and only while support_level is L0
UNKNOWN_OK_FIELDS = ("required_sensors", "rate_hz", "license", "checkpoint_url", "code_url")
REQUIRED_FIELDS = (
    "name",
    "slot",
    "boundary",
    "paradigm",
    "architecture",
    "required_sensors",
    "rate_hz",
    "license",
    "checkpoint_url",
    "code_url",
    "support_level",
    "target_support_level",
    "evidence",
    "results",
)
CARDS_DIR = Path(__file__).resolve().parent / "cards"
REPO_ROOT = Path(__file__).resolve().parents[2]


class CardValidationError(ValueError):
    def __init__(self, problems: list[str], source: str = "card"):
        self.problems = problems
        super().__init__(f"{source}: " + "; ".join(problems))


@dataclass(frozen=True)
class BackendCard:
    name: str
    slot: str
    boundary: str
    paradigm: str
    architecture: str
    required_sensors: list[str] | str
    rate_hz: float | str
    license: str
    checkpoint_url: str
    code_url: str
    support_level: str
    target_support_level: str
    evidence: list[dict[str, str]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)


def _level(card: dict[str, Any], key: str, problems: list[str]) -> int | None:
    v = card.get(key)
    if v not in SUPPORT_LEVELS:
        problems.append(f"{key} must be one of {SUPPORT_LEVELS}, got {v!r}")
        return None
    return SUPPORT_LEVELS.index(v)


def validate_card(
    data: dict[str, Any],
    *,
    check_paths: bool = False,
    repo_root: Path | None = None,
    source: str = "card",
) -> BackendCard:
    """Validate a parsed card; raises CardValidationError listing every problem found."""
    problems: list[str] = []
    if not isinstance(data, dict):
        raise CardValidationError(["card must be a mapping"], source)

    missing = [k for k in REQUIRED_FIELDS if k not in data]
    extra = [k for k in data if k not in REQUIRED_FIELDS]
    if missing:
        problems.append(f"missing fields: {missing}")
    if extra:
        problems.append(f"unknown fields: {extra}")
    if missing:
        raise CardValidationError(problems, source)

    if not isinstance(data["name"], str) or not re.fullmatch(r"[a-z0-9_]+", data["name"]):
        problems.append("name must match [a-z0-9_]+")
    if data["slot"] not in SLOTS:
        problems.append(f"slot must be one of {SLOTS}, got {data['slot']!r}")
    elif data["boundary"] != SLOT_BOUNDARY[data["slot"]].value:
        problems.append(f"boundary {data['boundary']!r} does not match slot {data['slot']!r} ({SLOT_BOUNDARY[data['slot']].value})")
    if data["boundary"] not in BOUNDARIES:
        problems.append(f"boundary must be one of {BOUNDARIES}, got {data['boundary']!r}")
    if not isinstance(data["paradigm"], str) or not data["paradigm"].strip() or data["paradigm"] == UNKNOWN:
        problems.append("paradigm must be a non-empty string")
    if data["architecture"] not in ARCHITECTURES:
        problems.append(f"architecture must be one of {ARCHITECTURES}, got {data['architecture']!r}")

    level = _level(data, "support_level", problems)
    target = _level(data, "target_support_level", problems)
    if level is not None and target is not None and target < level:
        problems.append("target_support_level must be >= support_level")

    # "unknown" placeholders: only L0, only on the allowed fields
    for k in UNKNOWN_OK_FIELDS:
        if data[k] == UNKNOWN and level is not None and level > 0:
            problems.append(f"{k} may be 'unknown' only at L0")
    sensors = data["required_sensors"]
    if sensors != UNKNOWN and (
        isinstance(sensors, str) or not isinstance(sensors, list) or not all(isinstance(s, str) and s for s in sensors)
    ):
        problems.append("required_sensors must be a list of sensor id strings (or 'unknown' at L0)")
    rate = data["rate_hz"]
    if rate != UNKNOWN and (isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0):
        problems.append("rate_hz must be a positive number (or 'unknown' at L0)")
    for k in ("license", "checkpoint_url", "code_url"):
        if not isinstance(data[k], str) or not data[k].strip():
            problems.append(f"{k} must be a non-empty string")

    evidence, results = data["evidence"], data["results"]
    if not isinstance(evidence, list):
        problems.append("evidence must be a list")
        evidence = []
    if not isinstance(results, list):
        problems.append("results must be a list")
        results = []
    kinds: set[str] = set()
    root = repo_root or REPO_ROOT
    for i, e in enumerate(evidence):
        if not (isinstance(e, dict) and set(e) == {"kind", "ref"} and e["kind"] in EVIDENCE_KINDS and isinstance(e["ref"], str) and e["ref"].strip()):
            problems.append(f"evidence[{i}] must be {{kind: {'|'.join(EVIDENCE_KINDS)}, ref: <non-empty str>}}")
            continue
        kinds.add(e["kind"])
        if check_paths and e["kind"] == "file" and not (root / e["ref"]).exists():
            problems.append(f"evidence[{i}] file does not exist: {e['ref']}")
    for i, r in enumerate(results):
        ok = isinstance(r, dict) and {"metric", "value", "eval", "source"} <= set(r) <= {"metric", "value", "eval", "source", "trained_by_us"}
        if not ok:
            problems.append(f"results[{i}] needs metric, value, eval, source (optional trained_by_us)")
        elif "trained_by_us" in r and not isinstance(r["trained_by_us"], bool):
            problems.append(f"results[{i}].trained_by_us must be bool")

    if level is not None:
        if level >= 1 and not evidence:
            problems.append(f"{data['support_level']} requires non-empty evidence")
        if level >= 2:
            if not results:
                problems.append(f"{data['support_level']} requires a results entry")
            if "report" not in kinds:
                problems.append(f"{data['support_level']} requires a 'report' evidence entry")
        if level >= 3 and not any(isinstance(r, dict) and r.get("trained_by_us") is True for r in results):
            problems.append("L3 requires a results entry with trained_by_us: true")

    if problems:
        raise CardValidationError(problems, source)
    return BackendCard(**{k: data[k] for k in REQUIRED_FIELDS})


def load_card(path: str | Path, *, check_paths: bool = False) -> BackendCard:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    card = validate_card(data, check_paths=check_paths, source=str(path.name))
    if card.name != path.stem:
        raise CardValidationError([f"name {card.name!r} must match file name {path.stem!r}"], path.name)
    return card


def load_all_cards(cards_dir: str | Path = CARDS_DIR, *, check_paths: bool = False) -> list[BackendCard]:
    """All cards under ``cards_dir``, sorted by (slot, name)."""
    cards = [load_card(p, check_paths=check_paths) for p in sorted(Path(cards_dir).glob("*.yaml"))]
    return sorted(cards, key=lambda c: (c.slot, c.name))
