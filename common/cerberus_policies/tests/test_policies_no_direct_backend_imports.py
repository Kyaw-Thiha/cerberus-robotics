"""Architecture rule: nothing outside common/cerberus_policies imports a backend module directly;
use registry.get_backend. Mirrors common/cerberus_worlds/tests/test_no_direct_backend_imports.py."""

import ast
from pathlib import Path

SKIP_DIRS = {".pixi", ".git", "third_party", "build", "install", "log", "__pycache__"}
NEEDLE = "cerberus_policies.backends"


def find_violations(source: str) -> list[int]:
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(NEEDLE in a.name for a in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if (
                NEEDLE in mod
                or (mod.endswith("cerberus_policies") and any(a.name == "backends" for a in node.names))
                or (
                    node.level > 0
                    and (
                        mod == "backends"
                        or mod.startswith("backends.")
                        or (not mod and any(a.name == "backends" for a in node.names))
                    )
                )
            ):
                lines.append(node.lineno)
    return lines


def _repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "pixi.toml").exists():
            return p
    raise RuntimeError("pixi.toml not found above test file")


def test_scanner_detects_violations():
    assert find_violations("import common.cerberus_policies.backends.null_loco\n") == [1]
    assert find_violations("x = 1\nfrom common.cerberus_policies.backends.null_loco import N\n") == [2]
    assert find_violations("from common.cerberus_policies import backends\n") == [1]
    assert find_violations("from .backends import null_loco\n") == [1]
    assert find_violations("from common.cerberus_policies import registry\n") == []


def test_no_direct_backend_imports():
    root = _repo_root()
    own = root / "common" / "cerberus_policies"
    bad = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if SKIP_DIRS & set(rel.parts) or own in path.parents:
            continue
        try:
            hits = find_violations(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        bad += [f"{rel}:{n}" for n in hits]
    assert not bad, "Direct backend imports (use registry.get_backend):\n" + "\n".join(bad)
