"""Architecture rule: nothing outside common/cerberus_worlds imports a
backend module directly; use registry.get_backend."""

import ast
from pathlib import Path

SKIP_DIRS = {".pixi", ".git", "third_party", "build", "install", "log", "__pycache__"}
NEEDLE = "cerberus_worlds.backends"


def find_violations(source: str) -> list[int]:
    """Line numbers of imports referencing cerberus_worlds.backends."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(NEEDLE in a.name for a in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if NEEDLE in mod or mod.endswith("cerberus_worlds") and any(a.name == "backends" for a in node.names) or node.level > 0 and (
                mod == "backends"
                or mod.startswith("backends.")
                or (not mod and any(a.name == "backends" for a in node.names))
            ):
                lines.append(node.lineno)
    return lines


def _repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "pixi.toml").exists():
            return p
    raise RuntimeError("pixi.toml not found above test file")


def test_scanner_detects_violations():
    assert find_violations("import common.cerberus_worlds.backends.arena_backend\n") == [1]
    assert find_violations("x = 1\nfrom common.cerberus_worlds.backends.arena_backend import A\n") == [2]
    assert find_violations("from common.cerberus_worlds import backends\n") == [1]
    assert find_violations("from .backends import arena_backend\n") == [1]
    assert find_violations("from common.cerberus_worlds import registry\n") == []


def test_no_direct_backend_imports():
    root = _repo_root()
    own = root / "common" / "cerberus_worlds"
    bad = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if SKIP_DIRS & set(rel.parts) or own in path.parents:
            continue
        try:
            src = path.read_text(encoding="utf-8")
            hits = find_violations(src)
        except (SyntaxError, UnicodeDecodeError):
            continue
        bad += [f"{rel}:{n}" for n in hits]
    assert not bad, "Direct backend imports (use registry.get_backend):\n" + "\n".join(bad)
