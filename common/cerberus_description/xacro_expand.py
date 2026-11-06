"""Expand a ROS xacro file to URDF XML without a ROS/ament environment.

`$(find <pkg>)` needs an ament/catkin package index, which a plain pixi/pip environment does not have. This helper
copies the xacro's top-level directory (e.g. `urdf/`) into a temporary package root, rewrites the package references
textually and calls the `xacro` python API:

- `$(find <pkg>)/<copied dir>/...` -> the temporary copy (so `xacro:include` finds the sibling xacro files);
- `$(find <pkg>)/<anything else>/...` (e.g. `meshes/`) -> the REAL package directory;
- `package://<pkg>/...` -> the REAL package directory, so mesh paths in the output are absolute and point at the
  real files, not at the temporary copy.

Pure python at import time: `xacro` is imported inside `expand_xacro`.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

DEFAULT_PACKAGE_NAME = "cerberus_description"
DEFAULT_PACKAGE_DIR = Path(__file__).resolve().parent
_SUBSTITUTED_SUFFIXES = (".xacro", ".urdf")


def xacro_available() -> bool:
    """True when the `xacro` python package can be imported."""
    try:
        import xacro  # noqa: F401
    except ImportError:
        return False
    return True


def _rewrite_text(
    text: str, package_name: str, package_dir: Path, copy_root: Path, copy_name: str
) -> str:
    """Resolve `$(find pkg)[/rel]` and `package://pkg/` references in one file's text."""
    pkg = re.escape(package_name)
    real = package_dir.as_posix()

    def find_repl(match: re.Match) -> str:
        rel = match.group("rel") or ""
        top = rel.lstrip("/").split("/", 1)[0]
        if top == copy_name:
            return copy_root.as_posix() + rel
        return real + rel

    text = re.sub(
        rf"\$\(\s*find\s+{pkg}\s*\)(?P<rel>/[^/\"'<>\s]+(?:/[^\"'<>\s]*)?)?",
        find_repl,
        text,
    )
    return text.replace(f"package://{package_name}/", real + "/")


def _strip_comments(node) -> None:
    """Remove every XML comment node below `node` (a `xml.dom.minidom` node), in place."""
    for child in list(node.childNodes):
        if child.nodeType == child.COMMENT_NODE:
            node.removeChild(child)
        else:
            _strip_comments(child)


def expand_xacro(
    xacro_path: str | Path,
    args: Mapping[str, str] | None = None,
    package_name: str = DEFAULT_PACKAGE_NAME,
    package_dir: str | Path | None = None,
) -> str:
    """Expand `xacro_path` with xacro arguments `args` (e.g. `{'sim': 'true'}`) and return the URDF XML string.

    `package_dir` is the real directory of the package `package_name` (default: `common/cerberus_description`, the
    directory of this module). Raises FileNotFoundError for a missing xacro and ImportError if `xacro` is missing.
    """
    try:
        import xacro
    except ImportError as e:
        raise ImportError(
            f"expanding xacro needs the `xacro` python package (pixi default env has it): {e}"
        ) from e

    xacro_path = Path(xacro_path).expanduser().resolve()
    if not xacro_path.is_file():
        raise FileNotFoundError(f"xacro file not found: {xacro_path}")
    pkg_dir = (
        Path(package_dir).expanduser().resolve()
        if package_dir is not None
        else DEFAULT_PACKAGE_DIR
    )
    if not pkg_dir.is_dir():
        raise FileNotFoundError(f"package directory not found: {pkg_dir}")

    # The directory to copy: the xacro's top-level dir under the package (`urdf/`), else its own parent directory.
    try:
        copy_name = xacro_path.relative_to(pkg_dir).parts[0]
        source_dir = pkg_dir / copy_name
        if not source_dir.is_dir():  # xacro lives directly in the package root
            raise ValueError("xacro is not inside a subdirectory of the package")
    except ValueError:
        source_dir = xacro_path.parent
        copy_name = source_dir.name
    mappings = {str(k): str(v) for k, v in (args or {}).items()}

    with tempfile.TemporaryDirectory(prefix="xacro_expand_") as tmp:
        copy_root = Path(tmp).resolve() / package_name
        dest_dir = copy_root / copy_name
        shutil.copytree(source_dir, dest_dir)
        for path in dest_dir.rglob("*"):
            if path.is_file() and path.suffix in _SUBSTITUTED_SUFFIXES:
                original = path.read_text(encoding="utf-8")
                rewritten = _rewrite_text(
                    original, package_name, pkg_dir, copy_root, copy_name
                )
                if rewritten != original:
                    path.write_text(rewritten, encoding="utf-8")
        doc = xacro.process_file(
            str(dest_dir / xacro_path.relative_to(source_dir)), mappings=mappings
        )
        # xacro's header comment names the temporary file: drop all comments so the output is deterministic.
        _strip_comments(doc)
        return doc.toprettyxml(indent="  ")
