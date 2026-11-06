#!/usr/bin/env python3
"""Pod-side: pull the smallest GRScenes-100 scene(s) from Hugging Face and
stage them, with the Materials/models they reference, into our PRIVATE R2 bucket.

Source: HF dataset `InternRobotics/GRScenes` (not gated, CC BY-NC-SA 4.0 -- keep
everything private, never publish), files `scenes/GRScenes-100/home_scenes.z01`,
`.z02` and `home_scenes.zip` (one split zip, ~50 GB; `--category home`) or
`commercial_scenes.zip` (23.8 GB; `--category commercial`). The archive root is
`{Materials,models,scenes/<id>_usd}` under one top folder (`home_scenes/`,
`commercial_scenes/`; older notes say `target_69_new/`); each scene folder holds
`start_result_*.usd`, `metadata.json` and `Materials`/`models` symlinks to
`../../Materials`, `../../models`. R2 does not store symlinks, so the upload tree
contains REAL files only and `fetch_grscenes_from_r2.py` recreates the links.

R2 layout (default prefix `worlds/grscenes/home`):

    <bucket>/worlds/grscenes/home/{Materials,models,scenes/<id>_usd}/...
    <bucket>/worlds/grscenes/home/manifest.json

Steps (all idempotent / restartable): open the archive, list members, rank scenes by
size of start_result_navigation.usd (smallest first), extract the chosen scenes +
the models/Materials they reference, build the upload tree, write manifest.json,
`rclone copy`, `rclone size` check, print a JSON summary line.

How the archive is opened (`remote_zip.py`): by default over HTTP range requests, so
only the central directory and the chosen members are transferred (a handful of GB
instead of 50-74 GB, minutes instead of hours). `--all` (every scene) or
`--skip-download` (parts already in --stage) use the local path instead: download
the parts (curl -C -), let 7z read the split zip (else `zip -s 0 ... --out` join).

Modes: `--extract-only` stops after extraction (no R2 needed) and can link the tree
to `--link-to` (the `data/scenes/grscenes/<category>` layout); `--from-tree` skips
the archive and uploads the scenes in `--scene-id/--scene-ids-file` (plus the shared
models/Materials they reference, resolved from the already extracted files in
--stage) -- used to upload only admitted scenes.

Needs on the pod: python3 stdlib; for the local path curl, unzip, zip, 7z; for
upload rclone and the env vars R2_ACCOUNT_ID, R2_ACCESS_KEY_ID,
R2_SECRET_ACCESS_KEY, R2_BUCKET.

Usage:
    python3 scripts/fetch_grscenes_to_r2.py --dry-run
    python3 scripts/fetch_grscenes_to_r2.py --stage /workspace/grscenes --scenes 1
    python3 scripts/fetch_grscenes_to_r2.py --category commercial --scenes 2 --extract-only
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, NamedTuple

import remote_zip
from sync_to_r2 import REMOTE, _rclone_env, _require_env

HF_REPO = "InternRobotics/GRScenes"
HF_API = f"https://huggingface.co/api/datasets/{HF_REPO}"
HF_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
PARTS_DIR = "scenes/GRScenes-100"
PART_NAMES = ["home_scenes.z01", "home_scenes.z02", "home_scenes.zip"]
COMMERCIAL_PART_NAMES = ["commercial_scenes.zip"]
# Sizes from the HF listing at planning time; used for --dry-run when offline.
KNOWN_SIZES = {
    "home_scenes.z01": 21474836480,
    "home_scenes.z02": 21474836480,
    "home_scenes.zip": 7999812294,
    "commercial_scenes.zip": 23792807959,
}
JOINED_NAME = "home_joined.zip"
DEFAULT_STAGE = "/workspace/grscenes"
DEFAULT_PREFIX = "worlds/grscenes/home"
CATEGORIES = {
    "home": (PART_NAMES, DEFAULT_PREFIX),
    "commercial": (COMMERCIAL_PART_NAMES, "worlds/grscenes/commercial"),
}
LICENSE_NOTE = (
    "GRScenes-100 (InternRobotics/GRScenes), CC BY-NC-SA 4.0. Non-commercial, share-alike. "
    "Private mirror for internal research use only; do not publish or redistribute."
)
DISK_MARGIN = 2 * 1024**3
UNRESOLVED_FALLBACK_FRACTION = 0.3

PATH_EXTS = (
    "usd",
    "usda",
    "usdc",
    "usdz",
    "mdl",
    "png",
    "jpg",
    "jpeg",
    "exr",
    "hdr",
    "tga",
    "bmp",
)
_EXT_ALT = "|".join(PATH_EXTS)
PATH_RE = re.compile(
    rf"[^\s\"'@<>()\[\]{{}},;]+\.(?:{_EXT_ALT})(?![A-Za-z0-9_])", re.IGNORECASE
)
SCENE_MEMBER_RE = re.compile(
    r"^(?P<root>(?:.*/)?)scenes/(?P<id>[^/]+_usd)/(?P<rest>.*)$"
)
SCENE_USD_NAMES = ("start_result_navigation.usd", "start_result_raw.usd")
MDL_IMPORT_RE = re.compile(r"^\s*import\s+\.?::\s*([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
"""``import ::KooPbr::KooMtl;`` (absolute) or ``import .::OmniUe4Function import *;`` (relative) -> the module name.

MDL module imports are NOT paths, so ``PATH_RE`` never matches them. That is why GRScenes' ``Materials/KooPbr.mdl``
and ``KooPbr_maps.mdl`` were left out of every selective extraction until 2026-10-11: 280 of the dataset's 299
``.mdl`` files import them, nothing in the USDs or the metadata references them, and a missing module makes the
material fail to compile (``MDLC comp error ... C120``), which the RTX renderer then retries on EVERY launch --
minutes of wasted startup per scene on a cache that can never warm (see docs/arena_integration_log.md).
"""
SCENE_USD_GLOB_RE = re.compile(r"^start_result_[^/]+\.usd[a-z]?$")


class Member(NamedTuple):
    name: str
    size: int
    is_link: bool = False


class Archive(NamedTuple):
    path: Path
    kind: str  # "7z" | "unzip" | "zipfile" | "remote"
    remote: Any = None  # remote_zip.RemoteZip for kind "remote"


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    log(" ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kwargs)


def _probe(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run without raising; for capability checks."""
    return subprocess.run(cmd, check=False, capture_output=True, text=True, **kwargs)


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return str(n)


def check_free(path: Path, need: int, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    log(
        f"disk check [{label}]: need ~{human(need)} (+{human(DISK_MARGIN)} margin), free {human(free)}"
    )
    if free < need + DISK_MARGIN:
        raise SystemExit(
            f"Not enough free disk for '{label}': need {human(need + DISK_MARGIN)}, have {human(free)}."
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Hugging Face
# ----------------------------------------------------------------------------


def fetch_hf_info(timeout: float = 30.0, parts: list[str] = PART_NAMES) -> dict:
    """Return {"sha": revision, "sizes": {part_name: bytes}} (in `parts` order) from the HF API."""
    with urllib.request.urlopen(f"{HF_API}?blobs=true", timeout=timeout) as resp:
        data = json.load(resp)
    sizes = {}
    for sib in data.get("siblings", []):
        rel = sib.get("rfilename", "")
        if rel.startswith(PARTS_DIR + "/") and rel.split("/")[-1] in parts:
            sizes[rel.split("/")[-1]] = sib.get("size")
    missing = [p for p in parts if not sizes.get(p)]
    if missing:
        raise SystemExit(f"HF API listing lacks sizes for {missing}")
    return {"sha": data.get("sha", "unknown"), "sizes": {p: sizes[p] for p in parts}}


def download_parts(stage: Path, sizes: dict[str, int]) -> None:
    for name, want in sizes.items():
        dest = stage / name
        if dest.exists() and dest.stat().st_size > want:
            log(f"{name} larger than expected, removing")
            dest.unlink()
        if dest.exists() and dest.stat().st_size == want:
            log(f"{name} already complete ({human(want)})")
            continue
        have = dest.stat().st_size if dest.exists() else 0
        check_free(stage, want - have, f"download {name}")
        url = f"{HF_RESOLVE}/{PARTS_DIR}/{name}"
        _run(
            [
                "curl",
                "-fL",
                "-C",
                "-",
                "--retry",
                "10",
                "--retry-delay",
                "5",
                "--retry-all-errors",
                "-o",
                str(dest),
                url,
            ]
        )
        got = dest.stat().st_size
        if got != want:
            raise SystemExit(
                f"{name}: size {got} != expected {want} (re-run to resume)"
            )


# ----------------------------------------------------------------------------
# archive access
# ----------------------------------------------------------------------------


def parse_unzip_zl(text: str) -> list[Member]:
    """Parse `unzip -Zl` (zipinfo long) output."""
    members = []
    for line in text.splitlines():
        parts = line.split(None, 8)
        if (
            len(parts) < 9
            or not parts[3].isdigit()
            or not re.match(r"^[-dlcbps?][-rwxsStTa]+$", parts[0])
        ):
            continue  # header / summary lines
        name = parts[8]
        if name.endswith("/") or parts[0].startswith("d"):
            continue
        members.append(Member(name, int(parts[3]), parts[0].startswith("l")))
    return members


def parse_7z_slt(text: str) -> list[Member]:
    """Parse `7z l -slt` output."""
    members = []
    for block in re.split(r"\n\s*\n", text):
        fields = {}
        for line in block.splitlines():
            if " = " in line:
                k, v = line.split(" = ", 1)
                fields[k.strip()] = v
        if "Path" not in fields or "Size" not in fields or "Folder" not in fields:
            continue  # archive header block etc
        if fields["Folder"].strip() == "+":
            continue
        attrs = fields.get("Attributes", "")
        is_link = bool(re.search(r"(^|\s)l[-r][-w][-x]", attrs))
        size = fields["Size"].strip()
        members.append(
            Member(
                fields["Path"].replace("\\", "/"),
                int(size) if size.isdigit() else 0,
                is_link,
            )
        )
    return members


def list_members(arc: Archive) -> list[Member]:
    if arc.kind == "remote":
        return [Member(m.name, m.size, m.is_link) for m in arc.remote.members() if not m.is_dir]
    if arc.kind == "zipfile":
        with zipfile.ZipFile(arc.path) as zf:
            return [
                Member(i.filename, i.file_size, stat.S_ISLNK(i.external_attr >> 16))
                for i in zf.infolist()
                if not i.is_dir()
            ]
    if arc.kind == "7z":
        out = subprocess.run(
            ["7z", "l", "-slt", str(arc.path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return parse_7z_slt(out)
    out = subprocess.run(
        ["unzip", "-Zl", str(arc.path)], check=True, capture_output=True, text=True
    ).stdout
    return parse_unzip_zl(out)


def _unzip_escape(name: str) -> str:
    return re.sub(r"([\[\]*?\\])", r"\\\1", name)


def extract_members(arc: Archive, names: list[str], out_dir: Path) -> None:
    """Extract exactly `names` into out_dir (original relative paths kept)."""
    if not names:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    if arc.kind == "remote":
        next_mark = [1 << 30]

        def progress(done: int, files: int) -> None:
            if done >= next_mark[0]:
                log(f"  extracted {human(done)} ({files} files in this batch)")
                next_mark[0] = (done >> 30 << 30) + (1 << 30)

        arc.remote.extract(names, out_dir, on_progress=progress)
    elif arc.kind == "zipfile":
        with zipfile.ZipFile(arc.path) as zf:
            for n in names:
                zf.extract(n, out_dir)
    elif arc.kind == "7z":
        # UNVERIFIED: 7z `-i@listfile` extraction from a split .z01/.z02/.zip set (needs a pod run).
        list_file = out_dir / ".extract_list.txt"
        list_file.write_text("\n".join(names) + "\n")
        try:
            _run(
                [
                    "7z",
                    "x",
                    "-y",
                    "-bd",
                    "-spd",
                    f"-o{out_dir}",
                    str(arc.path),
                    f"-i@{list_file}",
                ],
                stdout=subprocess.DEVNULL,
            )
        finally:
            list_file.unlink(missing_ok=True)
    else:
        for i in range(0, len(names), 200):
            chunk = [_unzip_escape(n) for n in names[i : i + 200]]
            _run(["unzip", "-o", "-q", str(arc.path), "-d", str(out_dir), *chunk])


def make_readable(stage: Path, sizes: dict[str, int]) -> Archive:
    """Return an Archive usable for listing/extracting, joining the split zip if 7z cannot read it."""
    main_zip = next(stage / n for n in sizes if n.endswith(".zip"))
    joined = stage / JOINED_NAME
    if len(sizes) == 1:  # not split: nothing to join, 7z or unzip read it directly
        if shutil.which("7z"):
            r = _probe(["7z", "l", "-slt", str(main_zip)])
            if r.returncode == 0 and parse_7z_slt(r.stdout):
                return Archive(main_zip, "7z")
        return Archive(main_zip, "unzip")
    if joined.exists():
        r = _probe(["unzip", "-Zl", str(joined)]) if shutil.which("unzip") else None
        if r is not None and r.returncode == 0 and parse_unzip_zl(r.stdout):
            log(f"{JOINED_NAME} already present and lists fine; reusing")
            return Archive(joined, "unzip")
        log(f"{JOINED_NAME} exists but does not list; removing")
        joined.unlink()
    if shutil.which("7z"):
        # UNVERIFIED: that 7z opens the .z01/.z02/.zip set directly (hence the join fallback).
        r = _probe(["7z", "l", "-slt", str(main_zip)])
        if r.returncode == 0 and parse_7z_slt(r.stdout):
            log(
                "7z reads the split archive directly; no join needed (no extra disk used)"
            )
            return Archive(main_zip, "7z")
        log("7z could not read the split archive; falling back to zip -s 0 join")
    check_free(stage, sum(sizes.values()), "join split zip")
    _run(["zip", "-s", "0", str(main_zip), "--out", str(joined)])
    r = _probe(["unzip", "-Zl", str(joined)])
    if r.returncode != 0 or not parse_unzip_zl(r.stdout):
        raise SystemExit(
            f"Joined zip {joined} does not list; parts kept for retry.\n{r.stderr[-500:]}"
        )
    return Archive(joined, "unzip")


def delete_parts_if_joined(stage: Path, arc: Archive, keep_zips: bool, names: list[str] = PART_NAMES) -> None:
    if keep_zips or arc.kind in ("7z", "remote"):
        return  # with 7z the parts ARE the archive
    for name in names:
        p = stage / name
        if p.exists():
            log(f"deleting {name} (joined zip lists OK)")
            p.unlink()


# ----------------------------------------------------------------------------
# scene ranking + reference resolution
# ----------------------------------------------------------------------------


def normalize_scene_id(sid: str) -> str:
    sid = sid.strip().strip("/")
    return sid if sid.endswith("_usd") else sid + "_usd"


def rank_scenes(members: list[Member]) -> list[tuple[str, int, str, str]]:
    """Return [(scene_dir_id, size, usd_name, root)] sorted smallest-first.

    Size = start_result_navigation.usd if present, else start_result_raw.usd.
    """
    found: dict[str, dict] = {}
    for m in members:
        if m.is_link:
            continue
        match = SCENE_MEMBER_RE.match(m.name)
        if not match or match["rest"] not in SCENE_USD_NAMES:
            continue
        entry = found.setdefault(match["id"], {"root": match["root"]})
        entry[match["rest"]] = m.size
    ranked = []
    for sid, e in found.items():
        for usd in SCENE_USD_NAMES:
            if usd in e:
                ranked.append((sid, e[usd], usd, e["root"]))
                break
    ranked.sort(key=lambda t: (t[1], t[0]))
    return ranked


def choose_scenes(
    members: list[Member], n: int | None, scene_ids: list[str] | None
) -> tuple[list[tuple], list[tuple]]:
    """The `n` smallest scenes (all of them for ``n=None``) or the explicit `scene_ids`, plus the full ranking."""
    ranking = rank_scenes(members)
    if not ranking:
        raise SystemExit(
            "No scenes/<id>_usd/start_result_{navigation,raw}.usd found in the archive."
        )
    if scene_ids:
        by_id = {r[0]: r for r in ranking}
        chosen = []
        for sid in map(normalize_scene_id, scene_ids):
            if sid not in by_id:
                raise SystemExit(
                    f"Scene {sid} not in archive. Known (first 10): {[r[0] for r in ranking[:10]]}"
                )
            chosen.append(by_id[sid])
        return chosen, ranking
    return (ranking if n is None else ranking[: max(1, n)]), ranking


def looks_like_path(s: str) -> bool:
    return bool(PATH_RE.search(s))


def collect_path_strings(obj) -> set[str]:
    """Collect every path-like substring (ending in a known asset extension) from any JSON shape."""
    out: set[str] = set()

    def visit_str(s: str) -> None:
        s = s.strip()
        if not s:
            return
        if "\n" not in s and re.search(rf"\.(?:{_EXT_ALT})$", s, re.IGNORECASE):
            out.add(s)  # whole string is a path (may contain spaces)
        for m in PATH_RE.finditer(s):
            out.add(m.group(0))

    def walk(o) -> None:
        if isinstance(o, str):
            visit_str(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                visit_str(str(k))
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(obj)
    return out


def resolve_ref(
    ref: str, member_set: set[str], root: str, base_dir: str | None = None
) -> str | None:
    """Map a reference string onto a zip member name, or None."""
    r = urllib.parse.unquote(ref.strip()).replace("\\", "/")
    r = (
        re.sub(r"^(?:file:/*|omniverse://[^/]+/)", "/", r)
        if re.match(r"^(file:|omniverse://)", r)
        else r
    )
    cands = []
    if base_dir is not None and not r.startswith("/"):
        cands.append(posixpath.normpath(posixpath.join(base_dir, r)))
    parts = [p for p in r.split("/") if p not in ("", ".", "..")]
    for i, part in enumerate(parts):
        if part in ("models", "Materials"):
            cands.append(root + "/".join(parts[i:]))
    cands.append(root + "/".join(parts))
    for c in cands:
        if c in member_set:
            return c
    return None


def mdl_or_usd_text_refs(path: Path) -> set[str]:
    """Path-like strings in a text asset (.mdl / .usda / ascii .usd); empty for binary files."""
    try:
        with open(path, "rb") as f:
            head = f.read(16 * 1024 * 1024)
    except OSError:
        return set()
    suffix = path.suffix.lower()
    if suffix not in (".mdl", ".usda", ".json") and not head.startswith(b"#usda"):
        return set()
    if b"\x00" in head[:4096]:
        return set()
    return {m.group(0) for m in PATH_RE.finditer(head.decode("utf-8", errors="ignore"))}


def mdl_module_imports(path: Path) -> set[str]:
    """Module names a text ``.mdl`` file imports; empty for anything else (or unreadable)."""
    if path.suffix.lower() != ".mdl":
        return set()
    try:
        with open(path, "rb") as f:
            head = f.read(4 * 1024 * 1024)
    except OSError:
        return set()
    return {m.group(1) for m in MDL_IMPORT_RE.finditer(head.decode("utf-8", errors="ignore"))}


def module_member(name: str, member_set: set[str], root: str) -> str | None:
    """Archive member holding MDL module ``name``: ``<root>Materials/<name>.mdl``, else any ``<...>/<name>.mdl``.

    The archive keeps every ``.mdl`` in one ``Materials/`` directory (1677 of them, verified 2026-10-11), so the
    direct hit is the normal case; the suffix search is the general fallback for a dataset that nests modules.
    """
    direct = f"{root}Materials/{name}.mdl"
    if direct in member_set:
        return direct
    suffix = f"/{name}.mdl"
    return next((m for m in sorted(member_set) if m.endswith(suffix)), None)


def expand_siblings(
    name: str, root: str, members_by_dir: dict[str, list[str]]
) -> list[str]:
    """For a resolved model file also take its folder tree (textures live beside/below it).

    The real archive nests assets as `models/<cat>/<sub>/<kind>/<hash>/instance.usd`, so the
    asset folder is the file's own directory (taking the first path component would grab a
    whole category, tens of GB). Only for files at least `models/<x>/<file>` deep.
    """
    rel = name[len(root) :]
    top, _, tail = rel.partition("/")
    if top != "models" or tail.count("/") < 1:
        return [name]
    folder = posixpath.dirname(name) + "/"
    return [n for d, lst in members_by_dir.items() if d.startswith(folder) for n in lst]


def select_members(
    members: list[Member],
    chosen: list[tuple],
    metadata_by_scene: dict[str, object],
    root: str,
) -> dict:
    """Resolve metadata references onto members. Returns selection info (no extraction)."""
    non_link = [m for m in members if not m.is_link]
    sizes = {m.name: m.size for m in non_link}
    member_set = set(sizes)
    by_dir: dict[str, list[str]] = {}
    for n in sizes:
        by_dir.setdefault(posixpath.dirname(n) + "/", []).append(n)
    resolved: set[str] = set()
    unresolved: set[str] = set()
    for meta in metadata_by_scene.values():
        for ref in collect_path_strings(meta):
            hit = resolve_ref(ref, member_set, root)
            if hit is None:
                unresolved.add(ref)
                continue
            resolved.add(hit)
    files: set[str] = set()
    for hit in resolved:
        files.update(expand_siblings(hit, root, by_dir))
    return {
        "resolved": resolved,
        "unresolved": unresolved,
        "files": files,
        "sizes": sizes,
        "member_set": member_set,
    }


def scene_members(members: list[Member], sid: str, root: str) -> list[Member]:
    prefix = f"{root}scenes/{sid}/"
    return [m for m in members if m.name.startswith(prefix) and not m.is_link]


def read_scene_metadata(extract_dir: Path, root: str, sid: str) -> object:
    p = extract_dir / (root + f"scenes/{sid}/metadata.json")
    if not p.exists():
        log(f"WARNING: no metadata.json for {sid}")
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as e:
        log(f"WARNING: unreadable metadata.json for {sid}: {e}")
        return {}


def extract_needed(
    arc: Archive | None, names: list[str], sizes: dict[str, int], extract_dir: Path
) -> int:
    """Extract members not already present with the right size. Returns bytes newly extracted.
    ``arc=None`` means the files are already on disk (``--from-tree``): nothing to do."""
    if arc is None:
        return 0
    todo = []
    for n in sorted(set(names)):
        p = extract_dir / n
        if p.is_file() and not p.is_symlink() and p.stat().st_size == sizes.get(n, -1):
            continue
        todo.append(n)
    need = sum(sizes.get(n, 0) for n in todo)
    check_free(extract_dir, need, "extract")
    extract_members(arc, todo, extract_dir)
    return need


def extract_selection(
    arc: Archive | None,
    members: list[Member],
    chosen: list[tuple],
    extract_dir: Path,
    max_bytes: int | None,
) -> dict:
    """Steps (d): extract scene folders, parse metadata, extract referenced assets (+ closure over text assets)."""
    roots = {c[3] for c in chosen}
    if len(roots) != 1:
        raise SystemExit(f"Scenes live under different roots: {roots}")
    root = roots.pop()
    sizes_all = {m.name: m.size for m in members if not m.is_link}
    scene_files = [m.name for c in chosen for m in scene_members(members, c[0], root)]
    extract_needed(arc, scene_files, sizes_all, extract_dir)
    metas = {c[0]: read_scene_metadata(extract_dir, root, c[0]) for c in chosen}
    scene_sel = select_members(members, chosen, metas, root)
    model_metadata = sorted(
        n
        for n in scene_sel["files"]
        if n.startswith(root + "models/") and n.endswith("/metadata.json")
    )
    extract_needed(arc, model_metadata, sizes_all, extract_dir)
    model_metas = {}
    for n in model_metadata:
        try:
            model_metas[n] = json.loads((extract_dir / n).read_text())
        except (OSError, ValueError) as e:
            log(f"WARNING: unreadable model metadata {n}: {e}")
    sel = select_members(members, chosen, {**metas, **model_metas}, root)
    total_refs = len(sel["resolved"]) + len(sel["unresolved"])
    frac = len(sel["unresolved"]) / total_refs if total_refs else 1.0
    log(
        f"metadata refs: {len(sel['resolved'])} resolved, {len(sel['unresolved'])} unresolved"
    )
    for u in sorted(sel["unresolved"])[:20]:
        log(f"  unresolved: {u}")
    files = set(sel["files"])
    fallback = False
    if frac > UNRESOLVED_FALLBACK_FRACTION:
        fallback = True
        mat = [n for n in sel["member_set"] if n.startswith(root + "Materials/")]
        log(
            f"{frac:.0%} unresolved -> extracting whole Materials/: {len(mat)} files, "
            f"{human(sum(sizes_all[n] for n in mat))}"
        )
        files.update(mat)

    def cap(names: set[str]) -> None:
        total = sum(sizes_all.get(n, 0) for n in names | set(scene_files))
        if max_bytes is not None and total > max_bytes:
            raise SystemExit(
                f"--max-bytes exceeded: selection is {human(total)} > cap {human(max_bytes)}"
            )

    cap(files)
    extract_needed(arc, sorted(files), sizes_all, extract_dir)
    # closure: scan extracted text assets (mdl/usda) for further textures / sublayers.
    scanned: set[str] = set()
    module_files: set[str] = set()
    for _ in range(6):
        new: set[str] = set()
        for n in sorted(files):
            if n in scanned or n not in sizes_all:
                continue
            scanned.add(n)
            for ref in mdl_or_usd_text_refs(extract_dir / n):
                hit = resolve_ref(
                    ref, sel["member_set"], root, base_dir=posixpath.dirname(n)
                )
                if hit and hit not in files:
                    new.add(hit)
            # MDL module imports are not paths, so PATH_RE never sees them: pull in the module files the
            # extracted materials import (see mdl_module_imports for why this matters).
            for mod in mdl_module_imports(extract_dir / n):
                hit = module_member(mod, sel["member_set"], root)
                if hit and hit not in files:
                    new.add(hit)
                    module_files.add(hit)
        if not new:
            break
        files |= new
        cap(files)
        extract_needed(arc, sorted(new), sizes_all, extract_dir)
    if module_files:
        log(
            f"mdl module closure: {len(module_files)} module file(s) pulled in by imports: "
            f"{sorted(n.rsplit('/', 1)[-1] for n in module_files)}"
        )
    total_bytes = sum(sizes_all.get(n, 0) for n in files | set(scene_files))
    log(
        f"extracted assets: {len(files)} files + {len(scene_files)} scene files, {human(total_bytes)}"
    )
    return {
        "root": root,
        "asset_files": sorted(files),
        "scene_files": scene_files,
        "resolved": len(sel["resolved"]),
        "unresolved": len(sel["unresolved"]),
        "materials_fallback": fallback,
        "mdl_modules": sorted(module_files),
        "total_bytes": total_bytes,
    }


# ----------------------------------------------------------------------------
# upload tree + manifest + rclone
# ----------------------------------------------------------------------------


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if (
            dst.is_file()
            and not dst.is_symlink()
            and dst.stat().st_size == src.stat().st_size
        ):
            return
        dst.unlink()
    try:
        os.link(src, dst)  # hardlink: no extra disk
    except OSError:
        shutil.copy2(src, dst)


def build_upload_tree(
    extract_dir: Path, root: str, names: list[str], upload_dir: Path
) -> list[Path]:
    """Mirror `<root>/<name>` into upload_dir/<name> as REAL files (symlinks skipped)."""
    out = []
    for n in sorted(set(names)):
        src = extract_dir / n
        if src.is_symlink() or not src.is_file():
            continue
        rel = n.removeprefix(root)
        dst = upload_dir / rel
        _link_or_copy(src, dst)
        out.append(dst)
    return out


def write_manifest(
    upload_dir: Path, hf_sha: str, scene_ids: list[str], prefix: str
) -> dict:
    files = [
        p for p in upload_dir.rglob("*") if p.is_file() and p.name != "manifest.json"
    ]
    usd_hashes = {
        p.relative_to(upload_dir).as_posix(): sha256_file(p)
        for p in sorted(files)
        if p.parent.parent.name == "scenes" and SCENE_USD_GLOB_RE.match(p.name)
    }
    manifest = {
        "source": f"huggingface.co/datasets/{HF_REPO}",
        "hf_revision": hf_sha,
        "prefix": prefix,
        "scene_ids": scene_ids,
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "start_result_sha256": usd_hashes,
        "license": LICENSE_NOTE,
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }
    (upload_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def local_totals(upload_dir: Path) -> tuple[int, int]:
    files = [p for p in upload_dir.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def rclone_upload(upload_dir: Path, bucket: str, prefix: str, rclone_env: dict) -> dict:
    remote = f"{REMOTE}:{bucket}/{prefix.strip('/')}/"
    _run(
        [
            "rclone",
            "copy",
            str(upload_dir),
            remote,
            "--transfers",
            "8",
            "--checksum",
            "--stats",
            "30s",
            "-v",
        ],
        env=rclone_env,
    )
    out = subprocess.run(
        ["rclone", "size", "--json", remote],
        env=rclone_env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    remote_size = json.loads(out)
    count, total = local_totals(upload_dir)
    if remote_size.get("bytes") == total and remote_size.get("count") == count:
        status = "match"
    elif remote_size.get("bytes", 0) > total:
        status = "remote_larger"  # earlier runs' scenes may live under the same prefix
    else:
        status = "MISMATCH"
    log(
        f"rclone size: remote {remote_size} vs local count={count} bytes={total} -> {status}"
    )
    return {
        "remote": remote,
        "remote_count": remote_size.get("count"),
        "remote_bytes": remote_size.get("bytes"),
        "local_count": count,
        "local_bytes": total,
        "status": status,
    }


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def members_from_tree(extract_dir: Path) -> list[Member]:
    """Members of an already extracted tree (names relative to `extract_dir`, symlinks flagged), so the
    selection logic can run without the archive (``--from-tree``)."""
    members = []
    for dirpath, dirnames, filenames in os.walk(extract_dir):
        for name in [*dirnames, *filenames]:
            path = Path(dirpath) / name
            rel = path.relative_to(extract_dir).as_posix()
            if path.is_symlink():
                members.append(Member(rel, 0, True))
            elif path.is_file():
                members.append(Member(rel, path.stat().st_size))
    return members


def read_scene_ids(args) -> list[str] | None:
    ids = list(args.scene_id or [])
    if args.scene_ids_file:
        ids += [ln.strip() for ln in Path(args.scene_ids_file).read_text().splitlines() if ln.strip()]
    return ids or None


def link_tree(extract_dir: Path, root: str, link_to: str | None) -> Path:
    """Recreate the symlinks R2/zip extraction lost (`fetch_grscenes_from_r2.ensure_symlinks`), write env.sh and
    optionally link `link_to` -> the extracted tree. Returns the tree (``extract_dir/<root>``)."""
    import fetch_grscenes_from_r2 as from_r2

    tree = extract_dir / root.rstrip("/")
    made = from_r2.ensure_symlinks(tree)
    from_r2.write_env_sh(tree)
    log(f"created {made} relative symlink(s) in {tree}")
    if link_to:
        link = Path(link_to)
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            link.unlink()
            link.symlink_to(tree.resolve())
        elif link.is_dir():
            # A real directory (e.g. a single scene already pulled from R2): merge instead of failing, so the
            # two sources can share one layout. Scene folders and shared dirs are symlinked, never copied.
            added = merge_into_tree(tree, link)
            log(f"merged {added} new entr(ies) from {tree} into existing {link}")
        elif link.exists():
            raise SystemExit(f"--link-to {link} exists and is neither a directory nor a symlink")
        else:
            link.symlink_to(tree.resolve())
    return tree


def link_missing_files(src: Path, dst: Path) -> int:
    """Symlink every entry under `src` that `dst` lacks, recursing into real subdirectories.

    Returns how many links were added. Anything already present (including a symlink, which is left alone rather
    than fought over) is skipped, so this is idempotent and safe to re-run after every fetch.
    """
    added = 0
    for entry in sorted(src.iterdir()):
        dest = dst / entry.name
        if dest.is_symlink():
            continue
        if entry.is_dir() and not entry.is_symlink():
            if dest.exists() and not dest.is_dir():
                continue  # a file where a directory belongs: leave it for a human, never clobber
            dest.mkdir(parents=True, exist_ok=True)
            added += link_missing_files(entry, dest)
        elif not dest.exists():
            dest.symlink_to(entry.resolve())
            added += 1
    return added


def merge_into_tree(tree: Path, target: Path) -> int:
    """Link everything in `tree` into an existing `target` layout (``Materials/``, ``models/``, ``scenes/<id>_usd/``).

    Returns how many entries were added; existing ones are left alone. Entries are symlinked, so the extracted tree
    stays the single copy on disk and a scene folder's own relative ``Materials``/``models`` links keep resolving.

    Shared directories (``Materials/``, ``models/``) are merged **file by file**, not skipped because the directory
    name already exists: a target built earlier from a different source (e.g. one scene pulled from R2) must still
    receive what a later, larger extraction adds. Skipping the directory is what left ``Materials/KooPbr.mdl`` -- and
    514 other material files -- invisible on 2026-10-11, so every material importing that module failed to compile
    (see docs/arena_integration_log.md).
    """
    added = 0
    for entry in sorted(tree.iterdir()):
        if entry.name == "scenes":
            (target / "scenes").mkdir(exist_ok=True)
            for scene_dir in sorted(entry.iterdir()):
                dest = target / "scenes" / scene_dir.name
                if not dest.exists():
                    dest.symlink_to(scene_dir.resolve())
                    added += 1
        else:
            dest = target / entry.name
            if dest.is_symlink():
                continue  # already this tree
            if not dest.exists():
                dest.symlink_to(entry.resolve())
                added += 1
            elif entry.is_dir():
                added += link_missing_files(entry, dest)
    return added


def print_plan(args, stage: Path, sizes: dict[str, int], sha: str, prefix: str) -> None:
    log("DRY RUN -- nothing is downloaded, extracted or uploaded")
    log(f"HF revision: {sha}")
    total = sum(sizes.values())
    remote = not (args.all or args.skip_download)
    if remote:
        log(f"1. read the central directory of {', '.join(f'{k} ({human(v)})' for k, v in sizes.items())} over HTTP ranges")
    else:
        log(
            f"1. download to {stage}: "
            + ", ".join(f"{k} {human(v)}" for k, v in sizes.items())
            + (" (skipped: --skip-download)" if args.skip_download else "")
        )
        for name in sizes:
            log(f"   curl -fL -C - --retry 10 -o {stage / name} {HF_RESOLVE}/{PARTS_DIR}/{name}")
        log(f"2. 7z l -slt (else: zip -s 0 ... --out {stage / JOINED_NAME}, needs ~{human(total)} free)")
    which = (
        f"scene(s) {args.scene_id}"
        if args.scene_id
        else "ALL scenes"
        if args.all
        else f"the {args.scenes} smallest scene(s)"
    )
    log(f"3. list members, rank scenes by start_result_navigation.usd size, choose {which}")
    log(
        f"4. extract chosen scenes + referenced models/Materials into {stage / 'extract'}"
        + (f" (cap {human(args.max_bytes)})" if args.max_bytes else "")
    )
    if args.extract_only:
        log("5. (--extract-only) recreate symlinks, write env.sh, stop")
    else:
        log(f"5. build real-file upload tree {stage / 'upload'}/{{Materials,models,scenes/<id>_usd}} + manifest.json")
        log(f"6. rclone copy -> {REMOTE}:$R2_BUCKET/{prefix.strip('/')}/ then rclone size")
    print(
        json.dumps(
            {
                "dry_run": True,
                "stage": str(stage),
                "prefix": prefix,
                "hf_revision": sha,
                "download_bytes": 0 if remote else total,
            }
        )
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--category", choices=sorted(CATEGORIES), default="home", help="archive to read (default %(default)s)")
    p.add_argument(
        "--stage", default=DEFAULT_STAGE, help="working directory (default %(default)s)"
    )
    p.add_argument(
        "--prefix",
        default=None,
        help="R2 destination prefix (default worlds/grscenes/<category>)",
    )
    p.add_argument(
        "--scenes",
        type=int,
        default=1,
        help="number of smallest scenes to take (default 1)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="take every scene; downloads the whole archive and extracts with 7z instead of range requests",
    )
    p.add_argument(
        "--scene-id",
        action="append",
        default=None,
        help="explicit scene id (repeatable; with or without _usd) overriding --scenes",
    )
    p.add_argument("--scene-ids-file", default=None, help="file with one scene id per line (added to --scene-id)")
    p.add_argument(
        "--extract-only",
        action="store_true",
        help="stop after extraction: no upload tree, no R2 credentials needed",
    )
    p.add_argument("--link-to", default=None, help="with --extract-only: symlink this path to the extracted tree")
    p.add_argument(
        "--from-tree",
        action="store_true",
        help="no archive: select --scene-id scenes and their assets from the tree already in --stage/extract, upload them",
    )
    p.add_argument(
        "--keep-zips",
        action="store_true",
        help="keep downloaded parts / joined zip after extraction",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="assume parts (or joined zip) are already in --stage (local 7z path)",
    )
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="abort if extracted selection exceeds this many bytes",
    )
    return p.parse_args(argv)


def open_archive(args, stage: Path, info: dict) -> Archive:
    """Remote (HTTP range) archive by default; downloaded parts + 7z for --all / --skip-download."""
    sizes = info["sizes"]
    if not (args.all or args.skip_download):
        urls = [f"{HF_RESOLVE}/{PARTS_DIR}/{n}" for n in sizes]
        vols = remote_zip.HttpVolumes(urls, list(sizes.values()))
        return Archive(stage, "remote", remote_zip.RemoteZip(vols))
    joined = stage / JOINED_NAME
    if not args.skip_download and not joined.exists():
        download_parts(stage, sizes)
    return make_readable(stage, sizes)


def main(argv=None) -> None:
    args = parse_args(argv)
    parts, default_prefix = CATEGORIES[args.category]
    prefix = args.prefix or default_prefix
    stage = Path(args.stage)
    extract_dir = stage / "extract"
    sha_file = stage / "hf_revision.txt"

    if args.from_tree:
        ids = read_scene_ids(args)
        if not ids:
            raise SystemExit("--from-tree needs --scene-id / --scene-ids-file")
        info = {"sha": sha_file.read_text().strip() if sha_file.is_file() else "unknown", "sizes": {}}
        arc, members = None, members_from_tree(extract_dir)
    else:
        try:
            info = fetch_hf_info(timeout=10 if args.dry_run else 30, parts=parts)
        except Exception as e:  # noqa: BLE001 - offline dry-run must still work
            if not args.dry_run and not args.skip_download:
                raise SystemExit(f"Could not read HF API: {e}")
            log(f"HF API unavailable ({e}); using built-in sizes")
            info = {"sha": "unknown", "sizes": {n: KNOWN_SIZES[n] for n in parts}}
        if args.dry_run:
            print_plan(args, stage, info["sizes"], info["sha"], prefix)
            return
        stage.mkdir(parents=True, exist_ok=True)
        sha_file.write_text(info["sha"] + "\n")
        arc = open_archive(args, stage, info)
        members = list_members(arc)
        log(f"archive lists {len(members)} members via {arc.kind}")
        if arc.kind != "remote":
            delete_parts_if_joined(stage, arc, args.keep_zips, list(info["sizes"]))
        ids = read_scene_ids(args)

    chosen, ranking = choose_scenes(members, None if args.all else args.scenes, ids)
    log("scene ranking (smallest first; size of start_result usd):")
    for sid, size, usd, _ in ranking:
        log(
            f"  {size:>14}  {sid}  [{usd}]{'  <== chosen' if any(c[0] == sid for c in chosen) else ''}"
        )

    result = extract_selection(arc, members, chosen, extract_dir, args.max_bytes)
    root = result["root"]
    scene_ids = [c[0] for c in chosen]
    if args.extract_only:
        tree = link_tree(extract_dir, root, args.link_to)
        if arc is not None and arc.kind != "remote" and not args.keep_zips:
            for name in [*info["sizes"], JOINED_NAME]:
                (stage / name).unlink(missing_ok=True)
        print(json.dumps({"scene_ids": scene_ids, "tree": str(tree), "hf_revision": info["sha"], "total_bytes": result["total_bytes"], "refs_unresolved": result["unresolved"], "materials_fallback": result["materials_fallback"], "mdl_modules": result["mdl_modules"]}))
        return

    env = _require_env()
    rclone_env = _rclone_env(env)
    upload_dir = stage / "upload"
    build_upload_tree(
        extract_dir, root, result["asset_files"] + result["scene_files"], upload_dir
    )
    manifest = write_manifest(upload_dir, info["sha"], scene_ids, prefix)
    up = rclone_upload(upload_dir, env["R2_BUCKET"], prefix, rclone_env)
    if arc is not None and arc.kind != "remote" and not args.keep_zips:
        for name in [*info["sizes"], JOINED_NAME]:
            (stage / name).unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "scene_ids": scene_ids,
                "prefix": prefix,
                "hf_revision": info["sha"],
                "refs_resolved": result["resolved"],
                "refs_unresolved": result["unresolved"],
                "materials_fallback": result["materials_fallback"],
                "mdl_modules": result["mdl_modules"],
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
                "upload": up,
            }
        )
    )
    if up["status"] == "MISMATCH":
        sys.exit(1)


if __name__ == "__main__":
    main()
