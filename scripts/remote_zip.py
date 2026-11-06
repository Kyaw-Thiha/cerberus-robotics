"""Read selected members of a (possibly split, possibly zip64) zip without downloading the whole archive.

``fetch_grscenes_to_r2.py`` needs a few GB out of the 74.7 GB GRScenes-100 archives (home: ``.z01`` + ``.z02`` +
``.zip`` split set; commercial: one zip). A zip keeps its central directory at the end, so listing it takes a few
range requests, and every member is addressable by (volume, offset): we fetch only the byte ranges of the members we
want (neighbouring members coalesced into one request), inflate them and verify the CRC. Standard library only;
Python's ``zipfile`` cannot be used because it refuses archives that span several disks.

Volumes are numbered like the zip "disk number": ``.z01`` is 0, ``.z02`` is 1, ``.zip`` is the last. A member's data
may straddle two volumes (``zip -s`` splits at a byte count), which ``read_global`` handles by addressing the volumes
as one concatenated byte space.
"""

from __future__ import annotations

import os
import posixpath
import random
import stat
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOCATOR_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CD_SIG = b"PK\x01\x02"
LOCAL_SIG = b"PK\x03\x04"
EOCD_SEARCH = 65_535 + 22 + 20  # max comment + EOCD + zip64 locator
METHOD_STORED, METHOD_DEFLATED = 0, 8
LOCAL_EXTRA_SLACK = 256  # local-header extra field budget when sizing a coalesced range (refetched if exceeded)
DEFAULT_MAX_GAP = 256 * 1024  # members closer than this share one range request
DEFAULT_MAX_SPAN = 64 * 1024 * 1024


class Volumes(Protocol):
    """Byte source of the archive volumes."""

    sizes: list[int]

    def read(self, volume: int, offset: int, n: int) -> bytes: ...


@dataclass(frozen=True)
class RemoteMember:
    name: str
    method: int
    crc: int
    compressed: int
    size: int
    volume: int
    offset: int  # of the local header, within `volume`
    name_len: int
    flags: int
    attr: int  # external attributes (unix mode in the high 16 bits)

    @property
    def is_dir(self) -> bool:
        return self.name.endswith("/")

    @property
    def is_link(self) -> bool:
        return stat.S_ISLNK(self.attr >> 16)


class LocalVolumes:
    """Volumes on local disk (tests, and archives already downloaded)."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = [Path(p) for p in paths]
        self.sizes = [p.stat().st_size for p in self.paths]

    def read(self, volume: int, offset: int, n: int) -> bytes:
        with open(self.paths[volume], "rb") as f:
            f.seek(offset)
            return f.read(n)


class HttpVolumes:
    """Volumes behind URLs that honour ``Range`` (Hugging Face ``resolve`` links redirect to a CDN that does).

    Range reads against Hugging Face are rate limited: an overnight run lost a whole archive to ``HTTP 429``
    with 8 concurrent readers and the plain exponential backoff. So reads are paced by a shared token bucket
    (``requests_per_s``, default from ``$REMOTE_ZIP_RPS``), 429/503 honour ``Retry-After`` when present, and the
    backoff carries jitter so parallel readers do not retry in lockstep.
    """

    DEFAULT_RETRIES = 8

    def __init__(self, urls: list[str], sizes: list[int], retries: int | None = None,
                 requests_per_s: float | None = None) -> None:
        self.urls = urls
        self.sizes = sizes
        self.retries = retries if retries is not None else self.DEFAULT_RETRIES
        if requests_per_s is None:
            requests_per_s = float(os.environ.get("REMOTE_ZIP_RPS", "6"))
        self._min_interval = 1.0 / requests_per_s if requests_per_s > 0 else 0.0
        self._lock = threading.Lock()
        self._next_at = 0.0

    def _wait_turn(self) -> None:
        """Serialise request starts so the whole process stays under the rate limit."""
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_at)
            self._next_at = start + self._min_interval
        delay = start - now
        if delay > 0:
            time.sleep(delay)

    def read(self, volume: int, offset: int, n: int) -> bytes:
        req = urllib.request.Request(self.urls[volume], headers={"Range": f"bytes={offset}-{offset + n - 1}"})
        err: Exception = OSError("no attempt made")
        for attempt in range(self.retries):
            self._wait_turn()
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = resp.read()
                if len(body) == n:
                    return body
                err = OSError(f"short read {len(body)}/{n} bytes")
            except OSError as e:  # URLError, HTTPError, timeouts, IncompleteRead are all OSError/HTTPException
                err = e
                if isinstance(e, urllib.error.HTTPError) and e.code in (429, 503):
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    if retry_after and retry_after.isdigit():
                        err = e  # fall through to the sleep below with the server's own delay
                        time.sleep(min(120.0, float(retry_after)))
                        continue
            if attempt == self.retries - 1:
                raise err
            time.sleep(min(60.0, 2.0**attempt) * (0.5 + random.random()))
        raise err  # pragma: no cover


def read_global(vols: Volumes, start: int, n: int) -> bytes:
    """`n` bytes from the concatenation of all volumes, starting at global offset `start`."""
    out = []
    base = 0
    for volume, size in enumerate(vols.sizes):
        if n <= 0:
            break
        if start < base + size:
            local = start - base
            take = min(n, size - local)
            out.append(vols.read(volume, local, take))
            start += take
            n -= take
        base += size
    if n > 0:
        raise EOFError("read past the end of the archive")
    return b"".join(out)


def global_offset(vols: Volumes, volume: int, offset: int) -> int:
    return sum(vols.sizes[:volume]) + offset


def _zip64_fields(extra: bytes, want: list[str]) -> dict[str, int]:
    """Parse the zip64 extra field (id 0x0001); `want` lists which values are stored, in spec order."""
    pos = 0
    while pos + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, pos)
        if tag == 1:
            vals = {}
            cur = pos + 4
            for key in want:
                width = 4 if key == "disk" else 8
                vals[key] = int.from_bytes(extra[cur : cur + width], "little")
                cur += width
            return vals
        pos += 4 + size
    return {}


def parse_central_directory(blob: bytes) -> list[RemoteMember]:
    members = []
    pos = 0
    while pos + 46 <= len(blob) and blob[pos : pos + 4] == CD_SIG:
        (_, _, _, flags, method, _, _, crc, csize, usize, nlen, elen, clen, disk, _, attr, off) = struct.unpack_from(
            "<4s6H3I5H2I", blob, pos
        )
        name = blob[pos + 46 : pos + 46 + nlen].decode("utf-8" if flags & 0x800 else "cp437")
        extra = blob[pos + 46 + nlen : pos + 46 + nlen + elen]
        want = [k for k, v in (("size", usize), ("compressed", csize), ("offset", off)) if v == 0xFFFFFFFF]
        if disk == 0xFFFF:
            want.append("disk")
        if want:
            z = _zip64_fields(extra, want)
            usize, csize, off, disk = z.get("size", usize), z.get("compressed", csize), z.get("offset", off), z.get("disk", disk)
        members.append(RemoteMember(name, method, crc, csize, usize, disk, off, nlen, flags, attr))
        pos += 46 + nlen + elen + clen
    return members


def list_members(vols: Volumes) -> list[RemoteMember]:
    """All members (files and directories) from the central directory in the last volume(s)."""
    last = len(vols.sizes) - 1
    tail_n = min(vols.sizes[last], EOCD_SEARCH)
    tail = vols.read(last, vols.sizes[last] - tail_n, tail_n)
    eocd_at = tail.rfind(EOCD_SIG)
    if eocd_at < 0:
        raise ValueError("no end-of-central-directory record: not a zip, or the last volume is not the .zip part")
    (_, _, cd_disk, _, entries, cd_size, cd_offset, _) = struct.unpack_from("<4s4H2IH", tail, eocd_at)
    locator_at = eocd_at - 20
    if locator_at >= 0 and tail[locator_at : locator_at + 4] == EOCD64_LOCATOR_SIG:
        _, rec_disk, rec_offset, _ = struct.unpack_from("<4sIQI", tail, locator_at)
        rec = read_global(vols, global_offset(vols, rec_disk, rec_offset), 56)
        if rec[:4] != EOCD64_SIG:
            raise ValueError("zip64 end-of-central-directory record not found at the locator's address")
        (_, _, _, _, _, cd_disk, _, entries, cd_size, cd_offset) = struct.unpack_from("<4sQ2H2I4Q", rec, 0)
    blob = read_global(vols, global_offset(vols, cd_disk, cd_offset), cd_size)
    members = parse_central_directory(blob)
    if len(members) != entries:
        raise ValueError(f"central directory lists {len(members)} entries, end record says {entries}")
    return members


def plan_groups(
    spans: list[tuple[int, int]], max_gap: int = DEFAULT_MAX_GAP, max_span: int = DEFAULT_MAX_SPAN
) -> list[list[int]]:
    """Group indices of `spans` (``(global_start, length)``) into runs fetched with one request each: consecutive
    (by start) spans whose gap is at most `max_gap` and whose combined extent stays within `max_span`."""
    order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    groups: list[list[int]] = []
    group_end = 0
    group_start = 0
    for i in order:
        start, length = spans[i]
        if groups and start - group_end <= max_gap and start + length - group_start <= max_span:
            groups[-1].append(i)
            group_end = max(group_end, start + length)
        else:
            groups.append([i])
            group_start, group_end = start, start + length
    return groups


def _safe_target(out_dir: Path, name: str) -> Path:
    norm = posixpath.normpath(name)
    if norm.startswith(("/", "..")) or "/../" in norm:
        raise ValueError(f"unsafe member path {name!r}")
    return out_dir / norm


def decode_member(member: RemoteMember, data: bytes) -> bytes:
    if member.flags & 1:
        raise ValueError(f"{member.name}: encrypted member")
    if member.method == METHOD_STORED:
        out = data
    elif member.method == METHOD_DEFLATED:
        out = zlib.decompressobj(-15).decompress(data)
    else:
        raise ValueError(f"{member.name}: unsupported compression method {member.method}")
    if len(out) != member.size or zlib.crc32(out) != member.crc:
        raise ValueError(f"{member.name}: size/CRC mismatch after extraction")
    return out


def extract_members(
    vols: Volumes,
    members: Iterable[RemoteMember],
    out_dir: Path,
    workers: int = 4,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Write `members` (files only) below `out_dir`, keeping their archive paths. Returns bytes written."""
    files = [m for m in members if not m.is_dir and not m.is_link]
    spans = [
        (global_offset(vols, m.volume, m.offset), 30 + m.name_len + LOCAL_EXTRA_SLACK + m.compressed) for m in files
    ]
    groups = plan_groups(spans)
    total_end = sum(vols.sizes)

    def fetch_group(group: list[int]) -> int:
        lo = min(spans[i][0] for i in group)
        hi = min(total_end, max(spans[i][0] + spans[i][1] for i in group))
        buf = read_global(vols, lo, hi - lo)
        written = 0
        for i in group:
            m, (start, _) = files[i], spans[i]
            rel = start - lo
            header = buf[rel : rel + 30]
            if header[:4] != LOCAL_SIG:
                raise ValueError(f"{m.name}: no local header at volume {m.volume} offset {m.offset}")
            name_len, extra_len = struct.unpack_from("<HH", header, 26)
            data_at = rel + 30 + name_len + extra_len
            data = buf[data_at : data_at + m.compressed]
            if len(data) != m.compressed:
                data = read_global(vols, start + 30 + name_len + extra_len, m.compressed)
            payload = decode_member(m, data)
            target = _safe_target(out_dir, m.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            written += len(payload)
        return written

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n_written, group in zip(pool.map(fetch_group, groups), groups, strict=True):
            done += n_written
            if on_progress:
                on_progress(done, len(files))
    return done


class RemoteZip:
    """A zip (split or not) behind `vols`: ``members()`` lists it once, ``extract(names, out_dir)`` writes members."""

    def __init__(self, vols: Volumes) -> None:
        self.vols = vols
        self._by_name: dict[str, RemoteMember] | None = None

    def members(self) -> list[RemoteMember]:
        if self._by_name is None:
            self._by_name = {m.name: m for m in list_members(self.vols)}
        return list(self._by_name.values())

    def extract(
        self, names: Iterable[str], out_dir: Path, on_progress: Callable[[int, int], None] | None = None
    ) -> int:
        self.members()
        assert self._by_name is not None
        return extract_members(self.vols, [self._by_name[n] for n in names], out_dir, on_progress=on_progress)
