import shutil
import stat
import struct
import subprocess
import urllib.error
import zipfile

import pytest
import remote_zip as rz

FILES = {
    "root/a.txt": b"alpha" * 1000,
    "root/sub/b.bin": bytes(range(256)) * 400,
    "root/sub/c.txt": b"c",
    "root/empty.txt": b"",
}


def _make_zip(path, compression=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(path, "w", compression) as zf:
        zf.writestr("root/", b"")
        for name, data in FILES.items():
            zf.writestr(name, data)
        link = zipfile.ZipInfo("root/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(link, "a.txt")


@pytest.mark.parametrize("compression", [zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED])
def test_list_and_extract_single_volume(tmp_path, compression):
    z = tmp_path / "one.zip"
    _make_zip(z, compression)
    vols = rz.LocalVolumes([z])
    members = {m.name: m for m in rz.list_members(vols)}
    assert members["root/"].is_dir and members["root/link"].is_link
    assert members["root/a.txt"].size == len(FILES["root/a.txt"])
    out = tmp_path / "out"
    wanted = [members[n] for n in (*FILES, "root/link", "root/")]
    written = rz.extract_members(vols, wanted, out)
    assert written == sum(len(v) for v in FILES.values())
    for name, data in FILES.items():
        assert (out / name).read_bytes() == data
    assert not (out / "root/link").exists()  # symlink members are never written


@pytest.mark.skipif(shutil.which("zip") is None, reason="needs the zip binary")
def test_split_archive_members_straddle_volumes(tmp_path):
    src = tmp_path / "src"
    (src / "d").mkdir(parents=True)
    expected = {}
    for i in range(12):
        data = bytes((i * 7 + j) % 251 for j in range(40_000))  # spans split points
        (src / "d" / f"f{i}.bin").write_bytes(data)
        expected[f"d/f{i}.bin"] = data
    z = tmp_path / "split.zip"
    subprocess.run(["zip", "-q", "-0", "-r", "-s", "64k", str(z), "d"], cwd=src.parent / "src", check=True)
    parts = sorted(tmp_path.glob("split.z*"))
    assert [p.suffix for p in parts][-1] == ".zip" and len(parts) >= 3
    ordered = [p for p in parts if p.suffix != ".zip"] + [z]
    vols = rz.LocalVolumes(ordered)
    members = {m.name: m for m in rz.list_members(vols)}
    assert {m.volume for m in members.values()} > {0}
    out = tmp_path / "out"
    rz.extract_members(vols, [members[n] for n in expected], out, workers=3)
    for name, data in expected.items():
        assert (out / name).read_bytes() == data


def test_crc_mismatch_is_detected(tmp_path):
    z = tmp_path / "bad.zip"
    _make_zip(z, zipfile.ZIP_STORED)
    raw = bytearray(z.read_bytes())
    at = raw.index(b"alphaalpha")
    raw[at] ^= 0xFF  # corrupt stored data
    z.write_bytes(bytes(raw))
    vols = rz.LocalVolumes([z])
    member = next(m for m in rz.list_members(vols) if m.name == "root/a.txt")
    with pytest.raises(ValueError, match="CRC"):
        rz.extract_members(vols, [member], tmp_path / "out")


def test_parse_central_directory_zip64_fields():
    name = b"big/file.bin"
    extra = struct.pack("<HHQQQ", 1, 24, 6_000_000_000, 5_000_000_000, 70_000_000_000) + struct.pack("<HH", 0x7855, 0)
    header = struct.pack(
        "<4s6H3I5H2I",
        rz.CD_SIG, 45, 45, 0x800, 8, 0, 0, 0xDEADBEEF, 0xFFFFFFFF, 0xFFFFFFFF, len(name), len(extra), 0, 0, 0, 0x81A40000, 0xFFFFFFFF,
    )  # fmt: skip
    (m,) = rz.parse_central_directory(header + name + extra)
    assert (m.name, m.size, m.compressed, m.offset, m.volume) == (
        "big/file.bin", 6_000_000_000, 5_000_000_000, 70_000_000_000, 0,
    )  # fmt: skip
    assert not m.is_link and not m.is_dir


def test_plan_groups_coalesces_close_spans_and_caps_span():
    spans = [(0, 100), (150, 100), (10_000, 50), (10_060, 50), (20_000, 500)]
    assert rz.plan_groups(spans, max_gap=100, max_span=1_000) == [[0, 1], [2, 3], [4]]
    assert rz.plan_groups(spans, max_gap=100_000, max_span=300) == [[0, 1], [2, 3], [4]]
    assert rz.plan_groups(spans, max_gap=100_000, max_span=100_000) == [[0, 1, 2, 3, 4]]
    assert rz.plan_groups([]) == []


def test_global_read_crosses_volumes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"0123456789")
    b.write_bytes(b"abcdef")
    vols = rz.LocalVolumes([a, b])
    assert rz.read_global(vols, 8, 5) == b"89abc"
    with pytest.raises(EOFError):
        rz.read_global(vols, 14, 5)


def test_unsafe_member_path_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        rz._safe_target(tmp_path, "../evil")
    with pytest.raises(ValueError, match="unsafe"):
        rz._safe_target(tmp_path, "/abs/evil")
    assert rz._safe_target(tmp_path, "root/a.txt") == tmp_path / "root/a.txt"


def test_remote_zip_wrapper_extracts_by_name(tmp_path):
    z = tmp_path / "one.zip"
    _make_zip(z)
    rzip = rz.RemoteZip(rz.LocalVolumes([z]))
    assert "root/a.txt" in {m.name for m in rzip.members()}
    rzip.extract(["root/sub/c.txt"], tmp_path / "out")
    assert (tmp_path / "out/root/sub/c.txt").read_bytes() == b"c"


def test_http_volumes_retries_429_with_retry_after(monkeypatch):
    """Hugging Face answered 429 and an overnight run lost the whole archive; the reader must back off and retry."""
    calls = {"n": 0}
    slept: list[float] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"abcd"

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "3"}, None)
        return _Resp()

    monkeypatch.setattr(rz.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(rz.time, "sleep", lambda s: slept.append(s))
    vols = rz.HttpVolumes(["http://x/part1"], [4], requests_per_s=0)
    assert vols.read(0, 0, 4) == b"abcd"
    assert calls["n"] == 3
    assert slept[:2] == [3.0, 3.0]  # the server's own Retry-After, not the exponential default


def test_http_volumes_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr(
        rz.urllib.request,
        "urlopen",
        lambda req, timeout=None: (_ for _ in ()).throw(urllib.error.HTTPError(req.full_url, 429, "x", None, None)),
    )
    monkeypatch.setattr(rz.time, "sleep", lambda s: None)
    vols = rz.HttpVolumes(["http://x/p"], [4], retries=3, requests_per_s=0)
    with pytest.raises(urllib.error.HTTPError):
        vols.read(0, 0, 4)


def test_http_volumes_paces_requests(monkeypatch):
    """A shared token bucket keeps concurrent readers under the rate limit instead of bursting."""
    monkeypatch.setattr(rz.time, "sleep", lambda s: None)
    vols = rz.HttpVolumes(["http://x/p"], [4], requests_per_s=2.0)
    with pytest.MonkeyPatch.context() as mp:
        clock = {"t": 100.0}
        mp.setattr(rz.time, "monotonic", lambda: clock["t"])
        vols._wait_turn()
        vols._next_at = 101.0  # pretend another reader just reserved the next slot
        assert vols._min_interval == pytest.approx(0.5)
        vols._wait_turn()
        assert vols._next_at == pytest.approx(101.5)
