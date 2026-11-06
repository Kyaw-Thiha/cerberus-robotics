import json
import stat
import subprocess
import zipfile

import fetch_grscenes_from_r2 as from_r2
import fetch_grscenes_to_r2 as to_r2
import pytest

ROOT = "target_69_new/"


def _add(zf, name, data=b"x"):
    zf.writestr(name, data)


def _add_link(zf, name, target):
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    zf.writestr(info, target)


@pytest.fixture
def archive(tmp_path):
    """Synthetic split-free zip mimicking target_69_new/ (3 scenes, symlinks, models, materials)."""
    path = tmp_path / "home_joined.zip"
    meta_a = {
        "objects": [
            {
                "usd": "../../models/chair/chair.usd",
                "mdl": "Materials/wood.mdl",
                "mat2": "Materials/metal.mdl",
            }
        ],
        "missing": "models/nope.usd",
    }
    meta_b = {
        "models": ["/abs/path/target_69_new/models/table/table.usd"],
        "mat": {"path": "../../Materials/metal.mdl"},
    }
    with zipfile.ZipFile(path, "w") as zf:
        _add(zf, ROOT + "scenes/a_usd/start_result_navigation.usd", b"a" * 500)
        _add(zf, ROOT + "scenes/a_usd/start_result_raw.usd", b"a" * 5)
        _add(zf, ROOT + "scenes/a_usd/metadata.json", json.dumps(meta_a))
        _add_link(zf, ROOT + "scenes/a_usd/Materials", "../../Materials")
        _add(zf, ROOT + "scenes/b_usd/start_result_navigation.usd", b"b" * 100)
        _add(zf, ROOT + "scenes/b_usd/start_result_raw.usd", b"b" * 900)
        _add(zf, ROOT + "scenes/b_usd/metadata.json", json.dumps(meta_b))
        _add_link(zf, ROOT + "scenes/b_usd/models", "../../models")
        _add(zf, ROOT + "scenes/c_usd/start_result_raw.usd", b"c" * 300)  # raw only
        _add(zf, ROOT + "scenes/c_usd/metadata.json", "{}")
        _add(
            zf,
            ROOT + "models/chair/chair.usd",
            b"#usda 1.0\n(\n)\n@./textures/chair.png@ ",
        )
        _add(zf, ROOT + "models/chair/textures/chair.png", b"png")
        _add(
            zf,
            ROOT + "models/chair/metadata.json",
            json.dumps({"models": [], "materials": ["./Materials/textures/from_model.png"]}),
        )
        _add(zf, ROOT + "models/table/table.usd", b"t" * 10)
        _add(zf, ROOT + "models/other/other.usd", b"o" * 10)
        _add(zf, ROOT + "Materials/wood.mdl", b'texture_2d("textures/wood_d.png")')
        _add(zf, ROOT + "Materials/textures/wood_d.png", b"w" * 20)
        _add(zf, ROOT + "Materials/metal.mdl", b"mdl 1.6;")
        _add(zf, ROOT + "Materials/unused.mdl", b"mdl 1.6;")
        _add(zf, ROOT + "Materials/textures/from_model.png", b"model texture")
    return to_r2.Archive(path, "zipfile")


# ---------------------------------------------------------------------------
# listing + ranking
# ---------------------------------------------------------------------------


def test_list_members_flags_symlinks(archive):
    members = {m.name: m for m in to_r2.list_members(archive)}
    assert members[ROOT + "scenes/a_usd/Materials"].is_link
    assert not members[ROOT + "models/table/table.usd"].is_link


def test_parse_unzip_zl_and_7z_slt():
    zl = (
        "Archive:  x.zip\n"
        "Zip file size: 100 bytes, number of entries: 3\n"
        "-rw-a--     3.0 unx      123 bX defN 24-Jan-01 00:00 target_69_new/scenes/a_usd/start result.usd\n"
        "lrwxrwxrwx  3.0 unx       16 bx stor 24-Jan-01 00:00 target_69_new/scenes/a_usd/Materials\n"
        "drwxr-xr-x  3.0 unx        0 bx stor 24-Jan-01 00:00 target_69_new/scenes/\n"
        "3 files, 139 bytes uncompressed\n"
    )
    ms = to_r2.parse_unzip_zl(zl)
    assert [(m.name, m.size, m.is_link) for m in ms] == [
        ("target_69_new/scenes/a_usd/start result.usd", 123, False),
        ("target_69_new/scenes/a_usd/Materials", 16, True),
    ]
    slt = (
        "Path = x.zip\nType = zip\n\n----------\n"
        "Path = d/f.usd\nFolder = -\nSize = 42\nAttributes = A -rw-r--r--\n\n"
        "Path = d\nFolder = +\nSize = 0\n\n"
        "Path = d/l\nFolder = -\nSize = 5\nAttributes = A lrwxrwxrwx\n"
    )
    ms = to_r2.parse_7z_slt(slt)
    assert [(m.name, m.size, m.is_link) for m in ms] == [
        ("d/f.usd", 42, False),
        ("d/l", 5, True),
    ]


def test_rank_scenes_smallest_first_with_raw_fallback(archive):
    ranking = to_r2.rank_scenes(to_r2.list_members(archive))
    assert [(r[0], r[1], r[2]) for r in ranking] == [
        ("b_usd", 100, "start_result_navigation.usd"),
        ("c_usd", 300, "start_result_raw.usd"),
        ("a_usd", 500, "start_result_navigation.usd"),
    ]
    assert ranking[0][3] == ROOT


def test_choose_scenes_n_and_explicit_id(archive):
    members = to_r2.list_members(archive)
    chosen, ranking = to_r2.choose_scenes(members, 2, None)
    assert [c[0] for c in chosen] == ["b_usd", "c_usd"] and len(ranking) == 3
    chosen, _ = to_r2.choose_scenes(members, 1, ["a"])
    assert [c[0] for c in chosen] == ["a_usd"]
    with pytest.raises(SystemExit):
        to_r2.choose_scenes(members, 1, ["zzz"])


# ---------------------------------------------------------------------------
# metadata reference parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "meta, expected",
    [
        ({"a": "models/x/y.usd"}, {"models/x/y.usd"}),
        (
            {"list": ["../../Materials/m.mdl", "not a path", 3]},
            {"../../Materials/m.mdl"},
        ),
        (
            {"nested": {"deep": [{"p": "/abs/models/a/b.USDA"}]}},
            {"/abs/models/a/b.USDA"},
        ),
        ({"models/key/as_path.usd": 1}, {"models/key/as_path.usd"}),
        (["Materials/tex/t.PNG", "x/y/z.jpg"], {"Materials/tex/t.PNG", "x/y/z.jpg"}),
        (
            {"s": "refs @models/a.usd@ and 'Materials/b.mdl'"},
            {"models/a.usd", "Materials/b.mdl"},
        ),
        ({"dir": "models/chair", "n": 5}, set()),
    ],
)
def test_collect_path_strings_shapes(meta, expected):
    assert to_r2.collect_path_strings(meta) == expected


def test_resolve_ref_variants(archive):
    member_set = {m.name for m in to_r2.list_members(archive)}
    r = to_r2.resolve_ref
    assert (
        r("../../models/chair/chair.usd", member_set, ROOT)
        == ROOT + "models/chair/chair.usd"
    )
    assert (
        r("/some/abs/target_69_new/Materials/wood.mdl", member_set, ROOT)
        == ROOT + "Materials/wood.mdl"
    )
    assert (
        r("models\\table\\table.usd", member_set, ROOT)
        == ROOT + "models/table/table.usd"
    )
    assert (
        r("file:///x/models/table/table.usd", member_set, ROOT)
        == ROOT + "models/table/table.usd"
    )
    assert r("models/nope.usd", member_set, ROOT) is None
    assert (
        r("textures/wood_d.png", member_set, ROOT, base_dir=ROOT + "Materials")
        == ROOT + "Materials/textures/wood_d.png"
    )


def test_select_members_expands_model_folder_not_all_models(archive):
    members = to_r2.list_members(archive)
    chosen, _ = to_r2.choose_scenes(members, 1, ["a"])
    meta = {"a_usd": json.loads(archive_meta(archive, "a_usd"))}
    sel = to_r2.select_members(members, chosen, meta, ROOT)
    assert sel["unresolved"] == {"models/nope.usd"}
    assert (
        ROOT + "models/chair/textures/chair.png" in sel["files"]
    )  # sibling tree of resolved model
    assert ROOT + "models/other/other.usd" not in sel["files"]


def archive_meta(archive, sid):
    with zipfile.ZipFile(archive.path) as zf:
        return zf.read(f"{ROOT}scenes/{sid}/metadata.json").decode()


# ---------------------------------------------------------------------------
# extraction, upload tree, manifest
# ---------------------------------------------------------------------------


# MDL module closure: the dataset's materials import modules that no USD/metadata references (KooPbr), so a
# selective extraction used to leave them behind and every material then failed to compile on the pod.
def _module_archive(tmp_path):
    """Own fixture: the shared `archive` fixture is asserted on by other tests, so don't add members to it."""
    path = tmp_path / "modules.zip"
    with zipfile.ZipFile(path, "w") as zf:
        _add(zf, ROOT + "scenes/a_usd/start_result_navigation.usd", b"a" * 500)
        _add(
            zf,
            ROOT + "scenes/a_usd/metadata.json",
            json.dumps({"objects": [{"usd": "../../models/chair/chair.usd", "mdl": "Materials/wood.mdl"}]}),
        )
        _add(zf, ROOT + "models/chair/chair.usd", b"#usda 1.0\n(\n)\n")
        _add(
            zf,
            ROOT + "Materials/wood.mdl",
            b"mdl 1.6;\nimport ::KooPbr::KooMtl;\nimport ::KooPbr_maps::KooPbr_bitmap;\n",
        )
        _add(zf, ROOT + "Materials/KooPbr.mdl", b"mdl 1.6;\nmodule KooMtl {}\n")
        _add(zf, ROOT + "Materials/KooPbr_maps.mdl", b"mdl 1.6;\n")
        _add(zf, ROOT + "Materials/unrelated.mdl", b"mdl 1.6;\n")
    return to_r2.Archive(path, "zipfile")


def test_merge_links_new_files_inside_an_existing_shared_dir(tmp_path):
    """A target built earlier (e.g. from one R2 scene) must still receive files a later extraction adds."""
    tree = tmp_path / "extract" / "root"
    target = tmp_path / "target"
    (tree / "Materials" / "textures").mkdir(parents=True)
    (tree / "Materials" / "wood.mdl").write_text("mdl 1.6;")
    (tree / "Materials" / "KooPbr.mdl").write_text("mdl 1.6;")
    (tree / "Materials" / "textures" / "wood_d.png").write_bytes(b"png")
    (tree / "models" / "chair").mkdir(parents=True)
    (tree / "models" / "chair" / "chair.usd").write_text("#usda 1.0")
    (target / "Materials").mkdir(parents=True)
    (target / "Materials" / "wood.mdl").write_text("already here")

    added = to_r2.merge_into_tree(tree, target)

    assert (target / "Materials" / "KooPbr.mdl").is_symlink()  # new file in an EXISTING shared dir
    assert (target / "Materials" / "textures" / "wood_d.png").is_symlink()  # and in a new subdir of it
    assert (target / "models").is_symlink()  # a whole new shared dir is linked as one entry
    assert (target / "models" / "chair" / "chair.usd").is_file()  # reachable through that link
    assert (target / "Materials" / "wood.mdl").read_text() == "already here"  # never clobbered
    assert added == 3  # 2 files in Materials + 1 new top-level dir
    assert to_r2.merge_into_tree(tree, target) == 0  # idempotent


def test_mdl_module_imports_parses_absolute_and_relative_forms(tmp_path):
    f = tmp_path / "m.mdl"
    f.write_text(
        "mdl 1.7;\n"
        "import ::state::normal;\n"
        "import ::KooPbr::KooMtl;\n"
        "import .::OmniUe4Function import *;\n"
        "// import ::commented_out::x;\n"
    )
    assert to_r2.mdl_module_imports(f) == {"state", "KooPbr", "OmniUe4Function"}
    assert to_r2.mdl_module_imports(tmp_path / "absent.mdl") == set()
    assert to_r2.mdl_module_imports(tmp_path / "not_mdl.usda") == set()


def test_module_member_prefers_the_materials_dir_then_any_match():
    members = {ROOT + "Materials/KooPbr.mdl", ROOT + "elsewhere/KooPbr.mdl"}
    assert to_r2.module_member("KooPbr", members, ROOT) == ROOT + "Materials/KooPbr.mdl"
    assert to_r2.module_member("Nested", {ROOT + "any/Nested.mdl"}, ROOT) == ROOT + "any/Nested.mdl"
    assert to_r2.module_member("Absent", members, ROOT) is None


def test_extraction_pulls_the_modules_its_materials_import(tmp_path):
    arc = _module_archive(tmp_path)
    members = to_r2.list_members(arc)
    chosen, _ = to_r2.choose_scenes(members, 1, ["a"])
    ext = tmp_path / "extract"
    result = to_r2.extract_selection(arc, members, chosen, ext, max_bytes=None)
    assert result["mdl_modules"] == [ROOT + "Materials/KooPbr.mdl", ROOT + "Materials/KooPbr_maps.mdl"]
    assert (ext / ROOT / "Materials/KooPbr.mdl").is_file()
    assert (ext / ROOT / "Materials/KooPbr_maps.mdl").is_file()
    assert not (ext / ROOT / "Materials/unrelated.mdl").exists()  # only what is imported


def test_extract_selection_upload_tree_and_manifest(archive, tmp_path):
    members = to_r2.list_members(archive)
    chosen, _ = to_r2.choose_scenes(members, 1, ["a"])
    ext = tmp_path / "extract"
    result = to_r2.extract_selection(archive, members, chosen, ext, max_bytes=None)
    assert result["root"] == ROOT
    assert result["unresolved"] == 1 and result["resolved"] == 4
    assert not result["materials_fallback"]
    names = set(result["asset_files"])
    assert ROOT + "Materials/textures/from_model.png" in names
    assert ROOT + "Materials/textures/wood_d.png" in names  # found by scanning wood.mdl
    assert ROOT + "Materials/unused.mdl" not in names
    assert not (
        ext / ROOT / "scenes/a_usd/Materials"
    ).exists()  # symlink members are never extracted

    up = tmp_path / "upload"
    to_r2.build_upload_tree(
        ext, ROOT, result["asset_files"] + result["scene_files"], up
    )
    assert (up / "scenes/a_usd/start_result_navigation.usd").is_file()
    assert (up / "models/chair/chair.usd").is_file()
    assert (up / "Materials/wood.mdl").is_file()
    assert not any(p.is_symlink() for p in up.rglob("*"))
    # idempotent
    to_r2.build_upload_tree(
        ext, ROOT, result["asset_files"] + result["scene_files"], up
    )

    manifest = to_r2.write_manifest(up, "deadbeef", ["a_usd"], "worlds/grscenes/home")
    on_disk = json.loads((up / "manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["hf_revision"] == "deadbeef" and manifest["scene_ids"] == ["a_usd"]
    files = [p for p in up.rglob("*") if p.is_file() and p.name != "manifest.json"]
    assert manifest["file_count"] == len(files)
    assert manifest["total_bytes"] == sum(p.stat().st_size for p in files)
    import hashlib

    assert manifest["start_result_sha256"] == {
        "scenes/a_usd/start_result_navigation.usd": hashlib.sha256(
            b"a" * 500
        ).hexdigest(),
        "scenes/a_usd/start_result_raw.usd": hashlib.sha256(b"a" * 5).hexdigest(),
    }
    assert "CC BY-NC-SA" in manifest["license"] and manifest["date_utc"].endswith("Z")


def test_materials_fallback_when_mostly_unresolved(archive, tmp_path):
    members = to_r2.list_members(archive)
    chosen, _ = to_r2.choose_scenes(
        members, 1, ["c"]
    )  # empty metadata -> everything unresolved
    result = to_r2.extract_selection(
        archive, members, chosen, tmp_path / "e", max_bytes=None
    )
    assert result["materials_fallback"]
    assert ROOT + "Materials/unused.mdl" in result["asset_files"]



def test_max_bytes_cap(archive, tmp_path):
    members = to_r2.list_members(archive)
    chosen, _ = to_r2.choose_scenes(members, 1, ["a"])
    with pytest.raises(SystemExit):
        to_r2.extract_selection(archive, members, chosen, tmp_path / "e", max_bytes=10)


def test_check_free_raises_when_disk_too_small(tmp_path, monkeypatch):
    monkeypatch.setattr(
        to_r2.shutil, "disk_usage", lambda p: type("U", (), {"free": 1024})()
    )
    with pytest.raises(SystemExit):
        to_r2.check_free(tmp_path, 10**9, "x")


# ---------------------------------------------------------------------------
# subprocess mocks: curl / rclone
# ---------------------------------------------------------------------------


def test_download_parts_resumes_and_verifies(tmp_path, monkeypatch):
    sizes = {n: 5 for n in to_r2.PART_NAMES}
    (tmp_path / "home_scenes.z01").write_bytes(b"12345")  # already complete -> skipped
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        out = cmd[cmd.index("-o") + 1]
        with open(out, "wb") as fh:
            fh.write(b"12345")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(to_r2.subprocess, "run", fake_run)
    to_r2.download_parts(tmp_path, sizes)
    assert len(calls) == 2
    assert all(
        c[0] == "curl"
        and "-C" in c
        and any(a.startswith("-") and "L" in a for a in c[1:3])
        for c in calls
    )
    assert calls[0][-1].endswith("/scenes/GRScenes-100/home_scenes.z02")


def test_rclone_upload_status(tmp_path, monkeypatch):
    up = tmp_path / "up"
    (up / "models").mkdir(parents=True)
    (up / "models/a.usd").write_bytes(b"abc")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[:2] == ["rclone", "size"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps({"count": 1, "bytes": 3}), stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(to_r2.subprocess, "run", fake_run)
    res = to_r2.rclone_upload(up, "bkt", "worlds/grscenes/home", {})
    assert calls[0][:2] == ["rclone", "copy"]
    assert calls[0][3] == "cerberus_r2:bkt/worlds/grscenes/home/"
    assert res["status"] == "match"


def test_dry_run_prints_plan_without_side_effects(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(to_r2.urllib.request, "urlopen", boom)
    to_r2.main(["--dry-run", "--stage", str(tmp_path / "s")])
    out = capsys.readouterr().out
    assert '"dry_run": true' in out
    assert not (tmp_path / "s").exists()


# ---------------------------------------------------------------------------
# from_r2
# ---------------------------------------------------------------------------


def test_from_r2_symlinks_and_env_sh(tmp_path):
    dest = tmp_path / "grscenes"
    (dest / "Materials").mkdir(parents=True)
    (dest / "models").mkdir()
    (dest / "scenes/a_usd").mkdir(parents=True)
    (dest / "scenes/b_usd").mkdir(parents=True)
    (dest / "scenes/b_usd/Materials").mkdir()  # real dir already there: left alone
    assert from_r2.find_scene_ids(dest) == ["a_usd", "b_usd"]
    assert from_r2.ensure_symlinks(dest) == 3
    link = dest / "scenes/a_usd/Materials"
    assert link.is_symlink() and str(link.readlink()) == "../../Materials"
    assert link.resolve() == (dest / "Materials").resolve()
    assert (dest / "scenes/a_usd/models").resolve() == (dest / "models").resolve()
    assert not (dest / "scenes/b_usd/Materials").is_symlink()
    assert from_r2.ensure_symlinks(dest) == 0  # idempotent
    line = from_r2.write_env_sh(dest)
    assert line == f"export MDL_SYSTEM_PATH={dest.resolve() / 'Materials'}"
    assert (dest / "env.sh").read_text() == line + "\n"

def test_from_r2_recreates_model_material_links(tmp_path):
    dest = tmp_path / "grscenes"
    (dest / "Materials").mkdir(parents=True)
    model = dest / "models/object/table/hash"
    model.mkdir(parents=True)
    (model / "instance.usd").write_text("")
    assert from_r2.ensure_symlinks(dest) == 1
    link = model / "Materials"
    assert link.is_symlink()
    assert link.resolve() == (dest / "Materials").resolve()
    assert from_r2.ensure_symlinks(dest) == 0


def test_from_r2_main_runs_rclone_and_prints(tmp_path, monkeypatch, capsys):
    for v in from_r2_env():
        monkeypatch.setenv(*v)
    dest = tmp_path / "d"
    calls = []

    def fake_run(cmd, **kw):
        if cmd[:2] == ["rclone", "version"]:  # the sync_to_r2.check_rclone preflight
            return subprocess.CompletedProcess(cmd, 0, stdout="rclone v1.75.1\n", stderr="")
        calls.append(cmd)
        (dest / "Materials").mkdir(parents=True, exist_ok=True)
        (dest / "models").mkdir(exist_ok=True)
        (dest / "scenes/s1_usd").mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(from_r2.subprocess, "run", fake_run)
    from_r2.main(["--dest", str(dest), "--scene-id", "s1"])
    cmd = calls[0]
    assert (
        cmd[:2] == ["rclone", "copy"]
        and cmd[2] == "cerberus_r2:bkt/worlds/grscenes/home/"
    )
    assert "/scenes/s1_usd/**" in cmd and "/Materials/**" in cmd
    out = capsys.readouterr().out
    assert "s1_usd" in out and "export MDL_SYSTEM_PATH=" in out
    assert (dest / "scenes/s1_usd/models").is_symlink()


def from_r2_env():
    return [
        ("R2_ACCOUNT_ID", "acc"),
        ("R2_ACCESS_KEY_ID", "k"),
        ("R2_SECRET_ACCESS_KEY", "s"),
        ("R2_BUCKET", "bkt"),
    ]


def test_expand_siblings_takes_only_the_asset_folder():
    root = "home_scenes/"
    a = root + "models/object/others/table/h1/instance.usd"
    by_dir = {
        root + "models/object/others/table/h1/": [a, root + "models/object/others/table/h1/metadata.json"],
        root + "models/object/others/table/h2/": [root + "models/object/others/table/h2/instance.usd"],
    }
    got = to_r2.expand_siblings(a, root, by_dir)
    assert sorted(got) == sorted(by_dir[root + "models/object/others/table/h1/"])


# ---------------------------------------------------------------------------
# many scenes: remote (HTTP range) archive, --all, categories, extract-only, --from-tree
# ---------------------------------------------------------------------------


def _remote(archive):
    import remote_zip

    return to_r2.Archive(
        archive.path.parent, "remote", remote_zip.RemoteZip(remote_zip.LocalVolumes([archive.path]))
    )


def test_choose_scenes_all_with_none(archive):
    members = to_r2.list_members(archive)
    chosen, ranking = to_r2.choose_scenes(members, None, None)
    assert chosen == ranking and len(chosen) == 3


def test_remote_archive_lists_and_extracts_like_the_zip(archive, tmp_path):
    remote = _remote(archive)
    assert {(m.name, m.size, m.is_link) for m in to_r2.list_members(remote)} == {
        (m.name, m.size, m.is_link) for m in to_r2.list_members(archive)
    }
    members = to_r2.list_members(remote)
    chosen, _ = to_r2.choose_scenes(members, 1, ["a"])
    result = to_r2.extract_selection(remote, members, chosen, tmp_path / "ext", max_bytes=None)
    assert result["unresolved"] == 1 and result["resolved"] == 4
    assert (tmp_path / "ext" / ROOT / "Materials/textures/wood_d.png").read_bytes() == b"w" * 20
    assert not (tmp_path / "ext" / ROOT / "scenes/a_usd/Materials").exists()


def test_extract_only_main_links_the_tree_without_r2_credentials(archive, tmp_path, monkeypatch, capsys):
    for var in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(to_r2, "fetch_hf_info", lambda timeout=30, parts=None: {"sha": "rev", "sizes": {"home_scenes.zip": 1}})
    monkeypatch.setattr(to_r2, "open_archive", lambda args, stage, info: _remote(archive))
    link = tmp_path / "data" / "grscenes" / "home"
    to_r2.main(["--stage", str(tmp_path / "s"), "--scenes", "2", "--extract-only", "--link-to", str(link)])
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["scene_ids"] == ["b_usd", "c_usd"] and summary["tree"].endswith(ROOT.rstrip("/"))
    assert link.is_symlink() and sorted(p.name for p in (link / "scenes").iterdir()) == ["b_usd", "c_usd"]
    assert (link / "scenes/b_usd/Materials").is_symlink() and (link / "env.sh").is_file()
    assert (tmp_path / "s" / "hf_revision.txt").read_text().strip() == "rev"
    to_r2.main(["--stage", str(tmp_path / "s"), "--scenes", "2", "--extract-only", "--link-to", str(link)])  # idempotent


def test_from_tree_uploads_only_the_requested_scenes_and_their_assets(archive, tmp_path, monkeypatch):
    for k, v in from_r2_env():
        monkeypatch.setenv(k, v)
    members = to_r2.list_members(archive)
    everything, _ = to_r2.choose_scenes(members, None, None)
    stage = tmp_path / "s"
    to_r2.extract_selection(archive, members, everything, stage / "extract", max_bytes=None)
    (stage / "hf_revision.txt").write_text("rev9\n")
    ids = tmp_path / "ids.txt"
    ids.write_text("a_usd\n")
    uploads = []

    def fake_run(cmd, **kw):
        if cmd[:2] == ["rclone", "version"]:  # the sync_to_r2.check_rclone preflight
            return subprocess.CompletedProcess(cmd, 0, stdout="rclone v1.75.1\n", stderr="")
        if cmd[:2] == ["rclone", "size"]:
            up = stage / "upload"
            n = sum(1 for p in up.rglob("*") if p.is_file())
            b = sum(p.stat().st_size for p in up.rglob("*") if p.is_file())
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"count": n, "bytes": b}), stderr="")
        uploads.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(to_r2.subprocess, "run", fake_run)
    to_r2.main(["--from-tree", "--stage", str(stage), "--scene-ids-file", str(ids)])
    up = stage / "upload"
    assert (up / "scenes/a_usd/start_result_navigation.usd").is_file()
    assert not (up / "scenes/b_usd").exists() and not (up / "scenes/c_usd").exists()
    assert (up / "models/chair/chair.usd").is_file() and not (up / "models/other/other.usd").exists()
    manifest = json.loads((up / "manifest.json").read_text())
    assert manifest["hf_revision"] == "rev9" and manifest["scene_ids"] == ["a_usd"]
    assert uploads[0][3] == "cerberus_r2:bkt/worlds/grscenes/home/"


def test_from_tree_needs_scene_ids(tmp_path):
    with pytest.raises(SystemExit):
        to_r2.main(["--from-tree", "--stage", str(tmp_path)])


def test_commercial_category_dry_run_plan(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(to_r2.urllib.request, "urlopen", boom)
    to_r2.main(["--dry-run", "--category", "commercial", "--stage", str(tmp_path / "s")])
    plan = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert plan["prefix"] == "worlds/grscenes/commercial" and plan["download_bytes"] == 0
    to_r2.main(["--dry-run", "--category", "commercial", "--all", "--stage", str(tmp_path / "s")])
    out = capsys.readouterr().out
    assert json.loads(out.strip().splitlines()[-1])["download_bytes"] == 23792807959 and "commercial_scenes.zip" in out


def test_single_volume_archive_needs_no_join(tmp_path, monkeypatch):
    (tmp_path / "commercial_scenes.zip").write_bytes(b"x")
    monkeypatch.setattr(to_r2.shutil, "which", lambda name: None)
    arc = to_r2.make_readable(tmp_path, {"commercial_scenes.zip": 1})
    assert (arc.kind, arc.path.name) == ("unzip", "commercial_scenes.zip")


def test_members_from_tree_flags_symlinks(tmp_path):
    (tmp_path / "r/scenes/a_usd").mkdir(parents=True)
    (tmp_path / "r/scenes/a_usd/x.usd").write_bytes(b"123")
    (tmp_path / "r/Materials").mkdir()
    (tmp_path / "r/scenes/a_usd/Materials").symlink_to("../../Materials")
    got = {m.name: m for m in to_r2.members_from_tree(tmp_path)}
    assert got["r/scenes/a_usd/x.usd"].size == 3 and not got["r/scenes/a_usd/x.usd"].is_link
    assert got["r/scenes/a_usd/Materials"].is_link and "r/Materials" not in got


def test_merge_into_tree_links_new_scenes_and_shared_dirs(tmp_path):
    """A real target directory (a scene already pulled from R2) must be merged with, not fatal."""
    import fetch_grscenes_to_r2 as f

    tree = tmp_path / "extract" / "home_scenes"
    (tree / "scenes" / "aaa_usd").mkdir(parents=True)
    (tree / "scenes" / "bbb_usd").mkdir()
    (tree / "Materials").mkdir()
    target = tmp_path / "data" / "home"
    (target / "scenes" / "aaa_usd").mkdir(parents=True)  # already present
    (target / "Materials").mkdir()

    assert f.merge_into_tree(tree, target) == 1  # only bbb_usd is new
    assert (target / "scenes" / "bbb_usd").is_symlink()
    assert (target / "scenes" / "aaa_usd").is_dir() and not (target / "scenes" / "aaa_usd").is_symlink()
    assert f.merge_into_tree(tree, target) == 0  # idempotent


def test_link_tree_merges_into_existing_directory(tmp_path, monkeypatch):
    import fetch_grscenes_to_r2 as f

    tree = tmp_path / "extract" / "home_scenes"
    (tree / "scenes" / "ccc_usd").mkdir(parents=True)
    (tree / "Materials").mkdir()
    target = tmp_path / "scenes_root"
    target.mkdir()
    monkeypatch.setattr(f, "log", lambda *a, **k: None)
    out = f.link_tree(tmp_path / "extract", "home_scenes", str(target))
    assert out == tree
    assert (target / "scenes" / "ccc_usd").is_symlink()
