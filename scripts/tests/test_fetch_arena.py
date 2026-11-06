import json
import subprocess
from pathlib import Path

import fetch_arena
import pytest

FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"
FAKE_LAB_SHA = "fedcba9876543210fedcba9876543210fedcba98"
LS_TREE = f"160000 commit {FAKE_LAB_SHA}\tsubmodules/IsaacLab\n"


@pytest.fixture
def fake_git(tmp_path, monkeypatch):
    """Redirect the destination into tmp_path and mock subprocess.run (no network)."""
    dest = tmp_path / "third_party" / "IsaacLab-Arena"
    monkeypatch.setattr(fetch_arena, "THIRD_PARTY_ARENA", dest)
    calls = []
    kwargs_seen = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        kwargs_seen.append(kwargs)
        if cmd[:2] == ["git", "clone"]:
            (Path(cmd[-1]) / "isaaclab_arena").mkdir(parents=True)
        stdout = ""
        if "rev-parse" in cmd:
            stdout = FAKE_SHA + "\n"
        elif "ls-tree" in cmd:
            stdout = LS_TREE
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(fetch_arena.subprocess, "run", fake_run)
    return dest, calls, kwargs_seen


def test_fetch_is_shallow_without_submodules_by_default(fake_git):
    dest, calls, _ = fake_git
    assert fetch_arena.fetch() is True
    assert calls[0] == [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        "main",
        "https://github.com/isaac-sim/IsaacLab-Arena",
        str(dest),
    ]
    assert calls[1] == ["git", "-C", str(dest), "rev-parse", "HEAD"]
    assert calls[2] == [
        "git",
        "-C",
        str(dest),
        "ls-tree",
        "HEAD",
        "submodules/IsaacLab",
    ]
    assert len(calls) == 3
    assert not any("submodule" in c for c in calls)


def test_fetch_disables_lfs_smudge(fake_git):
    _, _, kwargs_seen = fake_git
    fetch_arena.fetch()
    assert kwargs_seen[0]["env"]["GIT_LFS_SKIP_SMUDGE"] == "1"


def test_fetch_honours_ref_override(fake_git):
    _, calls, _ = fake_git
    fetch_arena.fetch(ref="release/0.3.0")
    assert calls[0][calls[0].index("--branch") + 1] == "release/0.3.0"


def test_fetch_with_submodules_uses_https_rewrite_and_skips_gr00t(fake_git):
    dest, calls, _ = fake_git
    fetch_arena.fetch(submodules=True)
    assert calls[1] == [
        "git",
        "-c",
        "url.https://github.com/.insteadOf=git@github.com:",
        "-C",
        str(dest),
        "submodule",
        "update",
        "--init",
        "--depth",
        "1",
        "--",
        "submodules/IsaacLab",
    ]
    assert not any("GR00T" in " ".join(c) for c in calls)
    assert json.loads((dest / ".cerberus_fetch.json").read_text())["submodules"] is True


def test_fetch_writes_metadata_with_both_shas(fake_git):
    dest, _, _ = fake_git
    fetch_arena.fetch()
    metadata = json.loads((dest / ".cerberus_fetch.json").read_text())
    assert metadata["commit"] == FAKE_SHA
    assert metadata["isaaclab_commit"] == FAKE_LAB_SHA
    assert metadata["ref"] == "main"
    assert metadata["submodules"] is False
    assert metadata["license"] == "Apache-2.0"
    assert metadata["url"] == "https://github.com/isaac-sim/IsaacLab-Arena"


def test_is_fetched_and_arena_path(fake_git):
    dest, _, _ = fake_git
    assert fetch_arena.is_fetched() is False
    with pytest.raises(FileNotFoundError, match="fetch_arena.py"):
        fetch_arena.arena_path()
    fetch_arena.fetch()
    assert fetch_arena.is_fetched() is True
    assert fetch_arena.arena_path() == dest


def test_second_fetch_is_a_noop(fake_git):
    _, calls, _ = fake_git
    assert fetch_arena.fetch() is True
    n_calls = len(calls)
    assert fetch_arena.fetch() is False
    assert len(calls) == n_calls


def test_second_fetch_with_submodules_only_fetches_submodules(fake_git):
    dest, calls, _ = fake_git
    fetch_arena.fetch()
    n_calls = len(calls)
    assert fetch_arena.fetch(submodules=True) is True
    assert len(calls) == n_calls + 1
    assert "submodule" in calls[n_calls]
    assert json.loads((dest / ".cerberus_fetch.json").read_text())["submodules"] is True
    assert fetch_arena.fetch(submodules=True) is False  # now complete


def test_force_refetches(fake_git):
    _, calls, _ = fake_git
    fetch_arena.fetch()
    n_calls = len(calls)
    assert fetch_arena.fetch(force=True) is True
    assert len(calls) == n_calls + 3
    assert calls[n_calls][:2] == ["git", "clone"]


def test_parse_ls_tree_sha():
    assert fetch_arena.parse_ls_tree_sha(LS_TREE) == FAKE_LAB_SHA
    assert fetch_arena.parse_ls_tree_sha("") is None
    # a plain file/blob entry at the same path is not a submodule
    assert (
        fetch_arena.parse_ls_tree_sha(
            f"100644 blob {FAKE_LAB_SHA}\tsubmodules/IsaacLab\n"
        )
        is None
    )
    # a different submodule is not matched
    assert (
        fetch_arena.parse_ls_tree_sha(
            f"160000 commit {FAKE_LAB_SHA}\tsubmodules/Isaac-GR00T\n"
        )
        is None
    )
