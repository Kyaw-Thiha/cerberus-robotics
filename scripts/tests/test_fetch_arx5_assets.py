import json
import subprocess
import sys
from pathlib import Path

import fetch_arx5_assets
import pytest

FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def fake_git(tmp_path, monkeypatch):
    """Redirect the destination into tmp_path and mock subprocess.run (no network)."""
    dest = tmp_path / "third_party" / "arx5-sdk"
    model_dest = tmp_path / "third_party" / "arx-model"
    monkeypatch.setattr(fetch_arx5_assets, "THIRD_PARTY_ARX5", dest)
    monkeypatch.setattr(fetch_arx5_assets, "THIRD_PARTY_ARX_MODEL", model_dest)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["git", "clone"]:
            target = Path(cmd[-1])
            if target == dest:
                (dest / "models").mkdir(parents=True)
                (dest / "models" / "X5.urdf").write_text("<robot/>")
            else:
                (target / "X5" / "X5A" / "urdf").mkdir(parents=True)
                (target / "X5" / "X5A" / "urdf" / "X5A.urdf").write_text("<robot/>")
        stdout = FAKE_SHA + "\n" if "rev-parse" in cmd else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(fetch_arx5_assets.subprocess, "run", fake_run)
    return dest, calls


def test_fetch_issues_sparse_shallow_filtered_git_commands_in_order(fake_git):
    dest, calls = fake_git
    assert fetch_arx5_assets.fetch(ref="main") is True
    assert calls[0] == [
        "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
        "--branch", "main", "https://github.com/real-stanford/arx5-sdk", str(dest),
    ]
    assert calls[1] == ["git", "-C", str(dest), "sparse-checkout", "set", "models"]
    assert calls[2] == ["git", "-C", str(dest), "rev-parse", "HEAD"]
    assert len(calls) == 3


def test_fetch_honours_ref_override(fake_git):
    _, calls = fake_git
    fetch_arx5_assets.fetch(ref="v1.2.3")
    assert calls[0][calls[0].index("--branch") + 1] == "v1.2.3"


def test_fetch_writes_metadata_with_sha_and_license(fake_git):
    dest, _ = fake_git
    fetch_arx5_assets.fetch(ref="main")
    metadata = json.loads((dest / ".cerberus_fetch.json").read_text())
    assert metadata["commit"] == FAKE_SHA
    assert metadata["ref"] == "main"
    assert metadata["license"] == "MIT, Copyright (c) 2024 Yihuai Gao"
    assert metadata["url"] == "https://github.com/real-stanford/arx5-sdk"


def test_second_fetch_is_a_noop(fake_git):
    _, calls = fake_git
    assert fetch_arx5_assets.fetch() is True
    n_calls = len(calls)
    assert fetch_arx5_assets.fetch() is False
    assert len(calls) == n_calls


def test_force_refetches(fake_git):
    dest, calls = fake_git
    fetch_arx5_assets.fetch()
    n_calls = len(calls)
    assert fetch_arx5_assets.fetch(force=True) is True
    assert len(calls) == n_calls + 3
    assert calls[n_calls][:2] == ["git", "clone"]
    assert (dest / ".cerberus_fetch.json").is_file()


def test_arm_urdf_path_happy_path(fake_git):
    dest, _ = fake_git
    fetch_arx5_assets.fetch()
    assert fetch_arx5_assets.arm_urdf_path("X5") == dest / "models" / "X5.urdf"


def test_arm_urdf_path_missing_raises_with_helpful_message(fake_git):
    with pytest.raises(FileNotFoundError, match="fetch_arx5_assets.py"):
        fetch_arx5_assets.arm_urdf_path("L5")


def test_arm_urdf_path_invalid_variant_raises(fake_git):
    with pytest.raises(ValueError, match="X7"):
        fetch_arx5_assets.arm_urdf_path("X7")


def test_variants_tuple_and_default():
    assert fetch_arx5_assets.VARIANTS == ("X5A", "X5", "L5", "X5_no_gripper", "X5_umi")
    assert fetch_arx5_assets.DEFAULT_VARIANT == "X5A"


def test_variant_source_table():
    assert fetch_arx5_assets.variant_source("X5A") == "arx-model"
    for v in ("X5", "L5", "X5_no_gripper", "X5_umi"):
        assert fetch_arx5_assets.variant_source(v) == "arx5-sdk"
    with pytest.raises(ValueError):
        fetch_arx5_assets.variant_source("X7")


def test_arx_model_fetch_issues_sparse_commands_and_records_sha(fake_git):
    _, calls = fake_git
    model_dest = fetch_arx5_assets.THIRD_PARTY_ARX_MODEL
    assert fetch_arx5_assets.fetch_arx_model() is True
    assert calls[0] == [
        "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
        "--branch", "master", "https://github.com/ARXroboticsX/ARX_Model", str(model_dest),
    ]
    assert calls[1] == ["git", "-C", str(model_dest), "sparse-checkout", "set", "X5/X5A"]
    assert calls[2] == ["git", "-C", str(model_dest), "rev-parse", "HEAD"]
    metadata = json.loads((model_dest / ".cerberus_fetch.json").read_text())
    assert metadata["commit"] == FAKE_SHA
    assert metadata["ref"] == "master"
    assert metadata["url"] == "https://github.com/ARXroboticsX/ARX_Model"
    assert "no LICENSE" in metadata["license"]
    # second call is a no-op
    assert fetch_arx5_assets.fetch_arx_model() is False
    assert len(calls) == 3


def test_arm_urdf_path_x5a(fake_git):
    model_dest = fetch_arx5_assets.THIRD_PARTY_ARX_MODEL
    with pytest.raises(FileNotFoundError, match="fetch_arx5_assets.py"):
        fetch_arx5_assets.arm_urdf_path("X5A")
    fetch_arx5_assets.fetch_arx_model()
    expected = model_dest / "X5" / "X5A" / "urdf" / "X5A.urdf"
    assert fetch_arx5_assets.arm_urdf_path("X5A") == expected
    assert fetch_arx5_assets.arm_urdf_path() == expected  # X5A is the default


def test_sdk_and_arx_model_fetches_are_independent(fake_git):
    fetch_arx5_assets.fetch()
    assert fetch_arx5_assets.is_fetched("arx5-sdk") and not fetch_arx5_assets.is_fetched("arx-model")
    fetch_arx5_assets.fetch_arx_model()
    assert fetch_arx5_assets.is_fetched("arx-model")


def test_cli_rejects_unknown_variant():
    result = subprocess.run([sys.executable, "scripts/fetch_arx5_assets.py", "--variant", "bogus"], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "bogus" in result.stderr
