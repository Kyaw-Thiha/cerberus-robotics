"""Redaction masks secret-looking values and nothing else."""

import json

import redact


def test_masks_key_secret_token_pass_case_insensitively():
    data = {"R2_ACCESS_KEY_ID": "abcd1234567890wxyz", "R2_SECRET_ACCESS_KEY": "supersecretvalue123",
            "VAST_API_KEY": "keykeykeykey12345", "HF_TOKEN": "hf_abcdefghijkl", "PASSWORD": "hunter2hunter2",
            "R2_ACCOUNT_ID": "account-is-not-secret", "image": "img:v1"}
    out = redact.redact(data)
    assert out["R2_ACCESS_KEY_ID"] == "abcd…wxyz"
    assert out["R2_SECRET_ACCESS_KEY"].startswith("supe") and out["R2_SECRET_ACCESS_KEY"].endswith("e123")
    assert out["VAST_API_KEY"] == "keyk…2345"
    assert out["HF_TOKEN"] == "hf_a…ijkl"
    assert out["PASSWORD"] == "hunt…ter2"
    assert out["R2_ACCOUNT_ID"] == "account-is-not-secret" and out["image"] == "img:v1"


def test_short_values_are_fully_masked():
    assert redact.redact({"token": "short"})["token"] == "***"


def test_nested_and_masked_json_contains_no_secret():
    payload = {"env": {"R2_SECRET_ACCESS_KEY": "topsecretvalue"}, "args": ["bash"], "n": 1}
    text = redact.masked_json(payload)
    assert "topsecretvalue" not in text
    parsed = json.loads(text)
    assert parsed["env"]["R2_SECRET_ACCESS_KEY"] == "tops…alue"
    assert parsed["args"] == ["bash"] and parsed["n"] == 1
