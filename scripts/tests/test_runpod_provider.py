"""RunPod job-mode payload: no SSH, entrypoint/start-cmd override, restricted env, CUDA fallback."""

import base64


import runpod_provider as rp


def test_job_payload_has_no_ssh_and_overrides_start():
    calls = []

    def transport(method, url, key, body=None):
        calls.append((method, url, body))
        return 200, '{"id": "pod123"}'

    status, body = rp.create_job_pod("key", image="img:v1", gpu_types=["NVIDIA A40"],
                                     env={"A": "1"}, bootstrap_b64="ZWNobyBoaQ==", transport=transport)
    assert status == 200 and body["id"] == "pod123"
    import json
    first = json.loads(calls[0][2])
    assert first["ports"] == []
    assert first["dockerEntrypoint"] == ["/bin/bash", "-c"]
    assert first["dockerStartCmd"] == [rp.SHORT_BOOTSTRAP]
    assert first["env"]["CERBERUS_BOOTSTRAP_B64"] == "ZWNobyBoaQ=="
    assert first["env"]["A"] == "1"
    assert first["allowedCudaVersions"] == ["13.0"]
    decoded = base64.b64decode(first["env"]["CERBERUS_BOOTSTRAP_B64"]).decode()
    assert decoded == "echo hi"


def test_job_payload_falls_back_filter_free_once():
    seen = []

    def transport(method, url, key, body=None):
        import json
        payload = json.loads(body)
        seen.append(payload)
        return (400, '{"error":"no stock"}') if "allowedCudaVersions" in payload else (200, '{"id":"p2"}')

    status, body = rp.create_job_pod("k", image="i", gpu_types=["NVIDIA A40"], env={},
                                     bootstrap_b64="eA==", transport=transport)
    assert status == 200 and body["id"] == "p2"
    assert len(seen) == 2 and "allowedCudaVersions" not in seen[1]


def test_stop_uses_post_and_the_laptop_headers():
    calls = []

    def transport(method, url, key, body=None):
        calls.append((method, url, key, body))
        return 200, "{}"

    status, _ = rp.stop_pod("secret-key", "pod123", transport=transport)
    assert status == 200
    method, url, key, payload = calls[0]
    assert method == "POST" and url.endswith("/pods/pod123/stop") and key == "secret-key"
    assert payload == b"{}"


def test_job_env_is_the_restricted_set(monkeypatch):
    for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.setenv(k, f"v-{k}")
    monkeypatch.setenv("RUNPOD_API_KEY", "api-secret")
    monkeypatch.setenv("WANDB_KEY", "nope")
    env = rp.job_env("migration/x/code/b.bundle", "job1", 90, "img:v1")
    assert set(env) == {"R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET",
                        "CERBERUS_CODE_BUNDLE", "CERBERUS_JOB_ID", "CERBERUS_DEADLINE_MINUTES",
                        "CERBERUS_IMAGE_TAG", "RUNPOD_API_KEY"}
    assert "nope" not in ",".join(env.values())
    assert env["CERBERUS_CODE_BUNDLE"].endswith("b.bundle")
