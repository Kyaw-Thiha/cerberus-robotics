"""Tests for pod creation and the dead-man stop helper (mocked HTTP; no network, no pod)."""

import json

import pytest

import create_pod
import pod_stop
from create_pod import DEFAULT_GPU_TYPES, PodSpec, build_payload, create_pod as create


def test_payload_orders_gpus_and_cuda_first_attempt():
    payload = build_payload(PodSpec())
    assert payload["allowedCudaVersions"] == ["13.0"]
    assert payload["gpuTypeIds"] == list(DEFAULT_GPU_TYPES)
    assert payload["imageName"] == create_pod.DEFAULT_IMAGE
    assert payload["ports"] == ["22/tcp"] and payload["cloudType"] == "SECURE"


def test_payload_always_sets_the_pod_guard_and_image_tag():
    payload = build_payload(PodSpec(extra_env={"FOO": "bar"}))
    assert payload["env"]["CERBERUS_ON_POD"] == "1"
    assert payload["env"]["FOO"] == "bar"
    assert payload["env"]["CERBERUS_IMAGE_TAG"] == payload["imageName"]


def test_payload_omits_the_mount_path_without_a_volume():
    assert "volumeMountPath" not in build_payload(PodSpec(volume_gb=0))
    assert build_payload(PodSpec(volume_gb=50))["volumeMountPath"] == "/workspace"


class Recorder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, body):
        self.calls.append((url, headers, json.loads(body) if body else b""))
        return self.responses.pop(0)


def test_create_returns_the_pod_id_and_never_leaks_the_key():
    transport = Recorder([(200, '{"id": "abc123"}')])
    status, body = create(PodSpec(), "secret-key", transport=transport)
    assert status == 200 and body["id"] == "abc123"
    url, headers, payload = transport.calls[0]
    assert url == create_pod.API_URL
    assert headers["Authorization"] == "Bearer secret-key"
    assert "secret-key" not in json.dumps(payload)


def test_retries_filter_free_after_13_0_fails(monkeypatch, capsys, tmp_path):
    transport = Recorder([(400, '{"error": "no capacity"}'), (200, '{"id": "xyz"}')])
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr(create_pod, "http_post", transport)
    rc = create_pod.main(["--creations-file", str(tmp_path / "c.json")])
    assert rc == 0
    first, second = transport.calls
    assert first[2]["allowedCudaVersions"] == ["13.0"]
    assert "allowedCudaVersions" not in second[2]
    assert "created pod xyz" in capsys.readouterr().out


def test_a_successful_first_attempt_is_the_only_attempt(monkeypatch, tmp_path):
    transport = Recorder([(200, '{"id": "abc"}')])
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr(create_pod, "http_post", transport)
    assert create_pod.main(["--creations-file", str(tmp_path / "c.json")]) == 0
    assert len(transport.calls) == 1


def test_creation_cap_is_four(monkeypatch, tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps([{"pod_id": str(i)} for i in range(4)]))
    transport = Recorder([])
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr(create_pod, "http_post", transport)
    assert create_pod.main(["--creations-file", str(f)]) == 3
    assert transport.calls == []


def test_api_key_is_passed_to_the_pod_but_never_printed(monkeypatch, capsys, tmp_path):
    transport = Recorder([(200, '{"id": "abc"}')])
    monkeypatch.setenv("RUNPOD_API_KEY", "secret-key")
    monkeypatch.setattr(create_pod, "http_post", transport)
    create_pod.main(["--creations-file", str(tmp_path / "c.json")])
    payload = transport.calls[0][2]
    assert payload["env"]["RUNPOD_API_KEY"] == "secret-key"
    assert "secret-key" not in capsys.readouterr().out


def test_cuda_13_2_is_rejected_before_any_request():
    with pytest.raises(SystemExit, match="not accepted by the API"):
        create_pod.validate_cuda_versions(("13.2",))
    payload = build_payload(PodSpec(cuda_versions=()))
    assert "allowedCudaVersions" not in payload


def test_missing_key_is_an_error_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert create_pod.main(["--creations-file", str(tmp_path / "c.json")]) == 2


def test_dry_run_prints_the_body_without_a_key(monkeypatch, capsys):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert create_pod.main(["--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["allowedCudaVersions"] == ["13.0"]


def test_env_flag_is_parsed():
    spec = create_pod.spec_from_args(create_pod.build_parser().parse_args(["--env", "A=1", "--env", "B=2"]))
    assert spec.env()["A"] == "1" and spec.env()["B"] == "2"


def test_bad_env_flag_is_rejected():
    with pytest.raises(SystemExit):
        create_pod.spec_from_args(create_pod.build_parser().parse_args(["--env", "NOVALUE"]))


def test_stop_request_targets_the_pod():
    transport = Recorder([(200, '{"id": "abc", "desiredStatus": "EXITED"}')])
    status, _ = pod_stop.stop_pod("abc", "k", transport=transport)
    assert status == 200
    url, headers, body = transport.calls[0]
    assert url == pod_stop.stop_url("abc")
    assert url.endswith("/pods/abc/stop")
    assert headers["Authorization"] == "Bearer k"
    assert body == b""


def test_deadline_from_now():
    import pod_stop as ps

    assert ps.deadline_from_now(now=1000, hours=3) == 1000 + 3 * 3600
    assert ps.deadline_from_now(now=1000, hours=0.5) == 1000 + 1800


def test_stop_dry_run_needs_no_key(monkeypatch, capsys):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert pod_stop.main(["--pod-id", "abc", "--dry-run"]) == 0
    assert "abc" in capsys.readouterr().out
