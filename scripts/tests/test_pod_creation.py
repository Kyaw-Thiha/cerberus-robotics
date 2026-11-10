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


def test_fallback_retries_with_cuda_13_2(monkeypatch, capsys):
    transport = Recorder([(400, '{"error": "no capacity"}'), (200, '{"id": "xyz"}')])
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr(create_pod, "http_post", transport)
    rc = create_pod.main(["--fallback"])
    assert rc == 0
    assert [call[2]["allowedCudaVersions"] for call in transport.calls] == [["13.0"], ["13.2"]]
    assert "created pod xyz" in capsys.readouterr().out


def test_no_fallback_means_one_attempt(monkeypatch):
    transport = Recorder([(400, '{"error": "no capacity"}')])
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr(create_pod, "http_post", transport)
    assert create_pod.main([]) == 1
    assert len(transport.calls) == 1


def test_missing_key_is_an_error_not_a_crash(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert create_pod.main([]) == 2


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
