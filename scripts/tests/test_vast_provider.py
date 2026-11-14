"""Vast.ai offer filtering, creation, stop and the laptop-side guard — all against a mocked transport."""

import json

import pytest

import vast_provider as vp


def offer(**kw):
    base = {"id": 1, "verified": True, "rentable": True, "reliability2": 0.99, "num_gpus": 1,
            "gpu_ram": 24576, "disk_space": 200.0, "dph_total": 0.55, "driver_version": "580.159.04",
            "gpu_name": "RTX A5000", "direct_port_count": 1}
    base.update(kw)
    return base


@pytest.mark.parametrize("driver,ok", [
    ("580.95.05", True), ("580.159.04", True), ("595.58.03", True),
    ("580.95.04", False), ("595.91.07", False), ("590.10.01", False), ("", False), ("unknown", False),
])
def test_driver_filter(driver, ok):
    assert vp.driver_ok(driver) is ok


def test_good_offer_passes():
    assert vp.offer_ok(offer())[0]


@pytest.mark.parametrize("kw,fragment", [
    ({"verified": False}, "verified"),
    ({"rentable": False}, "rentable"),
    ({"reliability2": 0.95}, "reliability"),
    ({"num_gpus": 2}, "GPU"),
    ({"gpu_ram": 16000}, "VRAM"),
    ({"disk_space": 40.0}, "disk"),
    ({"direct_port_count": 0}, "port"),
    ({"driver_version": "595.91.07"}, "driver"),
])
def test_rejections(kw, fragment):
    ok, why = vp.offer_ok(offer(**kw))
    assert not ok and fragment.lower() in why.lower()


def test_select_sorts_by_price_and_caps():
    cheap = offer(id=2, dph_total=0.4)
    dear = offer(id=3, dph_total=0.9)
    over = offer(id=4, dph_total=1.5)
    assert [o["id"] for o in vp.select_offers([dear, cheap, over])] == [2, 3]


def test_search_parses_offers_and_sends_the_key():
    calls = []

    def transport(url, method, headers, body):
        calls.append((url, method, headers, body))
        return 200, json.dumps({"offers": [offer()]})

    status, offers, note = vp.search_offers("k", transport=transport)
    assert status == 200 and len(offers) == 1 and note == ""
    url, method, headers, body = calls[0]
    assert "/bundles/" in url and method == "GET" and body is None
    assert headers["Authorization"] == "Bearer k"
    assert headers["User-Agent"] == vp.USER_AGENT


def test_search_failure_is_reported():
    status, offers, note = vp.search_offers("k", transport=lambda *a: (503, "busy"))
    assert status == 503 and offers == [] and "busy" in note


def test_create_body_and_headers():
    calls = []

    def transport(url, method, headers, body):
        calls.append((url, method, headers, body))
        return 200, json.dumps({"success": True, "new_contract": 777})

    status, body = vp.create_instance("k", 42, image="img:v1", disk_gb=80,
                                      env={"CERBERUS_ON_POD": "1"}, transport=transport)
    assert status == 200 and body["new_contract"] == 777
    url, method, headers, payload = calls[0]
    sent = json.loads(payload)
    assert url.endswith("/asks/42/") and method == "PUT"
    assert sent["image"] == "img:v1" and sent["disk"] == 80.0 and sent["env"]["CERBERUS_ON_POD"] == "1"
    assert headers["User-Agent"] == vp.USER_AGENT


def test_stop_and_destroy_hit_the_instance_path():
    calls = []

    def transport(url, method, headers, body):
        calls.append((url, method, headers, body))
        return 200, "{}"

    vp.stop_instance("k", "9", transport=transport)
    vp.destroy_instance("k", "9", transport=transport)
    assert calls[0][1] == "PUT" and json.loads(calls[0][3]) == {"state": "stopped"}
    assert calls[1][1] == "DELETE"
    assert calls[0][2]["User-Agent"] == vp.USER_AGENT


def test_instance_info_unwraps_and_handles_lists():
    payload = json.dumps({"instances": [{"ssh_host": "h", "ssh_port": 22, "actual_status": "running"}]})
    status, info = vp.instance_info("k", "1", transport=lambda *a: (200, payload))
    assert status == 200 and info["ssh_host"] == "h"


@pytest.mark.parametrize("elapsed,rate,expect", [
    (0, 0.55, "wait"),
    (89 * 60, 0.55, "wait"),
    (90 * 60, 0.55, "stop"),      # wall clock
    (2 * 3600, 2.0, "stop"),      # USD cap first (4.00 >= 3.00)
])
def test_guard_decision(elapsed, rate, expect):
    d = vp.guard_decision(started_epoch=1000.0, now_epoch=1000.0 + elapsed, rate_per_hour=rate)
    assert d.action == expect


def test_guard_reports_spend_and_remaining():
    d = vp.guard_decision(started_epoch=0, now_epoch=1800, rate_per_hour=0.6)
    assert d.spend_usd == pytest.approx(0.30)
    assert d.remaining_s == 3600
