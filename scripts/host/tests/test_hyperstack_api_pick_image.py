"""pick-image must skip images hard-restricted to another flavor family.

Evidence (2026-11-23): the R595 image returned by the old picker is ``hard``-restricted to ``n3-B300-SXM6x8``;
creating an L40 VM from it fails with ``compatibility_blocked`` after the request, so the restriction has to be
honoured before the POST.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

API_PATH = Path(__file__).resolve().parents[1] / "hyperstack_api.py"


def load_api():
    spec = importlib.util.spec_from_file_location("hyperstack_api_under_test", API_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REGION = "CANADA-1"


def images_payload():
    return {"images": [{"region_name": REGION, "images": [
        {"id": 175, "name": "Ubuntu Server 22.04 LTS R535 CUDA 12.2"},
        {"id": 517, "name": "Ubuntu Server 24.04 LTS R570 CUDA 12.8 with Docker"},
        {"id": 1624, "name": "Ubuntu Server 24.04 LTS R595 CUDA 13.2 with Docker",
         "flavor_restrictions": {"has_flavor_restrictions": True, "restriction_type": "hard",
                                 "compatible_flavors": [{"flavor_name": "n3-B300-SXM6x8"}]}},
    ]}]}


def test_restricted_image_is_skipped_for_a_different_flavor(monkeypatch):
    api = load_api()
    monkeypatch.setattr(api, "get", lambda path: images_payload())
    picked = api.pick_image(REGION, "n3-L40x1-bigroot")
    assert picked["id"] == 517


def test_restricted_image_is_allowed_for_its_own_flavor(monkeypatch):
    api = load_api()
    monkeypatch.setattr(api, "get", lambda path: images_payload())
    picked = api.pick_image(REGION, "n3-B300-SXM6x8")
    assert picked["id"] == 1624


def test_without_a_flavor_a_hard_restricted_image_is_skipped(monkeypatch):
    api = load_api()
    monkeypatch.setattr(api, "get", lambda path: images_payload())
    assert api.pick_image(REGION)["id"] == 517


def test_soft_restrictions_do_not_hide_the_newest_driver(monkeypatch):
    api = load_api()
    payload = {"images": [{"region_name": REGION, "images": [
        {"id": 517, "name": "Ubuntu Server 24.04 LTS R570 CUDA 12.8 with Docker"},
        {"id": 1640, "name": "Ubuntu Server 24.04 LTS R595 CUDA 13.2 with Docker",
         "flavor_restrictions": {"has_flavor_restrictions": True, "restriction_type": "soft",
                                 "compatible_flavors": [{"flavor_name": "n3-B300-SXM6x8"}]}},
    ]}]}
    monkeypatch.setattr(api, "get", lambda path: payload)
    assert api.pick_image(REGION, "n3-L40x1-bigroot")["id"] == 1640
