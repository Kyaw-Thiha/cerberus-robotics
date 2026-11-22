#!/usr/bin/env python3
"""Scene-id normalisation used everywhere a scene folder is named.

GRScenes folders end in ``_usd``; callers pass the id with or without that suffix (``tiny``, ``tiny_usd``,
``MV7J6NIKTKJZ2AABAAAAADQ8_usd``). Normalising to the folder name in one place is what keeps the fetch, the check and
the scene reference from disagreeing (the H4 run aborted on a doubled ``_usd``).
"""
from __future__ import annotations


def folder_name(scene_id: str) -> str:
    """``scene_id`` -> the USD folder name, with exactly one ``_usd`` suffix."""
    sid = scene_id.strip().strip("/")
    return sid if sid.endswith("_usd") else sid + "_usd"


def bare_id(scene_id: str) -> str:
    """``scene_id`` without the ``_usd`` suffix."""
    sid = folder_name(scene_id)
    return sid.removesuffix("_usd")
