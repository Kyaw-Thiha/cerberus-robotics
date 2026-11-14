#!/usr/bin/env python3
"""Mask secret-looking values before anything is logged.

Any key matching ``KEY|SECRET|TOKEN|PASS`` (case-insensitive) has its value replaced by a masked form that keeps only
a 4-character prefix and suffix. Used everywhere a request body, env dict or instance config is logged.
"""
from __future__ import annotations

import re
from typing import Any

SECRET_KEY = re.compile(r"KEY|SECRET|TOKEN|PASS", re.IGNORECASE)


def mask_value(value: Any) -> str:
    text = str(value)
    if len(text) <= 10:
        return "***"
    return f"{text[:4]}…{text[-4:]}"


def redact(data: Any) -> Any:
    """Recursively mask values whose key looks secret; leaves other values untouched."""
    if isinstance(data, dict):
        return {k: (mask_value(v) if isinstance(k, str) and SECRET_KEY.search(k) else redact(v))
                for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [redact(v) for v in data]
    return data


def masked_json(data: Any) -> str:
    import json

    return json.dumps(redact(data), default=str)
