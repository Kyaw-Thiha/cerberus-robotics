#!/usr/bin/env python3
"""Scan text artifacts for secret-looking values; mask them in place and report counts only.

Patterns cover the shapes this project's providers emit: ``R2_SECRET_ACCESS_KEY=…``, ``"RUNPOD_API_KEY":"rpa_…"``,
bearer tokens and long hex keys. Nothing is printed beyond counts and masked excerpts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from redact import mask_text  # noqa: E402

PATTERNS = {
    "r2_secret": re.compile(r"(?i)R2_(?:SECRET_ACCESS_KEY|ACCESS_KEY_ID)\s*[=:]\s*[\"']?[A-Za-z0-9/+=]{16,}"),
    "runpod_key": re.compile(r"rpa_[A-Za-z0-9]{12,}"),
    "bearer": re.compile(r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._\-]{16,}"),
    # Deliberately no bare hex rule: 40-hex git SHAs and sha256 digests are not secrets and masking them
    # corrupts documents. Secret shapes are covered by the key-named assignment/JSON patterns in redact.mask_text.
    "key_assignment": re.compile(r"(?i)\b[A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASS)[A-Z0-9_]*\s*[=:]\s*[\"']?[A-Za-z0-9/+=_\-]{16,}"),
}
TEXT_SUFFIXES = {".json", ".log", ".txt", ".csv", ".md", ".yaml", ".yml", ".jsonl"}


def scan_file(path: Path) -> dict[str, int]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    hits = {name: len(rx.findall(text)) for name, rx in PATTERNS.items()}
    if any(hits.values()):
        path.write_text(mask_text(text))
    return {k: v for k, v in hits.items() if v}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    total, files = {}, {}
    for root in args.paths:
        base = Path(root)
        for path in ([base] if base.is_file() else base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            hits = scan_file(path)
            if hits:
                files[str(path)] = hits
                for k, v in hits.items():
                    total[k] = total.get(k, 0) + v
    report = {"files": files, "totals": total}
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"files_with_hits": len(files), "totals": total}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
