#!/usr/bin/env python3
"""Stop (default) or destroy a Vast.ai instance. Shared headers/User-Agent with creation."""
from __future__ import annotations

import argparse
import os
import sys

from vast_provider import destroy_instance, instance_info, stop_instance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--destroy", action="store_true", help="destroy instead of stop")
    parser.add_argument("--self-test", action="store_true", help="authenticate with the stop headers only")
    args = parser.parse_args(argv)
    key = os.environ.get("VAST_API_KEY", "")
    if not key:
        print("vast_stop: VAST_API_KEY is not set", file=sys.stderr)
        return 2
    if args.self_test:
        status, _ = instance_info(key, args.id)
        print(f"vast_stop: self-test GET instance/{args.id} -> HTTP {status}")
        return 0 if status == 200 else 1
    status, text = (destroy_instance if args.destroy else stop_instance)(key, args.id)
    print(f"vast_stop: {'destroy' if args.destroy else 'stop'} {args.id} -> HTTP {status} {text[:200]}")
    return 0 if status < 400 else 1


if __name__ == "__main__":
    raise SystemExit(main())
