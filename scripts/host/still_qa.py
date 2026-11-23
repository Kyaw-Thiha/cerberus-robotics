#!/usr/bin/env python3
"""Apply the clip-QA still checks to PNG stills (the same numbers the gate uses on video frames).

Usage::

    python3 scripts/host/still_qa.py stills_dir/*.png --json stills_qa.json

Exit 0 when every still passes, 1 when at least one fails. Prints one line per still with the three measured
quantities, so a failure says which check failed instead of "the picture looks wrong".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clip_qa import still_checks


def check_image(path: Path) -> dict:
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    return {"path": str(path), **still_checks(rgb)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("images", nargs="+")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    results = [check_image(Path(p)) for p in args.images]
    for r in results:
        print(f"{Path(r['path']).name}: mean={r['mean_luminance']:.1f} clipped={r['clipped_fraction']:.3f} "
              f"hue_bins={r['hue_bins_over_5pct']} {'PASS' if r['passed'] else 'FAIL'}")
    failed = [r for r in results if not r["passed"]]
    if args.json:
        Path(args.json).write_text(json.dumps({"passed": not failed, "results": results}, indent=2) + "\n")
    print(f"{len(results) - len(failed)}/{len(results)} passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
