"""Run the repository's push-recovery sweep against the Newton task, unchanged, and record what happens.

`policy/locomotion/eval/push_recovery_eval.py` is imported and executed as-is (``__main__``), after this
spike's Newton task is registered, so the script's own CLI, sweep grid, contract check and CSV output are the
ones under test -- nothing about it is patched. If it cannot run kit-less, the error is captured verbatim.

    python -m spikes.newton.run_push_recovery --task <id> --platform go2 --terrain rough --headless \
        --checkpoint <ckpt.pt> --magnitudes 0,80 --terrain_levels 0,4 --trials_per_cell 20 --output <csv>
"""

from __future__ import annotations

import json
import runpy
import sys
import traceback
from pathlib import Path

from spikes.newton import common, tasks

SCRIPT = "policy/locomotion/eval/push_recovery_eval.py"


def main() -> int:
    args = sys.argv[1:]
    out = Path("/workspace/newton_out/push_sweep")
    for i, token in enumerate(args):
        if token == "--output" and i + 1 < len(args):
            out = Path(args[i + 1]).parent
    out.mkdir(parents=True, exist_ok=True)

    tasks.register()
    verdict = {"script": SCRIPT, "argv": args, "ran_unchanged": False}
    sys.argv = [SCRIPT, *args]
    try:
        runpy.run_path(str(common.CODE_ROOT / SCRIPT), run_name="__main__")
        verdict["ran_unchanged"] = True
    except SystemExit as exc:
        verdict["exit_code"] = exc.code
        verdict["ran_unchanged"] = exc.code in (0, None)
        verdict["error"] = None
    except Exception:  # noqa: BLE001 - the point is to record why it cannot run
        verdict["error"] = traceback.format_exc()
        verdict["error_class"] = traceback.format_exc().strip().splitlines()[-1]
    (out / "push_recovery_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"[push] ran_unchanged={verdict['ran_unchanged']} error={verdict.get('error_class')}", flush=True)
    if verdict.get("error"):
        print(verdict["error"][-2000:], flush=True)
    return 0 if verdict["ran_unchanged"] else 1


if __name__ == "__main__":
    from spikes.newton.common import run_stage

    raise SystemExit(run_stage(main))
