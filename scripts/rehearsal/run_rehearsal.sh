#!/usr/bin/env bash
# Rehearsal flow: the same stages as the physics job, with Kit and the driver replaced by stubs.
# Called by job_entrypoint.sh when CERBERUS_REHEARSAL=1. Env from the entrypoint: ROOT, REMOTE, JOB_ID, REPO.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${1:?root dir}"
REMOTE="${REMOTE:-cerberus_r2:${R2_BUCKET:?}}"
export REMOTE ROOT
mkdir -p "$ROOT"
cd "$REPO_ROOT"

stage() { echo "[rehearsal] $1"; }

# 1. Stub the image check: a fake isaaclab.sh that prints the startup marker and delegates to the stub app.
LAB_PATH="${ISAACLAB_PATH:-/workspace/rehearsal/lab}"
export ISAACLAB_PATH="$LAB_PATH"
mkdir -p "$LAB_PATH"
cp scripts/rehearsal/stub_lab.sh "$LAB_PATH/isaaclab.sh"
chmod +x "$LAB_PATH/isaaclab.sh"
LAB="$LAB_PATH/isaaclab.sh"
export CERBERUS_REHEARSAL_STUB="$REPO_ROOT/scripts/rehearsal/stub_app.py"
[ -f "$LAB" ] && stage "image check: stub lab present at $LAB"

# 2. Scene-id normalisation across every form, recorded for the report.
python3 - "$ROOT/scene_ids.json" <<'PY'
import json, sys
sys.path.insert(0, "scripts")
from scene_id import bare_id, folder_name
cases = ["tiny", "tiny_usd", "MV7J6NIKTKJZ2AABAAAAADQ8", "MV7J6NIKTKJZ2AABAAAAADQ8_usd", "/tiny_usd/"]
rows = [{"in": c, "folder": folder_name(c), "bare": bare_id(c)} for c in cases]
json.dump(rows, open(sys.argv[1], "w"), indent=2)
assert all(r["folder"] == r["bare"] + "_usd" for r in rows), rows
print("[rehearsal] scene-id forms:", [r["folder"] for r in rows])
PY

# 3. Disk check, then the tiny scene from R2 (same layout: scenes/<id>_usd, Materials, models).
df -h /workspace | tail -1
FREE_GB=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
stage "free container disk ${FREE_GB}G"
MIN_GB="${CERBERUS_REHEARSAL_MIN_FREE_GB:-10}"
[ "${FREE_GB:-0}" -ge "$MIN_GB" ] || { echo "[rehearsal] disk below ${MIN_GB} GB" >&2; exit 11; }
mkdir -p data/scenes/grscenes/home
rclone copy "$REMOTE/worlds/rehearsal" data/scenes/grscenes/home --no-check-dest >/dev/null
TINY="data/scenes/grscenes/home/scenes/$(python3 -c 'import sys; sys.path.insert(0,"scripts"); from scene_id import folder_name; print(folder_name("tiny"))')"
[ -f "$TINY/start_result_tiny.usd" ] || { echo "[rehearsal] tiny scene missing at $TINY" >&2; exit 12; }
stage "fetched tiny scene: $(du -sh "$TINY" | cut -f1)"

# 4. Gate with a stubbed driver string.
python3 - "$ROOT" <<'PY'
import sys
sys.path.insert(0, "scripts")
from pod_driver_gate import HostInfo, append_probe_row, CSV_PATH
info = HostInfo(driver="580.95.05", cuda="13.0", gpu="rehearsal", host="rehearsal-host",
                datacenter="local", image="rehearsal")
append_probe_row(CSV_PATH, info, stage="gate", verdict="ALLOW", tier="A", note="rehearsal stub driver")
print("[rehearsal] DRIVER GATE: TIER A (stub) driver=580.95.05")
PY

# 5. Launch gate through the same retry path (the stub prints Startup Complete after 2 s).
timeout 300 python3 scripts/launch_gate.py --max-consecutive-failures 10 --cap-minutes 5 -- \
  "$LAB" -p scripts/scene_eval.py --scene "grscenes:tiny" --num_envs 1 --steps 200 \
  --out-dir "$ROOT/stand_tiny" --headless || { echo "[rehearsal] launch gate failed" >&2; exit 13; }

# 6. Stand, scripted and navigation stubs, each through run_resilient + kit_retry like the real plan.
bash scripts/run_resilient.sh --name stand --max-restarts 5 -- \
  bash scripts/kit_retry.sh -- "$LAB" -p scripts/scene_eval.py --scene "grscenes:tiny" --num_envs 1 --steps 200 \
  --out-dir "$ROOT/stand_tiny" --headless || exit 14
python3 - "$ROOT/stand_tiny/assertions.json" <<'PY' || exit 15
import json, sys
d = json.load(open(sys.argv[1]))
print("[rehearsal] assertions_passed", d.get("passed"))
raise SystemExit(0 if d.get("passed") else 1)
PY
bash scripts/run_resilient.sh --name scripted --max-restarts 5 -- \
  bash scripts/kit_retry.sh -- "$LAB" -p scripts/scene_eval.py --scene "grscenes:tiny" --num_envs 1 --steps 200 \
  --scripted --out-dir "$ROOT/scripted_tiny" --headless || exit 16
bash scripts/run_resilient.sh --name nav --max-restarts 5 -- \
  bash scripts/kit_retry.sh -- "$LAB" -p scripts/nav_eval.py --scenes "grscenes:tiny" --pairs 2 --seed 0 --num_envs 1 \
  --min-distance 3 --max-distance 10 --max-time 120 --out-dir "$ROOT/nav_tiny" --headless || exit 17

# 7. Manifest check: upload the rehearsal root, then compare remote vs local listings.
rclone copy "$ROOT" "$REMOTE/jobs/$JOB_ID/rehearsal" --no-check-dest >/dev/null
python3 - "$ROOT" "$JOB_ID" <<'PY'
import json, os, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1]); job = sys.argv[2]
remote_root = f"cerberus_r2:{os.environ['R2_BUCKET']}/jobs/{job}/rehearsal"
listing = subprocess.run(["rclone", "lsjson", "--recursive", remote_root], capture_output=True, text=True)
remote = sorted(e["Path"] for e in json.loads(listing.stdout or "[]") if not e.get("IsDir"))
local = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
missing = sorted(set(local) - set(remote))
print("[rehearsal] manifest local=%d remote=%d missing=%s" % (len(local), len(remote), missing))
json.dump({"local": local, "remote": remote, "missing": missing},
          open(root / "manifest_check.json", "w"), indent=2)
raise SystemExit(0 if not missing else 1)
PY
[ $? -eq 0 ] || exit 18
stage "rehearsal complete"
