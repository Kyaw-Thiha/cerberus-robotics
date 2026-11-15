#!/usr/bin/env bash
# Build the lite scene from the preprocessed cache on the same CPU pod and upload it to R2.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${1:?root}"
SID_RAW="${CERBERUS_LITE_SCENE:-MV7J6NIKTKJZ2AABAAAAADQ8_usd}"
cd "$REPO_ROOT"
SID=$(python3 -c "import sys; sys.path.insert(0,'scripts'); from scene_id import folder_name; print(folder_name('$SID_RAW'))")
echo "[lite] scene $SID"

python3 -m pip install -q --no-input usd-core numpy >/dev/null 2>&1 || { echo "[lite] pip install failed" >&2; exit 2; }

df -h /workspace | tail -1
FREE_GB=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
MIN_GB="${CERBERUS_LITE_MIN_FREE_GB:-28}"
[ "${FREE_GB:-0}" -ge "$MIN_GB" ] || { echo "[lite] disk below ${MIN_GB} GB" >&2; exit 3; }

SCENE_USD=$(ls data/scenes/grscenes/home/scenes/"$SID"/start_result_*.usd 2>/dev/null | head -1)
if [ -z "$SCENE_USD" ]; then
    echo "[lite] fetching the preprocessed cache (~27 GB)"
    timeout 3600 python3 scripts/fetch_grscenes_from_r2.py --scene-id "$SID" || { echo "[lite] fetch failed" >&2; exit 4; }
    SCENE_USD=$(ls data/scenes/grscenes/home/scenes/"$SID"/start_result_*.usd 2>/dev/null | head -1)
fi
[ -n "$SCENE_USD" ] || { echo "[lite] no scene USD" >&2; exit 5; }
echo "[lite] source $SCENE_USD $(du -h "$SCENE_USD" | cut -f1)"

OUT="data/scenes/grscenes_lite/$SID"
python3 scripts/build_lite_scene.py --scene-usd "$SCENE_USD" --out-dir "$OUT" --scene-id "$SID" || exit 6

rclone mkdir "$REMOTE/worlds/grscenes_lite/$SID" 2>/dev/null || true
rclone copy "$OUT" "$REMOTE/worlds/grscenes_lite/$SID" --no-check-dest || { echo "[lite] upload failed" >&2; exit 7; }
cp "$OUT/lite_report.json" "$ROOT/lite_report.json"
python3 - "$ROOT/lite_report.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("[lite] bytes=%s triangles=%s iou=%s" % (d["lite_bytes"], d["triangles"], d["iou_blocked"]))
raise SystemExit(0 if d["lite_bytes"] < 100_000_000 and d["iou_blocked"] >= 0.98 else 1)
PY
