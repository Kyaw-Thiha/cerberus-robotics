#!/bin/bash
# K2 A/B: N launches of the minimal probe under one variant, counting starts and completions.
# usage: kit_ab_probe.sh <variant> <launches> <out.csv>
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-/workspace/cerberus-robotics}"
VARIANT="$1"; N="${2:-10}"; OUT="${3:-/home/ubuntu/cerberus-results/k2_${VARIANT}.csv}"
LAB="${ISAACLAB_PATH:-/workspace/isaaclab}/isaaclab.sh"
echo "variant,launch,exit,started,done,seconds" > "$OUT"
starts=0; dones=0
for i in $(seq 1 "$N"); do
  LOG="/tmp/k2_${VARIANT}_${i}.log"
  start=$(date +%s)
  case "$VARIANT" in
    v0) timeout 120 "$LAB" -p "$REPO_ROOT/scripts/host/kit_min_probe.py" --seconds 5 > "$LOG" 2>&1 ;;
    v1) timeout 120 taskset -c 0-7 "$LAB" -p "$REPO_ROOT/scripts/host/kit_min_probe.py" --seconds 5 > "$LOG" 2>&1 ;;
    v2) PXR_WORK_THREAD_LIMIT=1 OMP_NUM_THREADS=1 timeout 120 "$LAB" -p "$REPO_ROOT/scripts/host/kit_min_probe.py" --seconds 5 > "$LOG" 2>&1 ;;
    v4) timeout 120 "$LAB" -p "$REPO_ROOT/scripts/host/kit_min_probe.py" --seconds 5 --disable_cameras > "$LOG" 2>&1 ;;
    *) echo "unknown variant $VARIANT" >&2; exit 2 ;;
  esac
  rc=$?
  secs=$(( $(date +%s) - start ))
  st=0; dn=0
  grep -aq "PROBE STARTED" "$LOG" && st=1
  grep -aq "PROBE DONE" "$LOG" && dn=1
  starts=$((starts + st)); dones=$((dones + dn))
  echo "$VARIANT,$i,$rc,$st,$dn,$secs" >> "$OUT"
  echo "[k2 $VARIANT] $i/$N exit=$rc started=$st done=$dn ${secs}s" >&2
done
echo "[k2 $VARIANT] starts $starts/$N completes $dones/$N -> $OUT" >&2
