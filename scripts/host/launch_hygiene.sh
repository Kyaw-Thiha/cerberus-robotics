#!/usr/bin/env bash
# Runs ON THE INSTANCE before every relaunch: log leftover Kit/python processes, GPU compute apps, /dev/shm and
# lock files, then clean the remnants of the failed launch.
set -uo pipefail
OUT="${1:-/workspace/cerberus/outputs/launch_hygiene.log}"
mkdir -p "$(dirname "$OUT")"
{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) hygiene"
  echo "-- kit/python processes"
  pgrep -fa "isaac-sim/_isaac_sim/python[.]sh|kit/programs/kit|carb[.]tasking|scene_eval[.]py|nav_eval[.]py" | head -20
  echo "-- gpu compute apps"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -10
  echo "-- /dev/shm"
  df -h /dev/shm 2>/dev/null | tail -1
  ls -la /dev/shm 2>/dev/null | head -10
  echo "-- locks"
  find /tmp /isaac-sim/kit/data -maxdepth 3 \( -name "*.lock" -o -name "kit*.pid" \) 2>/dev/null | head -20
} >> "$OUT" 2>&1
# clean leftovers of the previous (dead) launch; bracket patterns so this script never matches itself
pkill -f "isaac-sim/_isaac_sim/python[.]sh" 2>/dev/null
pkill -f "kit/programs/kit" 2>/dev/null
pkill -f "carb[.]tasking" 2>/dev/null
sleep 2
rm -f /tmp/omni*.lock /tmp/kit*.lock 2>/dev/null
{
  echo "-- after cleanup: gpu compute apps"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -10
} >> "$OUT" 2>&1
exit 0
