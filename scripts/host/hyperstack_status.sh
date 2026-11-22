#!/usr/bin/env bash
# Show this run's Hyperstack VMs, spend estimate, caps and armed kill timers. Never prints a secret.
#
# Usage: scripts/host/hyperstack_status.sh [name|id] [--json]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source=hyperstack_lib.sh
source "$HERE/hyperstack_lib.sh"

TARGET=""; JSON=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --json) JSON=1; shift ;;
        -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
        *) TARGET="$1"; shift ;;
    esac
done

VMS="$(hs_api list-vms 2>/dev/null)" || VMS='{"instances":[]}'
USAGE="$(hs_usage)"
SPEND="$(echo "$USAGE" | cut -d' ' -f1)"; HOURS_USED="$(echo "$USAGE" | cut -d' ' -f2)"
TIMERS="$(systemctl --user list-timers 'cerberus-kill-*' --no-pager 2>/dev/null | grep -c 'cerberus-kill-' || true)"

if [[ $JSON -eq 1 ]]; then
    python3 - "$VMS" "$SPEND" "$HOURS_USED" "$TIMERS" "$TARGET" <<'PY'
import json, sys
vms, spend, hours, timers, target = sys.argv[1:6]
rows = json.loads(vms).get("instances", [])
if target:
    rows = [r for r in rows if str(r.get("id")) == target or r.get("name") == target]
print(json.dumps({"vms": rows, "spend_usd": float(spend), "hours_used": float(hours),
                  "armed_timers": int(timers)}, indent=2))
PY
    exit 0
fi

echo "spend ~\$$SPEND of \$$HS_TOTAL_SPEND; hours $HOURS_USED of $HS_TOTAL_HOURS; creations $(hs_creations) of $HS_MAX_CREATIONS; armed timers $TIMERS"
printf '%-12s %-30s %-10s %-16s %s\n' ID NAME STATUS FLOATING_IP FLAVOR
python3 - "$VMS" "$TARGET" <<'PY'
import json, sys
rows = json.loads(sys.argv[1]).get("instances", [])
if sys.argv[2]:
    rows = [r for r in rows if str(r.get("id")) == sys.argv[2] or r.get("name") == sys.argv[2]]
for r in rows:
    flavor = r.get("flavor") or {}
    print(f"{str(r.get('id','')):<12} {str(r.get('name','')):<30} {str(r.get('status','')):<10} "
          f"{str(r.get('floating_ip') or '-'):<16} {flavor.get('name','')}")
PY
