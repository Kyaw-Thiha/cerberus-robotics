#!/usr/bin/env bash
# Shared guards for the Brev wrappers. Sourced, never executed.
#
# The wrappers exist so a paid instance cannot outlive its budget or the operator's attention:
#   * price cap     1.50 USD/hour, checked before create (the price is passed in and must match the catalog)
#   * wall clock    3.5 hours, enforced by a systemd user timer armed BEFORE create and cancelled by brev_down.sh
#   * creations     4 per run, counted in a ledger alongside the spend estimate
#   * one live      creation is refused while an instance this run created is still alive
#   * ownership     exec/copy/down refuse a name that this run did not create (the owner's other instances are not ours)
#
# Everything the ledger records is non-secret: name, type, price, timestamps, the argv of the call.

BREV_MAX_PRICE_PER_HOUR="${CERBERUS_BREV_MAX_PRICE:-1.50}"
BREV_MAX_HOURS="${CERBERUS_BREV_MAX_HOURS:-3.5}"
BREV_MAX_CREATIONS="${CERBERUS_BREV_MAX_CREATIONS:-4}"
BREV_DIR="${CERBERUS_BREV_DIR:-$HOME/cerberus-results/brev}"
BREV_LEDGER="$BREV_DIR/ledger.jsonl"

brev_say() { echo "[brev $(date -u +%H:%M:%S)] $*"; }

brev_append_ledger() {  # brev_append_ledger <action> <name> <json-extra>
    mkdir -p "$BREV_DIR"
    local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '{"ts":"%s","action":"%s","name":"%s"%s}\n' "$ts" "$1" "$2" "${3:+,$3}" >> "$BREV_LEDGER"
}

brev_creations() {  # instances this run created (not deleted)
    [[ -f "$BREV_LEDGER" ]] || { echo 0; return; }
    python3 - "$BREV_LEDGER" <<'PY'
import json, sys
created, deleted = [], []
for line in open(sys.argv[1]):
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if row.get("action") == "create":
        created.append(row.get("name"))
    elif row.get("action") == "delete":
        deleted.append(row.get("name"))
print(len([n for n in created if n not in deleted]))
PY
}

brev_managed() {  # 0 when this run created <name>
    [[ -f "$BREV_LEDGER" ]] || return 1
    python3 - "$BREV_LEDGER" "$1" <<'PY'
import json, sys
name = sys.argv[2]
for line in open(sys.argv[1]):
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if row.get("action") == "create" and row.get("name") == name:
        sys.exit(0)
sys.exit(1)
PY
}

brev_logged_in() {
    timeout 30 brev ls --json >/dev/null 2>&1
}

# Count instances whose state is not stopped/deleted. Unknown output means "cannot tell", which refuses creation.
brev_live_count() {
    local out
    out="$(timeout 30 brev ls --json 2>/dev/null)" || { echo unknown; return; }
    python3 - "$out" <<'PY'
import json, sys
raw = sys.argv[1]
try:
    data = json.loads(raw)
except ValueError:
    print("unknown"); raise SystemExit
# The live API answers {"workspaces": [...]} and null when the org is empty; older builds used "instances".
rows = data
if isinstance(data, dict):
    for key in ("workspaces", "instances", "nodes"):
        if key in data:
            rows = data[key] or []
            break
if not isinstance(rows, list):
    print("unknown"); raise SystemExit
dead = {"STOPPED", "DELETED", "DELETING", "TERMINATED", "ERROR", "FAILED"}
live = 0
for row in rows:
    if not isinstance(row, dict):
        continue
    state = str(row.get("status") or row.get("state") or "").upper()
    if state and state not in dead:
        live += 1
print(live)
PY
}

# brev_guard_create <name> <price-per-hour>: every cap in one place, before anything is created.
brev_guard_create() {
    local name="$1" price="$2"
    if ! brev_logged_in; then
        brev_say "not logged in to Brev (run 'brev login'); refusing to create anything" >&2
        return 2
    fi
    if ! python3 -c "import sys; sys.exit(0 if float('$price') <= float('$BREV_MAX_PRICE_PER_HOUR') else 1)"; then
        brev_say "price \$$price/h exceeds the cap \$$BREV_MAX_PRICE_PER_HOUR/h; refusing" >&2
        return 3
    fi
    local made; made="$(brev_creations)"
    if (( made >= BREV_MAX_CREATIONS )); then
        brev_say "already created $made instance(s) this run (cap $BREV_MAX_CREATIONS); refusing" >&2
        return 4
    fi
    local live; live="$(brev_live_count)"
    if [[ "$live" == "unknown" ]]; then
        brev_say "cannot read 'brev ls --json' (not logged in?); refusing to create with an unknown live count" >&2
        return 5
    fi
    if (( live > 0 )); then
        brev_say "$live live instance(s) exist; this run keeps one live instance at a time, so stop yours first" >&2
        return 6
    fi
    return 0
}

# brev_arm_kill <name> <hours>: a transient user timer that stops and deletes the instance even if this agent dies.
brev_arm_kill() {
    local name="$1" hours="${2:-$BREV_MAX_HOURS}" unit="cerberus-kill-$1" when
    # Fractional hours (3.5) are not portable to date -d, so convert to whole seconds first.
    local seconds; seconds="$(python3 -c "print(int(float('$hours') * 3600))")"
    when="$(date -d "+$seconds seconds" '+%Y-%m-%d %H:%M:%S')"
    if [[ -z "$when" ]]; then
        brev_say "could not compute the kill deadline; refusing" >&2
        return 7
    fi
    systemd-run --user --unit="$unit" --on-calendar="$when" --timer-property=Persistent=yes \
        /bin/bash -lc "brev stop $name; sleep 60; brev delete $name" >/dev/null 2>&1 || {
        brev_say "could not arm the kill timer; refusing to create an instance without one" >&2
        return 7
    }
    if ! systemctl --user list-timers "$unit*" --no-pager 2>/dev/null | grep -q "$unit"; then
        brev_say "kill timer did not appear in 'systemctl --user list-timers'; cancelling and refusing" >&2
        brev_cancel_kill "$name"
        return 7
    fi
    brev_say "kill timer armed for $when (unit $unit)"
    return 0
}

brev_cancel_kill() {
    local unit="cerberus-kill-$1"
    systemctl --user stop "$unit.timer" 2>/dev/null
    systemctl --user stop "$unit.service" 2>/dev/null
    systemctl --user reset-failed "$unit.service" 2>/dev/null
    return 0
}

# brev_spend_estimate: sum price x hours-alive for creates that were not deleted, from the ledger alone.
brev_spend_estimate() {
    [[ -f "$BREV_LEDGER" ]] || { echo "0.00"; return; }
    python3 - "$BREV_LEDGER" <<'PY'
import json, sys
from datetime import datetime, timezone
rows = []
for line in open(sys.argv[1]):
    try:
        rows.append(json.loads(line))
    except ValueError:
        pass
open_since, total = {}, 0.0
for row in rows:
    ts = datetime.strptime(row["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if row.get("action") == "create":
        open_since[row["name"]] = (ts, float(row.get("price", 0) or 0))
    elif row.get("action") == "delete" and row["name"] in open_since:
        start, price = open_since.pop(row["name"])
        total += price * (ts - start).total_seconds() / 3600
now = datetime.now(timezone.utc)
for start, price in open_since.values():
    total += price * (now - start).total_seconds() / 3600
print(f"{total:.2f}")
PY
}
