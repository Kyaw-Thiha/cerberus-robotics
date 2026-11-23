#!/bin/bash
# Cerberus dead-man switch, VM side. Installed by scripts/host/deadman_setup.sh.
#
# Every 5 minutes: if the laptop's heartbeat file is older than CERBERUS_DEADMAN_MAX_AGE_MIN, or this VM is older
# than CERBERUS_DEADMAN_HARD_HOURS, copy the results and training directories to R2 (10-minute cap) and then delete
# this VM through the Hyperstack API using its own id. Log-only mode (CERBERUS_DEADMAN_LOG_ONLY=1) does everything
# except the copy and the delete, which is how it is tested before it is relied on.
set -uo pipefail
LOG=/var/log/cerberus-deadman.log
STATE=/etc/cerberus/vm.json
HB=/var/lib/cerberus/heartbeat
MAX_AGE_MIN="${CERBERUS_DEADMAN_MAX_AGE_MIN:-45}"
HARD_HOURS="${CERBERUS_DEADMAN_HARD_HOURS:-9.5}"
LOG_ONLY="${CERBERUS_DEADMAN_LOG_ONLY:-0}"
PREFIX="${CERBERUS_DEADMAN_PREFIX:-migration/session2/vm-backup}"

exec >>"$LOG" 2>&1
say() { echo "$(date -u +%FT%TZ) $*"; }

now=$(date +%s)
hb_seen=$(stat -c %Y "$HB" 2>/dev/null || echo 0)
hb_age_min=$(( (now - hb_seen) / 60 ))
created=$(python3 -c 'import json;print(int(json.load(open("/etc/cerberus/vm.json"))["created_epoch"]))' 2>/dev/null || echo "$now")
vm_id=$(python3 -c 'import json;print(json.load(open("/etc/cerberus/vm.json"))["id"])' 2>/dev/null || echo "")
age_min=$(( (now - created) / 60 ))
say "check: heartbeat ${hb_age_min} min old, vm ${age_min} min old, id ${vm_id:-unknown}"

trigger=""
if [ "$hb_age_min" -gt "$MAX_AGE_MIN" ]; then trigger="heartbeat stale (${hb_age_min} min > ${MAX_AGE_MIN})"; fi
if [ "$age_min" -gt "$(( ${HARD_HOURS%.*} * 60 + 30 ))" ]; then trigger="vm too old (${age_min} min)"; fi
if [ -z "$trigger" ]; then say "ok"; exit 0; fi
say "TRIGGER: $trigger"

if [ "$LOG_ONLY" = "1" ]; then
    say "log-only mode: would sync to $PREFIX and delete VM $vm_id"
    exit 0
fi

# 1. copy results and training outputs, 10-minute cap, never fail the delete on a copy error
set -a
# shellcheck disable=SC1091
. /etc/cerberus/r2.env
set +a
export RCLONE_CONFIG_R2_TYPE=s3 RCLONE_CONFIG_R2_PROVIDER=Cloudflare \
       RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
       RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
       RCLONE_CONFIG_R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
       RCLONE_CONFIG_R2_ACL=private
for dir in /home/ubuntu/cerberus-results /home/ubuntu/cerberus-robotics/policy/locomotion/runs; do
    [ -d "$dir" ] || continue
    timeout 540 rclone copy "$dir" "R2:${R2_BUCKET}/${PREFIX}/$(basename "$dir")" --transfers 4 \
        || say "copy failed for $dir (continuing to the delete)"
done
say "copy done"

# 2. delete this VM
key=$(cat /etc/cerberus/hyperstack.key)
code=$(curl -s -o /tmp/deadman_delete.json -w '%{http_code}' -X DELETE \
    -H "api_key: $key" "https://infrahub-api.nexgencloud.com/v1/core/virtual-machines/${vm_id}")
say "delete returned HTTP $code"
exit 0
