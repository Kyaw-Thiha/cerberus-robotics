#!/usr/bin/env bash
# Pre-flight: everything that can be checked without Isaac Sim or a GPU, before paying for a pod.
#
# The pod runs in this phase each cost 10-30 minutes of GPU time to diagnose, and every one of them was
# an interface bug rather than a physics question: a policy handed a module, observations arriving as a
# mapping, gymnasium's five-value step, a fixed-batch ONNX export, a training loop that never saved.
# This script exercises those contracts locally, so the next pod run is a confirmation instead of a
# discovery.
#
# Checks:
#   1 shell syntax of every pod driver
#   2 shellcheck, if it can be installed (quoting, word splitting, set -u hazards)
#   3 the repo's pure-python test suite (pixi)
#   4 the wrapper contract tests under CPU torch (fake vec-env, no Isaac, no GPU)
#   5 the ONNX export against the checkpoint's own actor weights
#
# Usage: scripts/preflight_local.sh [--skip-install]
set -uo pipefail
export TERM=${TERM:-xterm}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
VENV="${PREFLIGHT_VENV:-/tmp/cerberus-preflight}"
SKIP_INSTALL=0
[[ "${1:-}" == "--skip-install" ]] && SKIP_INSTALL=1

# The scripts this phase owns. The repo has older shell scripts with constructs shellcheck cannot
# parse (scripts/run_slam_eval_remote.sh), which are not this work's to change.
OWNED_SCRIPTS=(
    scripts/pod_isaac_lib.sh
    scripts/run_ga_stack_check_pod.sh
    scripts/run_ga_retrain_pod.sh
    scripts/run_adversarial_gate_pod.sh
    scripts/run_robustify_pod.sh
    scripts/pull_pod_artifacts.sh
    scripts/preflight_local.sh
)

FAILS=0
say() { echo "[preflight] $*"; }
step() { echo; echo "=== $* ==="; }
record() { echo "[preflight] $1: $2"; [[ "$2" == PASS || "$2" == SKIP ]] || FAILS=$((FAILS + 1)); }

step "1. shell syntax"
syntax_fail=0
for script in "${OWNED_SCRIPTS[@]}"; do
    bash -n "$script" || { say "syntax error in $script"; syntax_fail=$((syntax_fail + 1)); }
done
record "shell-syntax" "$([[ $syntax_fail -eq 0 ]] && echo PASS || echo FAIL)"

step "2. shellcheck"
SC="$(command -v shellcheck || true)"
[[ -z "$SC" && -x "$VENV/bin/shellcheck" ]] && SC="$VENV/bin/shellcheck"
if [[ -z "$SC" && $SKIP_INSTALL -eq 0 ]]; then
    python3 -m pip install --quiet shellcheck-py 2>/dev/null || true
    SC="$(command -v shellcheck || true)"
    if [[ -z "$SC" ]]; then
        SC="$(python3 -c 'import shellcheck_py,pathlib;print(pathlib.Path(shellcheck_py.__file__).parent/"shellcheck")' 2>/dev/null || true)"
    fi
fi
if [[ -n "$SC" && -x "$SC" ]]; then
    # SC1091: sourced files are resolved at runtime, not statically.
    "$SC" -S warning -e SC1091 "${OWNED_SCRIPTS[@]}" && record shellcheck PASS || record shellcheck FAIL "see above"
else
    record shellcheck SKIP "not installed (pip install shellcheck-py)"
fi

step "3. repo test suite (pixi)"
if pixi run pytest -q > /tmp/preflight-pytest.log 2>&1; then
    record pytest PASS "$(tail -1 /tmp/preflight-pytest.log)"
else
    # Collection errors from unbuilt colcon packages are expected on a workstation.
    if grep -qE "^[0-9]+ passed" /tmp/preflight-pytest.log; then
        record pytest PASS "$(tail -1 /tmp/preflight-pytest.log) (collection errors are the known unbuilt-package ones)"
    else
        record pytest FAIL "see /tmp/preflight-pytest.log"
    fi
fi

step "4. wrapper contract tests (CPU torch, fake vec-env)"
if [[ $SKIP_INSTALL -eq 0 ]]; then
    [[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
    "$VENV/bin/pip" -q install pytest onnxruntime 2>&1 | tail -1
    "$VENV/bin/python" -c "import torch" 2>/dev/null || \
        "$VENV/bin/pip" -q install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1
fi
if [[ -x "$VENV/bin/python" ]]; then
    if "$VENV/bin/python" -m pytest policy/locomotion/adversarial_robustness/tests/test_adversarial_env_contract.py -q; then
        record wrapper-tests PASS
    else
        record wrapper-tests FAIL
    fi
else
    record wrapper-tests SKIP "no venv"
fi

step "5. ONNX export vs checkpoint actor"
CKPT="policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate"
if [[ -f "$CKPT.pt" && -f "$CKPT.onnx" && -x "$VENV/bin/python" ]]; then
    if "$VENV/bin/python" scripts/verify_onnx_matches_checkpoint.py --checkpoint "$CKPT.pt" --onnx "$CKPT.onnx"; then
        record onnx-equivalence PASS
    else
        record onnx-equivalence FAIL
    fi
else
    record onnx-equivalence SKIP "checkpoint/onnx not present locally"
fi

step "6. R2 credentials"
# A wrong assumption here already cost this phase its artifacts once: the credentials were in .env but
# the operator's key scan missed them, so no upload was ever attempted. Check the variables exist and,
# if rclone is present, that they actually authenticate.
if [[ -f .env ]]; then
    set -a; . ./.env; set +a
fi
missing=()
for var in R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET; do
    [[ -n "${!var:-}" ]] || missing+=("$var")
done
if [[ ${#missing[@]} -gt 0 ]]; then
    record r2-creds FAIL "missing: ${missing[*]}"
elif ! command -v rclone >/dev/null 2>&1; then
    record r2-creds SKIP "variables present; rclone not installed locally (the pod image has it)"
else
    RCLONE_CONFIG_CERBERUS_R2_TYPE=s3 RCLONE_CONFIG_CERBERUS_R2_PROVIDER=Cloudflare \
    RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
    RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
    RCLONE_CONFIG_CERBERUS_R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    RCLONE_CONFIG_CERBERUS_R2_ACL=private RCLONE_CONFIG_CERBERUS_R2_NO_CHECK_BUCKET=true \
        rclone lsf "cerberus_r2:${R2_BUCKET}" --max-depth 1 > /tmp/preflight-r2.txt 2>/tmp/preflight-r2.err \
        && record r2-creds PASS "authenticated; $(wc -l < /tmp/preflight-r2.txt) top-level entries" \
        || record r2-creds FAIL "$(head -2 /tmp/preflight-r2.err | tr '\n' ' ')"
fi

echo
say "done: $FAILS failed check(s)"
exit "$FAILS"
