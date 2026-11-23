#!/usr/bin/env bash
# Package the showcase: gate every clip with the clip QA gate, stage the ones that pass, and (optionally) create a
# DRAFT GitHub release with them. Never publishes.
#
# usage: package_showcase.sh --results DIR --stage DIR [--release TAG] [--dry-run]
#
# Clip names map to camera kinds by their prefix (ego_*, chase_*, closeup_*, overview_*); the kinematic mission
# animation is a schematic and is gated by ffprobe only (`--camera none`), never by the photo checks.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

RESULTS=""; STAGE=""; TAG=""; DRY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --results) RESULTS="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --release) TAG="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) echo "package_showcase: unknown argument $1" >&2; exit 2 ;;
    esac
done
[[ -n "$RESULTS" && -n "$STAGE" ]] || { echo "usage: package_showcase.sh --results DIR --stage DIR [--release TAG]" >&2; exit 2; }
mkdir -p "$STAGE"

camera_for() {
    case "$(basename "$1")" in
        ego_*) echo ego ;;
        chase_*) echo chase ;;
        closeup_*) echo closeup ;;
        overview_*) echo overview ;;
        *) echo none ;;
    esac
}

PASS=0; FAIL=0; FAILED=()
for clip in "$RESULTS"/*.mp4 "$RESULTS"/clips/*.mp4; do
    [[ -f "$clip" ]] || continue
    base="$(basename "$clip")"
    cam="$(camera_for "$clip")"
    poses="$(dirname "$clip")/${base%.mp4}.poses.json"
    args=(--camera "$cam" --min-duration 3)
    [[ "$cam" == "none" ]] && args+=(--media-only)  # schematics: a top-down map is not a rendered photo
    if [[ -f "$poses" && "$cam" != "none" ]]; then
        args+=(--poses "$poses")
    elif [[ "$cam" != "none" ]]; then
        args+=(--no-robot-visibility)
    fi
    if python3 scripts/host/clip_qa.py "$clip" "${args[@]}" --json "$STAGE/${base%.mp4}.qa.json" > "$STAGE/${base%.mp4}.qa.txt" 2>&1; then
        PASS=$((PASS + 1))
        cp "$clip" "$STAGE/"
        echo "PASS $base (camera $cam)"
    else
        FAIL=$((FAIL + 1)); FAILED+=("$base")
        echo "FAIL $base (camera $cam) -- see $STAGE/${base%.mp4}.qa.txt"
    fi
done

echo "gated: $PASS passed, $FAIL failed"
(( FAIL == 0 )) || printf 'failed: %s\n' "${FAILED[*]}"

if [[ -n "$TAG" ]]; then
    if [[ $DRY -eq 1 ]]; then
        echo "[dry-run] gh release create $TAG --draft --title \"Showcase $TAG\" $STAGE/*.mp4"
    else
        # a DRAFT release only: nothing becomes public without the owner
        gh release create "$TAG" --draft --title "Showcase $TAG" --notes "Clips that passed the clip QA gate. See docs/showcase.md for what each one is." \
            $STAGE/*.mp4 || { echo "release creation failed; clips are staged in $STAGE" >&2; exit 1; }
        echo "draft release $TAG created"
    fi
fi
