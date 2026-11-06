#!/usr/bin/env bash
# Build the NEW-stack remote image from a MINIMAL temp build context.
# The repo root has no .dockerignore and ships ~8 GB of .pixi/third_party, so
# `docker build .` from the root would upload all of it; this copies only
# docker/remote_v6/* plus pixi.toml/pixi.lock (for the sim-ros2 stage) into a temp dir.
#
#   TAG=v6-dev ./docker/remote_v6/build.sh
#   ISAACLAB_REF=v3.0.0 ARENA_REF=<sha> TAG=v6-ga PUSH=1 ./docker/remote_v6/build.sh
#
# Env: TAG (default v6-dev), ISAACLAB_REF, ARENA_REF (default: Dockerfile ARGs),
#      ISAACLAB_INSTALL_SPEC (optional), PUSH=1 to push (off by default).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${TAG:-v6-dev}"
IMAGE="kevinkyawthiha/cerberus-remote-v6:${TAG}"

CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
cp -a "$HERE"/. "$CTX"/
cp "$HERE/../../pixi.toml" "$HERE/../../pixi.lock" "$CTX"/

ARGS=()
[ -n "${ISAACLAB_REF:-}" ] && ARGS+=(--build-arg "ISAACLAB_REF=${ISAACLAB_REF}")
[ -n "${ARENA_REF:-}" ] && ARGS+=(--build-arg "ARENA_REF=${ARENA_REF}")
[ -n "${ISAACLAB_INSTALL_SPEC:-}" ] && ARGS+=(--build-arg "ISAACLAB_INSTALL_SPEC=${ISAACLAB_INSTALL_SPEC}")

# The base image is on NGC: `docker login nvcr.io` first (user $oauthtoken).
DOCKER_BUILDKIT=1 docker build "${ARGS[@]}" -t "$IMAGE" -f "$CTX/Dockerfile" "$CTX"

if [ "${PUSH:-0}" = "1" ]; then
    docker push "$IMAGE"
else
    echo "PUSH!=1: not pushing $IMAGE"
fi

echo "image:  $IMAGE"
echo "id:     $(docker image inspect --format '{{.Id}}' "$IMAGE")"
DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null || true)"
echo "digest: ${DIGEST:-<none until pushed>}"
echo "size:   $(docker image inspect --format '{{.Size}}' "$IMAGE" | awk '{printf "%.2f GB\n", $1/1e9}')"
