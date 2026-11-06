#!/usr/bin/env bash
# Runs LOCALLY (where Docker already exists) -- extracts the compiled
# runtime artifact out of a pushed docker/slam/<name>/Dockerfile image
# (the whole /colcon_ws + its pixi env + its activation script), without
# ever running that image as a container, so it can be layered onto an
# already-running cerberus-remote pod at boot time instead of running as a
# separate container (see the SLAM-remote-orchestration plan's Context
# section for why: RunPod pods are one container each, and baking sim and
# backend together at Docker-build time would break the deliberate
# decoupling between docker/remote/Dockerfile and docker/slam/*/Dockerfile).
#
# `docker create` (never runs the image) + `docker cp` is the standard way
# to pull files out of an image's filesystem without executing it.
#
# Extracts the ENTIRE /colcon_ws, not just /colcon_ws/install -- every
# docker/slam/<name>/Dockerfile builds with `colcon build --symlink-install`,
# whose install/ tree is NOT self-contained: install/<pkg>/share/<pkg>/
# local_setup.bash (and package.xml, etc.) are ABSOLUTE symlinks into
# /colcon_ws/build/<pkg>/... and /colcon_ws/src/<pkg>/..., confirmed by
# hitting "not found: /colcon_ws/install/spark_fast_lio/share/spark_fast_lio/
# local_setup.bash" after extracting install/ alone against a real pod.
# Copying all of /colcon_ws (build/ + install/ + src/) and placing it back
# at the identical absolute path on the target makes those symlinks resolve
# correctly again -- cheaper than trying to dereference/flatten them.
#
# Renames the extracted activation script from the image's own
# /pixi-project/activate.sh to activate-<backend>.sh -- both
# docker/remote/Dockerfile's sim-ros2 env AND every docker/slam/<name>/
# Dockerfile use that exact same fixed filename for their own, DIFFERENT
# pixi env's activation script. Merged onto one running container
# (this extraction's whole point), the backend's copy would silently
# overwrite the sim image's own -- confirmed by reading both Dockerfiles'
# COPY lines, not guessed. The .pixi/envs/slam-<backend> directory itself
# doesn't collide (already uniquely named per backend), only this one
# generic activation-script filename does.
#
# orb_slam3 (and multicam_slam, per docker/slam/orb_slam3/Dockerfile's own
# comment about its "twin-shaped wrapper") need ONE MORE path this script
# does NOT pull: docker/slam/orb_slam3/Dockerfile COPYs
# third_party/slam/orb_slam3 to /workspace/orb_slam3 and symlinks
# /colcon_ws/src/lib, /colcon_ws/src/Thirdparty, /colcon_ws/src/include into
# it -- those symlinks resolve fine at BUILD time (colcon's own -I/-L
# search), but orb_slam3_node's compiled RPATH only ever points at its pixi
# env's lib/ (confirmed via `readelf -d`), never at /workspace/orb_slam3, so
# at RUNTIME the .so files there are invisible unless synced separately AND
# exported via LD_LIBRARY_PATH -- see orb_slam3/launch_algorithm.sh's own
# fix for the exact paths (lib/, Thirdparty/DBoW2/lib/, Thirdparty/g2o/lib/).
# This script intentionally does NOT extract /workspace/orb_slam3 -- it's
# large (Thirdparty builds) and this is a one-backend-specific need, not a
# general artifact shape; sync it manually per orb_slam3/launch_algorithm.sh's
# comments, e.g. via `docker cp <container>:/workspace/orb_slam3 ...` the
# same way this script pulls /colcon_ws, then tar-over-SSH to the pod at the
# identical /workspace/orb_slam3 path.
#
# Usage: scripts/extract_slam_backend_artifact.sh <backend-name> [image-tag]
set -eu

BACKEND="${1:?Usage: $0 <backend-name> [image-tag]}"
IMAGE_TAG="${2:-kevinkyawthiha/cerberus-slam-${BACKEND//_/-}:v1}"
# pixi.toml's [environments] table names every SLAM env with hyphens
# (slam-fast-lio2, not slam-fast_lio2) regardless of the backend's own
# underscore-style name (matching docker/slam/<name>/ and
# scripts/fetch_slam_backends.py) -- confirmed by hitting "Could not find
# the file /pixi-project/.pixi/envs/slam-fast_lio2 in container" against
# the real pushed image before adding this substitution.
PIXI_ENV_NAME="slam-${BACKEND//_/-}"

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/artifacts/slam_backends/${BACKEND}"
mkdir -p "$OUT_DIR"

echo "[extract_slam_backend_artifact] pulling $IMAGE_TAG..."
docker pull "$IMAGE_TAG"

CONTAINER_ID="$(docker create "$IMAGE_TAG")"
trap 'docker rm "$CONTAINER_ID" >/dev/null' EXIT

echo "[extract_slam_backend_artifact] extracting /colcon_ws, pixi env, and activation script..."
docker cp "$CONTAINER_ID:/colcon_ws" "$OUT_DIR/colcon_ws"
docker cp "$CONTAINER_ID:/pixi-project/.pixi/envs/${PIXI_ENV_NAME}" "$OUT_DIR/pixi-env"
docker cp "$CONTAINER_ID:/pixi-project/activate.sh" "$OUT_DIR/activate-${BACKEND}.sh"

echo "[extract_slam_backend_artifact] packing tarball..."
tar czf "$OUT_DIR.tar.gz" -C "$OUT_DIR" colcon_ws pixi-env "activate-${BACKEND}.sh"

echo "[extract_slam_backend_artifact] done: $OUT_DIR.tar.gz"
echo "[extract_slam_backend_artifact] sync to a pod with, e.g.:"
echo "  ssh -i ~/.ssh/cerberus_runpod_ed25519 -p <port> root@<host> mkdir -p /pixi-project/.pixi/envs"
echo "  cat $OUT_DIR.tar.gz | ssh -i ~/.ssh/cerberus_runpod_ed25519 -p <port> root@<host> \\"
echo "    'mkdir -p /tmp/slam_artifact && tar xzf - -C /tmp/slam_artifact && \\"
echo "     rm -rf /colcon_ws && mv /tmp/slam_artifact/colcon_ws /colcon_ws && \\"
echo "     mv /tmp/slam_artifact/pixi-env /pixi-project/.pixi/envs/${PIXI_ENV_NAME} && \\"
echo "     mv /tmp/slam_artifact/activate-${BACKEND}.sh /pixi-project/activate-${BACKEND}.sh && \\"
echo "     rmdir /tmp/slam_artifact'"
