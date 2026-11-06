# Cerberus remote image, NEW stack: Isaac Sim 6.2.0, Python 3.12, Isaac Lab
# v3.0.0 (GA), Isaac Lab Arena `main`. Lives side by side with docker/remote
# (OLD stack: Sim 5.1.0 / Lab v2.3.2 / py3.11, kept for checkpoint-dependent
# checks). Runs ONLY on a rented remote GPU pod (see docs/runpod.md).
#
# Sources each decision is based on (read 2026-10-05):
#   - docker/remote/{Dockerfile,start.sh,requirements.txt}: the behaviours pods
#     rely on (root user, sshd from $PUBLIC_KEY, /etc/cerberus_env.sh, Isaac
#     Lab at /workspace/isaaclab, ENTRYPOINT override, TERM, tmux/openssh).
#   - https://github.com/isaac-sim/IsaacLab-Arena/blob/main/docker/Dockerfile.isaaclab_arena
#     : FROM isaac-sim:6.1.0, `ln -s /isaac-sim _isaac_sim`, `isaaclab.sh -i`,
#     `python.sh -m pip install -e <arena>` (we skip [dev] and the GR00T /
#     openpi / cuRobo parts), Vulkan dir hiding.
#   - Arena main pyproject.toml: requires-python >=3.12,<3.13; runtime deps
#     already include lightwheel-sdk; [dev] adds simready-search which needs
#     the boto3 force-reinstall hack, so [dev] is deliberately NOT installed.
#   - Arena main `submodules/IsaacLab` pin and Arena main HEAD (GitHub API
#     contents/submodules?ref=main, commits/main). The defaults of the two REF
#     args below are now Lab v3.0.0 GA (e3f7d9ff...) and the Arena `main` commit
#     whose submodule pin is that same Lab SHA (276d9731...); see
#     docs/arena_integration_log.md, "Lab 3.0 GA upgrade". At first write
#     (2026-10-05) they were bb0c8e1b9 / 311d0f0fc.
#   - Isaac Lab @bb0c8e1b9 docker/Dockerfile.base: toml+pytest, apt deps via
#     tools/install_deps.py, carb env shim, `isaaclab.sh --install`, uninstall
#     imageio-ffmpeg and quadprog afterwards. `-i` grammar from
#     source/isaaclab/isaaclab/cli/__init__.py (`mimic`, `rl[rsl-rl]`, `all`).
#     rsl-rl-lib 5.4.1 is Lab's pin at that commit.
#   - https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_container.html
#     : base image tag, ACCEPT_EULA/PRIVACY_CONSENT env vars, /isaac-sim root.
#
# pixi stage (re-added 2026-10-06): the SLAM eval needs rclpy for the sim-to-ROS2
# relay (slam/slam/ground_truth/ros2_bridge_relay.py) and `evo` for scoring.
# Isaac Sim's own Python cannot host rclpy (different interpreter/ABI), so
# they live in the pixi `sim-ros2` env, copied to /pixi-project exactly like
# docker/remote/Dockerfile does. The SLAM backends themselves are NOT baked in:
# scripts/extract_slam_backend_artifact.sh layers them onto a running pod.
# Build context therefore needs pixi.toml + pixi.lock (docker/remote_v6/build.sh
# copies them from the repo root into its temp context).
#
# Cerberus code is NOT baked in (synced per pod); no scene datasets inside the
# image (GRScenes lives in private R2, fetched per pod with rclone).

ARG ISAACSIM_VERSION=6.2.0

FROM ghcr.io/prefix-dev/pixi:noble AS pixi-build
WORKDIR /pixi-project
COPY pixi.toml pixi.lock ./
RUN pixi install -e sim-ros2 --locked
RUN pixi shell-hook -e sim-ros2 --shell bash > /pixi-project/activate.sh

FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION}
ARG ISAACSIM_VERSION=6.2.0

COPY --from=pixi-build /pixi-project/.pixi/envs/sim-ros2 /pixi-project/.pixi/envs/sim-ros2
COPY --from=pixi-build /pixi-project/activate.sh /pixi-project/activate.sh

# Root throughout, like the old image and RunPod's own images (disposable pod).
USER root
SHELL ["/bin/bash", "-c"]

# Pinned git refs (any tag/branch/SHA reachable on GitHub).
# Pins as of 2026-10-25 (Lab 3.0 GA, released 2026-10-20; GA targets Sim 6.2, torch 2.11+cu130, rsl-rl 5.5.1):
#   Lab  = v3.0.0 @ e3f7d9ff (GA tag; 147 commits after the previous pin dcabdcda on release/3.0.0)
#   Arena= main @ 276d97313 (its submodules/IsaacLab pin IS e3f7d9ff, so Lab and Arena now agree; its
#          Dockerfile base is isaac-sim:6.2.0). Arena needs the `registry_name` class attribute (see
#          common/cerberus_worlds/arena_assets/*_embodiment.py).
# Earlier pins, for rollback: Lab dcabdcda + Arena 34bb35a85 + Sim 6.1.0 (tag v6-rel2),
# Lab bb0c8e1b9 + Arena 311d0f0fc (tag v6-dev), Lab v3.0.0-EA ae37b028e (tag v6-ea-dev).
ARG ISAACLAB_REF=e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4
ARG ARENA_REF=276d9731356e77e974173f6c47cf53d719a764a9
# Argument of `isaaclab.sh --install`. `all` is what Arena's own Dockerfile
# uses (core + mimic + teleop + newton + rsl-rl 5.x + visualizers); a narrower
# spec such as "mimic,rl[rsl-rl]" also works at the pinned commit. The grammar
# differs on older Lab lines (e.g. `rsl_rl`), hence an ARG.
ARG ISAACLAB_INSTALL_SPEC=all

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    DEBIAN_FRONTEND=noninteractive \
    TERM=xterm \
    ISAACLAB_PATH=/workspace/isaaclab \
    ISAACSIM_ROOT_PATH=/isaac-sim \
    ARENA_PATH=/workspace/IsaacLab-Arena \
    PIP_RETRIES=12

# Hide conflicting Vulkan files, if present (as Arena's Dockerfile does).
RUN if [ -e /usr/share/vulkan ] && [ -e /etc/vulkan ]; then \
      mv /usr/share/vulkan /usr/share/vulkan_hidden; \
    fi

# System packages: build tools for source installs, git, the pod tooling (zip/unzip/p7zip
# for scene archives, curl, sshd, tmux) and a plain system python3 for scripts with no
# Isaac Sim dependency. rclone is NOT installed here -- see the pinned install below.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git git-lfs ca-certificates ncurses-term wget \
      zip unzip p7zip-full curl openssh-server tmux python3 \
    && mkdir -p /var/run/sshd \
    && rm -rf /var/lib/apt/lists/*

# rclone for R2, pinned. The base image's apt rclone is v1.58.1, which rejects the `Cloudflare` S3 provider
# that scripts/{sync_to_r2,fetch_grscenes_to_r2}.py set: every R2 upload fails with `s3 provider "Cloudflare"
# not known`, which reads like a credentials problem. Installed from a checksummed release rather than
# `rclone.org/install.sh` (that fetches "latest" and makes the build non-reproducible). scripts/sync_to_r2.py
# also preflights the version, so an image built without this stays fail-fast rather than fail-late.
# NOTE the CERB_ prefix: a build ARG is an ENV var inside RUN, and rclone parses every RCLONE_* env var as a flag
# (RCLONE_VERSION=1.75.1 became `--version=1.75.1` and aborted `rclone version`, found by the first v6-rel2 build).
ARG CERB_RCLONE_RELEASE=1.75.1
ARG CERB_RCLONE_SHA_AMD64=982b5aa772841168f8e380f139e9e787b2a105403e32b94da8676a0e1c0a13ab
ARG CERB_RCLONE_SHA_ARM64=03f2504174034b6d004152ed7369251c9a9ec1f7e0836eda420f5c7a5ec0dff9
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64) rarch=amd64; rsum="${CERB_RCLONE_SHA_AMD64}" ;; \
      arm64) rarch=arm64; rsum="${CERB_RCLONE_SHA_ARM64}" ;; \
      *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/rclone.zip \
      "https://downloads.rclone.org/v${CERB_RCLONE_RELEASE}/rclone-v${CERB_RCLONE_RELEASE}-linux-${rarch}.zip"; \
    echo "${rsum}  /tmp/rclone.zip" | sha256sum -c -; \
    unzip -q /tmp/rclone.zip -d /tmp/rclone; \
    install -m 0755 "/tmp/rclone/rclone-v${CERB_RCLONE_RELEASE}-linux-${rarch}/rclone" /usr/local/bin/rclone; \
    rm -rf /tmp/rclone /tmp/rclone.zip; \
    rclone version

WORKDIR /workspace

# ---- Isaac Lab at ISAACLAB_REF -> /workspace/isaaclab -----------------------
RUN git init ${ISAACLAB_PATH} \
    && git -C ${ISAACLAB_PATH} remote add origin https://github.com/isaac-sim/IsaacLab.git \
    && git -C ${ISAACLAB_PATH} fetch --depth 1 origin ${ISAACLAB_REF} \
    && git -C ${ISAACLAB_PATH} checkout --detach FETCH_HEAD \
    && git -C ${ISAACLAB_PATH} rev-parse HEAD > /tmp/isaaclab_sha \
    && ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim \
    && chmod +x ${ISAACLAB_PATH}/isaaclab.sh \
    && find ${ISAACLAB_PATH} -type f -name "*.sh" -exec sed -i 's/\r$//' {} +

# Carbonite env shim and per-extension apt deps, as in Lab's Dockerfile.base.
# Guarded: the scripts may not exist on every Lab ref.
RUN if [ -f ${ISAACLAB_PATH}/docker/scripts/install_carb_env_shim.sh ]; then \
      bash ${ISAACLAB_PATH}/docker/scripts/install_carb_env_shim.sh ${ISAACSIM_ROOT_PATH}; \
    fi \
    && ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install toml pytest \
    && if [ -f ${ISAACLAB_PATH}/tools/install_deps.py ]; then \
      ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py apt ${ISAACLAB_PATH}/source \
      && apt-get -y autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# Install Isaac Lab into Isaac Sim's Python 3.12. imageio-ffmpeg is removed by
# Lab itself (robomimic dep) and re-added from requirements.txt below; quadprog
# is removed as in Lab's Dockerfile.base.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    ${ISAACLAB_PATH}/isaaclab.sh --install "${ISAACLAB_INSTALL_SPEC}" \
    && (${ISAACLAB_PATH}/isaaclab.sh -p -m pip uninstall -y imageio-ffmpeg quadprog || true)

# ---- Isaac Lab Arena at ARENA_REF, editable, no GR00T/openpi/cuRobo ---------
RUN git init ${ARENA_PATH} \
    && git -C ${ARENA_PATH} remote add origin https://github.com/isaac-sim/IsaacLab-Arena.git \
    && git -C ${ARENA_PATH} fetch --depth 1 origin ${ARENA_REF} \
    && git -C ${ARENA_PATH} checkout --detach FETCH_HEAD \
    && git -C ${ARENA_PATH} rev-parse HEAD > /tmp/arena_sha

# Base deps only (no [dev], no gr00t/openpi/curobo). lightwheel-sdk is an Arena dependency already; named
# explicitly so the requirement is visible here, and PINNED (1.0.3 is what v6-rel resolved, from its build log)
# so a rebuild re-resolves the same kitchens and the only delta in a new image is what we changed on purpose.
ARG LIGHTWHEEL_SDK_VERSION=1.0.3
RUN --mount=type=cache,target=/root/.cache/pip \
    ${ISAACSIM_ROOT_PATH}/python.sh -m pip install -e ${ARENA_PATH} "lightwheel-sdk==${LIGHTWHEEL_SDK_VERSION}"

# Cerberus additive packages (pinned), installed last.
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    ${ISAACSIM_ROOT_PATH}/python.sh -m pip install -r /tmp/requirements.txt

# Record what was actually built.
RUN ${ISAACSIM_ROOT_PATH}/python.sh -c "\
import json, sys, importlib.metadata as m; \
norm = lambda s: s.lower().replace('_','-').replace(' ','-'); \
v = lambda n: next((d.version for d in m.distributions() if norm(d.metadata.get('Name','')) == norm(n)), None); \
json.dump({ \
  'isaacsim_image': '${ISAACSIM_VERSION}', \
  'isaaclab_ref': '${ISAACLAB_REF}', 'isaaclab_sha': open('/tmp/isaaclab_sha').read().strip(), \
  'arena_ref': '${ARENA_REF}', 'arena_sha': open('/tmp/arena_sha').read().strip(), \
  'isaaclab_install_spec': '${ISAACLAB_INSTALL_SPEC}', \
  'python': sys.version.split()[0], \
  'packages': {n: v(n) for n in ['isaacsim','isaaclab','isaaclab_rl','isaaclab_mimic','isaaclab_arena','rsl-rl-lib','lightwheel-sdk','torch','numpy','warp-lang','wandb','imageio','imageio-ffmpeg']}, \
}, open('/etc/cerberus_versions.json','w'), indent=2)" \
    && cat /etc/cerberus_versions.json \
    && rm -f /tmp/isaaclab_sha /tmp/arena_sha

# Convenience aliases for interactive shells.
RUN echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> /root/.bashrc \
    && echo "alias python='${ISAACSIM_ROOT_PATH}/python.sh'" >> /root/.bashrc \
    && echo "alias pip='${ISAACSIM_ROOT_PATH}/python.sh -m pip'" >> /root/.bashrc

WORKDIR /workspace

COPY start.sh /start.sh
RUN chmod +x /start.sh

# The base image sets its own ENTRYPOINT (/isaac-sim/runheadless.sh); clear it
# by replacing it, otherwise CMD would only be passed to it as an argument.
ENTRYPOINT ["/start.sh"]
