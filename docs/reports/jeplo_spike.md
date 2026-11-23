# JEPLO spike: does the authors' release run, and what would an adapter cost?

**Answer: it did not run here.** The C++ sim-to-sim tool the authors ship could not be built, and the failure point
is precise (below). Nothing in this repository changed, no JEPLO code or dependency entered `pixi.toml`, and the
clone lives at `~/jeplo` on the VM, outside the repository. JEPLO is **GPLv3**; the isolation is deliberate.

This documents the *authors' demo with their released checkpoint*, not our pipeline. The backend card's support level
is unchanged.

## What was attempted

| step | result |
|---|---|
| `git clone --depth 1 https://github.com/ASIG-X/JEPLO ~/jeplo` | ok (outside the repo) |
| `apt-get install build-essential cmake libyaml-cpp-dev libspdlog-dev libboost-all-dev libglfw3-dev libfmt-dev libzmq3-dev libeigen3-dev` | ok |
| build `unitree_sdk2` (CycloneDDS, ddscxx) and install to `/usr/local` | ok |
| build `deployment/unitree_mujoco` (the authors' customised MuJoCo viewer/sim) | **failed** |

## Exact failure point

The clone does not include MuJoCo: `simulate/CMakeLists.txt` expects `simulate/mujoco/` to be a symlink to a MuJoCo
release, and `deployment/unitree_mujoco/readme.md` says `ln -s ~/.mujoco/mujoco-3.3.6 mujoco`. Three MuJoCo versions
were tried; each lacks one of the two symbols the authors' code uses:

| MuJoCo | first failure |
|---|---|
| 3.2.7 (git) | `depth_camera.h: mujoco::Simulate has no member user_images_new_` |
| 3.1.6 (git) | same (`user_images_new_`, `newimagerequest`) |
| 3.3.6 (release, the version the readme names) | `main.cc:669: mujoco::Simulate has no member render_callback` |

So the code sits between releases: it needs `user_images_new_` (MuJoCo 3.3.x) **and** `Simulate::render_callback`
(absent in 3.3.6), i.e. an unpublished MuJoCo commit. Building it needs the authors' pin (not in the repository) or a
local patch of their depth hook -- patching the hook out removes the depth image the policy consumes, which would no
longer be their demo.

A second route exists and was not attempted (session budget): `deployment/unitree_mujoco/simulate_python` (pip
MuJoCo + `unitree_sdk2_python` + a ZMQ depth publisher + `MuJoCo-LiDAR`). It needs the same depth pipeline and a DDS
bridge between two processes.

## What the release is, from its own repository

| item | value |
|---|---|
| input | a **60x25** (or 90/120x25) spherical depth image over ZMQ: `uint32 width`, `uint32 height`, then `width*height` float32 depth, row-major, row 0 = -4.5 deg elevation, row 24 = +45.5 deg, values normalised to [0, 1] with 1.0 = 2.0 m |
| sensor | Unitree **Mid-360** LiDAR (or Airy) simulated in MuJoCo, 20 Hz, ZMQ PUB on port 5590 |
| robot | Unitree Go2, MuJoCo `scene.xml`, DDS domain 1 over `lo` |
| command interface | Unitree SDK2 high-level commands (`SportModeState`, `LowCmd`/`LowState`) -- **not** our `LocoCommand` (vx, vy, wz at 50 Hz); an adapter has to translate both ways and re-time to our 50 Hz loop |
| training stack | Isaac Lab **2.3.2.post1** + Isaac Sim 5.1 via pip, `torch==2.7.0+cu128`, `rsl_rl` and `lejepa` installed editable; ours is Sim 6.2 / Lab 3.0 GA with torch pinned by the image |
| observation | depth image + proprioception in the authors' own layout, not our 235-D contract |
| checkpoint | shipped through the repo's release assets; the local clone carries no weights (the deployment pipeline downloads them) |

## Adapter estimate (one page)

1. **Sensor simulation**: our scene has no Mid-360. A 60x25 spherical depth image has to be produced from the scene
   (Isaac camera + projection, or the existing RTX-LiDAR path) at 20 Hz, normalised to [0, 1] with the 2 m clipping
   the authors use. ~1-2 weeks to make it faithful, a few days for a crude version.
2. **Observation assembly**: build their layout from our robot state, and keep our 235-D contract untouched for the
   existing checkpoint. ~3-5 days.
3. **Command/interface bridge**: DDS <-> our `LocoCommand` at 50 Hz with the rate limiter we already use. ~2-3 days.
4. **Joint order and asset**: their Go2 MuJoCo model and our fused Go2+ARX5 differ in joint names/order and in mass
   (18.5 kg vs their 15 kg); the policy would have to be retrained or at least re-validated on the arm body. ~1 week.
5. **Licence isolation**: GPLv3 code stays outside the repo, invoked as a separate process, with its own environment;
   no GPL source or binary is vendored. Already done here (`~/jeplo`).

Bottom line: the release is a real, reproducible demo *on its own pinned stack*; adapting it means re-implementing its
sensor simulation on our stack, which is a 3-4 week task and a different research direction from the scripted-arms
mission this repository is pursuing. Recommendation: keep it as a reference, do not integrate.
