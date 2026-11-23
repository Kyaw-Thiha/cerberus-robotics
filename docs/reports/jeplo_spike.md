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

## Attempt 2 (session 3)

**Result: the authors' released policy ran in their MuJoCo sim-to-sim loop on this laptop, headless, and walked.**
It did not run on their shipped binaries end to end: the TensorRT deployment path cannot load here, and the ROS2
depth generator was replaced by a small port of its documented ZMQ algorithm. Two local compatibility patches in the
clone were required. Nothing in this repository changed except this file, no dependency entered `pixi.toml`, and the
GPLv3 clone stays at `~/jeplo2` with its own venv. Hardware: RTX 3050 Laptop 4 GB (driver 615.78.08), headless EGL,
no X server/Xvfb.

### What ran

| step | result |
|---|---|
| fresh clone `~/jeplo2` @ `460e227` | ok |
| `uv venv --python 3.12 .venv`; `mujoco==3.3.6 numpy pyzmq pygame` | ok |
| `cyclonedds==0.10.2` python binding | ok after pointing `CYCLONEDDS_HOME` at a prefix built from `unitree_sdk2/thirdparty` (prebuilt `libddsc.so`); needs `bin/` to exist for its locator |
| `unitree_sdk2py` (editable from upstream), domain 1 / `lo` | ok; log: `selected interface "lo" is not multicast-capable: disabling multicast` |
| `unitree_sdk2` C++ install to `~/jeplo2/local` (prebuilt `libunitree_sdk2.a`, CycloneDDS x86_64 blobs) | ok, no DDS build |
| `onnxruntime-cpp=1.32.0` from conda-forge into `~/jeplo2/ortenv` | ok (CPU+CUDA providers; no TensorRT provider) |
| `cmake -S deploy -B deploy/build -DCMAKE_BUILD_TYPE=Release -Donnxruntime_DIR=/tmp/ortshim -DCMAKE_PREFIX_PATH=$HOME/jeplo2/local:$HOME/jeplo2/ortenv` | ok in ~30 s |
| `deployment/unitree_mujoco/simulate_python` headless | ok; MuJoCo viewer replaced by a no-op, `MUJOCO_GL=egl` |
| authors' checkpoint | **ran** |

The `onnxruntime_DIR` shim only adds a plain `onnxruntime` interface target around conda's
`onnxruntime::onnxruntime`; the repository's `CMakeLists.txt` expects the former and the released CMake config
exports only the latter.

The checkpoint is actually in the clone, not a release download: `deployment/go2_deploy/model/pretrained/policy.onnx`
(5.5 MB), `sensor_estimator.onnx` (8.2 MB) and ten `.npz` validation cases per model. On the CUDA execution provider
the authors' own test cases pass: policy `worst_max_abs=0.00503`, estimator `worst_max_abs=0.00230`.

### Local patches (all outside this repository)

1. `simulate_python/MuJoCo-LiDAR/src/mujoco_lidar/core_cpu/mjlidar_cpu.py`, CPU ray-casting backend, is written for an
   older MuJoCo. With `mujoco==3.3.6` it raises
   `TypeError: mj_multiRay(): incompatible function arguments` because it passes `float32` `pnt`/`vec` and a
   `normal=None` keyword that the 3.3.6 binding does not accept. Two lines: cast `pnt`/`vec` to `float64`, drop
   `normal=None`. MuJoCo-LiDAR is not vendored by JEPLO; the URL named in the brief
   (`github.com/tbelagia/MuJoCo-LiDAR`) is 404, `discoverse-dev/MuJoCo-LiDAR` is the live upstream whose
   `src/mujoco_lidar` layout the script expects.
2. `deployment/go2_deploy/deploy/src/deploy.cpp` hard-codes the TensorRT execution provider. Stock behaviour here:
   `[ONNXRuntimeError] : 1 : FAIL : Failed to load library .../libonnxruntime_providers_tensorrt.so with error:
   ... cannot open shared object file`, twice, then `Failed to initialize locomotion controller` and
   `quit locomotion control loop` (no commands sent). The patch replaces the TensorRT options block with
   `AppendExecutionProvider_CUDA`. A CPU-only variant of the same patch was tried first: it ran the policy but hit
   `[ERROR] OOD safety fault: max_tau=24.693 threshold=30.000 max_joint_vel=26.790 threshold=25.000` at ~14 s,
   consistent with inference not keeping the 50 Hz loop.

### Depth input

`deployment/lidar_depth_pub` was not built: it is a ROS2 node (`rclcpp`, `livox_ros_driver2`) and this Arch box has
no `/opt/ros`; its `--sim` branch is compiled into the same ROS2 translation unit. I ported just that branch to
`/tmp/depth_pub.py` (outside both repos): SUB `tcp://localhost:5590`, PUB `tcp://*:5560`, message `uint32 width`,
`uint32 height`, then `25*60` float32, 10 Hz. Projection is the documented one: elevation -4.5..45.5 deg into 25 rows,
azimuth -60..+60 deg into 60 columns, 2 m clip, min-reduce, downsample every 4th ray, 5-frame element-wise min
accumulation, normalised to [0,1]. Run with `--no-cage-mask`; the cage mask is synthetic and only affects the demo
look, not the interface.

### Run commands (4 terminals, all headless)

```bash
cd ~/jeplo2/deployment/unitree_mujoco/simulate_python
LIDAR_BACKEND=jax MUJOCO_GL=egl LD_LIBRARY_PATH=$HOME/jeplo2/cdds-home/lib \
  ~/jeplo2/.venv/bin/python /tmp/run_sim_record.py          # sim + 1280x720 EGL recorder

~/jeplo2/.venv/bin/python /tmp/depth_pub.py --no-cage-mask  # ZMQ depth 5590 -> 5560

cd ~/jeplo2/deployment/go2_deploy/output
LD_LIBRARY_PATH=$HOME/jeplo2/ortenv/lib:$HOME/jeplo2/local/lib:$HOME/jeplo2/cdds-home/lib \
  ./deploy --net lo --model-dir pretrained --ood-count-threshold 15 --raw-actions

~/jeplo2/.venv/bin/python /tmp/cmd_pub.py 0.6 30            # (vx,vy,wz) float32 on 5562
```

### Evidence that the policy stepped

A recorded 40 s wall run (812 rendered frames, 40.6 s sim): the controller reached `Locomotion control sequence
started, robot ready`, then kept a 50 Hz command stream while the base moved from x = -0.04 m to x = +7.1 m
(y within 1.8 m) on a 0.6 m/s forward command (mean ~0.27 m/s including the stair climb and one fall). The base
rises to z = 1.96 m on the default scene's stairs, loses the stair at t ~ 22 s (z 0.08 m), recovers, and is at
x = 6.73 m when the run ends.
No OOD safety fault in this run; peak `ood_count` was 10 against the threshold 15 (the CPU-provider run faulted).

Rates: physics loop targets 200 Hz, control 50 Hz, depth publish 10 Hz. The LiDAR is the bottleneck: the JAX backend
produced ~15 Hz at 24k rays (76 frames in 5 s) and the patched CPU backend ~8 Hz, so the sim runs slower than real
time; this is the asynchronous-evaluation degradation the README warns about on non-4090 hardware, not a code error.

### Clip

`~/cerberus-results/2026-11-23/jeplo/walking_30s.mp4` -- 30 s, 1280x720, 20 fps, H.264 (1.7 MB), cut from the run
above (frames 140-740, sim t ~ 7-37 s).

### What still does not hold

- Not the authors' exact pipeline: depth comes from a port of their algorithm, policy inference is CUDA EP instead
  of TensorRT, and the viewer is replaced for headless operation. Their shipped TensorRT binary path remains unrun.
- MuJoCo-LiDAR and the ONNX Runtime C++ install are extra environment work the README does not describe for
  `simulate_python`; they are pinned only in the clone (`~/jeplo2`) and in `/tmp`.
- The robot does eventually lose the stair in this run; the checkpoint walks but this is not a clean benchmark.

Setup took ~20 min of the 75-min box, the recorded run ~1 min; the remaining limit was hardware/real-time, not
missing code. GPLv3 isolation is intact: no JEPLO file, weight or dependency is in this repository or `pixi.toml`.
