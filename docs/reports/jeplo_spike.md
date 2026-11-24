# JEPLO spike — results

JEPLO (Yuan et al., arXiv:2609.15770) is the authors' released LiDAR locomotion policy for the Unitree Go2.
Their repository is at `third_party/jeplo` (GPL-3.0, not committed); our adapter is `plugins/jeplo/` (own GPL-3.0
LICENSE, never imported by the core). This report records what ran, what did not, and the numbers.

## J1 Contract (extracted from their repo)

- Stack: Isaac Lab / Isaac Sim 2.3.2.post1, Python 3.11, torch 2.7.0. Their LiDAR is an Isaac Lab `RayCaster`
  with a spherical-slice pattern, not RTX LiDAR.
- Rates: sim 200 Hz, decimation 4, **policy 50 Hz**; the **sensor estimator (JEPA) 10 Hz** (depth refreshed every
  5 policy steps).
- Observation (actor): 45-dim proprioception x 10 history frames (450) plus a 2x25x60 depth stack. Joint order is
  FL/FR/RL/RR hip, thigh, calf; default pose 0.1/-0.1 hips, 0.8/1.0 thighs, -1.5 calves. Scales: ang-vel 0.25,
  commands 2.0/0.25, joint pos 1.0, joint vel 0.05, actions 0.1. Action scale 0.25; PD 25 / 0.5 in sim.
- Commands (training): vx 0.5-1.0 m/s, vy 0.5-1.0 m/s, wz +-2.0 rad/s; resampled every 4 s. Deployment clips are
  wider (+-2.5 / +-1.5 / +-3.0) but the training distribution is what matters for zero-shot.
- LiDAR: 25 vertical rays over -4.5..45.5 deg, 60 horizontal over -60..60 deg, max range 2.0 m, noise 0.005 m,
  bottom 5 % dropout, 10-frame accumulation, published at 10 Hz as depth/2.0 in [0,1].
- Checkpoints: `policy.onnx` (proprio_hist 450 + high_feat 512 -> 12 actions) and `sensor_estimator.onnx`
  (prop_hist 450 + depth_stack 2x25x60 + hidden_in 512 -> feature 512, hidden_out 512).
- Their ray caster targets only `/World/ground`; our scene needs the house geometry, so we ray-cast the scan proxy
  (`/World/scan_proxy`, ceilings removed) instead.

## J2 Sensor (our implementation)

- `plugins/jeplo/sensor.py` builds the same 25x60 grid from Isaac Lab ray hits, with the authors' sensor offset
  (0.275, 0, 0.148) and 180-degree x rotation, 2.0 m range, Gaussian noise, bottom-row dropout, 10-frame
  accumulation, 10 Hz publication. Training-only random occlusion masks are **not** added (deployment has none).
- Numeric validation against a direct NumPy port of their projection on a synthetic room (floor, walls, a front
  wall): per-pixel agreement is exact, max |error| 0.0 m, 0 disagreeing bins; 221 of 1500 bins filled. Figure:
  `outputs/scratch/jeplo_sensor_validation.png`. Honest limitation: this is geometry-level, not a run of their Isaac
  Lab 2.3 simulator.

## J3 Adapter

- `plugins/jeplo/backend.py` implements the B1 `PolicyBackend`: 10-frame proprio history, 2-frame depth stack, a
  512-dim hidden state carried between 10 Hz estimator calls, and raw 12 actions converted to
  `default + 0.25 * action` joint targets. Commands pass through the shipped contract rate limiter.
- Contract tests (`plugins/jeplo/tests/test_contract.py`): ONNX shapes load, 100 steps -> 20 estimator calls,
  finite actions, targets for all 12 joints, a `height_scan` key is **rejected**, commands never exceed the
  contract range. All pass locally.

## J4 In our scene

PENDING — the zero-shot walk is the decisive test and is running now. Results and the honest outcome (including a
failure) go here.
