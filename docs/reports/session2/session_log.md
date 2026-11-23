# Session 2 log (arm first)

All times UTC unless stated. One line per decision.

2026-11-23T07:30Z preflight: keys present (HYP=1 R2=5 NGC=0), gh ok, disk 49G free, no VM, no kill timers
2026-11-23T07:38Z DECISION: pick-image skipped R595 (now hard-restricted to B300); patched pick_image + tests; created VM on R570 image
2026-11-23T07:38Z FIX: pick-image now skips hard flavor-restricted images (R595 is B300-only); tests added
2026-11-23T08:05Z VM 1177066 up, image R570; 35GB pull done, 25GB scene fetched; driver 570->595.58.03 installed, nouveau was holding the GPU (blacklisted, reboot)
2026-11-23T08:12Z K3 harness: scripts/host/run_until_complete.py + atomic_io + crash tests pass; clip_qa.py + 10 tests pass
2026-11-23T08:20Z LOOK ROOT CAUSE: all three scene variants (navigation/raw/interaction) use MDL shaders: 3282
  mdl:surface shader prims, 5013 asset inputs, 0 missing files on disk; MDL_SYSTEM_PATH empty because
  run_session.sh sets CERBERUS_SKIP_MDL_PATH=1. The flat red-orange is an unresolved MDL material, not a broken USD.
2026-11-23T08:25Z K1: launch failures on this host are the loader crash ('A crash has occurred', 14-15 s,
  exit -9) and occasionally a 36-80 s exit 1; crash dumps land in /isaac-sim/kit/data/Kit/IsaacLab/3.0/*.dmp.zip;
  dmesg shows Xid 43 once, after which torch could not see the GPU until a reboot.
2026-11-23T08:30Z FIX (our code, 4 bugs): observation asset_cfg ordering (fused USD is per-leg, the contract is
  per-part -> preserve_order=True with the contract list); event env_ids can be a slice; joint_pos dtype is a
  type() not a torch.dtype; train.py read args_cli.adversarial_env which the parser never defines.
2026-11-23T08:55Z BLOCKED (logged, not tuned): the arm randomisation events (per-episode joint offsets, payload
  mass) trip a PhysX device-side assert at the reset/startup event on this stack. Set/add methods, body_mass source
  and inertia consistency were all tried (3 hypotheses). Retrain runs with CERBERUS_ARM_EVENTS=0: the arm is held
  stowed by its own PD actuators, which is what the policy contract needs; offsets/payload randomisation is not
  applied and is reported as such.
