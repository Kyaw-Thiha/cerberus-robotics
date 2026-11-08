#!/usr/bin/env python3
"""Check that a checkpoint's ONNX export really is the same function as its actor.

The export is what the ROS node and any non-rsl-rl consumer will run, and an ONNX graph can load, have
the right shapes, and still be built from the wrong weights or the wrong activation. This compares the
two directly on random observations.

It needs ``torch`` and ``onnxruntime`` but NOT Isaac Lab, so it runs on a laptop:

    python3 -m venv /tmp/onnxcheck && /tmp/onnxcheck/bin/pip install torch --index-url \\
        https://download.pytorch.org/whl/cpu && /tmp/onnxcheck/bin/pip install onnxruntime
    /tmp/onnxcheck/bin/python scripts/verify_onnx_matches_checkpoint.py \\
        --checkpoint policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate.pt \\
        --onnx policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate.onnx

The actor is rebuilt from the checkpoint's own ``actor_state_dict`` shapes (the rsl-rl 5 split layout),
so no rsl-rl install is needed either. Tolerance is float32 noise (1e-4 by default).

Exit codes: 0 match, 1 mismatch, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_TOLERANCE = 1e-4


def rebuild_actor(state_dict, activation: str = "elu"):
    """nn.Sequential reconstructed from ``mlp.<i>.{weight,bias}`` keys."""
    import torch

    acts = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh, "gelu": torch.nn.GELU}
    if activation not in acts:
        raise ValueError(f"unsupported activation {activation!r}; expected one of {sorted(acts)}")
    layers = []
    index = 0
    while f"mlp.{index}.weight" in state_dict:
        weight = state_dict[f"mlp.{index}.weight"]
        layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
        layers.append(acts[activation]())
        index += 2  # weight/bias pairs sit at even indices, activations between them
    if not layers:
        raise KeyError("no mlp.<i>.weight keys in the state dict")
    layers.pop()  # the final activation belongs to the distribution head, not the exported actor
    net = torch.nn.Sequential(*layers)
    for layer_index in range(0, len(layers), 2):
        with torch.no_grad():
            net[layer_index].weight.copy_(state_dict[f"mlp.{layer_index}.weight"])
            net[layer_index].bias.copy_(state_dict[f"mlp.{layer_index}.bias"])
    net.eval()
    return net


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--activation", default="elu")
    args = parser.parse_args(argv)

    import numpy as np
    import onnxruntime as ort
    import torch

    loaded = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        print(f"[verify] {args.checkpoint} is not a checkpoint dict", file=sys.stderr)
        return 2
    state_dict = loaded.get("actor_state_dict") or loaded.get("model_state_dict")
    if state_dict is None:
        print(f"[verify] no actor_state_dict/model_state_dict in {args.checkpoint}", file=sys.stderr)
        return 2

    net = rebuild_actor(state_dict, args.activation)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        print(f"[verify] expected one ONNX input, got {[i.name for i in inputs]}", file=sys.stderr)
        return 2
    input_name = inputs[0].name
    obs_dim = net[0].in_features
    declared = inputs[0].shape[-1]
    if isinstance(declared, int) and declared != obs_dim:
        print(f"[verify] observation dim mismatch: checkpoint {obs_dim}, ONNX {declared}", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    worst = 0.0
    for trial in range(args.trials):
        obs = rng.normal(size=(1, obs_dim)).astype(np.float32)
        with torch.no_grad():
            expected = net(torch.from_numpy(obs)).numpy()
        actual = session.run(None, {input_name: obs})[0]
        if expected.shape != actual.shape:
            print(f"[verify] shape mismatch: torch {expected.shape} vs onnx {actual.shape}", file=sys.stderr)
            return 1
        worst = max(worst, float(np.max(np.abs(expected - actual))))

    # A graph that ignores its input would pass an all-zeros comparison; make sure it does not.
    zero = session.run(None, {input_name: np.zeros((1, obs_dim), np.float32)})[0]
    ones = session.run(None, {input_name: np.ones((1, obs_dim), np.float32)})[0]
    responds = float(np.max(np.abs(zero - ones))) > 1e-6

    ok = worst <= args.tolerance and responds
    print(f"[verify] worst |torch - onnx| = {worst:.3e} (tolerance {args.tolerance:g}), "
          f"input-responsive={responds} -> {'MATCH' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
