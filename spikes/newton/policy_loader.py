"""Load the PhysX-trained Go2 policy for evaluation on Newton (ONNX first, ``.pt`` actor as fallback).

Two containers exist on this stack and only one may load, so the loader tries both and records which
worked -- the same choice ``policy/locomotion/eval/push_recovery_eval.py`` documents:

* ``.onnx``: rsl-rl 5 exports with a **fixed batch of 1** and possibly external weights, so rows are fed
  one at a time. Native (numpy/onnxruntime) code, no rsl-rl needed.
* ``.pt``: rsl-rl 5 split format (``actor_state_dict``). The actor is a plain MLP, so it is rebuilt from
  the state-dict shapes rather than through ``OnPolicyRunner`` (which would need a vec-env wrapper and
  the agent cfg, and would drag the training stack into an inference path).

Neither path applies an observation normaliser: the contract says the policy has none, and the loader
reports any normaliser tensors it finds instead of silently applying them.
"""

from __future__ import annotations

import re
from pathlib import Path


def _find_actor_state(ckpt: dict) -> tuple[dict, str]:
    if "actor_state_dict" in ckpt:
        return ckpt["actor_state_dict"], "actor_state_dict"
    if "model_state_dict" in ckpt:
        state = {k[len("actor."):]: v for k, v in ckpt["model_state_dict"].items() if k.startswith("actor.")}
        if state:
            return state, "model_state_dict[actor.*]"
        return ckpt["model_state_dict"], "model_state_dict"
    raise KeyError(f"no actor weights in the checkpoint; top-level keys: {sorted(ckpt)[:12]}")


def _mlp_spec(state: dict) -> tuple[str, list[int]]:
    """``(prefix, [in_dim, hidden..., out_dim])`` from the linear-layer shapes in a state dict."""
    pattern = re.compile(r"^(?P<prefix>.*?)(?P<idx>\d+)\.weight$")
    layers: dict[int, tuple[str, int, int]] = {}
    for key, value in state.items():
        match = pattern.match(key)
        if match is None or getattr(value, "ndim", 0) != 2:
            continue
        index = int(match.group("idx"))
        layers[index] = (match.group("prefix"), int(value.shape[1]), int(value.shape[0]))
    if not layers:
        raise KeyError(f"no linear layers found in the actor state dict; keys: {sorted(state)[:12]}")
    prefix = layers[min(layers)][0]
    if any(meta[0] != prefix for meta in layers.values()):
        raise KeyError(f"linear layers use more than one prefix: {sorted({m[0] for m in layers.values()})}")
    dims = [layers[i][1] for i in sorted(layers)] + [layers[max(layers)][2]]
    return prefix, dims


def load_pt_actor(path: str, device: str = "cpu"):
    """Rebuild the actor MLP from a checkpoint and return ``(policy, info)``."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"{path}: expected a checkpoint dict, got {type(ckpt).__name__}")
    state, source_key = _find_actor_state(ckpt)
    prefix, dims = _mlp_spec(state)
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers)
    remapped = {}
    for key, value in state.items():
        match = re.match(rf"^{re.escape(prefix)}\d+\.(weight|bias)$", key)
        if match:
            remapped[f"{key[len(prefix):]}"] = value
    missing, unexpected = actor.load_state_dict(remapped, strict=False)
    if missing:
        raise RuntimeError(f"{path}: actor rebuild missing {missing}")
    actor.to(device).eval()
    normaliser_keys = [k for k in ckpt if "normal" in k.lower()]

    def policy(obs):
        obs = obs.detach().to(device=device, dtype=torch.float32) if hasattr(obs, "detach") \
            else torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            return actor(obs)

    info = {
        "source": "pt",
        "checkpoint": str(path),
        "actor_state_key": source_key,
        "mlp_prefix": prefix,
        "mlp_dims": dims,
        "activation": "elu (assumed for hidden layers; matches rsl-rl's MLPModel)",
        "normaliser_keys": normaliser_keys,
        "top_level_keys": sorted(ckpt),
        "iteration": ckpt.get("iter"),
    }
    return policy, info


def load_onnx_policy(path: str, device: str = "cuda:0"):
    from pathlib import Path as _Path

    import numpy as np
    import onnxruntime as ort
    import torch

    sidecar = _Path(str(path) + ".data")
    if sidecar.is_file():
        print(f"[policy] external weights present: {sidecar}", flush=True)
    providers = ["CPUExecutionProvider"] if device.startswith("cpu") else ["CUDAExecutionProvider",
                                                                          "CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), providers=providers)
    inputs = [i.name for i in session.get_inputs()]
    if len(inputs) != 1:
        raise RuntimeError(f"{path}: expected one ONNX input, got {inputs}")
    input_name = inputs[0]
    shape = session.get_inputs()[0].shape
    output_names = [o.name for o in session.get_outputs()]
    fixed_batch = shape[0] if shape and isinstance(shape[0], int) else None

    def policy(obs):
        obs_np = np.asarray(obs.detach().cpu() if hasattr(obs, "detach") else obs, dtype=np.float32)
        if fixed_batch is None or obs_np.shape[0] == fixed_batch:
            out = session.run(output_names, {input_name: obs_np})[0]
        else:
            out = np.concatenate([session.run(output_names, {input_name: obs_np[i:i + 1]})[0]
                                  for i in range(obs_np.shape[0])], axis=0)
        return torch.as_tensor(out, dtype=torch.float32, device=getattr(obs, "device", "cpu"))

    info = {"source": "onnx", "checkpoint": str(path), "input": input_name, "input_shape": list(shape),
            "outputs": output_names, "fixed_batch": fixed_batch,
            "providers": session.get_providers()}
    return policy, info


def load_policy(path: str, device: str = "cuda:0"):
    """Dispatch on the file suffix; returns ``(policy, info)``."""
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    if str(path).endswith(".onnx"):
        return load_onnx_policy(path, device)
    return load_pt_actor(path, "cpu")
