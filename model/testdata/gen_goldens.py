#!/usr/bin/env python3
"""Generate JSON golden fixtures for lmkit-go model block parity tests.

Runs on trig: `~/venvs/cuda/bin/python gen_goldens.py` (the Mac has no torch).
Requires torch (CPU tensors only). Deterministic (seeded). Each block writes
<block>.json with {config, inputs, weights, expected} as float32 row-major.
"""
import json, math, torch

torch.manual_seed(0)

def t2j(x):
    x = x.detach().to(torch.float32).contiguous()
    return {"shape": list(x.shape), "data": x.flatten().tolist()}

def write(name, config, inputs, weights, expected):
    obj = {"config": config,
           "inputs": {k: t2j(v) for k, v in inputs.items()},
           "weights": {k: t2j(v) for k, v in weights.items()},
           "expected": t2j(expected)}
    with open(f"{name}.json", "w") as f:
        json.dump(obj, f)
    print(f"wrote {name}.json")

def gen_rmsnorm():
    B, T, H, eps = 2, 3, 8, 1e-5
    x = torch.randn(B, T, H)
    scale = torch.randn(H)
    ms = x.pow(2).mean(-1, keepdim=True)
    y = x * torch.rsqrt(ms + eps) * scale
    write("rmsnorm", {"hidden": H, "rms_eps": eps},
          {"x": x}, {"scale": scale}, y)

if __name__ == "__main__":
    gen_rmsnorm()
