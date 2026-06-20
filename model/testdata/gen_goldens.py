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

def gen_rope():
    B, T, nH, hd, base = 2, 4, 2, 8, 10000.0
    x = torch.randn(B, T, nH, hd)
    positions = torch.arange(T)
    inv_freq = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))   # [hd/2]
    freqs = positions[:, None].float() * inv_freq[None, :]            # [T, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)                          # [T, hd]
    cos = emb.cos()[None, :, None, :]                                # [1,T,1,hd]
    sin = emb.sin()[None, :, None, :]
    def rotate_half(t):
        a, b = t[..., : hd // 2], t[..., hd // 2 :]
        return torch.cat([-b, a], dim=-1)
    y = x * cos + rotate_half(x) * sin
    write("rope", {"head_dim": hd, "rope_base": base, "seq_len": T},
          {"x": x}, {}, y)

def gen_swiglu():
    import torch.nn.functional as F
    B, T, H, F_ = 2, 3, 8, 16
    x = torch.randn(B, T, H)
    Wg = torch.randn(H, F_); Wu = torch.randn(H, F_); Wd = torch.randn(F_, H)
    y = (F.silu(x @ Wg) * (x @ Wu)) @ Wd
    write("swiglu", {"hidden": H, "ffn_hidden": F_},
          {"x": x}, {"Wg": Wg, "Wu": Wu, "Wd": Wd}, y)

if __name__ == "__main__":
    gen_rmsnorm()
    gen_rope()
    gen_swiglu()
