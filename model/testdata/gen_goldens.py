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

def gen_embedding():
    V, H, B, T = 10, 8, 2, 3
    table = torch.randn(V, H)
    ids = torch.randint(0, V, (B, T), dtype=torch.int32)
    embed = table[ids.long()]
    h = torch.randn(B, T, H)
    logits = h @ table.t()
    obj = {"config": {"vocab": V, "hidden": H},
           "inputs": {"ids": t2j(ids), "h": t2j(h)},
           "weights": {"table": t2j(table)},
           "expected_embed": t2j(embed),
           "expected_logits": t2j(logits)}
    with open("embedding.json", "w") as f:
        json.dump(obj, f)
    print("wrote embedding.json")

def gen_attention():
    B, T, H = 2, 4, 8
    nH, nKV, hd, base = 4, 2, 2, 10000.0   # H == nH*hd; nH % nKV == 0
    x = torch.randn(B, T, H)
    Wq = torch.randn(H, nH * hd); Wk = torch.randn(H, nKV * hd)
    Wv = torch.randn(H, nKV * hd); Wo = torch.randn(nH * hd, H)
    positions = torch.arange(T)
    inv_freq = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    emb = torch.cat([positions[:, None].float() * inv_freq[None, :]] * 2, dim=-1)
    cos = emb.cos()[None, :, None, :]; sin = emb.sin()[None, :, None, :]
    def rot(t):
        a, b = t[..., : hd // 2], t[..., hd // 2 :]
        return torch.cat([-b, a], dim=-1)
    q = (x @ Wq).view(B, T, nH, hd)
    k = (x @ Wk).view(B, T, nKV, hd)
    v = (x @ Wv).view(B, T, nKV, hd)
    q = q * cos + rot(q) * sin
    k = k * cos + rot(k) * sin
    rep = nH // nKV
    k = k.repeat_interleave(rep, dim=2)   # [B,T,nH,hd]
    v = v.repeat_interleave(rep, dim=2)
    q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)  # [B,nH,T,hd]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd)                   # [B,nH,T,T]
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    probs = (scores + mask).softmax(-1)
    out = (probs @ v).transpose(1, 2).reshape(B, T, nH * hd)             # [B,T,nH*hd]
    y = out @ Wo                                                         # [B,T,H]
    write("attention",
          {"hidden": H, "n_heads": nH, "n_kv_heads": nKV, "head_dim": hd,
           "rope_base": base, "seq_len": T},
          {"x": x}, {"Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo}, y)

import torch.nn.functional as F

def _rmsnorm(x, scale, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * scale

def _rope(x, hd, base):                       # x [B,T,nheads,hd]
    T = x.shape[1]
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    emb = torch.cat([torch.arange(T)[:, None].float() * inv[None, :]] * 2, dim=-1)
    cos = emb.cos()[None, :, None, :]; sin = emb.sin()[None, :, None, :]
    a, b = x[..., : hd // 2], x[..., hd // 2 :]
    return x * cos + torch.cat([-b, a], dim=-1) * sin

def _attention(x, w, nH, nKV, hd, base):
    B, T, _ = x.shape
    q = (x @ w["Wq"]).view(B, T, nH, hd); k = (x @ w["Wk"]).view(B, T, nKV, hd); v = (x @ w["Wv"]).view(B, T, nKV, hd)
    q = _rope(q, hd, base); k = _rope(k, hd, base)
    rep = nH // nKV
    k = k.repeat_interleave(rep, dim=2); v = v.repeat_interleave(rep, dim=2)
    q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
    s = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
    s = s + torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, nH * hd)
    return o @ w["Wo"]

def _swiglu(x, w):
    return (F.silu(x @ w["Wg"]) * (x @ w["Wu"])) @ w["Wd"]

def _decoder_layer(h, w, cfg):
    h = h + _attention(_rmsnorm(h, w["attn_norm"], cfg["eps"]), w, cfg["nH"], cfg["nKV"], cfg["hd"], cfg["base"])
    h = h + _swiglu(_rmsnorm(h, w["ffn_norm"], cfg["eps"]), w)
    return h

def _layer_weights(H, nH, nKV, hd, ffn):
    return {"attn_norm": torch.randn(H), "Wq": torch.randn(H, nH*hd), "Wk": torch.randn(H, nKV*hd),
            "Wv": torch.randn(H, nKV*hd), "Wo": torch.randn(nH*hd, H), "ffn_norm": torch.randn(H),
            "Wg": torch.randn(H, ffn), "Wu": torch.randn(H, ffn), "Wd": torch.randn(ffn, H)}

def gen_decoder_layer():
    torch.manual_seed(42)
    B, T, H, nH, nKV, hd, ffn = 2, 4, 8, 4, 2, 2, 16
    cfg = {"nH": nH, "nKV": nKV, "hd": hd, "base": 10000.0, "eps": 1e-5}
    h = torch.randn(B, T, H)
    w = _layer_weights(H, nH, nKV, hd, ffn)
    y = _decoder_layer(h, w, cfg)
    weights = {"attn_norm": w["attn_norm"], "Wq": w["Wq"], "Wk": w["Wk"], "Wv": w["Wv"],
               "Wo": w["Wo"], "ffn_norm": w["ffn_norm"], "Wgate": w["Wg"], "Wup": w["Wu"], "Wdown": w["Wd"]}
    write("decoder", {"hidden": H, "n_heads": nH, "n_kv_heads": nKV, "head_dim": hd,
                      "ffn_hidden": ffn, "rope_base": 10000.0, "rms_eps": 1e-5, "seq_len": T},
          {"h": h}, weights, y)

if __name__ == "__main__":
    gen_rmsnorm()
    gen_rope()
    gen_swiglu()
    gen_embedding()
    gen_attention()
    gen_decoder_layer()
