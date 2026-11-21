"""DiT-B: 132M-parameter diffusion transformer on 256x256 frames (32x32 patches, width 768, depth 12, 12 heads, 3D RoPE,
adaLN-zero conditioning on the noise level, no text input); Sec. C.2 of the supplement."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

PATCH, SIZE = 32, 256
GRID = SIZE // PATCH
TOK = GRID * GRID
PDIM = PATCH * PATCH * 3
torch._dynamo.config.cache_size_limit = 64
flex = torch.compile(flex_attention, dynamic=False)


def patchify(x: torch.Tensor) -> torch.Tensor:
    b, f = x.shape[:2]
    x = x.reshape(b, f, 3, GRID, PATCH, GRID, PATCH).permute(0, 1, 3, 5, 4, 6, 2)
    return x.reshape(b, f * TOK, PDIM)


def unpatchify(t: torch.Tensor, f: int) -> torch.Tensor:
    b = t.shape[0]
    t = t.reshape(b, f, GRID, GRID, PATCH, PATCH, 3).permute(0, 1, 6, 2, 4, 3, 5)
    return t.reshape(b, f, 3, SIZE, SIZE)


def timestep_embedding(sigma: torch.Tensor, dim: int = 256) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=sigma.device, dtype=torch.float32) / half)
    args = (sigma.float() * 1000)[..., None] * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Rope3D(nn.Module):
    def __init__(self, head_dim: int = 64, splits=(24, 20, 20), max_frames: int = 160):
        super().__init__()
        self.splits = splits
        tabs = []
        for d, n in zip(splits, (max_frames, GRID, GRID)):
            inv = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
            tabs.append(torch.outer(torch.arange(n).float(), inv))
        self.f, self.r, self.c = (nn.Buffer(t, persistent=False) for t in tabs)

    def angles(self, frame_ids: torch.Tensor) -> torch.Tensor:
        L = frame_ids.numel()
        rows = torch.arange(TOK, device=frame_ids.device) // GRID
        cols = torch.arange(TOK, device=frame_ids.device) % GRID
        rr = rows.repeat(L // TOK); cc = cols.repeat(L // TOK)
        return torch.cat([self.f[frame_ids], self.r[rr], self.c[cc]], dim=-1)

    @staticmethod
    def apply(x: torch.Tensor, cs) -> torch.Tensor:
        cos, sin = cs
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.flatten(-2)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3)
        self.qnorm = nn.RMSNorm(dim // heads); self.knorm = nn.RMSNorm(dim // heads)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(approximate="tanh"), nn.Linear(dim * 4, dim))
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, x, c, cs, block_mask, kv=None):
        b, L, d = x.shape
        fr = L // TOK
        s1, b1, g1, s2, b2, g2 = self.ada(c).chunk(6, dim=-1)
        h = (self.norm1(x).view(b, fr, TOK, d) * (1 + s1) + b1).view(b, L, d)
        q, k, v = self.qkv(h).reshape(b, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k = Rope3D.apply(self.qnorm(q), cs), Rope3D.apply(self.knorm(k), cs)
        q, k = q.to(v.dtype), k.to(v.dtype)
        if kv is None:
            a = flex(q, k, v, block_mask=block_mask) if block_mask is not None else F.scaled_dot_product_attention(q, k, v)
        else:
            a = F.scaled_dot_product_attention(q, torch.cat([kv[0], k], 2), torch.cat([kv[1], v], 2))
        x = x + (g1 * self.proj(a.transpose(1, 2).reshape(b, L, d)).view(b, fr, TOK, d)).view(b, L, d)
        h = (self.norm2(x).view(b, fr, TOK, d) * (1 + s2) + b2).view(b, L, d)
        x = x + (g2 * self.mlp(h).view(b, fr, TOK, d)).view(b, L, d)
        return x, (k, v)


class DiT(nn.Module):
    def __init__(self, dim: int = 768, depth: int = 12, heads: int = 12, bottleneck: int = 128):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(PDIM, bottleneck, bias=False), nn.Linear(bottleneck, dim))
        self.temb = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.rope = Rope3D(dim // heads)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, PDIM)
        nn.init.zeros_(self.final_ada.weight); nn.init.zeros_(self.final_ada.bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, tokens: torch.Tensor, sigma_tok: torch.Tensor, frame_ids: torch.Tensor, block_mask,
                cache=None, return_kv: bool = False):
        b, L, _ = tokens.shape
        fr = L // TOK
        x = self.embed(tokens)
        c = F.silu(self.temb(timestep_embedding(sigma_tok.view(b, fr, TOK)[:, :, :1])))
        ang = self.rope.angles(frame_ids)
        cs = (ang.cos()[None, None], ang.sin()[None, None])
        kvs = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, c, cs, block_mask, None if cache is None else cache[i])
            kvs.append(kv)
        s, b_ = self.final_ada(c).chunk(2, dim=-1)
        d = x.shape[-1]
        out = self.out((self.final_norm(x).view(b, fr, TOK, d) * (1 + s) + b_).view(b, L, d))
        return (out, kvs) if return_kv else out
