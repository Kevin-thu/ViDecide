"""DiT-B (132M parameters, pixel space) trained from scratch on FlowFree and Sokoban (Secs. 4.2-4.3 and C.2).

Modes: ``train`` (warmup, then constant learning rate), ``anneal`` (linear decay to zero, resumed from the full state
of ``train``) and ``eval`` (sampling with the rule verifiers). Arms: k1 / k8 (causal generation with K=1 / K=8, ours),
bidir (fixed-length bidirectional generation) and df (diffusion forcing, Tab. S7). Timesteps follow a logit-normal
distribution with mean --mu (log s of the training shift s; 2.303 for s=10). The training data are the pixel caches
of scripts/dit/build_cache.py in data/dit/ (--data-root to change).

Usage (see configs/dit/ for the paper settings):
    torchrun --standalone --nproc_per_node=1 scripts/dit/train.py train --config configs/dit/sokoban_k1.yaml --out OUT
    torchrun --standalone --nproc_per_node=1 scripts/dit/train.py anneal --config configs/dit/sokoban_k1.yaml \
        --out OUT_ANNEAL --resume OUT/step_010000/full_state.pt
    python scripts/dit/train.py eval --config configs/dit/sokoban_k1.yaml --out OUT_EVAL \
        --ckpt OUT_ANNEAL/step_011000/ema.pt [--eval-set hard_test]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dit_model import DiT, TOK, patchify, unpatchify
from videcide.config import parse_args_with_config
from videcide.paths import DATA_ROOT, data_path

DATA = DATA_ROOT / "dit"                   # pixel caches of scripts/dit/build_cache.py (--data-root)
# Clip length (frames) of the bidirectional baseline; longer training videos are left out of its training set.
FIXED = {"sokoban": 42, "flowfree": 67}
HIST_SIGMA = 0.0   # eval --hist-sigma: generated history enters the KV cache re-noised to this level (diffusion forcing)
CLEAN, NOISY = 0, 1


class Episodes:
    """Expert videos of one or more sets of a task in the pixel cache (DATA/<task>/<set>_frames.npy, _index.json).

    Videos longer than ``max_length`` frames are left out.
    """

    def __init__(self, task: str, sets: str | list[str], max_length: int | None = None):
        sets = sets.split(",") if isinstance(sets, str) else list(sets)
        self.frames, self.index = {}, []
        for name in sets:
            self.frames[name] = np.load(DATA / task / f"{name}_frames.npy", mmap_mode="r")
            self.index += [dict(r, set=name) for r in json.loads((DATA / task / f"{name}_index.json").read_text())
                           if max_length is None or r["length"] <= max_length]

    def __len__(self):
        return len(self.index)

    def get(self, i: int) -> np.ndarray:
        r = self.index[i]
        return np.asarray(self.frames[r["set"]][r["start"]:r["start"] + r["length"]])


def source_of(row: dict) -> dict:
    """Puzzle source of a cache row (its path is relative to the data root)."""
    return json.loads(data_path(row["source"]).read_text())


def to_tensor(frames: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float().div(127.5).sub(1)


def layout(slots: list[tuple[int, int]], device):
    copy = torch.tensor([c for c, _ in slots], device=device).repeat_interleave(TOK)
    fid = torch.tensor([f for _, f in slots], device=device).repeat_interleave(TOK)
    return copy, fid


_MASKS = {}
KBLOCK = 1


def block_of(f):
    return (f + KBLOCK - 1) // KBLOCK


def causal_mask(copy, fid, lens=None, B=None):
    L = copy.numel()
    key = (L, str(copy.device), KBLOCK)
    if key not in _MASKS:
        def mod(b, h, q, kv):
            qc, kc, qf, kf = copy[q], copy[kv], fid[q], fid[kv]
            clean_q = (qc == CLEAN) & (kc == CLEAN) & (kf <= qf)
            if KBLOCK == 1:
                noisy_q = (qc == NOISY) & (((kc == CLEAN) & (kf < qf)) | ((kc == NOISY) & (kf == qf)))
            else:
                bq, bk = (qf + KBLOCK - 1) // KBLOCK, (kf + KBLOCK - 1) // KBLOCK
                noisy_q = (qc == NOISY) & (((kc == CLEAN) & (bk < bq)) | ((kc == NOISY) & (bk == bq)))
            return clean_q | noisy_q
        _MASKS[key] = create_block_mask(mod, B=None, H=None, Q_LEN=L, KV_LEN=L, device=copy.device)
    return _MASKS[key]


def df_mask(fid):
    L = fid.numel()
    key = ("df", L, str(fid.device))
    if key not in _MASKS:
        def mod(b, h, q, kv):
            return fid[kv] <= fid[q]
        _MASKS[key] = create_block_mask(mod, B=None, H=None, Q_LEN=L, KV_LEN=L, device=fid.device)
    return _MASKS[key]


def sample_sigma(shape, mu, device, std=1.0):
    return torch.sigmoid(mu + std * torch.randn(shape, device=device))


def k1_loss(model, batch: list[np.ndarray], mu: float, device):
    B = len(batch)
    Fm = -(-max(len(v) for v in batch) // 8) * 8
    x = torch.zeros(B, Fm, 3, 256, 256, device=device)
    for i, v in enumerate(batch):
        x[i, :len(v)] = to_tensor(v, device)
    lens = torch.tensor([len(v) for v in batch], device=device)
    slots = [(CLEAN, f) for f in range(Fm)] + [(NOISY, f) for f in range(1, Fm)]
    copy, fid = layout(slots, device)
    sig = sample_sigma((B, Fm - 1), mu, device)
    if KBLOCK > 1:
        blk = torch.tensor([block_of(f) for f in range(1, Fm)], device=device) - 1
        sig = sample_sigma((B, int(blk.max()) + 1), mu, device)[:, blk]
    clean = x; target = x[:, 1:]
    eps = torch.randn_like(target)
    z = (1 - sig[..., None, None, None]) * target + sig[..., None, None, None] * eps
    tokens = torch.cat([patchify(clean), patchify(z)], dim=1)
    sig_tok = torch.cat([torch.zeros(B, Fm * TOK, device=device), sig.repeat_interleave(TOK, dim=1)], dim=1)
    mask = causal_mask(copy, fid, lens, B)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xh = model(tokens, sig_tok, fid, mask)[:, Fm * TOK:].float()
    err = ((patchify(target) - xh) ** 2).mean(-1).reshape(B, Fm - 1, TOK).mean(-1)
    err = err / sig.clamp(min=0.05) ** 2
    valid = (torch.arange(1, Fm, device=device)[None] < lens[:, None]).float()
    return (err * valid).sum() / valid.sum()


def df_loss(model, batch: list[np.ndarray], mu: float, device):
    B = len(batch)
    Fm = -(-max(len(v) for v in batch) // 8) * 8
    x = torch.zeros(B, Fm, 3, 256, 256, device=device)
    for i, v in enumerate(batch):
        x[i, :len(v)] = to_tensor(v, device)
    lens = torch.tensor([len(v) for v in batch], device=device)
    fid = torch.arange(Fm, device=device).repeat_interleave(TOK)
    sig = sample_sigma((B, Fm), mu, device)
    sig[:, 0] = 0.0
    eps = torch.randn_like(x)
    z = (1 - sig[..., None, None, None]) * x + sig[..., None, None, None] * eps
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xh = model(patchify(z), sig.repeat_interleave(TOK, dim=1), fid, df_mask(fid))[:, TOK:].float()
    err = ((patchify(x[:, 1:]) - xh) ** 2).mean(-1).reshape(B, Fm - 1, TOK).mean(-1)
    err = err / sig[:, 1:].clamp(min=0.05) ** 2
    valid = (torch.arange(1, Fm, device=device)[None] < lens[:, None]).float()
    return (err * valid).sum() / valid.sum()


def bidir_pad(v: np.ndarray, L: int) -> np.ndarray:
    return np.concatenate([v, np.repeat(v[-1:], L - len(v), axis=0)]) if len(v) < L else v


def bidir_loss(model, batch: list[np.ndarray], mu: float, device, L: int):
    B = len(batch)
    x = torch.stack([to_tensor(bidir_pad(v, L), device) for v in batch])
    slots = [(CLEAN, 0)] + [(NOISY, f) for f in range(1, L)]
    _, fid = layout(slots, device)
    sig = sample_sigma((B,), mu, device)
    target = x[:, 1:]; eps = torch.randn_like(target)
    z = (1 - sig[:, None, None, None, None]) * target + sig[:, None, None, None, None] * eps
    tokens = torch.cat([patchify(x[:, :1]), patchify(z)], dim=1)
    sig_tok = torch.cat([torch.zeros(B, TOK, device=device), sig[:, None].expand(B, (L - 1) * TOK)], dim=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xh = model(tokens, sig_tok, fid, None)[:, TOK:].float()
    err = ((patchify(target) - xh) ** 2).mean(-1).mean(-1) / sig.clamp(min=0.05) ** 2
    return err.mean()


def arm_loss(a, model, batch, device, L):
    if a.arm in ("k1", "k8"):
        return k1_loss(model, batch, a.mu, device)
    if a.arm == "df":
        return df_loss(model, batch, a.mu, device)
    return bidir_loss(model, batch, a.mu, device, L)


def length_batches(lengths, batch, rng):
    idx = list(range(len(lengths))); rng.shuffle(idx)
    out = []
    for s in range(0, len(idx), batch * 50):
        chunk = sorted(idx[s:s + batch * 50], key=lambda i: lengths[i])
        out += [chunk[k:k + batch] for k in range(0, len(chunk), batch) if len(chunk[k:k + batch]) == batch]
    rng.shuffle(out)
    return out


def train(a):
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.manual_seed(a.seed + rank)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ds = Episodes(a.task, "train"); val = Episodes(a.task, "val")
    L = FIXED[a.task]
    lengths = [r["length"] for r in ds.index]
    keep = [i for i in range(len(ds)) if a.arm in ("k1", "k8", "df") or lengths[i] <= L]
    model = DiT().to(device)
    ema = DiT().to(device); ema.load_state_dict(model.state_dict()); ema.requires_grad_(False)
    if a.compile:
        for blk in model.blocks:
            blk.compile(dynamic=False)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    step, total = 0, a.steps
    if a.resume:
        st = torch.load(Path(a.resume), map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        step = st["step"]
        total = step + a.anneal_steps if a.mode == "anneal" else a.steps
    base_step = step
    micro = a.batch // (world * a.accum)
    rng = random.Random(a.seed + 1000 * step)
    sub = [lengths[i] for i in keep]
    batches = []
    if rank == 0:
        cfg = {**vars(a), "world": world, "micro": micro, "train_rows": len(keep), "fixed_len": L,
               "params_M": sum(p.numel() for p in model.parameters()) / 1e6}
        (out / f"config_{a.mode}.json").write_text(json.dumps(cfg, indent=1, default=str))
        print(json.dumps(cfg, default=str), flush=True)
    log = open(out / "loss.jsonl", "a") if rank == 0 else None
    t0 = time.time()
    while step < total:
        if a.mode == "train":
            lr = a.lr * min(1.0, (step + 1) / a.warmup)
        else:
            lr = a.lr * max(0.0, 1 - (step - base_step) / a.anneal_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for k in range(a.accum):
            if not batches:
                batches = length_batches(sub, a.batch // a.accum, rng)
            gb = batches.pop()
            mine = [keep[gb[j]] for j in range(rank * micro, (rank + 1) * micro)]
            eps = [ds.get(i) for i in mine]
            loss = arm_loss(a, ddp, eps, device, L) / a.accum
            loss.backward(); tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); step += 1
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        if rank == 0 and step % 10 == 0:
            log.write(json.dumps({"step": step, "time": time.time(), "loss": tot, "grad_norm": float(gn), "lr": lr}) + "\n"); log.flush()
        if step % a.val_every == 0 or step == total:
            vl = val_loss(ema, val, a, device, L)
            if rank == 0:
                with open(out / "val_loss.jsonl", "a") as f:
                    f.write(json.dumps({"step": step, "val_loss": vl}) + "\n")
                print(f"step {step} loss {tot:.4f} val {vl:.4f} {(time.time() - t0) / (step - base_step):.2f}s/step", flush=True)
        if rank == 0 and (step % a.save_every == 0 or step == total):
            d = out / f"step_{step:06d}"; d.mkdir(exist_ok=True)
            torch.save({"ema": ema.state_dict(), "step": step}, d / "ema.pt")
            if step == total:
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step}, d / "full_state.pt")
    dist.barrier(); dist.destroy_process_group()


@torch.no_grad()
def val_loss(ema, val, a, device, L, n=128):
    g = torch.Generator(device="cpu").manual_seed(7)
    rng_state = torch.random.get_rng_state(); cuda_state = torch.cuda.get_rng_state(device)
    torch.manual_seed(7)
    idx = [i for i in range(len(val)) if a.arm in ("k1", "k8", "df") or val.index[i]["length"] <= L][:n]
    tot = 0.0
    for s in range(0, len(idx), 8):
        eps = [val.get(i) for i in idx[s:s + 8]]
        tot += arm_loss(a, ema, eps, device, L).item() * len(eps)
    torch.random.set_rng_state(rng_state); torch.cuda.set_rng_state(cuda_state, device)
    return tot / len(idx)


SAMPLE_SHIFT = 1.0


def euler_grid(steps: int, shift: float | None = None):
    s = SAMPLE_SHIFT if shift is None else shift
    t = torch.linspace(1.0, 0.0, steps + 1)
    return s * t / (1 + (s - 1) * t)


@torch.no_grad()
def sample_k1(model, first: torch.Tensor, steps: int, max_frames: int, is_eos, noise=None, prefix=None):
    device = first.device; B = first.shape[0]
    grid = euler_grid(steps)
    zeros = torch.zeros(B, TOK, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, cache = model(patchify(first[:, None]), zeros, torch.zeros(TOK, dtype=torch.long, device=device), None, return_kv=True)
    frames = [first]; done = torch.zeros(B, dtype=torch.bool); end = [None] * B
    start = 1
    if prefix is not None:
        for j in range(1, prefix.shape[1]):
            fid = torch.full((TOK,), j, dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, kv = model(patchify(prefix[:, j:j + 1]), zeros, fid, None, cache=cache, return_kv=True)
            cache = [(torch.cat([c[0], n[0]], 2), torch.cat([c[1], n[1]], 2)) for c, n in zip(cache, kv)]
            frames.append(prefix[:, j])
        start = prefix.shape[1]
    i = start
    while i < max_frames:
        n = min(KBLOCK - (i - 1) % KBLOCK, max_frames - i)
        fid = torch.arange(i, i + n, device=device).repeat_interleave(TOK)
        z = (torch.cat([noise(i + j) for j in range(n)], 1) if noise is not None
             else torch.randn(B, n, 3, 256, 256, device=device))
        for s0, s1 in zip(grid[:-1], grid[1:]):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                xh = model(patchify(z), torch.full((B, n * TOK), float(s0), device=device), fid, None, cache=cache).float()
            xh = unpatchify(xh, n).clamp(-1, 1)
            z = z + (float(s1) - float(s0)) * (z - xh) / max(float(s0), 1e-4)
        for j in range(n):
            frames.append(z[:, j].clamp(-1, 1))
            fj = torch.full((TOK,), i + j, dtype=torch.long, device=device)
            hist, hsig = frames[-1][:, None], zeros
            if HIST_SIGMA > 0:   # diffusion-forcing sampling: the history enters the cache lightly noised
                hist = (1 - HIST_SIGMA) * hist + HIST_SIGMA * torch.randn_like(hist); hsig = torch.full_like(zeros, HIST_SIGMA)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, kv = model(patchify(hist), hsig, fj, None, cache=cache, return_kv=True)
            cache = [(torch.cat([c[0], m[0]], 2), torch.cat([c[1], m[1]], 2)) for c, m in zip(cache, kv)]
            u8 = frames[-1].add(1).mul(127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            for b in range(B):
                if not done[b] and is_eos(u8[b]):
                    done[b] = True; end[b] = i + j + 1
        i += n
        if done.all():
            break
    video = torch.stack(frames, 1)
    return [video[b, :(end[b] or video.shape[1])] for b in range(B)]


@torch.no_grad()
def sample_bidir(model, first: torch.Tensor, steps: int, L: int):
    device = first.device; B = first.shape[0]
    _, fid = layout([(CLEAN, 0)] + [(NOISY, f) for f in range(1, L)], device)
    z = torch.randn(B, L - 1, 3, 256, 256, device=device)
    grid = euler_grid(steps)
    for s0, s1 in zip(grid[:-1], grid[1:]):
        sig_tok = torch.cat([torch.zeros(B, TOK, device=device), torch.full((B, (L - 1) * TOK), float(s0), device=device)], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xh = model(torch.cat([patchify(first[:, None]), patchify(z)], 1), sig_tok, fid, None)[:, TOK:].float()
        xh = unpatchify(xh, L - 1).clamp(-1, 1)
        z = z + (float(s1) - float(s0)) * (z - xh) / max(float(s0), 1e-4)
    video = torch.cat([first[:, None], z.clamp(-1, 1)], 1)
    return [video[b] for b in range(B)]


def evaluate(a):
    from checkers import CHECK
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from videcide.tasks.video_parser import find_eos_frame
    from videcide.tasks.video import write_mp4
    device = torch.device("cuda")
    ds = Episodes(a.task, a.eval_set)
    rows = list(range(len(ds)))
    if a.limit:
        rows = sorted(random.Random(20261007).sample(rows, a.limit))
    global SAMPLE_SHIFT
    SAMPLE_SHIFT = a.sample_shift
    torch.manual_seed(a.seed)
    model = DiT().to(device).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)[a.weights])
    out = Path(a.out); (out / "videos").mkdir(parents=True, exist_ok=True)
    strata = {}
    for i in rows:
        source = source_of(ds.index[i])
        grid = source.get("grid", [])
        shape = ((source["grid_size"], source["grid_size"]) if "grid_size" in source
                 else (len(grid), len(grid[0]) if grid else 0))
        if a.task=='sokoban':
            moves=source.get('num_moves',0)
            band='16-23' if 16<=moves<=23 else '24-27' if 24<=moves<=27 else '28-31' if 28<=moves<=31 else 'other'
            shape=(*shape,source.get('num_boxes',0),band)
        strata.setdefault(shape, []).append(i)
    for group in strata.values():
        group.sort(key=lambda i: hashlib.sha256(ds.index[i]["sample_id"].encode()).hexdigest())
    selected = []
    while len(selected) < min(a.save_videos, len(rows)):
        for shape in sorted(strata):
            if strata[shape] and len(selected) < a.save_videos:
                selected.append(strata[shape].pop(0))
    video_ids = set(selected)
    (out / "video_selection.json").write_text(json.dumps({
        "method": "hash-ranked round-robin over board shapes (Sokoban also over box counts and move bands), "
                  "independent of the outcomes",
        "sample_ids": [ds.index[i]["sample_id"] for i in selected],
    }, indent=2))
    is_eos = lambda f: find_eos_frame(f[None]) is not None
    results, t0 = [], time.time()
    for s in range(0, len(rows), a.batch):
        ids = rows[s:s + a.batch]
        first = torch.stack([to_tensor(ds.get(i)[:1], device)[0] for i in ids])
        noise = None
        if a.independent_noise:
            from rft_rollout import frame_noise
            sample_ids = [ds.index[i]['sample_id'] for i in ids]
            noise = lambda f: frame_noise(sample_ids, [a.seed] * len(ids), f, device)
        vids = (sample_k1(model, first, a.sample_steps, a.max_frames, is_eos, noise=noise) if a.arm in ("k1", "k8", "df")
                else sample_bidir(model, first, a.sample_steps, FIXED[a.task]))
        for i, v in zip(ids, vids):
            u8 = v.add(1).mul(127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            r = ds.index[i]
            m = CHECK[a.task](u8, source_of(r))
            if i in video_ids:
                write_mp4(out / "videos" / f"{r['sample_id']}.mp4", u8, fps=4)
            results.append({"sample_id": r["sample_id"], **{k: v for k, v in m.items() if not isinstance(v, np.ndarray)}})
    n = len(results); s_ = sum(r["success"] for r in results); v_ = sum(r.get("violation") is not None for r in results)
    index_path = DATA / a.task / f"{a.eval_set}_index.json"
    summary = {"n": n, "success": s_, "violation": v_, "eos": sum(bool(r.get("eos_detected")) for r in results),
               "set": a.eval_set, "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
               "ckpt": a.ckpt, "seconds": round(time.time() - t0, 1), "sample_steps": a.sample_steps, "sample_shift": a.sample_shift,
               "weights": a.weights, "seed": a.seed, "independent_per_case_noise": a.independent_noise}
    (out / "results.json").write_text(json.dumps({"summary": summary, "results": results}, default=str))
    print(json.dumps(summary), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("train", "anneal", "eval"))
    ap.add_argument("--task", required=True, choices=tuple(FIXED))
    ap.add_argument("--arm", required=True, choices=("k1", "k8", "bidir", "df"))
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--anneal-steps", type=int, default=1000)
    ap.add_argument("--resume")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--ckpt", help="eval: weights (ema.pt of a training step)")
    ap.add_argument("--limit", type=int, default=0, help="eval: a random subset of this many puzzles")
    ap.add_argument("--sample-steps", type=int, default=5); ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--save-videos", type=int, default=40)
    ap.add_argument("--eval-set", choices=("val", "test", "hard_test"), default="test")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--sample-shift", type=float, default=1.0)
    ap.add_argument("--weights", choices=('model', 'ema'), default='ema')
    ap.add_argument("--independent-noise", action='store_true',
                    help="eval: noise of every frame seeded by the puzzle (as in the post-training evaluations)")
    ap.add_argument("--data-root", type=Path, help="pixel caches (default: data/dit)")
    ap.add_argument("--hist-sigma", type=float, default=0.0, help="re-noise the generated history in the KV cache (0.1 for the df arm)")
    a = parse_args_with_config(ap)
    globals()['HIST_SIGMA'] = a.hist_sigma
    globals()['KBLOCK'] = 8 if a.arm == 'k8' else 1
    if a.data_root:
        global DATA
        DATA = a.data_root
    if a.mode == "eval":
        a.batch = a.batch if a.batch != 32 else 16
        evaluate(a)
    else:
        train(a)


if __name__ == '__main__':
    main()
