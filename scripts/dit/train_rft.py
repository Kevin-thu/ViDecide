"""Post-training of DiT-B on collected branches: CFT and its baselines (Tab. 5 and Tabs. S11-S12 of the paper).

Usage (one GPU; see configs/dit/ for the paper settings):
    torchrun --standalone --nproc_per_node=1 scripts/dit/train_rft.py --config configs/dit/cft_sokoban.yaml \\
        --init <SFT anneal>/step_011000/full_state.pt --round-dir ROUND --decisions ROUND/decisions.json --out OUT

ROUND is the collection folder of rft_rollout.py / rft_branch.py: ROUND/<task>/ (failed rollouts),
ROUND/branches/<task>/ (resampled continuations) and ROUND/roots/<task>/ (successful whole rollouts).
--mode:
    cft           the divergence frame d of every successful continuation reuses the noise of the failed rollout and
                  draws t=1 with probability --p-sigma1 (Tab. S12 varies it); half of the row's loss weight is on d
    decision_rft  the same continuations with fresh noise (--rft-sigma1-prob adds samples at t=1, Tab. S11)
    sample_rft    successful whole rollouts
    sft           continued SFT on expert videos
Expert videos of the training puzzles (--train-sets of the pixel cache) fill --sft-frac of every batch.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import train as T
from videcide.config import parse_args_with_config
from dit_model import DiT, TOK, patchify
from rft_rollout import frame_noise

MODES = ("cft", "decision_rft", "sample_rft", "sft")


def load_examples(round_dir: Path, task: str) -> list[dict]:
    """Successful continuations: the failed rollout (prefix up to the branch point f) and the continuation (suffix)."""
    ex = []
    for f in sorted((round_dir / "branches" / task).glob("branches.shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            for b in r["branches"]:
                p = f.parent / f"{r['sample_id']}_r{r['rollout']}_b{b['branch']}.npz"
                if b["success"] and p.exists():
                    ex.append({"sid": r["sample_id"], "k": r["rollout"], "f": r["fork"], "good_k": b["noise_rollout"],
                               "prefix": str(round_dir / task / f"{r['sample_id']}_r{r['rollout']}.npz"), "suffix": str(p)})
    return ex


def rft_loss(model, items, mu, device, fork_w=0.5):
    """Loss of a micro-batch. An item is (frames, f, eps, sigma, is_branch[, d]): the loss covers the frames from f on;
    for a branch, half of the weight (fork_w) lies on the divergence frame d (default f), where eps and sigma, if
    given, replace the training noise and the noise level."""
    B = len(items)
    Fm = -(-max(len(v) for v, *_ in items) // 8) * 8
    x = torch.zeros(B, Fm, 3, 256, 256, device=device)
    for i, (v, *_) in enumerate(items):
        x[i, :len(v)] = (torch.from_numpy(v).to(device) if v.dtype.kind == 'f' and v.shape[1] == 3 else T.to_tensor(v, device))
    lens = torch.tensor([len(v) for v, *_ in items], device=device)
    starts = torch.tensor([item[1] for item in items], device=device)
    slots = [(T.CLEAN, j) for j in range(Fm)] + [(T.NOISY, j) for j in range(1, Fm)]
    copy, fid = T.layout(slots, device)
    sig = T.sample_sigma((B, Fm - 1), mu, device)
    target = x[:, 1:]
    eps = torch.randn_like(target)
    for i, item in enumerate(items):
        _, f, e, s = item[:4]
        pf = item[5] if len(item) > 5 else f
        if e is not None:
            source_noise = e[0] if e.ndim == 4 else e
            assert source_noise.shape == eps[i, pf - 1].shape, 'the noise of the failed rollout must cover all RGB channels'
            eps[i, pf - 1] = source_noise
        if s is not None:
            sig[i, pf - 1] = s
    z = (1 - sig[..., None, None, None]) * target + sig[..., None, None, None] * eps
    tokens = torch.cat([patchify(x), patchify(z)], dim=1)
    sig_tok = torch.cat([torch.zeros(B, Fm * TOK, device=device), sig.repeat_interleave(TOK, dim=1)], dim=1)
    mask = T.causal_mask(copy, fid)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xh = model(tokens, sig_tok, fid, mask)[:, Fm * TOK:].float()
    err = ((patchify(target) - xh) ** 2).mean(-1).reshape(B, Fm - 1, TOK).mean(-1) / sig.clamp(min=0.05) ** 2
    j = torch.arange(1, Fm, device=device)[None]
    after = ((j < lens[:, None]) & (j > starts[:, None])).float()
    rest = (err * after).sum(1) / after.sum(1).clamp(min=1)
    fork_err = err.gather(1, (starts - 1).clamp(min=0)[:, None])[:, 0]
    is_branch = torch.tensor([item[4] if len(item) > 4 else item[1] > 1 or item[2] is not None for item in items], device=device)
    dec = torch.tensor([item[5] if len(item) > 5 else item[1] for item in items], device=device)
    if bool((dec != starts).any()):
        fork_err = err.gather(1, (dec - 1).clamp(min=0)[:, None])[:, 0]
        other = ((j < lens[:, None]) & (j >= starts[:, None]) & (j != dec[:, None])).float()
        rest = torch.where(dec != starts, (err * other).sum(1) / other.sum(1).clamp(min=1), rest)
    allv = ((j < lens[:, None]) & (j >= starts[:, None])).float()
    plain = (err * allv).sum(1) / allv.sum(1).clamp(min=1)
    per = torch.where(is_branch, fork_w * fork_err + (1 - fork_w) * rest, plain)
    return per.mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--task", required=True, choices=("flowfree", "sokoban"))
    ap.add_argument("--init", required=True, help="full_state.pt of the SFT model (anneal step 11,000)")
    ap.add_argument("--round-dir", required=True, help="collection folder (rft_rollout.py / rft_branch.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mu", type=float, default=2.303); ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=2); ap.add_argument("--sft-frac", type=float, default=0.25)
    ap.add_argument("--p-sigma1", type=float, default=0.5, help="cft: probability of t=1 at the divergence frame")
    ap.add_argument("--rft-sigma1-prob", type=float, default=0.0,
                    help="decision_rft: probability of t=1 at the divergence frame (0: timesteps from p_DF only)")
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--decay", type=int, default=0, help="final linear-decay steps; 0 keeps the learning rate constant after warmup")
    ap.add_argument("--fork-weight", type=float, default=0.5, help="share of a branch row's loss weight on the divergence frame")
    ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--data-root", type=Path, help="pixel caches (default: data/dit)")
    ap.add_argument("--train-sets", default="train,hard_train", help="sets of the cache that give the expert videos")
    ap.add_argument("--max-frames", type=int, default=80,
                    help="generation budget of the collection; longer expert videos are left out")
    ap.add_argument("--initial-ema-raw", action="store_true", help="start the EMA from the raw SFT weights")
    ap.add_argument("--exact-float-targets", action="store_true",
                    help="train on the float frames of the collected histories and continuations (saved with --keep-float)")
    ap.add_argument("--decisions", type=Path, help="JSON {branch: divergence frame d} of rft_branch.py decisions")
    ap.add_argument("--resume", type=Path, help="continue exactly from a full_state.pt of this run (weights, Adam, RNG)")
    a = parse_args_with_config(ap)
    stopped = {"requested": False}

    def request_stop(*_):
        stopped["requested"] = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    if a.data_root:
        T.DATA = a.data_root
    dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])); torch.cuda.set_device(device)
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ds = T.Episodes(a.task, a.train_sets, max_length=a.max_frames)
    if a.mode == "sample_rft":
        ex = [{"path": str(p)} for p in sorted((Path(a.round_dir) / "roots" / a.task).glob("*.npz"))]
    elif a.mode == "sft":
        ex = []
    else:
        ex = load_examples(Path(a.round_dir), a.task)
        if a.decisions:
            dd = json.loads(a.decisions.read_text())
            ex = [{**e, 'd': int(dd[Path(e['suffix']).stem])} for e in ex if Path(e['suffix']).stem in dd]
    model = DiT().to(device); ema = DiT().to(device)
    st = torch.load(a.resume or a.init, map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"])
    ema.load_state_dict(st['ema'] if a.resume else st['model'] if a.initial_ema_raw else st["ema"]); ema.requires_grad_(False)
    for blk in model.blocks:
        blk.compile(dynamic=False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    if "opt" in st:
        opt.load_state_dict(st["opt"])
    start_step = 0
    if a.resume:
        start_step = st['step']; saved_rng = st['rng']
        rng.setstate(saved_rng['sampling_python']); random.setstate(saved_rng['python'])
        np.random.set_state(saved_rng['numpy']); torch.set_rng_state(saved_rng['torch']); torch.cuda.set_rng_state_all(saved_rng['cuda'])
    micro = a.batch // a.accum
    n_sft = a.batch if a.mode == "sft" else int(round(a.batch * a.sft_frac))
    cfg = {**vars(a), "data_root": str(T.DATA), "decisions": str(a.decisions) if a.decisions else None,
           "resume": str(a.resume) if a.resume else None, "expert_videos": len(ds.index),
           "examples": len(ex), "puzzles": len({e.get("sid", Path(e.get("path", "x")).name.rsplit("_r", 1)[0]) for e in ex}),
           "init_step": st["step"], "sft_per_step": n_sft}
    (out / "config.json").write_text(json.dumps(cfg, indent=1)); print(json.dumps(cfg), flush=True)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    save_steps = set(range(a.save_every, a.steps + 1, a.save_every)) | set(range(1000, a.steps + 1, 1000)) | {a.steps}
    full_saves = a.steps // 1000 + int(a.steps % 1000 != 0)
    expected_bytes = parameter_bytes * (2 * len(save_steps) + 4 * full_saves)
    if shutil.disk_usage(out).free < expected_bytes + 2 * 2**30:
        raise RuntimeError(f"not enough free space in {out} for the checkpoints ({expected_bytes / 2**30:.1f} GiB)")
    log = open(out / "loss.jsonl", "a"); t0 = time.time()
    cache = {}

    def frames(path, exact):
        key = (path, exact)
        if key not in cache:
            if len(cache) > 4000:
                cache.clear()
            payload = np.load(path)
            if exact:
                assert 'float_frames' in payload, f'{path} has no float frames (collect with --keep-float)'
            cache[key] = payload['float_frames' if exact else 'frames']
        return cache[key]

    for step in range(start_step + 1, a.steps + 1):
        decay_scale = 1.0 if a.decay <= 0 else max(0.0, (a.steps - step) / a.decay)
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, step / max(1, a.warmup), decay_scale)
        rows = [("expert", rng.randrange(len(ds))) for _ in range(n_sft)]
        n_rft = a.batch - n_sft
        if a.mode != "sft":
            rows += [(a.mode, e) for e in rng.sample(ex, n_rft)]
        rng.shuffle(rows)
        opt.zero_grad(set_to_none=True); tot = 0.0
        for s in range(0, len(rows), micro):
            items = []
            for kind, e in rows[s:s + micro]:
                if kind == "expert":
                    items.append((ds.get(e), 1, None, None)); continue
                if kind == "sample_rft":
                    items.append((frames(e["path"], False), 1, None, None)); continue
                full = np.concatenate([frames(e["prefix"], a.exact_float_targets)[:e["f"]],
                                       frames(e["suffix"], a.exact_float_targets)])
                if kind == "cft":
                    pf = e.get('d', e['f'])
                    eps = frame_noise([e["sid"]], [e["k"]], pf, device)[0]
                    sig = 1.0 if rng.random() < a.p_sigma1 else float(torch.sigmoid(a.mu + torch.randn(())))
                    items.append((full, e["f"], eps, sig, True) + ((pf,) if 'd' in e else ()))
                else:
                    sig_r = 1.0 if (a.rft_sigma1_prob > 0 and rng.random() < a.rft_sigma1_prob) else None
                    items.append((full, e["f"], None, sig_r, True) + ((e['d'],) if 'd' in e else ()))
            if items:
                loss = rft_loss(model, items, a.mu, device, a.fork_weight) * len(items) / len(rows)
                loss.backward(); tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        if step % 10 == 0:
            log.write(json.dumps({"step": step, "time": time.time(), "loss": tot, "grad_norm": float(gn),
                                  "lr": opt.param_groups[0]["lr"]}) + "\n"); log.flush()
        if step % 100 == 0:
            print(f"step {step} loss {tot:.4f} {(time.time() - t0) / step:.2f}s/step", flush=True)
        if step in save_steps or stopped['requested']:
            d = out / f"step_{step:05d}"; d.mkdir(exist_ok=True)
            torch.save({"ema": ema.state_dict(), "model_raw": model.state_dict(), "step": step}, d / "ema.pt")
            if step % 1000 == 0 or step == a.steps or stopped['requested']:
                target = d / 'full_state.pt'
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                            "rng": {"sampling_python": rng.getstate(), "python": random.getstate(),
                                    "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                                    "cuda": torch.cuda.get_rng_state_all()}, "config": cfg},
                           d / "full_state.pt.tmp")
                (d / 'full_state.pt.tmp').replace(target)
                (out / 'latest.json').write_text(json.dumps({'step': step, 'full_state': str(target),
                                                             'stopped_early': stopped['requested']}))
        if stopped['requested']:
            break
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
