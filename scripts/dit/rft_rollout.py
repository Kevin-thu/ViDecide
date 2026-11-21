"""Roll out a DiT-B model on the training puzzles: the first step of collecting corrections for post-training.

    python scripts/dit/rft_rollout.py --task flowfree --ckpt <SFT anneal>/step_011000/ema.pt --rollouts 2 \\
        --keep-float --out ROUND [--shard i --shards n]
    python scripts/dit/rft_rollout.py --task sokoban --ckpt ... --rollouts 8 --max-frames 58 --keep-mixed-only \\
        --keep-float --out COLLECTION

Every rollout is judged by the verifier of checkers.py and logged in OUT/<task>/rollouts.shard<i>.jsonl. By default
the frames of failed rollouts are saved (OUT/<task>/<sample_id>_r<k>.npz) together with their branch point on
FlowFree (the first move of the failed walk that cannot be completed, fork_selection.boundary_of). With
--keep-mixed-only, the rollouts of a puzzle are saved only if some succeed and some fail, and the branch points are
chosen afterwards (rft_branch.py sokoban-forks). The noise of every frame is seeded by (puzzle, rollout, frame), so
that post-training can rebuild the noise of a failed rollout.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T
from dit_model import DiT
from checkers import CHECK

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def noise_seed(sample_id: str, k: int, frame: int) -> int:
    return int(hashlib.sha1(f"{sample_id}|{k}|{frame}".encode()).hexdigest()[:15], 16)


def frame_noise(ids, ks, frame, device, offset=0):
    out = []
    for sid, k in zip(ids, ks):
        g = torch.Generator(device=device).manual_seed(noise_seed(sid, k + offset, frame))
        out.append(torch.randn(1, 3, 256, 256, device=device, generator=g))
    return torch.stack(out)


def flowfree_fork(frames: np.ndarray, src: dict) -> dict:
    import cv2
    from videcide.tasks.flow_replay import evaluate_flowfree
    from videcide.training.fork_selection import boundary_of
    up = np.stack([cv2.resize(f, (384, 384), interpolation=cv2.INTER_NEAREST) for f in frames])
    trace = evaluate_flowfree(up, src, include_trace=True,
                              require_adjacent_color_switch=bool(src.get("continuous_global_walk", False)))["trace"]
    f, kind = boundary_of(trace, src)
    return {"fork": f, "kind": kind}


FORK = {"flowfree": flowfree_fork}   # Sokoban branch points: rft_branch.py sokoban-forks


def judge(task, frames, src):
    m = CHECK[task](frames, src)
    rec = {"success": bool(m["success"]), "violation": m.get("violation"), "eos": bool(m.get("eos_detected"))}
    if not rec["success"] and task in FORK:
        try:
            rec.update(FORK[task](frames, src))
        except Exception as e:
            rec.update({"fork": None, "kind": f"error:{type(e).__name__}"})
    return rec


def judge_and_save(task, frames, src, path, keep_success, float_frames=None):
    rec = judge(task, frames, src)
    if not rec["success"] or keep_success:
        np.savez_compressed(path, frames=frames, **({'float_frames': float_frames} if float_frames is not None else {}))
    return rec


def screen_mixed_group(task, group, out):
    """Judge all rollouts of a puzzle; save their frames only if some succeed and some fail."""
    records = []
    for sid, k, frames, source, float_frames in group:
        m = CHECK[task](frames, source)
        records.append({'success': bool(m['success']), 'violation': m.get('violation'),
                        'eos': bool(m.get('eos_detected')), 'fork': None, 'kind': 'not_screened_for_fork'})
    mixed = 0 < sum(r['success'] for r in records) < len(records)
    if mixed:
        for (sid, k, frames, source, float_frames), record in zip(group, records):
            np.savez_compressed(Path(out) / f'{sid}_r{k}.npz', frames=frames,
                                **({'float_frames': float_frames} if float_frames is not None else {}))
    for record in records:
        record.update(history_saved=mixed, group_rollouts=len(records))
    return records


def collect_mixed(a, ds, idx, model, out, log, done_keys, device, is_eos):
    assert a.batch % a.rollouts == 0, '--batch must be a multiple of --rollouts'
    todo = [i for i in idx if any((ds.index[i]['sample_id'], k) not in done_keys for k in range(a.rollouts))]
    per_batch = a.batch // a.rollouts
    pending, started = [], time.time()

    def consume(sid, ks, lengths, records):
        for k, n, record in zip(ks, lengths, records):
            if (sid, k) not in done_keys:
                write(log, a, sid, k, n, record); done_keys.add((sid, k))
    with ProcessPoolExecutor(a.workers) as pool:
        for start in range(0, len(todo), per_batch):
            puzzles = todo[start:start + per_batch]
            chunk = [(i, k) for i in puzzles for k in range(a.rollouts)]
            ids = [ds.index[i]['sample_id'] for i, k in chunk]; ks = [k for i, k in chunk]
            first = torch.stack([T.to_tensor(ds.get(i)[:1], device)[0] for i, k in chunk])
            vids = T.sample_k1(model, first, a.steps, a.max_frames, is_eos, noise=lambda f: frame_noise(ids, ks, f, device))
            for p, i in enumerate(puzzles):
                group = []
                for (_, k), sid, v in zip(chunk[p * a.rollouts:(p + 1) * a.rollouts], ids[p * a.rollouts:(p + 1) * a.rollouts],
                                          vids[p * a.rollouts:(p + 1) * a.rollouts]):
                    frames = v.add(1).mul(127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
                    group.append((sid, k, frames, T.source_of(ds.index[i]), v.cpu().numpy() if a.keep_float else None))
                pending.append((group[0][0], [x[1] for x in group], [len(x[2]) for x in group],
                                pool.submit(screen_mixed_group, a.task, group, str(out))))
            pending = drain_ready(pending, consume); log.flush()
            while len(pending) > 8 * per_batch:
                time.sleep(1); pending = drain_ready(pending, consume); log.flush()
            print(f'{min(start + per_batch, len(todo))}/{len(todo)} puzzles {time.time() - started:.0f}s', flush=True)
        for sid, ks, lengths, future in pending:
            consume(sid, ks, lengths, future.result())
    log.close()


def write(log, a, sid, k, n, rec):
    log.write(json.dumps({"sample_id": sid, "rollout": k, "frames": n, "steps": a.steps, **rec}) + "\n")


def drain_ready(pending, consume):
    remaining = []
    for item in pending:
        if item[3].done():
            consume(*item[:3], item[3].result())
        else:
            remaining.append(item)
    return remaining


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=("flowfree", "sokoban"))
    ap.add_argument("--ckpt", required=True, help="ema.pt of the model to roll out")
    ap.add_argument("--sets", default="train,hard_train", help="sets of the pixel cache whose puzzles are rolled out")
    ap.add_argument("--data-root", type=Path, help="pixel caches (default: data/dit)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample-ids", help="JSON list of puzzle ids to roll out (before sharding)")
    ap.add_argument("--rollouts", type=int, default=2, help="rollouts per puzzle")
    ap.add_argument("--steps", type=int, default=5, help="Euler steps per frame")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-frames", type=int, default=80,
                    help="generation budget in frames; puzzles whose expert video is longer are left out")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--keep-success", action="store_true", help="also save the frames of successful rollouts")
    ap.add_argument("--keep-mixed-only", action="store_true",
                    help="save the rollouts of a puzzle only if some succeed and some fail (no branch points)")
    ap.add_argument("--keep-float", action="store_true", help="also save the float frames (exact histories for branching)")
    a = ap.parse_args()
    if a.data_root:
        T.DATA = a.data_root
    from videcide.tasks.video_parser import find_eos_frame
    device = torch.device("cuda")
    ds = T.Episodes(a.task, a.sets, max_length=a.max_frames)
    idx = list(range(len(ds)))
    if a.sample_ids:
        selected = json.loads(Path(a.sample_ids).read_text())
        assert len(selected) == len(set(selected))
        by_id = {r["sample_id"]: i for i, r in enumerate(ds.index)}
        idx = [by_id[sid] for sid in selected]
    idx = idx[: a.limit or None][a.shard::a.shards]
    out = Path(a.out) / a.task; out.mkdir(parents=True, exist_ok=True)
    shard_lock = open(out / f'.shard{a.shard}.lock', 'a')
    fcntl.flock(shard_lock, fcntl.LOCK_EX)
    model = DiT().to(device).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["ema"])
    log = open(out / f"rollouts.shard{a.shard}.jsonl", "a")
    is_eos = lambda f: find_eos_frame(f[None]) is not None
    done_keys = set()
    for f in out.glob("rollouts.shard*.jsonl"):
        for line in open(f):
            r = json.loads(line); done_keys.add((r["sample_id"], r["rollout"]))
    if a.keep_mixed_only:
        collect_mixed(a, ds, idx, model, out, log, done_keys, device, is_eos)
        return
    jobs = [(i, k) for i in idx for k in range(a.rollouts) if (ds.index[i]["sample_id"], k) not in done_keys]
    print(f"{len(done_keys)} rollouts already logged; {len(jobs)} to do", flush=True)
    t0, pool, pending = time.time(), ProcessPoolExecutor(a.workers), []
    for s in range(0, len(jobs), a.batch):
        chunk = jobs[s:s + a.batch]
        ids = [ds.index[i]["sample_id"] for i, _ in chunk]; ks = [k for _, k in chunk]
        first = torch.stack([T.to_tensor(ds.get(i)[:1], device)[0] for i, _ in chunk])
        vids = T.sample_k1(model, first, a.steps, a.max_frames, is_eos, noise=lambda f: frame_noise(ids, ks, f, device))
        for (i, k), sid, v in zip(chunk, ids, vids):
            u8 = v.add(1).mul(127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            pending.append((sid, k, len(u8), pool.submit(judge_and_save, a.task, u8, T.source_of(ds.index[i]),
                                                       out / f"{sid}_r{k}.npz", a.keep_success,
                                                       v.cpu().numpy() if a.keep_float else None)))
        pending = drain_ready(pending, lambda sid, k, n, rec: write(log, a, sid, k, n, rec))
        log.flush()
        while len(pending) > 8 * a.batch:
            time.sleep(1)
            pending = drain_ready(pending, lambda sid, k, n, rec: write(log, a, sid, k, n, rec))
            log.flush()
        print(f"{min(s + a.batch, len(jobs))}/{len(jobs)} rollouts {time.time() - t0:.0f}s", flush=True)
    for sid, k, n, fut in pending:
        write(log, a, sid, k, n, fut.result())
    log.close()


if __name__ == "__main__":
    main()
