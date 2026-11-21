"""Corrections for DiT-B post-training: branch points, resampled continuations and divergence frames.

Sub-commands (ROUND is the collection folder; see rft_rollout.py for the rollouts):
    sokoban-forks   Sokoban branch points: the first frame of a failed rollout whose state breaks a rule or is proven
                    unsolvable by an exact solver (A*), then --backoff (4) frames earlier, walking further back until
                    the state before the branch point is readable and solvable. Writes OUT/sokoban/rollouts.shard0.jsonl
                    and links the rollouts there.
    (default)       resample --branches continuations of every failed rollout from its branch point (with
                    --fork-backoff frames earlier); writes OUT/<task>/branches.shard<i>.jsonl and the successful
                    continuations (OUT = ROUND/branches)
    roots           regenerate the successful whole rollouts (sample-level RFT) into OUT/<task>/ (OUT = ROUND/roots)
    decisions       the divergence frame d of every successful continuation (the first frame whose state differs from
                    the failed rollout) -> ROUND/decisions.json; on FlowFree, only continuations that diverge at most
                    --max-before-fatal moves before the fatal move are kept

Example (FlowFree; Sokoban: sokoban-forks first, then the same steps with ROUND = the output of sokoban-forks):
    python scripts/dit/rft_branch.py --task flowfree --ckpt CKPT --rollouts ROUND --out ROUND/branches --keep-float
    python scripts/dit/rft_branch.py --task flowfree --ckpt CKPT --rollouts ROUND --out ROUND/branches --keep-float \\
        --fork-backoff 3 --drop-clamped --branch-index-offset 4
    python scripts/dit/rft_branch.py roots --task flowfree --ckpt CKPT --rollouts ROUND --out ROUND/roots --mixed-only
    python scripts/dit/rft_branch.py decisions --task flowfree --round ROUND
"""
from __future__ import annotations

import argparse
import collections
import heapq
import itertools
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import train as T
from checkers import CHECK
from dit_model import DiT
from rft_rollout import frame_noise
from videcide.paths import data_path

BRANCH_BASE = 1000
SOLVER_BUDGET = 150_000


def branch_log(offset: int, shard: int | None = None) -> str:
    """Name (or, with shard None, name pattern) of the branch log of one pass (--branch-index-offset)."""
    suffix = "" if offset == 0 else f".b{offset}"
    return f"branches.shard{shard}{suffix}.jsonl" if shard is not None else rf"branches\.shard\d+{re.escape(suffix)}\.jsonl"


def judge_and_save(task, frames, fork, src, path, float_frames=None):
    metrics = CHECK[task](frames, src)
    rec = {'success': bool(metrics['success']), 'violation': metrics.get('violation'),
           'eos': bool(metrics.get('eos_detected'))}
    if rec["success"]:
        np.savez_compressed(path, frames=frames[fork:], **({'float_frames': float_frames[fork:]} if float_frames is not None else {}))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=("flowfree", "sokoban"))
    ap.add_argument("--ckpt", required=True, help="ema.pt of the model that made the rollouts")
    ap.add_argument("--rollouts", required=True, help="folder with <task>/rollouts.shard*.jsonl and the failed rollouts")
    ap.add_argument("--out", required=True, help="ROUND/branches")
    ap.add_argument("--branches", type=int, default=4, help="continuations per branch point")
    ap.add_argument("--per-puzzle", type=int, default=2, help="failed rollouts branched per puzzle")
    ap.add_argument("--steps", type=int, default=5); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--branch-base", type=int, default=BRANCH_BASE, help="noise seeds of the continuations")
    ap.add_argument("--branch-index-offset", type=int, default=0,
                    help="index of the first continuation; give each pass (e.g. each --fork-backoff) its own range")
    ap.add_argument("--data-root", type=Path, help="pixel caches (default: data/dit)")
    ap.add_argument("--sets", default="train,hard_train")
    ap.add_argument("--keep-float", action="store_true")
    ap.add_argument("--max-frames", type=int, default=80, help="generation budget (as in rft_rollout.py)")
    ap.add_argument("--fork-backoff", type=int, default=0, help="branch this many frames before the branch point")
    ap.add_argument("--drop-clamped", action="store_true",
                    help="with --fork-backoff: skip failures whose backed-off branch point would fall before frame 1")
    a = ap.parse_args()
    if a.data_root:
        T.DATA = a.data_root
    from videcide.tasks.video_parser import find_eos_frame
    device = torch.device("cuda")
    ds = T.Episodes(a.task, a.sets, max_length=a.max_frames); by_id = {r["sample_id"]: r for r in ds.index}
    fails = defaultdict(list)
    for f in Path(a.rollouts, a.task).glob("rollouts.shard*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if not r["success"] and isinstance(r.get("fork"), int) and r["fork"] >= 1 \
                    and Path(a.rollouts, a.task, f"{r['sample_id']}_r{r['rollout']}.npz").exists():
                fails[r["sample_id"]].append(r)
    picked = []
    for sid in sorted(fails)[a.shard::a.shards]:
        picked += sorted(fails[sid], key=lambda r: r["rollout"])[:a.per_puzzle]
    out = Path(a.out) / a.task; out.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in out.iterdir():
        if re.fullmatch(branch_log(a.branch_index_offset), f.name):
            for line in open(f):
                r = json.loads(line); done.add((r["sample_id"], r["rollout"]))
    picked = [r for r in picked if (r["sample_id"], r["rollout"]) not in done]
    if a.fork_backoff:
        if a.drop_clamped:
            picked = [r for r in picked if r["fork"] - a.fork_backoff >= 1]
        for r in picked:
            r["fork_boundary"] = r["fork"]; r["fork"] = max(1, r["fork"] - a.fork_backoff)
    by_fork = defaultdict(list)
    for r in picked:
        by_fork[r["fork"]].append(r)
    per_batch = max(1, a.batch // a.branches)
    jobs = [g[i:i + per_batch] for f, g in sorted(by_fork.items()) for i in range(0, len(g), per_batch)]
    random.Random(0).shuffle(jobs)
    print(f"{len(picked)} branch points ({len(done)} done), {len(jobs)} batches", flush=True)
    model = DiT().to(device).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["ema"])
    is_eos = lambda fr: find_eos_frame(fr[None]) is not None
    log = open(out / branch_log(a.branch_index_offset, a.shard), "a")
    pool, t0, n_done = ProcessPoolExecutor(a.workers), time.time(), 0
    pending = []
    for job in jobs:
        f = job[0]["fork"]
        pre = []
        for r in job:
            data = np.load(Path(a.rollouts, a.task, f"{r['sample_id']}_r{r['rollout']}.npz"))
            if 'float_frames' in data:
                pre.append(data['float_frames'][:f])
            else:
                pre.append(data['frames'][:f].transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1)
        rows, ids, ks = [], [], []
        for r, p in zip(job, pre):
            for j in range(a.branch_index_offset, a.branch_index_offset + a.branches):
                rows.append((r, j, p)); ids.append(r["sample_id"]); ks.append(a.branch_base + 100 * r["rollout"] + j)
        prefix = torch.from_numpy(np.stack([p for _, _, p in rows])).to(device).float()
        assert f < a.max_frames, 'branch point beyond the generation budget'
        vids = T.sample_k1(model, prefix[:, 0], a.steps, a.max_frames, is_eos, noise=lambda fr: frame_noise(ids, ks, fr, device), prefix=prefix)
        futs = defaultdict(list)
        for b, ((r, j, _), k, v) in enumerate(zip(rows, ks, vids)):
            assert torch.equal(v[:f], prefix[b]), 'the continuation changed the history'
            u8 = v.add(1).mul(127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            src = T.source_of(by_id[r["sample_id"]])
            path = out / f"{r['sample_id']}_r{r['rollout']}_b{j}.npz"
            futs[(r["sample_id"], r["rollout"])].append((j, k, pool.submit(judge_and_save, a.task, u8, f, src, path,
                v.cpu().numpy() if a.keep_float else None)))
        pending.append((job, futs))
        while len(pending) > 6 or (pending and job is jobs[-1]):
            jb, fu = pending.pop(0)
            for r in jb:
                br = [{"branch": j, "noise_rollout": k, **x.result()} for j, k, x in fu[(r["sample_id"], r["rollout"])]]
                log.write(json.dumps({"sample_id": r["sample_id"], "rollout": r["rollout"], "fork": r["fork"], "kind": r.get("kind"),
                                      "fork_boundary": r.get("fork_boundary", r["fork"]),
                                      "bad_noise": [r["sample_id"], r["rollout"], r["fork"]], "branches": br,
                                      "n_success": sum(b["success"] for b in br)}) + "\n")
                n_done += 1
            log.flush()
        print(f"{n_done}/{len(picked)} branch points {time.time() - t0:.0f}s", flush=True)
    log.close()


def check_save(task, frames, src, path):
    success = bool(CHECK[task](frames, src)["success"])
    if success:
        np.savez_compressed(path, frames=frames)
    return success


def roots_main():
    ap = argparse.ArgumentParser(description="Regenerate the successful whole rollouts (sample-level RFT).")
    ap.add_argument("--task", required=True, choices=("flowfree", "sokoban")); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rollouts", required=True, help="folder with <task>/rollouts.shard*.jsonl")
    ap.add_argument("--out", required=True, help="ROUND/roots")
    ap.add_argument("--steps", type=int, default=5); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--data-root", type=Path); ap.add_argument("--sets", default="train,hard_train")
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--mixed-only", action="store_true", help="only puzzles with both successful and failed rollouts")
    a = ap.parse_args()
    if a.data_root:
        T.DATA = a.data_root
    from videcide.tasks.video_parser import find_eos_frame
    device = torch.device("cuda")
    ds = T.Episodes(a.task, a.sets, max_length=a.max_frames); pos = {r["sample_id"]: i for i, r in enumerate(ds.index)}
    jobs, per = [], {}
    for f in sorted(Path(a.rollouts, a.task).glob("rollouts.shard*.jsonl")):
        for line in open(f):
            r = json.loads(line); per.setdefault(r["sample_id"], []).append(bool(r["success"]))
            if r["success"]:
                jobs.append((r["sample_id"], r["rollout"]))
    if a.mixed_only:
        jobs = [j for j in jobs if 0 < sum(per[j[0]]) < len(per[j[0]])]
    jobs = sorted(set(jobs))[a.shard::a.shards]
    out = Path(a.out) / a.task; out.mkdir(parents=True, exist_ok=True)
    model = DiT().to(device).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["ema"])
    is_eos = lambda fr: find_eos_frame(fr[None]) is not None
    pool, futs, t0 = ProcessPoolExecutor(24), [], time.time()
    for s in range(0, len(jobs), a.batch):
        chunk = jobs[s:s + a.batch]; ids = [x[0] for x in chunk]; ks = [x[1] for x in chunk]
        first = torch.stack([T.to_tensor(ds.get(pos[sid])[:1], device)[0] for sid in ids])
        vids = T.sample_k1(model, first, a.steps, a.max_frames, is_eos, noise=lambda fr: frame_noise(ids, ks, fr, device))
        for (sid, k), v in zip(chunk, vids):
            u8 = v.add(1).mul(127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            futs.append((sid, k, pool.submit(check_save, a.task, u8, T.source_of(ds.index[pos[sid]]), out / f"{sid}_r{k}.npz")))
        print(f"{min(s + a.batch, len(jobs))}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    with open(out / f"roots.shard{a.shard}.jsonl", "w") as log:
        for sid, k, fu in futs:
            log.write(json.dumps({"sample_id": sid, "rollout": k, "success": bool(fu.result())}) + "\n")


# ----- Sokoban branch points -------------------------------------------------------------------------------------

def solve_sokoban(grid, player, boxes, targets, budget=SOLVER_BUDGET):
    """Move-optimal A* (the solver of the Sokoban-Hard generator). Returns (moves or None, states searched); None
    with more than ``budget`` states searched means unknown, otherwise the state is unsolvable."""
    from videcide.tasks.sokoban_task import DIRS, floor_cells
    floors = set(floor_cells(grid)); goals = frozenset(targets)
    dead = {p for p in floors - goals if
            ((p[0] - 1, p[1]) not in floors or (p[0] + 1, p[1]) not in floors) and
            ((p[0], p[1] - 1) not in floors or (p[0], p[1] + 1) not in floors)}
    goals = tuple(sorted(goals)); assignments = tuple(itertools.permutations(goals))
    cache = {}

    def bound(bx):
        if bx not in cache:
            cache[bx] = min(sum(abs(b[0] - t[0]) + abs(b[1] - t[1]) for b, t in zip(bx, ts)) for ts in assignments)
        return cache[bx]
    start = (tuple(player), tuple(sorted(boxes)))
    costs = {start: 0}; parents = {start: None}; heap = [(bound(start[1]), 0, start)]
    while heap:
        _, cost, state = heapq.heappop(heap)
        if costs[state] != cost:
            continue
        pl, bx_tuple = state; bx = frozenset(bx_tuple)
        if bx == frozenset(goals):
            directions = []
            while parents[state] is not None:
                state, move = parents[state]; directions.append(move)
            return directions[::-1], len(costs)
        for name, (dr, dc) in DIRS.items():
            p = (pl[0] + dr, pl[1] + dc)
            if p not in floors:
                continue
            new_boxes = bx_tuple
            if p in bx:
                target = (p[0] + dr, p[1] + dc)
                if target not in floors or target in bx or target in dead:
                    continue
                new_boxes = tuple(sorted((bx - {p}) | {target}))
            new = (p, new_boxes); new_cost = cost + 1
            if new_cost < costs.get(new, 10 ** 9):
                costs[new] = new_cost; parents[new] = (state, name)
                heapq.heappush(heap, (new_cost + bound(new_boxes), new_cost, new))
                if len(costs) > budget:
                    return None, len(costs)
    return None, len(costs)


def sokoban_death(frames: np.ndarray, src: dict, budget: int) -> dict:
    """First frame of a failed rollout that breaks a rule or whose state is proven unsolvable (``fork``)."""
    from checkers import _decode, _legal
    from videcide.tasks.video_parser import find_eos_frame
    eos = find_eos_frame(frames); end = eos if eos is not None else len(frames)
    n_boxes = len(src["boxes_start"]); goals = [tuple(t) for t in src["targets"]]
    prev = None
    for i in range(end):
        agents, boxes, on_wall = _decode(frames[i], src)
        if on_wall or len(agents) != 1 or len(boxes) != n_boxes:
            return {"fork": i, "kind": "violation"}
        cur = (agents, boxes)
        if prev is not None and cur != prev and not _legal(prev, cur, src):
            return {"fork": i, "kind": "illegal_transition"}
        if cur != prev:
            plan, searched = solve_sokoban(src['grid'], agents[0], boxes, goals, budget=budget)
            if plan is None and searched > budget:
                return {"fork": None, "kind": "solver_unknown_state_limit"}
            if plan is None:
                return {"fork": i, "kind": "solver_dead"}
        prev = cur
    if eos is not None and eos >= 1 and prev is not None and set(prev[1]) != set(goals):
        return {"fork": eos, "kind": "premature_eos"}
    return {"fork": None, "kind": "no_eos" if eos is None else "unsolved_end"}


def sokoban_fork_one(job):
    rec, frames_path, src_path, backoff, budget = job
    from checkers import _decode
    src = json.loads(data_path(src_path).read_text()); goals = [tuple(t) for t in src['targets']]; n = len(src['boxes_start'])
    frames = np.load(frames_path)['frames']
    death = sokoban_death(frames, src, budget)
    if not isinstance(death["fork"], int) or death["fork"] < 1:
        return None
    D = death["fork"]; f = max(1, D - backoff)
    while f >= 1:
        agents, boxes, on_wall = _decode(frames[f - 1], src)
        if not on_wall and len(agents) == 1 and len(boxes) == n:
            plan, searched = solve_sokoban(src['grid'], agents[0], boxes, goals, budget=budget)
            if plan is not None:
                return {**rec, 'kind': death['kind'], 'fork': f, 'death': D}
            if searched > budget:
                return None
        f -= 1
    return None


def sokoban_forks_main():
    ap = argparse.ArgumentParser(description="Branch points of the failed Sokoban rollouts.")
    ap.add_argument('--rollouts', type=Path, required=True, help='folder of rft_rollout.py (sokoban/rollouts.shard*.jsonl)')
    ap.add_argument('--out', type=Path, required=True, help='ROUND: gets sokoban/rollouts.shard0.jsonl and links to the rollouts')
    ap.add_argument('--data-root', type=Path); ap.add_argument('--sets', default='train,hard_train')
    ap.add_argument('--max-frames', type=int, default=58)
    ap.add_argument('--backoff', type=int, default=4); ap.add_argument('--budget', type=int, default=SOLVER_BUDGET)
    ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    if a.data_root:
        T.DATA = a.data_root
    src = {r['sample_id']: r['source'] for r in T.Episodes('sokoban', a.sets, max_length=a.max_frames).index}
    col = a.rollouts / 'sokoban'
    recs = [json.loads(line) for f in sorted(col.glob('rollouts.shard*.jsonl')) for line in open(f)]
    fails = [r for r in recs if not r['success'] and r['sample_id'] in src
             and (col / f"{r['sample_id']}_r{r['rollout']}.npz").exists()]
    with ProcessPoolExecutor(a.workers) as ex:
        new = [r for r in ex.map(sokoban_fork_one, [(r, str(col / f"{r['sample_id']}_r{r['rollout']}.npz"), src[r['sample_id']],
                                                     a.backoff, a.budget) for r in fails], chunksize=8) if r is not None]
    out = a.out / 'sokoban'; out.mkdir(parents=True, exist_ok=True)
    with open(out / 'rollouts.shard0.jsonl', 'w') as f:
        for r in new:
            f.write(json.dumps(r) + '\n')
            link = out / f"{r['sample_id']}_r{r['rollout']}.npz"
            if not link.exists():
                os.symlink((col / link.name).resolve(), link)
    back = [r['death'] - r['fork'] for r in new]
    print(json.dumps({'failed_rollouts': len(fails), 'branch_points': len(new),
                      'mean_backoff': float(np.mean(back)) if back else None,
                      'backoff_hist': {int(k): int(v) for k, v in zip(*np.unique(back, return_counts=True))}}), flush=True)


# ----- divergence frames -------------------------------------------------------------------------------------------

def boards(frames, src):
    import cv2
    from videcide.tasks.flow_replay import evaluate_flowfree
    up = np.stack([cv2.resize(f, (384, 384), interpolation=cv2.INTER_NEAREST) for f in frames])
    trace = evaluate_flowfree(up, src, include_trace=True, require_adjacent_color_switch=True)['trace']
    state, out, t = {}, [], 0
    for i in range(len(frames)):
        while t < len(trace) and int(trace[t]['frame']) <= i:
            for r, c, col in trace[t].get('added', []):
                state[(int(r), int(c))] = int(col)
            t += 1
        out.append(dict(state))
    return out


def flowfree_decision(job):
    """First frame from the branch point on whose board differs from the failed rollout (None: no difference)."""
    stem, root_npz, branch_npz, f, src_path = job
    import cv2
    cv2.setNumThreads(1)
    src = json.loads(data_path(src_path).read_text())
    root = np.load(root_npz)['frames']; branch = np.concatenate([root[:f], np.load(branch_npz)['frames']])
    rb, bb = boards(root, src), boards(branch, src)
    for i in range(f, min(len(rb), len(bb))):
        if rb[i] != bb[i]:
            return stem, i
    return stem, None


def sokoban_decision(job):
    """First frame from the branch point whose state differs from the failed rollout (the branch point if none)."""
    stem, root_npz, branch_npz, f, src_path = job
    from checkers import _decode
    src = json.loads(data_path(src_path).read_text())
    root = np.load(root_npz)['frames']; branch = np.concatenate([root[:f], np.load(branch_npz)['frames']])
    state = lambda fr: (lambda a, b, w: (tuple(map(tuple, a)), tuple(sorted(map(tuple, b))), w))(*_decode(fr, src))
    for i in range(f, min(len(root), len(branch))):
        if state(root[i]) != state(branch[i]):
            return stem, i
    return stem, f


def decisions_main():
    ap = argparse.ArgumentParser(description="Divergence frames of the successful continuations -> ROUND/decisions.json.")
    ap.add_argument('--task', required=True, choices=('flowfree', 'sokoban'))
    ap.add_argument('--round', type=Path, required=True)
    ap.add_argument('--data-root', type=Path); ap.add_argument('--sets', default='train,hard_train')
    ap.add_argument('--max-before-fatal', type=int, default=4,
                    help='FlowFree: keep continuations that diverge at most this many moves before the fatal move')
    ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    if a.data_root:
        T.DATA = a.data_root
    src = {r['sample_id']: r['source'] for r in T.Episodes(a.task, a.sets).index}
    jobs, fatal = [], {}
    for fl in sorted((a.round / 'branches' / a.task).glob('branches.shard*.jsonl')):
        for line in open(fl):
            r = json.loads(line)
            for b in r['branches']:
                stem = f"{r['sample_id']}_r{r['rollout']}_b{b['branch']}"; bp = fl.parent / f'{stem}.npz'
                if b['success'] and bp.exists():
                    jobs.append((stem, str(a.round / a.task / f"{r['sample_id']}_r{r['rollout']}.npz"), str(bp),
                                 int(r['fork']), src[r['sample_id']]))
                    fatal[stem] = int(r.get('fork_boundary', r['fork']))
    with ProcessPoolExecutor(a.workers) as pool:
        found = dict(pool.map(flowfree_decision if a.task == 'flowfree' else sokoban_decision, jobs, chunksize=4))
    dec = {s: d for s, d in found.items() if d is not None
           and (a.task != 'flowfree' or fatal[s] - d <= a.max_before_fatal)}
    (a.round / 'decisions.json').write_text(json.dumps(dec, indent=0))
    fork = {j[0]: j[3] for j in jobs}
    print(json.dumps({'successful_continuations': len(jobs), 'with_decision': len(dec),
                      'puzzles': len({k.rsplit('_r', 1)[0] for k in dec}),
                      'd_minus_branch_point': dict(sorted(collections.Counter(min(d - fork[k], 8) for k, d in dec.items()).items()))}),
          flush=True)


if __name__ == '__main__':
    commands = {'roots': roots_main, 'sokoban-forks': sokoban_forks_main, 'decisions': decisions_main}
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        commands[sys.argv.pop(1)]()
    else:
        main()
