"""Sokoban-Hard and the harder Sokoban training puzzles of post-training.

    python scripts/benchmarks/generate_sokoban_hard.py --set hard_train   # 5,000 puzzles
    python scripts/benchmarks/generate_sokoban_hard.py --set hard_test    # 500 puzzles

Puzzles have two or three boxes (60% / 40%) in new random rooms of 7x7 to 9x9 cells including walls and a move-optimal
solution of 16-23, 24-27 or 28-31 moves (half, 30% and 20% of the puzzles), found by A* search. A room is carved by a
random walk, the boxes are placed on the targets and pulled away by reverse moves, and the puzzle is kept if A* solves it
within the move range. Rooms whose wall layout (up to rotation and reflection) occurs in the data already, including the
VideoRLVR puzzles, are rejected, so the hard sets are disjoint from all training puzzles when the training puzzles are
generated first. Every source stores its seed and policy: generate((set, index, boxes, band, seed, policy)) rebuilds it.
Sources, videos (four frames per move, verified by the strict verifier) and the manifest are written to
data/VideoRLVR/Sokoban (--out) in the layout of the released data.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import hashlib
import heapq
import itertools
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from videcide.paths import DATA_ROOT, benchmark_dir
from videcide.tasks.sokoban_task import DIRS, floor_cells, sokoban_source

BANDS = [(16, 23), (24, 27), (28, 31)]                  # optimal moves of the three difficulty bands
COUNTS = {'hard_train': [2500, 1500, 1000], 'hard_test': [250, 150, 100]}
BOX_SHARE = {2: .6, 3: .4}
SEED_BASE = 202610120000                                # seed of the k-th generation job: SEED_BASE + k
PROMPT = 'Solve the Sokoban puzzle. Move the agent and push every box onto a target. Boxes cannot be pulled and walls cannot be crossed.'


def random_room(n, rng):
    target = max(20, int(.55 * (n - 2) ** 2))
    opened = set(); cell = (rng.randrange(1, n - 1), rng.randrange(1, n - 1))
    while len(opened) < target:
        opened.add(cell); dr, dc = rng.choice(list(DIRS.values()))
        nxt = (cell[0] + dr, cell[1] + dc)
        if 1 <= nxt[0] < n - 1 and 1 <= nxt[1] < n - 1:
            cell = nxt
    return [''.join(' ' if (r, c) in opened else '#' for c in range(n)) for r in range(n)]


def canonical(grid, player, boxes, targets, layout_only=False):
    """Hash of a puzzle (or of its wall layout only) that is invariant to rotation and reflection."""
    import numpy as np
    a = np.zeros((len(grid), len(grid[0])), dtype=np.uint8)
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            a[r, c] = ch == '#'
    if not layout_only:
        for r, c in targets:
            a[r, c] |= 2
        for r, c in boxes:
            a[r, c] |= 4
        a[tuple(player)] |= 8
    variants = []
    for k in range(4):
        b = np.rot90(a, k)
        for v in [b, np.fliplr(b)]:
            variants.append(str(v.shape).encode() + v.tobytes())
    return hashlib.sha256(min(variants)).hexdigest()


def solve(grid, player, boxes, targets, budget=150000):
    """Move-optimal solution (A* with a box-assignment bound and simple dead squares), or None within the budget."""
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
            if new_cost < costs.get(new, 10**9):
                costs[new] = new_cost; parents[new] = (state, name)
                heapq.heappush(heap, (new_cost + bound(new_boxes), new_cost, new))
                if len(costs) > budget:
                    return None, len(costs)
    return None, len(costs)


def generate(spec):
    """One puzzle: (set, index, boxes, band, seed, policy) -> source."""
    set_name, index, boxes_count, tier, seed, policy = spec
    rng = random.Random(seed); lo, hi = policy['move_range']
    attempts = 0
    while True:
        attempts += 1; grid = random_room(rng.choice(policy['room_sizes']), rng)
        floors = set(floor_cells(grid))
        inner = [p for p in floors if sum((p[0] + dr, p[1] + dc) in floors for dr, dc in DIRS.values()) >= 3]
        if len(inner) < boxes_count + 1:
            continue
        targets = rng.sample(inner, boxes_count); bx = set(targets); pl = rng.choice(sorted(floors - bx))
        assignments = tuple(itertools.permutations(targets)); visited = {tuple(sorted(bx))}
        best = (0, pl, set(bx))
        for _ in range(rng.randrange(*policy['pull_iterations'])):
            reachable = {pl}; queue = [pl]
            for cell in queue:
                for dr, dc in DIRS.values():
                    p = (cell[0] + dr, cell[1] + dc)
                    if p in floors and p not in bx and p not in reachable:
                        reachable.add(p); queue.append(p)
            pulls = []
            for box in sorted(bx):
                for dr, dc in DIRS.values():
                    p = (box[0] + dr, box[1] + dc); nxt = (box[0] + 2 * dr, box[1] + 2 * dc)
                    new = tuple(sorted((bx - {box}) | {p}))
                    if p not in reachable or nxt not in floors or nxt in bx or new in visited:
                        continue
                    distance = min(sum(abs(b[0] - t[0]) + abs(b[1] - t[1]) for b, t in zip(new, ts)) for ts in assignments)
                    pulls.append((distance + rng.random() * 5, distance, nxt, new))
            if not pulls:
                break
            _, distance, pl, new = max(pulls); bx = set(new); visited.add(new)
            if distance >= best[0]:
                best = (distance, pl, set(bx))
        distance, pl, bx = best
        if distance < policy['minimum_assignment_distance'] or bx == set(targets):
            continue
        moves, searched = solve(grid, pl, bx, targets)
        if moves is None or not lo <= len(moves) <= hi:
            continue
        source = sokoban_source(f'sokoban_{set_name}_{index:05d}', grid, pl, sorted(bx), targets, moves,
                                {'generator': 'reverse_pull_astar', 'render_theme': 'classic', 'difficulty_tier': tier,
                                 'difficulty_range': [lo, hi], 'seed': seed, 'astar_states': searched,
                                 'generation_attempts': attempts, 'generation_policy': policy})
        from videcide.tasks.sokoban_task import replay
        _, _, verdict = replay(source); assert verdict['success'], verdict
        return source


def finish(item):
    """Write the source and the video of one puzzle; return its manifest row."""
    source, set_name, out = item
    out = Path(out)
    from videcide.tasks import sokoban_task
    from videcide.tasks.video import write_mp4
    from videcide.tasks.video_parser import parse_video
    video = sokoban_task.render_optimal(source, theme='classic', frames_per_move=4, start_hold=5)
    assert parse_video(video, source, 'sokoban')['success'], source['sample_id']
    sid = source['sample_id']
    rel = out.relative_to(DATA_ROOT) if out.is_relative_to(DATA_ROOT) else out
    source_path, video_path = f'sources/{set_name}/{sid}.json', f'videos/{set_name}/{sid}.mp4'
    source = {**source, 'native_frames': int(len(video)), 'prompt': PROMPT}
    for path in (source_path, video_path):
        (out / path).parent.mkdir(parents=True, exist_ok=True)
    (out / source_path).write_text(json.dumps(source))
    write_mp4(out / video_path, video)
    return {'sample_id': sid, 'task': 'sokoban', 'split': 'test' if set_name.endswith('test') else 'train', 'prompt': PROMPT,
            'theme': 'classic', 'size': f"{source['grid_rows']}x{source['grid_cols']}", 'native_frames': int(len(video)),
            'native_video': str(rel / video_path), 'source_path': str(rel / source_path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', required=True, choices=list(COUNTS))
    ap.add_argument('--out', type=Path, default=benchmark_dir('sokoban'), help='Sokoban folder of the data')
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--scale', type=float, default=1.0, help='fraction of the puzzles to generate (tests)')
    a = ap.parse_args()
    excluded = set()
    for path in (a.out / 'sources').rglob('*.json'):
        s = json.loads(path.read_text())
        excluded.add(canonical(s['grid'], s['player_start'], s['boxes_start'], s['targets'], True))
    print('wall layouts already in the data', len(excluded), flush=True)
    quotas = {(band, boxes): int(round(n * a.scale * BOX_SHARE[boxes])) for band, n in enumerate(COUNTS[a.set]) for boxes in (2, 3)}
    have, accepted, submitted = Counter(), [], 0
    with ProcessPoolExecutor(a.workers) as pool:
        while any(have[k] < q for k, q in quotas.items()):
            specs = []
            for (band, boxes), q in quotas.items():
                lo, hi = BANDS[band]
                policy = {'move_range': [lo, hi], 'room_sizes': [7, 8, 9], 'pull_iterations': [8, 25],
                          'minimum_assignment_distance': max(5, lo // 3)}
                for _ in range(q - have[band, boxes]):
                    submitted += 1
                    specs.append((a.set, submitted, boxes, band, SEED_BASE + submitted, policy))
            for s in pool.map(generate, specs, chunksize=1):          # in submission order: reproducible
                key = canonical(s['grid'], s['player_start'], s['boxes_start'], s['targets'], True)
                if key in excluded or have[s['difficulty_tier'], s['num_boxes']] >= quotas[s['difficulty_tier'], s['num_boxes']]:
                    continue
                excluded.add(key); have[s['difficulty_tier'], s['num_boxes']] += 1
                accepted.append({**s, 'canonical_layout_sha256': key})
            print(f'{len(accepted)} / {sum(quotas.values())} puzzles', flush=True)
        accepted.sort(key=lambda s: s['sample_id'])
        rows = list(pool.map(finish, [(s, a.set, str(a.out)) for s in accepted], chunksize=8))
    with open(a.out / f'{a.set}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(json.dumps({'set': a.set, 'puzzles': len(rows), 'by_band_and_boxes': {f'{BANDS[b]}:{n}': c for (b, n), c in sorted(have.items())}}))


if __name__ == '__main__':
    main()
