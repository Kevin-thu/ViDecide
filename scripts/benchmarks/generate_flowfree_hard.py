"""FlowFree-Hard and the harder FlowFree training puzzles of post-training, generated as in VideoRLVR.

    python scripts/benchmarks/generate_flowfree_hard.py --set hard_train   # 5,000 puzzles: 7x7-8x8 boards, then 5x5-8x8
    python scripts/benchmarks/generate_flowfree_hard.py --set hard_test    # 500 puzzles, 5x5-8x8 boards with 4-6 colors

Every board is a Hamiltonian path built with Warnsdorff's rule and cut into 4-6 contiguous color segments; the solution
walk visits the segments in order, so every color switch is adjacent to the previous cell. Layouts already in the data
(VideoRLVR and the existing hard sets) are rejected, so FlowFree-Hard is disjoint from all training puzzles when the
training puzzles are generated first. Sources, traces, videos (verified by the strict verifier) and the manifest are
written to data/VideoRLVR/FlowFree (--out) in the layout of the released data.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from videcide.paths import DATA_ROOT, benchmark_dir
from videcide.tasks.flowfree_task import evaluate_symbolic

P_END, P_MID = 0.19, 0.035     # probability that a segment marks its last cell / one middle cell as an extra endpoint
# Colors of the VideoRLVR FlowFree boards, most frequent first, at least 40 RGB units apart.
PALETTE = [[49, 203, 80], [48, 149, 219], [221, 49, 50], [249, 192, 32], [205, 90, 249], [180, 78, 220],
           [250, 135, 54], [222, 160, 32]]
STRATA = [((8, 6), .30), ((8, 5), .25), ((7, 6), .25), ((7, 5), .10), ((8, 4), .10)]   # (board size, colors), weight
SMALL = [((5, 4), 1), ((5, 5), 1), ((5, 6), 1)], [((6, 4), 1), ((6, 5), 1), ((6, 6), 1)]
# set -> parts of (count, (5x5 boards, 6x6 boards) or None, first index, seed)
SETS = {'hard_train': [(3000, None, 0, 20261127), (2000, (400, 600), 3000, 20261128)],
        'hard_test': [(500, (40, 60), 0, 20261129)]}
PROMPT = ('Fill the grid one cell at a time. Start at any colored endpoint and visit every remaining endpoint of that color. '
          'Then switch at an endpoint of a new color. Moves are horizontal or vertical, cannot cross or overlap, and every cell '
          'must be filled. Keep a single continuous walk across color changes: the next endpoint must be adjacent to the previous filled cell.')
D4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def layout(size: int, endpoints) -> str:
    return hashlib.sha256(json.dumps([size, sorted(sorted(tuple(c) for c in group) for group in endpoints)],
                                     sort_keys=True).encode()).hexdigest()


def warnsdorff_walk(n, rng):
    while True:
        c = (rng.randrange(n), rng.randrange(n)); seen = {c}; path = [c]
        while len(path) < n * n:
            nb = [(c[0] + a, c[1] + b) for a, b in D4 if 0 <= c[0] + a < n and 0 <= c[1] + b < n and (c[0] + a, c[1] + b) not in seen]
            if not nb:
                break
            deg = {x: sum(1 for a, b in D4 if 0 <= x[0] + a < n and 0 <= x[1] + b < n and (x[0] + a, x[1] + b) not in seen) for x in nb}
            m = min(deg.values()); c = rng.choice([x for x in nb if deg[x] == m]); seen.add(c); path.append(c)
        if len(path) == n * n:
            return path


def generate(n, k, palette, rng):
    cells = warnsdorff_walk(n, rng)
    cuts = sorted(rng.sample(range(1, n * n), k - 1))
    bounds = [0, *cuts, n * n]
    paths = [cells[a:b] for a, b in zip(bounds, bounds[1:])]
    endpoints = []
    for path in paths:
        marks = {path[0]}
        if len(path) > 1 and rng.random() < P_END:
            marks.add(path[-1])
        if len(path) > 2 and rng.random() < P_MID:
            marks.add(rng.choice(path[1:-1]))
        endpoints.append(sorted(marks))
    return {"grid_size": n, "paths": [[list(c) for c in p] for p in paths],
            "endpoints": [[list(c) for c in e] for e in endpoints], "source_colors_rgb": rng.sample(palette, k)}


def frames_for(n, k):
    return {5: 73, 6: 101, 7: 133}[n] if n < 8 else (173 if k <= 4 else 169)


def finish(item):
    """Write the source, trace and video of one puzzle; return its manifest row."""
    import cv2
    cv2.setNumThreads(1)
    from videcide.tasks.flowfree_task import render_order
    from videcide.tasks.flow_replay import evaluate_flowfree
    from videcide.tasks.video import write_mp4
    sid, source, set_name, out = item
    out = Path(out)
    cells = [tuple(c) for p in source['paths'] for c in p]
    colors = [i for i, p in enumerate(source['paths']) for _ in p]
    n = source['grid_size']
    assert len(set(cells)) == len(cells) == n * n
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(cells, cells[1:]))
    visited, used, active = set(), set(), None
    endpoints = [set(map(tuple, es)) for es in source['endpoints']]
    for cell, color in zip(cells, colors):
        if color != active:
            assert color not in used and cell in endpoints[color]
            assert active is None or endpoints[active] <= visited
            used.add(color); active = color
        visited.add(cell)
    trace = dict(cells=cells, colors=colors, unique_observed_order=True, split_merged_endpoint_groups=0,
                 corrected_cell_labels=0, repaired_endpoint_palette=False)
    rel = out.relative_to(DATA_ROOT) if out.is_relative_to(DATA_ROOT) else out
    trace_path, source_path, video = (f'traces/{set_name}/{sid}.json', f'sources/{set_name}/{sid}.json', f'videos/{set_name}/{sid}.mp4')
    frames = frames_for(n, len(source['paths']))
    source.update(sample_id=sid, observed_trace_path=str(rel / trace_path), frames=frames, continuous_global_walk=True, prompt=PROMPT)
    source['source_sha256'] = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    for path, value in ((trace_path, trace), (source_path, source)):
        (out / path).parent.mkdir(parents=True, exist_ok=True)
        (out / path).write_text(json.dumps(value))
    video_frames = render_order(source, trace, frames)
    verdict = evaluate_flowfree(video_frames, source, require_adjacent_color_switch=True)
    assert verdict['success'], (sid, verdict.get('failure_reason'))
    (out / video).parent.mkdir(parents=True, exist_ok=True)
    write_mp4(out / video, video_frames, fps=16)
    return dict(sample_id=sid, task='flowfree', split='test' if set_name.endswith('test') else 'train', prompt=PROMPT,
                theme='default', size=str(n), native_frames=frames, native_video=str(rel / video), source_path=str(rel / source_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', required=True, choices=list(SETS))
    ap.add_argument('--out', type=Path, default=benchmark_dir('flowfree'), help='FlowFree folder of the data')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--count', type=int, help='generate only this many puzzles per part (tests)')
    a = ap.parse_args()
    seen = {json.loads(p.read_text())['layout_sha256'] for p in (a.out / 'sources').rglob('*.json')}
    print('layouts already in the data', len(seen), flush=True)
    items, strata = [], Counter()
    for count, small, first, seed in SETS[a.set]:
        rng = random.Random(seed)
        count = a.count or count
        plan = []
        if small:
            for n_small, choices in zip(small, SMALL):
                plan += [rng.choices([s for s, _ in choices], weights=[w for _, w in choices])[0] for _ in range(n_small)]
        plan += [rng.choices([s for s, _ in STRATA], weights=[w for _, w in STRATA])[0] for _ in range(count - len(plan))]
        for i, (n, k) in enumerate(plan[:count]):
            while True:
                source = generate(n, k, PALETTE, rng)
                key = layout(n, source['endpoints'])
                if key not in seen and evaluate_symbolic(source)['success']:
                    break
            seen.add(key); strata[(n, k)] += 1
            source.update(generated=True, generator='warnsdorff_segments', layout_sha256=key, upstream_split=None, upstream_id=None)
            items.append((f'flowfree_{a.set}_{first + i:05d}', source, a.set, str(a.out)))
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(finish, items, chunksize=8))
    assert len({r['sample_id'] for r in rows}) == len(rows)
    with open(a.out / f'{a.set}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(json.dumps(dict(set=a.set, count=len(rows), strata={f'{n}x{n}_{k}colors': c for (n, k), c in sorted(strata.items())})),
          flush=True)


if __name__ == '__main__':
    main()
