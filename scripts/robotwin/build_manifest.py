"""Build RoboTwin-Plan manifest rows from rendered episodes (render_episodes.py).

    python scripts/robotwin/build_manifest.py --tasks blocksworld_tri,hanoi_tri --split train --out new_train.jsonl \
        [--episodes data/RoboTwin-Plan/episodes] [--seeds 0:1000] [--exclude data/RoboTwin-Plan/test.jsonl ...]

An episode becomes a row only if it passes the gate used for every released episode: replaying the simulator state of
every frame gives a symbolic arrangement, the last one must be the goal with every block at rest, and every goal block
must lie within 1 cm of its place. Episodes whose puzzle (start, goal, colours and pads) already occurs in an --exclude
manifest or earlier in the output are skipped, so that the splits share no puzzle. Rows have the format of the released
manifests; their paths are relative to the data root.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scene

XY_TOL, Z_TOL = 0.035, 0.006    # a block rests on a place if within 3.5 cm of it and at its stack height (+-6 mm)
GOAL_TOL = 0.01                 # every goal block within 1 cm of its place (or of the block below)


def load(row):
    meta = json.loads(scene.data_path(row['source_path']).read_text())
    states = json.loads(scene.data_path(row['state_path']).read_text())
    return meta, meta['info']['info'], states


def frame_state(actors, info, floor):
    n = len(info['sizes'])
    pos = np.array([actors[f'block{b + 1}'][:3] for b in range(n)])
    slots = np.asarray(info['slots'])
    slot_of = []
    for b in range(n):
        d = np.linalg.norm(slots - pos[b, :2], axis=1)
        slot_of.append(int(d.argmin()) if d.min() < XY_TOL else None)
    stacks, resting = [[] for _ in slots], [False] * n
    for s in range(len(slots)):
        members = sorted([b for b in range(n) if slot_of[b] == s], key=lambda b: pos[b, 2])
        level = floor + (.004 if info['pads'][s] is not None else 0.)
        for b in members:
            if abs(pos[b, 2] - (level + info['sizes'][b])) > Z_TOL:
                break
            resting[b] = True; stacks[s].append(b)
            level += 2 * info['sizes'][b]
    state = tuple(map(tuple, stacks)) if all(resting) else None
    return state, resting


def floor_height(states, info):
    zs = []
    for s, stack in enumerate(info['start']):
        below = .004 if info['pads'][s] is not None else 0.
        for b in stack:
            zs.append(states[0]['actors'][f'block{b + 1}'][2] - info['sizes'][b] - below)
            below += 2 * info['sizes'][b]
    return float(np.median(zs))


def gt_sequence(row):
    """Symbolic arrangement of every frame (None while a block is lifted), padded to the length of the video."""
    meta, info, states = load(row)
    floor = floor_height(states, info)
    seq = [frame_state(s['actors'], info, floor)[0] for s in states]
    seq += [seq[-1]] * (int(meta['frames']) - len(seq))
    return seq, info, meta


def gate(row):
    """(passed, reason) for an episode row."""
    try:
        seq, info, meta = gt_sequence(row)
    except Exception as e:
        return False, f'unreadable {e!r}'
    if seq[-1] != tuple(map(tuple, info['goal'])):
        return False, 'final arrangement is not the goal at rest'
    states = json.loads(scene.data_path(row['state_path']).read_text())
    a = states[-1]['actors']
    for slot, stack in enumerate(info['goal']):
        prev = np.asarray(info['slots'][slot])
        for b in stack:
            p = np.asarray(a[f'block{b + 1}'][:2])
            if np.linalg.norm(p - prev) > GOAL_TOL:
                return False, f'goal block {b} off by {np.linalg.norm(p - prev) * 100:.1f} cm'
            prev = p
    return True, 'ok'


def puzzle_key(meta):
    """Hash of the puzzle shown in an episode: start, goal, block colours and pad colours."""
    i = meta['info']['info']
    return hashlib.sha256(json.dumps([i['start'], i['goal'], i['colours'], i['pads']], sort_keys=True).encode()).hexdigest()


def relative(path: Path) -> str:
    """Path relative to the data root (symlinks are kept, not resolved), or absolute outside it."""
    path, root = Path(os.path.abspath(path)), Path(os.path.abspath(scene.DATA))
    return path.relative_to(root).as_posix() if root in path.parents else str(path)


def episode_row(folder: Path, split: str) -> dict:
    """Manifest row of a rendered episode folder <episodes>/<episode task>/<seed>."""
    meta = json.loads((folder / 'meta.json').read_text())
    task = meta['task']
    episode, family, subtask, prompt = scene.TASKS[task]
    info = meta['info']['info']; seed = int(meta['seed'])
    return {'sample_id': f'robotwin_{episode}_{seed:07d}', 'task': family, 'subtask': subtask, 'split': split,
            'prompt': prompt, 'seed': seed, 'blocks': len(info['colours']), 'moves': len(info['plan']),
            'native_frames': int(meta['frames']), 'native_video': relative(folder / 'video.mp4'),
            'source_path': relative(folder / 'meta.json'), 'state_path': relative(folder / 'state.json')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tasks', required=True, help=f'comma-separated RoboTwin tasks: {", ".join(scene.TASKS)}')
    ap.add_argument('--split', required=True, help='split field of the rows (train, val, test)')
    ap.add_argument('--out', type=Path, required=True, help='output manifest (JSONL)')
    ap.add_argument('--episodes', type=Path, default=scene.DATA / 'RoboTwin-Plan' / 'episodes')
    ap.add_argument('--seeds', help='start:stop range of seeds (default: every rendered episode)')
    ap.add_argument('--exclude', type=Path, nargs='*', default=[], help='manifests whose puzzles must not repeat')
    ap.add_argument('--limit', type=int, default=0, help='at most this many rows per task (in seed order)')
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split(':')) if a.seeds else (0, 1 << 62)
    seen = set()
    for path in a.exclude:
        seen |= {puzzle_key(json.loads(scene.data_path(r['source_path']).read_text())) for r in scene.read_manifest(path)}
    rows, report = [], Counter()
    for task in a.tasks.split(','):
        folder = a.episodes / scene.TASKS[task][0]
        episodes = sorted((d for d in folder.iterdir() if d.name.isdigit() and lo <= int(d.name) < hi), key=lambda d: int(d.name)) \
            if folder.is_dir() else []
        kept = 0
        for d in episodes:
            if a.limit and kept >= a.limit:
                break
            if not all((d / f).exists() for f in ('video.mp4', 'state.json', 'meta.json')):
                report[f'{task}: incomplete episode'] += 1; continue
            row = episode_row(d, a.split)
            ok, why = gate(row)
            if not ok:
                report[f'{task}: {why.split(" by ")[0]}'] += 1; continue
            key = puzzle_key(json.loads((d / 'meta.json').read_text()))
            if key in seen:
                report[f'{task}: puzzle already used'] += 1; continue
            seen.add(key); rows.append(row); kept += 1
        report[f'{task}: kept'] = kept
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(json.dumps({'rows': len(rows), 'by_subtask': Counter(r['subtask'] for r in rows), 'report': report},
                     indent=1, default=dict))


if __name__ == '__main__':
    main()
