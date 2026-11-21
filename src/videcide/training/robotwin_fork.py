"""Fork point of a failed RoboTwin-Plan rollout for post-training (Sec. 3.4 and C.4 of the paper).

The rollout's rest states (from the verifier) are compared with the shortest plan: the first detour (a move that does
not bring the arrangement closer to the goal) or the first rule violation marks the failure, and the fork is the first
still latent of the rest state before it. A rollout that reaches the goal but fails is forked before its first
violation, its strongest appearance defect, or the goal state it did not stop at.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import Any

import numpy as np

PHYSICAL_EXCLUDE = {'ended_not_goal', 'no_eos'}


def _moves(state, sizes):
    for s, src in enumerate(state):
        if not src:
            continue
        b = src[-1]
        for d, dst in enumerate(state):
            if d == s or (sizes is not None and dst and sizes[dst[-1]] < sizes[b]):
                continue
            new = [list(x) for x in state]; new[s].pop(); new[d].append(b)
            yield tuple(tuple(x) for x in new)


def _canon(state) -> tuple:
    return tuple(tuple(int(b) for b in s) for s in state)


@lru_cache(maxsize=4096)
def distances(goal: tuple, sizes: tuple | None, n_places: int, blocks: tuple) -> dict:
    dist = {goal: 0}; q = deque([goal])
    while q:
        st = q.popleft()
        for nxt in _moves(st, None if sizes is None else list(sizes)):
            if nxt not in dist:
                dist[nxt] = dist[st] + 1; q.append(nxt)
    return dist


def select_fork(result: dict[str, Any], info: dict[str, Any], max_latents: int = 49, appearance_excess=None) -> dict[str, Any]:
    rests = result.get('rest_states') or []
    if not rests:
        return {'status': 'no_rest_states'}
    goal = _canon(info['goal'])
    sizes = tuple(info['sizes']) if info.get('mode') == 'hanoi' else None
    blocks = tuple(sorted(b for s in info['start'] for b in s))
    dist = distances(goal, sizes, len(goal), blocks)
    states = [_canon(r['state']) for r in rests]
    d = [dist.get(s) for s in states]
    phys = [v for v in result.get('violations') or [] if v.get('type') not in PHYSICAL_EXCLUDE | {'unlike_expert_picture'}
            and v.get('frame') is not None]
    vframe = min((int(v['frame']) for v in phys), default=None)
    before = lambda f: max([i for i, r in enumerate(rests) if int(r['end']) < f], default=0)
    cands = []
    if not result.get('goal_reached'):
        detour = next((i for i in range(len(rests) - 1) if d[i] is None or d[i + 1] is None or d[i + 1] >= d[i]), None)
        if detour is not None:
            cands.append((detour, 'detour', int(rests[detour]['end']) + 1))
        if vframe is not None:
            cands.append((before(vframe), 'violation:' + min(phys, key=lambda v: int(v['frame']))['type'], vframe))
        if not cands:
            return {'status': 'no_fork_slow' if d and d[-1] else 'no_fork', 'distances': d}
    else:
        if vframe is not None:
            cands.append((before(vframe), 'violation:' + min(phys, key=lambda v: int(v['frame']))['type'], vframe))
        elif any(v.get('type') == 'unlike_expert_picture' for v in result.get('violations') or []) and appearance_excess is not None \
                and len(appearance_excess):
            f = int(np.argmax(appearance_excess)); cands.append((before(f), 'appearance_peak', f))
        elif any(v.get('type') == 'no_eos' for v in result.get('violations') or []):
            k = next((i for i, x in enumerate(d) if x == 0), None)
            if k is None:
                return {'status': 'no_fork_goal_unrested', 'distances': d}
            cands.append((k, 'no_stop_at_goal', int(rests[k]['end'])))
        if not cands:
            return {'status': 'no_fork', 'distances': d}
    k, reason, frame = min(cands, key=lambda c: (c[0], c[2]))
    r = rests[k]; lo, hi = int(r['start']), int(r['end'])
    p = max(2, -(-lo // 4) + 1)
    if 4 * (p - 1) > hi:
        return {'status': 'no_still_latent', 'rest': r, 'death': reason, 'death_frame': frame, 'distances': d}
    if p >= max_latents - 1:
        return {'status': 'too_late', 'prefix': p, 'death': reason, 'death_frame': frame, 'distances': d}
    return {'status': 'ok', 'prefix': p, 'rest_index': k, 'rest': r, 'fork_state': [list(s) for s in states[k]],
            'moves_left': d[k], 'death': reason, 'death_frame': frame, 'death_latent': 0 if frame == 0 else (frame - 1) // 4 + 1,
            'distances': d, 'latents_left': max_latents - p}
