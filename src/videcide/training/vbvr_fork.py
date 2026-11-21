from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

TARGET_BACKOFF = 4
BACK_STATES = {}
MAX_BACKOFF = 4
STILL = 1.5


def hanoi_distance(target: list[list[int]]):
    start = tuple(tuple(c) for c in target); dist = {start: 0}; q = deque([start])
    while q:
        s = q.popleft()
        for a in range(len(s)):
            if not s[a]:
                continue
            for b in range(len(s)):
                if a == b:
                    continue
                t = list(s); t[b] = s[b] + (s[a][-1],); t[a] = s[a][:-1]; t = tuple(t)
                if t not in dist:
                    dist[t] = dist[s] + 1; q.append(t)
    return dist


_SLIDE_DIST = {}


def slide_distance(target):
    key = tuple(target)
    if key not in _SLIDE_DIST:
        dist = {key: 0}; q = deque([key])
        while q:
            s = q.popleft(); b = s.index(0); r, c = divmod(b, 3)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 3 and 0 <= cc < 3:
                    t = list(s); j = rr * 3 + cc; t[b], t[j] = t[j], 0; t = tuple(t)
                    if t not in dist:
                        dist[t] = dist[s] + 1; q.append(t)
        _SLIDE_DIST[key] = dist
    return _SLIDE_DIST[key]


def slide_distance_ida(state, target, n, limit=40):
    goal = {v: divmod(i, n) for i, v in enumerate(target) if v}
    h0 = sum(abs(r - goal[v][0]) + abs(c - goal[v][1]) for i, v in enumerate(state) if v for r, c in [divmod(i, n)])
    s = list(state)

    def search(b, g, h, bound, prev):
        f = g + h
        if f > bound:
            return f
        if h == 0:
            return -1
        best = 10 ** 9; r, c = divmod(b, n)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < n and 0 <= cc < n):
                continue
            j = rr * n + cc
            if j == prev:
                continue
            v = s[j]; gr, gc = goal[v]
            dh = (abs(r - gr) + abs(c - gc)) - (abs(rr - gr) + abs(cc - gc))
            s[b], s[j] = v, 0
            t = search(j, g + 1, h + dh, bound, b)
            s[j], s[b] = v, 0
            if t == -1:
                return -1
            best = min(best, t)
        return best
    bound = h0
    while bound <= limit:
        t = search(s.index(0), 0, h0, bound, -1)
        if t == -1:
            return bound
        bound = t
    return None


def _slide_tuple(state, n):
    t = [0] * (n * n)
    for (r, c), v in state:
        t[r * n + c] = v
    return tuple(t)


_SOLVABLE = {}


def sokoban_solvable(grid, player, boxes, goals, max_states=300_000):
    key = (tuple(map(tuple, grid)), tuple(player), tuple(sorted(map(tuple, boxes))), tuple(sorted(map(tuple, goals))), max_states)
    if key not in _SOLVABLE:
        _SOLVABLE[key] = _sokoban_solvable(grid, player, boxes, goals, max_states)
    return _SOLVABLE[key]


def _sokoban_solvable(grid, player, boxes, goals, max_states=300_000):
    rows, cols = len(grid), len(grid[0]); goals = frozenset(map(tuple, goals))
    floor = lambda r, c: 0 <= r < rows and 0 <= c < cols and grid[r][c] != '#'
    dead = {(r, c) for r in range(rows) for c in range(cols) if floor(r, c) and (r, c) not in goals
            and (not floor(r - 1, c) or not floor(r + 1, c)) and (not floor(r, c - 1) or not floor(r, c + 1))}
    if any(tuple(b) in dead for b in boxes):
        return False
    start = (tuple(player), frozenset(map(tuple, boxes))); seen = {start}; q = deque([start])
    while q:
        (pr, pc), bx = q.popleft()
        if bx <= goals:
            return True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = pr + dr, pc + dc
            if not floor(nr, nc):
                continue
            nb = bx
            if (nr, nc) in bx:
                d = (nr + dr, nc + dc)
                if not floor(*d) or d in bx or d in dead:
                    continue
                nb = (bx - {(nr, nc)}) | {d}
            k = ((nr, nc), nb)
            if k not in seen:
                seen.add(k); q.append(k)
                if len(seen) > max_states:
                    return None
    return False


def death(task: str, m: dict[str, Any]) -> tuple[str, int] | None:
    cands = {}
    if m.get('violation'):
        f = m.get('violation_start') if m['violation'] == 'tile_hallucination' else None
        cands['violation:' + str(m['violation'])] = int(f if f is not None else m['violation_frame'])
    if task == 'sliding' and m.get('changes'):
        n = int(m['n'])
        if m.get('budget') is not None:
            tgt = _slide_tuple(m['target_state'], n)
            dist3 = slide_distance(tgt) if n == 3 else None
            for k, (i, st) in enumerate(m['changes']):
                d = dist3.get(_slide_tuple(st, 3)) if n == 3 else slide_distance_ida(_slide_tuple(st, n), tgt, n)
                if d is None or k + d > int(m['budget']):
                    cands['detour'] = int(i); break
        elif n == 3:
            dist = slide_distance(_slide_tuple(m['target_state'], 3)); prev = None
            for i, st in m['changes']:
                d = dist.get(_slide_tuple(st, 3))
                if prev is not None and (d is None or d >= prev):
                    cands['detour'] = int(i); break
                prev = d
        else:
            tgt = _slide_tuple(m['target_state'], n); prev = None
            for i, st in m['changes']:
                d = slide_distance_ida(_slide_tuple(st, n), tgt, n)
                if prev is not None and (d is None or d >= prev):
                    cands['detour'] = int(i); break
                prev = d
    if task == 'sokoban' and m.get('changes') and m.get('grid'):
        prev_boxes = None
        for i, (pl, bx) in m['changes']:
            if prev_boxes is not None and bx != prev_boxes and sokoban_solvable(m['grid'], pl, bx, m['goals']) is False:
                cands['unsolvable'] = int(i); break
            prev_boxes = bx
    if task == 'hanoi' and m.get('changes'):
        dist = hanoi_distance(m['target'])
        prev = None
        for i, st in m['changes']:
            d = dist.get(tuple(tuple(c) for c in st))
            if prev is not None and (d is None or d >= prev):
                cands['detour'] = int(i); break
            prev = d
    if not cands and not m.get('success') and m.get('eos_frame') is not None:
        cands['premature_eos'] = int(m['eos_frame'])
    if not cands:
        return None
    kind = min(cands, key=cands.get)
    return kind, cands[kind]


def select_fork(frames: np.ndarray, task: str, m: dict[str, Any], *, floor: int = 1,
                target_backoff: int | None = None) -> dict[str, Any]:
    if m.get('success'):
        return {'status': 'success'}
    d = death(task, m)
    if d is None:
        return {'status': 'no_death_found'}
    kind, frame = d
    changes = [(int(i), st) for i, st in (m.get('changes') or [])]
    if not changes:
        return select_fork_by_backoff(frames, task, m, floor=floor, target_backoff=target_backoff)
    f32 = frames.astype(np.float32)

    def still(f):
        nb = [g for g in (f - 1, f + 1) if 0 <= g < len(frames)]
        return all(float(np.abs(f32[f] - f32[g]).mean()) < STILL for g in nb)
    before = [i for i, _ in changes if i < frame]
    back = BACK_STATES.get(kind, 0)
    lo = before[-1 - back] if len(before) > back else 0
    tried = []
    for P in range(max(floor, -(-lo // 4) + 1), frame // 4 + 2):
        f = 4 * (P - 1)
        if f >= frame:
            break
        ok = P == 1 or still(f)
        tried.append({'prefix': P, 'frame': f, 'still': ok})
        if ok:
            death_latent = (frame - 1) // 4 + 1 if frame > 0 else 0
            return {'status': 'ok', 'rule': 'last_rest_state', 'kind': kind, 'death_frame': frame, 'death_latent': death_latent, 'prefix': P,
                    'fork_frame': f, 'rest_start': lo, 'backoff_latents': death_latent - P, 'tried': tried}
    P = max(floor, lo // 4 + 1); death_latent = (frame - 1) // 4 + 1 if frame > 0 else 0
    if 4 * (P - 1) < frame:
        return {'status': 'ok', 'rule': 'last_rest_state', 'kind': kind, 'death_frame': frame, 'death_latent': death_latent, 'prefix': P,
                'fork_frame': 4 * (P - 1), 'rest_start': lo, 'seam': 'boundary_before_rest', 'backoff_latents': death_latent - P,
                'tried': tried}
    return {'status': 'no_seam', 'rule': 'last_rest_state', 'kind': kind, 'death_frame': frame, 'rest_start': lo, 'tried': tried}


def select_fork_by_backoff(frames: np.ndarray, task: str, m: dict[str, Any], *, floor: int = 1,
                   target_backoff: int | None = None) -> dict[str, Any]:
    if m.get('success'):
        return {'status': 'success'}
    d = death(task, m)
    if d is None:
        return {'status': 'no_death_found'}
    kind, frame = d
    target_backoff = TARGET_BACKOFF if target_backoff is None else target_backoff
    death_latent = (frame - 1) // 4 + 1 if frame > 0 else 0
    top = max(floor, death_latent - target_backoff)
    f32 = frames.astype(np.float32)

    def still(f):
        nb = [g for g in (f - 1, f + 1) if 0 <= g < len(frames)]
        return all(float(np.abs(f32[f] - f32[g]).mean()) < STILL for g in nb)
    tried = []
    for P in range(top, max(floor, top - MAX_BACKOFF) - 1, -1):
        f = 4 * (P - 1)
        ok = P == 1 or still(f)
        tried.append({'prefix': P, 'frame': f, 'still': ok})
        if ok:
            return {'status': 'ok', 'kind': kind, 'death_frame': frame, 'death_latent': death_latent, 'prefix': P,
                    'fork_frame': f, 'backoff_latents': death_latent - P, 'target_backoff': target_backoff, 'tried': tried}
    return {'status': 'no_still_seam', 'kind': kind, 'death_frame': frame, 'tried': tried}
