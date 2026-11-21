from __future__ import annotations

import heapq
import itertools
from typing import Any

import numpy as np

SOLVER_BUDGET = 300_000
MAX_BACKOFF = 4
TARGET_BACKOFF = 4
MAX_TRANSIT = 2
DIRS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}


def solvable(grid: list[str], player, boxes, targets, budget: int = SOLVER_BUDGET) -> bool | None:
    floors = {(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch != '#'}
    goals = frozenset(tuple(t) for t in targets)
    dead = {p for p in floors - goals if
            ((p[0] - 1, p[1]) not in floors or (p[0] + 1, p[1]) not in floors) and
            ((p[0], p[1] - 1) not in floors or (p[0], p[1] + 1) not in floors)}
    if any(tuple(b) in dead for b in boxes):
        return False
    gl = tuple(sorted(goals)); assignments = tuple(itertools.permutations(gl)); cache = {}

    def bound(bx):
        if bx not in cache:
            cache[bx] = min(sum(abs(b[0] - t[0]) + abs(b[1] - t[1]) for b, t in zip(bx, ts)) for ts in assignments)
        return cache[bx]
    start = (tuple(player), tuple(sorted(tuple(b) for b in boxes)))
    costs = {start: 0}; heap = [(bound(start[1]), 0, start)]
    while heap:
        _, cost, state = heapq.heappop(heap)
        if costs[state] != cost:
            continue
        pl, bt = state; bx = frozenset(bt)
        if bx == goals:
            return True
        if len(costs) > budget:
            return None
        for dr, dc in DIRS.values():
            p = (pl[0] + dr, pl[1] + dc)
            if p not in floors:
                continue
            nb = bt
            if p in bx:
                t = (p[0] + dr, p[1] + dc)
                if t not in floors or t in bx or t in dead:
                    continue
                nb = tuple(sorted((bx - {p}) | {t}))
            new = (p, nb)
            if cost + 1 < costs.get(new, 1 << 30):
                costs[new] = cost + 1; heapq.heappush(heap, (cost + 1 + bound(nb), cost + 1, new))
    return False


def trace(frames: np.ndarray, source: dict[str, Any]) -> dict[str, Any]:
    from videcide.tasks.sokoban_checker import _import, level_from_source
    _, _, _, _, parse_rollout, is_eos = _import()
    level = level_from_source(source)
    p = parse_rollout(np.asarray(frames), level, 'classic', sustain_settle=2)
    eos = next((i for i, f in enumerate(frames) if is_eos(f)), None)
    end = eos if eos is not None else len(frames)
    st = p.per_frame_state; runs = []; i = 0
    while i < end:
        s = st[i]
        if s is None:
            i += 1; continue
        j = i
        while j + 1 < end and st[j + 1] == s:
            j += 1
        if j - i + 1 >= 2:
            runs.append((i, j, s))
        i = j + 1
    return {'parse': p, 'eos': eos, 'end': end, 'runs': runs, 'level': level}


def select_fork(frames: np.ndarray, source: dict[str, Any], *, floor: int = 1, tr: dict | None = None,
                override_fatal: int | None = None, require_solvable: bool = True,
                target_backoff: int | None = None) -> dict[str, Any]:
    tr = tr or trace(frames, source); p, runs, end, eos = tr['parse'], tr['runs'], tr['end'], tr['eos']
    target_backoff = TARGET_BACKOFF if target_backoff is None else target_backoff
    goals = frozenset(tuple(t) for t in source['targets']); grid = source['grid']
    cache: dict = {}

    def ok(s):
        key = (tuple(s.agent), tuple(sorted(s.boxes)))
        if key not in cache:
            cache[key] = solvable(grid, s.agent, s.boxes, goals)
        return cache[key]
    cands = {}
    for k, (a, b, s) in enumerate(runs):
        v = ok(s)
        if v is None:
            return {'status': 'unknown_solver', 'frame': a}
        if v is False:
            prev_end = runs[k - 1][1] if k else 0
            cands['solver_dead'] = (prev_end + 1, a)
            break
    if p.hall_pixel is not None:
        cands['hallucination'] = (p.hall_pixel, p.hall_pixel)
    if p.illegal_transitions:
        t = min(x['pixel_end'] for x in p.illegal_transitions)
        prev = max((b for a, b, _ in runs if b < t), default=0)
        cands['illegal_move'] = (prev + 1, t)
    solved = any(goals <= frozenset(s.boxes) for _, _, s in runs)
    if eos is not None and not solved and not cands:
        last = max((b for a, b, _ in runs if b < eos), default=0)
        cands['premature_eos'] = (last + 1, eos)
    if not cands:
        return {'status': 'success' if solved else 'no_death_found', 'eos': eos}
    kind = min(cands, key=lambda k: cands[k][0]); fatal, visible = cands[kind]
    if override_fatal is not None:
        kind, fatal, visible = 'visible_failure', int(override_fatal), int(override_fatal)
    elif target_backoff:
        fatal = max(1, fatal - 4 * int(target_backoff))
    settled = np.zeros(len(frames), bool); state_at = {}
    for a, b, s in runs:
        settled[a:b + 1] = True
        for f in range(a, b + 1):
            state_at[f] = s
    gap = np.zeros(len(frames), int); f = 0
    while f < end:
        if settled[f]:
            f += 1; continue
        g = f
        while g < end and not settled[g]:
            g += 1
        gap[f:g] = g - f; f = g
    onset = p.hall_pixel if p.hall_pixel is not None else 1 << 30
    top = (fatal - 1) // 4 + 1
    tried = []
    for P in range(top, max(floor, top - MAX_BACKOFF) - 1, -1):
        if P < floor:
            break
        f = 4 * (P - 1)
        latent = range(max(0, f - 3), f + 1)
        clean = bool(settled[f]) and all(settled[g] or gap[g] <= MAX_TRANSIT for g in latent) and max(latent) < onset
        state = state_at.get(f)
        good = bool(clean and state is not None and (ok(state) if require_solvable else True))
        tried.append({'prefix': P, 'frame': f, 'clean': bool(clean), 'solvable': None if state is None else ok(state)})
        if good:
            return {'status': 'ok', 'kind': kind, 'prefix': P, 'fork_frame': f, 'fatal_frame': fatal, 'target_backoff': target_backoff,
                    'death_frame': cands[kind][0] if kind in cands else fatal,
                    'visible_frame': visible, 'death_latent': (fatal - 1) // 4 + 1, 'backoff_latents': top - P,
                    'fork_state': {'agent': list(state.agent), 'boxes': sorted(map(list, state.boxes))},
                    'candidates': {k: list(v) for k, v in cands.items()}, 'tried': tried, 'eos': eos}
    return {'status': 'death_too_early' if top <= floor else 'no_clean_seam', 'kind': kind, 'fatal_frame': fatal,
            'candidates': {k: list(v) for k, v in cands.items()}, 'tried': tried, 'eos': eos}
