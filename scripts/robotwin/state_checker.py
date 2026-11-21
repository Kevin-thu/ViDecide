"""Rule checker of the RoboTwin-Plan verifier (Sec. B of the supplement).

check() turns the per-frame predictions of the state reader (state_reader.py infer) into a sequence of block
arrangements with a Viterbi decoder over all arrangements (with at most one block held), and reports every rule
violation: blocks that move without being grasped, duplicated, vanished or extra blocks, inconsistent reads, changes
between rest states that are not one legal move of a top block (including the size rule of Tower of Hanoi), a missing
EOS frame and a final arrangement that is not the goal. With appearance_path (DINOv2 patch distances of
appearance_scores.py), frames unlike every expert frame are flagged as well.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import cv2
import numpy as np

import scene
from scene import (APPEAR_PATCH_MAX, MIN_REST, OVERLAP_MIN_FRAMES, PERSIST, SEVERITY_MAX, arrangements,
                   first_persistent, legal_transition)

VERSION = 'state_checker'
SIGMA = 0.012
CLIP = -12.0
COST_PICK, COST_PLACE, COST_TELEPORT = 2.0, 2.0, 40.0
FAR = 0.03
MIN_CARRY = 6
MIN_SWITCH = 2
DUP_RATIO = 1.2
EXTRA_AREA = 0.3
MID, HELD_LOW, HELD_RUN = 4, 0.15, 4
GLITCH = 2
EDGE_PX = 16
SETTLE_TOL = 0.01


def states_of(nb, ns):
    out = []
    for held in [-1] + list(range(nb)):
        rest = [b for b in range(nb) if b != held]
        for arr in arrangements_of(rest, ns):
            out.append((arr, held))
    return out


def arrangements_of(blocks, ns):
    if not blocks:
        yield tuple(() for _ in range(ns)); return
    b, more = blocks[0], blocks[1:]
    for arr in arrangements_of(more, ns):
        for s_ in range(ns):
            for pos in range(len(arr[s_]) + 1):
                st = list(arr); st[s_] = arr[s_][:pos] + (b,) + arr[s_][pos:]; yield tuple(st)


def held_box(xyz, h, cam, size):
    x, y, z = xyz
    pts = np.array([scene.project((x + dx * h, y + dy * h, z + dz * h), cam) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)], np.float32)
    m = np.zeros((size, size), np.uint8); cv2.fillConvexPoly(m, cv2.convexHull(pts).astype(np.int32), 1); return m.astype(bool)


def colour_ratios(frames, states, xyz, geo):
    info = geo.info; cols = info['colours']; cam = geo.cam; size = cam['size']; nb = len(cols)
    card = np.zeros((size, size), bool)
    if geo.card is not None:
        x0, y0, x1, y1 = [int(v) for v in geo.card]; card[max(0, y0):y1, max(0, x0):x1] = True
    card[:EDGE_PX] = card[-EDGE_PX:] = True; card[:, :EDGE_PX] = card[:, -EDGE_PX:] = True

    def sil_of(arr, held, t, b):
        if held == b:
            return held_box(xyz[t, b], geo.sizes[b], cam, size)
        s_ = next(i for i, st in enumerate(arr) if b in st); st = arr[s_]
        return geo.cell_sil[(b, s_, round(sum(2 * geo.sizes[j] for j in st[:st.index(b)]), 4))]

    def hsv_of(f):
        f = f if f.shape[0] == size else cv2.resize(f, (size, size)); h = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        return h, (h[..., 1] > 120) & (h[..., 2] > 70)
    hsv0, sat0 = hsv_of(frames[0]); arr0 = states[0][0]
    blocks0 = np.zeros((size, size), np.uint8)
    for b in range(nb):
        blocks0 |= sil_of(arr0, -1, 0, b).astype(np.uint8)
    blocks0 = cv2.dilate(blocks0, np.ones((7, 7), np.uint8)).astype(bool)
    static = {c: cv2.dilate((scene.hue_mask(hsv0, c) & sat0 & ~blocks0).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) | card
              for c in scene.HUES}
    others = [c for c in scene.HUES if c not in cols]
    for q, pc in zip(geo.pad_quads, info.get('pads') or []):
        if pc in static:
            static[pc] |= q
    out = np.zeros((min(len(frames), len(states)), nb)); extra = np.zeros(len(out))
    observable = np.zeros_like(out)
    for t in range(len(out)):
        hsv, sat = hsv_of(frames[t]); arr, held = states[t]
        extra[t] = max([(scene.hue_mask(hsv, c) & sat & ~static[c]).sum() / geo.cube_area for c in others] or [0.])
        for b, c in enumerate(cols):
            sil = sil_of(arr, held, t, b)
            visible_area = max((sil & ~static[c]).sum(), 0.3 * sil.sum())
            den = sil.sum() if held == b else visible_area
            pixels = (scene.hue_mask(hsv, c) & sat & ~static[c]).sum()
            out[t, b] = pixels / max(den, 1.0)
            observable[t, b] = pixels / max(visible_area, 1.0)
    return out, extra, observable


def check(row, video_path, reads_path, appearance_path=None, debug=False):
    frames_all = scene.read_visible(video_path); eos = scene.find_eos_frame(frames_all)
    reads = json.loads(Path(reads_path).read_text()); T = len(reads)
    geo = scene.Geometry(row); info = geo.info; cols = info['colours']; nb = len(cols); slots = np.array(info['slots'], float)
    ns = len(slots); sizes = np.array(geo.sizes, float); floor = geo.floor; cam = geo.cam; e = geo.edge_px
    sts = states_of(nb, ns); index = {s: i for i, s in enumerate(sts)}; S_ = len(sts)
    cen = np.full((S_, nb, 3), np.nan)
    for i, (arr, held) in enumerate(sts):
        for s_, stack in enumerate(arr):
            h = 0.
            for b in stack:
                cen[i, b] = (slots[s_][0], slots[s_][1], floor + h + sizes[b]); h += 2 * sizes[b]
    held_of = np.array([h for _, h in sts])
    xyz = np.array([[r['blocks'][c]['xyz'] for c in cols] for r in reads])
    ph = np.clip(np.array([[r['blocks'][c]['held'] for c in cols] for r in reads]), 1e-4, 1 - 1e-4)
    pv = np.array([[r['blocks'][c]['visible'] for c in cols] for r in reads])
    E = np.zeros((T, S_))
    for b in range(nb):
        d2 = ((xyz[:, None, b, :] - cen[None, :, b, :]) ** 2).sum(-1) / (2 * SIGMA ** 2)
        rest_ll = np.log(1 - ph[:, b])[:, None] + np.maximum(-pv[:, b][:, None] * d2, CLIP)
        held_ll = np.log(ph[:, b])[:, None] + np.zeros((1, S_))
        E += np.where(held_of[None, :] == b, held_ll, rest_ll)
    src, dst, cost, kind = [], [], [], []

    def add(a, b_, c, k):
        src.append(a); dst.append(b_); cost.append(c); kind.append(k)
    for i, (arr, held) in enumerate(sts):
        add(i, i, 0., 0)
        if held == -1:
            for s_, stack in enumerate(arr):
                if stack:
                    top = stack[-1]; rest = list(arr); rest[s_] = stack[:-1]
                    add(i, index[(tuple(rest), top)], COST_PICK, 1)
                    for s2 in range(ns):
                        if s2 != s_:
                            moved = list(rest); moved[s2] = rest[s2] + (top,)
                            add(i, index[(tuple(moved), -1)], COST_TELEPORT, 3)
        else:
            for s_ in range(ns):
                placed = list(arr); placed[s_] = arr[s_] + (held,)
                add(i, index[(tuple(placed), -1)], COST_PLACE, 2)
    src, dst, cost, kind = map(np.array, (src, dst, cost, kind))
    order = np.lexsort((src, dst)); src, dst, cost, kind = src[order], dst[order], cost[order], kind[order]
    starts = np.r_[0, np.nonzero(np.diff(dst))[0] + 1]
    start = tuple(tuple(s) for s in info['start'])
    V = np.full(S_, -1e9); V[index[(start, -1)]] = 0.; V += E[0]
    back = np.zeros((T, S_), np.int64); back_kind = np.zeros((T, S_), np.int8)
    for t in range(1, T):
        cand = V[src] - cost
        best = np.maximum.reduceat(cand, starts)
        hit = cand == np.repeat(best, np.diff(np.r_[starts, len(cand)]))
        first = np.minimum.reduceat(np.where(hit, np.arange(len(cand)), len(cand)), starts)
        back[t] = src[first]; back_kind[t] = kind[first]
        V = best + E[t]
    path = [int(np.argmax(V))]; kinds = []
    for t in range(T - 1, 0, -1):
        kinds.append(int(back_kind[t, path[-1]])); path.append(int(back[t, path[-1]]))
    path = path[::-1]; kinds = [0] + kinds[::-1]
    violations = []
    arms = [[(a['x'], a['y']) for a in r['arms'] if a['in_view'] > 0.5] for r in reads]

    def arm_near(b, cell_xyz, t):
        p = scene.project(tuple(cell_xyz), cam)
        return any(np.hypot(x - p[0], y - p[1]) < 1.5 * e for u in range(max(0, t - 2), min(T, t + 3)) for x, y in arms[u])
    def slot_of(i, b):
        return next((s_ for s_, st in enumerate(sts[i][0]) if b in st), None)
    events = []
    for t in range(1, T):
        if kinds[t] == 1:
            b = sts[path[t]][1]; events.append((t, 1, b, slot_of(path[t - 1], b)))
        elif kinds[t] == 2:
            b = sts[path[t - 1]][1]; events.append((t, 2, b, slot_of(path[t], b)))
    clean, k_ = [], 0
    while k_ < len(events):
        if k_ + 1 < len(events):
            (t0, k0, b0, s0), (t1, k1, b1, s1) = events[k_], events[k_ + 1]
            if b0 == b1 and s0 == s1 and k0 != k1 and t1 - t0 <= GLITCH:
                k_ += 2; continue
        clean.append(events[k_]); k_ += 1
    for t in range(1, T):
        if kinds[t] == 3:
            b = next(b for b in range(nb) if not np.allclose(cen[path[t - 1], b], cen[path[t], b]))
            violations.append({'type': 'moved_without_gripper', 'frame': t, 'block': cols[b]})
    for t, k, b, s_ in clean:
        if k == 1 and not arm_near(b, cen[path[t - 1], b], t):
            violations.append({'type': 'held_without_pick', 'frame': t, 'block': cols[b]})
        elif k == 2 and not arm_near(b, cen[path[t], b], t):
            violations.append({'type': 'placed_without_gripper', 'frame': t, 'block': cols[b]})
    for (t0, k0, b0, s0), (t1, k1, b1, s1) in zip(clean, clean[1:]):
        if k0 == 1 and k1 == 2 and b0 == b1 and s0 != s1 and t1 - t0 < MIN_CARRY:
            violations.append({'type': 'moved_without_gripper', 'frame': t1, 'block': cols[b1], 'why': f'carry {t1 - t0} frames'})
        elif k0 == 2 and k1 == 1 and s0 != s1 and t1 - t0 < MIN_SWITCH:
            violations.append({'type': 'held_without_pick', 'frame': t1, 'block': cols[b1], 'why': f'pick {t1 - t0} frames after a put-down elsewhere'})
    bad = []
    for t in range(T):
        i = path[t]; flag = None
        for b in range(nb):
            under = held_of[i] != -1 and np.linalg.norm(xyz[t, held_of[i], :2] - cen[i, b, :2]) < FAR
            if pv[t, b] > 0.8 and held_of[i] != b and not under and np.linalg.norm(xyz[t, b] - cen[i, b]) > FAR:
                flag = cols[b]
        bad.append(flag)
    t = first_persistent([f is not None for f in bad])
    if t is not None:
        violations.append({'type': 'inconsistent', 'frame': t, 'block': bad[t]})
    ratio, extra, observable = colour_ratios(frames_all, [sts[i] for i in path], xyz, geo)
    existence = ratio
    t = first_persistent(list((ratio > DUP_RATIO).any(1)))
    if t is not None:
        violations.append({'type': 'duplicate', 'frame': t, 'block': cols[int(ratio[t].argmax())], 'ratio': round(float(ratio[t].max()), 2)})
    t = first_persistent(list(extra > EXTRA_AREA))
    if t is not None:
        violations.append({'type': 'extra_object', 'frame': t, 'area_cubes': round(float(extra[t]), 2)})
    hs_ = [sts[i][1] for i in path]; t = 0
    while t < len(ratio):
        if hs_[t] < 0:
            t += 1; continue
        t0 = t
        while t < len(ratio) and hs_[t] == hs_[t0]:
            t += 1
        run = 0
        for u in range(t0 + MID, t - MID):
            run = run + 1 if existence[u, hs_[t0]] < HELD_LOW else 0
            if run >= HELD_RUN:
                violations.append({'type': 'vanished', 'frame': u - HELD_RUN + 1, 'block': cols[hs_[t0]]}); break
        else:
            continue
        break
    rests = []
    for t in range(T):
        arr, held = sts[path[t]]
        if held != -1:
            continue
        if rests and rests[-1]['state'] == arr and rests[-1]['end'] == t:
            rests[-1]['end'] = t + 1
        else:
            rests.append({'state': arr, 'start': t, 'end': t + 1})
    rests = [r for r in rests if r['end'] - r['start'] >= MIN_REST]
    merged = []
    for r in rests:
        if merged and merged[-1]['state'] == r['state']:
            merged[-1]['end'] = r['end']
        else:
            merged.append(r)
    rests = merged
    goal = tuple(tuple(s) for s in info['goal']); hs = info['sizes'] if info['mode'] == 'hanoi' or info.get('size_rule') else None
    if not rests:
        violations.append({'type': 'no_rest_state', 'frame': 0})
    else:
        if rests[0]['state'] != start:
            violations.append({'type': 'start_changed', 'frame': rests[0]['start'], 'state': rests[0]['state']})
        all_states = None
        for a, b in zip(rests, rests[1:]):
            if legal_transition(a['state'], b['state'], hs):
                continue
            if b['start'] - a['end'] >= OVERLAP_MIN_FRAMES:
                all_states = all_states or list(arrangements(nb, ns))
                if any(legal_transition(a['state'], m, hs) and legal_transition(m, b['state'], hs) for m in all_states):
                    continue
            violations.append({'type': 'illegal_move', 'frame': b['start'], 'before': a['state'], 'after': b['state']})
    final = sts[path[-1]][0] if sts[path[-1]][1] == -1 else (rests[-1]['state'] if rests else None)
    arr, held = sts[path[-1]]
    if held != -1:
        p = xyz[-1, held]
        for s_ in range(ns):
            placed = list(arr); placed[s_] = arr[s_] + (held,); j = index[(tuple(placed), -1)]
            if np.linalg.norm(p - cen[j, held]) < SETTLE_TOL:
                final = tuple(placed)
                if rests and not legal_transition(rests[-1]['state'], final, hs):
                    violations.append({'type': 'illegal_move', 'frame': T, 'before': rests[-1]['state'], 'after': final})
                break
    if eos is None:
        violations.append({'type': 'no_eos', 'frame': T})
    elif final != goal:
        violations.append({'type': 'ended_not_goal', 'frame': T, 'final': final})
    if appearance_path and Path(appearance_path).exists():
        a = np.load(appearance_path).astype(np.float32)[:T]
        excess = np.clip(a - APPEAR_PATCH_MAX, 0, None).reshape(len(a), -1).sum(1); total = np.cumsum(excess)
        if len(total) and total[-1] > SEVERITY_MAX:
            violations.append({'type': 'unlike_expert_picture', 'frame': int(np.argmax(total > SEVERITY_MAX)), 'severity': float(total[-1])})
    violations.sort(key=lambda v: v['frame'])
    out = {'success': not violations, 'violation': violations[0]['type'] if violations else None, 'violations': violations,
           'goal_reached': final == goal, 'eos_frame': eos,
           'checker': VERSION,
           'rest_states': [{**r, 'state': [list(s) for s in r['state']]} for r in rests]}
    if debug:
        out['debug'] = {'path': [sts[i] for i in path], 'kinds': kinds, 'bad': bad, 'ratio': ratio.round(3).tolist(), 'extra': extra.round(3).tolist()}
    return out
