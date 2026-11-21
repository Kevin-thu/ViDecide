from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..paths import DATA_ROOT, data_path, manifest_path
from .video_parser import find_eos_frame

DATA = DATA_ROOT
GHOST = 1.6
S = 384


def read(path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path)); out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        out.append(f if f.shape[0] == S else cv2.resize(f, (S, S), interpolation=cv2.INTER_AREA))
    return np.stack(out)


def puzzle(row: dict) -> dict:
    """Parameters of a VBVR-MultiStep puzzle (the ``parameters`` of its source file, read by the checkers)."""
    return json.loads(data_path(row["source_path"]).read_text())["parameters"]


def content(frames):
    eos = find_eos_frame(frames)
    return frames[:eos] if eos is not None else frames, eos


def result(ok_goal, violation, eos, **extra):
    return {"success": bool(ok_goal and violation is None and eos is not None),
            "violation": violation[0] if violation else None, "violation_frame": violation[1] if violation else None,
            "goal_reached": bool(ok_goal), "eos_detected": eos is not None, "eos_frame": eos, **extra}


def patch_mean(img, y, x, r):
    y, x, r = int(round(y)), int(round(x)), max(1, int(round(r)))
    return img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3).astype(np.float32).mean(0)


DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def _sokoban_moves(grid, state):
    (a, bxs) = state
    for d in DIRS.values():
        n = (a[0] + d[0], a[1] + d[1])
        if grid[n[0]][n[1]] == "#":
            continue
        if n in bxs:
            dest = (n[0] + d[0], n[1] + d[1])
            if grid[dest[0]][dest[1]] == "#" or dest in bxs:
                continue
            yield (n, (bxs - {n}) | {dest})
        else:
            yield (n, bxs)


def _sokoban_reachable(grid, s0, s1, k):
    frontier = {s0}
    for _ in range(k):
        frontier = {t for s in frontier for t in _sokoban_moves(grid, s)}
        if s1 in frontier:
            return True
    return False


def sokoban(frames, meta, ref):
    st = meta["puzzle_state"]; grid = st["grid"]; R, C = len(grid), len(grid[0])
    goals = {tuple(g) for g in st["goals"]}; boxes0 = frozenset(tuple(b) for b in st["initial_boxes"])
    bright = ref[0].astype(np.int16).mean(-1) > 70
    ys_, xs_ = np.nonzero(bright)
    oy, ox = ys_.min(), xs_.min()
    ch, cw = (ys_.max() + 1 - oy) / R, (xs_.max() + 1 - ox) / C
    centre = lambda r, c: (oy + (r + 0.5) * ch, ox + (c + 0.5) * cw)
    rad = 0.33 * min(ch, cw)

    def frac(img, r, c):
        y, x = centre(r, c); y0, y1, x0, x1 = int(y - rad), int(y + rad) + 1, int(x - rad), int(x + rad) + 1
        p = img[max(0, y0):y1, max(0, x0):x1].reshape(-1, 3).astype(np.int16)
        R_, G_, B_ = p[:, 0], p[:, 1], p[:, 2]
        red_face = ((R_ > 180) & (G_ < 140) & (B_ < 140)).mean()
        yellow = ((R_ > 200) & (G_ > 170) & (B_ < 110)).mean()
        dark = ((R_ < 70) & (G_ < 70) & (B_ < 70)).mean()
        orange = ((R_ > 200) & (G_ > 120) & (G_ < 200) & (B_ < 90)).mean()
        on_goal = (r, c) in bare_yellow and (yellow + orange) > 0.3 and yellow < 0.8 * bare_yellow[(r, c)]
        agent = 1.0 if (red_face > 0.2 or orange > 0.22 or (yellow > 0.12 and dark > 0.01) or on_goal) else 0.0
        agent *= max(red_face, orange, yellow + orange)
        box = ((B_ > 140) & (R_ < 110) & (G_ < 160)).mean()
        boxg = ((G_ > 110) & (R_ < 110) & (B_ < 110) & (G_ > R_ + 30)).mean()
        return agent, box, boxg

    bare_yellow = {}
    for g in goals:
        img = ref[0] if g not in boxes0 and g != tuple(st["initial_player"]) else content(ref)[0][-1]
        y, x = centre(*g); p_ = img[int(y - rad):int(y + rad) + 1, int(x - rad):int(x + rad) + 1].reshape(-1, 3).astype(np.int16)
        bare_yellow[g] = float(((p_[:, 0] > 200) & (p_[:, 1] > 170) & (p_[:, 2] < 110)).mean())

    def decode(img):
        agents, bxs = [], set()
        for r in range(R):
            for c in range(C):
                if grid[r][c] == "#":
                    continue
                agent, box, boxg = frac(img, r, c)
                if agent > 0:
                    agents.append((agent, (r, c)))
                elif max(box, boxg) > 0.25:
                    bxs.add((r, c))
        agents.sort(reverse=True)
        if len(agents) > 1 and agents[0][0] > GHOST * agents[1][0]:
            agents = agents[:1]
        return [p for _, p in agents], frozenset(bxs)

    vis, eos = content(frames)
    states, violation, last_i = [], None, 0
    bad_run, changes = 0, []
    for i, img in enumerate(vis):
        agents, bxs = decode(img)
        if len(agents) != 1 or len(bxs) != len(boxes0):
            bad_run += 1
            need = 2 if len(agents) != 1 else 3
            if bad_run >= need:
                violation = violation or ("hallucination", i - need + 1)
            continue
        bad_run = 0
        cur = (agents[0], bxs)
        if states and cur != states[-1]:
            if not _sokoban_reachable(grid, states[-1], cur, min(i - last_i, 3)):
                violation = violation or ("illegal_move", i)
        if not states or cur != states[-1]:
            changes.append((i, [list(cur[0]), sorted(list(b) for b in cur[1])]))
        last_i = i
        states.append(cur)
    if states and states[0] != (tuple(st["initial_player"]), boxes0):
        violation = violation or ("wrong_start", 0)
    goal = bool(states) and set(states[-1][1]) == goals
    return result(goal, violation, eos, states=len(states), changes=changes, grid=grid, goals=sorted(list(g) for g in goals))


SLIDE_BANK = DATA / "VBVR-MultiStep/verifier/sliding_digit_bank.npz"   # digit templates of the Sliding Puzzle tiles
SLIDE_PERSIST = 6
SLIDE_STILL = 1.5
SLIDE_DUP_PERSIST = 10 ** 9
HANOI_AIR_MAX = 24


def _slide_geom(meta, ref):
    n = int(meta["size"])
    hsv = cv2.cvtColor(ref[0], cv2.COLOR_RGB2HSV)
    ys, xs = np.nonzero(hsv[..., 1] > 80)
    y0, x0 = ys.min(), xs.min()
    ch, cw = (ys.max() - y0 + 1) / n, (xs.max() - x0 + 1) / n

    def cell(img, r, c):
        return img[int(y0 + r * ch + 0.15 * ch):int(y0 + r * ch + 0.85 * ch),
                   int(x0 + c * cw + 0.15 * cw):int(x0 + c * cw + 0.85 * cw)].astype(np.int16)
    return n, cell


def _digit_feat(a):
    w = a.min(-1).astype(np.float32); bg = float(np.median(w))
    ys, xs = np.nonzero(w > bg + 0.6 * (255 - bg))
    if len(ys) < 3:
        return np.zeros(16 * 12, np.float32)
    g = a.astype(np.float32).mean(-1)
    f = cv2.resize(g[max(0, ys.min() - 1):ys.max() + 2, max(0, xs.min() - 1):xs.max() + 2], (12, 16),
                   interpolation=cv2.INTER_AREA).ravel()
    f = f - f.mean()
    return f / (np.linalg.norm(f) + 1e-6)


def _slide_replay(vis, meta, ref):
    n, cell = _slide_geom(meta, ref)
    start = {tuple(o["initial_position"]): o["value"] for o in meta["objects"] if o.get("value")}
    blank = next((r, c) for r in range(n) for c in range(n) if (r, c) not in start)
    sat = lambda img, p: float((cell(img, *p).max(-1) - cell(img, *p).min(-1)).mean())
    tile_sat = float(np.median([sat(ref[0], p) for p in start]))
    state, bad_run, snaps = dict(start), 0, []
    for i, img in enumerate(vis):
        sats = {(r, c): sat(img, (r, c)) for r in range(n) for c in range(n)}
        blanks = [p for p, v in sats.items() if v < 25]
        full = [p for p, v in sats.items() if v > 0.8 * tile_sat]
        if len(blanks) != 1 or len(full) != n * n - 1:
            bad_run = bad_run + 1 if len(blanks) == 0 or len(blanks) > 1 and len(full) == n * n - len(blanks) else 0
            if bad_run >= 4:
                return snaps, ("blank_hallucination", i - 3), state
            continue
        bad_run = 0
        b = blanks[0]
        if b != blank:
            if abs(b[0] - blank[0]) + abs(b[1] - blank[1]) != 1:
                return snaps, ("illegal_move", i), state
            state[blank] = state.pop(b); blank = b
        snaps.append((i, dict(state)))
    return snaps, None, state


def build_sliding_bank(n_videos: int = 60):
    rows = [json.loads(l) for l in open(manifest_path("vbvr", "train"))]
    rows = sorted((r for r in rows if r["task"] == "vbvr_sliding"), key=lambda r: r["sample_id"])
    bank = {}
    for r in rows[:n_videos]:
        meta = puzzle(r)
        ref = read(data_path(r["native_video"])); n, cell = _slide_geom(meta, ref)
        vis = content(ref)[0]; snaps, _, _ = _slide_replay(vis, meta, ref)
        for i, st in snaps[::3]:
            for p, v in st.items():
                bank.setdefault(f"{n}_{v}", []).append(_digit_feat(cell(vis[i], *p)))
    SLIDE_BANK.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SLIDE_BANK, **{k: np.stack(v) for k, v in bank.items()})


_BANK = None


def sliding(frames, meta, ref):
    global _BANK
    if _BANK is None:
        if not SLIDE_BANK.exists():
            build_sliding_bank()
        z = np.load(SLIDE_BANK); _BANK = {k: z[k] for k in z.files}
    n, cell = _slide_geom(meta, ref)
    bank = {int(k.split("_")[1]): v for k, v in _BANK.items() if int(k.split("_")[0]) == n}
    target = {tuple(o["target_position"]): o["value"] for o in meta["objects"] if o.get("value")}
    vis, eos = content(frames)
    snaps, violation, state = _slide_replay(vis, meta, ref)
    runs, run_start, extra = {}, {}, {}
    dup_run, dup_start = 0, None
    for i, st in snaps:
        if i == 0 or float(np.abs(vis[i].astype(np.float32) - vis[i - 1].astype(np.float32)).mean()) > SLIDE_STILL:
            continue
        reads = {p: min(bank, key=lambda u: float(((bank[u] - _digit_feat(cell(vis[i], *p))) ** 2).sum(1).min())) for p in st}
        if len(set(reads.values())) < len(reads):
            dup_start = i if dup_run == 0 else dup_start; dup_run += 1
            if dup_run >= SLIDE_DUP_PERSIST and (violation is None or dup_start < violation[1]):
                violation = ("tile_hallucination", dup_start); extra['violation_start'] = dup_start
        else:
            dup_run = 0
    for i, st in snaps:
        if violation is not None and violation[0] == 'tile_hallucination' and i > violation[1]:
            break
        for p, v in st.items():
            f = _digit_feat(cell(vis[i], *p))
            read_as = min(bank, key=lambda u: float(((bank[u] - f) ** 2).sum(1).min()))
            k, cnt = runs.get((p, v), (None, 0))
            runs[(p, v)] = (read_as, cnt + 1 if read_as == k else 1) if read_as != v else (None, 0)
            if runs[(p, v)][1] == 1 and read_as != v:
                run_start[(p, v)] = i
            if runs[(p, v)][1] >= SLIDE_PERSIST:
                hall = ("tile_hallucination", i)
                extra['violation_start'] = run_start.get((p, v), i)
                if violation is None or hall[1] < violation[1]:
                    violation = hall
                break
        else:
            continue
        break
    changes, prev = [], None
    for i, st in snaps:
        key = sorted((list(p), v) for p, v in st.items())
        if key != prev:
            changes.append((i, key)); prev = key
    return result(state == target, violation, eos, changes=changes, n=n,
                  target_state=sorted((list(p), v) for p, v in target.items()), **extra)


def hanoi(frames, meta, ref):
    objs = meta["objects"]; k = S / 1024
    cols = np.array([o["color"] for o in objs], np.float32)
    nstk = int(meta["num_stacks"])
    xs_cur = [128 * (s + 1) * k for s in range(nstk)]; xs_tgt = [(640 + 128 * s) * k for s in range(nstk)]
    ybase, dy = 792 * k, 64 * k
    bg = patch_mean(ref[0], 300 * k, 512 * k, 6)
    rad = 10 * k

    def stacks(img, xs):
        st = []
        for x in xs:
            col = []
            for p in range(len(objs) + 1):
                m = patch_mean(img, ybase - p * dy, x - 26 * k, 3)
                if ((bg - m) ** 2).sum() < ((cols - m) ** 2).sum(1).min():
                    break
                col.append(m)
            st.append(col)
        return st

    def identify(st):
        ms = [m for c in st for m in c]
        if len(ms) != len(objs):
            return None, "block"
        D = np.array([((cols - m) ** 2).sum(1) for m in ms])
        r, c = linear_sum_assignment(D)
        dmin = D.min(1)
        if (dmin > 6000).any():
            return None, "colour"
        if any(D[i, j] > 2500 and D[i, j] > 4 * dmin[i] for i, j in zip(r, c)):
            return None, "block"
        ids = iter(c[np.argsort(r)].tolist())
        return [[next(ids) for _ in col] for col in st], None

    tgt = [[] for _ in range(nstk)]
    for j, o in sorted(enumerate(objs), key=lambda t: t[1]["target_position"]):
        tgt[o["target_stack"]].append(j)
    start = [[] for _ in range(nstk)]
    for j, o in sorted(enumerate(objs), key=lambda t: t[1]["initial_position"]):
        start[o["initial_stack"]].append(j)
    vis, eos = content(frames)
    states, violation, bad_run, changes = [], None, 0, []
    air_run = multi_run = 0
    for i, img in enumerate(vis):
        tg, _ = identify(stacks(img, xs_tgt))
        if tg != tgt:
            violation = violation or ("target_panel_changed", i)
        cur = stacks(img, xs_cur)
        miss = len(objs) - sum(len(c) for c in cur)
        if miss > 0:
            air_run = air_run + 1 if miss == 1 else 0; multi_run = multi_run + 1 if miss >= 2 else 0
            if multi_run == 2:
                violation = violation or ("multi_block_hallucination", i - 1)
            if air_run == HANOI_AIR_MAX + 1:
                violation = violation or ("vanished_block_hallucination", i - HANOI_AIR_MAX)
            continue
        air_run = multi_run = 0
        state, problem = identify(cur)
        if problem:
            bad_run += 1
            if bad_run >= 2:
                violation = violation or (f"{problem}_hallucination", i - 1)
            continue
        bad_run = 0
        if states and state != states[-1]:
            prev = states[-1]
            src = [s for s in range(nstk) if len(state[s]) == len(prev[s]) - 1]
            dst = [s for s in range(nstk) if len(state[s]) == len(prev[s]) + 1]
            ok = len(src) == 1 and len(dst) == 1 and state[src[0]] == prev[src[0]][:-1] and \
                state[dst[0]] == prev[dst[0]] + [prev[src[0]][-1]] and \
                all(state[s] == prev[s] for s in range(nstk) if s not in (src[0], dst[0]))
            if not ok:
                violation = violation or ("illegal_move", i)
        if not states or state != states[-1]:
            changes.append((i, state))
        states.append(state)
    if states and states[0] != start:
        violation = violation or ("wrong_start", 0)
    goal = bool(states) and states[-1] == tgt
    moves = sum(1 for a, b in zip(states, states[1:]) if a != b)
    return result(goal, violation, eos, moves=moves, optimal_moves=int(meta["optimal_moves"]), changes=changes, target=tgt)


def snake(frames, meta, ref):
    W, H = int(meta["grid_w"]), int(meta["grid_h"]); init = meta["initial_state"]
    hsv = cv2.cvtColor(ref[0], cv2.COLOR_RGB2HSV)
    navy = (hsv[..., 0] > 100) & (hsv[..., 0] < 130) & (hsv[..., 1] > 100) & (hsv[..., 2] > 40)
    ys, xs = np.nonzero(navy)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    ch, cw = (y1 - y0) / H, (x1 - x0) / W
    centre = lambda r, c: (y0 + (r + 0.5) * ch, x0 + (c + 0.5) * cw)
    body0 = [tuple(p) for p in init["snake"]]; foods0 = {tuple(p) for p in init["foods"]}
    green = lambda img, r, c: (lambda m: m[1] > 150 and m[0] < 150 and m[2] < 150)(patch_mean(img, *centre(r, c), 0.25 * min(ch, cw)))
    swap = not all(green(ref[0], *p) for p in body0 if 0 <= p[0] < H and 0 <= p[1] < W) if all(0 <= p[0] < H and 0 <= p[1] < W for p in body0) else True
    conv = (lambda p: (p[1], p[0])) if swap else (lambda p: p)
    body0 = [conv(p) for p in body0]; foods0 = {conv(p) for p in foods0}
    def cells_of(img):
        return {(r, c) for r in range(H) for c in range(W)
                if (lambda m: m[1] > 150 and m[0] < 150)(patch_mean(img, *centre(r, c), 0.22 * min(ch, cw)))}
    for f in ref[1:]:
        new = cells_of(f) - set(body0)
        if len(new) == 1:
            nh = next(iter(new))
            if abs(nh[0] - body0[-1][0]) + abs(nh[1] - body0[-1][1]) == 1:
                body0 = body0[::-1]
            break

    def decode(img):
        snake_cells, food = set(), set()
        for r in range(H):
            for c in range(W):
                m = patch_mean(img, *centre(r, c), 0.22 * min(ch, cw))
                if m[1] > 150 and m[0] < 150:
                    snake_cells.add((r, c))
                elif m[0] > 150 and m[1] < 120:
                    food.add((r, c))
        return snake_cells, food
    vis, eos = content(frames)
    body, foods, violation, pending = list(body0), set(foods0), None, None
    moves = 0

    def step(cells, food):
        if cells == set(body) and food == foods:
            return None, body, foods
        head = body[0]; new = [p for p in cells if p not in body]
        if len(new) != 1:
            return "snake_hallucination", body, foods
        nh = new[0]
        if abs(nh[0] - head[0]) + abs(nh[1] - head[1]) != 1:
            return "illegal_move", body, foods
        ate = nh in foods
        expect = [nh] + (body if ate else body[:-1])
        if nh in (body if ate else body[:-1]):
            return "self_collision", body, foods
        if cells != set(expect):
            return "length_hallucination", body, foods
        if food != (foods - {nh}):
            return "food_hallucination", body, foods
        return None, expect, foods - {nh}
    for i, img in enumerate(vis):
        kind, nb, nf = step(*decode(img))
        if kind is None:
            moves += nb is not body
            body, foods, pending = nb, nf, None
        elif kind == "self_collision":
            violation = (kind, i); break
        elif pending is not None:
            violation = pending; break
        else:
            pending = (kind, i)
    goal = not foods
    return result(goal, violation, eos, food_left=len(foods), length=len(body), moves=moves)


AGENT = np.array([255, 165, 0], np.float32)


def agent_mask(img):
    r, g, b = (img[..., i].astype(np.int16) for i in range(3))
    return (np.abs(r - 255) < 40) & (np.abs(g - 165) < 40) & (b < 70)


def static_objects(ref0):
    hsv = cv2.cvtColor(ref0, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 1] > 60) & ~agent_mask(ref0)).astype(np.uint8)
    return cv2.dilate(m, np.ones((7, 7), np.uint8)) > 0


def agent_track(vis, merge=S / 10 * 0.6, static=None):
    out = []
    for img in vis:
        m = agent_mask(img).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
        parts = []
        for k in range(1, n):
            area = stats[k, cv2.CC_STAT_AREA]
            if area <= 3:
                continue
            if static is not None and area < 40 and static[lab == k].mean() > 0.9:
                continue
            parts.append((area, cent[k][1], cent[k][0]))
        if not parts:
            out.append(None); continue
        big = max(parts)
        near = [p for p in parts if np.hypot(p[1] - big[1], p[2] - big[2]) < merge]
        far = [p for p in parts if p not in near and p[0] > 25]
        area = sum(p[0] for p in near)
        if far:
            out.append(("multi",))
        elif area <= 25:
            out.append(None)
        else:
            out.append((sum(p[0] * p[1] for p in near) / area, sum(p[0] * p[2] for p in near) / area, 1))
    return out


def _cell_of(y, x, n=10):
    return int(y // (S / n)), int(x // (S / n))


def _conv(ref0, start):
    tr = agent_track([ref0])[0]
    if tr is None or tr[0] == "multi":
        return lambda p: tuple(p)
    cell = _cell_of(tr[0], tr[1])
    return (lambda p: tuple(p)) if tuple(start) == cell else (lambda p: (p[1], p[0]))


def halts(track, still=2.0):
    pts = [None if t is None or t[0] == "multi" else np.array(t[:2], float) for t in track]
    return {i for i in range(1, len(pts) - 1) if pts[i] is not None and pts[i - 1] is not None and pts[i + 1] is not None
            and np.linalg.norm(pts[i] - pts[i - 1]) < still and np.linalg.norm(pts[i + 1] - pts[i]) < still}


def waypoints(track, k=2, angle=30.0, still=2.0):
    pts = [None if t is None or t[0] == "multi" else np.array(t[:2], float) for t in track]
    out = []
    for i in range(len(pts)):
        if pts[i] is None:
            continue
        a, b = pts[max(0, i - k)], pts[min(len(pts) - 1, i + k)]
        if a is None or b is None:
            continue
        u, v = pts[i] - a, b - pts[i]
        if np.linalg.norm(u) < still or np.linalg.norm(v) < still:
            out.append(i); continue
        cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
        if np.degrees(np.arccos(np.clip(cos, -1, 1))) > angle:
            out.append(i)
    return out


def _agent_checks(vis, ch, ref, persist=3):
    track, run, last = agent_track(vis, static=static_objects(ref[0])), 0, None
    for i, t in enumerate(track):
        if t is None or t[0] == "multi":
            run += 1
            if run >= persist:
                return track, ("agent_hallucination", i - persist + 1)
            continue
        run = 0
        if last is not None and np.hypot(t[0] - last[0], t[1] - last[1]) > 1.2 * ch * (i - last[2]):
            return track, ("teleport", i)
        last = (t[0], t[1], i)
    return track, None


def tsp(frames, meta, ref):
    conv = lambda p: tuple(p)
    end = conv(meta["end"])
    items = {r["id"]: conv(r["position"]) for r in meta["rewards"]}
    ch = S / 10
    look = lambda img, c: patch_mean(img, (c[0] + .5) * ch, (c[1] + .5) * ch, 0.15 * ch)
    present = lambda img, c: ((look(img, c) - look(ref[-1], c)) ** 2).sum() > 900
    vis, eos = content(frames)
    track, violation = _agent_checks(vis, ch, ref)
    collected = set()
    if violation is None:
        stops, halted = set(waypoints(track)), halts(track)
        for i, (img, t) in enumerate(zip(vis, track)):
            if t is None or t[0] == "multi":
                continue
            cell = _cell_of(t[0], t[1])
            for k, c in items.items():
                if cell == c:
                    collected.add(k)
                elif k not in collected and not present(img, c):
                    violation = violation or ("item_vanished_unvisited", i)
            if cell == end and i in halted and i != len(vis) - 1 and len(collected) < len(items):
                violation = violation or ("end_before_all_items", i)
    last = next((t for t in reversed(track) if t and t[0] != "multi"), None)
    goal = last is not None and _cell_of(last[0], last[1]) == end and len(collected) == len(items)
    pts = [t for t in track if t and t[0] != "multi"]
    path = sum(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(pts, pts[1:])) / ch
    return result(goal, violation, eos, collected=len(collected), path_cells=round(float(path), 1))


def ordinal(frames, meta, ref):
    conv = lambda p: (p[1], p[0])
    end = conv(meta["end"])
    nums = {d["number"]: conv(d["position"]) for d in meta["number_positions"]}
    ch = S / 10
    vis, eos = content(frames)
    track, violation = _agent_checks(vis, ch, ref)
    order = []
    if violation is None:
        stops, halted = set(waypoints(track)), halts(track)
        for i, t in enumerate(track):
            if t is None or t[0] == "multi":
                continue
            cy, cx = t[0] / ch, t[1] / ch
            for k, c in nums.items():
                inside = abs(cy - (c[0] + .5)) < .3 and abs(cx - (c[1] + .5)) < .3
                if inside and k not in order and (k == len(order) + 1 or i in stops):
                    if k != len(order) + 1:
                        violation = violation or ("wrong_order", i)
                    order.append(k)
            at_end = abs(cy - (end[0] + .5)) < .3 and abs(cx - (end[1] + .5)) < .3
            if at_end and i in halted and i != len(track) - 1 and len(order) < len(nums):
                violation = violation or ("end_before_all_numbers", i)
    last = next((t for t in reversed(track) if t and t[0] != "multi"), None)
    goal = last is not None and _cell_of(last[0], last[1]) == end and order == sorted(nums)
    pts = [t for t in track if t and t[0] != "multi"]
    path = sum(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(pts, pts[1:])) / ch
    return result(goal, violation, eos, visited=order, path_cells=round(float(path), 1))


CHECK = {"sokoban": sokoban, "sliding": sliding, "hanoi": hanoi, "snake": snake, "tsp": tsp, "ordinal": ordinal}


def check(row: dict, frames: np.ndarray) -> dict:
    task = row["task"].replace("vbvr_", "")
    meta = puzzle(row)
    ref = read(data_path(row["native_video"]))
    return CHECK[task](frames, meta, ref)
