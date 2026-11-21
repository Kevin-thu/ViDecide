from __future__ import annotations

from typing import Any
from functools import lru_cache

import cv2
import numpy as np

from videcide.tasks.flow_replay import _filled_boards_and_scores, evaluate_flowfree, necessary_connectivity_deadlock
from videcide.tasks.flowfree_task import grid_neighbors


MARGIN = 28


CENTRED = 0.20


TENDENCY = 0.03


CURSOR_MIN = 0.12


CIRCLE_VOTES = 14


MAX_BACKOFF = 4


SOLVER_MAX_STATES = 2_000_000


POSTCHECK_MAX_STATES = 2_000_000


def cursor_offset(frame: np.ndarray, size: int, tip: tuple[int, int], source: dict[str, Any],
                     _return_xy: bool = False, reference_frame: np.ndarray | None = None) -> float | None:
    width = (frame.shape[0] - 2 * MARGIN) / size
    img = frame.astype(np.float32)
    dev = np.zeros(img.shape[:2], np.float32)
    valid = np.zeros(img.shape[:2], bool)
    endpoints = {tuple(map(int, c)) for group in source["endpoints"] for c in group}
    row, col = tip
    for r in range(max(0, row - 1), min(size, row + 2)):
        for c in range(max(0, col - 1), min(size, col + 2)):
            y0, y1 = int(MARGIN + (r + 0.1) * width), int(MARGIN + (r + 0.9) * width)
            x0, x1 = int(MARGIN + (c + 0.1) * width), int(MARGIN + (c + 0.9) * width)
            cell = img[y0:y1, x0:x1]
            base = np.median(cell.reshape(-1, 3), axis=0)
            d = np.linalg.norm(cell - base, axis=2)
            static = reference_frame is not None and not _endpoint_changed(frame, reference_frame, size, (r,c))
            if (r, c) in endpoints and ((r, c) != (row, col) or static):
                yy, xx = np.mgrid[y0:y1, x0:x1]
                cy, cx = MARGIN + (r + 0.5) * width, MARGIN + (c + 0.5) * width
                d[(yy - cy) ** 2 + (xx - cx) ** 2 < (0.42 * width) ** 2] = 0
            dev[y0:y1, x0:x1] = d
            valid[y0:y1, x0:x1] = True
    radius = max(3, int(round(width * 0.2)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2).astype(np.float32)
    kernel /= kernel.sum()
    score = cv2.filter2D(np.clip(dev / 120.0, 0, 1), -1, kernel)
    score[~valid] = 0
    if float(score.max()) < CURSOR_MIN:
        return None
    y, x = np.unravel_index(int(score.argmax()), score.shape)
    if _return_xy:
        return float(x), float(y)
    fc = (x - MARGIN) / width - 0.5
    fr = (y - MARGIN) / width - 0.5
    return float(np.hypot(fr - round(fr), fc - round(fc)))


def _endpoint_changed(frame: np.ndarray, reference: np.ndarray, size: int, cell: tuple[int,int]) -> bool:
    width = (frame.shape[0] - 2 * MARGIN) / size
    cx, cy = MARGIN + (cell[1] + .5) * width, MARGIN + (cell[0] + .5) * width
    radius = .18 * width
    x0, x1 = max(0,int(cx-radius)), min(frame.shape[1],int(cx+radius)+1)
    y0, y1 = max(0,int(cy-radius)), min(frame.shape[0],int(cy+radius)+1)
    yy, xx = np.mgrid[y0:y1,x0:x1]
    disk = (xx-cx)**2 + (yy-cy)**2 <= radius**2
    change = np.linalg.norm(frame[y0:y1,x0:x1].astype(float)-reference[y0:y1,x0:x1],axis=-1)[disk]
    return bool(np.mean(change >= 60) >= .12)


def cursor_circle(frame: np.ndarray, size: int, tip: tuple[int, int], source: dict[str, Any] | None = None,
                  reference_frame: np.ndarray | None = None):
    width = (frame.shape[0] - 2 * MARGIN) / size
    row, col = tip
    cy, cx = MARGIN + (row + 0.5) * width, MARGIN + (col + 0.5) * width
    half = width * 1.25
    y0, y1 = int(max(0, cy - half)), int(min(frame.shape[0], cy + half))
    x0, x1 = int(max(0, cx - half)), int(min(frame.shape[1], cx + half))
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    gray = cv2.cvtColor(np.ascontiguousarray(patch), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=width * 0.5, param1=80,
                               param2=CIRCLE_VOTES, minRadius=max(3, int(width * 0.16)),
                               maxRadius=int(width * 0.38))
    if circles is None:
        return None
    best = None
    for x, y, r in circles[0]:
        gx, gy = x0 + x, y0 + y
        fc, fr = (gx - MARGIN) / width - 0.5, (gy - MARGIN) / width - 0.5
        if abs(fr - row) > 1.05 or abs(fc - col) > 1.05:
            continue
        if source is not None:
            near = (int(round(fr)), int(round(fc)))
            ends = {tuple(map(int, c)) for g in source["endpoints"] for c in g}
            if near in ends and np.hypot(fr - near[0], fc - near[1]) < 0.15:
                if reference_frame is None:
                    if near != (row, col):
                        continue
                else:
                    if not _endpoint_changed(frame, reference_frame, size, near):
                        continue
        d = float(np.hypot(fr - row, fc - col))
        near = (int(round(fr)), int(round(fc)))
        on_endpoint = source is not None and near in {tuple(map(int, c)) for g in source["endpoints"] for c in g} \
            and np.hypot(fr - near[0], fc - near[1]) < 0.15
        key = (on_endpoint, d)
        if best is None or key < best[3]:
            best = (float(gx), float(gy), float(np.hypot(fr - round(fr), fc - round(fc))), key)
    return None if best is None else best[:3]


def cursor_detection(frame: np.ndarray, size: int, tip: tuple[int, int], source: dict[str, Any],
                     reference_frame: np.ndarray) -> dict[str, Any]:
    circle = cursor_circle(frame, size, tip, source, reference_frame)
    if circle is not None:
        return {"xy": list(circle[:2]), "offset": circle[2], "method": "ring"}
    xy = cursor_offset(frame, size, tip, source, _return_xy=True, reference_frame=reference_frame)
    if xy is not None:
        width = (frame.shape[0] - 2 * MARGIN) / size
        fc, fr = (xy[0] - MARGIN) / width - .5, (xy[1] - MARGIN) / width - .5
        near = (int(round(fr)), int(round(fc)))
        endpoints = {tuple(map(int, c)) for group in source["endpoints"] for c in group}
        if near in endpoints and np.hypot(fr - near[0], fc - near[1]) < .20:
            if not _endpoint_changed(frame, reference_frame, size, near):
                return {"xy": None, "offset": None, "method": "static_endpoint"}
        return {"xy": list(xy), "offset": float(np.hypot(fr - round(fr), fc - round(fc))),
                "method": "uncertain_blob"}
    return {"xy": None, "offset": None, "method": "no_cursor_evidence"}


def move_start_frame(frames: np.ndarray, source: dict[str, Any], trace: list[dict], frame: int) -> int:
    size = int(source["grid_size"])
    before = [e for e in trace if int(e["frame"]) < frame and e.get("active_tips")]
    if not before:
        return frame
    tip = before[-1]["active_tips"][-1]
    fatal = next((e for e in trace if int(e["frame"]) == frame), {})
    additions = fatal.get("added", [])
    target = None
    if additions:
        target = additions[-1]
        color = int(target[2])
        previous = [(int(c), int(r), int(col)) for e in trace if int(e["frame"]) < frame
                    for r, col, c in e.get("added", []) if int(c) == color]
        if previous:
            tip = previous[-1]
        else:
            return frame
    f = frame - 1
    while f > 0 and frame - f <= 4:
        detection = cursor_detection(frames[f], size, (int(tip[1]), int(tip[2])), source, frames[0])
        off = detection["offset"]
        entering = 0.0
        if target is not None and abs(int(target[0]) - int(tip[1])) + abs(int(target[1]) - int(tip[2])) == 1:
            from videcide.tasks.flow_replay import PALETTE
            entering = max(0.0, _edge_strip_fraction(frames[f], size, tip[1:], target[:2], PALETTE[int(target[2])])
                           - _edge_strip_fraction(frames[0], size, tip[1:], target[:2], PALETTE[int(target[2])]))
        if (off is None or off < CENTRED) and entering < TENDENCY:
            break
        f -= 1
    return f + 1


def _edge_strip_fraction(frame: np.ndarray, size: int, cell, towards, colour_rgb) -> float:
    width = (frame.shape[0] - 2 * MARGIN) / size
    (r0, c0), (r1, c1) = cell, towards
    top, left = MARGIN + r1 * width, MARGIN + c1 * width
    band, inset = 0.35 * width, 0.2 * width
    if r1 == r0 + 1:   ys, xs = (top, top + band), (left + inset, left + width - inset)
    elif r1 == r0 - 1: ys, xs = (top + width - band, top + width), (left + inset, left + width - inset)
    elif c1 == c0 + 1: ys, xs = (top + inset, top + width - inset), (left, left + band)
    else:              ys, xs = (top + inset, top + width - inset), (left + width - band, left + width)
    patch = frame[int(ys[0]):int(ys[1]), int(xs[0]):int(xs[1])].astype(np.float32)
    if patch.size == 0:
        return 0.0
    target = np.asarray(colour_rgb, np.float32)
    chroma = patch - patch.min(-1, keepdims=True); chroma /= chroma.sum(-1, keepdims=True).clip(min=1)
    tchroma = target - target.min(); tchroma /= max(float(tchroma.sum()), 1.0)
    bright = patch.max(-1); sat = (bright - patch.min(-1)) / bright.clip(min=1)
    match = (((chroma - tchroma) ** 2).sum(-1) < 0.02) & (sat >= 0.35) & (bright >= 100)
    return float(match.mean())


def direction_tendency(frames: np.ndarray, source: dict[str, Any], prefix: int, tip) -> float:
    from videcide.tasks.flow_replay import PALETTE
    size = int(source["grid_size"])
    lo, hi = max(0, 4 * (prefix - 2) + 1), 4 * (prefix - 1)
    window = frames[lo:hi + 1]
    if len(window) == 0:
        return 0.0
    boards, _ = _filled_boards_and_scores(window, size, len(source["endpoints"]))
    colour, row, col = int(tip[0]), int(tip[1]), int(tip[2])
    endpoints = {tuple(map(int, cell)) for group in source["endpoints"] for cell in group}
    worst = 0.0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r, c = row + dr, col + dc
        if 0 <= r < size and 0 <= c < size and boards[-1, r, c] == -1 and (r, c) not in endpoints:
            for frame in window:
                worst = max(worst, _edge_strip_fraction(frame, size, (row, col), (r, c), PALETTE[colour]))
    return worst


def _solve_board(board: np.ndarray, source: dict[str, Any], tips: list,
                 max_states: int) -> dict[str, Any]:
    used = sorted(int(c) for c in np.unique(board) if int(c) >= 0)
    active = sorted({int(t[0]) for t in tips})
    if not used:
        return find_completion(board, source, active_tips=[], color_runs=[], max_states=max_states)
    if not active:
        return {"status": "unknown", "states": 0, "reason": "missing_active_tip", "witness": None}
    states, unknown = 0, None
    results = []
    for color in active:
        runs = [c for c in used if c != color] + [color]
        result = find_completion(board, source, active_tips=[t for t in tips if int(t[0]) == color],
                                 color_runs=runs, max_states=max(0,max_states-states))
        states += result["states"]
        if result["status"] == "solvable":
            return {**result, "states": states}
        if result["status"] == "unknown":
            unknown = result
        results.append(result)
    if unknown is not None:
        return {**unknown, "states": states}
    reason = "exhaustive_search" if all(r["reason"] == "exhaustive_search" for r in results) else results[0]["reason"]
    return {"status": "unsolvable", "states": states, "reason": reason, "witness": None}


def _weak_deadlock(board: np.ndarray, source: dict[str, Any], tips: list) -> bool:
    used = sorted(int(c) for c in np.unique(board) if int(c) >= 0)
    active = sorted({int(t[0]) for t in tips})
    if not active:
        return not used and necessary_connectivity_deadlock(board,source,active_tips=[],color_runs=[]) is not None
    return all(necessary_connectivity_deadlock(board,source,
               active_tips=[t for t in tips if int(t[0])==color],
               color_runs=[c for c in used if c != color]+[color]) is not None for color in active)


def boundary_of(trace: list[dict], source: dict[str, Any],
                max_states: int = SOLVER_MAX_STATES, use_solver: bool = True) -> tuple[int | None, str | None]:
    if not use_solver:
        for event in trace:
            if event.get("local_violation"):
                return int(event["frame"]), "immediate_violation"
        return (int(trace[-1]["frame"]), "end_without_visible_violation") if trace else (None, None)
    size = int(source["grid_size"])
    board = np.full((size, size), -1, dtype=np.int8)
    snapshots = []
    weak = None
    for event in trace:
        for row, col, colour in event.get("added", []):
            board[int(row), int(col)] = int(colour)
        frame = int(event["frame"])
        if event.get("local_violation"):
            visible = frame
            break
        if weak is None and _weak_deadlock(board,source,event.get("active_tips", [])):
            weak = frame
        snapshots.append((frame, board.copy(), list(event.get("active_tips", []))))
    else:
        visible = None

    def dead(index: int) -> bool:
        _, board_, tips = snapshots[index]
        result = _solve_board(board_, source, tips, max_states)
        return result["status"] == "unsolvable" and result.get("reason") == "exhaustive_search"

    solver = None
    if snapshots and dead(len(snapshots) - 1):
        low, high = 0, len(snapshots) - 1
        while low < high:
            mid = (low + high) // 2
            if dead(mid):
                high = mid
            else:
                low = mid + 1
        solver = snapshots[low][0]
    found = [(f, k) for f, k in ((solver, "solver_dead"), (weak, "deadlock_entry"), (visible, "immediate_violation"))
             if f is not None]
    return min(found) if found else (None, None)


def fork_state(trace: list[dict], source: dict[str, Any], prefix: int,
               max_states: int = POSTCHECK_MAX_STATES) -> str:
    size = int(source["grid_size"])
    board = np.full((size, size), -1, dtype=np.int8)
    tips: list = []
    last = 4 * (int(prefix) - 1)
    for event in trace:
        if int(event["frame"]) > last:
            break
        for row, col, colour in event.get("added", []):
            board[int(row), int(col)] = int(colour)
        tips = list(event.get("active_tips", []))
    result = _solve_board(board, source, tips, max_states)
    if result["status"] == "unsolvable":
        return "unsolvable" if result.get("reason") == "exhaustive_search" else f"unknown:{result.get('reason')}"
    return "solvable" if result["status"] != "unknown" else f"unknown:{result.get('reason')}"


def seam_diagnostics(frames: np.ndarray, source: dict[str, Any], prefix: int,
                     trace: list[dict] | None = None) -> dict[str, Any]:
    last = 4 * (int(prefix) - 1)
    if trace is None:
        trace = evaluate_flowfree(frames[:last + 1], source, include_trace=True,
                                  require_adjacent_color_switch=bool(source.get('continuous_global_walk', False)))["trace"]
    lo = max(0, 4 * (int(prefix) - 2) + 1)
    tips: dict[int, tuple[int, int, int]] = {}
    entering = []
    for e in trace:
        f = int(e["frame"])
        if f > last:
            break
        if f < lo:
            entering = e.get("active_tips", [])
            continue
        if not tips:
            tips.update((int(t[0]), tuple(map(int, t))) for t in entering)
        for r, c, color in e.get("added", []):
            tips[int(color)] = (int(color), int(r), int(c))
        for t in e.get("active_tips", []):
            tips[int(t[0])] = tuple(map(int, t))
    observations = []
    painted = set()
    painted_by_color: dict[int,set] = {}
    for e in trace:
        if int(e["frame"]) > last:
            break
        painted.update((int(r), int(c)) for r, c, _ in e.get("added", []))
        for r,c,color in e.get("added", []):
            painted_by_color.setdefault(int(color),set()).add((int(r),int(c)))
    for tip in tips.values():
        detection = cursor_detection(frames[last], int(source["grid_size"]), tip[1:], source, frames[0])
        tendency = direction_tendency(frames, source, prefix, tip)
        remaining = [list(map(int, c)) for c in source["endpoints"][tip[0]] if tuple(map(int, c)) not in painted]
        observations.append({"tip": list(tip), **detection, "tendency": tendency,
                             "remaining_endpoints": remaining,
                             "painted_color_cells": len(painted_by_color.get(tip[0],set()))})
    offsets = [o["offset"] for o in observations if o["offset"] is not None]
    tendency = max((o["tendency"] for o in observations), default=0.0)
    offset = max(offsets, default=None)
    uncertain = any(o["method"] == "uncertain_blob" for o in observations)
    unfinished = any(o["offset"] is None and o["remaining_endpoints"] and o["painted_color_cells"] > 1
                     for o in observations)
    clean = tendency < TENDENCY and (offset is None or offset < CENTRED) and not uncertain and not unfinished
    reason = "stroke_tendency" if tendency >= TENDENCY else "off_centre" if offset is not None and offset >= CENTRED \
        else "cursor_uncertain" if uncertain else "cursor_disappeared_before_endpoints" if unfinished \
        else "centred" if offset is not None else "no_visible_cursor"
    return {"clean": clean, "reason": reason, "offset": offset, "tendency": tendency,
            "last_shared_frame": last, "observations": observations}


def select_fork(frames: np.ndarray, source: dict[str, Any], *, floor: int = 1,
                max_backoff: int = MAX_BACKOFF, trace: list[dict] | None = None,
                min_backoff: int = 0, use_solver: bool = True) -> dict[str, Any]:
    if trace is None:
        trace = evaluate_flowfree(frames, source, include_trace=True,
                                  require_adjacent_color_switch=bool(source.get('continuous_global_walk', False)))["trace"]
    record = _select_fork(frames, source, floor=floor, max_backoff=max_backoff, trace=trace, min_backoff=min_backoff,
                          use_solver=use_solver)
    if record.get("status") == "ok" and not use_solver:
        record["fork_state"] = "not_certified_solver_free"
        return record
    if record.get("status") == "ok":
        record["fork_state"] = fork_state(trace, source, record["prefix"])
        if record["fork_state"] == "unsolvable":
            record["status"] = "fork_unsolvable"
        elif record["fork_state"] != "solvable":
            record["status"] = "fork_state_unknown"
    return record


def _select_fork(frames: np.ndarray, source: dict[str, Any], *, floor: int,
                 max_backoff: int, trace: list[dict], min_backoff: int = 0, use_solver: bool = True) -> dict[str, Any]:
    frame, kind = boundary_of(trace, source, use_solver=use_solver)
    if frame is None:
        return {"status": "no_boundary"}
    move = move_start_frame(frames, source, trace, frame) if len(frames) else frame
    start = (max(1, move) - 1) // 4 + 1
    seen = []
    if start <= floor:
        return {"status": "dead_inside_history", "boundary_kind": kind, "boundary_frame": frame}
    for step in range(min_backoff, min_backoff + max_backoff + 1):
        candidate = start - step
        if candidate <= floor:
            break
        last = 4 * (candidate - 1)
        if not (0 <= last < len(frames)):
            continue
        evidence = seam_diagnostics(frames, source, candidate, trace)
        common = {"boundary_kind": kind, "boundary_frame": frame, "backoff": step,
                  "move_start_frame": move}
        if evidence["clean"]:
            return {"status": "ok", "prefix": candidate, "offset": evidence["offset"],
                    "reason": evidence["reason"], "tendency": evidence["tendency"],
                    "seam": evidence, **common}
        seen.append((candidate, evidence["offset"]))
    if seen:
        return {"status": "no_clean_seam", "boundary_kind": kind, "boundary_frame": frame}
    return {"status": "no_measurable_latent", "boundary_kind": kind, "boundary_frame": frame}


Cell = tuple[int, int]


def find_completion(
    board: np.ndarray,
    source: dict[str, Any],
    *,
    active_tips: list[list[int]] | tuple[tuple[int, int, int], ...],
    color_runs: list[int] | tuple[int, ...],
    max_states: int = 1_000_000,
) -> dict[str, Any]:
    size = int(source["grid_size"])
    if board.shape != (size, size):
        raise ValueError(f"expected {(size, size)} board, got {board.shape}")
    colors = len(source["endpoints"])
    continuous = bool(source.get('continuous_global_walk', False))
    endpoints = tuple(
        frozenset((int(row), int(col)) for row, col in group)
        for group in source["endpoints"]
    )
    endpoint_color = {
        cell: color for color, group in enumerate(endpoints) for cell in group
    }
    for cell, color in endpoint_color.items():
        if int(board[cell]) not in (-1, color):
            return {"status": "unsolvable", "states": 0, "witness": None,
                    "reason": "endpoint_wrong_color"}

    index = {(row, col): row * size + col for row in range(size) for col in range(size)}
    cells = tuple(index)
    full_mask = (1 << (size * size)) - 1
    filled = 0
    for cell in cells:
        if int(board[cell]) >= 0:
            filled |= 1 << index[cell]

    runs = tuple(int(color) for color in color_runs)
    if len(runs) != len(set(runs)) or any(not 0 <= color < colors for color in runs):
        return {"status": "unsolvable", "states": 0, "witness": None,
                "reason": "invalid_color_runs"}
    used = sum(1 << color for color in runs)
    active = runs[-1] if runs else -1
    tips = {
        (int(color), (int(row), int(col))) for color, row, col in active_tips
        if int(color) == active
    }
    if active >= 0 and not tips:
        return {"status": "unsolvable", "states": 0, "witness": None,
                "reason": "missing_active_tip"}
    initial_tips = sorted(cell for _, cell in tips) if active >= 0 else [(-1, -1)]

    endpoint_masks = []
    for group in endpoints:
        mask = 0
        for cell in group:
            mask |= 1 << index[cell]
        endpoint_masks.append(mask)

    states = 0
    hit_limit = False

    def reachable(start: Cell, allowed_mask: int) -> int:
        reached = 1 << index[start]
        stack = [start]
        while stack:
            cell = stack.pop()
            for neighbor in grid_neighbors(cell, size):
                bit = 1 << index[neighbor]
                if allowed_mask & bit and not reached & bit:
                    reached |= bit
                    stack.append(neighbor)
        return reached

    def endpoints_reachable(mask: int, current: int, tip: Cell, used_mask: int) -> bool:
        empty = full_mask ^ mask
        for color, targets_mask in enumerate(endpoint_masks):
            remaining = targets_mask & empty
            if not remaining:
                continue
            if used_mask & (1 << color):
                if color != current:
                    return False
                start = tip
            else:
                first = (remaining & -remaining).bit_length() - 1
                start = cells[first]
            other_endpoints = 0
            for other, other_mask in enumerate(endpoint_masks):
                if other != color:
                    other_endpoints |= other_mask
            allowed = (empty & ~other_endpoints) | (1 << index[start]) | remaining
            if remaining & ~reachable(start, allowed):
                return False
        return True

    @lru_cache(maxsize=None)
    def search(mask: int, current: int, tip_index: int, used_mask: int):
        nonlocal states, hit_limit
        states += 1
        if states > max_states:
            hit_limit = True
            return None
        if mask == full_mask:
            return ()
        tip = cells[tip_index] if current >= 0 else (-1, -1)
        if not endpoints_reachable(mask, current, tip, used_mask):
            return None

        actions: list[tuple[int, Cell, bool]] = []
        if current >= 0:
            for neighbor in grid_neighbors(tip, size):
                bit = 1 << index[neighbor]
                owner = endpoint_color.get(neighbor)
                if not mask & bit and (owner is None or owner == current):
                    onward = sum(
                        not mask & (1 << index[nxt])
                        for nxt in grid_neighbors(neighbor, size)
                    )
                    actions.append((onward, neighbor, False))
        can_switch = current < 0 or not (endpoint_masks[current] & ~mask)
        if can_switch:
            for color, group in enumerate(endpoints):
                if used_mask & (1 << color):
                    continue
                for endpoint in group:
                    if continuous and current >= 0 and endpoint not in grid_neighbors(tip, size):
                        continue
                    if not mask & (1 << index[endpoint]):
                        actions.append((len(grid_neighbors(endpoint, size)), endpoint, True))
        actions.sort(key=lambda item: item[0])

        for _, cell, switching in actions:
            next_color = endpoint_color[cell] if switching else current
            bit = 1 << index[cell]
            suffix = search(
                mask | bit,
                next_color,
                index[cell],
                used_mask | (1 << next_color),
            )
            if suffix is not None:
                return ((next_color, cell[0], cell[1]),) + suffix
        return None

    for tip in initial_tips:
        witness = search(filled, active, index[tip] if active >= 0 else -1, used)
        if witness is not None:
            return {"status": "solvable", "states": states,
                    "witness": [list(move) for move in witness], "reason": None}
    if hit_limit:
        return {"status": "unknown", "states": states, "witness": None,
                "reason": "state_limit"}
    return {"status": "unsolvable", "states": states, "witness": None,
            "reason": "exhaustive_search"}
