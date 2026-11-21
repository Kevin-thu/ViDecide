"""Rule verifiers of the DiT-B videos (256 x 256, one frame per move): Sokoban reads the agent and the boxes per frame,
FlowFree upsamples the frames and runs the strict FlowFree verifier."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from videcide.tasks import sokoban_task
from videcide.tasks.video_parser import find_eos_frame, parse_video

C = sokoban_task.CLASSIC
PROTO = {"floor": C["floor"], "goal": C["goal"], "box": C["box"], "box_on_goal": C["box_on_goal"], "agent": C["agent"]}
NAMES = list(PROTO)
COLORS = np.array([PROTO[n] for n in NAMES], dtype=np.float32)
MOVES = {(-1, 0), (1, 0), (0, -1), (0, 1)}


def _decode(frame: np.ndarray, src: dict) -> tuple[tuple, frozenset, bool]:
    meta = {**src, "render_theme": "classic"}
    tile, ox, oy = sokoban_task._layout(meta)
    s = frame.shape[0] / sokoban_task.CANVAS_SIZE
    rad = max(1, int(round(0.12 * tile * s)))
    agents, boxes, on_wall = [], set(), False
    for r, row in enumerate(src["grid"]):
        for c, ch in enumerate(row):
            y = int(round((oy + (r + 0.5) * tile) * s)); x = int(round((ox + (c + 0.5) * tile) * s))
            patch = frame[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1].reshape(-1, 3).astype(np.float32)
            label = NAMES[int(np.argmin(((COLORS - patch.mean(0)) ** 2).sum(1)))]
            if ch == "#":
                if label in ("box", "box_on_goal", "agent"):
                    wall_rgb = np.array(C["wall"], np.float32)
                    if ((patch.mean(0) - wall_rgb) ** 2).sum() > ((COLORS[NAMES.index(label)] - patch.mean(0)) ** 2).sum():
                        on_wall = True
                continue
            if label == "agent":
                agents.append((r, c))
            elif label in ("box", "box_on_goal"):
                boxes.add((r, c))
    return tuple(agents), frozenset(boxes), on_wall


def _legal(prev, cur, src) -> bool:
    (a0,), b0 = prev
    (a1,), b1 = cur
    d = (a1[0] - a0[0], a1[1] - a0[1])
    if d not in MOVES or src["grid"][a1[0]][a1[1]] == "#":
        return False
    if a1 in b0:
        dest = (a1[0] + d[0], a1[1] + d[1])
        if src["grid"][dest[0]][dest[1]] == "#" or dest in b0:
            return False
        return b1 == (b0 - {a1}) | {dest}
    return b1 == b0


def check_sokoban(frames: np.ndarray, src: dict) -> dict:
    eos = find_eos_frame(frames)
    end = eos if eos is not None else len(frames)
    goals = {tuple(t) for t in src["targets"]}
    n_boxes = len(src["boxes_start"])
    states, violation, moves = [], None, 0
    for i in range(end):
        agents, boxes, on_wall = _decode(frames[i], src)
        if on_wall:
            violation = violation or ("drawn_on_wall", i)
        if len(agents) != 1 or len(boxes) != n_boxes:
            violation = violation or ("hallucination", i)
            continue
        cur = (agents, boxes)
        if states and cur != states[-1]:
            if not _legal(states[-1], cur, src):
                violation = violation or ("illegal_transition", i)
            moves += 1
        states.append(cur)
    solved = bool(states) and set(states[-1][1]) == goals
    start_ok = bool(states) and states[0] == ((tuple(src["player_start"]),), frozenset(tuple(b) for b in src["boxes_start"]))
    if states and not start_ok:
        violation = violation or ("wrong_start", 0)
    return {"success": bool(solved and violation is None and eos is not None), "solved": solved,
            "violation": violation[0] if violation else None, "violation_frame": violation[1] if violation else None,
            "moves": moves, "eos_frame": eos, "eos_detected": eos is not None, "frames": len(frames)}


def check_flowfree(frames: np.ndarray, src: dict) -> dict:
    import cv2
    up = np.stack([cv2.resize(f, (384, 384), interpolation=cv2.INTER_NEAREST) for f in frames])
    m = parse_video(up, src, "flowfree")
    reason = m.get("failure_reason")
    return {"success": bool(m["success"]), "violation": None if m["success"] or (reason or "incomplete") == "incomplete" else reason,
            "eos_detected": bool(m.get("eos_detected")), "eos_frame": m.get("eos_frame"), "failure_reason": reason,
            "frames": len(frames)}


CHECK = {"sokoban": check_sokoban, "flowfree": check_flowfree}
