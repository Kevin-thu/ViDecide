from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Literal, Optional, Tuple, Any

import cv2
import numpy as np


WALL = 0


FLOOR = 1


GOAL = 2


DIRECTIONS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


@dataclass(frozen=True)
class SokobanState:
    agent: Tuple[int, int]
    boxes: FrozenSet[Tuple[int, int]]


@dataclass
class SokobanLevel:
    grid: np.ndarray
    agent_start: Tuple[int, int]
    box_starts: FrozenSet[Tuple[int, int]]
    goals: FrozenSet[Tuple[int, int]]
    level_id: str
    source: str
    difficulty: str = "medium"
    safe_squares: FrozenSet[Tuple[int, int]] = field(default_factory=frozenset, repr=False)

    @property
    def rows(self) -> int:
        return self.grid.shape[0]

    @property
    def cols(self) -> int:
        return self.grid.shape[1]

    @property
    def n_boxes(self) -> int:
        return len(self.box_starts)

    @property
    def initial_state(self) -> SokobanState:
        return SokobanState(self.agent_start, self.box_starts)

    def grid_hash(self) -> str:
        return hashlib.md5(self.grid.tobytes()).hexdigest()


CANVAS_SIDE = 384


HOLD_FRAMES = 8


ABSTRACT_PALETTES: Dict[str, Dict[str, Tuple[int,int,int]]] = {
    "classic": {
        "wall":        (80,  60,  40),
        "floor":       (210, 190, 160),
        "goal":        (200, 60,  60),
        "box":         (160, 120, 40),
        "box_on_goal": (60,  160, 60),
        "agent":       (240, 200, 40),
        "bg":          (180, 160, 130),
    },
    "modern_flat": {
        "wall":        (50,  50,  60),
        "floor":       (240, 240, 245),
        "goal":        (230, 80,  80),
        "box":         (80,  140, 220),
        "box_on_goal": (80,  200, 120),
        "agent":       (255, 160, 0),
        "bg":          (230, 230, 235),
    },
    "dark": {
        "wall":        (40,  40,  50),
        "floor":       (60,  60,  70),
        "goal":        (255, 80,  80),
        "box":         (200, 180, 60),
        "box_on_goal": (80,  255, 120),
        "agent":       (100, 200, 255),
        "bg":          (20,  20,  25),
    },
    "pastel": {
        "wall":        (160, 150, 170),
        "floor":       (245, 240, 250),
        "goal":        (240, 140, 140),
        "box":         (140, 180, 220),
        "box_on_goal": (140, 210, 160),
        "agent":       (255, 200, 150),
        "bg":          (235, 230, 240),
    },
}


SPRITE_THEMES = {
    "warehouse", "cartoon", "pixel_art", "retro", "nature",
    "industrial", "fantasy", "arctic", "desert",
}


def _compute_layout(rows: int, cols: int):
    tile_size = min(CANVAS_SIDE // max(rows, cols), 38)
    offset_y = (CANVAS_SIDE - rows * tile_size) // 2
    offset_x = (CANVAS_SIDE - cols * tile_size) // 2
    return tile_size, offset_x, offset_y


def _tile_center(r: int, c: int, tile_size: int, offset_x: int, offset_y: int):
    cx = offset_x + c * tile_size + tile_size // 2
    cy = offset_y + r * tile_size + tile_size // 2
    return cx, cy


def _sq_dist(px: np.ndarray, color: Tuple[int,int,int]) -> int:
    c = np.array(color, dtype=np.int32)
    return int(np.sum((px - c) ** 2))


def _classify_pixel(px: np.ndarray, palette: Dict[str, Tuple[int,int,int]]) -> str:
    best, best_d = "floor", float("inf")
    for cat, color in palette.items():
        d = _sq_dist(px, color)
        if d < best_d:
            best, best_d = cat, d
    return best


def _is_red_border_frame(frame_rgb: np.ndarray, border_px: int = 14,
                          red_ratio: float = 2.5, min_val: int = 100) -> bool:
    top    = frame_rgb[:border_px, :, :]
    bottom = frame_rgb[-border_px:, :, :]
    left   = frame_rgb[:, :border_px, :]
    right  = frame_rgb[:, -border_px:, :]
    for strip in (top, bottom, left, right):
        r = strip[:,:,0].mean()
        g = strip[:,:,1].mean()
        b = strip[:,:,2].mean()
        if r < min_val or r < red_ratio * g or r < red_ratio * b:
            return False
    return True


def _is_rewind_frame(frame_rgb: np.ndarray, border_px: int = 14,
                     magenta_thresh: float = 0.08) -> bool:
    top    = frame_rgb[:border_px, :, :].astype(np.float32)
    bottom = frame_rgb[-border_px:, :, :].astype(np.float32)
    left   = frame_rgb[:, :border_px, :].astype(np.float32)
    right  = frame_rgb[:, -border_px:, :].astype(np.float32)
    for strip in (top, bottom, left, right):
        r, g, b = strip[:,:,0].mean(), strip[:,:,1].mean(), strip[:,:,2].mean()
        if not (r > 150 and b > 120 and g < r * 0.6):
            return False
    return True


@dataclass(frozen=True)
class VisualComponent:
    kind: str
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: int


@dataclass(frozen=True)
class VisualFrame:
    boxes: list[VisualComponent]
    agents: list[VisualComponent]
    wall_violations: list[tuple[int, int]]
    goal_marker_missing: list[tuple[int, int]] = ()


def _near_color_mask(crop: np.ndarray, color: tuple[int, int, int],
                     *, tolerance: int) -> np.ndarray:
    diff = np.abs(crop.astype(np.int16) - np.asarray(color, dtype=np.int16))
    return np.max(diff, axis=-1) <= tolerance


def _components_from_mask(mask: np.ndarray, kind: str,
                          *, min_area: int) -> list[VisualComponent]:
    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    comps: list[VisualComponent] = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = cents[label]
        comps.append(VisualComponent(
            kind=kind,
            centroid=(float(cy), float(cx)),
            bbox=(y, x, y + h - 1, x + w - 1),
            area=area,
        ))
    return comps


def decode_visual_frame(frame_rgb: np.ndarray, level: SokobanLevel,
                        palette: dict,
                        tile_size: int, offset_x: int,
                        offset_y: int) -> VisualFrame:
    rows, cols = level.rows, level.cols
    y0, y1 = offset_y, offset_y + rows * tile_size
    x0, x1 = offset_x, offset_x + cols * tile_size
    crop = frame_rgb[y0:y1, x0:x1]

    board_mask = np.zeros(crop.shape[:2], dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if level.grid[r, c] != WALL:
                yy0, yy1 = r * tile_size, (r + 1) * tile_size
                xx0, xx1 = c * tile_size, (c + 1) * tile_size
                board_mask[yy0:yy1, xx0:xx1] = True

    min_box_area = int(0.20 * tile_size * tile_size)
    min_agent_area = int(0.05 * tile_size * tile_size)
    box_mask = (
        _near_color_mask(crop, palette["box"], tolerance=58)
        | _near_color_mask(crop, palette["box_on_goal"], tolerance=58)
    )
    agent_mask = _near_color_mask(crop, palette["agent"], tolerance=58)
    boxes = _components_from_mask(
        box_mask & board_mask,
        "box",
        min_area=min_box_area,
    )
    agents = _components_from_mask(
        agent_mask & board_mask,
        "agent",
        min_area=min_agent_area,
    )

    wall_violations: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if level.grid[r, c] != WALL:
                continue
            cx, cy = _tile_center(r, c, tile_size, offset_x, offset_y)
            px = frame_rgb[cy, cx].astype(np.int32)
            if _classify_pixel(px, palette) not in ("wall", "bg"):
                wall_violations.append((r, c))

    goal_marker_missing: list[tuple[int, int]] = []
    cover_threshold = 0.15
    tile_area = tile_size * tile_size
    for (r, c) in level.goals:
        yy0, yy1 = r * tile_size, (r + 1) * tile_size
        xx0, xx1 = c * tile_size, (c + 1) * tile_size
        box_cov = float(box_mask[yy0:yy1, xx0:xx1].sum()) / tile_area
        agent_cov = float(agent_mask[yy0:yy1, xx0:xx1].sum()) / tile_area
        if box_cov >= cover_threshold or agent_cov >= cover_threshold:
            continue
        cx_, cy_ = _tile_center(r, c, tile_size, offset_x, offset_y)
        px = frame_rgb[cy_, cx_].astype(np.int32)
        if _classify_pixel(px, palette) != "goal":
            goal_marker_missing.append((r, c))

    return VisualFrame(boxes=boxes, agents=agents,
                       wall_violations=wall_violations,
                       goal_marker_missing=goal_marker_missing)


def _component_to_cell(component: VisualComponent, level: SokobanLevel,
                       tile_size: int,
                       *, max_center_dist: float,
                       max_extent: float) -> Optional[tuple[int, int]]:
    y, x = component.centroid
    r = int(y // tile_size)
    c = int(x // tile_size)
    if not (0 <= r < level.rows and 0 <= c < level.cols):
        return None
    if level.grid[r, c] == WALL:
        return None

    center_y = r * tile_size + tile_size / 2.0
    center_x = c * tile_size + tile_size / 2.0
    if abs(y - center_y) > max_center_dist * tile_size:
        return None
    if abs(x - center_x) > max_center_dist * tile_size:
        return None

    y0, x0, y1, x1 = component.bbox
    if (y1 - y0 + 1) > max_extent * tile_size:
        return None
    if (x1 - x0 + 1) > max_extent * tile_size:
        return None
    return (r, c)


def snap_visual_to_state(visual: VisualFrame, level: SokobanLevel,
                         tile_size: int) -> Optional[SokobanState]:
    if len(visual.boxes) != level.n_boxes or len(visual.agents) != 1:
        return None

    box_cells: list[tuple[int, int]] = []
    for comp in visual.boxes:
        cell = _component_to_cell(
            comp, level, tile_size,
            max_center_dist=0.36,
            max_extent=1.35,
        )
        if cell is None:
            return None
        box_cells.append(cell)
    if len(set(box_cells)) != level.n_boxes:
        return None

    agent_cell = _component_to_cell(
        visual.agents[0], level, tile_size,
        max_center_dist=0.42,
        max_extent=1.35,
    )
    if agent_cell is None:
        return None
    return SokobanState(agent_cell, frozenset(box_cells))


@dataclass
class Event:
    kind: Literal["move", "push", "undo_move", "undo_push", "think"]
    direction: str
    agent_from: Tuple[int, int]
    agent_to: Tuple[int, int]
    box_from: Tuple[int, int] | None = None
    box_to: Tuple[int, int] | None = None
    reason: str | None = None
    focus_cells: Tuple[Tuple[int, int], ...] | None = None


_DELTA_TO_DIR = {v: k for k, v in DIRECTIONS.items()}


@dataclass
class ParseResult:
    events: list[Event]
    pixel_at_event_end: list[int]
    hall_pixel: Optional[int]
    hall_reason: Optional[str]
    goal_marker_hallucination: list[tuple[int, int]]
    illegal_undo_indices: list[int]
    illegal_transitions: list[dict]
    n_frames: int
    n_bad_frames: int
    per_frame_state: list[Optional[SokobanState]]
    per_frame_bad: list[bool]


def _state_diff_to_event(s1: SokobanState, s2: SokobanState) -> Optional[Event]:
    if s1 == s2:
        return None
    dr = s2.agent[0] - s1.agent[0]
    dc = s2.agent[1] - s1.agent[1]
    lost = s1.boxes - s2.boxes
    gained = s2.boxes - s1.boxes

    if abs(dr) + abs(dc) == 1 and not lost and not gained:
        return Event("move", _DELTA_TO_DIR[(dr, dc)], s1.agent, s2.agent)

    if len(lost) == 1 and len(gained) == 1:
        bf = next(iter(lost))
        bt = next(iter(gained))
        bdr = bt[0] - bf[0]
        bdc = bt[1] - bf[1]
        if abs(bdr) + abs(bdc) != 1:
            return None
        if (abs(dr) + abs(dc) == 1
                and bdr == dr and bdc == dc
                and bf == (s1.agent[0] + dr, s1.agent[1] + dc)):
            return Event("push", _DELTA_TO_DIR[(dr, dc)],
                         s1.agent, s2.agent, bf, bt)
        d_undo = _DELTA_TO_DIR.get((-bdr, -bdc))
        if d_undo is None:
            return None
        expected_post_agent = (2 * bt[0] - bf[0], 2 * bt[1] - bf[1])
        if s2.agent != expected_post_agent:
            return None
        return Event("undo_push", d_undo, s1.agent, s2.agent, bf, bt)
    return None


def parse_rollout(mp4_pixels: np.ndarray, level: SokobanLevel, theme: str,
                  *,
                  sustain_settle: int = 2,
                  sustain_bad: int = 12) -> ParseResult:
    if theme in SPRITE_THEMES:
        raise ValueError(f"sprite theme {theme!r} not supported by eval parser")
    palette = ABSTRACT_PALETTES.get(theme)
    if palette is None:
        raise ValueError(f"unknown theme {theme!r}")
    rows, cols = level.rows, level.cols
    tile_size, offset_x, offset_y = _compute_layout(rows, cols)
    n_boxes = level.n_boxes
    n_frames = int(mp4_pixels.shape[0])
    first_eos_frame: Optional[int] = None
    for fi in range(n_frames):
        if _is_red_border_frame(mp4_pixels[fi]):
            first_eos_frame = fi
            break
    parse_end = first_eos_frame if first_eos_frame is not None else n_frames

    per_frame_state: list[Optional[SokobanState]] = [None] * n_frames
    per_frame_bad = [False] * n_frames
    bad_reasons: list[Optional[str]] = [None] * n_frames
    per_frame_box_count: list[Optional[int]] = [None] * n_frames
    per_frame_goal_missing: list[list[tuple[int, int]]] = [
        [] for _ in range(n_frames)
    ]

    for fi in range(min(HOLD_FRAMES, parse_end)):
        per_frame_state[fi] = level.initial_state
        per_frame_box_count[fi] = n_boxes

    for fi in range(HOLD_FRAMES, parse_end):
        f = mp4_pixels[fi]
        if _is_rewind_frame(f):
            continue
        visual = decode_visual_frame(
            f, level, palette, tile_size, offset_x, offset_y,
        )
        n_det = len(visual.boxes)
        per_frame_box_count[fi] = n_det
        n_agents = len(visual.agents)
        if n_det != n_boxes:
            per_frame_bad[fi] = True
            bad_reasons[fi] = f"box_count={n_det}!={n_boxes}"
        elif visual.wall_violations:
            per_frame_bad[fi] = True
            bad_reasons[fi] = f"wall_violations={visual.wall_violations}"
        elif n_agents > 1:
            per_frame_bad[fi] = True
            bad_reasons[fi] = f"multi_agent={n_agents}"
        per_frame_goal_missing[fi] = list(visual.goal_marker_missing)
        if not per_frame_bad[fi]:
            per_frame_state[fi] = snap_visual_to_state(visual, level, tile_size)

    hall_pixel = None
    hall_reason = None
    bad_run = 0
    bad_start: Optional[int] = None
    bad_reason_at_start: Optional[str] = None

    def _bad_threshold(reason: Optional[str]) -> int:
        reason = reason or ""
        if reason.startswith("multi_agent=") or reason.startswith("wall_violations="):
            return 3
        if reason.startswith("goal_marker_missing="):
            return 3
        if reason.startswith("box_count="):
            return sustain_bad
        return sustain_bad

    for fi in range(parse_end):
        reason = bad_reasons[fi]
        if per_frame_bad[fi] and reason == bad_reason_at_start:
            if bad_start is None:
                bad_start = fi
            bad_run += 1
            if bad_run >= _bad_threshold(bad_reason_at_start) and hall_pixel is None:
                hall_pixel = bad_start
                hall_reason = bad_reasons[bad_start]
        elif per_frame_bad[fi]:
            bad_start = fi
            bad_reason_at_start = reason
            bad_run = 1
            if bad_run >= _bad_threshold(bad_reason_at_start) and hall_pixel is None:
                hall_pixel = bad_start
                hall_reason = bad_reasons[bad_start]
        else:
            bad_run = 0
            bad_start = None
            bad_reason_at_start = None

    if hall_pixel is None:
        tail_end: Optional[int] = None
        tail_count: Optional[int] = None
        for fi in range(parse_end - 1, -1, -1):
            n_det = per_frame_box_count[fi]
            if n_det is None:
                continue
            tail_end = fi
            tail_count = n_det
            break
        if tail_count is not None and tail_count < n_boxes:
            tail_start = tail_end
            tail_run = 1
            for fi in range(tail_end - 1, -1, -1):
                n_det = per_frame_box_count[fi]
                if n_det is not None and n_det < n_boxes:
                    tail_start = fi
                    tail_run += 1
                    continue
                if n_det is None:
                    continue
                break
            if tail_run >= max(3, sustain_settle):
                hall_pixel = tail_start
                hall_reason = f"box_count={tail_count}!={n_boxes}_at_end"

    miss_run = 0
    miss_start: Optional[int] = None
    last_boxes_before_miss: Optional[frozenset] = None
    for fi in range(parse_end):
        if per_frame_state[fi] is not None:
            last_boxes_before_miss = per_frame_state[fi].boxes
            miss_run = 0
            miss_start = None
            continue
        if per_frame_bad[fi]:
            miss_run = 0
            miss_start = None
            continue
        if miss_start is None:
            miss_start = fi
        miss_run += 1
        if miss_run >= sustain_bad and hall_pixel is None:
            hall_pixel = miss_start
            hall_reason = "agent_missing_stuck"

    settled_runs: list[tuple[int, int, SokobanState]] = []
    fi = 0
    while fi < parse_end:
        s = per_frame_state[fi]
        if s is None:
            fi += 1
            continue
        end = fi
        while end + 1 < parse_end and per_frame_state[end + 1] == s:
            end += 1
        if end - fi + 1 >= sustain_settle:
            settled_runs.append((fi, end, s))
        fi = end + 1

    def _transition_crosses_real_hallucination(prev_end: int, next_start: int) -> bool:
        for fj in range(prev_end + 1, next_start):
            if not per_frame_bad[fj]:
                continue
            reason = bad_reasons[fj] or ""
            if reason.startswith("box_count="):
                continue
            return True
        return False

    events: list[Event] = []
    pixel_at_event_end: list[int] = []
    push_stack: list[Event] = []
    illegal_undo_indices: list[int] = []
    illegal_transitions: list[dict] = []
    for i in range(len(settled_runs) - 1):
        prev_end = settled_runs[i][1]
        next_start = settled_runs[i + 1][0]
        crosses_bad = _transition_crosses_real_hallucination(prev_end, next_start)
        s_a = settled_runs[i][2]
        s_b = settled_runs[i + 1][2]
        ev = _state_diff_to_event(s_a, s_b)
        if ev is None:
            if s_a == s_b:
                continue
            lost = sorted(s_a.boxes - s_b.boxes)
            gained = sorted(s_b.boxes - s_a.boxes)
            box_changed = bool(lost or gained)
            if box_changed and not crosses_bad:
                illegal_transitions.append({
                    "pixel_end": next_start,
                    "agent_from": list(s_a.agent),
                    "agent_to": list(s_b.agent),
                    "boxes_lost": [list(b) for b in lost],
                    "boxes_gained": [list(b) for b in gained],
                    "reason": "no_legal_single_action_fits_diff",
                })
            continue
        if ev.kind == "push":
            push_stack.append(ev)
        elif ev.kind == "undo_push":
            top = push_stack[-1] if push_stack else None
            legal = (
                top is not None
                and top.box_to == ev.box_from
                and top.box_from == ev.box_to
            )
            if legal:
                push_stack.pop()
        events.append(ev)
        pixel_at_event_end.append(next_start)

    sustain_goal = 8
    cell_uncovered_missing = {(r, c): 0 for (r, c) in level.goals}
    seen_persistent: set[tuple[int, int]] = set()
    for fi in range(parse_end):
        st = per_frame_state[fi]
        miss = per_frame_goal_missing[fi]
        if not miss:
            continue
        for cell in miss:
            if st is not None and cell in st.boxes:
                continue
            if st is not None and cell == st.agent:
                continue
            cell_uncovered_missing[cell] += 1
            if cell_uncovered_missing[cell] >= sustain_goal:
                seen_persistent.add(cell)
    goal_marker_hallucination = sorted(seen_persistent)

    return ParseResult(
        events=events,
        pixel_at_event_end=pixel_at_event_end,
        hall_pixel=hall_pixel,
        hall_reason=hall_reason,
        goal_marker_hallucination=goal_marker_hallucination,
        illegal_undo_indices=illegal_undo_indices,
        illegal_transitions=illegal_transitions,
        n_frames=n_frames,
        n_bad_frames=sum(per_frame_bad[:parse_end]),
        per_frame_state=per_frame_state,
        per_frame_bad=per_frame_bad,
    )


def _import():
    return FLOOR, GOAL, WALL, SokobanLevel, parse_rollout, _is_red_border_frame


def level_from_source(source: dict[str, Any]):
    FLOOR, GOAL, WALL, SokobanLevel, _, _ = _import()
    targets = {tuple(cell) for cell in source["targets"]}
    grid = np.array([[WALL if ch == "#" else (GOAL if (r, c) in targets else FLOOR)
                      for c, ch in enumerate(row)] for r, row in enumerate(source["grid"])], dtype=np.uint8)
    return SokobanLevel(grid=grid, agent_start=tuple(source["player_start"]),
                        box_starts=frozenset(tuple(cell) for cell in source["boxes_start"]),
                        goals=frozenset(targets), level_id=str(source.get("sample_id", "puzzle")),
                        source="videorlvr")


def _seen_two_step(frames, level, theme, states, t):
    end = int(t['pixel_end'])
    after = states[end] if end < len(states) else None
    if after is None:
        return False
    i = end - 1
    while i >= 0 and (states[i] is None or states[i] == after):
        i -= 1
    if i < 0:
        return False
    before = states[i]
    while i > 0 and (states[i - 1] is None or states[i - 1] == before):
        i -= 1
    if abs(after.agent[0] - before.agent[0]) + abs(after.agent[1] - before.agent[1]) != 2:
        return False
    rows, cols = level.grid.shape
    free = lambda c: 0 <= c[0] < rows and 0 <= c[1] < cols and int(level.grid[c]) != 0
    def step(agent, boxes, d):
        n = (agent[0] + d[0], agent[1] + d[1])
        if not free(n): return None
        if n in boxes:
            m = (n[0] + d[0], n[1] + d[1])
            if not free(m) or m in boxes: return None
            return n, (boxes - {n}) | {m}
        return n, boxes
    start = (tuple(before.agent), frozenset(map(tuple, before.boxes)))
    goal = (tuple(after.agent), frozenset(map(tuple, after.boxes)))
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    middles = set()
    for d1 in dirs:
        m = step(*start, d1)
        if m and any(step(*m, d2) == goal for d2 in dirs):
            middles.add(m)
    if not middles:
        return False
    tile, ox, oy = _compute_layout(level.rows, level.cols)
    for fi in range(i, end):
        st = snap_visual_to_state(decode_visual_frame(frames[fi], level, ABSTRACT_PALETTES[theme], tile, ox, oy), level, tile)
        if st is not None and (tuple(st.agent), frozenset(map(tuple, st.boxes))) in middles:
            return True
    return False


def check_sokoban_video(frames: np.ndarray, source: dict[str, Any], theme: str = "classic",
                        sustain_settle: int = 2, pulls_illegal: bool = True) -> dict[str, Any]:
    _, _, _, _, parse_rollout, is_eos = _import()
    level = level_from_source(source)
    eos = next((i for i, frame in enumerate(frames) if is_eos(frame)), None)
    end = eos if eos is not None else len(frames)
    parsed = parse_rollout(np.asarray(frames), level, theme, sustain_settle=sustain_settle)
    states = [s for s in parsed.per_frame_state[:end] if s is not None]
    on_goal = [len(level.goals & s.boxes) for s in states]
    solved_any = bool(on_goal) and max(on_goal) >= len(level.goals)
    solved = bool(on_goal) and on_goal[-1] >= len(level.goals)
    goal_vanished = [list(c) for c in parsed.goal_marker_hallucination]
    illegal = len(parsed.illegal_transitions) + len(parsed.illegal_undo_indices)
    illegal_merged = 0
    if parsed.illegal_transitions:
        explained = [t for t in parsed.illegal_transitions if _seen_two_step(np.asarray(frames), level, theme, parsed.per_frame_state, t)]
        illegal_merged, illegal = len(explained), illegal - len(explained)
    pulls = sum(e.kind == "undo_push" for e in parsed.events)
    if pulls_illegal:
        illegal += pulls
    hallucinated = parsed.hall_pixel is not None
    return {
        "success": bool(solved and not hallucinated and illegal == 0 and not goal_vanished),
        "solved": solved,
        "solved_any_time": solved_any,
        "goal_vanished": bool(goal_vanished),
        "illegal_merged_fast_moves": illegal_merged,
        "pulls": pulls,
        "hallucination": parsed.hall_reason,
        "hall_frame": parsed.hall_pixel,
        "illegal_transitions": illegal,
        "events": len(parsed.events),
        "pushes": sum(e.kind == "push" for e in parsed.events),
        "boxes_on_goals_max": max(on_goal) if on_goal else 0,
        "n_goals": len(level.goals),
        "eos_frame": eos,
        "goal_marker_hallucination": [list(c) for c in parsed.goal_marker_hallucination],
        "checker": "sokoban_strict" if pulls_illegal else "sokoban_strict_pulls_allowed",
    }
