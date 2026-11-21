from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .video import CANVAS_SIZE, add_eos_border, pad_to_vae_length


Cell = tuple[int, int]
FloatCell = tuple[float, float]

DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

BG = (226, 228, 235)
WALL = (52, 56, 68)
FLOOR = (244, 245, 249)
GRID = (210, 214, 224)
GOAL = (231, 83, 86)
BOX = (71, 139, 221)
BOX_BORDER = (42, 94, 166)
BOX_ON_GOAL = (72, 187, 117)
AGENT = (249, 165, 47)
AGENT_BORDER = (174, 104, 22)

CLASSIC = {
    "bg": (180, 160, 130),
    "wall": (80, 60, 40),
    "border": (50, 35, 20),
    "floor": (210, 190, 160),
    "goal": (200, 60, 60),
    "box": (160, 120, 40),
    "box_on_goal": (60, 160, 60),
    "box_border": (100, 75, 25),
    "agent": (240, 200, 40),
    "agent_border": (20, 20, 20),
}
THEMES = ("default", "classic")


def replay(
    metadata: dict[str, Any],
    directions: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grid = metadata["grid"]
    rows = int(metadata["grid_rows"])
    cols = int(metadata["grid_cols"])
    actions = list(metadata["directions"] if directions is None else directions)
    player: Cell = tuple(metadata["player_start"])
    boxes: set[Cell] = {tuple(cell) for cell in metadata["boxes_start"]}
    targets: set[Cell] = {tuple(cell) for cell in metadata["targets"]}
    states = [{"player": list(player), "boxes": [list(cell) for cell in sorted(boxes)]}]
    events: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None

    for step, action in enumerate(actions):
        if action not in DIRECTIONS:
            failure = {"step": step, "reason": "unknown_action", "action": action}
            break
        dr, dc = DIRECTIONS[action]
        next_player = player[0] + dr, player[1] + dc
        if not (
            0 <= next_player[0] < rows
            and 0 <= next_player[1] < cols
        ) or grid[next_player[0]][next_player[1]] == "#":
            failure = {"step": step, "reason": "wall", "action": action}
            break

        box_from: Cell | None = None
        box_to: Cell | None = None
        if next_player in boxes:
            box_from = next_player
            box_to = next_player[0] + dr, next_player[1] + dc
            if not (
                0 <= box_to[0] < rows
                and 0 <= box_to[1] < cols
            ) or grid[box_to[0]][box_to[1]] == "#" or box_to in boxes:
                failure = {"step": step, "reason": "illegal_push", "action": action}
                break
            boxes.remove(box_from)
            boxes.add(box_to)

        events.append(
            {
                "action": action,
                "player_from": list(player),
                "player_to": list(next_player),
                "box_from": list(box_from) if box_from is not None else None,
                "box_to": list(box_to) if box_to is not None else None,
            }
        )
        player = next_player
        states.append(
            {"player": list(player), "boxes": [list(cell) for cell in sorted(boxes)]}
        )

    solved = boxes == targets and failure is None and len(events) == len(actions)
    result = {
        "success": solved,
        "legal": failure is None and len(events) == len(actions),
        "solved": boxes == targets,
        "moves": len(actions),
        "pushes": sum(event["box_from"] is not None for event in events),
        "failure": failure,
        "final_boxes": [list(cell) for cell in sorted(boxes)],
    }
    return states, events, result


def _layout(metadata: dict[str, Any]) -> tuple[int, int, int]:
    rows = int(metadata["grid_rows"])
    cols = int(metadata["grid_cols"])
    if metadata.get("render_theme", "default") == "classic":
        tile = min(38, CANVAS_SIZE // max(rows, cols))
    else:
        tile = min(42, (CANVAS_SIZE - 24) // max(rows, cols))
    offset_x = (CANVAS_SIZE - cols * tile) // 2
    offset_y = (CANVAS_SIZE - rows * tile) // 2
    return tile, offset_x, offset_y


def _center(metadata: dict[str, Any], cell: FloatCell) -> tuple[float, float]:
    tile, ox, oy = _layout(metadata)
    r, c = cell
    return ox + (c + 0.5) * tile, oy + (r + 0.5) * tile


def _draw_frame_classic(
    metadata: dict[str, Any],
    state: dict[str, Any],
    player_position: FloatCell | None,
    moving_box_from: Cell | None,
    moving_box_position: FloatCell | None,
    moving_box_to: Cell | None,
) -> np.ndarray:
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), CLASSIC["bg"])
    draw = ImageDraw.Draw(image)
    tile, ox, oy = _layout(metadata)
    targets = {tuple(cell) for cell in metadata["targets"]}
    for r, row in enumerate(metadata["grid"]):
        for c, value in enumerate(row):
            box = [ox + c * tile, oy + r * tile, ox + (c + 1) * tile, oy + (r + 1) * tile]
            fill = CLASSIC["wall"] if value == "#" else CLASSIC["floor"]
            draw.rectangle(box, fill=fill, outline=CLASSIC["border"], width=1)
    goal_radius = tile * 0.2
    for target in targets:
        x, y = _center(metadata, target)
        draw.polygon([(x, y - goal_radius), (x + goal_radius, y),
                      (x, y + goal_radius), (x - goal_radius, y)], fill=CLASSIC["goal"])
    half = tile / 2 - (max(1, tile // 16) + 1)

    def draw_box(cell: FloatCell, on_goal: bool) -> None:
        x, y = _center(metadata, cell)
        draw.rectangle([x - half, y - half, x + half, y + half],
                       fill=CLASSIC["box_on_goal"] if on_goal else CLASSIC["box"],
                       outline=CLASSIC["box_border"], width=1)

    for box in {tuple(cell) for cell in state["boxes"]}:
        if moving_box_from is None or box != moving_box_from:
            draw_box(box, box in targets)
    if moving_box_position is not None:
        destination = moving_box_to if moving_box_to is not None else (
            round(moving_box_position[0]), round(moving_box_position[1]))
        draw_box(moving_box_position, tuple(destination) in targets)
    player = tuple(state["player"]) if player_position is None else player_position
    x, y = _center(metadata, player)
    radius = tile * 0.3
    draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                 fill=CLASSIC["agent"], outline=CLASSIC["agent_border"], width=2)
    return np.asarray(image)


def _draw_frame(
    metadata: dict[str, Any],
    state: dict[str, Any],
    *,
    player_position: FloatCell | None = None,
    moving_box_from: Cell | None = None,
    moving_box_position: FloatCell | None = None,
    moving_box_to: Cell | None = None,
) -> np.ndarray:
    if metadata.get("render_theme", "default") == "classic":
        return _draw_frame_classic(metadata, state, player_position,
                                   moving_box_from, moving_box_position, moving_box_to)
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BG)
    draw = ImageDraw.Draw(image)
    tile, ox, oy = _layout(metadata)
    grid = metadata["grid"]
    targets = {tuple(cell) for cell in metadata["targets"]}
    boxes = {tuple(cell) for cell in state["boxes"]}

    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            x0, y0 = ox + c * tile, oy + r * tile
            x1, y1 = x0 + tile, y0 + tile
            if value == "#":
                draw.rounded_rectangle(
                    [x0 + 1, y0 + 1, x1 - 1, y1 - 1],
                    radius=max(2, tile // 9),
                    fill=WALL,
                )
            else:
                draw.rectangle([x0, y0, x1, y1], fill=FLOOR, outline=GRID, width=1)

    goal_radius = tile * 0.23
    for target in targets:
        x, y = _center(metadata, target)
        draw.ellipse(
            [x - goal_radius, y - goal_radius, x + goal_radius, y + goal_radius],
            outline=GOAL,
            width=max(2, tile // 10),
        )

    box_half = tile * 0.34
    for box in boxes:
        if moving_box_from is not None and box == moving_box_from:
            continue
        x, y = _center(metadata, box)
        color = BOX_ON_GOAL if box in targets else BOX
        draw.rounded_rectangle(
            [x - box_half, y - box_half, x + box_half, y + box_half],
            radius=max(2, tile // 9),
            fill=color,
            outline=BOX_BORDER,
            width=max(1, tile // 16),
        )
    if moving_box_position is not None:
        x, y = _center(metadata, moving_box_position)
        draw.rounded_rectangle(
            [x - box_half, y - box_half, x + box_half, y + box_half],
            radius=max(2, tile // 9),
            fill=BOX,
            outline=BOX_BORDER,
            width=max(1, tile // 16),
        )

    player = (
        tuple(state["player"])
        if player_position is None
        else player_position
    )
    x, y = _center(metadata, player)
    radius = tile * 0.27
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=AGENT,
        outline=AGENT_BORDER,
        width=max(2, tile // 12),
    )
    return np.asarray(image)


def _interpolate(a: Cell, b: Cell, t: float) -> FloatCell:
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def _append_action_frames(
    frames: list[np.ndarray],
    metadata: dict[str, Any],
    states: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    frames_per_move: int,
    transition_mode: str = "continuous",
) -> None:
    if transition_mode not in {"continuous", "discrete"}:
        raise ValueError(f"unsupported transition_mode: {transition_mode}")
    for index, event in enumerate(events):
        before = states[index]
        player_from = tuple(event["player_from"])
        player_to = tuple(event["player_to"])
        box_from = (
            tuple(event["box_from"]) if event["box_from"] is not None else None
        )
        box_to = tuple(event["box_to"]) if event["box_to"] is not None else None
        for step in range(1, frames_per_move + 1):
            if transition_mode == "discrete":
                if step < frames_per_move:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(
                        _draw_frame(
                            metadata,
                            before,
                            player_position=player_to,
                            moving_box_from=box_from,
                            moving_box_position=box_to,
                        )
                    )
                continue
            amount = step / frames_per_move
            player_position = _interpolate(player_from, player_to, amount)
            moving_box = (
                _interpolate(box_from, box_to, amount)
                if box_from is not None and box_to is not None
                else None
            )
            frames.append(
                _draw_frame(
                    metadata,
                    before,
                    player_position=player_position,
                    moving_box_from=box_from,
                    moving_box_position=moving_box,
                    moving_box_to=box_to,
                )
            )


def render_optimal(
    metadata: dict[str, Any],
    *,
    frames_per_move: int = 3,
    theme: str = "default",
    transition_mode: str = "continuous",
    start_hold: int = 6,
) -> np.ndarray:
    if theme not in THEMES:
        raise ValueError(f"unknown Sokoban theme: {theme}")
    metadata = {**metadata, "render_theme": theme}
    states, events, result = replay(metadata)
    if not result["success"]:
        raise ValueError(f"invalid Sokoban source: {result}")
    frames: list[np.ndarray] = []
    for _ in range(start_hold):
        frames.append(_draw_frame(metadata, states[0]))
    _append_action_frames(
        frames,
        metadata,
        states,
        events,
        frames_per_move=frames_per_move,
        transition_mode=transition_mode,
    )
    for _ in range(8):
        frames.append(_draw_frame(metadata, states[-1]))
    return add_eos_border(pad_to_vae_length(np.stack(frames)))


DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


MAX_SOKOBAN_STATES = 3_000_000


def sokoban_solve(walls, player, boxes, targets, max_states=MAX_SOKOBAN_STATES):
    from collections import deque
    targets = frozenset(targets)
    rows, cols = len(walls), len(walls[0])

    def floor(cell):
        r, c = cell
        return 0 <= r < rows and 0 <= c < cols and walls[r][c] != "#"

    dead = set()
    for r in range(rows):
        for c in range(cols):
            if floor((r, c)) and (r, c) not in targets:
                vert = not floor((r - 1, c)) or not floor((r + 1, c))
                horiz = not floor((r, c - 1)) or not floor((r, c + 1))
                if vert and horiz:
                    dead.add((r, c))
    start = (tuple(player), frozenset(map(tuple, boxes)))
    parents = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        pl, bx = state
        if bx <= targets:
            moves = []
            while parents[state]:
                state, name = parents[state]
                moves.append(name)
            return moves[::-1]
        for name, (dr, dc) in DIRS.items():
            nxt = (pl[0] + dr, pl[1] + dc)
            if not floor(nxt):
                continue
            nb = bx
            if nxt in bx:
                dest = (nxt[0] + dr, nxt[1] + dc)
                if not floor(dest) or dest in bx or dest in dead:
                    continue
                nb = (bx - {nxt}) | {dest}
            key = (nxt, nb)
            if key not in parents:
                parents[key] = (state, name)
                queue.append(key)
                if len(parents) > max_states:
                    return None
    return None


def floor_cells(grid):
    return [(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch != "#"]


def sokoban_source(sample_id, grid, player, boxes, targets, moves, extra=None):
    return {"sample_id": sample_id, "grid_rows": len(grid), "grid_cols": len(grid[0]), "num_boxes": len(boxes),
            "grid": grid, "player_start": list(player), "boxes_start": [list(b) for b in boxes],
            "targets": [list(t) for t in targets], "num_moves": len(moves), "directions": moves,
            "num_frames": len(moves) + 1, **(extra or {})}
