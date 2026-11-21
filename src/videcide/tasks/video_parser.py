from __future__ import annotations

from collections import deque
from typing import Any, Iterable

import numpy as np

from .video import EOS_COLOR


Cell = tuple[int, int]


def find_eos_frame(frames: np.ndarray, *, min_fraction: float = 0.70) -> int | None:
    height, width = frames.shape[1:3]
    border = max(3, int(round(10 * min(height, width) / 384)))
    mask = np.zeros((height, width), dtype=bool)
    mask[:border] = True
    mask[-border:] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    target = EOS_COLOR.astype(np.float32)
    for index, frame in enumerate(frames):
        pixels = frame[mask].astype(np.float32)
        close_to_target = np.linalg.norm(pixels - target, axis=-1) <= 75.0
        red = pixels[:, 0]
        other = np.maximum(pixels[:, 1], pixels[:, 2])
        red_dominant = (red >= 60.0) & (red >= 1.5 * other) & (red - other >= 30.0)
        fraction = float((close_to_target | red_dominant).mean())
        if fraction >= min_fraction:
            return index
    return None


def _color_fraction(
    frame: np.ndarray,
    center: tuple[float, float],
    color: tuple[int, int, int],
    *,
    radius: int,
    threshold: float,
) -> float:
    x, y = center
    x0 = max(0, int(round(x)) - radius)
    x1 = min(frame.shape[1], int(round(x)) + radius + 1)
    y0 = max(0, int(round(y)) - radius)
    y1 = min(frame.shape[0], int(round(y)) + radius + 1)
    patch = frame[y0:y1, x0:x1].astype(np.float32)
    distance = np.linalg.norm(patch - np.asarray(color, dtype=np.float32), axis=-1)
    return float((distance <= threshold).mean())


def _flow_center(size: int, cell: Cell, canvas: int) -> tuple[float, float]:
    margin = 28
    width = (canvas - 2 * margin) / size
    row, col = cell
    return margin + (col + 0.5) * width, margin + (row + 0.5) * width


def _neighbors(cell: Cell, rows: int, cols: int) -> Iterable[Cell]:
    row, col = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nxt = row + dr, col + dc
        if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols:
            yield nxt


def _connected(cells: set[Cell], rows: int, cols: int) -> bool:
    if not cells:
        return False
    reached = {next(iter(cells))}
    queue = deque(reached)
    while queue:
        for nxt in _neighbors(queue.popleft(), rows, cols):
            if nxt in cells and nxt not in reached:
                reached.add(nxt)
                queue.append(nxt)
    return reached == cells


def _maze_layout(frame: np.ndarray, source: dict[str, Any]) -> tuple[int, int, int]:
    grid = source["grid"]
    rows, cols = len(grid), len(grid[0])
    cell_size = max(4, (frame.shape[0] - 24) // max(rows, cols))
    return cell_size, (frame.shape[1] - cols * cell_size) // 2, (frame.shape[0] - rows * cell_size) // 2


def _maze_ink(frame: np.ndarray, base: np.ndarray, source: dict[str, Any]) -> np.ndarray:
    """Per-pixel mask of drawn ink (line or cursor) in one frame, independent of the line colour.

    Generated videos drift in tone over a rollout and draw the line in varying shades, so neither a fixed line colour
    nor "changed since frame 0" works. Ink is a pixel that is far from this frame's own wall colour (median of the
    wall cells) and from this frame's own free-corridor colour (median of the corridor cells still close to the
    frame-0 corridor colour), and that changed since frame 0 (so the static start and goal markers never count).
    """
    grid = source["grid"]
    rows, cols = len(grid), len(grid[0])
    cell, ox, oy = _maze_layout(frame, source)

    def centres(cells):
        return np.array([[int(oy + (r + 0.5) * cell), int(ox + (c + 0.5) * cell)] for r, c in cells])

    walls = centres([(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != 0])
    corridors = centres([(r, c) for r in range(1, rows, 2) for c in range(1, cols, 2)])
    f = frame.astype(np.float32)
    b = base.astype(np.float32)
    wall_now = np.median(f[walls[:, 0], walls[:, 1]], axis=0)
    corridor0 = np.median(b[corridors[:, 0], corridors[:, 1]], axis=0)
    corr_now = f[corridors[:, 0], corridors[:, 1]]
    free = corr_now[np.linalg.norm(corr_now - corridor0, axis=1) < 60]
    corridor_now = np.median(free, axis=0) if len(free) else corridor0
    # corridor margin 90: the line of every theme is at least 150 from its corridor colour, while generated frames
    # show faint smudges 60-90 away from a white corridor
    return ((np.linalg.norm(f - wall_now, axis=-1) > 50) & (np.linalg.norm(f - corridor_now, axis=-1) > 90)
            & (np.linalg.norm(f - b, axis=-1) > 40))


def _disk_fraction(mask: np.ndarray, center: tuple[float, float], radius: int) -> float:
    x, y = center
    x0, x1 = max(0, int(round(x)) - radius), min(mask.shape[1], int(round(x)) + radius + 1)
    y0, y1 = max(0, int(round(y)) - radius), min(mask.shape[0], int(round(y)) + radius + 1)
    patch = mask[y0:y1, x0:x1]
    return float(patch.mean()) if patch.size else 0.0


def _maze_occupied(frame: np.ndarray, source: dict[str, Any], base: np.ndarray | None = None) -> set[Cell]:
    """Cells the drawn line covers in one frame (see _maze_ink).

    The start and goal cells are covered by their markers, so an endpoint counts only if a drawn neighbour leads into
    it and there is ink at a probe 0.4 cell towards that neighbour or at its own centre (the cursor arriving there).
    """
    if base is None:
        base = frame
    grid = source["grid"]
    rows, cols = len(grid), len(grid[0])
    cell_size, offset_x, offset_y = _maze_layout(frame, source)
    ink = _maze_ink(frame, base, source)

    def centre(cell: Cell) -> tuple[float, float]:
        return offset_x + (cell[1] + 0.5) * cell_size, offset_y + (cell[0] + 0.5) * cell_size

    start = tuple(source["start"])
    goal = tuple(source["goal"])
    occupied: set[Cell] = set()
    radius = max(2, int(cell_size * 0.30))
    for row in range(rows):
        for col in range(cols):
            if (row, col) not in (start, goal) and _disk_fraction(ink, centre((row, col)), radius) >= 0.30:
                occupied.add((row, col))
    for end in (start, goal):
        ex, ey = centre(end)
        for nxt in _neighbors(end, rows, cols):
            if nxt not in occupied:
                continue
            nx, ny = centre(nxt)
            # either the line visibly enters the cell (probe) or the cell itself changed (the cursor arrived or
            # left); a line that stops one cell short of the goal leaves the goal ring untouched
            if (_disk_fraction(ink, (ex + 0.4 * (nx - ex), ey + 0.4 * (ny - ey)), max(1, int(cell_size * 0.08))) >= 0.3
                    or _disk_fraction(ink, (ex, ey), radius) >= 0.3):
                occupied.add(end)
                break
    return occupied


MAZE_MAX_EXTRA_CELLS = 2      # a short stub into a dead end that the line backs out of may stay drawn


def parse_maze_video(frames: np.ndarray, source: dict[str, Any]) -> dict[str, Any]:
    """Maze verifier: the drawing process and the final path.

    Frames from the first EOS frame on are ignored. A cell counts at frame t only if it is detected at t and t + 1
    (blur and flicker filter). Process rules, checked at every frame: nothing is drawn on a wall (line or cursor), and
    the drawn cells stay one connected drawing grown from the start (no separate segments). Entering a dead end and
    erasing it again is allowed. Final rule: the last frame before the EOS frame shows a connected drawing on
    corridors that holds the start and the goal; as the maze is a tree, this drawing contains the unique solution
    path, and at most MAZE_MAX_EXTRA_CELLS cells beyond it are accepted.
    """
    grid = source["grid"]
    rows, cols = len(grid), len(grid[0])
    start = tuple(source["start"])
    goal = tuple(source["goal"])
    eos = find_eos_frame(frames)
    visible = frames[:eos] if eos else frames
    if len(visible) == 0:
        visible = frames[:1]
    base = frames[0]
    raw = [_maze_occupied(frame, source, base) for frame in visible]
    stable = [raw[i] & raw[i + 1] for i in range(len(raw) - 1)] or raw
    walls = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != 0}
    process_violation, violation_frame = None, None
    streak: dict[Cell, int] = {}
    established: set[Cell] = set()
    retracted_frame = None
    for t, cells in enumerate(stable):
        if cells & walls:
            process_violation, violation_frame = "trail_on_wall", t
            break
        # Erasing: a cell drawn for 3 consecutive frames and then missing for 2 consecutive frames (cells that the
        # line is just entering flicker around the coverage threshold for a frame). Recorded, not a violation.
        nxt = stable[t + 1] if t + 1 < len(stable) else cells
        lost = {c for c in established if c not in cells and c not in nxt}
        if lost and retracted_frame is None:
            retracted_frame = t
            established -= lost
        for c in cells - {start, goal}:
            streak[c] = streak.get(c, 0) + 1
            if streak[c] >= 3:
                established.add(c)
        for c in list(streak):
            if c not in cells:
                streak[c] = 0
        # The start marker and the cursor cover the start cell while the first cells are drawn, so the start may be
        # implied during the process; the final check requires it to be entered.
        grown = cells | ({start} if cells else set())
        if grown and not _connected(grown, rows, cols):
            process_violation, violation_frame = "not_connected_to_start", t
            break
    final = raw[-1]
    on_corridor = not (final & walls)
    connected = _connected(final, rows, cols) if final else False
    reference = len(source.get("solution_path", [])) or len(final)
    extra = len(final) - reference
    simple_path = start in final and goal in final and extra <= MAZE_MAX_EXTRA_CELLS
    final_ok = bool(on_corridor and connected and simple_path)
    return {
        "success": bool(final_ok and process_violation is None),
        "final_path_ok": final_ok,
        "process_violation": process_violation,
        "process_violation_frame": violation_frame,
        "retracted_frame": retracted_frame,
        "on_corridor": on_corridor,
        "connected": connected,
        "simple_path": simple_path,
        "parsed_cells": len(final),
        "reference_cells": len(source.get("solution_path", [])),
        "eos_frame": eos,
    }


def parse_flowfree_video(frames: np.ndarray, source: dict[str, Any]) -> dict[str, Any]:
    from .flow_replay import evaluate_flowfree
    return evaluate_flowfree(frames, source,
                             require_adjacent_color_switch=bool(source.get('continuous_global_walk', False)))


def parse_video(frames: np.ndarray, source: dict[str, Any], task: str) -> dict[str, Any]:
    """Strict verdict of a FlowFree, Sokoban or Maze video (frames F x H x W x 3, RGB uint8) of the puzzle ``source``.

    The video is cut at its first EOS frame. ``success`` requires the task rules to hold and, for Sokoban and Maze,
    an EOS frame; FlowFree succeeds as soon as the board is filled without a violation.
    """
    eos_frame = find_eos_frame(frames)
    visible = frames if eos_frame is None else frames[: eos_frame + 1]

    def parse(candidate: np.ndarray) -> dict[str, Any]:
        if task == "flowfree":
            return parse_flowfree_video(candidate, source)
        if task == "sokoban":
            from .sokoban_checker import check_sokoban_video
            return check_sokoban_video(candidate, source, sustain_settle=int(source.get("sustain_settle", 2)))
        if task == "maze":
            return parse_maze_video(candidate, source)
        raise ValueError(f"unsupported task: {task}")

    result = parse(visible)
    task_frame = len(visible) - 1
    if task == "maze" and eos_frame is not None and not result["success"]:
        # the final path may be read from one of the three frames before the EOS frame (the EOS border can hide the
        # last cell of a path that ends at the edge of the canvas)
        for candidate in range(len(visible) - 2, max(-1, len(visible) - 5), -1):
            earlier = parse(visible[: candidate + 1])
            if earlier["success"]:
                result = earlier
                task_frame = candidate
                break
    task_success = bool(result["success"])
    result.update(
        {
            "task_success": task_success,
            "eos_detected": eos_frame is not None,
            "eos_frame": eos_frame,
            "task_frame": result.get("first_success_frame") if task == "flowfree" and task_success else task_frame,
            "visible_frames": len(visible),
            "success": bool(task_success and (task == "flowfree" or eos_frame is not None)),
        }
    )
    return result
