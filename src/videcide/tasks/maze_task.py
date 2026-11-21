"""Maze (VideoRLVR): the project renderer of the expert videos and the symbolic rule check of a path.

A maze is a grid of wall (1) and corridor (0) cells with a start and a goal cell. The expert video draws the solution
path as one continuous anti-aliased stroke from the start to the goal, 1.5 frames per move on average, on a 384 x 384
canvas: four opening frames, the moves, and eight frames holding the solved maze, the last four with the red EOS border.
The colours follow the seven palettes of the VideoRLVR maze images (``render_theme`` of a source), and the prompt names
the colours that are drawn (:func:`maze_prompt`). The video verifier is ``parse_maze_video`` in ``video_parser.py``.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from .video import CANVAS_SIZE, add_eos_border, pad_to_vae_length


Cell = tuple[int, int]
FloatCell = tuple[float, float]

WALL = 1
PATH = 0
REJECT_COLOR = (244, 139, 45)
CURSOR_COLOR = (255, 224, 75)
TRACE_SUPERSAMPLE = 4

DEFAULT_THEME = {
    "bg": (25, 28, 35), "wall": (44, 49, 59), "path": (239, 241, 245), "grid": (205, 210, 218),
    "start": (48, 190, 105), "goal": (224, 62, 68), "ring_fill": (255, 255, 255),
    "solution": (45, 132, 235), "reject": REJECT_COLOR, "cursor": CURSOR_COLOR,
}


def _palette(wall, path, start, goal, trail, agent) -> dict[str, tuple[int, int, int]]:
    grid = tuple(round(0.85 * p + 0.15 * w) for p, w in zip(path, wall))
    # The agent dot keeps its VideoRLVR colour unless it would vanish on the trail.
    cursor = agent if np.linalg.norm(np.subtract(agent, trail)) >= 100 else CURSOR_COLOR
    return {"bg": wall, "wall": wall, "path": path, "grid": grid, "start": start,
            "goal": goal, "ring_fill": path, "solution": trail,
            "reject": REJECT_COLOR, "cursor": cursor}


# The seven palettes of the VideoRLVR maze images (wall, corridor, start block, goal block, agent dot), with a vivid
# trail colour per palette (at least 150 from the corridor and 100 from the wall colour).
THEMES = {
    "default": DEFAULT_THEME,
    "charcoal": _palette((10, 10, 10), (60, 60, 60), (12, 194, 90), (176, 48, 50),
                         (90, 140, 255), (2, 150, 246)),
    "forest": _palette((32, 58, 31), (179, 209, 157), (73, 188, 75), (191, 81, 34),
                       (230, 120, 30), (94, 60, 16)),
    "ice": _palette((7, 29, 79), (148, 197, 228), (54, 205, 192), (214, 103, 93),
                    (20, 70, 200), (247, 203, 18)),
    "mono": _palette((39, 39, 39), (255, 255, 255), (18, 240, 15), (214, 5, 11),
                     (45, 132, 235), (0, 100, 241)),
    "lavender": _palette((179, 158, 197), (248, 237, 245), (148, 214, 151), (229, 149, 155),
                         (120, 60, 200), (144, 183, 228)),
    "midnight": _palette((13, 12, 26), (27, 28, 49), (26, 245, 112), (211, 26, 99),
                         (160, 60, 255), (0, 196, 247)),
    "sand": _palette((78, 57, 38), (217, 200, 157), (114, 172, 76), (194, 60, 57),
                     (40, 90, 200), (49, 103, 145)),
}
VIDEORLVR_THEMES = tuple(name for name in THEMES if name != "default")

# Plain colour names of each theme's (trail, start marker, goal marker), so that the prompt describes what is drawn.
THEME_COLOR_NAMES = {
    "default": ("blue", "green", "red"),
    "charcoal": ("blue", "green", "red"),
    "forest": ("orange", "green", "red"),
    "ice": ("dark blue", "teal", "red"),
    "mono": ("blue", "green", "red"),
    "lavender": ("purple", "green", "pink"),
    "midnight": ("violet", "green", "pink"),
    "sand": ("blue", "green", "red"),
}


def maze_prompt(theme: str = "default") -> str:
    trail, start, goal = THEME_COLOR_NAMES[theme]
    return (f"Solve the maze by drawing one continuous {trail} path from the {start} start "
            f"to the {goal} goal without crossing a wall.")


def neighbors(grid: list[list[int]], cell: Cell) -> Iterable[Cell]:
    rows, cols = len(grid), len(grid[0])
    r, c = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == PATH:
            yield nr, nc


def evaluate_path(grid: list[list[int]], path: list[Cell], start: Cell, goal: Cell) -> dict[str, Any]:
    """Rules of a symbolic path: in bounds, on corridors, unit steps, from the start to the goal."""
    in_bounds = all(0 <= r < len(grid) and 0 <= c < len(grid[0]) for r, c in path)
    on_corridor = in_bounds and all(grid[r][c] == PATH for r, c in path)
    adjacent = all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))
    starts = bool(path) and path[0] == start
    reaches = bool(path) and path[-1] == goal
    return {
        "success": bool(in_bounds and on_corridor and adjacent and starts and reaches),
        "in_bounds": in_bounds,
        "on_corridor": on_corridor,
        "adjacent": adjacent,
        "starts_at_start": starts,
        "reaches_goal": reaches,
        "moves": max(0, len(path) - 1),
    }


def _layout(grid: list[list[int]]) -> tuple[int, int, int]:
    rows, cols = len(grid), len(grid[0])
    margin = 12
    cell = max(4, (CANVAS_SIZE - 2 * margin) // max(rows, cols))
    offset_x = (CANVAS_SIZE - cols * cell) // 2
    offset_y = (CANVAS_SIZE - rows * cell) // 2
    return cell, offset_x, offset_y


def _center(grid: list[list[int]], cell_pos: FloatCell) -> tuple[float, float]:
    cell, ox, oy = _layout(grid)
    r, c = cell_pos
    return ox + (c + 0.5) * cell, oy + (r + 0.5) * cell


def _base_image(grid: list[list[int]], start: Cell, goal: Cell, palette: dict = DEFAULT_THEME) -> Image.Image:
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), palette["bg"])
    draw = ImageDraw.Draw(image)
    cell, ox, oy = _layout(grid)
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            x0, y0 = ox + c * cell, oy + r * cell
            x1, y1 = x0 + cell, y0 + cell
            fill = palette["wall"] if value == WALL else palette["path"]
            draw.rectangle([x0, y0, x1, y1], fill=fill)
            if value == PATH and cell >= 10:
                draw.rectangle([x0, y0, x1, y1], outline=palette["grid"], width=1)
    _draw_endpoint(draw, grid, start, palette["start"], filled=True, palette=palette)
    _draw_endpoint(draw, grid, goal, palette["goal"], filled=False, palette=palette)
    return image


def _draw_endpoint(draw: ImageDraw.ImageDraw, grid: list[list[int]], cell_pos: Cell, color: tuple[int, int, int], *,
                   filled: bool, palette: dict = DEFAULT_THEME) -> None:
    cell, _, _ = _layout(grid)
    x, y = _center(grid, cell_pos)
    radius = max(3, cell * 0.30)
    box = [x - radius, y - radius, x + radius, y + radius]
    if filled:
        draw.ellipse(box, fill=color, outline=palette["ring_fill"], width=max(1, cell // 12))
    else:
        draw.ellipse(box, fill=palette["ring_fill"], outline=color, width=max(2, cell // 7))


def _draw_frame(grid: list[list[int]], start: Cell, goal: Cell, accepted: list[FloatCell],
                rejected: list[FloatCell] | None = None, cursor: FloatCell | None = None, *,
                palette: dict = DEFAULT_THEME) -> np.ndarray:
    image = _base_image(grid, start, goal, palette)
    draw = ImageDraw.Draw(image)
    cell, _, _ = _layout(grid)
    width = max(4, int(cell * 0.5))

    def draw_trace(points: list[FloatCell], color: tuple[int, int, int]) -> None:
        # Anti-aliased stroke: a coverage mask rasterised at TRACE_SUPERSAMPLE x and box-filtered down.
        if not points:
            return
        scale = TRACE_SUPERSAMPLE
        mask = Image.new("L", (image.width * scale, image.height * scale), 0)
        pen = ImageDraw.Draw(mask)
        pixels = [(x * scale, y * scale) for x, y in (_center(grid, point) for point in points)]
        stroke = width * scale
        if len(pixels) >= 2:
            pen.line(pixels, fill=255, width=stroke, joint="curve")
        for x, y in (pixels[0], pixels[-1]):
            pen.ellipse([x - stroke / 2, y - stroke / 2, x + stroke / 2, y + stroke / 2], fill=255)
        image.paste(color, (0, 0, image.width, image.height), mask.resize(image.size, Image.BOX))

    draw_trace(accepted, palette["solution"])
    if rejected:
        draw_trace(rejected, palette["reject"])
    _draw_endpoint(draw, grid, start, palette["start"], filled=True, palette=palette)
    _draw_endpoint(draw, grid, goal, palette["goal"], filled=False, palette=palette)
    if cursor is not None:
        x, y = _center(grid, cursor)
        radius = max(3, cell * 0.22)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=palette["cursor"], outline=(25, 25, 25),
                     width=max(1, cell // 12))
    return np.asarray(image)


def _interpolate(a: Cell, b: Cell, t: float) -> FloatCell:
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def _motion_steps(move_index: int, average_frames_per_move: float) -> int:
    before = math.floor(move_index * average_frames_per_move)
    after = math.floor((move_index + 1) * average_frames_per_move)
    return max(1, after - before)


def render_optimal(grid: list[list[int]], path: list[Cell], start: Cell, goal: Cell, *, theme: str = "default",
                   frames_per_move: float = 1.5) -> np.ndarray:
    """Expert video (F x 384 x 384 x 3, uint8) that draws ``path`` from ``start`` to ``goal``.

    The released videos are ``render_optimal(source["grid"], source["solution_path"], source["start"], source["goal"],
    theme=source["render_theme"])``, written at 16 fps with ``video.write_mp4``.
    """
    palette = THEMES[theme]
    path = [tuple(cell) for cell in path]
    start, goal = tuple(start), tuple(goal)
    result = evaluate_path(grid, path, start, goal)
    if not result["success"]:
        raise ValueError(f"invalid maze path: {result}")
    frames: list[np.ndarray] = []
    accepted: list[FloatCell] = [path[0]]
    for _ in range(4):
        frames.append(_draw_frame(grid, start, goal, accepted, cursor=path[0], palette=palette))
    for index, (a, b) in enumerate(zip(path, path[1:])):
        move_frames = _motion_steps(index, frames_per_move)
        for step in range(1, move_frames + 1):
            tip = _interpolate(a, b, step / move_frames)
            frames.append(_draw_frame(grid, start, goal, [*accepted, tip], cursor=tip, palette=palette))
        accepted.append(b)
    for _ in range(8):
        frames.append(_draw_frame(grid, start, goal, accepted, cursor=goal, palette=palette))
    return add_eos_border(pad_to_vae_length(np.stack(frames)))
