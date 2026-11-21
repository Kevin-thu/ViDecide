from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path
from types import ModuleType
from typing import Any
from itertools import permutations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .video import CANVAS_SIZE, add_eos_border, pad_to_vae_length, read_last_bgr


Cell = tuple[int, int]


FloatCell = tuple[float, float]


BG = (24, 26, 33)


CELL_BG = (38, 42, 52)


GRID = (74, 80, 94)


ENDPOINT_STROKE = (245, 245, 245)


CURSOR = (255, 255, 255)


CURSOR_RADIUS_FRACTION = 0.28


FAILURE_MARK_COLOR = (235, 40, 48)


PALETTE = [
    (237, 58, 68),
    (53, 132, 238),
    (55, 190, 103),
    (246, 194, 65),
    (166, 88, 228),
    (33, 190, 202),
    (247, 126, 54),
    (235, 86, 164),
]


def load_upstream_evaluator(code_root: Path) -> ModuleType:
    evaluator_path = code_root / "src" / "eval" / "evaluate_flowfree.py"
    spec = importlib.util.spec_from_file_location(
        "videorlvr_evaluate_flowfree",
        evaluator_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_grid_size(prompt: str) -> int:
    match = re.search(r"(\d+)[×x](\d+)", prompt)
    if not match or match.group(1) != match.group(2):
        raise ValueError(f"cannot parse square FlowFree size from prompt: {prompt}")
    size = int(match.group(1))
    if size not in (5, 6, 7, 8):
        raise ValueError(f"unsupported FlowFree size: {size}")
    return size


def extract_source(
    input_image: Path,
    solution_video: Path,
    prompt: str,
    evaluator: ModuleType,
) -> dict[str, Any]:
    size = parse_grid_size(prompt)
    first_bgr = cv2.imread(str(input_image))
    if first_bgr is None:
        raise ValueError(f"cannot read FlowFree input image: {input_image}")
    last_bgr = read_last_bgr(solution_video)
    upstream = evaluator.compute_flowfree_success([first_bgr, last_bgr])
    if upstream["success"] != 1:
        raise ValueError(f"upstream GT fails its own strict evaluator: {upstream}")

    endpoints_raw = evaluator._detect_endpoints(first_bgr, size)
    endpoint_positions = {(int(r), int(c)) for r, c, _ in endpoints_raw}
    cell_colors = evaluator._get_cell_colors(last_bgr, size)
    groups = evaluator._group_cells_by_color(cell_colors, 60.0)

    endpoint_to_group: dict[Cell, int] = {}
    for r, c, _ in endpoints_raw:
        cell = int(r), int(c)
        for group_id, (_, group_cells) in groups.items():
            if cell in group_cells:
                endpoint_to_group[cell] = int(group_id)
                break

    extracted: list[dict[str, Any]] = []
    for group_id, (representative_bgr, raw_cells) in groups.items():
        cells = {(int(r), int(c)) for r, c in raw_cells}
        own_endpoints = {
            cell for cell, mapped_group in endpoint_to_group.items()
            if mapped_group == group_id
        }
        if not own_endpoints:
            raise ValueError(f"color group {group_id} has no detected endpoint")

        if len(cells) == 1:
            path = [next(iter(cells))]
        else:
            other_endpoints = endpoint_positions - own_endpoints
            valid_ends: set[Cell] = set()
            for r, c in cells:
                if (r, c) in own_endpoints:
                    valid_ends.add((r, c))
                if any(
                    (r + dr, c + dc) in other_endpoints
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                ):
                    valid_ends.add((r, c))

            path = []
            for start in sorted(own_endpoints):
                candidate = evaluator._trace_path(
                    start,
                    cells,
                    valid_ends=valid_ends - {start},
                    max_backtracks=100_000,
                )
                if (
                    len(candidate) == len(cells)
                    and candidate[-1] in valid_ends - {start}
                ):
                    path = [(int(r), int(c)) for r, c in candidate]
                    break
            if not path:
                raise ValueError(
                    f"cannot reconstruct group {group_id}: "
                    f"cells={len(cells)} endpoints={sorted(own_endpoints)}"
                )

        b, g, r = [int(round(value)) for value in representative_bgr]
        extracted.append(
            {
                "path": [list(cell) for cell in path],
                "endpoints": [list(cell) for cell in sorted(own_endpoints)],
                "source_rgb": [r, g, b],
            }
        )

    extracted.sort(
        key=lambda group: (
            group["endpoints"][0] if group["endpoints"] else [99, 99],
            group["path"][0],
        )
    )
    source = {
        "grid_size": size,
        "paths": [group["path"] for group in extracted],
        "endpoints": [group["endpoints"] for group in extracted],
        "source_colors_rgb": [group["source_rgb"] for group in extracted],
        "upstream_metrics": upstream,
    }
    result = evaluate_symbolic(source)
    if not result["success"]:
        raise ValueError(f"reconstructed FlowFree source is invalid: {result}")
    return source


def evaluate_symbolic(source: dict[str, Any]) -> dict[str, Any]:
    size = int(source["grid_size"])
    paths = [
        [tuple(int(value) for value in cell) for cell in path]
        for path in source["paths"]
    ]
    endpoints = [
        {tuple(int(value) for value in cell) for cell in group}
        for group in source["endpoints"]
    ]
    in_bounds = all(
        0 <= r < size and 0 <= c < size
        for path in paths
        for r, c in path
    )
    adjacent = all(
        abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        for path in paths
        for a, b in zip(path, path[1:])
    )
    flat = [cell for path in paths for cell in path]
    no_overlap = len(flat) == len(set(flat))
    full_cover = len(set(flat)) == size * size
    starts_at_endpoint = all(
        bool(path) and path[0] in own
        for path, own in zip(paths, endpoints)
    )
    all_endpoints_visited = all(
        own <= set(path)
        for path, own in zip(paths, endpoints)
    )
    valid_ends = True
    endpoint_union = set().union(*endpoints) if endpoints else set()
    for path, own in zip(paths, endpoints):
        if len(path) <= 1:
            continue
        other = endpoint_union - own
        end = path[-1]
        valid_ends &= (
            end in own
            or any(
                (end[0] + dr, end[1] + dc) in other
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
        )
    success = all(
        (
            in_bounds,
            adjacent,
            no_overlap,
            full_cover,
            starts_at_endpoint,
            all_endpoints_visited,
            valid_ends,
        )
    )
    return {
        "success": success,
        "in_bounds": in_bounds,
        "adjacent": adjacent,
        "no_overlap": no_overlap,
        "full_cover": full_cover,
        "starts_at_endpoint": starts_at_endpoint,
        "all_endpoints_visited": all_endpoints_visited,
        "valid_ends": valid_ends,
        "cells": len(set(flat)),
        "num_paths": len(paths),
    }


def _palette_of(source: dict[str, Any]) -> list:
    return [tuple(c) for c in source["render_palette"]] if source.get("render_palette") else PALETTE


def _cell_center(size: int, cell: FloatCell) -> tuple[float, float]:
    margin = 28
    board = CANVAS_SIZE - 2 * margin
    width = board / size
    r, c = cell
    return margin + (c + 0.5) * width, margin + (r + 0.5) * width


def _cell_box(size: int, cell: Cell, inset: float = 2.0) -> list[float]:
    margin = 28
    board = CANVAS_SIZE - 2 * margin
    width = board / size
    r, c = cell
    return [
        margin + c * width + inset,
        margin + r * width + inset,
        margin + (c + 1) * width - inset,
        margin + (r + 1) * width - inset,
    ]


def _draw_frame(
    source: dict[str, Any],
    drawn_paths: list[list[Cell]],
    cursor: FloatCell | None = None,
    cursor_color_index: int | None = None,
    blocked_cell: Cell | None = None,
) -> np.ndarray:
    size = int(source["grid_size"])
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BG)
    draw = ImageDraw.Draw(image)
    margin = 28
    board = CANVAS_SIZE - 2 * margin
    cell = board / size

    for r in range(size):
        for c in range(size):
            x0 = margin + c * cell
            y0 = margin + r * cell
            draw.rounded_rectangle(
                [x0 + 2, y0 + 2, x0 + cell - 2, y0 + cell - 2],
                radius=max(3, int(cell * 0.08)),
                fill=CELL_BG,
                outline=GRID,
                width=1,
            )

    for color_index, path in enumerate(drawn_paths):
        color = _palette_of(source)[color_index % len(_palette_of(source))]
        for filled_cell in path:
            draw.rounded_rectangle(
                _cell_box(size, filled_cell),
                radius=max(3, int(cell * 0.08)),
                fill=color,
                outline=GRID,
                width=1,
            )

    endpoint_radius = max(8, int(cell * 0.31))
    for color_index, group in enumerate(source["endpoints"]):
        color = _palette_of(source)[color_index % len(_palette_of(source))]
        for raw_cell in group:
            point = int(raw_cell[0]), int(raw_cell[1])
            x, y = _cell_center(size, point)
            draw.ellipse(
                [
                    x - endpoint_radius,
                    y - endpoint_radius,
                    x + endpoint_radius,
                    y + endpoint_radius,
                ],
                fill=color,
                outline=ENDPOINT_STROKE,
                width=max(2, int(cell * 0.07)),
            )

    if cursor is not None:
        x, y = _cell_center(size, cursor)
        radius = max(9, int(cell * CURSOR_RADIUS_FRACTION))
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=CURSOR,
            outline=(25, 25, 25),
            width=max(2, int(cell * 0.06)),
        )
        core_radius = max(4, int(cell * 0.12))
        core_color = (
            _palette_of(source)[cursor_color_index % len(_palette_of(source))]
            if cursor_color_index is not None
            else (35, 38, 46)
        )
        draw.ellipse(
            [
                x - core_radius,
                y - core_radius,
                x + core_radius,
                y + core_radius,
            ],
            fill=core_color,
            outline=(25, 25, 25),
            width=max(1, int(cell * 0.03)),
        )

    if blocked_cell is not None:
        x0, y0, x1, y1 = _cell_box(size, blocked_cell, inset=5.0)
        stroke = max(4, int(cell * 0.09))
        under_stroke = stroke + max(3, int(cell * 0.06))
        pad = max(6, int(cell * 0.17))
        draw.rounded_rectangle(
            [x0, y0, x1, y1],
            radius=max(3, int(cell * 0.08)),
            outline=(255, 255, 255),
            width=under_stroke,
        )
        draw.rounded_rectangle(
            [x0, y0, x1, y1],
            radius=max(3, int(cell * 0.08)),
            outline=FAILURE_MARK_COLOR,
            width=stroke,
        )
        for start, end in (
            ((x0 + pad, y0 + pad), (x1 - pad, y1 - pad)),
            ((x1 - pad, y0 + pad), (x0 + pad, y1 - pad)),
        ):
            draw.line([start, end], fill=(255, 255, 255), width=under_stroke)
            draw.line([start, end], fill=FAILURE_MARK_COLOR, width=stroke)
    return np.asarray(image)


def _interpolate(a: Cell, b: Cell, t: float) -> FloatCell:
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def _motion_steps(move_index: int, average_frames_per_move: float) -> int:
    if average_frames_per_move < 1.0:
        raise ValueError("average_frames_per_move must be at least 1")
    before = math.floor(move_index * average_frames_per_move)
    after = math.floor((move_index + 1) * average_frames_per_move)
    return after - before


def render_optimal(
    source: dict[str, Any],
    *,
    frames_per_move: float = 2.5,
    transition_mode: str = "continuous",
) -> np.ndarray:
    if transition_mode not in {"continuous", "discrete"}:
        raise ValueError(f"unsupported transition_mode: {transition_mode}")
    result = evaluate_symbolic(source)
    if not result["success"]:
        raise ValueError(f"invalid FlowFree source: {result}")
    paths = [
        [tuple(int(value) for value in cell) for cell in path]
        for path in source["paths"]
    ]
    drawn: list[list[Cell]] = [[] for _ in paths]
    frames: list[np.ndarray] = []
    for _ in range(4):
        frames.append(_draw_frame(source, drawn))
    move_index = 0
    for color_index, path in enumerate(paths):
        drawn[color_index] = [path[0]]
        frames.append(
            _draw_frame(
                source,
                drawn,
                cursor=path[0],
                cursor_color_index=color_index,
            )
        )
        prefix: list[Cell] = [path[0]]
        for a, b in zip(path, path[1:]):
            move_frames = _motion_steps(move_index, frames_per_move)
            move_index += 1
            for step in range(1, move_frames + 1):
                tip = (
                    _interpolate(a, b, step / move_frames)
                    if transition_mode == "continuous"
                    else (b if step == move_frames else a)
                )
                drawn[color_index] = [*prefix, *([b] if step == move_frames else [])]
                frames.append(
                    _draw_frame(
                        source,
                        drawn,
                        cursor=tip,
                        cursor_color_index=color_index,
                    )
                )
            prefix.append(b)
            drawn[color_index] = list(prefix)
        frames.append(_draw_frame(source, drawn))
    for _ in range(8):
        frames.append(_draw_frame(source, drawn))
    return add_eos_border(pad_to_vae_length(np.stack(frames)))


def grid_neighbors(cell: Cell, size: int) -> list[Cell]:
    r, c = cell
    return [
        (nr, nc)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        if 0 <= nr < size and 0 <= nc < size
    ]


def recover_order(video: str | Path, source: dict) -> dict:
    n = int(source['grid_size'])
    reader = cv2.VideoCapture(str(video))
    ok, first = reader.read()
    if not ok:
        raise ValueError(f'Unreadable original video: {video}')
    expected = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    reader.set(cv2.CAP_PROP_POS_FRAMES, expected - 1)
    ok, final = reader.read()
    if not ok:
        raise ValueError('Unreadable original final frame')
    h, w = first.shape[:2]
    yy = np.array([[(r + v) * h / n for c in range(n) for v in (.2, .2, .8, .8)]
                   for r in range(n)]).reshape(n, n, 4).astype(int)
    xx = np.array([[(c + v) * w / n for c in range(n) for v in (.2, .8, .2, .8)]
                   for r in range(n)]).reshape(n, n, 4).astype(int)
    before = first[yy, xx].astype(np.float32)
    delta = final[yy, xx].astype(np.float32) - before
    contrast = np.linalg.norm(delta, axis=-1).mean(-1)
    if contrast.min() < 30:
        raise ValueError('Unidentifiable original cell fill; do not infer its trajectory')
    reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
    observations = []
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        current = frame[yy, xx].astype(np.float32) - before
        fraction = (current * delta).sum(-1) / np.maximum((delta * delta).sum(-1), 1)
        observations.append(np.median(fraction, axis=-1) >= .65)
    reader.release()
    boards = np.stack(observations)
    if len(boards) != expected or boards[0].any() or not boards[-1].all():
        raise ValueError('Original decoded coverage or first frame is inconsistent')
    if (boards[:-1] & ~boards[1:]).any():
        raise ValueError('Original cell fill is non-monotonic or visually ambiguous')
    times = boards.argmax(axis=0)
    endpoints, dot_refs = [], []
    split_groups = []
    for old_color, group in enumerate(source['endpoints']):
        parts = []
        for raw_cell in group:
            cell = tuple(raw_cell)
            r, c = cell
            dot = first[int((r + .5) * h / n), int((c + .5) * w / n)].astype(float)
            part = next((p for p in parts if np.linalg.norm(dot - p[0]) < 25), None)
            if part is None:
                parts.append((dot, [cell]))
            else:
                part[1].append(cell)
        parts.sort(key=lambda p: (-len(p[1]), min(p[1])))
        dot_refs.append(parts[0][0])
        endpoints.append(set(parts[0][1]))
        split_groups.extend(parts[1:])
    for dot, cells in split_groups:
        dot_refs.append(dot)
        endpoints.append(set(cells))
    unique_refs, unique_endpoints = [], []
    for dot, cells in zip(dot_refs, endpoints):
        existing = next((i for i, ref in enumerate(unique_refs) if np.linalg.norm(dot - ref) < 25), None)
        if existing is None:
            unique_refs.append(dot)
            unique_endpoints.append(set(cells))
        else:
            unique_endpoints[existing].update(cells)
    merged_duplicates = len(endpoints) - len(unique_endpoints)
    dot_refs, endpoints = unique_refs, unique_endpoints
    if len(endpoints) > 8:
        raise ValueError('Observed palette exceeds the project renderer palette')
    cy = np.array([[(r + v) * h / n for c in range(n) for v in (.35, .35, .65, .65)]
                   for r in range(n)]).reshape(n, n, 4).astype(int)
    cx = np.array([[(c + v) * w / n for c in range(n) for v in (.35, .65, .35, .65)]
                   for r in range(n)]).reshape(n, n, 4).astype(int)
    def chroma(rgb):
        rgb = rgb - rgb.min(axis=-1, keepdims=True)
        return rgb / np.maximum(np.linalg.norm(rgb, axis=-1, keepdims=True), 1)
    fill = chroma(np.median(final[cy, cx].astype(float), axis=-2))
    refs = chroma(np.stack(dot_refs))
    distances = np.linalg.norm(fill[:, :, None, :] - refs[None, None, :, :], axis=-1)
    labels = distances.argmin(axis=-1)
    ranked = np.sort(distances, axis=-1)
    if ranked[..., 0].max() > .18 or (ranked[..., 1] - ranked[..., 0]).min() < .06:
        raise ValueError('Original filled-cell colour is ambiguous relative to its endpoint palette')
    candidates = [()]
    for time in sorted(set(times.ravel())):
        group = [tuple(map(int, p)) for p in np.argwhere(times == time)]
        if len(group) > 7:
            raise ValueError('Too many simultaneous first appearances to recover an observed order')
        next_candidates = []
        for previous in candidates:
            for ordering in permutations(group):
                extended = previous + ordering
                if not previous and ordering[0] not in endpoints[int(labels[ordering[0]])]:
                    continue
                tail = (previous[-1:] + ordering) if previous else ordering
                if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in zip(tail, tail[1:])):
                    continue
                next_candidates.append(extended)
        candidates = list(dict.fromkeys(next_candidates))
        if not candidates or len(candidates) > 1024:
            raise ValueError('Original frame order does not identify a continuous walk')
    if len(candidates) != 1:
        raise ValueError('Several walks fit original frames; review instead of choosing one silently')
    order = candidates[0]
    visited, used, active = set(), set(), None
    for cell in order:
        color = int(labels[cell])
        if color != active:
            if color in used or cell not in endpoints[color] or (
                active is not None and not endpoints[active] <= visited
            ):
                raise ValueError('Original color switching does not satisfy its endpoint rule')
            used.add(color)
            active = color
        visited.add(cell)
    if not all(group <= visited for group in endpoints):
        raise ValueError('An original endpoint was never visited')
    corrected = dict(source)
    corrected['endpoints'] = [[list(c) for c in sorted(group)] for group in endpoints]
    corrected['paths'] = [[list(c) for c in order if labels[c] == color] for color in range(len(endpoints))]
    corrected['source_colors_rgb'] = [list(map(int, dot[::-1])) for dot in dot_refs]
    return dict(cells=[list(c) for c in order], colors=[int(labels[c]) for c in order],
                first_fill_frames=[int(times[c]) for c in order], original_frames=expected,
                unique_observed_order=True, minimum_cell_contrast=float(contrast.min()),
                corrected_source=corrected, split_merged_endpoint_groups=len(split_groups),
                merged_duplicate_endpoint_groups=merged_duplicates,
                repaired_endpoint_palette=corrected['endpoints'] != source['endpoints'],
                corrected_cell_labels=sum(int(labels[tuple(c)] != color)
                                          for color, path in enumerate(source['paths']) for c in path))


def render_order(source: dict, order: dict, target_frames: int) -> np.ndarray:
    source = order.get('corrected_source', source)
    cells = [tuple(c) for c in order['cells']]
    colors = order['colors']
    n = int(source['grid_size'])
    if len(cells) != n * n or len(set(cells)) != len(cells):
        raise ValueError('Order must visit every cell exactly once')
    if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in zip(cells, cells[1:])):
        raise ValueError('Global walk is not four-adjacent')
    drawn = [[] for _ in source['paths']]
    frames = [_draw_frame(source, drawn) for _ in range(4)]
    drawn[colors[0]].append(cells[0])
    frames.append(_draw_frame(source, drawn, cursor=cells[0], cursor_color_index=colors[0]))
    move_budget = int(target_frames) - 4 - 1 - 8
    if move_budget < len(cells) - 1 or target_frames % 4 != 1:
        raise ValueError('Original project frame budget is insufficient or not VAE aligned')
    for i, (a, b) in enumerate(zip(cells, cells[1:])):
        count = ((i + 1) * move_budget // (len(cells) - 1)
                 - i * move_budget // (len(cells) - 1))
        for step in range(1, count + 1):
            if step == count:
                drawn[colors[i + 1]].append(b)
            frames.append(_draw_frame(source, drawn, cursor=_interpolate(a, b, step / count),
                                     cursor_color_index=colors[i + 1] if step == count else colors[i]))
    frames.extend(_draw_frame(source, drawn) for _ in range(8))
    if len(frames) != target_frames:
        raise AssertionError('Matched frame count changed')
    return add_eos_border(np.stack(frames))
