from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from .flowfree_task import PALETTE, grid_neighbors

VERSION = "flowfree_strict_replay"
Cell = tuple[int, int]


@lru_cache(maxsize=32)
def _fill_samples(canvas: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    width = (canvas - 56) / size
    ys, xs = [], []
    for row in range(size):
        for col in range(size):
            cy, cx = 28 + (row + .5) * width, 28 + (col + .5) * width
            yy, xx = [], []
            for dy in (-.34, .34):
                for dx in (-.34, .34):
                    for py in (-.025, 0, .025):
                        for px in (-.025, 0, .025):
                            yy.append(round(cy + (dy + py) * width))
                            xx.append(round(cx + (dx + px) * width))
            ys.append(yy); xs.append(xx)
    return np.asarray(ys), np.asarray(xs)


def _filled_boards_and_scores(frames: np.ndarray, size: int,
                              colors: int, palette=None) -> tuple[np.ndarray, np.ndarray]:
    if frames.ndim != 4 or frames.shape[1] != frames.shape[2]:
        raise ValueError("Expected square RGB video")
    ys, xs = _fill_samples(frames.shape[1], size)
    pixels = frames[:, ys, xs].astype(np.float32)
    palette = np.asarray(palette[:colors] if palette is not None else PALETTE[:colors], np.float32)
    palette -= palette.min(-1, keepdims=True)
    palette /= palette.sum(-1, keepdims=True)
    chroma = pixels - pixels.min(-1, keepdims=True)
    chroma /= chroma.sum(-1, keepdims=True).clip(min=1)
    distance = np.stack([((chroma - color) ** 2).sum(-1) for color in palette], axis=-1)
    nearest = distance.argmin(-1)
    brightest = pixels.max(-1)
    saturation = (brightest - pixels.min(-1)) / brightest.clip(min=1)
    colored = (saturation >= .4) & (brightest >= 120)
    scores = np.stack([((nearest == c) & colored).mean(-1) for c in range(colors)], axis=-1)
    best = scores.argmax(-1)
    labels = np.where(scores.max(-1) >= .5, best, -1)
    return (labels.astype(np.int8).reshape(len(frames), size, size),
            scores.reshape(len(frames), size, size, colors))


def preserve_cursor_occluded_paint(boards: np.ndarray, frames: np.ndarray,
                                   size: int, scores: np.ndarray | None = None) -> int:
    from .video_parser import _color_fraction, _flow_center

    corrected = 0
    radius = max(3, int(((frames.shape[1] - 56) / size) * .27))
    for index in range(1, len(boards)):
        missing = np.argwhere((boards[index - 1] >= 0) & (boards[index] < 0))
        for row, col in missing:
            previous_color = int(boards[index - 1, row, col])
            center = _flow_center(size, (int(row), int(col)), frames.shape[1])
            cursor = _color_fraction(
                frames[index], center, (255, 255, 255), radius=radius, threshold=65.0
            ) >= .05
            weak_same_color = (
                scores is not None and scores[index, row, col, previous_color] >= .20
            )
            if cursor or weak_same_color:
                boards[index, row, col] = previous_color
                corrected += 1
    return corrected


def _adjacent(a: Cell, b: Cell) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _connected(cells: set[Cell]) -> bool:
    if not cells:
        return False
    reached = {next(iter(cells))}
    stack = list(reached)
    while stack:
        at = stack.pop()
        new = {cell for cell in cells - reached if _adjacent(at, cell)}
        reached.update(new); stack.extend(new)
    return reached == cells


def necessary_connectivity_deadlock(
    board: np.ndarray,
    source: dict[str, Any],
    *,
    active_tips: list[list[int]] | tuple[tuple[int, int, int], ...],
    color_runs: list[int] | tuple[int, ...],
) -> dict[str, Any] | None:
    size = int(source["grid_size"])
    if board.shape != (size, size):
        raise ValueError(f"expected {(size, size)} board, got {board.shape}")
    groups = [[tuple(map(int, cell)) for cell in group] for group in source["endpoints"]]
    all_endpoints = {cell for group in groups for cell in group}
    runs = [int(color) for color in color_runs]
    used = set(runs)
    active = runs[-1] if runs else None
    tips_by_color: dict[int, list[Cell]] = {}
    for color, row, col in active_tips:
        tips_by_color.setdefault(int(color), []).append((int(row), int(col)))

    empty = {tuple(map(int, cell)) for cell in np.argwhere(board < 0)}

    def reaches_all(start: Cell, targets: set[Cell], own_endpoints: set[Cell]) -> bool:
        allowed = (empty - (all_endpoints - own_endpoints)) | {start} | targets
        reached, stack = {start}, [start]
        while stack:
            cell = stack.pop()
            for neighbor in grid_neighbors(cell, size):
                if neighbor in allowed and neighbor not in reached:
                    reached.add(neighbor)
                    stack.append(neighbor)
        return targets <= reached

    for color, endpoints in enumerate(groups):
        if len(endpoints) < 2:
            continue
        if color not in used:
            targets = set(endpoints)
            if not reaches_all(endpoints[0], targets, set(endpoints)):
                return {"reason": "unstarted_endpoints_disconnected", "color": color,
                        "endpoints": [list(cell) for cell in endpoints]}
            continue
        if color != active:
            continue
        remaining = {cell for cell in endpoints if int(board[cell]) != color}
        if not remaining:
            continue
        tips = tips_by_color.get(color, [])
        if tips and any(reaches_all(tip, remaining, set(endpoints)) for tip in tips):
            continue
        return {"reason": "active_tip_disconnected_from_remaining_endpoint", "color": color,
                "tips": [list(cell) for cell in tips],
                "remaining_endpoints": [list(cell) for cell in sorted(remaining)]}
    return None


def replay_boards(boards: np.ndarray, source: dict[str, Any], *,
                  eos_frame: int | None = None, include_trace: bool = False,
                  require_adjacent_color_switch: bool = False) -> dict:
    size = int(source["grid_size"])
    groups = [{tuple(cell) for cell in group} for group in source["endpoints"]]
    endpoints = {cell: color for color, group in enumerate(groups) for cell in group}
    boards = boards[:eos_frame + 1] if eos_frame is not None else boards
    if not len(boards):
        raise ValueError("Empty video")
    states = {(None, None, frozenset())}
    previous = np.full((size, size), -1, dtype=np.int8)
    trace, runs = [], []
    failure_frame = success_frame = None
    failure_reason = None
    failure_cells = []
    max_added = 0
    monotonic = True
    for frame_index, board in enumerate(boards):
        removed = [tuple(map(int, cell)) for cell in np.argwhere((previous >= 0) & (board != previous))]
        new = [tuple(map(int, cell)) for cell in np.argwhere((previous < 0) & (board >= 0))]
        wrong_endpoints = [cell for cell, color in endpoints.items() if board[cell] >= 0 and board[cell] != color]
        unexpected_colors = [tuple(map(int, cell)) for cell in np.argwhere(board >= len(groups))]
        max_added = max(max_added, len(new))
        reason = None
        implicated = new
        if frame_index == 0 and new:
            reason = "initial_paint_without_trajectory"
        elif removed:
            reason = "paint_removed_or_recolored"; implicated = removed; monotonic = False
        elif unexpected_colors:
            reason = "color_not_in_puzzle"; implicated = unexpected_colors
        elif wrong_endpoints:
            reason = "endpoint_wrong_color"; implicated = wrong_endpoints
        elif new:
            cells = tuple(new)
            full_mask = (1 << len(cells)) - 1
            old_visited = {tuple(map(int, c)) for c in np.argwhere(previous >= 0)}

            @lru_cache(maxsize=None)
            def extend(mask, color, tip, used):
                if mask == full_mask:
                    return frozenset({(color, tip, used)})
                visited = old_visited | {cell for j, cell in enumerate(cells) if mask & (1 << j)}
                result = set()
                for j, cell in enumerate(cells):
                    if mask & (1 << j):
                        continue
                    next_color = int(board[cell])
                    if next_color == color:
                        allowed = _adjacent(tip, cell)
                    else:
                        allowed = (next_color not in used and cell in groups[next_color]
                                   and (color is None or groups[color] <= visited)
                                   and (not require_adjacent_color_switch or color is None or _adjacent(tip, cell)))
                    if allowed:
                        result.update(extend(mask | (1 << j), next_color, cell, used | {next_color}))
                return frozenset(result)

            next_states = set()
            for color, tip, used in states:
                next_states.update(extend(0, color, tip, used))
            if not next_states:
                new_colors = {int(board[cell]) for cell in new}
                reason = "non_continuous_single_tip"
                if len(new_colors) == 1:
                    next_color = next(iter(new_colors))
                    if all(color != next_color for color, _, _ in states):
                        if all(next_color in used for _, _, used in states):
                            reason = "return_to_completed_color"
                        elif all(color is not None and not groups[color] <= old_visited
                                 for color, _, _ in states):
                            reason = "switch_before_visiting_all_endpoints"
                        elif not any(cell in groups[next_color] for cell in new):
                            reason = "new_color_does_not_start_at_endpoint"
            else:
                states = next_states
                for cell in new:
                    color = int(board[cell])
                    if not runs or color != runs[-1]:
                        runs.append(color)
        if reason is not None:
            failure_frame, failure_reason, failure_cells = frame_index, reason, implicated
        solved = reason is None and bool((board >= 0).all())
        if include_trace:
            trace.append({"frame": frame_index, "painted_cells": int((board >= 0).sum()),
                          "added": [[*cell, int(board[cell])] for cell in new],
                          "local_violation": reason is not None, "reason": reason,
                          "active_tips": [[color, *tip] for color, tip, _ in sorted(states, key=str)
                                          if tip is not None]})
        if solved:
            success_frame = frame_index
        if reason is not None or solved:
            break
        previous = board

    success = success_frame is not None
    terminal_frame = (success_frame if success else failure_frame if failure_frame is not None
                      else len(boards) - 1)
    final = boards[success_frame] if success else boards[-1]
    full_cover = bool((final >= 0).all())
    endpoint_colors = all(final[cell] == color for cell, color in endpoints.items())
    color_valid = [_connected({tuple(map(int, c)) for c in np.argwhere(final == color)})
                   and all(final[cell] == color for cell in group)
                   for color, group in enumerate(groups)]
    version = VERSION + '_adjacent_switch' if require_adjacent_color_switch else VERSION
    result = {"evaluator_version": version, "success": success, "task_success": success,
              "outcome": "success" if success else "rule_violation" if failure_frame is not None else "incomplete",
              "first_success_frame": success_frame, "first_failure_frame": failure_frame,
              "failure_reason": failure_reason, "failure_cells": [list(c) for c in failure_cells],
              "terminal_frame": terminal_frame, "terminal_latent_index": (terminal_frame + 3) // 4,
              "eos_frame": eos_frame, "eos_detected": eos_frame is not None,
              "full_cover": full_cover, "endpoint_colors": bool(endpoint_colors), "color_valid": color_valid,
              "final_board_success": bool(full_cover and endpoint_colors and all(color_valid)),
              "parsed_cells": int((final >= 0).sum()), "colors": len(groups),
              "painted_cells": sum(final[r, c] >= 0 for r in range(size) for c in range(size)
                                   if (r, c) not in endpoints),
              "target_painted_cells": size * size - len(endpoints),
              "temporal_legal": failure_frame is None, "monotonic_coloring": monotonic,
              "single_agent_path": failure_frame is None, "continuous_single_tip": failure_frame is None,
              "color_runs": runs, "max_cells_added_per_frame": max_added,
              "temporal_failures": [failure_reason] if failure_reason else [],
              "cursor_occluded_label_frames_filled": 0}
    result["painted_cells"] = int(result["painted_cells"])
    if include_trace:
        result["trace"] = trace
    return result


def evaluate_flowfree(frames: np.ndarray, source: dict[str, Any], *, include_trace: bool = False,
                      require_adjacent_color_switch: bool = False) -> dict:
    from .video_parser import find_eos_frame
    eos = find_eos_frame(frames)
    visible = frames[:eos + 1] if eos is not None else frames
    boards, scores = _filled_boards_and_scores(
        visible, int(source["grid_size"]), len(source["endpoints"]), source.get("render_palette")
    )
    corrected = preserve_cursor_occluded_paint(
        boards, visible, int(source["grid_size"]), scores
    )
    result = replay_boards(boards, source, eos_frame=eos, include_trace=include_trace,
                           require_adjacent_color_switch=require_adjacent_color_switch)
    result["cursor_occluded_label_frames_filled"] = corrected
    return result
