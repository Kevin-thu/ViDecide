"""Frames of the DiT-B videos: 256 x 256 pixels, one frame per move, and one EOS frame with a red border.

FlowFree videos fill one cell per frame along the drawing order of the puzzle's trace (``observed_trace_path``);
Sokoban videos show one state per move.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from videcide.paths import data_path
from videcide.tasks import flowfree_task, sokoban_task
from videcide.tasks.video import add_eos_border

SIZE = 256


def _small(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def _eos(frames: list[np.ndarray]) -> np.ndarray:
    last = add_eos_border(frames[-1][None].copy(), duration=1, width=7)[0]
    return np.stack(frames + [last])


def sokoban(source: dict) -> np.ndarray:
    meta = {**source, "render_theme": "classic"}
    states, _, result = sokoban_task.replay(meta)
    if not result["success"]:
        raise ValueError(f"invalid Sokoban source: {result}")
    return _eos([_small(sokoban_task._draw_frame(meta, s)) for s in states])


def flowfree(source: dict) -> np.ndarray:
    trace = json.loads(data_path(source["observed_trace_path"]).read_text())
    cells = [tuple(c) for c in trace["cells"]]
    colors = trace["colors"]
    n = int(source["grid_size"])
    if (not source.get("continuous_global_walk") or len(cells) != n * n
            or len(set(cells)) != n * n or len(colors) != len(cells)
            or any(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in zip(cells, cells[1:]))):
        raise ValueError(f"{source.get('sample_id')}: the trace is not a single walk over all cells")
    drawn = [[] for _ in source["paths"]]
    frames = [_small(flowfree_task._draw_frame(source, drawn))]
    for cell, color in zip(cells, colors):
        drawn[color].append(cell)
        frames.append(_small(flowfree_task._draw_frame(source, drawn, cursor=cell, cursor_color_index=color)))
    frames.append(_small(flowfree_task._draw_frame(source, drawn)))
    return _eos(frames)


def video_length(task: str, source: dict) -> int:
    """Frames of the expert video. FlowFree: the empty board, one frame per cell, the full board without the cursor
    and the EOS frame; Sokoban: one frame per state (the start and every move) and the EOS frame."""
    if task == "flowfree":
        return int(source["grid_size"]) ** 2 + 3
    return int(source["num_moves"]) + 2


RENDER = {"sokoban": sokoban, "flowfree": flowfree}
