"""Strict verifiers on expert videos rendered with the project renderer and on corrupted copies (CPU only)."""
import json
from pathlib import Path

import numpy as np
import pytest

from videcide.paths import data_path, manifest_path, read_manifest
from videcide.tasks import flowfree_task, maze_task, sokoban_task
from videcide.tasks.video import add_eos_border
from videcide.tasks.video_parser import find_eos_frame, parse_video

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "assets" / "examples"


def source(name):
    return json.loads((EXAMPLES / f"{name}.json").read_text())


@pytest.fixture(scope="module")
def sokoban():
    src = source("sokoban_test_00756")
    return src, sokoban_task.render_optimal(src, theme="classic", frames_per_move=4, start_hold=5)


@pytest.fixture(scope="module")
def flowfree():
    src = source("flowfree_test_00005")
    return src, flowfree_task.render_optimal(src)


def stop_early(video, frame):
    """The video stops at `frame` and ends with EOS frames, before the goal is reached."""
    return add_eos_border(np.concatenate([video[:frame], np.repeat(video[frame - 1:frame], 8, axis=0)]))


def test_eos_detection(sokoban):
    _, video = sokoban
    eos = find_eos_frame(video)
    assert eos is not None and eos == len(video) - 4          # the last four frames carry the red border
    assert find_eos_frame(video[:eos]) is None


def test_sokoban_expert_passes(sokoban):
    src, video = sokoban
    verdict = parse_video(video, src, "sokoban")
    assert verdict["success"] and verdict["solved"] and verdict["illegal_transitions"] == 0


def first_push_frames(src):
    """Frames of the first move that pushes a box (five opening frames, then four frames per move)."""
    states, _, _ = sokoban_task.replay(src, src["directions"])
    k = next(i for i, (a, b) in enumerate(zip(states, states[1:])) if a["boxes"] != b["boxes"])
    return 5 + 4 * k, 5 + 4 * (k + 1)


@pytest.mark.parametrize("corruption", ["box_teleport", "stops_early", "no_eos"])
def test_sokoban_corruptions_fail(sokoban, corruption):
    src, video = sokoban
    eos = find_eos_frame(video)
    a, b = first_push_frames(src)
    bad = {"box_teleport": np.concatenate([video[:a], video[b:]]),   # the first push is skipped
           "stops_early": stop_early(video, 50),
           "no_eos": video[:eos]}[corruption]
    assert not parse_video(bad, src, "sokoban")["success"]


def test_flowfree_expert_passes(flowfree):
    src, video = flowfree
    verdict = parse_video(video, src, "flowfree")
    assert verdict["success"] and verdict["outcome"] == "success"


@pytest.mark.parametrize("corruption", ["paint_removed", "stops_early"])
def test_flowfree_corruptions_fail(flowfree, corruption):
    src, video = flowfree
    bad = {"paint_removed": np.concatenate([video[:60], video[50:60][::-1], video[60:]]),
           "stops_early": stop_early(video, 80)}[corruption]
    verdict = parse_video(bad, src, "flowfree")
    assert not verdict["success"]
    assert verdict["outcome"] == {"paint_removed": "rule_violation", "stops_early": "incomplete"}[corruption]


# A 7 x 7 maze (1 = wall) with a dead end at the top right; the solution winds from the top left to the bottom right.
MAZE = {"grid": [[1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 1, 0, 1], [1, 1, 1, 0, 1, 0, 1], [1, 0, 0, 0, 0, 0, 1],
                 [1, 0, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1]],
        "start": [1, 1], "goal": [5, 5], "render_theme": "forest",
        "solution_path": [[1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1], [4, 1], [5, 1], [5, 2], [5, 3], [5, 4],
                          [5, 5]]}


@pytest.fixture(scope="module")
def maze():
    return MAZE, maze_task.render_optimal(MAZE["grid"], MAZE["solution_path"], MAZE["start"], MAZE["goal"],
                                          theme=MAZE["render_theme"])


def test_maze_expert_passes(maze):
    src, video = maze
    assert len(video) == 4 + 18 + 8 + 3            # opening, 12 moves at 1.5 frames, final hold, padding to 4k + 1
    verdict = parse_video(video, src, "maze")
    assert verdict["success"] and verdict["process_violation"] is None
    assert verdict["parsed_cells"] == len(src["solution_path"])


@pytest.mark.parametrize("corruption", ["stops_early", "no_eos", "trail_on_wall"])
def test_maze_corruptions_fail(maze, corruption):
    src, video = maze
    eos = find_eos_frame(video)
    if corruption == "trail_on_wall":                # the verifier sees a wall where the line passes
        src = {**src, "grid": [row[:] for row in src["grid"]]}
        src["grid"][3][2] = 1
    bad = {"stops_early": stop_early(video, 14), "no_eos": video[:eos], "trail_on_wall": video}[corruption]
    verdict = parse_video(bad, src, "maze")
    assert not verdict["success"]
    if corruption == "trail_on_wall":
        assert verdict["process_violation"] == "trail_on_wall"
    if corruption == "stops_early":
        assert not verdict["final_path_ok"]


VBVR_TEST = manifest_path("vbvr", "test")


@pytest.mark.skipif(not VBVR_TEST.exists(), reason="needs the VBVR-MultiStep data in data/ (or $VIDECIDE_DATA)")
def test_vbvr_reference_videos_pass():
    from videcide.tasks import vbvr_checkers as V

    by_task = {}
    for row in read_manifest(VBVR_TEST):
        by_task.setdefault(row["task"], row)
    for task, row in sorted(by_task.items()):
        frames = V.read(data_path(row["native_video"]))
        assert V.check(row, frames)["success"], task


@pytest.mark.skipif(not manifest_path("sokoban", "test").exists(), reason="needs the data in data/ (or $VIDECIDE_DATA)")
@pytest.mark.parametrize("task", ["flowfree", "sokoban", "maze"])
def test_game_reference_videos_pass(task):
    from videcide.tasks.video import read_mp4

    for name in ("test", "hard_test") if task != "maze" else ("test", "train"):
        row = read_manifest(manifest_path(task, name))[0]
        source = json.loads(data_path(row["source_path"]).read_text())
        assert parse_video(read_mp4(data_path(row["native_video"])), source, task)["success"], (task, name)
