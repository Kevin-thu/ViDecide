"""Render the FlowFree, Sokoban and Maze videos of a manifest from their sources, as in the released data.

    python scripts/benchmarks/render_videos.py data/VideoRLVR/Sokoban/hard_test.jsonl --check   # compare with the files
    python scripts/benchmarks/render_videos.py MANIFEST --out DIR                               # write DIR/<native_video>

FlowFree videos fill one cell per frame along the walk of the puzzle's trace (``observed_trace_path``), Sokoban videos
show each move or push over four frames after a five-frame opening hold, and Maze videos draw the solution path at 1.5
frames per move in the colours of the puzzle's theme (``render_theme``) after a four-frame opening hold; all end with
eight frames that hold the final state, the last four with the red EOS border, and are written at 16 fps. Every rendered video must pass the strict
verifier and have the length given in the manifest. ``--check`` writes the videos to a temporary folder and compares
them byte for byte with the files the manifest names.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from videcide.paths import data_path, read_manifest


def render(row: dict):
    from videcide.tasks import flowfree_task, maze_task, sokoban_task
    from videcide.tasks.video_parser import parse_video

    source = json.loads(data_path(row["source_path"]).read_text())
    if row["task"] == "flowfree":
        trace = json.loads(data_path(source["observed_trace_path"]).read_text())
        frames = flowfree_task.render_order(source, trace, int(row["native_frames"]))
    elif row["task"] == "sokoban":
        frames = sokoban_task.render_optimal(source, theme="classic", frames_per_move=4, start_hold=5)
    elif row["task"] == "maze":
        frames = maze_task.render_optimal(source["grid"], source["solution_path"], source["start"], source["goal"],
                                          theme=source["render_theme"])
    else:
        raise ValueError(f"{row['sample_id']}: task {row['task']} is not rendered from sources")
    if len(frames) != int(row["native_frames"]):
        raise ValueError(f"{row['sample_id']}: rendered {len(frames)} frames, the manifest says {row['native_frames']}")
    if not parse_video(frames, source, row["task"])["success"]:
        raise ValueError(f"{row['sample_id']}: the rendered video fails the verifier")
    return frames


def one(job) -> str:
    import cv2
    from videcide.tasks.video import write_mp4

    cv2.setNumThreads(1)
    row, out, check = job
    frames = render(row)
    target = Path(out) / row["native_video"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not check:
        raise FileExistsError(target)
    write_mp4(target, frames, fps=16)
    if not check:
        return "written"
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return "identical" if digest(target) == digest(data_path(row["native_video"])) else "different"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, help="root of the written videos (their manifest paths below it)")
    parser.add_argument("--check", action="store_true", help="compare the rendered videos with the existing files")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if bool(args.out) == args.check:
        parser.error("give either --out or --check")
    rows = read_manifest(args.manifest)[:args.limit or None]
    with tempfile.TemporaryDirectory() as scratch, ProcessPoolExecutor(args.workers) as pool:
        out = scratch if args.check else args.out
        counts = Counter(pool.map(one, [(r, str(out), args.check) for r in rows], chunksize=4))
    print(json.dumps({"manifest": str(args.manifest), "videos": len(rows), **counts}))


if __name__ == "__main__":
    main()
