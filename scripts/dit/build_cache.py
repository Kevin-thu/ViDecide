"""Pixel caches of DiT-B, rendered from the released dataset.

    python scripts/dit/build_cache.py --task sokoban          # all sets -> data/dit/sokoban/
    python scripts/dit/build_cache.py --task flowfree --sets train,val,test

Every set of data/VideoRLVR/<Game>/<set>.jsonl becomes <out>/<task>/<set>_frames.npy (the uint8 frames of all expert
videos, 256 x 256, concatenated) and <set>_index.json (sample_id, source, start, length). Sets: train, val and test
(VideoRLVR), hard_train (the 5,000 harder training puzzles used for post-training) and hard_test (FlowFree-Hard,
Sokoban-Hard). Each expert video is replayed by the DiT-B verifier (checkers.py) before it is written. A complete
set is not rebuilt unless --overwrite is given. All sets take about 160 GB for FlowFree and 65 GB for Sokoban.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from render_discrete import RENDER, SIZE, video_length
from videcide.paths import DATA_ROOT, data_path, manifest_path, read_manifest

SETS = ("train", "val", "test", "hard_train", "hard_test")


def render(job: tuple[str, str]) -> np.ndarray:
    os.environ["OMP_NUM_THREADS"] = "1"
    import cv2
    cv2.setNumThreads(1)
    from checkers import CHECK
    task, source_path = job
    source = json.loads(data_path(source_path).read_text())
    frames = RENDER[task](source)
    verdict = CHECK[task](frames, source)
    if not verdict["success"] or len(frames) != video_length(task, source):
        raise ValueError(f"{source_path}: the expert video fails the verifier or has the wrong length: {verdict}")
    return frames


def build_set(task: str, name: str, out: Path, workers: int, limit: int) -> dict:
    rows = read_manifest(manifest_path(task, name))
    if limit:
        rows = rows[:limit]
    sources = [r["source_path"] for r in rows]
    lengths = [video_length(task, json.loads(data_path(s).read_text())) for s in sources]
    tmp = out / f"{name}_frames.npy.tmp"
    frames = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(sum(lengths), SIZE, SIZE, 3))
    index, start = [], 0
    chunk = 8 * workers
    with ProcessPoolExecutor(workers) as pool:
        for lo in range(0, len(rows), chunk):
            jobs = [(task, s) for s in sources[lo:lo + chunk]]
            for row, video in zip(rows[lo:lo + chunk], pool.map(render, jobs, chunksize=1)):
                frames[start:start + len(video)] = video
                index.append({"sample_id": row["sample_id"], "source": row["source_path"], "start": start,
                              "length": len(video)})
                start += len(video)
            print(f"[{task}/{name}] {len(index)}/{len(rows)}", flush=True)
    frames.flush()
    del frames
    tmp.replace(out / f"{name}_frames.npy")
    (out / f"{name}_index.json").write_text(json.dumps(index))
    return {"task": task, "set": name, "videos": len(index), "frames": start,
            "min_length": min(lengths), "max_length": max(lengths)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=tuple(RENDER))
    ap.add_argument("--sets", default=",".join(SETS), help="comma-separated sets (default: all)")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "dit", help="cache root (default: data/dit)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="only the first N puzzles of each set (for a quick check)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out = a.out / a.task
    out.mkdir(parents=True, exist_ok=True)
    for name in a.sets.split(","):
        if name not in SETS:
            ap.error(f"unknown set {name!r}; expected one of {SETS}")
        if (out / f"{name}_index.json").exists() and not a.overwrite:
            print(f"[{a.task}/{name}] exists, skipped (--overwrite to rebuild)", flush=True)
            continue
        print(json.dumps(build_set(a.task, name, out, a.workers, a.limit)), flush=True)


if __name__ == "__main__":
    main()
