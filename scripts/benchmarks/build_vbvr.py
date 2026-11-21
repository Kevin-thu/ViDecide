"""VBVR-MultiStep in the layout of the released data, built from the upstream dataset.

Download Video-Reason/VBVR-MultiStep (https://huggingface.co/datasets/Video-Reason/VBVR-MultiStep) to --upstream, then
    python scripts/benchmarks/build_vbvr.py splits      # train (5,000 per task, distinct puzzles) and val (50 per task)
    python scripts/benchmarks/build_vbvr.py clean-test  # test: 100 puzzles per task whose color-free layout is not in train
    python scripts/benchmarks/build_vbvr.py speeds      # agent speed of TSP Reward Collection and Ordinal Number Sequence
    python scripts/benchmarks/build_vbvr.py build       # videos, sources and train / val / test manifests in --out

The six planning tasks are Sokoban (Multi-07), Sliding Puzzle (08), Tower of Hanoi (09), Snake Routing (10), TSP Reward
Collection (11) and Ordinal Number Sequence (12). Videos lose their idle frames (TSP Reward Collection and Ordinal Number
Sequence are replayed at a constant agent speed, Sokoban at two frames per move), are resized to 384x384, and end with
eight frames that hold the final state, the last four with the red EOS border. Sources are the upstream
question_metadata.json of every puzzle, which the checkers read.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from videcide.paths import DATA_ROOT
from videcide.tasks.video import CANVAS_SIZE, add_eos_border, pad_to_vae_length, write_mp4

TASKS = {"07": "sokoban", "08": "sliding", "09": "hanoi", "10": "snake", "11": "tsp", "12": "ordinal"}
FOLDERS = {"sokoban": "sokoban", "sliding": "sliding_puzzle", "hanoi": "tower_of_hanoi", "snake": "snake_routing",
           "tsp": "tsp_reward_collection", "ordinal": "ordinal_number_sequence"}   # task -> folder of its media
END_HOLD = 8
TRAIN_ROWS, TEST_ROWS, VAL_ROWS = 5000, 100, 50


def read_video(data: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(data)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def changed(a: np.ndarray, b: np.ndarray, thr: int = 40, min_pixels: int = 30) -> bool:
    return int((np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(-1) > thr).sum()) > min_pixels


def motion_span(frames: np.ndarray) -> tuple[int, int]:
    idx = [i for i in range(1, len(frames)) if changed(frames[i - 1], frames[i])]
    return (idx[0], idx[-1]) if idx else (1, 0)


def finish(frames: list[np.ndarray] | np.ndarray) -> np.ndarray:
    frames = [cv2.resize(f, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_AREA) for f in frames]
    frames += [frames[-1]] * END_HOLD
    return add_eos_border(pad_to_vae_length(np.stack(frames)))


def trim_holds(frames: np.ndarray) -> list[np.ndarray]:
    first, last = motion_span(frames)
    return [frames[0]] + list(frames[first:last + 1])


HANOI_TEXT_TOP = 958
HANOI_DIVIDER_END = 975
HANOI_HEADER_BOTTOM = 50
HANOI_HEADER_SHIFT = 32


def hanoi_blank_caption(frames: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for f in frames:
        f = f.copy()
        background = np.median(f[HANOI_TEXT_TOP - 20:HANOI_TEXT_TOP - 4, :40].reshape(-1, 3), axis=0)
        reference = f[HANOI_TEXT_TOP - 8].copy()
        divider = np.abs(reference.astype(np.int16) - background).max(-1) > 20
        f[HANOI_TEXT_TOP:] = background
        f[HANOI_TEXT_TOP:HANOI_DIVIDER_END, divider] = reference[divider]
        header = f[:HANOI_HEADER_BOTTOM].copy()
        glyph = np.abs(header.astype(np.int16) - background).max(-1) > 30
        f[:HANOI_HEADER_BOTTOM] = background
        moved = np.zeros((HANOI_HEADER_BOTTOM + HANOI_HEADER_SHIFT, glyph.shape[1]), dtype=bool)
        moved[HANOI_HEADER_SHIFT:] = glyph
        f[:HANOI_HEADER_BOTTOM + HANOI_HEADER_SHIFT][moved] = header[glyph]
        out.append(f)
    return out


SOKOBAN_FRAMES_PER_MOVE = 2


def sokoban_uniform(frames: np.ndarray, meta: dict) -> list[np.ndarray]:
    moves = len(meta["parameters"]["solution_state"]["solution_path"])
    idx = [i for i in range(1, len(frames)) if changed(frames[i - 1], frames[i], min_pixels=500)]
    if len(idx) != moves:
        raise ValueError(f"{len(idx)} visible jumps for {moves} moves")
    out = [frames[0]]
    for i in idx:
        out += [frames[i]] * SOKOBAN_FRAMES_PER_MOVE
    return out


AGENT_RGB = (255, 165, 0)


def agent_mask(img: np.ndarray) -> np.ndarray:
    r, g, b = (img[..., i].astype(np.int16) for i in range(3))
    return (np.abs(r - AGENT_RGB[0]) < 25) & (np.abs(g - AGENT_RGB[1]) < 25) & (b < 50)


def agent_track(frames: np.ndarray) -> tuple[np.ndarray, list[int]]:
    pts, idx = [], []
    for i, f in enumerate(frames):
        ys, xs = np.nonzero(agent_mask(f))
        if len(xs) > 50:
            pts.append((xs.mean(), ys.mean()))
            idx.append(i)
    return np.array(pts), idx


def source_speed(frames: np.ndarray) -> float:
    pts, _ = agent_track(frames)
    steps = np.hypot(*np.diff(pts, axis=0).T)
    moving = steps[steps > 0.5]
    return float(np.median(moving)) if len(moving) else 0.0


def agent_replay(frames: np.ndarray, speed: float) -> list[np.ndarray]:
    pts, idx = agent_track(frames)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    keep = np.concatenate([[True], seg > 0.5])
    pts, idx = pts[keep], [i for i, k in zip(idx, keep) if k]
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    first, last = frames[0], frames[-1]
    m0 = cv2.dilate(agent_mask(first).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    m1 = cv2.dilate(agent_mask(last).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    background = first.copy()
    background[m0] = last[m0]
    diff = (np.abs(first.astype(np.int16) - last.astype(np.int16)).max(-1) > 40) & ~m0 & ~m1
    count, labels = cv2.connectedComponents(cv2.dilate(diff.astype(np.uint8), np.ones((5, 5), np.uint8)))
    items = []
    for k in range(1, count):
        ys, xs = np.nonzero(labels == k)
        if len(xs) < 30:
            continue
        centre = np.array([xs.mean(), ys.mean()])
        taken_at = arc[int(np.argmin(np.hypot(*(pts - centre).T)))]
        items.append((labels == k, taken_at))
    ys, xs = np.nonzero(agent_mask(first))
    radius = float(np.sqrt(len(xs) / np.pi))
    colour = tuple(int(v) for v in np.median(first[agent_mask(first)], axis=0))
    n = max(1, int(np.ceil(arc[-1] / speed)))
    out = [first]
    for j in range(1, n + 1):
        s = min(arc[-1], j * speed)
        img = background.copy()
        for mask, taken_at in items:
            if s >= taken_at:
                img[mask] = last[mask]
        x = np.interp(s, arc, pts[:, 0])
        y = np.interp(s, arc, pts[:, 1])
        cv2.circle(img, (int(round(x * 16)), int(round(y * 16))), int(round(radius * 16)), colour, -1,
                   lineType=cv2.LINE_AA, shift=4)
        out.append(img)
    return out


def build(task: str, frames: np.ndarray, meta: dict, speed: dict) -> np.ndarray:
    if task in ("tsp", "ordinal"):
        return finish(agent_replay(frames, speed[task]))
    if task == "sokoban":
        return finish(sokoban_uniform(frames, meta))
    if task == "hanoi":
        return finish(hanoi_blank_caption(trim_holds(frames)))
    return finish(trim_holds(frames))


def iter_shard(path: Path):
    groups: dict[str, dict[str, bytes]] = {}
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = member.name.split("/")
            groups.setdefault(parts[-2], {})[parts[-1]] = tar.extractfile(member).read()
    for name, files in sorted(groups.items()):
        yield name, files


COLOUR_KEYS = {"tile_color_theme", "color", "colors", "colour", "theme", "shape", "reward_order", "generation_method",
               "min_moves", "max_moves"}


def clean_layout(task, p):
    if task == "sliding":
        return tuple(sorted((tuple(o["initial_position"]), o["value"]) for o in p["objects"]))
    if task == "hanoi":
        return tuple(sorted((o["initial_stack"], o["initial_position"], o["target_stack"], o["target_position"]) for o in p["objects"]))
    if task == "sokoban":
        s = p["puzzle_state"]
        return (tuple(map(tuple, s["grid"])), tuple(s["initial_player"]), tuple(sorted(map(tuple, s["initial_boxes"]))), tuple(sorted(map(tuple, s["goals"]))))
    strip = lambda o: {k: strip(v) for k, v in o.items() if k not in COLOUR_KEYS} if isinstance(o, dict) else [strip(x) for x in o] if isinstance(o, list) else o
    return json.dumps(strip(p), sort_keys=True)




def shard_job(args):
    code, path, speed, wanted, out = args
    task, out = TASKS[code], Path(out)
    rows = []
    for name, files in iter_shard(Path(path)):
        index = int(name.rsplit("_", 1)[-1])
        split = wanted.get(index)
        if split is None:
            continue
        sample_id = f"vbvr_{task}_{index:05d}"
        meta = json.loads(files["question_metadata.json"])
        try:
            frames = read_video(files["ground_truth.mp4"])
            video = build(task, frames, meta, speed)
        except Exception as exc:
            rows.append({"sample_id": sample_id, "task": f"vbvr_{task}", "split": split, "error": repr(exc)})
            continue
        video_path = Path("videos") / FOLDERS[task] / f"{sample_id}.mp4"
        source_path = Path("sources") / FOLDERS[task] / f"{sample_id}.json"
        write_mp4(out / video_path, video)
        (out / source_path).parent.mkdir(parents=True, exist_ok=True)
        (out / source_path).write_text(json.dumps({k: v for k, v in meta.items() if k != "timestamp"}, indent=2) + "\n")
        prefix = out.relative_to(DATA_ROOT) if out.is_relative_to(DATA_ROOT) else out
        rows.append({"sample_id": sample_id, "task": f"vbvr_{task}", "split": split,
                     "prompt": files["prompt.txt"].decode().strip(), "native_frames": int(len(video)),
                     "native_video": str(prefix / video_path), "source_path": str(prefix / source_path)})
    return rows


def shards(upstream: Path, code: str, limit_shards: int = 0) -> list[Path]:
    paths = sorted((upstream / "questions").glob(f"Multi-{code}_*.tar.gz"))
    return paths[:limit_shards] if limit_shards else paths


def metadata_rows(upstream: Path, code: str) -> list[tuple[int, dict]]:
    import pandas as pd
    d = pd.read_parquet(glob.glob(str(upstream / "data" / "metadata_shards" / f"Multi-{code}_*.parquet"))[0])
    return sorted((int(s.rsplit("_", 1)[-1]), json.loads(m)["parameters"]) for s, m in zip(d.sample_id, d.metadata_json))


def splits_main(a) -> None:
    """Train: the first 5,000 distinct puzzles (topped up from later indices); val: 50 more distinct puzzles."""
    splits, report = {}, {}
    for code, task in TASKS.items():
        rows = [(i, json.dumps(p, sort_keys=True)) for i, p in metadata_rows(a.upstream, code)]
        seen, out = set(), {}
        for index, key in rows:
            if index < TRAIN_ROWS and key not in seen:
                out[index] = "train"
                seen.add(key)
        n_skip = n_val = n_topup = 0
        for index, key in rows:
            if index < TRAIN_ROWS or key in seen:
                continue
            if n_skip < TEST_ROWS:              # the original test puzzles: replaced by clean-test, never used
                out[index] = "original_test"; n_skip += 1
            elif n_val < VAL_ROWS:
                out[index] = "val"; n_val += 1
            elif sum(v == "train" for v in out.values()) < TRAIN_ROWS:
                out[index] = "train"; n_topup += 1
            else:
                break
            seen.add(key)
        splits[task] = out
        report[task] = {"train": sum(v == "train" for v in out.values()), "train_topped_up": n_topup, "val": n_val}
    a.work.mkdir(parents=True, exist_ok=True)
    (a.work / "splits.json").write_text(json.dumps({"report": report, "splits": splits}))
    print(json.dumps(report, indent=1))


def clean_test_main(a) -> None:
    """Test: 100 puzzles per task, outside the other splits, whose color-free layout occurs neither in train nor twice."""
    table = json.loads((a.work / "splits.json").read_text())
    report = {}
    for code, task in TASKS.items():
        old = table["splits"][task]
        rows = metadata_rows(a.upstream, code)
        seen = {clean_layout(task, p) for i, p in rows if old.get(str(i)) == "train"}
        pick, used = {}, set()
        for i, p in rows:
            if i < TRAIN_ROWS or str(i) in old:
                continue
            k = clean_layout(task, p)
            if k in seen or k in used:
                continue
            pick[str(i)] = "test"; used.add(k)
            if len(pick) == TEST_ROWS:
                break
        table["splits"][task] = {**{i: s for i, s in old.items() if s != "original_test"}, **pick}
        report[task] = {"test": len(pick), "train_layouts": len(seen)}
    table["report_clean_test"] = report
    (a.work / "splits.json").write_text(json.dumps(table))
    print(json.dumps(report))


def stage_speeds(a) -> None:
    out = {}
    for code in ("11", "12"):
        speeds = [source_speed(read_video(files["ground_truth.mp4"])) for _, files in iter_shard(shards(a.upstream, code)[0])]
        out[TASKS[code]] = float(np.median(speeds))
        print(TASKS[code], "agent speed (px per frame of the 1024 canvas): median %.2f p10 %.2f p90 %.2f" % (
            np.median(speeds), np.percentile(speeds, 10), np.percentile(speeds, 90)))
    a.work.mkdir(parents=True, exist_ok=True)
    (a.work / "agent_speed.json").write_text(json.dumps(out, indent=1))


def build_main(a) -> None:
    speed = json.loads((a.work / "agent_speed.json").read_text())
    splits = json.loads((a.work / "splits.json").read_text())["splits"]
    if any(s not in ("train", "val", "test") for table in splits.values() for s in table.values()):
        raise SystemExit("run clean-test before build")
    jobs = []
    for code in a.tasks.split(","):
        for path in shards(a.upstream, code, a.limit_shards):
            lo, hi = (int(x) for x in re.search(r"_(\d{5})-(\d{5})\.tar\.gz$", path.name).groups())
            table = splits[TASKS[code]]
            wanted = {i: table[str(i)] for i in range(lo, hi + 1) if str(i) in table}
            if wanted:
                jobs.append((code, str(path), speed, wanted, str(a.out)))
    rows = []
    with ProcessPoolExecutor(a.workers) as pool:
        for i, part in enumerate(pool.map(shard_job, jobs)):
            rows += part
            if i % 50 == 0:
                print(f"{i + 1}/{len(jobs)} shards, {len(rows)} rows", flush=True)
    errors = [r for r in rows if "error" in r]
    for split in ("train", "val", "test"):
        with open(a.out / f"{split}.jsonl", "w") as f:
            for r in rows:
                if r["split"] == split and "error" not in r:
                    f.write(json.dumps(r) + "\n")
    print(f"wrote {a.out}/{{train,val,test}}.jsonl: {len(rows) - len(errors)} rows, {len(errors)} errors")
    for r in errors[:10]:
        print(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("splits", "clean-test", "speeds", "build"))
    ap.add_argument("--upstream", type=Path, default=Path("data/upstream/VBVR-MultiStep"), help="upstream dataset")
    ap.add_argument("--work", type=Path, default=Path("outputs/vbvr_build"), help="splits.json and agent_speed.json")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "VBVR-MultiStep", help="build: output folder")
    ap.add_argument("--tasks", default="07,08,09,10,11,12", help="build: upstream task codes")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit-shards", type=int, default=0, help="build: first N shards per task (tests)")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    {"splits": splits_main, "clean-test": clean_test_main, "speeds": stage_speeds, "build": build_main}[a.stage](a)


if __name__ == "__main__":
    main()
