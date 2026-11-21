"""Training latents: encode the training and validation videos of a benchmark with the Wan2.2 VAE into an LMDB.

    python scripts/benchmarks/build_latents.py lengths --task flowfree                  # data/latents/flowfree/lengths.json
    python scripts/benchmarks/build_latents.py encode  --task flowfree --variants native,fixed_length [--shard i --shards n]
    python scripts/benchmarks/build_latents.py merge   --task flowfree --variant native     # data/latents/flowfree/native/

--task is flowfree, sokoban, maze, vbvr or robotwin; the rows are those of the train and val manifests of the benchmark in
data/. ``--set hard_train`` encodes the 5,000 harder training puzzles of FlowFree or Sokoban instead
(data/latents/<task>_hard_train/native), the expert anchors of post-training. Every shard of ``encode`` runs on one
GPU; ``merge`` joins the shards into ``<variant>/lmdb`` and ``<variant>/metadata.json``, the files named by
``data_path`` and ``metadata_path`` in configs/sft/.

FlowFree, Sokoban and Maze videos are rendered from their sources, frame for frame as the released videos before
compression; VBVR-MultiStep and RoboTwin-Plan videos are decoded from the released mp4 files. Variants:
    native         every video at its native duration (causal models)
    fixed_length   every video holds its last frame up to the fixed length of the bidirectional baseline of the task
                   (RoboTwin-Plan: of each sub-task, the longest training video of the sub-task); longer videos are
                   left out
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from videcide.paths import DATA_ROOT, data_path, manifest_path, read_manifest

TASKS = ("flowfree", "sokoban", "maze", "vbvr", "robotwin")
RENDERED = ("flowfree", "sokoban", "maze")            # videos rendered from their sources
FIXED_FRAMES = {"flowfree": 173, "sokoban": 173, "maze": 105, "vbvr": 165}   # fixed lengths of the bidirectional baselines
LMDB_MAP_SIZE = 400 * 1024 ** 3
MERGE_MAP_SIZE = 600 * 1024 ** 3


def output_root(task: str, set_name: str) -> Path:
    return DATA_ROOT / "latents" / (task if set_name == "train" else f"{task}_{set_name}")


def manifest_rows(task: str, set_name: str) -> list[dict]:
    """Rows to encode: train + val, or the rows of another set (hard_train)."""
    names = ("train", "val") if set_name == "train" else (set_name,)
    return [row for name in names for row in read_manifest(manifest_path(task, name))]


def render(task: str, source_path: str) -> tuple[np.ndarray, dict]:
    """Expert video of a FlowFree, Sokoban or Maze puzzle, rendered from its source (as the released videos)."""
    from videcide.tasks import flowfree_task, maze_task, sokoban_task

    source = json.loads(data_path(source_path).read_text())
    if task == "sokoban":
        video = sokoban_task.render_optimal(source, theme="classic", frames_per_move=4, start_hold=5)
        extra = {"theme": "classic", "action_count": int(source["num_moves"])}
    elif task == "flowfree":
        from videcide.tasks.flow_replay import evaluate_flowfree

        order = json.loads(data_path(source["observed_trace_path"]).read_text())
        video = flowfree_task.render_order(source, order, int(source["frames"]))
        verdict = evaluate_flowfree(video, source, require_adjacent_color_switch=True)
        if not verdict["success"]:
            raise ValueError(f"FlowFree render fails the replay: {source['sample_id']}: {verdict['failure_reason']}")
        extra = {"theme": "default", "action_count": int(source["grid_size"]) ** 2 - 1}
    elif task == "maze":
        video = maze_task.render_optimal(source["grid"], source["solution_path"], source["start"], source["goal"],
                                         theme=source["render_theme"])
        extra = {"theme": source["render_theme"], "action_count": len(source["solution_path"]) - 1}
    else:
        raise ValueError(f"{task} videos are read from their mp4 files")
    return video, {"sample_id": source["sample_id"], "source_path": source_path, **extra}


def load_video(job: tuple[str, dict]) -> tuple[np.ndarray, dict]:
    task, row = job
    if task in RENDERED:
        video, meta = render(task, row["source_path"])
    else:
        from videcide.tasks.video import read_mp4

        video = read_mp4(data_path(row["native_video"]))
        meta = {"sample_id": row["sample_id"], "source_path": row["source_path"], "theme": None, "action_count": None}
    if len(video) != int(row["native_frames"]):
        raise ValueError(f"{row['sample_id']}: {len(video)} frames, the manifest says {row['native_frames']}")
    return video, {**meta, "split": row["split"], "prompt": row["prompt"], "native_frames": int(row["native_frames"]),
                   "subtask": row.get("subtask", row["task"])}


def _single_thread_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)


def round_up_to_vae_length(frames: int) -> int:
    """Smallest length of the form 4k + 1 that holds ``frames`` frames."""
    return frames + (-(frames - 1)) % 4


def stage_lengths(task: str, set_name: str) -> None:
    rows = manifest_rows(task, set_name)
    train = [r for r in rows if r["split"] == "train"]
    frames = np.array([r["native_frames"] for r in train])
    summary = {"task": task, "set": set_name, "rows": len(rows), "train_rows": len(train),
               "native_mean": float(frames.mean()), "native_min": int(frames.min()), "native_max": int(frames.max())}
    if task == "robotwin":
        summary["durations_by_subtask"] = {
            sub: {"freeze_frames": round_up_to_vae_length(max(r["native_frames"] for r in train if r["subtask"] == sub)),
                  "train_rows": sum(r["subtask"] == sub for r in train)}
            for sub in sorted({r["subtask"] for r in train})}
    else:
        summary["freeze_frames"] = FIXED_FRAMES.get(task)
    out = output_root(task, set_name)
    out.mkdir(parents=True, exist_ok=True)
    keep = ("sample_id", "split", "subtask", "native_frames")
    (out / "lengths.json").write_text(json.dumps({"summary": summary, "rows": [
        {k: r.get(k, r["task"] if k == "subtask" else None) for k in keep} for r in rows]}) + "\n")
    print(json.dumps(summary), flush=True)


def stage_encode(task: str, set_name: str, shard: int, shards: int, batch: int, workers: int, device: str,
                 variants: tuple[str, ...], limit: int = 0) -> None:
    import lmdb
    import torch

    from videcide.paths import add_causal_forcing_to_path
    add_causal_forcing_to_path()
    from utils.wan_wrapper import WanVAEWrapper

    root = output_root(task, set_name)
    lengths = json.loads((root / "lengths.json").read_text())["summary"]
    durations = lengths.get("durations_by_subtask")

    def fixed_frames(meta: dict) -> int:
        return durations[meta["subtask"]]["freeze_frames"] if durations else int(lengths["freeze_frames"])

    rows = manifest_rows(task, set_name)
    rows = sorted(rows, key=lambda r: (r["native_frames"], r["sample_id"]))
    rows = [r for r in rows if int(hashlib.sha256(r["sample_id"].encode()).hexdigest()[:8], 16) % shards == shard]
    if limit:
        rows = rows[:limit]
    out = root / "shards"
    out.mkdir(parents=True, exist_ok=True)
    done_marker = out / f"DONE_{'-'.join(variants)}_shard{shard:02d}_of{shards:02d}{f'_limit{limit}' if limit else ''}.json"
    if done_marker.exists():
        print(f"[encode] {task} shard {shard} already complete", flush=True)
        return
    vae = WanVAEWrapper(model_name="checkpoints/Wan2.2-TI2V-5B").to(device=device, dtype=torch.bfloat16).eval()
    torch.cuda.reset_peak_memory_stats()
    envs = {v: lmdb.open(str(out / f"lmdb_{v}_shard{shard:02d}"), map_size=LMDB_MAP_SIZE) for v in variants}
    metadata = {v: [] for v in variants}
    spatial = None
    started = time.time()

    def encode(videos: list[np.ndarray]) -> np.ndarray:
        x = torch.from_numpy(np.stack(videos)).to(device=device).permute(0, 4, 1, 2, 3)
        x = x.to(torch.bfloat16).div_(127.5).sub_(1.0)
        with torch.no_grad():
            return vae.encode_to_latent(x).to(torch.float16).cpu().numpy()

    def write(variant: str, items: list[tuple[np.ndarray, dict]], latents: np.ndarray) -> None:
        nonlocal spatial
        with envs[variant].begin(write=True) as txn:
            for (video, meta), latent in zip(items, latents):
                index = len(metadata[variant])
                n = int(latent.shape[0])
                mask = np.zeros(n, dtype=np.float32)
                mask[1:] = 1.0                          # latent frame 0 is the given first frame o_0
                txn.put(f"latents_{index}_data".encode(), latent.tobytes())
                txn.put(f"latents_{index}_nframes".encode(), str(n).encode())
                txn.put(f"prompts_{index}_data".encode(), meta["prompt"].encode())
                txn.put(f"validity_mask_{index}_data".encode(), mask.tobytes())
                metadata[variant].append({
                    "entry_id": f"{meta['sample_id']}:{variant}", "sample_id": meta["sample_id"],
                    "training_noise_group_id": meta["sample_id"], "task": task, "split": meta["split"],
                    "variant": variant, "prompt": meta["prompt"], "source_path": meta["source_path"],
                    "theme": meta.get("theme"), "action_count": meta.get("action_count"), "subtask": meta["subtask"],
                    "pixel_frames": int(len(video)), "native_frames": meta["native_frames"], "row_idx": index,
                    "actual_latent_frames": n, "validity_mask_start": 1, "validity_mask_end": n,
                    "encode_resolution": 384, "model_name": "Wan2.2-TI2V-5B"})
                spatial = tuple(int(v) for v in latent.shape[1:])

    pending = {v: [] for v in variants}

    def flush(variant: str, force: bool = False) -> None:
        """Encode full batches of equal-length videos (and every remaining one with force)."""
        queue = pending[variant]
        newest = len(queue[-1][0]) if queue else None
        groups: dict[int, list] = {}
        for item in queue:
            groups.setdefault(len(item[0]), []).append(item)
        keep = []
        for length, items in groups.items():
            while len(items) >= batch:
                write(variant, items[:batch], encode([video for video, _ in items[:batch]]))
                items = items[batch:]
            if items and (force or length != newest):
                write(variant, items, encode([video for video, _ in items]))
            else:
                keep.extend(items)
        pending[variant] = keep

    def view(variant: str, video: np.ndarray, meta: dict) -> np.ndarray | None:
        if variant == "native":
            return video
        hold = fixed_frames(meta) - len(video)
        if hold < 0:
            return None                                  # longer than the fixed length: left out
        return np.concatenate([video, np.repeat(video[-1:], hold, axis=0)]) if hold else video

    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_single_thread_worker) as pool:
        jobs = iter((task, r) for r in rows)
        running = deque(pool.submit(load_video, job) for job in [next(jobs, None) for _ in range(2 * workers)] if job)
        count = 0
        while running:
            video, meta = running.popleft().result()
            job = next(jobs, None)
            if job is not None:
                running.append(pool.submit(load_video, job))
            for variant in variants:
                clip = view(variant, video, meta)
                if clip is not None:
                    pending[variant].append((clip, meta))
                    flush(variant)
            count += 1
            if count % 50 == 0:
                rate = count / (time.time() - started)
                print(f"[encode] {task} shard {shard}: {count}/{len(rows)} {rate * 3600:.0f}/h", flush=True)
    for variant in variants:
        flush(variant, force=True)
    for variant, env in envs.items():
        with env.begin(write=True) as txn:
            txn.put(b"latents_shape", " ".join(map(str, (len(metadata[variant]), 80, *spatial))).encode())
            txn.put(b"prompts_shape", str(len(metadata[variant])).encode())
            txn.put(b"variable_length", b"true")
        env.sync()
        env.close()
        (out / f"metadata_{variant}_shard{shard:02d}.json").write_text(json.dumps(metadata[variant], indent=1) + "\n")
    summary = {"task": task, "set": set_name, "shard": shard, "shards": shards, "rows": len(rows),
               **{f"{v}_rows": len(metadata[v]) for v in variants}, "batch": batch,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3, "seconds": round(time.time() - started, 1)}
    done_marker.write_text(json.dumps(summary, indent=2) + "\n")
    print("[encode] done " + json.dumps(summary), flush=True)


def stage_merge(task: str, set_name: str, variant: str) -> None:
    import lmdb

    root = output_root(task, set_name)
    shards = sorted((root / "shards").glob(f"lmdb_{variant}_shard*"))
    if not shards:
        raise FileNotFoundError(f"no {variant} shards in {root / 'shards'}; run encode first")
    out = root / variant
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    target = lmdb.open(str(out / "lmdb"), map_size=MERGE_MAP_SIZE)
    merged, shape = [], None
    for shard in shards:
        index = shard.name.rsplit("shard", 1)[1]
        metadata = json.loads((root / "shards" / f"metadata_{variant}_shard{index}.json").read_text())
        source = lmdb.open(str(shard), readonly=True, lock=False)
        with source.begin() as read, target.begin(write=True) as write:
            shape = read.get(b"latents_shape").decode().split()
            if int(shape[0]) != len(metadata):
                raise RuntimeError(f"{shard}: {shape[0]} rows in LMDB vs {len(metadata)} in metadata")
            for row in metadata:
                old, new = int(row["row_idx"]), len(merged)
                for key in ("latents_{}_data", "latents_{}_nframes", "prompts_{}_data", "validity_mask_{}_data"):
                    value = read.get(key.format(old).encode())
                    if value is None:
                        raise KeyError(f"{shard}: missing {key.format(old)}")
                    write.put(key.format(new).encode(), value)
                merged.append({**row, "row_idx": new})
        source.close()
    with target.begin(write=True) as write:
        write.put(b"latents_shape", " ".join([str(len(merged)), *shape[1:]]).encode())
        write.put(b"prompts_shape", str(len(merged)).encode())
        write.put(b"variable_length", b"true")
    target.sync()
    target.close()
    (out / "metadata.json").write_text(json.dumps(merged, indent=1) + "\n")
    print(json.dumps({"dir": str(out), "rows": len(merged), "shards": [s.name for s in shards]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("lengths", "encode", "merge"))
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--set", default="train", choices=("train", "hard_train"),
                        help="train: the train and val manifests; hard_train: the harder post-training puzzles")
    parser.add_argument("--variants", default="native,fixed_length", help="encode: comma-separated variants")
    parser.add_argument("--variant", choices=("native", "fixed_length"), help="merge: the variant to merge")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4, help="videos per VAE call")
    parser.add_argument("--workers", type=int, default=32, help="processes that render or decode the videos")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0,
                        help="encode only the first (shortest) N rows of the shard, for a quick test")
    args = parser.parse_args()
    if args.set == "hard_train" and args.task not in ("flowfree", "sokoban"):
        parser.error("--set hard_train exists for flowfree and sokoban only")
    if args.stage == "lengths":
        stage_lengths(args.task, args.set)
    elif args.stage == "encode":
        variants = tuple(args.variants.split(","))
        if not set(variants) <= {"native", "fixed_length"}:
            parser.error(f"unknown variants {variants}")
        stage_encode(args.task, args.set, args.shard, args.shards, args.batch, args.workers, args.device, variants, args.limit)
    else:
        if args.variant is None:
            parser.error("merge needs --variant")
        stage_merge(args.task, args.set, args.variant)


if __name__ == "__main__":
    main()
