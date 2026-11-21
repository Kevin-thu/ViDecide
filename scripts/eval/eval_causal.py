"""Evaluate a causal model (K=1 or K=8) on a test set: generate until the EOS frame and apply the verifier.

Protocol of the paper: 5 sampling steps per segment, no classifier-free guidance, seed 20261025, and the EOS frame must
appear within the first 320 frames (20 s at 16 fps); videos are cut to these frames. FlowFree, Sokoban and Maze are
judged here; VBVR-MultiStep and RoboTwin-Plan videos are written for scripts/eval/score_vbvr.py and
scripts/robotwin/verifier.py.

Usage:
    python scripts/eval/eval_causal.py --run-dir checkpoints/videcide/sokoban_cft --task sokoban --set hard_test \
        --output outputs/eval/sokoban_cft/hard_test
``--set`` names a manifest of the task in data/ (test, hard_test, ood_*); ``--eval-manifest`` takes any manifest
instead. ``--run-dir`` is a released model folder (config.json + model.safetensors) or a training run
(resolved_config.json; its final checkpoint unless --checkpoint is given).
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from videcide.generation.common import (default_checkpoint, load_generator_state_dict, load_run_config, read_jsonl,
                                        row_seed, write_json)
from videcide.paths import DATA_ROOT, manifest_path
from videcide.generation.rollout_engine import RolloutEngine, RolloutJob
from videcide.tasks.video import write_mp4
from videcide.tasks.video_parser import find_eos_frame, parse_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        help="default: <run-dir>/model.safetensors, or the final checkpoint of a training run")
    parser.add_argument("--task", required=True, choices=("flowfree", "sokoban", "maze", "vbvr", "robotwin"),
                        help="vbvr / robotwin: generation only (scored by score_vbvr.py / robotwin/verifier.py)")
    parser.add_argument("--set", default="test", help="manifest of the task in data/: test, hard_test, ood_* (default: test)")
    parser.add_argument("--eval-manifest", type=Path, help="evaluate the rows of this manifest instead of --set")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="root of the paths in the manifest (default: data)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampling-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--generated-latents", type=int, default=79, help="latent frames generated after o_0 (at most)")
    parser.add_argument("--max-frames", type=int, default=320, help="frames within which the EOS frame must appear (20 s)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0, help="this process takes rows with index %% shards == shard")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--guidance", type=float, default=1.0, help="CFG scale (1.0 = no guidance, as in the paper)")
    parser.add_argument("--context-sigma", type=float, default=0.0,
                        help="re-noise the generated history to this level in the KV cache (0.1 for the diffusion-forcing models of Tab. S7)")
    args = parser.parse_args()
    from videcide.generation import wan_setup   # needs the Causal-Forcing checkout and a GPU

    from omegaconf import OmegaConf

    config = load_run_config(args.run_dir)
    config.guidance_scale = float(args.guidance)
    if args.checkpoint is None:
        args.checkpoint = default_checkpoint(args.run_dir, config)
    if not bool(config.causal):
        raise ValueError("eval_causal handles causal models only")
    block = int(config.num_frame_per_block)
    if (args.generated_latents % block) and block > 1:
        args.generated_latents += block - args.generated_latents % block
    if block > 1 and not bool(config.get("independent_first_frame", True)):
        raise ValueError("block models must have an independent first frame (history = the given first latent)")
    dtype = torch.bfloat16
    device = torch.device("cuda")
    if os.environ.get("VIDECIDE_GPU_MEMORY_GIB"):
        budget = float(os.environ["VIDECIDE_GPU_MEMORY_GIB"]) * 2**30
        torch.cuda.set_per_process_memory_fraction(budget / torch.cuda.get_device_properties(device).total_memory)
    generator = wan_setup.WanDiffusionWrapper(**OmegaConf.to_container(config.model_kwargs, resolve=True), is_causal=True)
    state = load_generator_state_dict(args.checkpoint)
    incompat = generator.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(str(incompat))
    del state
    generator = generator.to(device=device, dtype=dtype).eval()
    model_name = str(config.model_kwargs.model_name)
    text_encoder = wan_setup.WanTextEncoder(model_name=model_name).to(device=device, dtype=dtype).eval()
    vae = wan_setup.WanVAEWrapper(model_name=model_name).to(device=device, dtype=dtype).eval()
    pipe = wan_setup.CausalDiffusionInferencePipeline(config, device=str(device), generator=generator,
                                                   text_encoder=text_encoder, vae=vae)
    pipe.frame_seq_length = 144
    pipe.sampling_steps = args.sampling_steps
    pipe.disable_tqdm = True
    runner = SimpleNamespace(pipeline=pipe, trainer=SimpleNamespace(device=device, dtype=dtype), generators=None)

    def eos_only(frames, source, include_trace=False):
        return {"outcome": "incomplete", "eos_detected": find_eos_frame(frames[:args.max_frames]) is not None}

    engine = RolloutEngine(runner, stop_on="eos", evaluate=eos_only, block=block)
    engine.context_sigma = float(args.context_sigma)
    vbvr = args.task in ("vbvr", "robotwin")
    prefix = {"vbvr": "vbvr_", "robotwin": "robotwin_"}.get(args.task, "")
    manifest = args.eval_manifest or manifest_path(args.task, args.set)
    rows = [r for r in read_jsonl(manifest) if (r["task"].startswith(prefix) if vbvr else r["task"] == args.task)]
    if args.limit:
        rows = rows[:args.limit]
    rows = [row for index, row in enumerate(rows) if index % args.shards == args.shard]
    jobs = []
    for index, row in enumerate(rows):
        _, pixels = wan_setup.first_frame_tensor(args.data_root / row["native_video"], 384, str(device), dtype)
        with torch.no_grad():
            first = vae.encode_to_latent(pixels)[0].cpu()
        seed = row_seed(args.seed, row)
        jobs.append(RolloutJob(key=str(row["sample_id"]), history=first, prompt=str(row["prompt"]),
                               source={} if vbvr else json.loads((args.data_root / row["source_path"]).read_text()),
                               seed=seed, noise_seed=seed, noise_latents=args.generated_latents,
                               max_latents=1 + args.generated_latents, meta={"row": row, "index": index}))
    (args.output / "videos").mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=16)
    results, pending = [], []
    began = time.time()

    def finish(result):
        row, index = result.job.meta["row"], result.job.meta["index"]
        frames = result.frames[:args.max_frames]
        path = args.output / "videos" / f"{index:04d}_{row['sample_id']}.mp4"
        write_mp4(path, frames)
        if vbvr:
            metrics = {"success": False, "task_success": False, "outcome": "unscored",
                       "eos_detected": find_eos_frame(frames) is not None}
        else:
            metrics = parse_video(frames, result.job.source, args.task)
        return {"sample_id": row["sample_id"], "task": row["task"], "split": row.get("split", "test"),
                "group_index": 0, "seed": result.job.seed, "mp4": str(path), "generated_frames": len(frames),
                "generated_latent_frames": int(result.latents.shape[0]), "stopped_early": result.stop == "terminal",
                **metrics}

    def on_result(result):
        pending.append(pool.submit(finish, result))
        done = len(pending)
        if done % 50 == 0:
            print(f"[eval] {done}/{len(jobs)} {done / (time.time() - began) * 3600:.0f}/h", flush=True)

    engine.run(jobs, batch_size=args.batch, on_result=on_result)
    results = [f.result() for f in pending]
    results.sort(key=lambda r: r["sample_id"])
    if vbvr:   # scored afterwards
        scorer = "scripts/eval/score_vbvr.py" if args.task == "vbvr" else "scripts/robotwin/verifier.py"
        summary = {args.task: {"n": len(results), "eos": sum(bool(r["eos_detected"]) for r in results),
                               "strict_success": f"not scored here: run {scorer}"}}
    else:
        summary = {args.task: {"n": len(results), "strict_success": sum(bool(r["success"]) for r in results),
                               "task_success": sum(bool(r["task_success"]) for r in results),
                               "eos": sum(bool(r["eos_detected"]) for r in results)}}
    write_json(args.output / "results.json", {
        "run_dir": str(args.run_dir), "checkpoint": str(args.checkpoint), "manifest": str(manifest), "causal": True,
        "num_frame_per_block": block, "sampling_steps": args.sampling_steps, "seed": args.seed,
        "context_sigma": float(args.context_sigma), "cfg": float(config.guidance_scale), "max_frames": args.max_frames,
        "seconds": round(time.time() - began, 1), "engine_stats": engine.stats, "summary": summary, "results": results})
    print(json.dumps(summary), f"{time.time() - began:.0f}s", flush=True)


if __name__ == "__main__":
    main()
