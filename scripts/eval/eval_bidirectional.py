"""Evaluate a fixed-length bidirectional model: denoise the whole clip jointly, then apply the verifier.

Protocol of the paper: 5 UniPC steps, no classifier-free guidance, seed 20261025; the clip length is the training
length of the model (RoboTwin-Plan: the length of each sub-task, from the lengths.json written by
scripts/benchmarks/build_latents.py, --durations-json). FlowFree, Sokoban and Maze are judged here; VBVR-MultiStep and
RoboTwin-Plan videos are scored with scripts/eval/score_vbvr.py and scripts/robotwin/verifier.py.

Usage:
    python scripts/eval/eval_bidirectional.py --run-dir outputs/sft/sokoban_bidir_stage2 --task sokoban \
        --output outputs/eval/sokoban_bidir
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from videcide.generation.common import (default_checkpoint, load_generator_state_dict, load_run_config, read_jsonl,
                                        row_seed, write_json)
from videcide.paths import DATA_ROOT, manifest_path
from videcide.tasks.video import write_mp4
from videcide.tasks.video_parser import find_eos_frame, parse_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="default: the final checkpoint of the run")
    parser.add_argument("--task", required=True, choices=("flowfree", "sokoban", "maze", "vbvr", "robotwin"),
                        help="vbvr / robotwin: generation only (scored by score_vbvr.py / robotwin/verifier.py)")
    parser.add_argument("--set", default="test", help="manifest of the task in data/: test, hard_test, ood_* (default: test)")
    parser.add_argument("--eval-manifest", type=Path, help="evaluate the rows of this manifest instead of --set")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="root of the paths in the manifest (default: data)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampling-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-pixel-frames", type=int, default=317)
    parser.add_argument("--no-parse", action="store_true")
    parser.add_argument('--durations-json',type=Path,help='per-subtask fixed lengths of the RoboTwin-Plan model (lengths json of build_latents.py)')
    parser.add_argument('--guidance', type=float, default=1.0, help='CFG scale (1.0 = no guidance, as in the paper)')
    parser.add_argument('--length-latents', type=int, default=0, help='generate this many latents instead of the trained clip length')
    parser.add_argument('--chain', type=int, default=1, help='up to this many clips: a clip without an EOS frame is continued '
                        'from its last frame (re-encoded as the next first frame, fresh noise)')
    args = parser.parse_args()
    from videcide.generation import wan_setup   # needs the Causal-Forcing checkout and a GPU

    from omegaconf import OmegaConf
    config = load_run_config(args.run_dir)
    config.guidance_scale = float(args.guidance)
    if bool(config.causal):
        raise ValueError("eval_bidirectional handles bidirectional runs only")
    if args.checkpoint is None:
        args.checkpoint = default_checkpoint(args.run_dir, config)
    dtype, device = torch.bfloat16, torch.device("cuda")
    if os.environ.get("VIDECIDE_GPU_MEMORY_GIB"):
        budget = float(os.environ["VIDECIDE_GPU_MEMORY_GIB"]) * 2**30
        torch.cuda.set_per_process_memory_fraction(budget / torch.cuda.get_device_properties(device).total_memory)
    generator = wan_setup.WanDiffusionWrapper(**OmegaConf.to_container(config.model_kwargs, resolve=True), is_causal=False)
    generator.seq_len = 0
    state = load_generator_state_dict(args.checkpoint)
    incompat = generator.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(str(incompat))
    del state
    generator = generator.to(device=device, dtype=dtype).eval()
    model_name = str(config.model_kwargs.model_name)
    text_encoder = wan_setup.WanTextEncoder(model_name=model_name).to(device=device, dtype=dtype).eval()
    vae = wan_setup.WanVAEWrapper(model_name=model_name).to(device=device, dtype=dtype).eval()
    length = args.length_latents or int(config.image_or_video_shape[1])
    durations=(json.loads(args.durations_json.read_text())['summary']['durations_by_subtask'] if args.durations_json else {})

    vbvr = args.task in ("vbvr", "robotwin")
    prefix = {"vbvr": "vbvr_", "robotwin": "robotwin_"}.get(args.task, "")
    manifest = args.eval_manifest or manifest_path(args.task, args.set)
    rows = [r for r in read_jsonl(manifest) if (r["task"].startswith(prefix) if vbvr else r["task"] == args.task)]
    if args.limit:
        rows = rows[:args.limit]
    rows = [row for index, row in enumerate(rows) if index % args.shards == args.shard]
    if durations:rows.sort(key=lambda r:(r['subtask'],r['sample_id']))
    (args.output / "videos").mkdir(parents=True, exist_ok=True)
    unconditional = {k: v.detach() for k, v in text_encoder([config.negative_prompt]).items()}
    pool = ThreadPoolExecutor(max_workers=16)
    pending, began = [], time.time()

    def finish(index, row, frames, seed, source, clip_length):
        path = args.output / "videos" / f"{index:04d}_{row['sample_id']}.mp4"
        write_mp4(path, frames)
        if vbvr:
            metrics = {"success": False, "task_success": False, "outcome": "unscored",
                       "eos_detected": find_eos_frame(frames) is not None}
        else:
            metrics = ({"success": False, "task_success": False, "eos_detected": False, "unscored": True}
                       if args.no_parse else parse_video(frames, source, args.task))
        return {"sample_id": row["sample_id"], "task": row["task"], "split": row.get("split", "test"),
                "group_index": 0, "seed": seed, "mp4": str(path), "generated_frames": len(frames),
                "generated_latent_frames": clip_length, **metrics}

    chunks=[]
    for row in rows:
        if not chunks or len(chunks[-1])>=args.batch or (durations and chunks[-1][0]['subtask']!=row['subtask']):chunks.append([])
        chunks[-1].append(row)
    start=0
    for chunk in chunks:
        if durations and not args.length_latents:length=(durations[chunk[0]['subtask']]['freeze_frames']-1)//4+1
        firsts, noises, seeds, conds = [], [], [], []
        for row in chunk:
            _, pixels = wan_setup.first_frame_tensor(args.data_root / row["native_video"], 384, str(device), dtype)
            with torch.no_grad():
                firsts.append(vae.encode_to_latent(pixels))
            seed = row_seed(args.seed, row)
            rng = torch.Generator(device=device).manual_seed(seed)
            noises.append(torch.randn((1, length, 48, 24, 24), generator=rng, device=device, dtype=dtype))
            seeds.append(seed)
            conds.append(text_encoder([str(row["prompt"])]))
        batch = len(chunk)
        single = float(config.guidance_scale) == 1.0
        cond = {k: torch.cat([c[k] for c in conds] + ([] if single else [unconditional[k].expand(batch, *unconditional[k].shape[1:])]))
                for k in conds[0]}

        def denoise(initial, latents):
            latents[:, 0] = initial[:, 0]
            scheduler = wan_setup.FlowUniPCMultistepScheduler(num_train_timesteps=int(config.num_train_timestep),
                                                           shift=1, use_dynamic_shifting=False)
            scheduler.set_timesteps(args.sampling_steps, device=device, shift=float(config.timestep_shift))
            with torch.no_grad():
                for value in scheduler.timesteps:
                    if single:
                        timesteps = value * torch.ones((batch, length), device=device, dtype=torch.float32)
                        guided, _ = generator(latents, cond, timesteps)
                    else:
                        timesteps = value * torch.ones((2 * batch, length), device=device, dtype=torch.float32)
                        flow, _ = generator(torch.cat([latents, latents]), cond, timesteps)
                        flow_cond, flow_uncond = flow.chunk(2)
                        guided = flow_uncond + float(config.guidance_scale) * (flow_cond - flow_uncond)
                    latents = scheduler.step(guided, value, latents, return_dict=False)[0]
                    latents[:, 0] = initial[:, 0]
            return latents

        def decode(latents):
            with torch.no_grad():
                return [wan_setup.to_rgb_frames((vae.decode_to_pixel(latents[i:i + 1]) * 0.5 + 0.5).clamp(0, 1), 10 ** 6)
                        for i in range(latents.shape[0])]

        all_frames = decode(denoise(torch.cat(firsts), torch.cat(noises).clone()))
        for seg in range(1, args.chain):
            if all(find_eos_frame(f) is not None for f in all_frames):
                break
            nxt, nz = [], []
            for i, f in enumerate(all_frames):
                last = torch.from_numpy(f[-1].astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None, :, None]
                with torch.no_grad():
                    nxt.append(vae.encode_to_latent(last.to(device=device, dtype=dtype)))
                rng = torch.Generator(device=device).manual_seed(seeds[i] + 1_000_003 * seg)
                nz.append(torch.randn((1, length, 48, 24, 24), generator=rng, device=device, dtype=dtype))
            more = decode(denoise(torch.cat(nxt), torch.cat(nz)))
            all_frames = [f if find_eos_frame(f) is not None else np.concatenate([f, m[1:]]) for f, m in zip(all_frames, more)]
        for i, row in enumerate(chunk):
            frames = all_frames[i][:args.max_pixel_frames]
            source = {} if vbvr else json.loads((args.data_root / row["source_path"]).read_text())
            pending.append(pool.submit(finish, start + i, row, frames, seeds[i], source,length))
        done = start + batch
        print(f"[eval] {done}/{len(rows)} {done / (time.time() - began) * 3600:.0f}/h", flush=True)
        start=done

    results = sorted((f.result() for f in pending), key=lambda r: r["sample_id"])
    summary = {args.task: {"n": len(results), "strict_success": sum(bool(r["success"]) for r in results),
                           "task_success": sum(bool(r["task_success"]) for r in results),
                           "eos": sum(bool(r["eos_detected"]) for r in results)}}
    write_json(args.output / "results.json", {
        "run_dir": str(args.run_dir), "checkpoint": str(args.checkpoint), "manifest": str(manifest), "causal": False,
        "sampling_steps": args.sampling_steps, "cfg": float(config.guidance_scale), "sampler": f"batched bidirectional UniPC, batch {args.batch}, " + ("conditional only (CFG 1)" if float(config.guidance_scale) == 1.0 else "CFG in one 2B forward"),
        "clip_latents": length, "seconds": round(time.time() - began, 1), "summary": summary, "results": results})
    print(json.dumps(summary), f"{time.time() - began:.0f}s", flush=True)


if __name__ == "__main__":
    main()
