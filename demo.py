"""Solve one puzzle with a released ViDecide model and check the video with the rule verifier.

The model generates the solution segment by segment from the first frame and the task prompt, and stops at its own
visual EOS frame (a frame whose border is at least 70% red) within 20 s. FlowFree, Sokoban and Maze videos are then
replayed by the strict rule verifier.

Examples (run from the repository root):
    python demo.py --model checkpoints/videcide/sokoban_cft --example sokoban_test_00756    # 3 boxes, 39 moves
    python demo.py --model checkpoints/videcide/flowfree_cft --example flowfree_test_00005  # 8x8, 5 colors
    python demo.py --model checkpoints/videcide/sokoban_cft --task sokoban \\
        --first-frame my_puzzle.png --source my_puzzle.json
    python demo.py --example sokoban_test_00756 --verify outputs/demo/sokoban_test_00756.mp4   # verifier only (CPU)
    python demo.py --task maze --source data/VideoRLVR/Maze/sources/test/maze_test_00000.json \\
        --verify data/VideoRLVR/Maze/videos/test/maze_test_00000.mp4                  # any video of a puzzle

--model is a released model folder (config.json + model.safetensors) or a training run directory together with
--checkpoint. Generation needs one GPU and the Wan2.2-TI2V-5B text encoder and VAE in checkpoints/ (see README).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from videcide.tasks.video import FPS, read_mp4, write_mp4  # noqa: E402
from videcide.tasks.video_parser import find_eos_frame, parse_video  # noqa: E402

EXAMPLES = ROOT / "assets" / "examples" / "examples.jsonl"
VERIFIED_TASKS = ("flowfree", "sokoban", "maze")
MAX_FRAMES = 320          # 20 s at 16 fps: the EOS frame must appear within these frames


def load_example(name: str) -> dict:
    rows = {r["example"]: r for r in map(json.loads, EXAMPLES.read_text().splitlines())}
    if name not in rows:
        raise SystemExit(f"unknown example {name!r}; available: {', '.join(sorted(rows))}")
    row = dict(rows[name])
    row["first_frame"] = ROOT / row["first_frame"]
    row["source_path"] = ROOT / row["source_path"]
    return row


def read_first_frame(path: Path) -> np.ndarray:
    """First frame of a video, or an image, as RGB uint8."""
    import cv2

    if path.suffix.lower() in (".mp4", ".avi", ".mov", ".webm"):
        return read_mp4(path)[0]
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def summarize(frames: np.ndarray, task: str, source: dict | None) -> dict:
    eos = find_eos_frame(frames)
    summary = {"frames": int(len(frames)), "eos_frame": eos,
               "eos_time_s": None if eos is None else round(eos / FPS, 2)}
    if task in VERIFIED_TASKS and source is not None:
        verdict = parse_video(frames, source, task)
        summary["strict_success"] = bool(verdict["success"])
        if task == "flowfree":
            summary.update(outcome=verdict.get("outcome"), failure_reason=verdict.get("failure_reason"),
                           first_failure_frame=verdict.get("first_failure_frame"))
        elif task == "maze" and not verdict["success"]:
            extra = int(verdict.get("parsed_cells") or 0) - int(verdict.get("reference_cells") or 0)
            summary["failure_reason"] = (
                {"trail_on_wall": "path drawn on a wall", "not_connected_to_start": "drawing not connected to the start"}
                .get(verdict.get("process_violation")) or
                ("final path on a wall" if not verdict.get("on_corridor") else
                 "final drawing not connected" if not verdict.get("connected") else
                 f"final drawing has {extra} cells beyond the solution path" if not verdict.get("simple_path") and extra > 2 else
                 "final path does not join the start and the goal" if not verdict.get("simple_path") else
                 "no EOS frame"))
        elif not verdict["success"]:
            summary["failure_reason"] = (
                "hallucinated or vanished object" if verdict.get("hallucination") or verdict.get("goal_vanished") else
                "illegal move or push" if verdict.get("illegal_transitions") or verdict.get("pulls") else
                "goal not reached" if not verdict.get("solved") else "no EOS frame")
    else:
        summary["strict_success"] = None
        summary["note"] = ("score VBVR-MultiStep videos with scripts/eval/score_vbvr.py and RoboTwin-Plan videos with "
                           "scripts/robotwin/verifier.py")
    return summary


def generate(args, prompt: str, first_rgb: np.ndarray, source: dict | None, row: dict) -> np.ndarray:
    import cv2
    import torch
    from types import SimpleNamespace
    from omegaconf import OmegaConf

    from videcide.generation import wan_setup
    from videcide.generation.common import default_checkpoint, load_generator_state_dict, load_run_config, row_seed
    from videcide.generation.rollout_engine import RolloutEngine, RolloutJob

    config = load_run_config(args.model)
    config.guidance_scale = float(args.guidance)
    if not bool(config.causal):
        raise SystemExit("demo.py runs causal models; use scripts/eval/eval_bidirectional.py for bidirectional ones")
    block = int(config.num_frame_per_block)
    generated = args.max_latents - 1
    if generated % block:
        generated += block - generated % block
    checkpoint = args.checkpoint or default_checkpoint(args.model, config)
    device, dtype = torch.device("cuda"), torch.bfloat16

    began = time.time()
    generator = wan_setup.WanDiffusionWrapper(**OmegaConf.to_container(config.model_kwargs, resolve=True),
                                              is_causal=True)
    missing = generator.load_state_dict(load_generator_state_dict(checkpoint), strict=True)
    if missing.missing_keys or missing.unexpected_keys:
        raise RuntimeError(str(missing))
    generator = generator.to(device=device, dtype=dtype).eval()
    model_name = str(config.model_kwargs.model_name)
    text_encoder = wan_setup.WanTextEncoder(model_name=model_name).to(device=device, dtype=dtype).eval()
    vae = wan_setup.WanVAEWrapper(model_name=model_name).to(device=device, dtype=dtype).eval()
    pipe = wan_setup.CausalDiffusionInferencePipeline(config, device=str(device), generator=generator,
                                                      text_encoder=text_encoder, vae=vae)
    pipe.frame_seq_length = 144          # 24 x 24 latent tokens per frame at 384 x 384
    pipe.sampling_steps = args.steps
    pipe.disable_tqdm = True
    print(f"[demo] model loaded in {time.time() - began:.0f}s from {checkpoint}", flush=True)

    resized = cv2.resize(first_rgb, (384, 384), interpolation=cv2.INTER_AREA)
    pixels = torch.from_numpy(resized.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None, :, None]
    with torch.no_grad():
        first = vae.encode_to_latent(pixels.to(device=device, dtype=dtype))[0].cpu()

    def eos_only(frames, _source, include_trace=False):
        return {"outcome": "incomplete", "eos_detected": find_eos_frame(frames[:MAX_FRAMES]) is not None}

    runner = SimpleNamespace(pipeline=pipe, trainer=SimpleNamespace(device=device, dtype=dtype), generators=None)
    engine = RolloutEngine(runner, stop_on="eos", evaluate=eos_only, block=block)
    seed = row_seed(args.seed, row)       # the examples keep the sampling noise of the paper's evaluation
    job = RolloutJob(key=row["sample_id"], history=first, prompt=prompt, source=source or {}, seed=seed, noise_seed=seed,
                     noise_latents=generated, max_latents=1 + generated)
    began = time.time()
    result = engine.run_batch([job])[0]
    print(f"[demo] generated {int(result.latents.shape[0])} latent frames in {time.time() - began:.1f}s "
          f"({'EOS' if result.stop == 'terminal' else 'length limit'})", flush=True)
    return result.frames[:MAX_FRAMES]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, help="released model folder or training run directory")
    parser.add_argument("--checkpoint", type=Path, help="weights (default: <model>/model.safetensors)")
    parser.add_argument("--example", help="name of a puzzle in assets/examples/examples.jsonl")
    parser.add_argument("--task", choices=("flowfree", "sokoban", "maze", "vbvr", "robotwin"))
    parser.add_argument("--first-frame", type=Path, help="initial observation o_0 (image or video)")
    parser.add_argument("--source", type=Path, help="puzzle JSON for the FlowFree / Sokoban / Maze verifier")
    parser.add_argument("--prompt", help="task prompt (default: the example's or the source's prompt)")
    parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--steps", type=int, default=5, help="denoising steps per segment (paper: 5)")
    parser.add_argument("--guidance", type=float, default=1.0, help="CFG scale (paper: 1.0, no guidance)")
    parser.add_argument("--seed", type=int, default=20261025, help="base seed (paper protocol: 20261025)")
    parser.add_argument("--max-latents", type=int, default=80, help="latent frames including o_0 (80 = 20 s)")
    parser.add_argument("--verify", type=Path, help="only run the verifier on this video (no GPU needed)")
    args = parser.parse_args()

    if args.example:
        row = load_example(args.example)
        task, sample_id = row["task"], row["sample_id"]
        first_frame, source_path, prompt = row["first_frame"], row["source_path"], row["prompt"]
    else:
        if not (args.task and (args.first_frame or args.verify)):
            parser.error("give --example, or --task and --first-frame (--task, --source and --verify to only verify)")
        task, first_frame, source_path = args.task, args.first_frame, args.source
        sample_id = (first_frame or args.verify).stem
        row = {"sample_id": sample_id}
        prompt = None
    source = json.loads(Path(source_path).read_text()) if source_path else None
    prompt = args.prompt or prompt or (source or {}).get("prompt")

    if args.verify:
        frames = read_mp4(args.verify)
    else:
        if args.model is None:
            parser.error("--model is required for generation")
        if not prompt:
            parser.error("no prompt: pass --prompt")
        frames = generate(args, prompt, read_first_frame(Path(first_frame)), source, row)
        eos = find_eos_frame(frames)
        video = frames if eos is None else frames[:eos + 1]   # the video ends at the model's own EOS frame
        path = args.out / f"{args.example or sample_id}.mp4"
        write_mp4(path, video)
        print(f"[demo] video: {path}", flush=True)
    summary = summarize(frames, task, source)
    print(json.dumps({"example": args.example, "task": task, "sample_id": sample_id, **summary}, indent=1))


if __name__ == "__main__":
    main()
