"""Export a Wan2.2-TI2V-5B training checkpoint as a release model folder (config.json + model.safetensors).

A training checkpoint (``checkpoint_model_<step>/model.pt``) holds the full training state (fp32 weights, AdamW
moments, scheduler and RNG states, about 60 GB). Inference only needs the generator weights; the evaluation casts
them to bf16, so a bf16 export (about 10 GB) gives the same videos.

    python scripts/export_model.py --run-dir outputs/cft/sokoban \\
        --checkpoint outputs/cft/sokoban/checkpoint_model_053700/model.pt --out checkpoints/videcide/sokoban_cft

The folder works with demo.py, scripts/eval/eval_causal.py (--run-dir) and the post-training collectors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Keys of the run config that inference needs (see scripts/eval/eval_causal.py and the inference pipeline).
INFERENCE_KEYS = ("causal", "num_frame_per_block", "independent_first_frame", "i2v", "num_train_timestep",
                  "timestep_shift", "guidance_scale", "negative_prompt", "frame_buckets", "image_or_video_shape",
                  "teacher_forcing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="training run (resolved_config.json)")
    parser.add_argument("--checkpoint", type=Path, help="default: the final checkpoint of the run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--base-model", default="checkpoints/Wan2.2-TI2V-5B",
                        help="model_kwargs.model_name written to config.json (Wan2.2 text encoder, VAE, DiT config)")
    parser.add_argument("--name", help="model name recorded in config.json")
    args = parser.parse_args()

    import torch
    from safetensors.torch import save_file
    from videcide.generation.common import default_checkpoint, load_generator_state_dict, load_run_config

    config = load_run_config(args.run_dir)
    checkpoint = args.checkpoint or default_checkpoint(args.run_dir, config)
    state = load_generator_state_dict(checkpoint)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tensors = {k: v.detach().to(dtype).contiguous() for k, v in state.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "model.safetensors"), metadata={"format": "pt"})
    resolved = json.loads((args.run_dir / "resolved_config.json").read_text())
    exported = {k: resolved[k] for k in INFERENCE_KEYS if k in resolved}
    exported["model_kwargs"] = {**resolved["model_kwargs"], "model_name": args.base_model}
    exported["guidance_scale"] = 1.0
    exported.update(name=args.name or args.out.name, task=resolved.get("task"), dtype=args.dtype,
                    training_steps=int(resolved["max_iters"]),
                    training_shift=float(resolved["model_kwargs"]["timestep_shift"]),
                    rft_mode=resolved.get("rft_mode"),
                    source_checkpoint_step=_step(checkpoint))
    (args.out / "config.json").write_text(json.dumps(exported, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    size = sum(t.numel() * t.element_size() for t in tensors.values()) / 2**30
    print(f"wrote {args.out}/model.safetensors ({len(tensors)} tensors, {size:.1f} GiB) and config.json")


def _step(checkpoint: Path) -> int | None:
    name = Path(checkpoint).parent.name
    return int(name.rsplit("_", 1)[1]) if name.startswith("checkpoint_model_") else None


if __name__ == "__main__":
    main()
