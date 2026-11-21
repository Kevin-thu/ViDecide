"""Wan2.2-TI2V-5B components from the patched Causal-Forcing checkout, and frame helpers for inference."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "CAUSAL_FLEX_ATTENTION_COMPILE_MODE", "max-autotune-no-cudagraphs"
)
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import cv2
import numpy as np
import torch


if not hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config._config["recompile_limit"] = (
        torch._dynamo.config._config["cache_size_limit"]
    )
if not hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
    torch._dynamo.config._config["accumulated_recompile_limit"] = (
        torch._dynamo.config._config["accumulated_cache_size_limit"]
    )

from videcide.paths import add_causal_forcing_to_path

REFERENCE_ROOT = add_causal_forcing_to_path()

from pipeline.causal_diffusion_inference import (
    CausalDiffusionInferencePipeline,
)
from utils.wan_wrapper import (
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from videcide.tasks.video import read_mp4

__all__ = ["CausalDiffusionInferencePipeline", "FlowUniPCMultistepScheduler", "WanDiffusionWrapper", "WanTextEncoder",
           "WanVAEWrapper", "first_frame_tensor", "to_rgb_frames"]


def first_frame_tensor(
    path: Path, resolution: int, device: str, dtype: torch.dtype
) -> tuple[np.ndarray, torch.Tensor]:
    frame = read_mp4(path)[0]
    resized = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 127.5 - 1.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    return frame, tensor.to(device=device, dtype=dtype)


def to_rgb_frames(video: torch.Tensor, max_frames: int) -> np.ndarray:
    array = (
        video[0].detach().float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        * 255.0
    ).round().astype(np.uint8)
    if array.shape[1:3] != (384, 384):
        array = np.stack(
            [cv2.resize(frame, (384, 384), interpolation=cv2.INTER_CUBIC) for frame in array]
        )
    return array[:max_frames]
