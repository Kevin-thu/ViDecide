from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import cv2
import imageio.v2 as imageio
import numpy as np


FPS = 16
CANVAS_SIZE = 384
EOS_COLOR = np.array([230, 30, 30], dtype=np.uint8)


def nearest_vae_length(frame_count: float) -> int:
    scaled = (float(frame_count) - 1.0) / 4.0
    k = max(1, math.floor(scaled + 0.5))
    return 4 * k + 1


def pad_to_vae_length(frames: np.ndarray) -> np.ndarray:
    target = nearest_vae_length(len(frames))
    if target < len(frames):
        target += 4
    if target == len(frames):
        return frames
    pad = np.repeat(frames[-1][None], target - len(frames), axis=0)
    return np.concatenate([frames, pad], axis=0)


def add_eos_border(
    frames: np.ndarray,
    duration: int = 4,
    width: int = 10,
) -> np.ndarray:
    out = frames.copy()
    for frame in out[-min(duration, len(out)):]:
        frame[:width, :, :] = EOS_COLOR
        frame[-width:, :, :] = EOS_COLOR
        frame[:, :width, :] = EOS_COLOR
        frame[:, -width:, :] = EOS_COLOR
    return out


def timewarp_frames(frames: np.ndarray, target_frames: int) -> np.ndarray:
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected F,H,W,3 frames, got {frames.shape}")
    indices = np.rint(np.linspace(0, len(frames) - 1, target_frames)).astype(int)
    return frames[indices]


def write_mp4(
    path: Path,
    frames: np.ndarray | Iterable[np.ndarray],
    fps: int = FPS,
    *,
    crf: int = 18,
) -> int:
    array = np.asarray(list(frames) if not isinstance(frames, np.ndarray) else frames)
    if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"expected uint8 F,H,W,3 array, got {array.dtype} {array.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=None,
        macro_block_size=16,
        pixelformat="yuv420p",
        output_params=["-crf", str(crf), "-movflags", "+faststart"],
    )
    try:
        for frame in array:
            writer.append_data(frame)
    finally:
        writer.close()
    return len(array)


def read_mp4(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"no decodable frames: {path}")
    return np.stack(frames)


def read_last_bgr(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, count - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot decode final frame: {path}")
    return frame
