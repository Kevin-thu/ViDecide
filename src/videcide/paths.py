"""Locations shared by the scripts.

All scripts are meant to be run from the repository root, where ``data/``, ``checkpoints/`` and ``outputs/`` live.
``data/`` (or ``$VIDECIDE_DATA``) holds the released dataset with its three benchmark folders::

    data/VideoRLVR/{FlowFree,Sokoban,Maze}/   <set>.jsonl, videos/<set>/, sources/<set>/ (FlowFree: traces/<set>/)
    data/VBVR-MultiStep/                      <set>.jsonl, videos/<task>/, sources/<task>/, verifier/
    data/RoboTwin-Plan/                       <set>.jsonl, episodes/<task>/<seed>/
    data/latents/<task>/<variant>/            training latents built with scripts/benchmarks/build_latents.py

Every path stored in the manifests and source files is relative to the data root. The patched Causal-Forcing
checkout is created by ``scripts/setup/setup_causal_forcing.sh`` in ``third_party/Causal-Forcing``; set
``CAUSAL_FORCING_ROOT`` to use a checkout elsewhere.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("VIDECIDE_DATA", "data"))

# Benchmark folder of each task name used by the scripts (--task) and by the training configs.
BENCHMARKS = {"flowfree": "VideoRLVR/FlowFree", "sokoban": "VideoRLVR/Sokoban", "maze": "VideoRLVR/Maze",
              "vbvr": "VBVR-MultiStep", "robotwin": "RoboTwin-Plan"}


def data_path(path: str | Path) -> Path:
    """Resolve a path stored in a dataset file.

    The released datasets store paths relative to the data root (``data/`` or ``$VIDECIDE_DATA``); a path that already
    exists as written (absolute, or relative to the working directory) is returned unchanged.
    """
    path = Path(path)
    return path if path.is_absolute() or path.exists() else DATA_ROOT / path


def benchmark_dir(task: str) -> Path:
    """Folder of a benchmark in the data root, e.g. ``data/VideoRLVR/Sokoban`` for ``sokoban``."""
    if task not in BENCHMARKS:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(BENCHMARKS)}")
    return DATA_ROOT / BENCHMARKS[task]


def manifest_path(task: str, name: str = "test") -> Path:
    """Manifest of one set, e.g. ``manifest_path("sokoban", "hard_test")`` -> data/VideoRLVR/Sokoban/hard_test.jsonl.

    Sets: train, val, test for every benchmark; hard_train (the 5,000 harder post-training puzzles) and hard_test
    (FlowFree-Hard, Sokoban-Hard) for FlowFree and Sokoban; ood_* for the out-of-distribution sets.
    """
    return benchmark_dir(task) / f"{name}.jsonl"


def read_manifest(path: str | Path) -> list[dict]:
    """Rows of a JSONL manifest."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def latents_dir(task: str, variant: str = "native") -> Path:
    """Training latents of a task: ``data/latents/<task>/<variant>`` with ``lmdb/`` and ``metadata.json``.

    ``native`` keeps the native duration of every video (causal models); ``fixed_length`` holds the last frame up to
    a fixed length per task (bidirectional baseline).
    """
    return DATA_ROOT / "latents" / task / variant


def causal_forcing_root() -> Path:
    """Return the patched Causal-Forcing checkout (``$CAUSAL_FORCING_ROOT`` or ``third_party/Causal-Forcing``)."""
    return Path(os.environ.get("CAUSAL_FORCING_ROOT", REPO_ROOT / "third_party" / "Causal-Forcing")).resolve()


def _torch_compat() -> None:
    """The patched modules use the PyTorch >= 2.7 names of two Dynamo limits; alias them on PyTorch 2.6."""
    os.environ.setdefault("CAUSAL_FLEX_ATTENTION_COMPILE_MODE", "max-autotune-no-cudagraphs")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    import torch

    config = torch._dynamo.config
    for new_name, old_name in (("recompile_limit", "cache_size_limit"),
                               ("accumulated_recompile_limit", "accumulated_cache_size_limit")):
        if not hasattr(config, new_name):
            config._config[new_name] = config._config[old_name]  # type: ignore[attr-defined]


def add_causal_forcing_to_path() -> Path:
    """Put the patched Causal-Forcing checkout on ``sys.path`` and return it."""
    _torch_compat()
    root = causal_forcing_root()
    if not (root / "utils" / "wan_wrapper.py").is_file():
        raise FileNotFoundError(
            f"Causal-Forcing not found at {root}. Run `bash scripts/setup/setup_causal_forcing.sh` "
            "or set CAUSAL_FORCING_ROOT to a patched checkout.")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
