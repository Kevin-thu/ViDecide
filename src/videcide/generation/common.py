"""Small helpers shared by the generation, training and evaluation scripts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from contextlib import contextmanager
import fcntl

import torch.distributed as dist


def stable_sample_seed(base_seed: int, sample_id: str) -> int:
    """Seed of one sample: the base seed plus a hash of its id (modulo 2^63 - 1)."""
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big")
    return (int(base_seed) + offset) % (2**63 - 1)


def row_seed(base_seed: int, row: dict) -> int:
    """Sampling seed of a manifest row.

    The evaluation and training manifests of the released data store the offset of every row (``seed_offset``), which
    reproduces the sampling noise of the paper's evaluation and post-training collection; without it the offset is the
    hash of the sample id, as in :func:`stable_sample_seed`.
    """
    if row.get("seed_offset") is None:
        return stable_sample_seed(base_seed, str(row["sample_id"]))
    return (int(base_seed) + int(row["seed_offset"])) % (2**63 - 1)


SEED_FIELDS = ("seed_offset", "continuation_seed_offsets")


def with_seed_fields(rows: list[dict], *manifests: Path) -> list[dict]:
    """``rows`` with the seed fields of the manifest rows of the same sample id (rows read from the metadata of a
    training run, or from a copied subset, may lack them)."""
    fields = {}
    for manifest in manifests:
        if Path(manifest).is_file():
            for row in read_jsonl(Path(manifest)):
                fields[str(row["sample_id"])] = {k: row[k] for k in SEED_FIELDS if k in row}
    return [{**fields.get(str(row["sample_id"]), {}), **row} for row in rows]


def rollout_seed_fields(row: dict, k: int) -> dict:
    """Fields of the record of rollout ``k`` of a manifest row that seed its continuations (:func:`continuation_seed`)."""
    offsets = row.get("continuation_seed_offsets") or []
    return {"continuation_seed_offset": int(offsets[k])} if k < len(offsets) else {}


def continuation_seed(base_seed: int, rollout: dict) -> int:
    """Seed of a continuation of a rollout record: the base seed plus the hash of the rollout key (``<sample id>_r<k>``),
    or plus the rollout's ``continuation_seed_offset``, taken from the training manifest, which reproduces the paper's
    collection."""
    if rollout.get("continuation_seed_offset") is None:
        return stable_sample_seed(base_seed, str(rollout["key"]))
    return (int(base_seed) + int(rollout["continuation_seed_offset"])) % (2**63 - 1)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@contextmanager
def checkpoint_write_lock(path):
    if not path:
        yield
        return
    distributed=dist.is_initialized()
    leader=not distributed or dist.get_rank()==0
    handle=None
    if leader:
        target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
        handle=target.open('a');fcntl.flock(handle,fcntl.LOCK_EX)
    if distributed:dist.barrier()
    try:
        yield
    finally:
        if distributed:dist.barrier()
        if handle is not None:handle.close()


def native_eos_metadata(stored_latents, eos_frame, loss_start=1, block=1):
    stored_latents, loss_start, block = int(stored_latents), int(loss_start), int(block)
    if stored_latents < 2 or block < 1 or not 1 <= loss_start < stored_latents:
        raise ValueError('Invalid stored length, block or supervised suffix')
    if block > 1 and ((stored_latents-1) % block or (loss_start-1) % block):
        raise ValueError('K-block history and stored trajectory must end on block boundaries')
    actual = stored_latents
    if eos_frame is not None:
        eos_frame = int(eos_frame)
        if not 0 <= eos_frame <= 4*(stored_latents-1):
            raise ValueError('EOS frame lies outside the saved causal-VAE video')
        generated = (eos_frame+3)//4
        actual = min(stored_latents, 1 + block*((generated+block-1)//block))
    if actual <= loss_start:
        raise ValueError('No supervised suffix before EOS; reject this training row')
    return dict(actual_latent_frames=actual, stored_latent_frames=stored_latents,
                native_eos_frame=eos_frame, validity_mask_end=actual)


def native_eos_loss_metadata(stored_latents, eos_frame, loss_start=1, block=1):
    row = native_eos_metadata(stored_latents, eos_frame, loss_start, block)
    end = row['actual_latent_frames'] if eos_frame is None else 1 + (int(eos_frame) + 3) // 4
    if end <= int(loss_start):
        raise ValueError('No supervised suffix through the EOS-containing latent')
    row['validity_mask_end'] = end
    row['eos_loss_mask'] = 'native_eos_latent_stop'
    if eos_frame is not None:
        row['eos_target_pad_from'] = end
    return row


def load_run_config(run_dir: Path):
    """Model config of a training run (``resolved_config.json``) or of a released model folder (``config.json``)."""
    from omegaconf import OmegaConf

    run_dir = Path(run_dir)
    for name in ("resolved_config.json", "config.json"):
        if (run_dir / name).is_file():
            return OmegaConf.create(json.loads((run_dir / name).read_text(encoding="utf-8")))
    raise FileNotFoundError(f"no resolved_config.json or config.json in {run_dir}")


def load_generator_state_dict(path: Path) -> dict:
    """Generator weights from a training checkpoint (``model.pt``) or a released ``model.safetensors``."""
    import torch

    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return state["generator"] if isinstance(state, dict) and "generator" in state else state


def default_checkpoint(run_dir: Path, config: Any) -> Path:
    """``model.safetensors`` of a released model folder, else the final ``checkpoint_model_*/model.pt`` of a run."""
    run_dir = Path(run_dir)
    if (run_dir / "model.safetensors").is_file():
        return run_dir / "model.safetensors"
    return run_dir / f"checkpoint_model_{int(getattr(config, 'max_iters')):06d}" / "model.pt"
