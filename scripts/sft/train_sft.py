"""Supervised fine-tuning of Wan2.2-TI2V-5B: causal (K=1 / K=8, teacher forcing) or bidirectional.

Usage (normally through scripts/sft/train.sh):
    torchrun --nproc_per_node=N scripts/sft/train_sft.py --config-path configs/sft/flowfree_k1.yaml \
        --logdir outputs/sft/flowfree_k1 [key=value ...]

The experiment config is merged over Causal-Forcing's configs/default_config.yaml; a top-level ``_base_`` key
(a path relative to the config file, or a list of paths) is merged first. Extra ``key=value`` arguments override
the merged config, and ``???`` marks values that must be given on the command line.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "CAUSAL_FLEX_ATTENTION_COMPILE_MODE", "max-autotune-no-cudagraphs"
)
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import torch
from omegaconf import OmegaConf
from videcide.config import load_config, missing_values
from videcide.generation.common import checkpoint_write_lock

import torch.distributed.device_mesh as device_mesh_module


if "backend_override" not in __import__("inspect").signature(
    device_mesh_module.init_device_mesh
).parameters:
    _init_device_mesh = device_mesh_module.init_device_mesh

    def _compat_init_device_mesh(*args: Any, **kwargs: Any):
        kwargs.pop("backend_override", None)
        return _init_device_mesh(*args, **kwargs)

    device_mesh_module.init_device_mesh = _compat_init_device_mesh


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

import model.diffusion as diffusion_module
import trainer.diffusion as trainer_module
from model.diffusion import CausalDiffusion
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class SFTDiffusion(CausalDiffusion):

    def _initialize_models(self, args: Any, device: torch.device) -> None:
        model_kwargs = getattr(args, "model_kwargs", {})
        model_name = getattr(model_kwargs, "model_name", "Wan2.1-T2V-1.3B")
        is_causal = bool(getattr(args, "causal", True))
        self.generator = WanDiffusionWrapper(**model_kwargs, is_causal=is_causal)
        if not is_causal:
            self.generator.seq_len = 0
            self.generator.enable_gradient_checkpointing = lambda: (
                self.generator.model._set_gradient_checkpointing(None, value=True)
            )
        self.generator.model.requires_grad_(True)
        self.text_encoder = WanTextEncoder(model_name=model_name)
        self.text_encoder.requires_grad_(False)
        self.vae = WanVAEWrapper(model_name=model_name)
        self.vae.requires_grad_(False)
        generator_numel = sum(parameter.numel() for parameter in self.generator.parameters())
        trainable_numel = sum(
            parameter.numel()
            for parameter in self.generator.parameters()
            if parameter.requires_grad
        )
        if trainable_numel != generator_numel:
            raise RuntimeError(
                "SFT trains the full generator: "
                f"trainable={trainable_numel} total={generator_numel}"
            )
        print(
            f"[full-finetune] generator_trainable={trainable_numel} generator_total={generator_numel} "
            "text_encoder_trainable=0 vae_trainable=0",
            flush=True,
        )
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        self.is_causal = is_causal

    def _get_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
        uniform_timestep: bool = False,
    ) -> torch.Tensor:
        if self.is_causal:
            index = super()._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                num_frame_per_block,
                uniform_timestep=uniform_timestep,
            )
            if not self.teacher_forcing and self.independent_first_frame:
                index[:, 0] = int(self.scheduler.timesteps.numel()) - 1
            return index
        return torch.randint(
            min_timestep,
            max_timestep,
            (batch_size, 1),
            device=self.device,
            dtype=torch.long,
        ).repeat(1, num_frame)

    def generator_loss(self, *args: Any, **kwargs: Any):
        validity_mask = kwargs.get("validity_mask")
        if validity_mask is not None and bool((validity_mask[:, 0] != 0).any()):
            raise ValueError("latent frame zero (the given first frame) must be loss-masked")
        if self.is_causal and self.teacher_forcing:
            return super().generator_loss(*args, **kwargs)

        if self.is_causal:
            original_timesteps = self.scheduler.timesteps
            original_sigmas = self.scheduler.sigmas
            original_weights = self.scheduler.linear_timesteps_weights
            self.scheduler.timesteps = torch.cat(
                [original_timesteps, original_timesteps.new_zeros(1)]
            )
            self.scheduler.sigmas = torch.cat(
                [original_sigmas, original_sigmas.new_zeros(1)]
            )
            self.scheduler.linear_timesteps_weights = torch.cat(
                [original_weights, original_weights.new_zeros(1)]
            )
            try:
                return super().generator_loss(*args, **kwargs)
            finally:
                self.scheduler.timesteps = original_timesteps
                self.scheduler.sigmas = original_sigmas
                self.scheduler.linear_timesteps_weights = original_weights

        original_noise = diffusion_module._randn_like_with_sample_seeds

        def clean_clamped_noise(
            reference: torch.Tensor,
            sample_seeds: list[int] | tuple[int, ...] | None,
        ) -> torch.Tensor:
            noise = original_noise(reference, sample_seeds)
            noise[:, 0] = reference[:, 0]
            return noise

        diffusion_module._randn_like_with_sample_seeds = clean_clamped_noise
        try:
            return super().generator_loss(*args, **kwargs)
        finally:
            diffusion_module._randn_like_with_sample_seeds = original_noise


class SFTTrainer(trainer_module.Trainer):

    def _resume_from_full_state(self, ckpt_path):
        super()._resume_from_full_state(ckpt_path)
        if (bool(getattr(self.config, "require_optimizer_resume", False))
                or bool(getattr(self.config, "restore_rank_state", False))):
            states = list(self.generator_optimizer.state.values())
            if not states or any(
                not {"step", "exp_avg", "exp_avg_sq"}.issubset(state)
                or float(state["step"]) <= 0 for state in states
            ):
                raise RuntimeError("Continuation requires restored Adam moments; fresh optimizer fallback is forbidden")
        if not bool(getattr(self.config, "resume_reset_lr_scheduler", False)) and self.lr_scheduler is not None:
            state = self.lr_scheduler.state_dict()
            if "_cf_schedule_start_step" in state:
                self.lr_scheduler = self._build_lr_scheduler(
                    schedule_start_step=state["_cf_schedule_start_step"],
                    schedule_steps=state["_cf_schedule_steps"],
                    warmup_steps=state["_cf_warmup_steps"],
                    base_lr=state["base_lrs"][0],
                )
                self.lr_scheduler.load_state_dict(state)
                self._set_lr_scheduler_step(self.step)
        if bool(getattr(self.config, "restore_rank_state", False)):
            rank = torch.distributed.get_rank()
            state = torch.load(Path(ckpt_path).parent / f"rank_state_{rank}.pt",
                               map_location="cpu", weights_only=False)
            if state["step"] != self.step or state["world_size"] != torch.distributed.get_world_size():
                raise RuntimeError("Rank-state step/topology mismatch; do not approximate resume")
            if state.get("data_step_origin", 0) != int(getattr(self.config, "data_step_origin", 0)):
                raise RuntimeError("Data-cursor phase origin changed; do not approximate resume")
            loader = self.dataloader.gi_frame.f_locals["dl"]
            if bool(getattr(self.config, 'robotwin_matched_batching', False)) or bool(getattr(self.config, 'bucket_batching', False)):
                self.sampler = loader.sampler
            if loader.num_workers or int(self.config.gradient_accumulation_steps) != 1:
                raise RuntimeError("an exact data cursor requires zero workers and accumulation=1")
            if len(loader) != state["batches_per_pass"]:
                raise RuntimeError("Data-loader length changed on resume")
            self.sampler.set_epoch(state["sampler_epoch"])
            self.dataloader = self._resume_data_cycle(loader, state)
            self._restore_rank_rng(state)
            print(f"[resume] rank={rank} full RNG and batch cursor={state['batch_offset']} restored", flush=True)

    @staticmethod
    def _restore_rank_rng(state):
        import random
        import numpy as np
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state_all(state["cuda"])

    def _resume_data_cycle(self, loader, state):
        offset = state["batch_offset"]
        iterator = iter(loader)
        for _ in range(offset):
            next(iterator)
        if offset:
            self._restore_rank_rng(state)
        yield from iterator
        yield from trainer_module.cycle(loader)

    def _save_rank_state(self):
        if not bool(getattr(self.config, "restore_rank_state", False)):
            return
        import random
        import numpy as np
        variables = self.dataloader.gi_frame.f_locals
        loader = variables.get("dl", variables.get("loader"))
        if loader is None or loader.num_workers or int(self.config.gradient_accumulation_steps) != 1:
            raise RuntimeError("cannot save the exact data cursor")
        rank = torch.distributed.get_rank()
        data_origin = int(getattr(self.config, "data_step_origin", 0))
        if self.step < data_origin:
            raise RuntimeError("Data-cursor origin exceeds the current training step")
        root = Path(self.output_path) / f"checkpoint_model_{int(self.step):06d}"
        root.mkdir(parents=True, exist_ok=True)
        state = {"step": int(self.step), "world_size": torch.distributed.get_world_size(),
                 "batches_per_pass": len(loader), "batch_offset": (int(self.step) - data_origin) % len(loader),
                 "data_step_origin": data_origin,
                 "sampler_epoch": int(self.sampler.epoch), "python": random.getstate(),
                 "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                 "cuda": torch.cuda.get_rng_state_all()}
        target = root / f"rank_state_{rank}.pt"
        torch.save(state, str(target) + ".tmp")
        os.replace(str(target) + ".tmp", target)
        torch.distributed.barrier()

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        if bool(getattr(config, 'robotwin_matched_batching', False)) or bool(getattr(config, 'bucket_batching', False)):
            variables = self.dataloader.gi_frame.f_locals
            loader = variables.get('dl', variables.get('loader'))
            self.sampler = loader.sampler

    _stop_requested = False

    @classmethod
    def request_stop(cls, *_: Any) -> None:
        cls._stop_requested = True

    def train_one_step(self, batch: Any) -> Any:
        result = super().train_one_step(batch)
        if not hasattr(self, "_val_rows"):
            self._val_setup()
        interval = int(getattr(self.config, "val_interval", 1000) or 1000)
        if self._val_rows and (int(self.step) in (1, 250, 500) or int(self.step) % interval == 0):
            self._val_loss()
        rolling = int(getattr(self.config, "rolling_full_state_iters", 0) or 0)
        stop = SFTTrainer._stop_requested or (Path(self.output_path) / "STOP_SAVE").exists()
        flag = torch.tensor([int(stop)], device=self.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        stop = bool(flag.item())
        if stop or (rolling and int(self.step) % rolling == 0):
            self._save_rolling(reason="stop" if stop else "rolling")
        if stop:
            if self.is_main_process:
                (Path(self.output_path) / "STOPPED.json").write_text(
                    json.dumps({"step": int(self.step), "checkpoint": str(Path(self.output_path) / "checkpoint_rolling")})
                    + "\n")
                (Path(self.output_path) / "STOP_SAVE").unlink(missing_ok=True)
                print(f"[stop] saved full state at step {self.step}; exiting", flush=True)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
            sys.exit(0)
        return result

    def _val_setup(self) -> None:
        import hashlib

        self._val_rows = []
        n = int(getattr(self.config, "val_samples", 0) or 0)
        if not n:
            return
        task = str(self.config.task)
        dataset = trainer_module.VariableLengthLatentLMDBDataset(
            self.config.data_path, max_pair=int(1e8),
            metadata_path=self.config.metadata_path,
            metadata_filter={"task": task, "split": "val"},
            frame_buckets=trainer_module._configured_frame_buckets(self.config),
        )
        order = sorted(range(len(dataset)), key=lambda i: hashlib.sha256(
            str(dataset.metadata_rows[i]["sample_id"]).encode()).hexdigest())[:n]
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if len(order) % world:
            order = order[:len(order) - len(order) % world]
        self._val_dataset = dataset
        self._val_rows = order[rank::world]
        if self.is_main_process:
            print(f"[val] {len(order)} fixed rows of {len(dataset)} held out, "
                  f"every {int(getattr(self.config, 'val_interval', 1000))} steps", flush=True)

    @torch.no_grad()
    def _val_loss(self) -> None:
        common_shift = getattr(self.config, "val_timestep_shift", None)
        if common_shift is None:
            return self._val_loss_impl()
        from utils.scheduler import FlowMatchScheduler

        generator = trainer_module.unwrap_module(self.model.generator)
        original_model = self.model.scheduler
        original_generator = generator.scheduler
        common = FlowMatchScheduler(shift=float(common_shift), sigma_min=0.0, extra_one_step=True)
        common.set_timesteps(int(self.config.num_train_timestep), training=True)
        common.timesteps = common.timesteps.to(self.device)
        generator.scheduler = common
        self.model.scheduler = generator.get_scheduler()
        try:
            return self._val_loss_impl()
        finally:
            self.model.scheduler = original_model
            generator.scheduler = original_generator

    @torch.no_grad()
    def _val_loss_impl(self) -> None:
        import hashlib

        results = []
        for index in self._val_rows:
            batch = trainer_module.collate_with_none([self._val_dataset[index]])
            sample_id = str(self._val_dataset.metadata_rows[index]["sample_id"])
            seed = int(hashlib.sha256(f"val:{sample_id}".encode()).hexdigest()[:8], 16)
            clean = batch["clean_latent"].to(device=self.device, dtype=self.dtype)
            valid = batch.get("valid_frames")
            mask = batch.get("validity_mask")
            mask = mask.to(device=self.device, dtype=self.dtype) if mask is not None else None
            context = batch.get("clean_x_latents")
            context = context.to(device=self.device, dtype=self.dtype) if context is not None else None
            conditional = self.model.text_encoder(text_prompts=batch["prompts"])
            with torch.random.fork_rng(devices=[self.device]):
                torch.manual_seed(seed)
                loss, _ = self.model.generator_loss(
                    image_or_video_shape=list(clean.shape), conditional_dict=conditional,
                    unconditional_dict=self.unconditional_dict, clean_latent=clean,
                    initial_latent=clean[:, 0:1], valid_frames=int(valid[0]) if valid is not None and mask is None else None,
                    clean_context_override=context, validity_mask=mask,
                    action_labels=self._batch_action_labels(batch, clean.shape[1]),
                    action_loss_weight=self.action_loss_weight,
                    action_loss_use_validity_mask=self.action_loss_use_validity_mask,
                    action_class_weights=self.action_class_weights,
                    action_loss_fixed_timestep=self.action_loss_fixed_timestep,
                    action_head_only=self.action_head_only,
                    training_noise_seeds=[seed] if bool(getattr(self.config, "paired_training_noise", False)) else None,
                )
            results.append((sample_id, float(loss.item())))
        gathered = [results]
        if torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, results)
        if self.is_main_process:
            rows = sorted(r for part in gathered for r in part)
            losses = [loss for _, loss in rows]
            record = {"step": int(self.step), "val_loss": sum(losses) / len(losses), "n": len(losses),
                      "median": sorted(losses)[len(losses) // 2], "per_row": dict(rows)}
            if getattr(self.config, "val_timestep_shift", None) is not None:
                record["validation_timestep_shift"] = float(self.config.val_timestep_shift)
            with (Path(self.output_path) / "val_loss.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"[val] step {self.step} val_loss {record['val_loss']:.5f} (n={len(losses)})", flush=True)

    def _save_rolling(self, reason: str) -> None:
        with checkpoint_write_lock(getattr(self.config, "checkpoint_io_lock", None)):
            return self._save_rolling_unlocked(reason)

    def _save_rolling_unlocked(self, reason: str) -> None:
        import shutil

        original = bool(getattr(self.config, "save_model_only", False))
        self.config.save_model_only = False
        self._save_rank_state()
        torch.cuda.empty_cache()
        try:
            trainer_module.Trainer.save(self)
        finally:
            self.config.save_model_only = original
            torch.cuda.empty_cache()
        if self.is_main_process:
            root = Path(self.output_path)
            fresh = root / f"checkpoint_model_{int(self.step):06d}"
            rolling, previous = root / "checkpoint_rolling", root / "checkpoint_rolling.previous"
            if rolling.exists():
                rolling.rename(previous)
            fresh.rename(rolling)
            (rolling / "ROLLING.json").write_text(json.dumps({"step": int(self.step), "reason": reason}) + "\n")
            if previous.exists():
                shutil.rmtree(previous)
            print(f"[rolling] full state at step {self.step} -> {rolling}", flush=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def save(self) -> None:
        with checkpoint_write_lock(getattr(self.config, "checkpoint_io_lock", None)):
            return self._save_named_checkpoint()

    def _save_named_checkpoint(self) -> None:
        named_steps = getattr(self.config, "named_checkpoint_steps", None)
        if (
            named_steps is not None
            and int(self.step) < int(self.config.max_iters)
            and int(self.step) not in {int(step) for step in named_steps}
        ):
            return
        original_model_only = bool(getattr(self.config, "save_model_only", False))
        terminal_full_state = (
            bool(getattr(self.config, "save_full_state_final", True))
            and int(self.step) >= int(self.config.max_iters)
        )
        if original_model_only and terminal_full_state:
            print(
                "[save] terminal checkpoint: saving full training state",
                flush=True,
            )
            self.config.save_model_only = False
        if not self.config.save_model_only:
            self._save_rank_state()
        try:
            return super().save()
        finally:
            self.config.save_model_only = original_model_only

TASKS = ("flowfree", "sokoban", "maze", "vbvr", "robotwin")   # latents built by scripts/benchmarks/build_latents.py --task

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_known_args()


def validate_config(config: Any) -> None:
    matched_robo = bool(getattr(config, 'robotwin_matched_batching', False))
    if matched_robo and (str(getattr(config, 'task', '')) != 'robotwin' or int(config.batch_size) != 2):
        raise ValueError('Matched RoboTwin batching requires robotwin and microbatch2')
    named = getattr(config, 'named_checkpoint_steps', None) or []
    unreachable = [s for s in named if int(s) < int(config.max_iters) and int(s) % int(config.save_iters)]
    if unreachable and not getattr(config, 'no_save', False):
        raise ValueError(f'named checkpoint steps {unreachable} are not multiples of save_iters={config.save_iters}')
    causal = bool(config.causal)
    teacher_forcing = bool(config.teacher_forcing)
    block = int(config.num_frame_per_block)
    if causal and not bool(config.independent_first_frame):
        raise ValueError("causal models require independent_first_frame=true")
    if causal and not teacher_forcing and block != 1:
        raise ValueError("causal diffusion forcing is currently supported only for K1")
    if not causal and (teacher_forcing or block != 1):
        raise ValueError(
            "bidirectional models require teacher_forcing=false and block=1"
        )
    if not causal:
        frame_buckets = [int(value) for value in config.frame_buckets]
        prebuild_buckets = [int(value) for value in config.prebuild_buckets]
        latent_frames = int(config.image_or_video_shape[1])
        valid_buckets = (max(frame_buckets) == latent_frames if matched_robo else frame_buckets == [latent_frames])
        if not valid_buckets or prebuild_buckets:
            raise ValueError(
                "bidirectional arm requires a single frame bucket equal to the "
                f"image_or_video_shape latent length ({latent_frames}) and prebuild_buckets=[]"
            )
    bucket_batching = bool(getattr(config, 'bucket_batching', False))
    if bucket_batching and matched_robo:
        raise ValueError('bucket_batching and robotwin_matched_batching are exclusive')
    if int(config.batch_size) != 1 and not (matched_robo or bucket_batching):
        raise ValueError("the current variable-length validity-mask path requires batch_size=1 (or bucket_batching)")
    if not bool(config.variable_length):
        raise ValueError("training requires metadata-backed validity masks (variable_length=true)")
    if bool(getattr(config, "save_model_only", False)) and not bool(
        getattr(config, "save_full_state_final", True)
    ):
        raise ValueError(
            "milestones may be weights-only, but the terminal "
            "checkpoint must save full training state"
        )
    task = getattr(config, "task", None)
    if task is not None:
        task = str(task)
        if task not in TASKS:
            raise ValueError(f"unsupported task={task}; expected one of {TASKS}")
        metadata_filter = OmegaConf.to_container(
            getattr(config, "metadata_filter", {}), resolve=True
        )
        tasks = [task]
        if metadata_filter not in ({"task": task}, {"task": task, "split": "train"}):
            raise ValueError(
                f"task={task} requires metadata_filter.task "
                "(optionally with split=train when a validation split is held out)"
            )
        metadata_rows = json.loads(Path(str(config.metadata_path)).read_text())
        matching = [
            row
            for row in metadata_rows
            if row.get("task") in tasks and row.get("split") == "train"
        ]
        if "split" not in metadata_filter and any(row.get("split") != "train" for row in metadata_rows
                                                  if row.get("task") in tasks):
            raise ValueError("metadata holds non-train rows; add metadata_filter.split=train")
        expected_rows = int(getattr(config, "expected_train_rows", 1000))
        if len(matching) != expected_rows:
            raise ValueError(
                f"task={task} selected {len(matching)} train rows; "
                f"expected {expected_rows}"
            )
        print(
            f"[task-data] task={task} train_rows={len(matching)} "
            f"metadata={config.metadata_path}",
            flush=True,
        )


def main() -> None:
    args, dotlist = parse_args()
    default_config = OmegaConf.load(REFERENCE_ROOT / "configs/default_config.yaml")
    experiment_config = load_config(args.config_path)
    config = OmegaConf.merge(default_config, experiment_config)
    if dotlist:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(dotlist))
    missing = missing_values(config)
    if missing:
        raise ValueError(f"Set these config values on the command line (key=value): {missing}")
    config.config_name = args.config_path.stem
    config.logdir = str(args.logdir.resolve())
    config.wandb_save_dir = str((args.logdir / "wandb").resolve())
    config.no_save = bool(args.no_save)
    config.no_visualize = True
    config.disable_wandb = bool(args.disable_wandb)
    config.tf = bool(config.teacher_forcing)
    validate_config(config)

    args.logdir.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.to_container(config, resolve=True)
    (args.logdir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[mode] "
        + json.dumps(
            {
                "causal": bool(config.causal),
                "teacher_forcing": bool(config.teacher_forcing),
                "num_frame_per_block": int(config.num_frame_per_block),
                "conditioning": (
                    "causal clean-history double stream"
                    if config.causal and config.teacher_forcing
                    else (
                        "causal single-stream diffusion forcing with clean frame zero"
                        if config.causal
                        else "bidirectional clean-clamped frame zero"
                    )
                ),
                "loss_mask": "latent suffix [1,T)",
                "task": getattr(config, "task", None),
                "learning_rate": float(config.lr),
                "max_iters": int(config.max_iters),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.validate_only:
        return
    trainer_module.CausalDiffusion = SFTDiffusion
    num_workers = int(getattr(config, "num_workers", 0))
    original_dataloader = torch.utils.data.DataLoader

    def configured_dataloader(*loader_args: Any, **loader_kwargs: Any):
        loader_kwargs["num_workers"] = num_workers
        if bool(getattr(config, 'robotwin_matched_batching', False)):
            from videcide.training.batching import MatchedSubtaskSampler, collate_native_or_freeze
            dataset = loader_args[0] if loader_args else loader_kwargs['dataset']
            old_sampler = loader_kwargs['sampler']
            loader_kwargs['sampler'] = MatchedSubtaskSampler(dataset, old_sampler.num_replicas,
                old_sampler.rank, int(config.batch_size), int(config.seed))
            loader_kwargs['collate_fn'] = collate_native_or_freeze
        elif bool(getattr(config, 'bucket_batching', False)):
            from videcide.training.batching import BucketBatchSampler, collate_bucket_batch
            dataset = loader_args[0] if loader_args else loader_kwargs['dataset']
            old_sampler = loader_kwargs['sampler']
            loader_kwargs['sampler'] = BucketBatchSampler(dataset, old_sampler.num_replicas,
                old_sampler.rank, int(config.batch_size), int(config.seed))
            loader_kwargs['collate_fn'] = collate_bucket_batch
        return original_dataloader(*loader_args, **loader_kwargs)

    torch.utils.data.DataLoader = configured_dataloader
    print(f"[data] num_workers={num_workers}", flush=True)
    try:
        import signal

        signal.signal(signal.SIGUSR1, SFTTrainer.request_stop)
        signal.signal(signal.SIGTERM, SFTTrainer.request_stop)
        trainer = SFTTrainer(config)
        torch.cuda.reset_peak_memory_stats()
        try:
            trainer.train()
        finally:
            print(
                "[cuda-memory] "
                + json.dumps(
                    {
                        "max_allocated_gib": round(
                            torch.cuda.max_memory_allocated() / 1024**3, 3
                        ),
                        "max_reserved_gib": round(
                            torch.cuda.max_memory_reserved() / 1024**3, 3
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        torch.utils.data.DataLoader = original_dataloader


if __name__ == "__main__":
    main()
