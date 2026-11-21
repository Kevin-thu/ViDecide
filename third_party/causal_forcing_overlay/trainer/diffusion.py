# Modified by the ViDecide authors from Causal-Forcing (https://github.com/thu-ml/Causal-Forcing,
# commit 3e7cc99, Apache-2.0): warmup + cosine/constant schedules with resets on resume, exact full-state resume,
# variable-length bucketed data, paired training noise, FSDP2 options. See NOTICE.
import gc
import hashlib
import logging
import random
import json

from model import CausalDiffusion
from utils.dataset import (
    collate_with_none,
    cycle,
    LengthBucketDistributedSampler,
    PairedPlanDistributedSampler,
    LatentLMDBDataset,
    MixedLatentLMDBDataset,
    StratifiedMixedLatentDataset,
    VariableLengthLatentLMDBDataset,
)
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import time
import os
import math
import numpy as np
from utils.distributed import (
    EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job,
    init_parallel_mesh, get_dp_group, get_tp_group, get_tp_size, get_sp_size,
    get_sp_mesh, get_sp_group,
    unwrap_module, fsdp2_wrap, compute_clip_grad_norm_2d,
)
from utils.tp_parallel import parallelize_causal_model, tp_sum_replicated_grads
from utils.sequence_parallel import enable_sp_on_model, sp_sum_replicated_grads
from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)


def _paired_training_noise_seed(group_id: str, step: int, base_seed: int) -> int:
    payload = f"{int(base_seed)}\0{int(step)}\0{group_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _configured_frame_buckets(config, field: str = "frame_buckets") -> list[int] | None:
    value = getattr(config, field, None)
    return None if value is None else [int(bucket) for bucket in value]


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        tp_size = int(getattr(config, "tp_size", 1) or 1)
        sp_size = int(getattr(config, "sp_size", 1) or 1)
        dp_size_cfg = getattr(config, "dp_size", None)
        self.parallel_mesh = init_parallel_mesh(
            tp_size=tp_size, dp_size=dp_size_cfg, sp_size=sp_size,
        )
        self.tp_size = get_tp_size()
        self.sp_size = get_sp_size()
        self.tp_group = get_tp_group()
        self.sp_group = get_sp_group()
        self.tp_mesh = self.parallel_mesh["tp"] if self.parallel_mesh is not None else None
        self.sp_mesh = get_sp_mesh()
        self.dp_group = get_dp_group()

        if self.sp_size > 1:
            _tf = bool(getattr(config, "teacher_forcing", False))
            _mult = 2 if _tf else 1
            if getattr(config, "variable_length", False):
                from utils.dataset import FRAME_BUCKETS as _FB
                bad = [F for F in _FB if (_mult * F) % self.sp_size != 0]
            else:
                Fcfg = int(config.image_or_video_shape[1])
                bad = [] if (_mult * Fcfg) % self.sp_size == 0 else [Fcfg]
            if bad:
                raise ValueError(
                    f"sp_size={self.sp_size} but frame buckets {bad} don't "
                    f"satisfy ({_mult}·F) % sp == 0 (teacher_forcing={_tf}). "
                    f"Pick sp_size=2 (works with any odd F under TF), or adjust "
                    f"FRAME_BUCKETS to include multiples of sp_size."
                )
        self.sp_enabled = self.sp_size > 1
        self._use_sp_now = False

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        _world_rank = global_rank
        _denom = max(1, self.tp_size * self.sp_size)
        _dp_rank_for_seed = _world_rank // _denom
        set_seed(config.seed + _dp_rank_for_seed)

        if self.is_main_process and not self.disable_wandb:
            _wkey = getattr(config, "wandb_key", "") or ""
            if _wkey:
                wandb.login(host=config.wandb_host, key=_wkey)
            else:
                wandb.login(host=config.wandb_host)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=getattr(config, "wandb_name", None) or config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        self.model = CausalDiffusion(config, device=self.device)
        self.action_loss_weight = float(getattr(config, "action_loss_weight", 0.0) or 0.0)
        self.action_loss_use_validity_mask = bool(
            getattr(config, "action_loss_use_validity_mask", True)
        )
        self.action_class_weights = getattr(config, "action_class_weights", None)
        if self.action_class_weights is not None:
            self.action_class_weights = [float(x) for x in self.action_class_weights]
        self.action_loss_fixed_timestep = getattr(config, "action_loss_fixed_timestep", None)
        self.action_head_only = bool(getattr(config, "action_head_only", False))
        self.action_head_enabled = (
            self.action_loss_weight > 0
            or float(getattr(config, "rl_action_loss_weight", 0.0) or 0.0) > 0
            or bool(getattr(config, "eval_use_action_head", False))
        )
        self.action_labels_by_row = None
        self.action_labels_by_source_row = None
        if self.action_head_enabled:
            num_actions = int(getattr(config, "action_num_classes", 13))
            self.model.generator.add_action_head(num_actions=num_actions)
            if bool(getattr(config, "freeze_generator_except_action_head", False)):
                for name, param in self.model.generator.named_parameters():
                    param.requires_grad_("action_head" in name)
            if bool(getattr(config, "detach_action_features", False)) or self.action_head_only:
                self.model.generator.detach_action_features = True
            if self.is_main_process:
                print(f"[action-supervision] enabled: num_actions={num_actions} "
                      f"weight={self.action_loss_weight} "
                      f"rl_weight={float(getattr(config, 'rl_action_loss_weight', 0.0) or 0.0)}")
                if bool(getattr(config, "freeze_generator_except_action_head", False)):
                    trainable = [
                        (name, int(param.numel()))
                        for name, param in self.model.generator.named_parameters()
                        if param.requires_grad
                    ]
                    print(
                        "[action-supervision] trainable before FSDP: "
                        f"count={len(trainable)} numel={sum(x[1] for x in trainable)} "
                        f"names={[x[0] for x in trainable[:12]]}",
                        flush=True,
                    )
        if self.tp_size > 1:
            parallelize_causal_model(
                self.model.generator, self.tp_mesh, use_sp=self._use_sp_now
            )
        if self.sp_size > 1:
            enable_sp_on_model(self.model.generator, self.sp_mesh)
        self.skip_generator_fsdp = bool(getattr(config, "skip_generator_fsdp", False))
        if self.skip_generator_fsdp:
            world_size = dist.get_world_size()
            if self.tp_size != 1 or self.sp_size != 1:
                raise ValueError(
                    "skip_generator_fsdp only supports data parallel runs with "
                    "tp=sp=1 "
                    f"run (world={world_size}, tp={self.tp_size}, sp={self.sp_size})."
                )
            self.model.generator = self.model.generator.to(
                device=self.device, dtype=self.dtype,
            )
            if world_size > 1:
                self.model.generator = DDP(
                    self.model.generator,
                    device_ids=[self.device],
                    output_device=self.device,
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                )
            if self.is_main_process:
                mode = "DDP" if world_size > 1 else "plain module"
                print(
                    f"[fsdp] skip_generator_fsdp=true; generator uses {mode}",
                    flush=True,
                )
        if self.tp_size > 1:
            from wan.modules.causal_model import CausalWanAttentionBlock
            dp_submesh = self.parallel_mesh["dp"]
            self.model.generator = fsdp2_wrap(
                self.model.generator,
                dp_mesh=dp_submesh,
                mixed_precision=config.mixed_precision,
                transformer_cls=CausalWanAttentionBlock,
            )
            self.model.text_encoder = fsdp2_wrap(
                self.model.text_encoder,
                dp_mesh=dp_submesh,
                mixed_precision=config.mixed_precision,
            )
        else:
            if not self.skip_generator_fsdp:
                self.model.generator = fsdp_wrap(
                    self.model.generator,
                    sharding_strategy=config.sharding_strategy,
                    mixed_precision=config.mixed_precision,
                    wrap_strategy=config.generator_fsdp_wrap_strategy,
                    process_group=self.dp_group,
                )
            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
                process_group=self.dp_group,
            )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        if self.is_main_process and self.action_head_enabled:
            opt_numel = sum(
                int(param.numel())
                for group in self.generator_optimizer.param_groups
                for param in group["params"]
            )
            print(
                f"[action-supervision] optimizer trainable numel={opt_numel}",
                flush=True,
            )

        self.lr_scheduler = self._build_lr_scheduler()

        if getattr(config, "variable_length", False):
            frame_buckets = _configured_frame_buckets(config)
            base_dataset = VariableLengthLatentLMDBDataset(
                config.data_path, max_pair=int(1e8),
                metadata_path=getattr(config, "metadata_path", None),
                metadata_filter=getattr(config, "metadata_filter", None),
                frame_buckets=frame_buckets,
            )
            onpolicy_shard_path = getattr(config, "onpolicy_shard_path", None)
            if onpolicy_shard_path:
                onpolicy_dataset = VariableLengthLatentLMDBDataset(
                    onpolicy_shard_path, max_pair=int(1e8),
                    metadata_path=getattr(config, "onpolicy_metadata_path", None),
                    metadata_filter=getattr(config, "onpolicy_metadata_filter", None),
                    frame_buckets=(
                        _configured_frame_buckets(config, "onpolicy_frame_buckets")
                        or frame_buckets
                    ),
                )
                if bool(getattr(config, "length_bucket_stratified_sampler", False)):
                    dataset = StratifiedMixedLatentDataset(base_dataset, onpolicy_dataset)
                else:
                    dataset = MixedLatentLMDBDataset(
                        base_dataset,
                        onpolicy_dataset,
                        mix_start=float(getattr(config, "onpolicy_mix_start", 0.2)),
                        mix_end=float(getattr(config, "onpolicy_mix_end", 0.5)),
                        anneal_steps=int(getattr(config, "onpolicy_anneal_steps", 8000)),
                    )
                print(f"[trainer] MixedLatentLMDBDataset: base={len(base_dataset)} op={len(onpolicy_dataset)}")
            else:
                dataset = base_dataset
        else:
            dataset = LatentLMDBDataset(config.data_path, max_pair=int(1e8))

        self.dataset = dataset
        if self.parallel_mesh is not None:
            _dp_submesh = self.parallel_mesh["dp"]
            _sampler_num_replicas = _dp_submesh.size()
            _sampler_rank = _dp_submesh.get_local_rank()
        else:
            _sampler_num_replicas = dist.get_world_size()
            _sampler_rank = dist.get_rank()
        if bool(getattr(config, "paired_plan_sampler", False)):
            if int(config.batch_size) != 1:
                raise ValueError("paired_plan_sampler requires batch_size=1")
            if not bool(getattr(config, "paired_training_noise", False)):
                raise ValueError("paired_plan_sampler requires paired_training_noise=true")
            sampler = PairedPlanDistributedSampler(
                dataset,
                num_replicas=_sampler_num_replicas,
                rank=_sampler_rank,
                seed=int(getattr(config, "seed", 0) or 0),
            )
            if self.is_main_process:
                print(
                    "[trainer] PairedPlanDistributedSampler enabled: "
                    f"pairs={len(sampler)} dp_replicas={_sampler_num_replicas}",
                    flush=True,
                )
        elif bool(getattr(config, "length_bucket_stratified_sampler", False)) and hasattr(dataset, "sample_infos"):
            sampler = LengthBucketDistributedSampler(
                dataset,
                num_replicas=_sampler_num_replicas,
                rank=_sampler_rank,
                onpolicy_prob=float(getattr(config, "onpolicy_mix_start", 0.5)),
                mixed_group_prob=float(getattr(config, "length_bucket_mixed_group_prob", 0.5)),
                seed=int(getattr(config, "seed", 0) or 0),
                drop_last=True,
            )
            if self.is_main_process:
                print(
                    "[trainer] LengthBucketDistributedSampler enabled: "
                    f"dp_replicas={_sampler_num_replicas} rank={_sampler_rank} "
                    f"onpolicy_prob={float(getattr(config, 'onpolicy_mix_start', 0.5))} "
                    f"mixed_group_prob={float(getattr(config, 'length_bucket_mixed_group_prob', 0.5))}",
                    flush=True,
                )
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=_sampler_num_replicas,
                rank=_sampler_rank,
                shuffle=True, drop_last=True)
        self.sampler = sampler
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8,
            collate_fn=collate_with_none)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        if getattr(config, "variable_length", False):
            from utils.dataset import FRAME_BUCKETS
            active_frame_buckets = frame_buckets or list(FRAME_BUCKETS)
            prebuild_buckets = getattr(config, "prebuild_buckets", None)
            if prebuild_buckets is not None:
                prebuild_list = list(prebuild_buckets)
            else:
                prebuild_list = list(active_frame_buckets)
            if dist.get_rank() == 0:
                print(f"Variable-length mode: buckets={active_frame_buckets}, "
                      f"prebuild={prebuild_list}, loss masked to valid frames")
            gen = unwrap_module(self.model.generator)
            inner = gen.model if hasattr(gen, "model") else gen
            frame_seqlen = config.image_or_video_shape[3] * config.image_or_video_shape[4] // (
                inner.patch_size[1] * inner.patch_size[2])
            todo = [nf for nf in prebuild_list if nf not in inner._block_mask_cache]
            if todo:
                import os as _os, time as _time
                from concurrent.futures import ProcessPoolExecutor
                import multiprocessing as _mp
                from wan.modules.causal_model import CausalWanModel
                _t0 = _time.time()
                nfb = inner.num_frame_per_block
                independent_first = bool(inner.independent_first_frame)

                cache_dir = _os.path.join("data", ".block_mask_cache")
                _os.makedirs(cache_dir, exist_ok=True)

                def _cache_path(nf):
                    i2v_suffix = "_I1" if independent_first else ""
                    return _os.path.join(
                        cache_dir, f"tf_F{nf}_S{frame_seqlen}_B{nfb}{i2v_suffix}.pt",
                    )

                disk_hits = 0
                disk_miss = []
                for nf in todo:
                    cpath = _cache_path(nf)
                    if _os.path.exists(cpath):
                        try:
                            payload = torch.load(
                                cpath, map_location="cpu", weights_only=False,
                            )
                            cpu_mask = payload["block_mask"]
                            helpers = payload["helpers"]
                            gpu_mask = CausalWanModel._finalize_tf_mask(
                                cpu_mask, helpers, self.device,
                            )
                            inner._block_mask_cache[nf] = gpu_mask
                            disk_hits += 1
                            if dist.get_rank() == 0:
                                print(f"  loaded cached block_mask for {nf} frames")
                        except Exception as exc:
                            if dist.get_rank() == 0:
                                print(f"  WARN bad cache for F={nf}: {exc}; rebuilding")
                            disk_miss.append(nf)
                    else:
                        disk_miss.append(nf)

                if disk_miss:
                    ctx = _mp.get_context("spawn")
                    with ProcessPoolExecutor(
                        max_workers=min(len(disk_miss), 16), mp_context=ctx,
                    ) as ex:
                        futures = {
                            ex.submit(CausalWanModel._build_tf_mask_cpu,
                                      nf, frame_seqlen, nfb, independent_first): nf
                            for nf in disk_miss
                        }
                        for fut in futures:
                            nf = futures[fut]
                            cpu_mask, helpers = fut.result()
                            if dist.get_rank() == 0:
                                try:
                                    torch.save(
                                        {"block_mask": cpu_mask, "helpers": helpers},
                                        _cache_path(nf),
                                    )
                                except Exception as exc:
                                    print(f"  WARN failed to write cache for F={nf}: {exc}")
                            gpu_mask = CausalWanModel._finalize_tf_mask(
                                cpu_mask, helpers, self.device,
                            )
                            inner._block_mask_cache[nf] = gpu_mask
                            if dist.get_rank() == 0:
                                print(f"  pre-built block_mask for {nf} frames")

                if dist.get_rank() == 0:
                    print(f"  prebuild done in {_time.time() - _t0:.1f}s "
                          f"({disk_hits} cached, {len(disk_miss)} built)")

        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            if self.tp_size > 1:
                raise NotImplementedError(
                    "EMA under tp>1 requires DCP-based shadow storage (Phase 4). "
                    "Disable EMA (set ema_weight=0) for now, or train with tp_size=1."
                )
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        resume_from = getattr(config, "resume_from", None) or None
        generator_ckpt = getattr(config, "generator_ckpt", None) or None
        if resume_from and generator_ckpt:
            print(
                f"[resume] resume_from set; ignoring generator_ckpt="
                f"{generator_ckpt}"
            )
            generator_ckpt = None

        if generator_ckpt:
            print(f"Loading pretrained generator weights from {generator_ckpt}")
            state_dict = torch.load(
                generator_ckpt, map_location="cpu", weights_only=False)
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                gen_sd = state_dict["generator_ema"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            action_head_enabled = self.action_head_enabled
            if self.tp_size > 1:
                from torch.distributed.checkpoint.state_dict import (
                    set_model_state_dict, StateDictOptions,
                )
                opts = StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=False,
                    strict=not action_head_enabled,
                )
                set_model_state_dict(self.model.generator, state_dict, options=opts)
            else:
                if action_head_enabled:
                    load_target = (
                        unwrap_module(self.model.generator)
                        if self.skip_generator_fsdp
                        else self.model.generator
                    )
                    incompat = load_target.load_state_dict(state_dict, strict=False)
                    bad_missing = [k for k in incompat.missing_keys if "action_head" not in k]
                    bad_unexpected = [k for k in incompat.unexpected_keys if "action_head" not in k]
                    if bad_missing or bad_unexpected:
                        raise RuntimeError(
                            "Generator checkpoint mismatch outside action_head: "
                            f"missing={bad_missing[:10]} unexpected={bad_unexpected[:10]}"
                        )
                    if self.is_main_process:
                        print("[action-supervision] initialized action_head from scratch; "
                              f"ignored missing keys={len(incompat.missing_keys)}")
                else:
                    load_target = (
                        unwrap_module(self.model.generator)
                        if self.skip_generator_fsdp
                        else self.model.generator
                    )
                    load_target.load_state_dict(state_dict, strict=True)

        if resume_from:
            self._resume_from_full_state(resume_from)


        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm = float(getattr(config, "max_grad_norm", 10.0))
        self.previous_time = None
        self.delta_mean = None
        self.rtf_ema_ratio = getattr(self.config, "rtf_ema_ratio", 0.9)
        self.rtf_single_gpu_batch = getattr(self.config, "rtf_single_gpu_batch", 1)
        self.given_first_chunk = getattr(self.config, "given_first_chunk", True)

        action_label_path = getattr(config, "action_label_path", None)
        if self.action_loss_weight > 0:
            if not action_label_path:
                raise ValueError("action_loss_weight > 0 requires action_label_path")
            data = np.load(action_label_path, allow_pickle=True)
            ignore_arr = data["ignore_index"] if "ignore_index" in data.files else np.asarray([-100])
            self.action_ignore_index = int(ignore_arr[0])
            self.action_labels_by_row = {
                int(row_idx): np.asarray(labels, dtype=np.int64)
                for row_idx, labels in zip(data["row_idx"], data["labels"])
            }
            if "source" in data.files:
                self.action_labels_by_source_row = {
                    (str(source), int(row_idx)): np.asarray(labels, dtype=np.int64)
                    for source, row_idx, labels in zip(data["source"], data["row_idx"], data["labels"])
                }
            if self.is_main_process:
                print(f"[action-supervision] loaded {len(self.action_labels_by_row)} rows "
                      f"from {action_label_path}")

    def _batch_action_labels(self, batch, num_frame: int):
        if self.action_labels_by_row is None:
            return None
        row_idx = batch.get("row_idx", None)
        if row_idx is None:
            return None
        if torch.is_tensor(row_idx):
            row_ids = [int(x) for x in row_idx.detach().cpu().view(-1).tolist()]
        else:
            row_ids = [int(row_idx)]
        sample_source = batch.get("_sample_source", None)
        if sample_source is None:
            sources = [None] * len(row_ids)
        elif isinstance(sample_source, (list, tuple)):
            sources = [str(x) for x in sample_source]
        else:
            sources = [str(sample_source)] * len(row_ids)
        labels = []
        for source, rid in zip(sources, row_ids):
            arr = None
            if self.action_labels_by_source_row is not None and source is not None:
                arr = self.action_labels_by_source_row.get((source, rid))
            if arr is None:
                arr = self.action_labels_by_row.get(rid)
            out = torch.full((num_frame,), self.action_ignore_index, dtype=torch.long)
            if arr is not None:
                n = min(num_frame, int(arr.shape[0]))
                out[:n] = torch.from_numpy(arr[:n]).long()
            labels.append(out)
        return torch.stack(labels, dim=0).to(device=self.device)

    def _action_head_grad_norm(self) -> float | None:
        gen = unwrap_module(self.model.generator)
        action_head = getattr(gen, "action_head", None)
        if action_head is None:
            return None
        total_sq = 0.0
        seen = False
        for param in action_head.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            total_sq += float(torch.sum(grad * grad).cpu())
            seen = True
        if not seen:
            return 0.0
        return math.sqrt(total_sq)

    def _optimizer_grad_norm(self) -> float:
        total_sq = 0.0
        for group in self.generator_optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach().float()
                total_sq += float(torch.sum(grad * grad).cpu())
        return math.sqrt(total_sq)

    def _build_lr_scheduler(
            self, schedule_start_step=0, schedule_steps=None,
            warmup_steps=None, base_lr=None):
        max_iters_for_sched = int(getattr(self.config, "max_iters", 0) or 0)
        if max_iters_for_sched <= 0:
            return None

        schedule_start_step = int(schedule_start_step or 0)
        if schedule_steps is None:
            schedule_steps = max_iters_for_sched - schedule_start_step
        schedule_steps = max(1, int(schedule_steps))

        if warmup_steps is None:
            warmup_steps = int(getattr(self.config, "warmup_steps", 0) or 0)
        warmup_steps = max(0, min(int(warmup_steps), schedule_steps))

        if base_lr is not None:
            base_lr = float(base_lr)
            for group in self.generator_optimizer.param_groups:
                group["lr"] = base_lr
                group["initial_lr"] = base_lr

        schedule_type = str(
            getattr(self.config, "lr_schedule", "cosine") or "cosine"
        ).lower()

        import math

        def lr_lambda(current_step):
            local_step = max(0, int(current_step) - schedule_start_step)
            if warmup_steps > 0 and local_step < warmup_steps:
                return float(local_step) / float(max(1, warmup_steps))
            if schedule_type == "constant":
                return 1.0
            progress = float(local_step - warmup_steps) / float(
                max(1, schedule_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.generator_optimizer, lr_lambda)
        scheduler._cf_schedule_start_step = schedule_start_step
        scheduler._cf_schedule_steps = schedule_steps
        scheduler._cf_warmup_steps = warmup_steps
        scheduler._cf_schedule_type = schedule_type
        return scheduler

    def _set_lr_scheduler_step(self, step):
        if self.lr_scheduler is None:
            return
        step = int(step)
        lrs = [
            base_lr * lr_lambda(step)
            for lr_lambda, base_lr in zip(
                self.lr_scheduler.lr_lambdas, self.lr_scheduler.base_lrs)
        ]
        for group, lr in zip(self.generator_optimizer.param_groups, lrs):
            group["lr"] = lr
        self.lr_scheduler.last_epoch = step
        self.lr_scheduler._last_lr = lrs

    def _reset_lr_scheduler_for_resume(self):
        schedule_steps = getattr(self.config, "resume_lr_schedule_steps", None)
        if schedule_steps is None:
            schedule_steps = max(
                1, int(getattr(self.config, "max_iters", 0) or 0) - self.step)
        schedule_steps = int(schedule_steps)

        warmup_steps = getattr(self.config, "resume_lr_warmup_steps", 0)
        if warmup_steps is None:
            warmup_steps = 0

        base_lr = getattr(self.config, "resume_lr", None)
        if base_lr is None:
            base_lr = getattr(self.config, "lr")

        self.lr_scheduler = self._build_lr_scheduler(
            schedule_start_step=self.step,
            schedule_steps=schedule_steps,
            warmup_steps=warmup_steps,
            base_lr=base_lr,
        )
        self._set_lr_scheduler_step(self.step)

        if dist.get_rank() == 0:
            lr_now = self.generator_optimizer.param_groups[0]["lr"]
            print(
                f"[resume-lr] reset scheduler at step={self.step}: "
                f"schedule_steps={schedule_steps}, "
                f"warmup_steps={int(warmup_steps)}, base_lr={float(base_lr):.3e}, "
                f"schedule={getattr(self.lr_scheduler, '_cf_schedule_type', 'cosine')}, "
                f"current_lr={lr_now:.3e}")

    def _resume_from_full_state(self, ckpt_path):
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP, StateDictType,
        )
        from torch.distributed.fsdp.api import (
            FullOptimStateDictConfig, FullStateDictConfig,
        )

        print(f"[resume] Loading full-state checkpoint: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        required = ("generator", "optimizer", "step", "rng_state")
        missing = [k for k in required if k not in sd]
        if missing:
            raise ValueError(
                f"resume_from={ckpt_path} is missing keys {missing} — it was "
                f"not produced by the current trainer's save(). Use "
                f"generator_ckpt=... instead to do a weights-only fresh-start."
            )

        from torch.distributed.checkpoint.state_dict import (
            set_model_state_dict, set_optimizer_state_dict, StateDictOptions,
        )

        gen_sd = sd["generator"]
        fixed = {}
        for k, v in gen_sd.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[k] = v
        opts = StateDictOptions(
            full_state_dict=True, broadcast_from_rank0=False, strict=True,
        )

        _resume_sample_keys = []
        if dist.get_rank() == 0:
            inner = unwrap_module(self.model.generator)
            named = list(inner.named_parameters())

            def disk_key(k):
                for c in (k, k.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")):
                    if c in fixed:
                        return c
                return None

            # Under FSDP, rank 0 holds a part of the parameters only: check four parameters that it holds completely
            # (a partly held parameter would be compared with the mean of the whole tensor on disk).
            whole = [k for k, p in named if disk_key(k) is not None and p.numel() == fixed[disk_key(k)].numel()]
            if whole:
                idx = sorted({0, len(whole) // 4, len(whole) // 2, len(whole) - 1})
                _resume_sample_keys = [whole[i] for i in idx]
                _pre = {
                    k: dict(named)[k].detach().float().abs().mean().item()
                    for k in _resume_sample_keys
                }
                _disk_norms = {k: fixed[disk_key(k)].float().abs().mean().item() for k in _resume_sample_keys}

        set_model_state_dict(self.model.generator, fixed, options=opts)

        if dist.get_rank() == 0:
            for k, p in unwrap_module(self.model.generator).named_parameters():
                if k in _resume_sample_keys:
                    post = p.detach().float().abs().mean().item()
                    pre = _pre[k]
                    disk = _disk_norms.get(k, float("nan"))
                    same_as_pre = abs(post - pre) / (pre + 1e-12) < 1e-6
                    matches_disk = abs(post - disk) / (disk + 1e-12) < 1e-3
                    tag = "OK" if matches_disk else (
                        "STALE" if same_as_pre else "OFF"
                    )
                    print(f"[resume-verify] {tag}  {k}  "
                          f"pre={pre:.6e}  post={post:.6e}  "
                          f"disk={disk:.6e}")

        load_optimizer = bool(getattr(self.config, "resume_load_optimizer", True))
        if not load_optimizer:
            if dist.get_rank() == 0:
                print("[resume-optim] skip optimizer state by config; "
                      "Adam moments will be rebuilt from the resumed weights")
        else:
            osd = sd["optimizer"]
            dp_world_size = (
                dist.get_world_size(self.dp_group)
                if self.dp_group is not None
                else dist.get_world_size()
            )
            optimizer_resharded = (
                dp_world_size > 1
                and str(getattr(self.config, "sharding_strategy", "full")) != "no_shard"
            )
            if optimizer_resharded:
                optim_opts = StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=False,
                )
                set_optimizer_state_dict(
                    self.model.generator,
                    self.generator_optimizer,
                    osd,
                    options=optim_opts,
                )
                if dist.get_rank() == 0:
                    print(
                        f"[resume-optim] DCP resharded optimizer state for "
                        f"dp_size={dp_world_size}"
                    )
            live_named = list(unwrap_module(self.model.generator).named_parameters())
            def _normalize(k):
                return (k.replace("_fsdp_wrapped_module.", "")
                         .replace("_checkpoint_wrapped_module.", ""))
            live_name_to_id = {}
            for i, (name, _) in enumerate(live_named):
                live_name_to_id[_normalize(name)] = i

            disk_state = {} if optimizer_resharded else osd.get("state", {})
            remapped_state = {}
            n_matched = 0
            n_missing = 0
            for disk_key, val in disk_state.items():
                norm = _normalize(disk_key)
                if norm in live_name_to_id:
                    remapped_state[live_name_to_id[norm]] = val
                    n_matched += 1
                else:
                    n_missing += 1
            if dist.get_rank() == 0 and not optimizer_resharded:
                print(f"[resume-optim] matched {n_matched}/{len(disk_state)} "
                      f"param entries (missing {n_missing})")

            new_param_groups = []
            for pg in (() if optimizer_resharded else osd.get("param_groups", [])):
                new_pg = dict(pg)
                new_pg["params"] = list(range(len(live_named)))
                new_param_groups.append(new_pg)

            rebuilt_osd = {"state": remapped_state, "param_groups": new_param_groups}
            try:
                if not optimizer_resharded:
                    self.generator_optimizer.load_state_dict(rebuilt_osd)

                bad_shapes = []
                for group in self.generator_optimizer.param_groups:
                    for p in group["params"]:
                        state = self.generator_optimizer.state.get(p, {})
                        exp_avg = state.get("exp_avg")
                        if exp_avg is not None and tuple(exp_avg.shape) != tuple(p.shape):
                            bad_shapes.append((tuple(exp_avg.shape), tuple(p.shape)))
                            if len(bad_shapes) >= 4:
                                break
                    if len(bad_shapes) >= 4:
                        break
                if bad_shapes:
                    if optimizer_resharded:
                        raise RuntimeError(
                            "DCP optimizer reshard produced parameter/state "
                            f"shape mismatches: {bad_shapes}"
                        )
                    import collections
                    self.generator_optimizer.state = collections.defaultdict(dict)
                    if dist.get_rank() == 0:
                        print("[resume-optim] WARN: loaded optimizer state has "
                              f"shape mismatches {bad_shapes}; discarded optimizer "
                              "state so Adam can rebuild moments")
                elif dist.get_rank() == 0:
                    if optimizer_resharded:
                        print(
                            f"[resume-optim] verified local optimizer-state "
                            f"shapes for {len(self.generator_optimizer.state)} params"
                        )
                    state_after = self.generator_optimizer.state_dict()["state"]
                    if state_after:
                        sample_id = next(iter(state_after))
                        sample = state_after[sample_id]
                        if "exp_avg" in sample:
                            ea = sample["exp_avg"]
                            ea_norm = ea.float().abs().mean().item() if hasattr(ea, "float") else 0.0
                            print(f"[resume-optim] post-load exp_avg sample id={sample_id} "
                                  f"mean_abs={ea_norm:.4e} (should be non-zero ~1e-6)")
            except Exception as exc:
                if optimizer_resharded:
                    raise RuntimeError(
                        "failed to restore optimizer state for the current "
                        "FSDP DP topology"
                    ) from exc
                if dist.get_rank() == 0:
                    print(f"[resume-optim] WARN: load_state_dict failed: {exc}; "
                          f"falling back to fresh optim state (Adam will rebuild moments)")

        self.step = int(sd["step"])

        reset_lr_scheduler = bool(
            getattr(self.config, "resume_reset_lr_scheduler", False))
        if self.lr_scheduler is not None:
            if reset_lr_scheduler:
                self._reset_lr_scheduler_for_resume()
            elif sd.get("lr_scheduler") is not None:
                self.lr_scheduler.load_state_dict(sd["lr_scheduler"])

        from utils.misc import set_seed
        _world_rank = dist.get_rank()
        _denom = max(1, self.tp_size * self.sp_size)
        _dp_rank_for_seed = _world_rank // _denom
        _resume_seed = int(getattr(self.config, "seed", 0) or 0) + _dp_rank_for_seed
        set_seed(_resume_seed)
        if dist.get_rank() == 0:
            print(f"[resume] RNG re-seeded via set_seed({_resume_seed}) for SP-rank "
                  f"sync (replaces saved cuda RNG which would desync rank 1's "
                  f"cuda:1 stream — see comment in trainer/diffusion.py).")

        if hasattr(self, "sampler"):
            self.sampler.set_epoch(self.step)

        snap = sd.get("config_snapshot", {}) or {}
        drifted = []
        for k in ("seed", "warmup_steps", "max_iters", "lr",
                  "lr_schedule",
                  "resume_reset_lr_scheduler", "resume_lr_schedule_steps",
                  "resume_lr_warmup_steps", "resume_lr"):
            if k in snap:
                cur = getattr(self.config, k, None)
                if cur is not None and cur != snap[k]:
                    drifted.append(f"{k}: saved={snap[k]} current={cur}")
        if drifted:
            print(f"[resume] NOTE: config drift since checkpoint — "
                  f"{'; '.join(drifted)}. This is allowed (e.g. extending "
                  f"max_iters on a continuation run) but double-check if "
                  f"any of these were unintentional.")

        print(f"[resume] Restored full state: step={self.step}, "
              f"lr={self.generator_optimizer.param_groups[0]['lr']:.3e}")

    def save(self):
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict, get_optimizer_state_dict, StateDictOptions,
        )
        print("Start gathering distributed model states...")

        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        generator_state_dict = get_model_state_dict(
            self.model.generator, options=opts,
        )

        gathered_norms = {}
        sample_keys = []
        if self.is_main_process:
            tensor_keys = [
                k for k, v in generator_state_dict.items()
                if torch.is_tensor(v) and v.is_floating_point()
            ]
            if tensor_keys:
                idx = [0, len(tensor_keys) // 4, len(tensor_keys) // 2, len(tensor_keys) - 1]
                sample_keys = [tensor_keys[i] for i in idx]
                for k in sample_keys:
                    gathered_norms[k] = generator_state_dict[k].detach().float().abs().mean().item()
                print(f"[save-verify] gathered sample params: {sample_keys}")
            else:
                print("[save-verify] WARN: gathered state has no floating tensors; "
                      "verify skipped")

        model_only = bool(getattr(self.config, "save_model_only", False))
        state_dict = {
            "generator": generator_state_dict,
            "step": int(self.step),
            "config_snapshot": {
                "seed": int(getattr(self.config, "seed", 0) or 0),
                "warmup_steps": int(getattr(self.config, "warmup_steps", 0) or 0),
                "max_iters": int(getattr(self.config, "max_iters", 0) or 0),
                "lr": float(getattr(self.config, "lr", 0.0) or 0.0),
                "lr_schedule": str(getattr(self.config, "lr_schedule", "cosine")),
                "max_grad_norm": float(getattr(self.config, "max_grad_norm", 10.0) or 0.0),
                "rl_chunk_len_latents": int(getattr(self.config, "rl_chunk_len_latents", 0) or 0),
                "rl_best_k": int(getattr(self.config, "rl_best_k", 0) or 0),
                "rl_worst_k": int(getattr(self.config, "rl_worst_k", 0) or 0),
                "rl_finite_check_interval": int(getattr(self.config, "rl_finite_check_interval", 0) or 0),
                "rl_reward_normalizer": float(getattr(self.config, "rl_reward_normalizer", 0.0) or 0.0),
                "config_name": getattr(self.config, "config_name", ""),
                "wandb_name": getattr(self.config, "wandb_name", ""),
            },
        }
        if model_only:
            state_dict["weights_only"] = True
            if self.is_main_process:
                print("[save] save_model_only=true: omitting optimizer, scheduler, and RNG state")
        else:
            optim_state_dict = get_optimizer_state_dict(
                self.model.generator,
                self.generator_optimizer,
                options=opts,
            )
            import numpy as _np
            import random as _random
            state_dict.update({
                "optimizer": optim_state_dict,
                "lr_scheduler": (self.lr_scheduler.state_dict()
                                 if self.lr_scheduler is not None else None),
                "rng_state": {
                    "python": _random.getstate(),
                    "numpy": _np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            })
        if not model_only and self.config.ema_start_step < self.step:
            state_dict["generator_ema"] = self.generator_ema.full_state_dict(
                self.model.generator,
            )

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            ckpt_path = os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt")
            tmp_ckpt_path = ckpt_path + ".tmp"
            torch.save(state_dict, tmp_ckpt_path)
            os.replace(tmp_ckpt_path, ckpt_path)
            print("Model saved to", ckpt_path)

            if gathered_norms:
                disk_sd = torch.load(ckpt_path, map_location="cpu",
                                     weights_only=False)["generator"]
                disk_keys = set(disk_sd.keys())
                all_match = True
                checked = 0
                for k in sample_keys:
                    gathered = gathered_norms.get(k)
                    if gathered is None:
                        continue
                    dk = k if k in disk_keys else None
                    if dk is None:
                        print(f"[save-verify] MISSING-ON-DISK  gathered={k}")
                        all_match = False
                        continue
                    dsk = disk_sd[dk].float().abs().mean().item()
                    rel = abs(gathered - dsk) / (gathered + 1e-12)
                    ok = rel < 1e-3
                    all_match = all_match and ok
                    mark = "OK" if ok else "MISMATCH"
                    print(f"[save-verify] {mark}  gathered={k}  disk={dk}  "
                          f"gathered={gathered:.6e}  disk={dsk:.6e}  rel_diff={rel:.3e}")
                    checked += 1
                if checked == 0:
                    print("[save-verify] WARN: no gathered key matched disk; "
                          "verify inconclusive")
                if not all_match:
                    print("[save-verify] !! SAVE WRITEBACK BROKEN — "
                          "disk content differs from gathered weights")
        barrier()

    def train_one_step(self, batch):
        if self.step % 20 == 0:
            torch.cuda.empty_cache()
        if hasattr(self.dataset, "set_step"):
            self.dataset.set_step(self.step)

        text_prompts = batch["prompts"]
        if not self.config.load_raw_video:
            clean_latent = batch["clean_latent"].to(
                device=self.device, dtype=self.dtype)
        else:
            frames = batch["frames"].to(
                device=self.device, dtype=self.dtype)

            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(
                    frames).to(device=self.device, dtype=self.dtype)
        valid_frames = batch.get("valid_frames", None)
        if valid_frames is not None:
            valid_frames = int(valid_frames[0])
        clean_x_latents = batch.get("clean_x_latents", None)
        if clean_x_latents is not None:
            clean_x_latents = clean_x_latents.to(device=self.device, dtype=self.dtype)
        op_validity_mask = batch.get("validity_mask", None)
        if op_validity_mask is not None:
            op_validity_mask = op_validity_mask.to(device=self.device, dtype=self.dtype)
        decision_latent_indices = batch.get("decision_latent_index", None)
        if decision_latent_indices is not None:
            decision_latent_indices = decision_latent_indices.to(
                device=self.device, dtype=torch.long
            )
            if op_validity_mask is not None:
                decision_positions = torch.nn.functional.one_hot(
                    decision_latent_indices.reshape(-1),
                    num_classes=clean_latent.shape[1],
                ).to(device=self.device, dtype=torch.bool)
                if not bool((op_validity_mask.to(dtype=torch.bool) & decision_positions).any(dim=1).all()):
                    raise ValueError(
                        "decision_latent_index must lie inside the supervised full suffix"
                    )
        if (
            str(getattr(self.config, "decision_loss_mode", "equal")) == "balanced"
            or str(getattr(self.config, "decision_sigma_mode", "normal")) == "pure"
        ) and decision_latent_indices is None:
            raise ValueError(
                "balanced decision loss or pure decision sigma requires "
                "decision_latent_index in every metadata row"
            )
        image_latent = clean_latent[:, 0:1, ]

        batch_size = len(text_prompts)
        image_or_video_shape = list(clean_latent.shape)
        action_labels = self._batch_action_labels(batch, clean_latent.shape[1])
        training_noise_seeds = None
        if bool(getattr(self.config, "paired_training_noise", False)):
            noise_groups = batch.get("training_noise_group_id")
            if noise_groups is None:
                raise ValueError(
                    "paired_training_noise=true requires training_noise_group_id "
                    "in every dataset row"
                )
            if isinstance(noise_groups, str):
                noise_groups = [noise_groups]
            if len(noise_groups) != batch_size:
                raise ValueError(
                    f"expected {batch_size} noise groups, got {len(noise_groups)}"
                )
            training_noise_seeds = [
                _paired_training_noise_seed(
                    str(group_id), self.step, int(getattr(self.config, "seed", 0))
                )
                for group_id in noise_groups
            ]

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        self.generator_optimizer.zero_grad()
        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent,
            valid_frames=valid_frames if op_validity_mask is None else None,
            clean_context_override=clean_x_latents,
            validity_mask=op_validity_mask,
            action_labels=action_labels,
            action_loss_weight=self.action_loss_weight,
            action_loss_use_validity_mask=self.action_loss_use_validity_mask,
            action_class_weights=self.action_class_weights,
            action_loss_fixed_timestep=self.action_loss_fixed_timestep,
            action_head_only=self.action_head_only,
            training_noise_seeds=training_noise_seeds,
            decision_latent_indices=decision_latent_indices,
        )
        generator_loss.backward()
        if self.sp_size > 1:
            sp_sum_replicated_grads(self.model.generator, self.sp_group)
        optimizer_grad_norm = self._optimizer_grad_norm()
        if self.tp_size > 1:
            generator_grad_norm = compute_clip_grad_norm_2d(
                list(self.model.generator.parameters()), self.max_grad_norm,
            )
        elif hasattr(self.model.generator, "clip_grad_norm_"):
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm)
        else:
            generator_grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.generator.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
        action_head_grad_norm = self._action_head_grad_norm()
        self.generator_optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.step += 1

        current_lr = self.generator_optimizer.param_groups[0]["lr"]
        sample_source = batch.get("_sample_source", None)
        if isinstance(sample_source, (list, tuple)):
            source_values = [str(value) for value in sample_source]
        elif sample_source is None:
            source_values = []
        else:
            source_values = [str(sample_source)]
        rank_sample_source = (
            source_values[0]
            if source_values and all(value == source_values[0] for value in source_values)
            else ("mixed" if source_values else "unknown")
        )

        source_stats = torch.zeros(5, device=self.device, dtype=torch.float64)
        if rank_sample_source == "base":
            source_stats[0] = 1.0
            source_stats[3] = generator_loss.detach().double()
        elif rank_sample_source == "onpolicy":
            source_stats[1] = 1.0
            source_stats[4] = generator_loss.detach().double()
        else:
            source_stats[2] = 1.0
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(source_stats, op=dist.ReduceOp.SUM, group=self.dp_group)

        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
            "lr": current_lr,
            "dp_base_batch_count": float(source_stats[0].item()),
            "dp_onpolicy_batch_count": float(source_stats[1].item()),
            "dp_unknown_batch_count": float(source_stats[2].item()),
        }
        if source_stats[0] > 0:
            wandb_loss_dict["base_generator_loss"] = float(
                (source_stats[3] / source_stats[0]).item()
            )
        if source_stats[1] > 0:
            wandb_loss_dict["onpolicy_generator_loss"] = float(
                (source_stats[4] / source_stats[1]).item()
            )
        if action_head_grad_norm is not None:
            wandb_loss_dict["action_head_grad_norm"] = float(action_head_grad_norm)
        wandb_loss_dict["optimizer_grad_norm"] = float(optimizer_grad_norm)
        for _k in (
            "diffusion_loss",
            "training_timestep_index",
            "training_sigma",
            "training_loss_weight",
            "paired_training_noise",
            "decision_loss",
            "rest_loss",
            "decision_loss_frames",
            "rest_loss_frames",
            "decision_timestep_index",
            "decision_sigma",
            "decision_training_loss_weight",
            "pure_sigma_decision_frames",
            "action_loss",
            "action_acc",
            "prefix_kv_loss_start",
            "prefix_kv_loss_end",
            "prefix_kv_suffix_frames",
            "prefix_kv_dense_prefix_frames",
            "prefix_kv_dense_suffix_frames",
            "prefix_kv_dense_suffix_bucket_frames",
            "prefix_kv_dense_prefix_original_frames",
            "suffix_loss_frames",
            "suffix_loss_ratio",
            "er_context_applied",
            "er_context_frames",
            "er_bank_size",
            "er_used_clean_input",
        ):
            if _k in log_dict:
                _v = log_dict[_k]
                wandb_loss_dict[_k] = float(_v.item() if torch.is_tensor(_v) else _v)

        if self.is_main_process:
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)
            try:
                import json as _json
                import time as _time
                _logdir = getattr(self.config, "logdir", "") or "/tmp"
                os.makedirs(_logdir, exist_ok=True)
                _jsonl_path = os.path.join(_logdir, "loss.jsonl")
                _rec = {
                    "step": self.step,
                    "time": _time.time(),
                    "loss": wandb_loss_dict["generator_loss"],
                    "grad_norm": wandb_loss_dict["generator_grad_norm"],
                    "lr": wandb_loss_dict["lr"],
                    "rank0_sample_source": rank_sample_source,
                }
                for _k in (
                    "diffusion_loss",
                    "training_timestep_index",
                    "training_sigma",
                    "training_loss_weight",
                    "paired_training_noise",
                    "decision_loss",
                    "rest_loss",
                    "decision_loss_frames",
                    "rest_loss_frames",
                    "decision_timestep_index",
                    "decision_sigma",
                    "decision_training_loss_weight",
                    "pure_sigma_decision_frames",
                    "action_loss",
                    "action_acc",
                    "base_generator_loss",
                    "onpolicy_generator_loss",
                    "dp_base_batch_count",
                    "dp_onpolicy_batch_count",
                    "dp_unknown_batch_count",
                    "prefix_kv_loss_start",
                    "prefix_kv_loss_end",
                    "prefix_kv_suffix_frames",
                    "prefix_kv_dense_prefix_frames",
                    "prefix_kv_dense_suffix_frames",
                    "prefix_kv_dense_suffix_bucket_frames",
                    "prefix_kv_dense_prefix_original_frames",
                    "suffix_loss_frames",
                    "suffix_loss_ratio",
                    "er_context_applied",
                    "er_context_frames",
                    "er_bank_size",
                    "er_used_clean_input",
                    "action_head_grad_norm",
                    "optimizer_grad_norm",
                ):
                    if _k in wandb_loss_dict:
                        _rec[_k] = wandb_loss_dict[_k]
                with open(_jsonl_path, "a") as _fh:
                    _fh.write(_json.dumps(_rec) + "\n")
                if self.step % 10 == 0:
                    print(
                        f"[train] step={self.step:6d}  "
                        f"loss={_rec['loss']:.4f}  "
                        f"grad_norm={_rec['grad_norm']:.4f}  "
                        f"lr={_rec['lr']:.2e}",
                        flush=True,
                    )
            except Exception as _e:
                print(f"[train] WARN: failed to write loss jsonl: {_e}", flush=True)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()


    def train(self):

        max_iters = getattr(self.config, "max_iters", 0)
        while True:
            batch = next(self.dataloader)
            self.train_one_step(batch)
            if max_iters and self.step >= max_iters:
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                reserved_gb = torch.cuda.max_memory_reserved() / 1e9
                print(
                    f"[mem] rank={dist.get_rank()} "
                    f"peak_alloc={peak_gb:.2f}GB  peak_reserved={reserved_gb:.2f}GB",
                    flush=True,
                )
                if not self.config.no_save:
                    torch.cuda.empty_cache()
                    self.save()
                    torch.cuda.empty_cache()
                break

            if (not self.config.no_save) and self.step % self.config.save_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
