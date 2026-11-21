# Modified by the ViDecide authors from Causal-Forcing (https://github.com/thu-ml/Causal-Forcing,
# commit 3e7cc99, Apache-2.0): teacher-forcing loss on variable-length videos with validity masks, per-sample seeded training
# noise, Wan2.2 loss weighting. See NOTICE.
import random
import math
from typing import Tuple
import torch
import torch.nn.functional as F

from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


def _randn_like_with_sample_seeds(
    reference: torch.Tensor,
    sample_seeds: list[int] | tuple[int, ...] | None,
) -> torch.Tensor:
    if sample_seeds is None:
        return torch.randn_like(reference)
    if len(sample_seeds) != reference.shape[0]:
        raise ValueError(
            f"expected {reference.shape[0]} noise seeds, got {len(sample_seeds)}"
        )
    chunks = []
    for sample, seed in zip(reference, sample_seeds, strict=True):
        generator = torch.Generator(device=reference.device)
        generator.manual_seed(int(seed) % (2**63 - 1))
        chunks.append(torch.randn(
            sample.shape,
            generator=generator,
            device=reference.device,
            dtype=reference.dtype,
        ))
    return torch.stack(chunks, dim=0)


def _training_loss_weight(
    scheduler,
    timestep: torch.Tensor,
    *,
    batch_size: int,
    num_frame: int,
    override: float | None,
) -> torch.Tensor:
    if override is None:
        return scheduler.training_weight(timestep).unflatten(
            0, (batch_size, num_frame)
        )
    return torch.full(
        (batch_size, num_frame),
        float(override),
        device=timestep.device,
        dtype=torch.float32,
    )


class CausalDiffusion(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = args.guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.teacher_forcing = getattr(args, "teacher_forcing", False)
        self.training_timestep_min_index = int(getattr(
            args, "training_timestep_min_index", 0
        ))
        self.training_timestep_max_index = int(getattr(
            args,
            "training_timestep_max_index",
            self.scheduler.num_train_timesteps,
        ))
        if not (
            0 <= self.training_timestep_min_index
            < self.training_timestep_max_index
            <= self.scheduler.num_train_timesteps
        ):
            raise ValueError(
                "training timestep index range must satisfy "
                "0 <= min < max <= num_train_timesteps"
            )
        self.fixed_training_timestep_index = getattr(
            args, "fixed_training_timestep_index", None
        )
        if self.fixed_training_timestep_index is not None:
            self.fixed_training_timestep_index = int(self.fixed_training_timestep_index)
            if not 0 <= self.fixed_training_timestep_index < self.scheduler.num_train_timesteps:
                raise ValueError(
                    "fixed_training_timestep_index must be in "
                    f"[0, {self.scheduler.num_train_timesteps})"
                )
        self.fixed_training_loss_weight = getattr(
            args, "fixed_training_loss_weight", None
        )
        if self.fixed_training_loss_weight is not None:
            self.fixed_training_loss_weight = float(self.fixed_training_loss_weight)
            if self.fixed_training_timestep_index is None:
                raise ValueError(
                    "fixed_training_loss_weight requires fixed_training_timestep_index"
                )
            if not math.isfinite(self.fixed_training_loss_weight) or self.fixed_training_loss_weight <= 0:
                raise ValueError("fixed_training_loss_weight must be finite and positive")
        self.decision_loss_mode = str(getattr(args, "decision_loss_mode", "equal"))
        if self.decision_loss_mode not in {"equal", "balanced"}:
            raise ValueError("decision_loss_mode must be 'equal' or 'balanced'")
        self.decision_sigma_mode = str(getattr(args, "decision_sigma_mode", "normal"))
        if self.decision_sigma_mode not in {"normal", "pure"}:
            raise ValueError("decision_sigma_mode must be 'normal' or 'pure'")
        self.pure_decision_loss_weight = getattr(args, "pure_decision_loss_weight", None)
        if self.decision_sigma_mode == "pure":
            if self.pure_decision_loss_weight is None:
                raise ValueError(
                    "decision_sigma_mode=pure requires pure_decision_loss_weight "
                    "so endpoint loss cannot silently be zero"
                )
            self.pure_decision_loss_weight = float(self.pure_decision_loss_weight)
            if not math.isfinite(self.pure_decision_loss_weight) or self.pure_decision_loss_weight <= 0:
                raise ValueError("pure_decision_loss_weight must be finite and positive")

        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)
        self.action_ignore_index = int(getattr(args, "action_ignore_index", -100))
        self.use_error_recycling = bool(getattr(args, "use_error_recycling", False))
        if self.use_error_recycling:
            self.er_num_grids = int(getattr(args, "error_recycling_num_grids", 50))
            self.er_buffer_k = int(getattr(args, "error_recycling_buffer_k", 500))
            self.er_chunk_frames = int(getattr(args, "error_recycling_chunk_frames", 3))
            self.er_context_prob = float(getattr(args, "error_recycling_context_prob", 0.9))
            self.er_clean_prob = float(getattr(args, "error_recycling_clean_prob", 0.5))
            self.er_clean_update_prob = float(getattr(args, "error_recycling_clean_update_prob", 0.1))
            self.er_modulate_factor = float(getattr(args, "error_recycling_modulate_factor", 0.0))
            self.er_sample_from_all_grids = bool(
                getattr(args, "error_recycling_sample_from_all_grids", True)
            )
            self.er_replacement = str(
                getattr(args, "error_recycling_replacement_strategy", "l2_batch")
            )
            self.er_skip_first_frame = bool(getattr(args, "error_recycling_skip_first_frame", True))
            grid_ids = torch.linspace(
                0, len(self.scheduler.timesteps) - 1, self.er_num_grids
            ).round().long()
            self.er_timestep_grids = self.scheduler.timesteps[grid_ids].detach().float().cpu()
            self.er_context_error_buffer = {i: [] for i in range(self.er_num_grids)}
            self.er_iteration = 0

    def _er_grid_idx(self, timestep) -> int:
        if isinstance(timestep, torch.Tensor):
            value = float(timestep.detach().flatten()[0].float().cpu().item())
        else:
            value = float(timestep)
        return int(torch.argmin((self.er_timestep_grids - value).abs()).item())

    def _er_total_size(self) -> int:
        if not self.use_error_recycling:
            return 0
        return sum(len(v) for v in self.er_context_error_buffer.values())

    def _er_add_context_error(self, error_chunk: torch.Tensor, timestep) -> None:
        grid = self._er_grid_idx(timestep)
        chunk_cpu = error_chunk.detach().to(device="cpu", dtype=torch.float16).contiguous()
        bucket = self.er_context_error_buffer[grid]
        if len(bucket) < self.er_buffer_k:
            bucket.append(chunk_cpu)
            return
        if self.er_replacement == "fifo":
            bucket.pop(0)
            bucket.append(chunk_cpu)
        elif self.er_replacement == "l2_batch":
            stored = torch.stack(bucket).flatten(start_dim=1).float()
            new = chunk_cpu.flatten().float()
            replace_idx = int(torch.argmin(torch.norm(stored - new.unsqueeze(0), p=2, dim=1)).item())
            bucket[replace_idx] = chunk_cpu
        else:
            bucket[random.randrange(len(bucket))] = chunk_cpu

    def _er_sample_context_error(self, like: torch.Tensor, timestep) -> torch.Tensor | None:
        if self.er_sample_from_all_grids:
            available = [
                chunk
                for bucket in self.er_context_error_buffer.values()
                for chunk in bucket
            ]
        else:
            available = self.er_context_error_buffer[self._er_grid_idx(timestep)]
        if not available:
            return None
        chunk = random.choice(available).to(device=like.device, dtype=like.dtype)
        if self.er_modulate_factor > 0:
            lo = 1.0 - self.er_modulate_factor
            hi = 1.0 + self.er_modulate_factor
            chunk = chunk * random.uniform(lo, hi)
        return chunk

    def _er_random_start(self, num_frames: int, chunk_frames: int) -> int | None:
        if num_frames <= 0:
            return None
        start_min = 1 if self.er_skip_first_frame and num_frames > 1 else 0
        max_start = num_frames - chunk_frames
        if max_start < start_min:
            return None
        return random.randint(start_min, max_start)

    def _er_apply_context_error(
        self,
        clean_context: torch.Tensor,
        timestep: torch.Tensor,
        valid_frames: int | None,
    ) -> tuple[torch.Tensor, dict]:
        info = {
            "er_context_applied": torch.tensor(0.0, device=clean_context.device),
            "er_context_frames": torch.tensor(0.0, device=clean_context.device),
            "er_bank_size": torch.tensor(float(self._er_total_size()), device=clean_context.device),
            "er_used_clean_input": torch.tensor(1.0, device=clean_context.device),
        }
        if not self.use_error_recycling or not self.teacher_forcing:
            return clean_context, info
        self.er_iteration += 1
        if random.random() < self.er_clean_prob:
            return clean_context, info
        if self._er_total_size() == 0 or random.random() >= self.er_context_prob:
            info["er_used_clean_input"] = torch.tensor(0.0, device=clean_context.device)
            return clean_context, info
        num_frames = int(valid_frames or clean_context.shape[1])
        chunk_frames = min(self.er_chunk_frames, num_frames)
        start = self._er_random_start(num_frames, chunk_frames)
        if start is None:
            return clean_context, info
        sampled = self._er_sample_context_error(
            clean_context[:, start:start + chunk_frames],
            timestep[:, start] if timestep.ndim == 2 else timestep,
        )
        if sampled is None:
            return clean_context, info
        if sampled.shape[1] != chunk_frames:
            sampled = sampled[:, :chunk_frames]
        out = clean_context.clone()
        out[:, start:start + sampled.shape[1]] = out[:, start:start + sampled.shape[1]] + sampled
        info.update({
            "er_context_applied": torch.tensor(1.0, device=clean_context.device),
            "er_context_frames": torch.tensor(float(sampled.shape[1]), device=clean_context.device),
            "er_used_clean_input": torch.tensor(0.0, device=clean_context.device),
        })
        return out, info

    def _er_update_context_bank(
        self,
        x0_pred: torch.Tensor,
        clean_latent: torch.Tensor,
        timestep: torch.Tensor,
        valid_frames: int | None,
        *,
        context_applied: bool,
    ) -> None:
        if not self.use_error_recycling or not self.teacher_forcing:
            return
        if (not context_applied) and random.random() >= self.er_clean_update_prob:
            return
        num_frames = int(valid_frames or clean_latent.shape[1])
        chunk_frames = min(self.er_chunk_frames, num_frames)
        start = self._er_random_start(num_frames, chunk_frames)
        if start is None:
            return
        residual = x0_pred[:, start:start + chunk_frames] - clean_latent[:, start:start + chunk_frames]
        self._er_add_context_error(
            residual,
            timestep[:, start] if timestep.ndim == 2 else timestep,
        )

    def _initialize_models(self, args, device):
        model_kwargs = getattr(args, "model_kwargs", {})
        model_name = getattr(model_kwargs, "model_name", "Wan2.1-T2V-1.3B")
        self.generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder(model_name=model_name)
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(model_name=model_name)
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        valid_frames=None,
        clean_context_override: torch.Tensor = None,
        validity_mask: torch.Tensor = None,
        action_labels: torch.Tensor = None,
        action_loss_weight: float = 0.0,
        action_loss_use_validity_mask: bool = True,
        action_class_weights=None,
        action_loss_fixed_timestep=None,
        action_head_only: bool = False,
        prefix_kv_cache=None,
        prefix_frame_count: int = 0,
        training_noise_seeds: list[int] | tuple[int, ...] | None = None,
        decision_latent_indices: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, dict]:
        noise = _randn_like_with_sample_seeds(clean_latent, training_noise_seeds)
        batch_size, num_frame = image_or_video_shape[:2]
        decision_mask = None
        if decision_latent_indices is not None:
            decision_latent_indices = decision_latent_indices.to(
                device=self.device, dtype=torch.long
            ).reshape(-1)
            if decision_latent_indices.numel() != batch_size:
                raise ValueError(
                    f"expected {batch_size} decision indices, got "
                    f"{decision_latent_indices.numel()}"
                )
            if not bool(((decision_latent_indices >= 1) & (decision_latent_indices < num_frame)).all()):
                raise ValueError(
                    f"decision indices must be in generated suffix [1, {num_frame})"
                )
            decision_mask = F.one_hot(
                decision_latent_indices, num_classes=num_frame
            ).to(dtype=torch.bool)
        if self.decision_loss_mode == "balanced" and decision_mask is None:
            raise ValueError("decision_loss_mode=balanced requires decision_latent_indices")
        if self.decision_sigma_mode == "pure" and decision_mask is None:
            raise ValueError("decision_sigma_mode=pure requires decision_latent_indices")

        if self.fixed_training_timestep_index is None:
            index = self._get_timestep(
                self.training_timestep_min_index,
                self.training_timestep_max_index,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=False,
            )
        else:
            index = torch.full(
                (batch_size, num_frame),
                self.fixed_training_timestep_index,
                device=self.device,
                dtype=torch.long,
            )
        if self.decision_sigma_mode == "pure":
            index = torch.where(decision_mask, torch.zeros_like(index), index)
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        training_target = self.scheduler.training_target(clean_latent, noise, timestep)

        clean_context = clean_context_override if clean_context_override is not None else clean_latent
        er_info = {}
        if self.use_error_recycling and prefix_kv_cache is None:
            clean_context, er_info = self._er_apply_context_error(
                clean_context, timestep, valid_frames
            )
        if self.noise_augmentation_max_timestep > 0:
            index_clean_aug = self._get_timestep(
                self.noise_augmentation_max_timestep,
                1000,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=False
            )
            timestep_clean_aug = self.scheduler.timesteps[index_clean_aug].to(dtype=self.dtype, device=self.device)
            clean_latent_aug = self.scheduler.add_noise(
                clean_context.flatten(0, 1),
                noise.flatten(0, 1),
                timestep_clean_aug.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        else:
            clean_latent_aug = clean_context
            timestep_clean_aug = None

        use_action_loss = action_labels is not None and float(action_loss_weight) > 0.0
        use_fixed_action_timestep = (
            use_action_loss
            and action_loss_fixed_timestep is not None
        )
        generator_out = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_latent_aug if self.teacher_forcing else None,
            aug_t=timestep_clean_aug if self.teacher_forcing else None,
            return_action_logits=use_action_loss and not use_fixed_action_timestep,
            prefix_kv_cache=prefix_kv_cache,
            prefix_frame_count=int(prefix_frame_count),
        )
        if use_action_loss and not use_fixed_action_timestep:
            flow_pred, x0_pred, action_logits = generator_out
        else:
            flow_pred, x0_pred = generator_out
            action_logits = None

        if use_fixed_action_timestep:
            if isinstance(action_loss_fixed_timestep, str):
                fixed_t = self.scheduler.timesteps[-1]
            else:
                fixed_t = torch.as_tensor(
                    float(action_loss_fixed_timestep),
                    device=self.device,
                    dtype=self.scheduler.timesteps.dtype,
                )
            action_timestep = fixed_t.to(device=self.device, dtype=self.dtype).reshape(1, 1).expand(batch_size, num_frame)
            action_noisy_latents = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                noise.flatten(0, 1),
                action_timestep.flatten(0, 1),
            ).unflatten(0, (batch_size, num_frame))
            _, _, action_logits = self.generator(
                noisy_image_or_video=action_noisy_latents,
                conditional_dict=conditional_dict,
                timestep=action_timestep,
                clean_x=clean_latent_aug if self.teacher_forcing else None,
                aug_t=timestep_clean_aug if self.teacher_forcing else None,
                return_action_logits=True,
                prefix_kv_cache=prefix_kv_cache,
                prefix_frame_count=int(prefix_frame_count),
            )
        if self.use_error_recycling and prefix_kv_cache is None:
            with torch.no_grad():
                self._er_update_context_bank(
                    x0_pred=x0_pred.detach(),
                    clean_latent=clean_latent.detach(),
                    timestep=timestep.detach(),
                    valid_frames=valid_frames,
                    context_applied=bool(
                        er_info.get("er_context_applied", torch.tensor(0)).item() > 0.5
                    ),
                )
        loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction='none'
        ).mean(dim=(2, 3, 4))
        training_loss_weight = _training_loss_weight(
            self.scheduler,
            timestep,
            batch_size=batch_size,
            num_frame=num_frame,
            override=self.fixed_training_loss_weight,
        )
        if self.decision_sigma_mode == "pure":
            training_loss_weight = torch.where(
                decision_mask,
                torch.full_like(training_loss_weight, self.pure_decision_loss_weight),
                training_loss_weight,
            )
        weighted_frame_loss = loss * training_loss_weight
        loss = weighted_frame_loss

        suffix_loss_frames = None
        suffix_loss_ratio = None
        if validity_mask is not None:
            frame_mask = validity_mask.to(device=loss.device, dtype=loss.dtype)
            if frame_mask.shape != loss.shape:
                raise ValueError(
                    f"validity_mask shape {tuple(frame_mask.shape)} does not match loss "
                    f"shape {tuple(loss.shape)}"
                )
            denom = frame_mask.sum()
            suffix_loss_frames = denom.detach()
            suffix_loss_ratio = (
                denom / max(1, int(frame_mask.numel()))
            ).detach()
            if decision_mask is not None and (
                self.decision_loss_mode == "balanced" or self.decision_sigma_mode == "pure"
            ) and not bool((frame_mask.to(dtype=torch.bool) & decision_mask).any(dim=1).all()):
                raise ValueError(
                    "each decision latent must be inside the supervised suffix"
                )
            if denom == 0:
                loss = loss.sum() * 0.0
            elif self.decision_loss_mode == "balanced":
                decision_weight = frame_mask * decision_mask.to(dtype=loss.dtype)
                rest_weight = frame_mask * (~decision_mask).to(dtype=loss.dtype)
                decision_denom = decision_weight.sum()
                rest_denom = rest_weight.sum()
                if decision_denom <= 0 or rest_denom <= 0:
                    raise ValueError(
                        "balanced decision loss requires one valid decision frame and "
                        "at least one valid non-decision suffix frame per batch"
                    )
                decision_loss = (loss * decision_weight).sum() / decision_denom
                rest_loss = (loss * rest_weight).sum() / rest_denom
                loss = 0.5 * (decision_loss + rest_loss)
            else:
                loss = (loss * frame_mask).sum() / denom
        elif valid_frames is not None:
            frame_mask = torch.zeros(batch_size, num_frame, device=loss.device)
            frame_mask[:, :valid_frames] = 1.0
            loss = loss * frame_mask
            loss = loss.sum() / frame_mask.sum()
        else:
            loss = loss.mean()

        diffusion_loss = loss
        action_loss = None
        action_acc = None
        if use_action_loss:
            labels = action_labels.to(device=action_logits.device, dtype=torch.long)
            labels = labels[:, :num_frame].contiguous()
            if labels.shape[1] < num_frame:
                pad = labels.new_full(
                    (labels.shape[0], num_frame - labels.shape[1]),
                    self.action_ignore_index,
                )
                labels = torch.cat([labels, pad], dim=1)

            action_mask = labels.ne(self.action_ignore_index)
            if validity_mask is not None and action_loss_use_validity_mask:
                action_mask = action_mask & validity_mask.to(
                    device=action_logits.device, dtype=torch.bool
                )
            elif valid_frames is not None:
                valid = torch.zeros(batch_size, num_frame, device=action_logits.device, dtype=torch.bool)
                valid[:, :valid_frames] = True
                action_mask = action_mask & valid

            if action_mask.any():
                masked_labels = labels.masked_fill(~action_mask, self.action_ignore_index)
                ce_weight = None
                if action_class_weights is not None:
                    ce_weight = torch.as_tensor(
                        action_class_weights,
                        device=action_logits.device,
                        dtype=action_logits.dtype,
                    )
                action_loss = F.cross_entropy(
                    action_logits.reshape(-1, action_logits.shape[-1]),
                    masked_labels.reshape(-1),
                    weight=ce_weight,
                    ignore_index=self.action_ignore_index,
                )
                pred = action_logits.argmax(dim=-1)
                action_acc = (pred[action_mask] == labels[action_mask]).float().mean()
                if action_head_only:
                    loss = float(action_loss_weight) * action_loss
                else:
                    loss = diffusion_loss + float(action_loss_weight) * action_loss
            else:
                action_loss = diffusion_loss.detach() * 0.0
                action_acc = diffusion_loss.detach() * 0.0

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach(),
            "diffusion_loss": diffusion_loss.detach(),
            "training_timestep_index": index.float().mean().detach(),
            "training_sigma": (
                timestep.float().mean() / self.scheduler.num_train_timesteps
            ).detach(),
            "training_loss_weight": training_loss_weight.float().mean().detach(),
        }
        if decision_mask is not None:
            decision_weight = (
                validity_mask.to(device=loss.device, dtype=loss.dtype)
                if validity_mask is not None
                else torch.ones_like(loss if loss.ndim == 2 else training_loss_weight)
            ) * decision_mask.to(dtype=training_loss_weight.dtype)
            per_frame_loss = weighted_frame_loss
            log_dict.update({
                "decision_loss": (per_frame_loss * decision_weight).sum().detach()
                / decision_weight.sum().detach(),
                "decision_loss_frames": decision_weight.sum().detach(),
                "decision_timestep_index": index.masked_select(decision_mask).float().mean().detach(),
                "decision_sigma": (
                    timestep.masked_select(decision_mask).float().mean()
                    / self.scheduler.num_train_timesteps
                ).detach(),
                "decision_training_loss_weight": training_loss_weight.masked_select(
                    decision_mask
                ).float().mean().detach(),
            })
            if validity_mask is not None:
                rest_weight = validity_mask.to(device=loss.device, dtype=loss.dtype) * (~decision_mask).to(dtype=loss.dtype)
                if rest_weight.sum() > 0:
                    log_dict["rest_loss"] = (per_frame_loss * rest_weight).sum().detach() / rest_weight.sum().detach()
                    log_dict["rest_loss_frames"] = rest_weight.sum().detach()
            if self.decision_sigma_mode == "pure":
                log_dict["pure_sigma_decision_frames"] = decision_weight.sum().detach()
        if training_noise_seeds is not None:
            log_dict["paired_training_noise"] = torch.tensor(
                1.0, device=clean_latent.device
            )
        if self.use_error_recycling:
            log_dict.update(er_info)
            log_dict["er_bank_size"] = torch.tensor(
                float(self._er_total_size()), device=clean_latent.device
            )
        if suffix_loss_frames is not None:
            log_dict["suffix_loss_frames"] = suffix_loss_frames
            log_dict["suffix_loss_ratio"] = suffix_loss_ratio
        if action_loss is not None:
            log_dict["action_loss"] = action_loss.detach()
            log_dict["action_acc"] = action_acc.detach()
        return loss, log_dict
