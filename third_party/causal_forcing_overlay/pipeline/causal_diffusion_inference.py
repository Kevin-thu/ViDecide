# Modified by the ViDecide authors from Causal-Forcing (https://github.com/thu-ml/Causal-Forcing,
# commit 3e7cc99, Apache-2.0): Wan2.2-TI2V-5B dimensions, KV caches sized to the video, segment-wise
# sampling helpers used by the rollout engine. See NOTICE.
from tqdm import tqdm
from typing import Callable, List, Optional
import os
import torch

from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None,
            need_vae = True
    ):
        super().__init__()
        model_kwargs = getattr(args, "model_kwargs", {})
        model_name = getattr(model_kwargs, "model_name", "Wan2.1-T2V-1.3B")
        self.generator = WanDiffusionWrapper(
            **model_kwargs, is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder(model_name=model_name) if text_encoder is None else text_encoder
        if need_vae:
            self.vae = WanVAEWrapper(model_name=model_name) if vae is None else vae

        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = 50
        self.sample_solver = 'unipc'
        self.shift = args.timestep_shift
        self.disable_tqdm = bool(getattr(args, "disable_tqdm", False))

        self.frame_seq_length = 1560
        inner_generator = self.generator.model if hasattr(self.generator, "model") else self.generator
        self.num_transformer_blocks = int(getattr(inner_generator, "num_layers", 30))
        self.num_attention_heads = int(getattr(inner_generator, "num_heads", 12))
        self.attention_head_dim = int(
            getattr(inner_generator, "dim", self.num_attention_heads * 128)
            // self.num_attention_heads
        )

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0,
        return_video=True,
        early_stop_patience: Optional[int] = None,
        early_stop_threshold: float = 0.015,
        eos_cosine_threshold: Optional[float] = None,
        eos_confirm_frames: int = 1,
        eos_candidate_validator: Optional[Callable[[torch.Tensor], bool]] = None,
        return_action_logits: bool = False,
        action_logit_last_n: int = 2,
    ) -> torch.Tensor:
        if os.environ.get("CAUSAL_EOS_COSINE_DEFAULT"):
            configured = os.environ["CAUSAL_EOS_COSINE_DEFAULT"]
            if eos_cosine_threshold is None and configured.strip().lower() not in ("off", "none", "0", ""):
                eos_cosine_threshold = float(configured)
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        needed_cache = num_output_frames * self.frame_seq_length
        if self.kv_cache_pos is None or self.kv_cache_pos[0]["k"].shape[1] < needed_cache:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                num_frames=num_output_frames
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        current_start_frame = start_frame_index
        cache_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        prev_block_latent = None
        stagnant_blocks = 0
        last_generated_end = cache_start_frame
        action_logits_out = [] if return_action_logits else None
        action_logit_last_n = max(1, int(action_logit_last_n))
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            sample_scheduler = self._initialize_sample_scheduler(noise)
            timesteps_iter = sample_scheduler.timesteps
            n_denoise_steps = len(timesteps_iter)
            block_action_logits = []
            for denoise_idx, t in enumerate(tqdm(
                    timesteps_iter, disable=self.disable_tqdm)):
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                )

                if return_action_logits:
                    flow_pred_cond, _, action_logits_cond = self.generator(
                        noisy_image_or_video=latent_model_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length,
                        return_action_logits=True,
                    )
                    if denoise_idx >= n_denoise_steps - action_logit_last_n:
                        block_action_logits.append(action_logits_cond.detach())
                else:
                    flow_pred_cond, _ = self.generator(
                        noisy_image_or_video=latent_model_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length
                    )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0

            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents
            if return_action_logits:
                if block_action_logits:
                    action_logits_out.append(torch.stack(block_action_logits, dim=0).mean(dim=0))
                else:
                    raise RuntimeError("return_action_logits=True but no action logits were recorded")

            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )

            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames
            last_generated_end = cache_start_frame

            if early_stop_patience is not None:
                curr = latents.detach().float()
                if prev_block_latent is not None and curr.shape == prev_block_latent.shape:
                    delta = (curr - prev_block_latent).abs().mean().item()
                    print(f"[blk {last_generated_end:3d}] Δlatent={delta:.4f}", flush=True)
                    if delta < early_stop_threshold:
                        stagnant_blocks += 1
                    else:
                        stagnant_blocks = 0
                    if stagnant_blocks >= early_stop_patience:
                        print(f"[early-stop] Δ<{early_stop_threshold} "
                              f"for {stagnant_blocks} blocks; stopping at "
                              f"latent frame {last_generated_end}", flush=True)
                        break
                prev_block_latent = curr

            if eos_cosine_threshold is not None or eos_candidate_validator is not None:
                curr_flat = latents.detach().float().reshape(batch_size, -1)
                if prev_block_latent is None:
                    _eos_consecutive = 0
                else:
                    prev_flat = prev_block_latent if prev_block_latent.dim() == 2 else prev_block_latent.reshape(batch_size, -1)
                    cos_sim = torch.nn.functional.cosine_similarity(
                        prev_flat, curr_flat, dim=-1).mean().item()
                    generated_so_far = last_generated_end - num_input_frames
                    if generated_so_far <= 5:
                        _eos_consecutive = 0
                    elif eos_cosine_threshold is None or cos_sim < eos_cosine_threshold:
                        _eos_consecutive += 1
                    else:
                        _eos_consecutive = 0
                    if eos_cosine_threshold is not None:
                        print(f"[blk {last_generated_end:3d}] cos_sim={cos_sim:.4f}"
                              f" (thr={eos_cosine_threshold})", flush=True)
                    if _eos_consecutive >= eos_confirm_frames:
                        if eos_candidate_validator is not None:
                            confirmed = eos_candidate_validator(
                                output[:, :last_generated_end])
                            if not confirmed:
                                print("[eos-stop] EOS candidate rejected by "
                                      "validator; continuing", flush=True)
                                _eos_consecutive = 0
                                if early_stop_patience is None:
                                    prev_block_latent = latents.detach().float()
                                continue
                        detail = (f"cosine_sim < {eos_cosine_threshold} "
                                  if eos_cosine_threshold is not None
                                  else "validator accepted ")
                        print(f"[eos-stop] {detail}for {_eos_consecutive} "
                              f"frames; EOS at latent frame {last_generated_end}",
                              flush=True)
                        break
                if early_stop_patience is None:
                    prev_block_latent = latents.detach().float()

        output = output[:, :last_generated_end]
        action_logits_tensor = None
        if return_action_logits:
            if action_logits_out:
                action_logits_tensor = torch.cat(action_logits_out, dim=1)
            else:
                action_logits_tensor = torch.empty(
                    batch_size, 0, 0, device=output.device, dtype=torch.float32)

        if return_video:
            video = self.vae.decode_to_pixel(output)
            video = (video * 0.5 + 0.5).clamp(0, 1)

            if return_latents:
                if return_action_logits:
                    return video, output, action_logits_tensor
                return video, output
            else:
                if return_action_logits:
                    return video, action_logits_tensor
                return video
        else:
            if return_action_logits:
                return output, action_logits_tensor
            return output

    def inference_for_cd(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        record_step_indices: List[int],
        initial_latent: Optional[torch.Tensor] = None,
        start_frame_index: int = 0
    ):
        self.sampling_steps = 48
        batch_size, num_frames, num_channels, height, width = noise.shape

        if (not self.independent_first_frame) or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames

        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self.text_encoder(text_prompts=[self.args.negative_prompt] * len(text_prompts))

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        if self.kv_cache_pos is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)

        sample_scheduler_probe = self._initialize_sample_scheduler(noise)
        T = len(sample_scheduler_probe.timesteps)
        record_step_indices = sorted(set(int(i) for i in record_step_indices))
        if len(record_step_indices) == 0:
            raise ValueError("record_step_indices must be non-empty")
        if record_step_indices[0] < 0 or record_step_indices[-1] >= T:
            raise ValueError(f"record_step_indices out of range: valid=[0,{T-1}], got={record_step_indices}")
        record_set = set(record_step_indices)

        current_start_frame = start_frame_index
        cache_start_frame = 0

        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0

            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        all_num_frames = [self.num_frame_per_block] * num_blocks

        full_chunk_record = []
        for current_num_frames in all_num_frames:
            noisy_input = noise[:, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            chunk_records = []

            sample_scheduler = self._initialize_sample_scheduler(noise)
            for progress_id, t in enumerate(tqdm(
                    sample_scheduler.timesteps, disable=self.disable_tqdm)):
                if progress_id in record_set:
                    print(f'{progress_id}: {t} saved')
                    chunk_records.append(latents.detach().clone())

                timestep = t * torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.float32)

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (flow_pred_cond - flow_pred_uncond)
                latents = sample_scheduler.step(flow_pred, t, latents, return_dict=False)[0]

            chunk_records.append(latents.detach().clone())
            chunk_records = torch.stack(chunk_records, dim=1)

            full_chunk_record.append(chunk_records)
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            timestep0 = torch.zeros([batch_size, current_num_frames], device=noise.device, dtype=torch.float32)
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )


            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames


        full_chunk_record = torch.cat(full_chunk_record, dim=2)

        return full_chunk_record


    def inference_for_genuine_cd(
        self,
        noisy_input: torch.Tensor,
        conditional_dict = None,
        unconditional_dict = None,
        text_prompts = None,
        initial_latent: Optional[torch.Tensor] = None,
        timestep_idx=0,
        sampling_steps=48,
        chunksize = 3
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noisy_input.shape
        assert num_frames == chunksize
        if initial_latent is not None:
            num_input_frames = initial_latent.shape[1]
            assert num_input_frames % chunksize == 0
            num_output_frames = num_frames + num_input_frames
        else:
            num_output_frames = num_frames

        if conditional_dict is None:
            assert text_prompts is not None
            conditional_dict = self.text_encoder(
                text_prompts=text_prompts
            )
            unconditional_dict = self.text_encoder(
                text_prompts=[self.args.negative_prompt] * len(text_prompts)
            )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noisy_input.device,
            dtype=noisy_input.dtype
        )

        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noisy_input.dtype,
                device=noisy_input.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noisy_input.dtype,
                device=noisy_input.device
            )
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)

        current_start_frame = 0
        cache_start_frame = 0
        timestep = torch.ones([batch_size, 1], device=noisy_input.device, dtype=torch.int64) * 0


        if initial_latent is not None:
            num_input_blocks = num_input_frames // chunksize
            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + chunksize]
                output[:, cache_start_frame:cache_start_frame + chunksize] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += chunksize
                cache_start_frame += chunksize


        latents = noisy_input
        sample_scheduler = self._initialize_sample_scheduler(noisy_input, sampling_steps=sampling_steps)
        t = sample_scheduler.timesteps[timestep_idx]
        latent_model_input = latents
        timestep = t * torch.ones(
            [batch_size, chunksize], device=noisy_input.device, dtype=torch.float32
        )
        flow_pred_cond, _ = self.generator(
            noisy_image_or_video=latent_model_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache_pos,
            crossattn_cache=self.crossattn_cache_pos,
            current_start=current_start_frame * self.frame_seq_length,
            cache_start=cache_start_frame * self.frame_seq_length
        )
        flow_pred_uncond, _ = self.generator(
            noisy_image_or_video=latent_model_input,
            conditional_dict=unconditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache_neg,
            crossattn_cache=self.crossattn_cache_neg,
            current_start=current_start_frame * self.frame_seq_length,
            cache_start=cache_start_frame * self.frame_seq_length
        )
        flow_pred = flow_pred_uncond + self.args.guidance_scale * (
            flow_pred_cond - flow_pred_uncond)

        latents = sample_scheduler.step(
            flow_pred,
            t,
            latents,
            return_dict=False)[0]

        return latents


    def _initialize_kv_cache(self, batch_size, dtype, device, num_frames=None):
        kv_cache_pos = []
        kv_cache_neg = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            min_size = (num_frames or 0) * self.frame_seq_length
            kv_cache_size = max(32760, min_size)

        for _ in range(self.num_transformer_blocks):
            kv_cache_pos.append({
                "k": torch.zeros([batch_size, kv_cache_size, self.num_attention_heads, self.attention_head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, self.num_attention_heads, self.attention_head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_neg.append({
                "k": torch.zeros([batch_size, kv_cache_size, self.num_attention_heads, self.attention_head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, self.num_attention_heads, self.attention_head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache_pos = kv_cache_pos
        self.kv_cache_neg = kv_cache_neg

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        crossattn_cache_pos = []
        crossattn_cache_neg = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros([batch_size, 512, self.num_attention_heads, self.attention_head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, self.num_attention_heads, self.attention_head_dim], dtype=dtype, device=device),
                "is_init": False
            })
            crossattn_cache_neg.append({
                "k": torch.zeros([batch_size, 512, self.num_attention_heads, self.attention_head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, self.num_attention_heads, self.attention_head_dim], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache_pos = crossattn_cache_pos
        self.crossattn_cache_neg = crossattn_cache_neg

    def _initialize_sample_scheduler(self, noise, sampling_steps=-1):
        if sampling_steps == -1:
            sampling_steps = self.sampling_steps
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        elif self.sample_solver == 'sde-dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
                algorithm_type="sde-dpmsolver++",
                solver_order=1)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler
