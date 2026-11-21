# Modified by the ViDecide authors from Causal-Forcing (https://github.com/thu-ml/Causal-Forcing,
# commit 3e7cc99, Apache-2.0): Wan2.2-TI2V-5B support (48-channel Wan2.2 VAE through diffusers, model folders) and
# KV-cache arguments for segment-wise generation. See NOTICE.
import types
from pathlib import Path
from typing import List, Optional
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel


WAN21_VAE_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
]
WAN21_VAE_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
]
WAN22_TI2V5B_VAE_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.157, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667
]
WAN22_TI2V5B_VAE_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744
]


def _wan_model_dir(model_name: str) -> Path:
    path = Path(model_name)
    if path.exists():
        return path
    return Path("wan_models") / model_name


def _vae_spec(model_name: str):
    if "Wan2.2-TI2V-5B" in model_name:
        return "Wan2.2_VAE.pth", 48, WAN22_TI2V5B_VAE_MEAN, WAN22_TI2V5B_VAE_STD
    return "Wan2.1_VAE.pth", 16, WAN21_VAE_MEAN, WAN21_VAE_STD


class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_name="Wan2.1-T2V-1.3B") -> None:
        super().__init__()
        model_dir = _wan_model_dir(model_name)

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=str(model_dir / "google/umt5-xxl/"), seq_len=512, clean='whitespace')

    @property
    def device(self):
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_name="Wan2.1-T2V-1.3B"):
        super().__init__()
        model_dir = _wan_model_dir(model_name)
        vae_name, z_dim, mean, std = _vae_spec(model_name)
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.z_dim = z_dim
        self._diffusers_vae = "Wan2.2-TI2V-5B" in model_name

        if self._diffusers_vae:
            from diffusers import AutoencoderKLWan
            # <model>-Diffusers/vae next to the model folder as given, or next to the folder a symlink points to
            candidates = [Path(str(model_dir) + "-Diffusers") / "vae", Path(str(model_dir.resolve()) + "-Diffusers") / "vae"]
            vae_dir = next((path for path in candidates if path.exists()), candidates[0])
            if not vae_dir.exists():
                raise FileNotFoundError(
                    f"Wan2.2 TI2V-5B diffusers VAE not found: {vae_dir}"
                )
            self.model = AutoencoderKLWan.from_pretrained(str(vae_dir))
            self.model.eval().requires_grad_(False)
        else:
            self.model = _video_vae(
                pretrained_path=model_dir / vae_name,
                z_dim=z_dim,
            ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if self._diffusers_vae:
            model_param = next(self.model.parameters())
            if model_param.device != pixel.device or model_param.dtype != dtype:
                self.model.to(device=pixel.device, dtype=dtype)
            model_dtype = next(self.model.parameters()).dtype
            latent = self.model.encode(pixel.to(device=pixel.device, dtype=model_dtype)).latent_dist.mode()
            scale = [self.mean.to(device=device, dtype=model_dtype),
                     1.0 / self.std.to(device=device, dtype=model_dtype)]
            output = (latent - scale[0].view(1, self.z_dim, 1, 1, 1)) * \
                scale[1].view(1, self.z_dim, 1, 1, 1)
            output = output.to(dtype=dtype)
        else:
            output = [
                self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
                for u in pixel
            ]
            output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache and not self._diffusers_vae:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        if self._diffusers_vae:
            model_param = next(self.model.parameters())
            if model_param.device != device or model_param.dtype != dtype:
                self.model.to(device=device, dtype=dtype)
            model_dtype = next(self.model.parameters()).dtype
            zs = zs.to(dtype=model_dtype)
            scale = [self.mean.to(device=device, dtype=model_dtype),
                     1.0 / self.std.to(device=device, dtype=model_dtype)]
            zs = zs / scale[1].view(1, self.z_dim, 1, 1, 1) + \
                scale[0].view(1, self.z_dim, 1, 1, 1)
            output = decode_function(zs, return_dict=False)[0].float().clamp_(-1, 1)
        else:
            output = []
            for u in zs:
                output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
            output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0
    ):
        super().__init__()

        model_dir = _wan_model_dir(model_name)
        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                str(model_dir), local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.model = WanModel.from_pretrained(str(model_dir))
        self.model.eval()

        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760
        self.action_head = None
        self.detach_action_features = False
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def add_action_head(self, num_actions: int, hidden_dim: Optional[int] = None) -> None:
        hidden_dim = int(hidden_dim or getattr(self.model, "dim"))
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, int(num_actions)),
        )
        self.action_head.requires_grad_(True)

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        self._cls_pred_branch = nn.Sequential(
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,

        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        return_features: Optional[bool] = False,
        return_action_logits: Optional[bool] = False,

        cache_start: Optional[int] = None,
        kv_cache_no_update: Optional[bool] = False,
        prefix_kv_cache: Optional[List[dict]] = None,
        prefix_frame_count: Optional[int] = 0,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        _B, _F, _C, _H, _W = noisy_image_or_video.shape
        _ps1, _ps2 = self.model.patch_size[1], self.model.patch_size[2]
        _tokens_per_frame = (_H // _ps1) * (_W // _ps2)
        _multiplier = 2 if clean_x is not None else 1
        _dynamic_seq_len = _F * _tokens_per_frame * _multiplier
        _seq_len = max(self.seq_len, _dynamic_seq_len)

        logits = None
        frame_features = None
        if kv_cache is not None:
            if return_features or return_action_logits:
                raise NotImplementedError(
                    "KV inference does not expose frame features/action logits; "
                    "run a separate teacher-forcing read pass for action logits."
                )
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=_seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                kv_cache_no_update=bool(kv_cache_no_update),
            )
            flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=_seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                    return_features=return_features or return_action_logits,
                    prefix_kv_cache=prefix_kv_cache,
                    prefix_frame_count=int(prefix_frame_count or 0),
                )
                if return_features or return_action_logits:
                    flow_pred, frame_features = flow_pred
                flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=_seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=_seq_len
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits
        if return_action_logits:
            if self.action_head is None:
                raise RuntimeError("return_action_logits=True but action_head is not initialized")
            head_dtype = next(self.action_head.parameters()).dtype
            head_features = frame_features.to(dtype=head_dtype)
            if bool(getattr(self, "detach_action_features", False)):
                head_features = head_features.detach()
            action_logits = self.action_head(head_features).float()
            if return_features:
                return flow_pred, pred_x0, frame_features, action_logits
            return flow_pred, pred_x0, action_logits
        if return_features:
            return flow_pred, pred_x0, frame_features

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        self.get_scheduler()
