# Modified by the ViDecide authors from Causal-Forcing (https://github.com/thu-ml/Causal-Forcing,
# commit 3e7cc99, Apache-2.0): teacher-forcing attention masks for K-frame segments on a clean history
# (FlexAttention), variable-length buckets and KV-cache inference for Wan2.2-TI2V-5B. See NOTICE.
from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
import os

_dynamo_recompile_limit = int(os.environ.get("CAUSAL_DYNAMO_RECOMPILE_LIMIT", "4096"))
torch._dynamo.config.recompile_limit = _dynamo_recompile_limit
torch._dynamo.config.accumulated_recompile_limit = _dynamo_recompile_limit
torch._dynamo.config.cache_size_limit = max(
    int(getattr(torch._dynamo.config, "cache_size_limit", 8)),
    _dynamo_recompile_limit,
)
_flex_compile_mode = os.environ.get(
    "CAUSAL_FLEX_ATTENTION_COMPILE_MODE",
    "max-autotune-no-cudagraphs",
)
if _flex_compile_mode.lower() in ("0", "false", "none", "eager"):
    pass
else:
    flex_attention = torch.compile(
        flex_attention,
        dynamic=False,
        mode=_flex_compile_mode,
    )


def _frame_buckets_from_env(name: str, default: str) -> list[int]:
    raw = os.environ.get(name, default)
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return sorted(set(out))


_PREFIX_KV_PREFIX_FRAME_BUCKETS = _frame_buckets_from_env(
    "CAUSAL_PREFIX_KV_PREFIX_FRAME_BUCKETS",
    "0,9,17,25,33,41,49,57,65,73,81,89,97,105,113,121,129,145,161,181,213,241",
)


def _bucket_frame_count(num_frames: int, buckets: list[int]) -> int:
    num_frames = int(num_frames)
    for bucket in buckets:
        if bucket >= num_frames:
            return int(bucket)
    return num_frames


def _teacher_forcing_frame_blocks(
    num_frames: int,
    num_frame_per_block: int,
    independent_first_frame: bool,
) -> list[tuple[int, int]]:
    num_frames = int(num_frames)
    block_size = int(num_frame_per_block)
    if num_frames <= 0 or block_size <= 0:
        raise ValueError("num_frames and num_frame_per_block must be positive")

    blocks = []
    start = 0
    if independent_first_frame:
        blocks.append((0, 1))
        start = 1
    remaining = num_frames - start
    if remaining % block_size:
        layout = "1 + N * block_size" if independent_first_frame else "N * block_size"
        raise ValueError(
            f"num_frames={num_frames} must follow {layout} for block_size={block_size}"
        )
    while start < num_frames:
        blocks.append((start, start + block_size))
        start += block_size
    return blocks


def _placeholder_mask_mod(b, h, q_idx, kv_idx):
    return q_idx >= 0


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        sp_mesh=None,
        kv_cache_no_update: bool = False,
        prefix_kv_cache=None,
        prefix_kv_tokens: int = 0,
        prefix_kv_storage_tokens: int = 0,
        prefix_frame_count: int = 0,
    ):
        b, s, d = *x.shape[:2], self.head_dim
        sp_size = sp_mesh.size() if sp_mesh is not None else 1
        if cache_start is None:
            cache_start = current_start

        def qkv_fn(x):
            q = self.norm_q(self.q(x))
            k = self.norm_k(self.k(x))
            v = self.v(x)
            n_local = q.shape[-1] // d
            q = q.view(b, s, n_local, d)
            k = k.view(b, s, n_local, d)
            v = v.view(b, s, n_local, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        n = q.shape[2]

        if prefix_kv_cache is not None:
            full_s = s * sp_size
            assert full_s % 2 == 0, (
                "prefix-KV dense TF expects clean/noisy suffix halves"
            )
            suffix_tokens = full_s // 2
            assert suffix_tokens == seq_lens[0].item(), (
                f"suffix token mismatch: got {suffix_tokens}, "
                f"seq_lens={seq_lens[0].item()}"
            )

            if sp_size > 1:
                from utils.sequence_parallel import (
                    ulysses_seq_to_head, ulysses_head_to_seq,
                )
                sp_rank = sp_mesh.get_local_rank()
                if sp_size != 2:
                    raise NotImplementedError(
                        "prefix-KV dense TF currently supports sp_size=2, "
                        "matching the existing TF clean/noisy split."
                    )
                roped_q = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=int(prefix_frame_count)
                ).type_as(v)
                roped_k = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=int(prefix_frame_count)
                ).type_as(v)
                q_cur = ulysses_seq_to_head(roped_q, sp_mesh)
                k_cur = ulysses_seq_to_head(roped_k, sp_mesh)
                v_cur = ulysses_seq_to_head(v, sp_mesh)

                h_local = k_cur.shape[2]
                h0 = sp_rank * h_local
                h1 = h0 + h_local
                p_end = int(prefix_kv_cache["local_end_index"].item())
                p_keep = int(prefix_kv_tokens) if prefix_kv_tokens else p_end
                p_storage = int(prefix_kv_storage_tokens) if prefix_kv_storage_tokens else p_keep
                p_storage = max(p_storage, p_keep)
                p_start = max(0, p_end - p_keep)
                prefix_k = prefix_kv_cache["k"][:, p_start:p_end, h0:h1]
                prefix_v = prefix_kv_cache["v"][:, p_start:p_end, h0:h1]
                p_pad = p_storage - prefix_k.shape[1]
                if p_pad > 0:
                    zpad = torch.zeros(
                        prefix_k.shape[0], p_pad, prefix_k.shape[2], prefix_k.shape[3],
                        device=prefix_k.device, dtype=prefix_k.dtype,
                    )
                    prefix_k = torch.cat([prefix_k, zpad], dim=1)
                    prefix_v = torch.cat([prefix_v, zpad], dim=1)
                k_all = torch.cat([prefix_k, k_cur], dim=1)
                v_all = torch.cat([prefix_v, v_cur], dim=1)
                q_all = q_cur

                q_len = q_all.shape[1]
                kv_len = k_all.shape[1]
                q_pad = math.ceil(q_len / 128) * 128 - q_len
                kv_pad = math.ceil(kv_len / 128) * 128 - kv_len
                if q_pad > 0:
                    q_all = torch.cat([
                        q_all,
                        torch.zeros(
                            q_all.shape[0], q_pad, q_all.shape[2], q_all.shape[3],
                            device=q_all.device, dtype=q_all.dtype,
                        ),
                    ], dim=1)
                if kv_pad > 0:
                    zpad = torch.zeros(
                        k_all.shape[0], kv_pad, k_all.shape[2], k_all.shape[3],
                        device=k_all.device, dtype=k_all.dtype,
                    )
                    k_all = torch.cat([k_all, zpad], dim=1)
                    v_all = torch.cat([v_all, zpad], dim=1)

                attn_out = flex_attention(
                    query=q_all.transpose(2, 1),
                    key=k_all.transpose(2, 1),
                    value=v_all.transpose(2, 1),
                    block_mask=block_mask,
                )
                if q_pad > 0:
                    attn_out = attn_out[:, :, :-q_pad]
                attn_out = attn_out.transpose(2, 1)
                x = ulysses_head_to_seq(attn_out, sp_mesh)
            else:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                for ii in range(2):
                    rq = causal_rope_apply(
                        q_chunk[ii], grid_sizes, freqs,
                        start_frame=int(prefix_frame_count),
                    ).type_as(v)
                    rk = causal_rope_apply(
                        k_chunk[ii], grid_sizes, freqs,
                        start_frame=int(prefix_frame_count),
                    ).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)
                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                p_end = int(prefix_kv_cache["local_end_index"].item())
                p_keep = int(prefix_kv_tokens) if prefix_kv_tokens else p_end
                p_storage = int(prefix_kv_storage_tokens) if prefix_kv_storage_tokens else p_keep
                p_storage = max(p_storage, p_keep)
                p_start = max(0, p_end - p_keep)
                prefix_k = prefix_kv_cache["k"][:, p_start:p_end]
                prefix_v = prefix_kv_cache["v"][:, p_start:p_end]
                p_pad = p_storage - prefix_k.shape[1]
                if p_pad > 0:
                    zpad = torch.zeros(
                        prefix_k.shape[0], p_pad, prefix_k.shape[2], prefix_k.shape[3],
                        device=prefix_k.device, dtype=prefix_k.dtype,
                    )
                    prefix_k = torch.cat([prefix_k, zpad], dim=1)
                    prefix_v = torch.cat([prefix_v, zpad], dim=1)
                k_all = torch.cat(
                    [prefix_k, roped_key],
                    dim=1,
                )
                v_all = torch.cat(
                    [prefix_v, v],
                    dim=1,
                )
                q_len = roped_query.shape[1]
                kv_len = k_all.shape[1]
                q_pad = math.ceil(q_len / 128) * 128 - q_len
                kv_pad = math.ceil(kv_len / 128) * 128 - kv_len
                if q_pad > 0:
                    roped_query = torch.cat([
                        roped_query,
                        torch.zeros(
                            roped_query.shape[0], q_pad,
                            roped_query.shape[2], roped_query.shape[3],
                            device=roped_query.device, dtype=v.dtype,
                        ),
                    ], dim=1)
                if kv_pad > 0:
                    zpad = torch.zeros(
                        k_all.shape[0], kv_pad, k_all.shape[2], k_all.shape[3],
                        device=k_all.device, dtype=v.dtype,
                    )
                    k_all = torch.cat([k_all, zpad], dim=1)
                    v_all = torch.cat([v_all, zpad], dim=1)

                x = flex_attention(
                    query=roped_query.transpose(2, 1),
                    key=k_all.transpose(2, 1),
                    value=v_all.transpose(2, 1),
                    block_mask=block_mask,
                )
                if q_pad > 0:
                    x = x[:, :, :-q_pad]
                x = x.transpose(2, 1)
        elif kv_cache is None:
            full_s = s * sp_size
            is_tf = (full_s == seq_lens[0].item() * 2)

            if sp_size > 1:
                from utils.sequence_parallel import (
                    ulysses_seq_to_head, ulysses_head_to_seq,
                )
                sp_rank = sp_mesh.get_local_rank()
                if is_tf:
                    roped_q = rope_apply(q, grid_sizes, freqs).type_as(v)
                    roped_k = rope_apply(k, grid_sizes, freqs).type_as(v)
                else:
                    F_frames = int(grid_sizes[0, 0].item())
                    start_frame = sp_rank * (F_frames // sp_size)
                    roped_q = causal_rope_apply(
                        q, grid_sizes, freqs, start_frame=start_frame
                    ).type_as(v)
                    roped_k = causal_rope_apply(
                        k, grid_sizes, freqs, start_frame=start_frame
                    ).type_as(v)

                q_full = ulysses_seq_to_head(roped_q, sp_mesh)
                k_full = ulysses_seq_to_head(roped_k, sp_mesh)
                v_full = ulysses_seq_to_head(v, sp_mesh)

                L_full = q_full.shape[1]
                padded_length = math.ceil(L_full / 128) * 128 - L_full
                if padded_length > 0:
                    pad_shape = (
                        q_full.shape[0], padded_length,
                        q_full.shape[2], q_full.shape[3],
                    )
                    zpad = torch.zeros(
                        pad_shape, device=q_full.device, dtype=v_full.dtype,
                    )
                    q_full = torch.cat([q_full, zpad], dim=1)
                    k_full = torch.cat([k_full, zpad], dim=1)
                    v_full = torch.cat([v_full, zpad], dim=1)

                attn_out = flex_attention(
                    query=q_full.transpose(2, 1),
                    key=k_full.transpose(2, 1),
                    value=v_full.transpose(2, 1),
                    block_mask=block_mask,
                )
                if padded_length > 0:
                    attn_out = attn_out[:, :, :-padded_length]
                attn_out = attn_out.transpose(2, 1)

                x = ulysses_head_to_seq(attn_out, sp_mesh)
            elif is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )
                if padded_length > 0:
                    x = x[:, :, :-padded_length]
                x = x.transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )
                if padded_length > 0:
                    x = x[:, :, :-padded_length]
                x = x.transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            num_new_tokens = roped_query.shape[1]
            if kv_cache_no_update:
                local_end_index = int(kv_cache["local_end_index"].item())
                keep = max(0, self.max_attention_size - num_new_tokens)
                prefix_start = max(0, local_end_index - keep)
                attn_k = torch.cat(
                    [kv_cache["k"][:, prefix_start:local_end_index], roped_key],
                    dim=1,
                )
                attn_v = torch.cat(
                    [kv_cache["v"][:, prefix_start:local_end_index], v],
                    dim=1,
                )
                x = attention(roped_query, attn_k, attn_v)
            else:
                kv_cache_size = kv_cache["k"].shape[1]
                if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                        num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                    num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                    num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                    kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    local_end_index = kv_cache["local_end_index"].item() + current_end - \
                        kv_cache["global_end_index"].item() - num_evicted_tokens
                    local_start_index = local_end_index - num_new_tokens
                    kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                    kv_cache["v"][:, local_start_index:local_end_index] = v
                else:
                    local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                    local_start_index = local_end_index - num_new_tokens
                    kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                    kv_cache["v"][:, local_start_index:local_end_index] = v
                x = attention(
                    roped_query,
                    kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                    kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
                )
                kv_cache["global_end_index"].fill_(current_end)
                kv_cache["local_end_index"].fill_(local_end_index)

        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        kv_cache_no_update: bool = False,
        prefix_kv_cache=None,
        prefix_kv_tokens: int = 0,
        prefix_kv_storage_tokens: int = 0,
        prefix_frame_count: int = 0,
    ):
        sp_mesh = getattr(self, "_sp_mesh", None)

        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start,
            sp_mesh=sp_mesh,
            kv_cache_no_update=kv_cache_no_update,
            prefix_kv_cache=prefix_kv_cache,
            prefix_kv_tokens=prefix_kv_tokens,
            prefix_kv_storage_tokens=prefix_kv_storage_tokens,
            prefix_frame_count=prefix_frame_count,
        )

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        cross_attn_type = 'i2v_cross_attn' if model_type == 'i2v' else 't2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        self.head = CausalHead(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None
        self._block_mask_cache = {}
        self._prefix_kv_tf_mask_cache = {}

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module=None, value=False, enable=None, gradient_checkpointing_func=None):
        if enable is not None:
            value = enable
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        total_length = num_frames * frame_seqlen

        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)


        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1,
        independent_first_frame: bool = False,
    ) -> BlockMask:
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        frame_blocks = _teacher_forcing_frame_blocks(
            num_frames, num_frame_per_block, independent_first_frame,
        )
        for frame_start, frame_end in frame_blocks:
            clean_start = frame_start * frame_seqlen
            clean_end = frame_end * frame_seqlen
            context_ends[clean_start:clean_end] = clean_end

            noisy_start = clean_ends + clean_start
            noisy_end = clean_ends + clean_end
            noise_noise_starts[noisy_start:noisy_end] = noisy_start
            noise_noise_ends[noisy_start:noisy_end] = noisy_end
            noise_context_ends[noisy_start:noisy_end] = clean_start

        def attention_mask(b, h, q_idx, kv_idx):
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        _seq_tokens = total_length + padded_length
        if _seq_tokens > 50000:
            _cpu_helpers = [t.cpu() for t in [context_ends, noise_context_starts,
                            noise_context_ends, noise_noise_starts, noise_noise_ends]]
            context_ends, noise_context_starts, noise_context_ends, \
                noise_noise_starts, noise_noise_ends = _cpu_helpers
            block_mask = create_block_mask(attention_mask, B=None, H=None,
                                           Q_LEN=_seq_tokens, KV_LEN=_seq_tokens,
                                           _compile=False, device='cpu')
            block_mask = block_mask.to(device)
            context_ends = context_ends.to(device)
            noise_context_starts = noise_context_starts.to(device)
            noise_context_ends = noise_context_ends.to(device)
            noise_noise_starts = noise_noise_starts.to(device)
            noise_noise_ends = noise_noise_ends.to(device)
            def attention_mask_gpu(b, h, q_idx, kv_idx):
                clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
                C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
                C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
                noise_mask = (q_idx >= clean_ends) & (C1 | C2)
                eye_mask = q_idx == kv_idx
                return eye_mask | clean_mask | noise_mask
            block_mask.mask_mod = attention_mask_gpu
        else:
            block_mask = create_block_mask(attention_mask, B=None, H=None,
                                           Q_LEN=_seq_tokens, KV_LEN=_seq_tokens,
                                           _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                                padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_prefix_kv_teacher_forcing_mask(
        device: torch.device | str,
        prefix_tokens: int,
        suffix_frames: int,
        frame_seqlen: int = 1560,
        num_frame_per_block=1,
        prefix_storage_tokens: int | None = None,
    ) -> BlockMask:
        prefix_tokens = int(prefix_tokens)
        prefix_storage_tokens = (
            int(prefix_storage_tokens)
            if prefix_storage_tokens is not None
            else prefix_tokens
        )
        prefix_storage_tokens = max(prefix_storage_tokens, prefix_tokens)
        suffix_tokens = int(suffix_frames) * int(frame_seqlen)
        q_length = suffix_tokens * 2
        kv_length = prefix_storage_tokens + suffix_tokens * 2
        q_padded = math.ceil(q_length / 128) * 128
        kv_padded = math.ceil(kv_length / 128) * 128
        attention_block_size = int(frame_seqlen) * int(num_frame_per_block)

        def attention_mask(b, h, q_idx, kv_idx):
            q_valid = q_idx < q_length
            kv_valid = (kv_idx < prefix_tokens) | (
                (kv_idx >= prefix_storage_tokens) & (kv_idx < kv_length)
            )

            prefix_mask = kv_idx < prefix_tokens
            clean_kv_pos = kv_idx - prefix_storage_tokens
            noisy_kv_pos = kv_idx - prefix_storage_tokens - suffix_tokens
            clean_kv = (clean_kv_pos >= 0) & (clean_kv_pos < suffix_tokens)
            noisy_kv = (noisy_kv_pos >= 0) & (noisy_kv_pos < suffix_tokens)

            clean_q = q_idx < suffix_tokens
            clean_q_pos = q_idx
            clean_block_end = (
                (clean_q_pos // attention_block_size) + 1
            ) * attention_block_size
            clean_mask = clean_q & (
                prefix_mask | (clean_kv & (clean_kv_pos < clean_block_end))
            )

            noisy_q = (q_idx >= suffix_tokens) & (q_idx < q_length)
            noisy_q_pos = q_idx - suffix_tokens
            noisy_block_start = (
                noisy_q_pos // attention_block_size
            ) * attention_block_size
            noisy_block_end = noisy_block_start + attention_block_size
            noisy_mask = noisy_q & (
                prefix_mask
                | (clean_kv & (clean_kv_pos < noisy_block_start))
                | (
                    noisy_kv
                    & (noisy_kv_pos >= noisy_block_start)
                    & (noisy_kv_pos < noisy_block_end)
                )
            )

            return q_valid & kv_valid & (clean_mask | noisy_mask)

        block_mask = create_block_mask(
            attention_mask, B=None, H=None,
            Q_LEN=q_padded, KV_LEN=kv_padded,
            _compile=False, device=device,
        )
        return block_mask

    @staticmethod
    def _build_tf_mask_cpu(num_frames: int, frame_seqlen: int,
                           num_frame_per_block: int = 1,
                           independent_first_frame: bool = False):
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        clean_ends = num_frames * frame_seqlen
        seq_tokens = total_length + padded_length

        context_ends = torch.zeros(seq_tokens, dtype=torch.long)
        noise_context_starts = torch.zeros(seq_tokens, dtype=torch.long)
        noise_context_ends = torch.zeros(seq_tokens, dtype=torch.long)
        noise_noise_starts = torch.zeros(seq_tokens, dtype=torch.long)
        noise_noise_ends = torch.zeros(seq_tokens, dtype=torch.long)

        frame_blocks = _teacher_forcing_frame_blocks(
            num_frames, num_frame_per_block, independent_first_frame,
        )
        for frame_start, frame_end in frame_blocks:
            clean_start = frame_start * frame_seqlen
            clean_end = frame_end * frame_seqlen
            context_ends[clean_start:clean_end] = clean_end
            noisy_start = clean_ends + clean_start
            noisy_end = clean_ends + clean_end
            noise_noise_starts[noisy_start:noisy_end] = noisy_start
            noise_noise_ends[noisy_start:noisy_end] = noisy_end
            noise_context_ends[noisy_start:noisy_end] = clean_start

        def attention_mask_cpu(b, h, q_idx, kv_idx):
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)
            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(
            attention_mask_cpu, B=None, H=None,
            Q_LEN=seq_tokens, KV_LEN=seq_tokens,
            _compile=False, device="cpu",
        )
        block_mask.mask_mod = _placeholder_mask_mod
        helpers = {
            "clean_ends": clean_ends,
            "independent_first_frame": bool(independent_first_frame),
            "context_ends": context_ends,
            "noise_context_starts": noise_context_starts,
            "noise_context_ends": noise_context_ends,
            "noise_noise_starts": noise_noise_starts,
            "noise_noise_ends": noise_noise_ends,
        }
        return block_mask, helpers

    @staticmethod
    def _finalize_tf_mask(block_mask, helpers, device):
        block_mask = block_mask.to(device)
        clean_ends = helpers["clean_ends"]
        context_ends = helpers["context_ends"].to(device)
        noise_context_starts = helpers["noise_context_starts"].to(device)
        noise_context_ends = helpers["noise_context_ends"].to(device)
        noise_noise_starts = helpers["noise_noise_starts"].to(device)
        noise_noise_ends = helpers["noise_noise_ends"].to(device)

        def attention_mask_dev(b, h, q_idx, kv_idx):
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)
            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask.mask_mod = attention_mask_dev
        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        total_length = num_frames * frame_seqlen

        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        ends[:frame_seqlen] = frame_seqlen

        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)


        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        kv_cache_no_update: bool = False,
    ):

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "kv_cache_no_update": kv_cache_no_update,
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "kv_cache_no_update": kv_cache_no_update,
                    }
                )
                x = block(x, **kwargs)

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
        return_features: bool = False,
        prefix_kv_cache: dict = None,
        prefix_frame_count: int = 0,
    ):
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        current_nf = x.shape[2]
        frame_seqlen = x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2])
        prefix_kv_tokens = 0
        prefix_kv_storage_tokens = 0
        if prefix_kv_cache is not None:
            prefix_kv_tokens = int(prefix_kv_cache[0]["local_end_index"].item())
            prefix_kv_frames = math.ceil(prefix_kv_tokens / frame_seqlen) if prefix_kv_tokens else 0
            prefix_kv_storage_frames = _bucket_frame_count(
                prefix_kv_frames, _PREFIX_KV_PREFIX_FRAME_BUCKETS,
            )
            prefix_kv_storage_tokens = int(prefix_kv_storage_frames) * int(frame_seqlen)
            mask_key = (prefix_kv_tokens, prefix_kv_storage_tokens, current_nf)
            if mask_key in self._prefix_kv_tf_mask_cache:
                self.block_mask = self._prefix_kv_tf_mask_cache[mask_key]
            else:
                self.block_mask = self._prepare_prefix_kv_teacher_forcing_mask(
                    device,
                    prefix_tokens=prefix_kv_tokens,
                    prefix_storage_tokens=prefix_kv_storage_tokens,
                    suffix_frames=current_nf,
                    frame_seqlen=frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block,
                )
                self._prefix_kv_tf_mask_cache[mask_key] = self.block_mask
        elif current_nf in self._block_mask_cache:
            self.block_mask = self._block_mask_cache[current_nf]
        elif self.block_mask is None or getattr(self, '_block_mask_nf', None) != current_nf:
            if clean_x is not None:
                self.block_mask = self._prepare_teacher_forcing_mask(
                    device, num_frames=current_nf,
                    frame_seqlen=frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block,
                    independent_first_frame=self.independent_first_frame,
                )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=current_nf,
                        frame_seqlen=frame_seqlen,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=current_nf,
                        frame_seqlen=frame_seqlen,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
            self._block_mask_cache[current_nf] = self.block_mask
            self._block_mask_nf = current_nf

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        sp_mesh = getattr(self.blocks[0], "_sp_mesh", None) if len(self.blocks) else None
        sp_size = sp_mesh.size() if sp_mesh is not None else 1

        if sp_size > 1:
            from utils.sequence_parallel import sp_scatter, sp_all_gather
            sp_rank = sp_mesh.get_local_rank()
            F_total = e0.shape[1]
            assert F_total % sp_size == 0, (
                f"SP requires frame count ({F_total}) divisible by sp_size ({sp_size})"
            )
            F_local = F_total // sp_size
            e_local = e0[:, sp_rank * F_local : (sp_rank + 1) * F_local, :, :]
            x = sp_scatter(x, sp_mesh)
            kwargs_sp = dict(kwargs)
            kwargs_sp["e"] = e_local
        else:
            kwargs_sp = kwargs

        for block_index, block in enumerate(self.blocks):
            block_kwargs = kwargs_sp
            if prefix_kv_cache is not None:
                block_kwargs = dict(kwargs_sp)
                block_kwargs["prefix_kv_cache"] = prefix_kv_cache[block_index]
                block_kwargs["prefix_kv_tokens"] = prefix_kv_tokens
                block_kwargs["prefix_kv_storage_tokens"] = prefix_kv_storage_tokens
                block_kwargs["prefix_frame_count"] = int(prefix_frame_count)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **block_kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **block_kwargs)

        if sp_size > 1:
            x = sp_all_gather(x, sp_mesh)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]
        frame_features = None
        if return_features:
            frame_seqlen = int(frame_seqlen)
            frame_features = x.reshape(
                x.shape[0], current_nf, frame_seqlen, x.shape[-1]
            ).mean(dim=2)
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        x = self.unpatchify(x, grid_sizes)
        out = torch.stack(x)
        if return_features:
            return out, frame_features
        return out

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        nn.init.zeros_(self.head.head.weight)
