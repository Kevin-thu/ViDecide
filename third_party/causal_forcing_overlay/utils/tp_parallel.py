# Added by the ViDecide authors to the Causal-Forcing code base (Apache-2.0): tensor parallelism. See NOTICE.
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        SequenceParallel,
        parallelize_module,
    )
    _TP_AVAILABLE = True
except ImportError:
    _TP_AVAILABLE = False


class TPRowRMSNorm(nn.Module):

    def __init__(self, full_dim: int, tp_mesh, eps: float = 1e-6):
        super().__init__()
        self.full_dim = int(full_dim)
        self.eps = float(eps)
        self.tp_mesh = tp_mesh
        self.tp_group = tp_mesh.get_group()
        tp_world = tp_mesh.size()
        assert self.full_dim % tp_world == 0, (
            f"full_dim={self.full_dim} not divisible by tp_world={tp_world}"
        )
        self.local_dim = self.full_dim // tp_world
        weight_full = torch.ones(self.full_dim)
        self.weight = nn.Parameter(distribute_tensor(weight_full, tp_mesh, [Shard(0)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        local_sq = x_fp32.pow(2).sum(dim=-1, keepdim=True)
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM, group=self.tp_group)
        rms = torch.rsqrt(local_sq / self.full_dim + self.eps)
        w_local = self.weight.to_local() if hasattr(self.weight, "to_local") else self.weight
        return (x_fp32 * rms).type_as(x) * w_local


class _SPScatterSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp_mesh):
        ctx.tp_mesh = tp_mesh
        ctx.tp_rank = tp_mesh.get_local_rank()
        ctx.tp_size = tp_mesh.size()
        if ctx.tp_size == 1:
            return x
        L = x.shape[1]
        assert L % ctx.tp_size == 0, (
            f"sp_scatter_seq: seq_len {L} not divisible by tp_size {ctx.tp_size}"
        )
        chunk = L // ctx.tp_size
        return x[:, ctx.tp_rank * chunk : (ctx.tp_rank + 1) * chunk].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.tp_size == 1:
            return grad_output, None
        B, L_local = grad_output.shape[0], grad_output.shape[1]
        full_shape = (B, L_local * ctx.tp_size) + tuple(grad_output.shape[2:])
        out = torch.zeros(
            full_shape, device=grad_output.device, dtype=grad_output.dtype
        )
        out[:, ctx.tp_rank * L_local : (ctx.tp_rank + 1) * L_local] = grad_output
        return out, None


class _SPAllGatherSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp_mesh):
        ctx.tp_mesh = tp_mesh
        ctx.tp_group = tp_mesh.get_group()
        ctx.tp_rank = tp_mesh.get_local_rank()
        ctx.tp_size = tp_mesh.size()
        if ctx.tp_size == 1:
            return x
        B, L_local = x.shape[0], x.shape[1]
        full_shape = (B, L_local * ctx.tp_size) + tuple(x.shape[2:])
        gathered = torch.empty(full_shape, device=x.device, dtype=x.dtype)
        out_chunks = [
            gathered[:, i * L_local : (i + 1) * L_local] for i in range(ctx.tp_size)
        ]
        dist.all_gather(out_chunks, x.contiguous(), group=ctx.tp_group)
        return gathered

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.tp_size == 1:
            return grad_output, None
        chunk = grad_output.shape[1] // ctx.tp_size
        return (
            grad_output[:, ctx.tp_rank * chunk : (ctx.tp_rank + 1) * chunk].contiguous(),
            None,
        )


def tp_sum_replicated_grads(model, tp_group) -> None:
    if tp_group is None:
        return
    for p in model.parameters():
        if p.grad is None:
            continue
        if hasattr(p.grad, "placements"):
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=tp_group)


def sp_scatter_seq(x: torch.Tensor, tp_mesh) -> torch.Tensor:
    return _SPScatterSeq.apply(x, tp_mesh)


def sp_all_gather_seq(x: torch.Tensor, tp_mesh) -> torch.Tensor:
    return _SPAllGatherSeq.apply(x, tp_mesh)


def _replace_qk_norm(attn_module: nn.Module, tp_mesh) -> None:
    tp_rank = tp_mesh.get_local_rank()
    tp_world = tp_mesh.size()
    for name in ("norm_q", "norm_k"):
        orig = getattr(attn_module, name, None)
        if orig is None or not hasattr(orig, "weight"):
            continue
        full_dim = int(orig.weight.shape[0])
        eps = float(getattr(orig, "eps", 1e-6))
        new_norm = TPRowRMSNorm(full_dim, tp_mesh, eps=eps)
        local_dim = full_dim // tp_world
        start = tp_rank * local_dim
        local_slice = orig.weight.detach()[start : start + local_dim].to(
            device=new_norm.weight.to_local().device,
            dtype=new_norm.weight.to_local().dtype,
        )
        with torch.no_grad():
            new_norm.weight.to_local().copy_(local_slice)
        setattr(attn_module, name, new_norm)


def _build_block_plan(block: nn.Module, use_sp: bool):
    plan = {
        "self_attn.q": ColwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(-1), use_local_output=True
        ),
        "self_attn.k": ColwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(-1), use_local_output=True
        ),
        "self_attn.v": ColwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(-1), use_local_output=True
        ),
        "self_attn.o": RowwiseParallel(
            input_layouts=Shard(-1), output_layouts=Replicate(), use_local_output=True
        ),
        "cross_attn.q": ColwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(-1), use_local_output=True
        ),
        "cross_attn.k": ColwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(-1), use_local_output=True
        ),
        "cross_attn.v": ColwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(-1), use_local_output=True
        ),
        "cross_attn.o": RowwiseParallel(
            input_layouts=Shard(-1), output_layouts=Replicate(), use_local_output=True
        ),
        "ffn.0": ColwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(-1), use_local_output=True
        ),
        "ffn.2": RowwiseParallel(
            input_layouts=Shard(-1), output_layouts=Replicate(), use_local_output=True
        ),
    }
    return plan


def parallelize_causal_block(block: nn.Module, tp_mesh, use_sp: bool = False) -> None:
    plan = _build_block_plan(block, use_sp=use_sp)
    parallelize_module(block, tp_mesh, plan)
    _replace_qk_norm(block.self_attn, tp_mesh)
    _replace_qk_norm(block.cross_attn, tp_mesh)
    if use_sp:
        block._sp_tp_mesh = tp_mesh


def parallelize_causal_model(model: nn.Module, tp_mesh, use_sp: bool = False) -> nn.Module:
    if tp_mesh is None or (_TP_AVAILABLE and tp_mesh.size() == 1):
        return model
    if not _TP_AVAILABLE:
        raise RuntimeError(
            "torch.distributed.tensor.parallel not available — need torch>=2.2"
        )

    target = model
    depth = 0
    while not hasattr(target, "blocks") and hasattr(target, "model"):
        target = target.model
        depth += 1
        if depth > 5:
            break
    if not hasattr(target, "blocks"):
        raise AttributeError(
            "parallelize_causal_model: could not find `blocks` attribute "
            "(expected CausalWanModel.blocks)"
        )

    if dist.is_initialized() and dist.get_rank() == 0:
        print(
            f"[tp] parallelizing {len(target.blocks)} causal blocks "
            f"(tp={tp_mesh.size()}, sp={use_sp})"
        )
    for blk in target.blocks:
        parallelize_causal_block(blk, tp_mesh, use_sp=use_sp)
    return model
