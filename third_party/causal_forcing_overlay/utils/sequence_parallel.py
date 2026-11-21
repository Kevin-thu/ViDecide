# Added by the ViDecide authors to the Causal-Forcing code base (Apache-2.0): Ulysses sequence parallelism. See NOTICE.
from __future__ import annotations

import torch
import torch.distributed as dist


def _all_to_all(x: torch.Tensor, sp_mesh, scatter_dim: int, gather_dim: int) -> torch.Tensor:
    sp_size = sp_mesh.size()
    if sp_size == 1:
        return x
    sp_group = sp_mesh.get_group()

    chunks = [c.contiguous() for c in x.chunk(sp_size, dim=scatter_dim)]
    recv = [torch.empty_like(chunks[0]) for _ in range(sp_size)]
    dist.all_to_all(recv, chunks, group=sp_group)
    return torch.cat(recv, dim=gather_dim)


class _AllToAllScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sp_mesh, scatter_dim, gather_dim):
        ctx.sp_mesh = sp_mesh
        ctx.sp_size = sp_mesh.size()
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        if ctx.sp_size == 1:
            return x
        return _all_to_all(x, sp_mesh, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.sp_size == 1:
            return grad_output, None, None, None
        grad = _all_to_all(grad_output, ctx.sp_mesh, ctx.gather_dim, ctx.scatter_dim)
        return grad, None, None, None


def ulysses_seq_to_head(x: torch.Tensor, sp_mesh) -> torch.Tensor:
    return _AllToAllScatter.apply(x, sp_mesh, 2, 1)


def ulysses_head_to_seq(x: torch.Tensor, sp_mesh) -> torch.Tensor:
    return _AllToAllScatter.apply(x, sp_mesh, 1, 2)


class _SeqScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sp_mesh):
        ctx.sp_mesh = sp_mesh
        ctx.sp_size = sp_mesh.size()
        ctx.sp_rank = sp_mesh.get_local_rank() if ctx.sp_size > 1 else 0
        if ctx.sp_size == 1:
            return x
        L = x.shape[1]
        assert L % ctx.sp_size == 0, f"seq {L} not divisible by sp_size {ctx.sp_size}"
        chunk = L // ctx.sp_size
        return x[:, ctx.sp_rank * chunk : (ctx.sp_rank + 1) * chunk].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.sp_size == 1:
            return grad_output, None
        sp_group = ctx.sp_mesh.get_group()
        B, L_local = grad_output.shape[0], grad_output.shape[1]
        full_shape = (B, L_local * ctx.sp_size) + tuple(grad_output.shape[2:])
        out = torch.empty(full_shape, device=grad_output.device, dtype=grad_output.dtype)
        out_chunks = [out[:, i * L_local : (i + 1) * L_local] for i in range(ctx.sp_size)]
        dist.all_gather(out_chunks, grad_output.contiguous(), group=sp_group)
        return out, None


class _SeqAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sp_mesh):
        ctx.sp_mesh = sp_mesh
        ctx.sp_size = sp_mesh.size()
        ctx.sp_rank = sp_mesh.get_local_rank() if ctx.sp_size > 1 else 0
        if ctx.sp_size == 1:
            return x
        sp_group = sp_mesh.get_group()
        B, L_local = x.shape[0], x.shape[1]
        full_shape = (B, L_local * ctx.sp_size) + tuple(x.shape[2:])
        gathered = torch.empty(full_shape, device=x.device, dtype=x.dtype)
        out_chunks = [gathered[:, i * L_local : (i + 1) * L_local] for i in range(ctx.sp_size)]
        dist.all_gather(out_chunks, x.contiguous(), group=sp_group)
        return gathered

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.sp_size == 1:
            return grad_output, None
        chunk = grad_output.shape[1] // ctx.sp_size
        return (
            grad_output[:, ctx.sp_rank * chunk : (ctx.sp_rank + 1) * chunk].contiguous(),
            None,
        )


def sp_scatter(x: torch.Tensor, sp_mesh) -> torch.Tensor:
    return _SeqScatter.apply(x, sp_mesh)


def sp_all_gather(x: torch.Tensor, sp_mesh) -> torch.Tensor:
    return _SeqAllGather.apply(x, sp_mesh)


def sp_sum_replicated_grads(model, sp_group) -> None:
    if sp_group is None:
        return
    for p in model.parameters():
        if p.grad is None:
            continue
        if hasattr(p.grad, "placements"):
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=sp_group)


def enable_sp_on_model(model, sp_mesh) -> None:
    if sp_mesh is None or sp_mesh.size() == 1:
        return
    target = model
    for _ in range(6):
        if hasattr(target, "blocks"):
            break
        if hasattr(target, "model"):
            target = target.model
        else:
            break
    if not hasattr(target, "blocks"):
        raise AttributeError(
            "enable_sp_on_model: could not find `blocks` attribute "
            "(expected CausalWanModel.blocks)"
        )
    for blk in target.blocks:
        blk._sp_mesh = sp_mesh
    import torch.distributed as _dist
    if _dist.is_initialized() and _dist.get_rank() == 0:
        print(f"[sp] Ulysses SP activated on {len(target.blocks)} blocks "
              f"(sp_size={sp_mesh.size()})")
