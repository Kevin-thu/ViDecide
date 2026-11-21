# Modified by the ViDecide authors from Causal-Forcing (https://github.com/thu-ml/Causal-Forcing,
# commit 3e7cc99, Apache-2.0): FSDP2, tensor- and sequence-parallel helpers. See NOTICE.
from datetime import timedelta
from functools import partial
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

try:
    from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
    _DEVICE_MESH_AVAILABLE = True
except ImportError:
    DeviceMesh = None
    _DEVICE_MESH_AVAILABLE = False

_PARALLEL_MESH = None


def init_parallel_mesh(tp_size: int = 1, dp_size=None, sp_size: int = 1):
    global _PARALLEL_MESH
    if not _DEVICE_MESH_AVAILABLE:
        if tp_size != 1 or sp_size != 1:
            raise RuntimeError(
                "init_device_mesh not available; tp_size/sp_size > 1 requires torch>=2.2"
            )
        _PARALLEL_MESH = None
        return None

    world = dist.get_world_size()
    tp_size = max(1, int(tp_size))
    sp_size = max(1, int(sp_size))
    if dp_size in (None, 0, "null"):
        denom = tp_size * sp_size
        if world % denom != 0:
            raise ValueError(
                f"world_size={world} is not divisible by tp_size*sp_size={denom}. "
                f"Pick a valid combination (dp×tp×sp == world)."
            )
        dp_size = world // denom
    if dp_size * tp_size * sp_size != world:
        raise ValueError(
            f"dp_size*tp_size*sp_size must equal world_size: "
            f"{dp_size}*{tp_size}*{sp_size} != {world}"
        )

    backend_override = {}
    mesh_timeout_minutes = int(os.environ.get("CAUSAL_MESH_TIMEOUT_MINUTES", "30"))
    for dim_name, dim_size in (("dp", dp_size), ("tp", tp_size), ("sp", sp_size)):
        if int(dim_size) <= 1:
            continue
        opts = dist.ProcessGroupNCCL.Options()
        opts._timeout = timedelta(minutes=mesh_timeout_minutes)
        backend_override[dim_name] = ("nccl", opts)

    mesh = init_device_mesh(
        "cuda",
        (dp_size, tp_size, sp_size),
        mesh_dim_names=("dp", "tp", "sp"),
        backend_override=backend_override or None,
    )
    _PARALLEL_MESH = mesh
    if dist.get_rank() == 0:
        print(
            f"[parallel] DeviceMesh initialised: dp={dp_size}, tp={tp_size}, "
            f"sp={sp_size} (world={world}, mesh_timeout={mesh_timeout_minutes}min)"
        )
    return mesh


def get_parallel_mesh():
    return _PARALLEL_MESH


def get_dp_group():
    if _PARALLEL_MESH is None:
        return None
    tp_size = _PARALLEL_MESH["tp"].size()
    try:
        sp_size = _PARALLEL_MESH["sp"].size()
    except (KeyError, IndexError):
        sp_size = 1
    if tp_size == 1 and sp_size == 1:
        return None
    return _PARALLEL_MESH["dp"].get_group()


def get_tp_group():
    if _PARALLEL_MESH is None:
        return None
    tp_mesh = _PARALLEL_MESH["tp"]
    if tp_mesh.size() == 1:
        return None
    return tp_mesh.get_group()


def get_tp_size() -> int:
    if _PARALLEL_MESH is None:
        return 1
    return _PARALLEL_MESH["tp"].size()


def get_dp_size() -> int:
    if _PARALLEL_MESH is None:
        return dist.get_world_size() if dist.is_initialized() else 1
    return _PARALLEL_MESH["dp"].size()


def get_sp_size() -> int:
    if _PARALLEL_MESH is None:
        return 1
    try:
        return _PARALLEL_MESH["sp"].size()
    except (KeyError, IndexError):
        return 1


def get_sp_group():
    if _PARALLEL_MESH is None:
        return None
    try:
        sp_mesh = _PARALLEL_MESH["sp"]
    except (KeyError, IndexError):
        return None
    if sp_mesh.size() == 1:
        return None
    return sp_mesh.get_group()


def get_sp_mesh():
    if _PARALLEL_MESH is None:
        return None
    try:
        sp_mesh = _PARALLEL_MESH["sp"]
    except (KeyError, IndexError):
        return None
    if sp_mesh.size() == 1:
        return None
    return sp_mesh


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, cpu_offload=False, process_group=None):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    _orig_strategy = sharding_strategy
    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    if process_group is not None and sharding_strategy in (
        ShardingStrategy.HYBRID_SHARD,
        ShardingStrategy._HYBRID_SHARD_ZERO2,
    ):
        import warnings as _w
        _w.warn(
            f"fsdp_wrap: sharding_strategy={_orig_strategy!r} is incompatible "
            "with a single-PG DP sub-group (tp>1). Downgrading to FULL_SHARD.",
            RuntimeWarning,
            stacklevel=2,
        )
        sharding_strategy = ShardingStrategy.FULL_SHARD

    fsdp_kwargs = dict(
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False,
    )
    if process_group is not None:
        fsdp_kwargs["process_group"] = process_group

    module = FSDP(module, **fsdp_kwargs)
    return module


def compute_clip_grad_norm_2d(params, max_norm: float, eps: float = 1e-6):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    total_sq = torch.zeros((), device="cuda", dtype=torch.float32)
    for p in params:
        g = p.grad
        if g is None:
            continue
        g = g.detach()
        if hasattr(g, "placements") and hasattr(g, "device_mesh"):
            mesh = g.device_mesh
            placements = g.placements
            mesh_world = mesh.size()
            repl_factor = 1
            for i, pl in enumerate(placements):
                is_repl = getattr(pl, "is_replicate", None)
                if callable(is_repl) and is_repl():
                    repl_factor *= mesh.size(i)
            total_replication = (world_size // mesh_world) * repl_factor
            g_local = g.to_local().float()
            total_sq = total_sq + g_local.pow(2).sum() / total_replication
        else:
            g_local = g.float()
            total_sq = total_sq + g_local.pow(2).sum() / world_size
    if dist.is_initialized():
        dist.all_reduce(total_sq, op=dist.ReduceOp.SUM)
    total_norm = total_sq.sqrt()
    clip_coef = torch.where(
        total_norm > 0,
        (max_norm / (total_norm + eps)).clamp(max=1.0),
        torch.ones((), device=total_norm.device, dtype=total_norm.dtype),
    )
    for p in params:
        if p.grad is not None:
            p.grad.mul_(clip_coef)
    return total_norm


def unwrap_module(m):
    inner = getattr(m, "module", None)
    if isinstance(inner, torch.nn.Module):
        return inner
    return m


def fsdp2_wrap(
    module,
    dp_mesh,
    mixed_precision: bool = False,
    cpu_offload: bool = False,
    transformer_cls=None,
):
    try:
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
    except ImportError as e:
        raise RuntimeError(
            "FSDP2 (fully_shard) not available; need torch>=2.4 for clean FSDP+TP composition."
        ) from e

    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    offload_policy = None
    if cpu_offload:
        offload_policy = CPUOffloadPolicy()

    kwargs = {"mesh": dp_mesh}
    if mp_policy is not None:
        kwargs["mp_policy"] = mp_policy
    if offload_policy is not None:
        kwargs["offload_policy"] = offload_policy

    if transformer_cls is not None:
        for sub in module.modules():
            if isinstance(sub, transformer_cls):
                fully_shard(sub, **kwargs)
    fully_shard(module, **kwargs)
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:
        init_method = f"tcp://[{host}]:{port}"
    else:
        init_method = f"tcp://{host}:{port}"
    timeout_minutes = int(os.environ.get("CAUSAL_DIST_TIMEOUT_MINUTES", "30"))
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timedelta(minutes=timeout_minutes))
    torch.cuda.set_device(local_rank)


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        for n, p in fsdp_module.module.named_parameters():
            self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        for n, p in fsdp_module.module.named_parameters():
            self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1. - d)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, fsdp_module):
        for n, p in fsdp_module.module.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(dtype=p.dtype, device=p.device))

    @torch.no_grad()
    def full_state_dict(self, fsdp_module):
        live_state = {}
        for n, p in fsdp_module.module.named_parameters():
            live_state[n] = p.detach().clone()
        for n, p in fsdp_module.module.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(dtype=p.dtype, device=p.device))

        checkpoint = fsdp_state_dict(fsdp_module)
        shadow_checkpoint = {}
        for n in self.shadow:
            k = n
            if k not in checkpoint and k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            if k in checkpoint:
                shadow_checkpoint[n] = checkpoint[k]
        for n, p in fsdp_module.module.named_parameters():
            if n in live_state:
                p.data.copy_(live_state[n].to(dtype=p.dtype, device=p.device))

        return shadow_checkpoint
