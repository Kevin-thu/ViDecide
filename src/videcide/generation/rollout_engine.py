from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Iterable

import numpy as np
import torch


@dataclass
class RolloutJob:
    key: str
    history: torch.Tensor
    prompt: str
    source: dict[str, Any]
    seed: int
    max_latents: int
    history_key: str | None = None
    noise_seed: int | None = None
    noise_latents: int | None = None
    noise_offset: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutResult:
    job: RolloutJob
    latents: torch.Tensor
    frames: np.ndarray
    verdict: dict[str, Any]
    stop: str
    observed_at: int


def _is_terminal(verdict: dict[str, Any]) -> bool:
    return verdict["outcome"] in ("success", "rule_violation") or bool(verdict.get("eos_detected"))


def _take(value, index: torch.Tensor):
    return value.index_select(0, index) if torch.is_tensor(value) else value


class BatchedIncrementalDecoder:

    def __init__(self, vae, chunk: int = 4):
        if not vae._diffusers_vae or vae.model.use_tiling:
            raise ValueError("incremental decoding requires the untiled Diffusers Wan VAE")
        self.vae = vae
        self.chunk = chunk
        self.cache: list | None = None
        self.decoded = 0
        self.frames: list[list[np.ndarray]] = []

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> None:
        from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

        vae, model = self.vae, self.vae.model
        if latents.shape[1] <= self.decoded:
            return
        batch = latents.shape[0]
        if self.cache is None:
            model.clear_cache()
            self.cache = list(model._feat_map)
            self.frames = [[] for _ in range(batch)]
        model.to(device=latents.device, dtype=latents.dtype)
        mean = vae.mean.to(latents).view(1, vae.z_dim, 1, 1, 1)
        inv_std = (1.0 / vae.std.to(latents)).view(1, vae.z_dim, 1, 1, 1)
        pieces = []
        for lo in range(0, batch, self.chunk):
            rows = torch.arange(lo, min(lo + self.chunk, batch), device=latents.device)
            cache = [c.index_select(0, rows.to(c.device)) if torch.is_tensor(c) else c for c in self.cache]
            z = latents[lo:lo + len(rows), self.decoded:].permute(0, 2, 1, 3, 4)
            x = model.post_quant_conv(z / inv_std + mean)
            for i in range(x.shape[2]):
                out = model.decoder(x[:, :, i:i + 1], feat_cache=cache, feat_idx=[0],
                                    first_chunk=self.decoded + i == 0)
                if model.config.patch_size is not None:
                    out = unpatchify(out, model.config.patch_size)
                rgb = ((out.float().clamp(-1, 1) * .5 + .5).clamp(0, 1)
                       .permute(0, 2, 3, 4, 1).mul(255).round().to(torch.uint8).cpu().numpy())
                for j in range(rgb.shape[0]):
                    self.frames[lo + j].append(rgb[j])
                del out
            pieces.append((lo, cache))
            del x, z
            if self._writable(cache, len(rows)):
                for idx, c in enumerate(cache):
                    if torch.is_tensor(c):
                        self.cache[idx][lo:lo + len(rows)].copy_(c)
                pieces.pop()
        if pieces:
            if len(pieces) * self.chunk < batch and any(torch.is_tensor(c) for c in self.cache):
                raise RuntimeError("decoder cache changed shape after its first decode")
            caches = [cache for _, cache in pieces]
            self.cache = [torch.cat([cache[idx] for cache in caches]) if torch.is_tensor(caches[0][idx])
                          else caches[0][idx] for idx in range(len(caches[0]))]
        self.decoded = latents.shape[1]

    def _writable(self, cache: list, rows: int) -> bool:
        for old, new in zip(self.cache, cache):
            if torch.is_tensor(new) != torch.is_tensor(old):
                return False
            if torch.is_tensor(new) and (old.shape[1:] != new.shape[1:] or new.shape[0] != rows):
                return False
            if not torch.is_tensor(new) and old != new:
                return False
        return True

    def video(self, slot: int) -> np.ndarray:
        return np.concatenate(self.frames[slot])

    def select(self, index: torch.Tensor) -> None:
        if self.cache is not None:
            self.cache = [_take(c, index.to(c.device)) if torch.is_tensor(c) else c for c in self.cache]
        self.frames = [list(self.frames[i]) for i in index.tolist()]


class RolloutEngine:
    """Batched segment-wise generation with a KV cache of the clean history; stops a slot at its EOS frame (or at a
    rule violation when stop_on='rule'), and compacts the batch as slots finish."""
    context_sigma = 0.0  # > 0: history enters the KV cache re-noised to this level (diffusion-forcing sampling)

    def __init__(self, runner, *, interval: int = 4, success_grace: int | None = None, stop_on: str = "rule",
                 evaluate: Callable[..., dict] | None = None, eval_workers: int = 16, block: int = 1,
                 prefill_mode: str = 'frame'):
        from videcide.tasks.flow_replay import evaluate_flowfree

        self.runner = runner
        self.pipe = runner.pipeline
        self.device = runner.trainer.device
        self.dtype = runner.trainer.dtype
        self.block = int(block)
        if self.block < 1:
            raise ValueError("block must be positive")
        history_chunks(1, self.block, prefill_mode)
        self.prefill_mode = prefill_mode
        self.interval = interval if self.block == 1 else self.block * max(1, -(-interval // self.block))
        self.success_grace = success_grace
        if stop_on not in ("rule", "eos"):
            raise ValueError(f"unknown stop_on {stop_on}")
        self.stop_on = stop_on
        self.evaluate = evaluate or evaluate_flowfree
        self.pool = ThreadPoolExecutor(max_workers=eval_workers)
        self.stats = {"batches": 0, "slot_steps": 0, "generate_s": 0.0, "prefill_s": 0.0,
                      "decode_s": 0.0, "evaluate_s": 0.0, "compact_s": 0.0}

    def _kv(self, rows: int, frames: int) -> list[dict]:
        pipe = self.pipe
        shape = (rows, frames * pipe.frame_seq_length, pipe.num_attention_heads, pipe.attention_head_dim)
        return [{"k": torch.empty(shape, dtype=self.dtype, device=self.device),
                 "v": torch.empty(shape, dtype=self.dtype, device=self.device),
                 "global_end_index": torch.zeros(1, dtype=torch.long, device=self.device),
                 "local_end_index": torch.zeros(1, dtype=torch.long, device=self.device)}
                for _ in range(pipe.num_transformer_blocks)]

    def _cross(self, rows: int) -> list[dict]:
        pipe = self.pipe
        shape = (rows, 512, pipe.num_attention_heads, pipe.attention_head_dim)
        return [{"k": torch.zeros(shape, dtype=self.dtype, device=self.device),
                 "v": torch.zeros(shape, dtype=self.dtype, device=self.device), "is_init": False}
                for _ in range(pipe.num_transformer_blocks)]

    @property
    def halves(self) -> int:
        return 1 if float(self.pipe.args.guidance_scale) == 1.0 else 2

    def _cfg_index(self, index: torch.Tensor, half: int) -> torch.Tensor:
        return torch.cat([index, index + half]) if self.halves == 2 else index

    @staticmethod
    def _regather_kv(kv: list[dict], rows: torch.Tensor) -> list[dict]:
        out = []
        for entry in kv:
            end = int(entry["global_end_index"].item())
            m = len(rows)
            base_k, base_v = entry.get("base_k", entry["k"]), entry.get("base_v", entry["v"])
            if base_k.shape[0] < m:
                raise ValueError("cache buffer has fewer rows than requested")
            k = entry["k"][:, :end].index_select(0, rows)
            v = entry["v"][:, :end].index_select(0, rows)
            base_k[:m, :end].copy_(k)
            base_v[:m, :end].copy_(v)
            del k, v
            out.append({**entry, "k": base_k[:m], "v": base_v[:m], "base_k": base_k, "base_v": base_v})
        return out

    def _select_cross(self, cross: list[dict], rows: torch.Tensor) -> list[dict]:
        return [{"k": c["k"].index_select(0, rows), "v": c["v"].index_select(0, rows), "is_init": c["is_init"]}
                for c in cross]

    def denoise_latent(self, x, both, kv, cross, position, gens, alive, jobs):
        g, n, dev = self.halves, x.shape[0], x.device
        scheduler = self.pipe._initialize_sample_scheduler(x)
        for t in scheduler.timesteps:
            ts = t * torch.ones([g * n, x.shape[1]], device=dev, dtype=torch.float32)
            flow, _ = self._forward(torch.cat([x] * g), both, ts, kv, cross, position)
            if g == 2:
                fc, fu = flow.chunk(2)
                flow = fu + self.pipe.args.guidance_scale * (fc - fu)
            x = scheduler.step(flow, t, x, return_dict=False)[0]
        return x

    def _forward(self, x, both, timestep, kv, cross, position):
        fsl = self.pipe.frame_seq_length
        return self.pipe.generator(noisy_image_or_video=x, conditional_dict=both, timestep=timestep,
                                   kv_cache=kv, crossattn_cache=cross,
                                   current_start=position * fsl, cache_start=position * fsl)

    def _conditions(self, prompts: list[str]) -> dict:
        unique = list(dict.fromkeys(prompts))
        encoded = {p: self.pipe.text_encoder(text_prompts=[p]) for p in unique}
        if self.halves == 1:
            return {k: torch.cat([encoded[p][k][:1] for p in prompts]) for k in encoded[unique[0]]}
        negative = self.pipe.text_encoder(text_prompts=[self.pipe.args.negative_prompt])
        return {k: torch.cat([encoded[p][k][:1] for p in prompts] + [negative[k][:1]] * len(prompts))
                for k in negative}

    def _prefill_history(self, history, both, kv, cross):
        g = self.halves
        for lo, hi in history_chunks(history.shape[1], self.block, self.prefill_mode):
            zero = torch.zeros((g * history.shape[0], hi - lo),
                               device=self.device, dtype=torch.int64)
            self._forward(torch.cat([history[:, lo:hi]] * g), both, zero, kv, cross, lo)

    @torch.no_grad()
    def run_batch(self, jobs: list[RolloutJob]) -> list[RolloutResult]:
        H = int(jobs[0].history.shape[0])
        if any(int(job.history.shape[0]) != H for job in jobs):
            raise ValueError("a batch must share one history length")
        cap = max(int(job.max_latents) for job in jobs)
        if cap <= H:
            raise ValueError("latent cap must exceed the history")
        K = self.block
        if (cap - H) % K or any((int(job.max_latents) - H) % K for job in jobs):
            raise ValueError(f"generated length must be a multiple of the block ({K})")
        dev, dt = self.device, self.dtype
        self.stats["batches"] += 1

        keys = [job.history_key or job.key for job in jobs]
        unique = list(dict.fromkeys(keys))
        first = {key: keys.index(key) for key in unique}
        expand = torch.tensor([unique.index(key) for key in keys], device=dev)
        u, b = len(unique), len(jobs)

        started = time.perf_counter()
        history = torch.stack([jobs[first[key]].history for key in unique]).to(dev, dt)
        both = self._conditions([jobs[first[key]].prompt for key in unique])
        g = self.halves
        kv_full = self._kv(g * b, cap)
        kv = [{**e, "k": e["k"][:g * u], "v": e["v"][:g * u], "base_k": e["k"], "base_v": e["v"]} for e in kv_full]
        del kv_full
        cross = self._cross(g * u)
        self._prefill_history(history, both, kv, cross)
        decoder = BatchedIncrementalDecoder(self.pipe.vae)
        t0 = time.perf_counter()
        decoder.decode(history)
        self.stats["decode_s"] += time.perf_counter() - t0
        rows = self._cfg_index(expand, u)
        kv = self._regather_kv(kv, rows)
        cross = self._select_cross(cross, rows)
        both = {k: v.index_select(0, rows) for k, v in both.items()}
        decoder.select(expand)
        self.stats["prefill_s"] += time.perf_counter() - started

        out = torch.zeros((b, cap, *history.shape[2:]), device=dev, dtype=dt)
        out[:, :H] = history.index_select(0, expand)
        noise = torch.cat([torch.randn(
            (1, max(cap - H + job.noise_offset, job.noise_latents or 0), *history.shape[2:]), device=dev, dtype=dt,
            generator=torch.Generator(device=dev).manual_seed(
                job.seed + 100_000_000 if job.noise_seed is None else job.noise_seed))[:, job.noise_offset:job.noise_offset + cap - H]
            for job in jobs])
        gens = [torch.Generator(device=dev).manual_seed(job.seed) for job in jobs]
        alive = list(range(b))
        solved_at: dict[int, int] = {}
        results: list[RolloutResult | None] = [None] * b
        caps = torch.tensor([int(job.max_latents) for job in jobs])

        for position in range(H, cap, K):
            t0 = time.perf_counter()
            n = len(alive)
            x = noise[:, position - H:position - H + K]
            self.runner.generators = gens
            x = self.denoise_latent(x, both, kv, cross, position, gens, alive, jobs)
            out[:, position:position + K] = x
            ctx, cts = x, torch.zeros([g * n, K], device=dev, dtype=torch.int64)
            if self.context_sigma > 0:  # diffusion-forcing sampling: cache the new segment re-noised to context_sigma
                gen = torch.Generator(device=dev).manual_seed(1_000_003 * position + 7)
                ctx = (1 - self.context_sigma) * x + self.context_sigma * torch.randn(x.shape, device=dev, dtype=x.dtype, generator=gen)
                cts = torch.full([g * n, K], self.context_sigma * 1000.0, device=dev, dtype=torch.float32)
            self._forward(torch.cat([ctx] * g), both, cts, kv, cross, position)
            self.runner.generators = None
            self.stats["generate_s"] += time.perf_counter() - t0
            self.stats["slot_steps"] += n * K

            length = position + K
            check_length = length - H if K > 1 else length
            if check_length % self.interval and length != cap and not bool((caps[alive] == length).any()):
                continue
            t0 = time.perf_counter()
            decoder.decode(out[:, :length])
            self.stats["decode_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            videos = [decoder.video(row) for row in range(n)]
            verdicts = list(self.pool.map(lambda pair: self.evaluate(pair[0], jobs[pair[1]].source),
                                          zip(videos, alive)))
            self.stats["evaluate_s"] += time.perf_counter() - t0
            keep = []
            for row, (job_index, verdict) in enumerate(zip(alive, verdicts)):
                job = jobs[job_index]
                capped = length >= int(job.max_latents)
                terminal = _is_terminal(verdict) if self.stop_on == "rule" else bool(verdict.get("eos_detected"))
                if terminal and self.success_grace is not None and verdict["outcome"] == "success" \
                        and not verdict.get("eos_detected"):
                    first = solved_at.setdefault(job_index, length)
                    terminal = length >= first + self.success_grace
                if terminal or capped:
                    full = self.evaluate(videos[row], job.source, include_trace=True)
                    results[job_index] = RolloutResult(
                        job=job, latents=out[row, :length].detach().cpu().clone(), frames=videos[row],
                        verdict=full, stop="terminal" if terminal else "cap", observed_at=length)
                else:
                    keep.append(row)
            if not keep:
                break
            if len(keep) < n:
                t0 = time.perf_counter()
                index = torch.tensor(keep, device=dev)
                rows = self._cfg_index(index, n)
                kv = self._regather_kv(kv, rows)
                cross = self._select_cross(cross, rows)
                both = {k: v.index_select(0, rows) for k, v in both.items()}
                out = out.index_select(0, index)
                noise = noise.index_select(0, index)
                decoder.select(index)
                gens = [gens[i] for i in keep]
                alive = [alive[i] for i in keep]
                self.stats["compact_s"] += time.perf_counter() - t0
        del kv, cross, out, noise, decoder
        if any(result is None for result in results):
            raise RuntimeError("rollout engine lost a slot")
        return results

    def run(self, jobs: Iterable[RolloutJob], *, batch_size: Callable[[RolloutJob], int] | int,
            on_result: Callable[[RolloutResult], None] | None = None,
            order: Callable[[RolloutJob], Any] | None = None) -> list[RolloutResult]:
        groups: dict[tuple, list[RolloutJob]] = {}
        for job in jobs:
            size = batch_size(job) if callable(batch_size) else int(batch_size)
            groups.setdefault((int(job.history.shape[0]), size, int(job.max_latents)), []).append(job)
        collected = []
        for (_, size, _), members in sorted(groups.items(), key=lambda item: item[0]):
            members.sort(key=lambda job: ((order(job),) if order else ()) + (job.history_key or job.key,))
            for start in range(0, len(members), size):
                for result in self.run_batch(members[start:start + size]):
                    if on_result is not None:
                        on_result(result)
                    else:
                        collected.append(result)
                torch.cuda.empty_cache()
        return collected


def history_chunks(length, block, mode):
    if length < 1 or block < 1:
        raise ValueError('History length and block size must be positive')
    if mode == 'frame':
        return [(i, i + 1) for i in range(length)]
    if mode != 'native_block':
        raise ValueError(f'Unknown prefill mode: {mode}')
    if (length - 1) % block:
        raise ValueError('Native history must end at a complete block boundary')
    return [(0, 1)] + [(i, i + block) for i in range(1, length, block)]
