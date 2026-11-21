from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import torch
from torch.utils.data import Sampler
from torch.utils.data._utils.collate import default_collate


def bucket_for(nframes: int, buckets: list[int]) -> int:
    for b in buckets:
        if nframes <= b:
            return b
    return buckets[-1]


class BucketBatchSampler(Sampler):

    def __init__(self, dataset, num_replicas: int, rank: int, microbatch: int, seed: int = 0):
        self.num_replicas, self.rank, self.microbatch, self.seed, self.epoch = num_replicas, rank, microbatch, seed, 0
        self.global_batch = num_replicas * microbatch
        self.groups: dict[int, list[int]] = defaultdict(list)
        rows = dataset.metadata_rows
        for index in range(len(dataset)):
            n = None
            if rows is not None and rows[index].get('actual_latent_frames') is not None:
                n = int(rows[index]['actual_latent_frames'])
            if n is None:
                n = dataset.nframes_for_index(index)
            self.groups[bucket_for(n, dataset.frame_buckets)].append(index)
        self.steps_per_pass = sum(math.ceil(len(v) / self.global_batch) for v in self.groups.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps_per_pass * self.microbatch

    def _key(self, s: str) -> bytes:
        return hashlib.sha256(f'{self.seed}:{self.epoch}:{s}'.encode()).digest()

    def global_batches(self) -> list[tuple[int, list[int]]]:
        out = []
        for bucket, rows in sorted(self.groups.items()):
            rows = sorted(rows, key=lambda i: self._key(f'r{i}'))
            need = (-len(rows)) % self.global_batch
            rows = rows + (rows * math.ceil(need / len(rows)))[:need]
            for start in range(0, len(rows), self.global_batch):
                out.append((bucket, rows[start:start + self.global_batch]))
        out.sort(key=lambda b: self._key(f'b{b[0]}:{b[1][0]}:{b[1][-1]}'))
        return out

    def __iter__(self):
        start = self.rank * self.microbatch
        for _, rows in self.global_batches():
            yield from rows[start:start + self.microbatch]


def weight_rows(batch: dict) -> dict:
    mask = batch.get('validity_mask')
    if mask is None or mask.shape[0] == 1:
        return batch
    n = mask.sum(dim=1, keepdim=True)
    if bool((n <= 0).any()):
        raise ValueError('every row needs at least one supervised latent')
    batch['validity_mask'] = mask * (n.mean() / n)
    return batch


def collate_bucket_batch(items):
    from utils.dataset import collate_with_none
    shapes = {tuple(item['clean_latent'].shape) for item in items}
    if len(shapes) != 1:
        raise ValueError(f'bucket batch mixes shapes {sorted(shapes)}')
    batch = collate_with_none(items)
    if batch.get('validity_mask') is None:
        raise ValueError('bucket batching needs metadata validity masks (validity_mask_start / _end)')
    return weight_rows(batch)


class MatchedSubtaskSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, microbatch, seed=0):
        self.dataset, self.num_replicas, self.rank = dataset, num_replicas, rank
        self.microbatch, self.seed, self.epoch = microbatch, seed, 0
        self.groups = defaultdict(list)
        for index, row in enumerate(dataset.metadata_rows):
            self.groups[row['subtask']].append((row['sample_id'], index))
        self.global_batch = num_replicas * microbatch
        self.steps_per_pass = sum(math.ceil(len(v) / self.global_batch) for v in self.groups.values())

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.steps_per_pass * self.microbatch

    def __iter__(self):
        batches = []
        def order(s):
            return hashlib.sha256(f'{self.seed}:{self.epoch}:{s}'.encode()).digest()
        for task, group in sorted(self.groups.items()):
            rows = sorted(group, key=lambda r: order(r[0]))
            needed = (-len(rows)) % self.global_batch
            rows += (rows * math.ceil(needed / len(rows)))[:needed]
            for start in range(0, len(rows), self.global_batch):
                batches.append((task, start, rows[start:start + self.global_batch]))
        batches.sort(key=lambda b: order(f'{b[0]}:{b[1]}'))
        for _, _, batch in batches:
            start = self.rank * self.microbatch
            yield from (i for _, i in batch[start:start + self.microbatch])


def collate_native_or_freeze(batch):
    maximum = max(item['clean_latent'].shape[0] for item in batch)
    padded = []
    for item in batch:
        copy = dict(item)
        for key in ['clean_latent', 'clean_x_latents', 'validity_mask']:
            value = copy.get(key)
            if value is not None and len(value) < maximum:
                copy[key] = torch.cat([value, value.new_zeros((maximum-len(value), *value.shape[1:]))])
        padded.append(copy)
    return {key: None if any(item[key] is None for item in padded) else default_collate([item[key] for item in padded]) for key in padded[0]}
