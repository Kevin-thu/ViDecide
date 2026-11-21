# Modified by the ViDecide authors from Causal-Forcing (https://github.com/thu-ml/Causal-Forcing,
# commit 3e7cc99, Apache-2.0): variable-length latent LMDB datasets with metadata and length-bucketed samplers. See NOTICE.
from utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset, Sampler
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
from torch.utils.data._utils.collate import default_collate


def collate_with_none(batch):
    if not batch:
        return {}
    elem = batch[0]
    if isinstance(elem, dict):
        result = {}
        for key in elem:
            vals = [d[key] for d in batch]
            if any(v is None for v in vals):
                result[key] = None
            else:
                result[key] = default_collate(vals)
        return result
    return default_collate(batch)


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class LatentLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "clean_latent": torch.tensor(latents, dtype=torch.float32)[-1]
        }


FRAME_BUCKETS = [9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]


def _row_matches(row: dict, flt: dict) -> bool:
    for k, spec in flt.items():
        val = row.get(k)
        if isinstance(spec, dict):
            if "in" in spec and val not in spec["in"]:
                return False
            if "nin" in spec and val in spec["nin"]:
                return False
            if "ne" in spec and val == spec["ne"]:
                return False
            if "le" in spec and not (val is not None and val <= spec["le"]):
                return False
            if "ge" in spec and not (val is not None and val >= spec["ge"]):
                return False
        elif isinstance(spec, (list, tuple, set)):
            if val not in spec:
                return False
        else:
            if val != spec:
                return False
    return True


def _bucket_for(nframes: int, buckets=None) -> int:
    if buckets is None:
        buckets = FRAME_BUCKETS
    for b in buckets:
        if nframes <= b:
            return b
    return buckets[-1]


def _cost_bucket_for(nframes: int) -> int:
    cost_buckets = FRAME_BUCKETS + [129, 145, 161, 181, 213, 241]
    for b in cost_buckets:
        if nframes <= b:
            return b
    return cost_buckets[-1]


class VariableLengthLatentLMDBDataset(Dataset):

    def __init__(self, data_path: str, max_pair: int = int(1e8),
                 metadata_path: str | None = None,
                 metadata_filter: dict | None = None,
                 category_repeats: dict | None = None,
                 frame_buckets: list[int] | None = None):
        import json as _json
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)
        self.frame_buckets = (
            [int(bucket) for bucket in frame_buckets]
            if frame_buckets is not None
            else list(FRAME_BUCKETS)
        )
        if not self.frame_buckets:
            raise ValueError("frame_buckets must not be empty")

        with self.env.begin() as txn:
            shape_str = txn.get(b"latents_shape").decode()
            shape = tuple(int(x) for x in shape_str.split())
            self.n_total = shape[0]
            self.spatial_shape = shape[2:]

        self.metadata_rows = None
        if metadata_path is not None:
            with open(metadata_path) as f:
                rows = _json.load(f)
            rows_by_idx = {int(r["row_idx"]): r for r in rows if "row_idx" in r}
        else:
            rows = None
            rows_by_idx = {}

        if metadata_path is not None and (metadata_filter or category_repeats):
            try:
                from omegaconf import OmegaConf
                flt = OmegaConf.to_container(metadata_filter, resolve=True) \
                    if metadata_filter is not None and not isinstance(metadata_filter, dict) \
                    else (dict(metadata_filter) if metadata_filter is not None else {})
                repeat_cfg = OmegaConf.to_container(category_repeats, resolve=True) \
                    if category_repeats is not None and not isinstance(category_repeats, dict) \
                    else category_repeats
            except ImportError:
                flt = dict(metadata_filter) if metadata_filter is not None else {}
                repeat_cfg = category_repeats
            matched_rows = [row for row in rows if _row_matches(row, flt)]
            self.index = self._build_repeated_index(matched_rows, repeat_cfg)
            print(f"[VariableLengthLatentLMDBDataset] metadata filter "
                  f"selected {len(self.index)}/{self.n_total} rows")
        else:
            self.index = list(range(self.n_total))
        if rows_by_idx:
            self.metadata_rows = [rows_by_idx.get(int(i), {"row_idx": int(i)})
                                  for i in self.index]

        self.max_pair = max_pair

    def _build_repeated_index(self, rows: list[dict], cfg: dict | None) -> list[int]:
        if not cfg:
            return [int(row["row_idx"]) for row in rows]
        field = cfg.get("field", "agent_type")
        repeats = cfg.get("repeats")
        if not repeats:
            raise ValueError("category_repeats requires repeats")
        index = []
        for label, repeat in repeats.items():
            repeat = int(repeat)
            if repeat <= 0:
                continue
            pool = [int(row["row_idx"]) for row in rows if row.get(field) == label]
            if not pool:
                raise ValueError(
                    f"category_repeats has empty pool for {field}={label}"
                )
            index.extend(pool * repeat)
        if not index:
            raise ValueError("category_repeats produced an empty index")
        return index

    def __len__(self):
        return min(len(self.index), self.max_pair)

    def nframes_for_index(self, idx: int) -> int:
        real_idx = self.index[idx]
        with self.env.begin() as txn:
            return int(txn.get(f"latents_{real_idx}_nframes".encode()).decode())

    def metadata_for_index(self, idx: int) -> dict | None:
        if self.metadata_rows is None:
            return None
        return self.metadata_rows[idx]

    def __getitem__(self, idx):
        real_idx = self.index[idx]
        metadata = self.metadata_for_index(idx)
        with self.env.begin() as txn:
            nframes = int(txn.get(f"latents_{real_idx}_nframes".encode()).decode())
            shape = (nframes,) + self.spatial_shape
            raw = txn.get(f"latents_{real_idx}_data".encode())
            latent = np.frombuffer(raw, dtype=np.float16).reshape(shape).copy()
            prompt = txn.get(f"prompts_{real_idx}_data".encode()).decode()

            clean_x_raw = txn.get(f"clean_x_latents_{real_idx}_data".encode())
            validity_raw = txn.get(f"validity_mask_{real_idx}_data".encode())

        metadata_validity = None
        if metadata is not None and "prompt_override" in metadata:
            prompt_override = metadata["prompt_override"]
            if not isinstance(prompt_override, str) or not prompt_override.strip():
                raise ValueError(f"row {real_idx} metadata prompt_override must be non-empty text")
            prompt = prompt_override
        if validity_raw is None and metadata is not None:
            has_start = "validity_mask_start" in metadata
            has_end = "validity_mask_end" in metadata
            if has_start != has_end:
                raise ValueError(
                    f"row {real_idx} metadata must define both validity_mask_start "
                    "and validity_mask_end"
                )
            if has_start:
                start = int(metadata["validity_mask_start"])
                end = int(metadata["validity_mask_end"])
                if not 0 <= start < end <= nframes:
                    raise ValueError(
                        f"row {real_idx} invalid metadata validity range "
                        f"[{start}, {end}) for {nframes} frames"
                    )
                metadata_validity = np.zeros(nframes, dtype=np.float32)
                metadata_validity[start:end] = 1.0

        latent_t = torch.from_numpy(latent).float()
        bucket = _bucket_for(nframes, self.frame_buckets)
        if nframes < bucket:
            pad = torch.zeros(bucket - nframes, *self.spatial_shape)
            latent_t = torch.cat([latent_t, pad], dim=0)

        batch = {
            "prompts": prompt,
            "clean_latent": latent_t,
            "valid_frames": nframes,
            "row_idx": real_idx,
            "clean_x_latents": None,
            "validity_mask": None,
        }
        if metadata is not None and metadata.get("decision_latent_index") is not None:
            decision_latent_index = int(metadata["decision_latent_index"])
            if not 1 <= decision_latent_index < nframes:
                raise ValueError(
                    f"row {real_idx} decision_latent_index={decision_latent_index} "
                    f"must be in generated suffix [1, {nframes})"
                )
            batch["decision_latent_index"] = decision_latent_index
        if metadata is not None and metadata.get("training_noise_group_id") is not None:
            batch["training_noise_group_id"] = str(metadata["training_noise_group_id"])

        if clean_x_raw is not None and validity_raw is None and metadata_validity is None:
            raise ValueError(
                f"row {real_idx} has clean_x_latents but no validity_mask"
            )

        if validity_raw is not None or metadata_validity is not None:
            vm = (
                np.frombuffer(validity_raw, dtype=np.float32).reshape((nframes,)).copy()
                if validity_raw is not None
                else metadata_validity
            )
            vm_t = torch.from_numpy(vm)
            if nframes < bucket:
                vm_t = torch.cat([vm_t, torch.zeros(bucket - nframes)], dim=0)
            batch["validity_mask"]   = vm_t

        if clean_x_raw is not None:
            cx = np.frombuffer(clean_x_raw, dtype=np.float16).reshape(shape).copy()
            cx_t = torch.from_numpy(cx).float()
            if nframes < bucket:
                cx_t = torch.cat([cx_t, pad], dim=0)
            batch["clean_x_latents"] = cx_t

        return batch


class PairedPlanDistributedSampler(Sampler[int]):

    def __init__(
        self,
        dataset: Dataset,
        *,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ) -> None:
        if num_replicas != 2:
            raise ValueError(
                "PairedPlanDistributedSampler requires exactly DP=2; "
                f"got {num_replicas}"
            )
        if rank not in (0, 1):
            raise ValueError(f"PairedPlanDistributedSampler invalid rank {rank}")
        if not hasattr(dataset, "metadata_for_index"):
            raise TypeError("paired Plan sampler requires metadata-aware dataset")
        groups: dict[str, list[tuple[str, int]]] = {}
        for index in range(len(dataset)):
            metadata = dataset.metadata_for_index(index)
            if metadata is None:
                raise ValueError("paired Plan sampler requires metadata for every row")
            pair_id = metadata.get("pair_id")
            plan_id = metadata.get("plan_id")
            noise_group = metadata.get("training_noise_group_id")
            if not all(isinstance(value, str) and value for value in (pair_id, plan_id, noise_group)):
                raise ValueError(
                    f"dataset index {index} needs non-empty pair_id, plan_id, and "
                    "training_noise_group_id"
                )
            if noise_group != pair_id:
                raise ValueError(
                    f"dataset index {index} noise group must equal pair_id for paired Plan training"
                )
            groups.setdefault(pair_id, []).append((plan_id, index))
        self.pairs: list[tuple[int, int]] = []
        for pair_id, siblings in sorted(groups.items()):
            if len(siblings) != 2:
                raise ValueError(f"{pair_id}: expected exactly two siblings, got {len(siblings)}")
            siblings.sort()
            if siblings[0][0] == siblings[1][0]:
                raise ValueError(f"{pair_id}: siblings must have distinct plan_id values")
            self.pairs.append((siblings[0][1], siblings[1][1]))
        if not self.pairs:
            raise ValueError("paired Plan sampler found no pairs")
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.pairs)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        order = np.random.default_rng(self.seed + self.epoch).permutation(len(self.pairs))
        return iter([self.pairs[int(pair_index)][self.rank] for pair_index in order])


class MixedLatentLMDBDataset(Dataset):

    def __init__(
        self,
        base_dataset: Dataset,
        onpolicy_dataset: Dataset,
        *,
        mix_start: float = 0.2,
        mix_end: float = 0.5,
        anneal_steps: int = 8000,
        seed: int = 0,
    ):
        self.base      = base_dataset
        self.onpolicy  = onpolicy_dataset
        self.mix_start = mix_start
        self.mix_end   = mix_end
        self.anneal_steps = anneal_steps
        self._step     = 0
        self._rng      = np.random.default_rng(seed)

    def set_step(self, step: int):
        self._step = step

    @property
    def onpolicy_prob(self) -> float:
        frac = min(1.0, self._step / max(1, self.anneal_steps))
        return self.mix_start + frac * (self.mix_end - self.mix_start)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if self._rng.random() < self.onpolicy_prob:
            op_idx = int(self._rng.integers(len(self.onpolicy)))
            item = self.onpolicy[op_idx]
            item["_sample_source"] = "onpolicy"
            item["_sample_local_idx"] = op_idx
            return item
        item = self.base[idx]
        item["_sample_source"] = "base"
        item["_sample_local_idx"] = idx
        return item


class StratifiedMixedLatentDataset(Dataset):

    def __init__(self, base_dataset: Dataset, onpolicy_dataset: Dataset):
        self.base = base_dataset
        self.onpolicy = onpolicy_dataset
        self.sample_infos: list[dict] = []
        self._build_infos()

    def _dataset_nframes(self, dataset: Dataset, idx: int) -> int:
        if hasattr(dataset, "nframes_for_index"):
            return int(dataset.nframes_for_index(idx))
        item = dataset[idx]
        vf = item.get("valid_frames") if isinstance(item, dict) else None
        if vf is not None:
            return int(vf)
        return int(item["clean_latent"].shape[0])

    def _dataset_meta(self, dataset: Dataset, idx: int) -> dict:
        if hasattr(dataset, "metadata_for_index"):
            meta = dataset.metadata_for_index(idx)
            return dict(meta) if meta is not None else {}
        return {}

    def _add_source(self, source: str, dataset: Dataset) -> None:
        for local_idx in range(len(dataset)):
            meta = self._dataset_meta(dataset, local_idx)
            nframes = int(meta.get("actual_latent_frames") or self._dataset_nframes(dataset, local_idx))
            bucket = _cost_bucket_for(nframes)
            label = (
                meta.get("controller_row_type")
                or meta.get("repair_repair_reason")
                or meta.get("agent_type")
                or source
            )
            self.sample_infos.append({
                "source": source,
                "local_idx": local_idx,
                "nframes": nframes,
                "bucket": bucket,
                "label": str(label),
            })

    def _build_infos(self) -> None:
        self._add_source("base", self.base)
        self._add_source("onpolicy", self.onpolicy)

    def __len__(self):
        return len(self.sample_infos)

    def set_step(self, step: int):
        return None

    def __getitem__(self, idx):
        info = self.sample_infos[int(idx)]
        if info["source"] == "base":
            item = self.base[info["local_idx"]]
        else:
            item = self.onpolicy[info["local_idx"]]
        item["_sample_source"] = info["source"]
        item["_sample_local_idx"] = int(info["local_idx"])
        return item


class LengthBucketDistributedSampler(Sampler):

    def __init__(
        self,
        dataset: StratifiedMixedLatentDataset,
        *,
        num_replicas: int,
        rank: int,
        onpolicy_prob: float = 0.5,
        mixed_group_prob: float = 0.5,
        seed: int = 0,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.onpolicy_prob = float(onpolicy_prob)
        self.mixed_group_prob = float(mixed_group_prob)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.by_bucket_source: dict[tuple[int, str], list[int]] = {}
        self.by_bucket: dict[int, list[int]] = {}
        self.by_source: dict[str, list[int]] = {}
        self.source_buckets: dict[str, list[int]] = {}
        self.source_bucket_prob: dict[str, np.ndarray] = {}
        for idx, info in enumerate(dataset.sample_infos):
            bucket = int(info["bucket"])
            source = str(info["source"])
            self.by_bucket_source.setdefault((bucket, source), []).append(idx)
            self.by_bucket.setdefault(bucket, []).append(idx)
            self.by_source.setdefault(source, []).append(idx)
        self.buckets = sorted(self.by_bucket)
        if not self.buckets:
            raise ValueError("LengthBucketDistributedSampler got an empty dataset")
        counts = np.array([len(self.by_bucket[b]) for b in self.buckets], dtype=np.float64)
        self.bucket_prob = counts / counts.sum()
        for source in self.by_source:
            sb = sorted(b for (b, s) in self.by_bucket_source if s == source)
            self.source_buckets[source] = sb
            sc = np.array([len(self.by_bucket_source[(b, source)]) for b in sb], dtype=np.float64)
            self.source_bucket_prob[source] = sc / sc.sum()
        self.common_buckets = sorted(
            set(self.source_buckets.get("base", [])) & set(self.source_buckets.get("onpolicy", []))
        )
        if self.common_buckets:
            cc = np.array(
                [
                    min(
                        len(self.by_bucket_source.get((b, "base"), [])),
                        len(self.by_bucket_source.get((b, "onpolicy"), [])),
                    )
                    for b in self.common_buckets
                ],
                dtype=np.float64,
            )
            self.common_bucket_prob = cc / cc.sum()
        else:
            self.common_bucket_prob = np.array([], dtype=np.float64)
        self.mixed_buckets = list(self.source_buckets.get("onpolicy", []))
        self.mixed_bucket_prob = self.source_bucket_prob.get(
            "onpolicy", np.array([], dtype=np.float64)
        )

    def __len__(self):
        n = len(self.dataset) // self.num_replicas if self.drop_last else int(np.ceil(len(self.dataset) / self.num_replicas))
        return max(1, n)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _shuffled_pools(self, rng: np.random.Generator) -> dict[tuple[int, str], list[int]]:
        pools: dict[tuple[int, str], list[int]] = {}
        for key, vals in self.by_bucket_source.items():
            arr = list(vals)
            rng.shuffle(arr)
            pools[key] = arr
        for b, vals in self.by_bucket.items():
            key = (b, "*")
            arr = list(vals)
            rng.shuffle(arr)
            pools[key] = arr
        return pools

    def _pop(self, pools: dict[tuple[int, str], list[int]], bucket: int, source: str, rng: np.random.Generator) -> int:
        key = (bucket, source)
        source_rows = self.by_bucket_source.get(key)
        if source_rows:
            if not pools.get(key):
                pools[key] = list(source_rows)
                rng.shuffle(pools[key])
            return pools[key].pop()

        nearest = sorted(self.buckets, key=lambda b: abs(b - bucket))
        for b in nearest:
            key = (b, source)
            source_rows = self.by_bucket_source.get(key)
            if source_rows:
                if not pools.get(key):
                    pools[key] = list(source_rows)
                    rng.shuffle(pools[key])
                return pools[key].pop()
        source_pool = self.by_source.get(source)
        if not source_pool:
            raise RuntimeError(f"no rows available for sampler source {source!r}")
        return int(rng.choice(source_pool))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        pools = self._shuffled_pools(rng)
        out: list[int] = []
        for _ in range(len(self)):
            use_mixed_group = (
                self.mixed_buckets
                and self.by_source.get("base")
                and self.by_source.get("onpolicy")
                and self.num_replicas > 1
                and rng.random() < self.mixed_group_prob
            )
            group = []
            if use_mixed_group:
                bucket = int(rng.choice(
                    self.mixed_buckets,
                    p=self.mixed_bucket_prob,
                ))
                n_onpolicy = int(round(self.num_replicas * self.onpolicy_prob))
                n_onpolicy = min(max(1, n_onpolicy), self.num_replicas - 1)
                sources = ["onpolicy"] * n_onpolicy + ["base"] * (self.num_replicas - n_onpolicy)
                rng.shuffle(sources)
                for source in sources:
                    group.append(self._pop(pools, bucket, source, rng))
            else:
                group_source = "onpolicy" if rng.random() < self.onpolicy_prob else "base"
                if group_source in self.source_buckets and self.source_buckets[group_source]:
                    source_buckets = self.source_buckets[group_source]
                    bucket = int(rng.choice(source_buckets, p=self.source_bucket_prob[group_source]))
                else:
                    bucket = int(rng.choice(self.buckets, p=self.bucket_prob))
                for _rank in range(self.num_replicas):
                    group.append(self._pop(pools, bucket, group_source, rng))
            out.append(group[self.rank])
        return iter(out)


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))


        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        self.transform = transform
        data_dir = Path(data_dir)

        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        aspect_ratio = metadata_path.stem.split('_')[-1]

        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


def cycle(dl):
    while True:
        for data in dl:
            yield data
