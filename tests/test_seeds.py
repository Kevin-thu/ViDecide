"""Sampling seeds of the evaluation and collection scripts: the offsets stored in the released manifests replace the
hash of the sample id and of the rollout key."""
import hashlib
import json

from videcide.generation.common import (continuation_seed, rollout_seed_fields, row_seed, stable_sample_seed,
                                        with_seed_fields)


def offset(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def test_stored_offsets_give_the_seeds_of_the_original_id(tmp_path):
    # "b" is the released id of a puzzle that was sampled as "a"
    row = {"sample_id": "b", "seed_offset": offset("a"), "continuation_seed_offsets": [offset("a_r0"), offset("a_r1")]}
    assert row_seed(7, row) == stable_sample_seed(7, "a")
    for k in range(2):
        rollout = {"key": f"b_r{k}", **rollout_seed_fields(row, k)}
        assert continuation_seed(5, rollout) == stable_sample_seed(5, f"a_r{k}")
    # without stored offsets: the hash of the id and of the rollout key
    assert row_seed(7, {"sample_id": "b"}) == stable_sample_seed(7, "b")
    assert rollout_seed_fields(row, 2) == {} and continuation_seed(5, {"key": "b_r2"}) == stable_sample_seed(5, "b_r2")
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    joined = with_seed_fields([{"sample_id": "b", "prompt": "p"}, {"sample_id": "c"}], manifest)
    assert joined[0] == {**row, "prompt": "p"} and joined[1] == {"sample_id": "c"}
