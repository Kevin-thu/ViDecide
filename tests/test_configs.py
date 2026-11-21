"""The released configs resolve and carry the settings of the paper (Tabs. S3 and S4 of the supplement)."""
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from videcide.config import load_config, missing_values

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs" / "sft").glob("[!_]*.yaml")) + sorted((ROOT / "configs" / "cft").glob("*.yaml"))

# config: (max_iters, total_batch_size, num_frame_per_block, lr, warmup, causal)
SFT = {
    "flowfree_k1": (23750, 2, 1, 1e-5, 500, True), "flowfree_k1_stage2": (47500, 2, 1, 3e-6, 500, True),
    "sokoban_k1": (23750, 2, 1, 1e-5, 500, True), "sokoban_k1_stage2": (47500, 2, 1, 3e-6, 500, True),
    "flowfree_k8": (23750, 2, 8, 1e-5, 500, True), "sokoban_bidir_stage2": (47100, 2, 1, 3e-6, 500, False),
    "vbvr_k1": (37500, 4, 1, 1e-5, 500, True), "vbvr_k8": (37500, 4, 8, 1e-5, 500, True),
    "vbvr_bidir": (37283, 4, 1, 1e-5, 500, False),
    "robotwin_k8": (16700, 6, 8, 1e-5, 500, True), "robotwin_k1": (16700, 6, 1, 1e-5, 500, True),
    "robotwin_bidir": (25000, 4, 1, 1e-5, 500, False),
    "maze_k1": (23750, 2, 1, 1e-5, 500, True), "maze_k8": (23750, 2, 8, 1e-5, 500, True),
    "maze_bidir": (23750, 2, 1, 1e-5, 500, False),
}
# benchmark: (updates, warmup, start step, block)
CFT = {"flowfree": (7600, 1000, 47500, 1), "sokoban": (6200, 1000, 47500, 1), "vbvr": (9800, 1000, 37500, 1),
       "robotwin": (3200, 800, 16700, 8)}


def resolved(path, overrides=()):
    config = load_config(path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    return config


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_config_resolves(path):
    config = resolved(path)
    missing = set(missing_values(config))
    assert missing <= {"resume_from", "generator_ckpt", "metadata_path", "sft_checkpoint"}, missing
    assert config.model_kwargs.timestep_shift == 10.0          # decision-focused schedule, s = 10
    assert config.timestep_shift == 5.0                         # sampling shift
    assert config.guidance_scale == 1.0                         # no classifier-free guidance at inference
    assert config.lr_schedule == "cosine" and config.max_grad_norm == 1.0
    assert (config.beta1, config.beta2, config.weight_decay) == (0.9, 0.999, 0.01)
    assert config.total_batch_size % config.batch_size == 0
    if not config.causal:
        assert list(config.frame_buckets)[-1] == config.image_or_video_shape[1] and not config.prebuild_buckets


@pytest.mark.parametrize("name", sorted(SFT))
def test_sft_schedule(name):
    config = resolved(ROOT / "configs" / "sft" / f"{name}.yaml")
    iters, batch, block, lr, warmup, causal = SFT[name]
    assert (config.max_iters, config.total_batch_size, config.num_frame_per_block) == (iters, batch, block)
    assert (config.lr, config.warmup_steps, config.causal) == (lr, warmup, causal)
    if name.endswith("_stage2"):
        assert (config.resume_lr, config.resume_lr_warmup_steps) == (3e-6, 1500)
        assert config.resume_reset_lr_scheduler and config.require_optimizer_resume


@pytest.mark.parametrize("name", sorted(CFT))
def test_cft_recipe(name):
    config = resolved(ROOT / "configs" / "cft" / f"{name}.yaml", ["sft_checkpoint=sft.pt", "metadata_path=m.json"])
    updates, warmup, start, block = CFT[name]
    assert config.rft_mode == "cft" and config.fork_sigma1_prob == 0.5
    assert (config.lr, config.resume_lr, config.lr_schedule) == (1e-6, 1e-6, "cosine")
    assert (config.resume_lr_schedule_steps, config.resume_lr_warmup_steps, config.warmup_steps) == (updates, warmup, warmup)
    assert config.max_iters == start + updates and config.data_step_origin == start
    assert (config.total_batch_size, config.batch_size, config.num_frame_per_block) == (3, 1, block)
    assert config.resume_from == config.rft_phase_entry_checkpoint == "sft.pt"


def test_dit_configs():
    load = lambda name: yaml.safe_load((ROOT / "configs" / "dit" / name).read_text())
    for name in ("flowfree_k1.yaml", "sokoban_k1.yaml"):
        c = load(name)
        assert (c["batch"], c["lr"], c["warmup"], c["ema"], c["steps"], c["anneal_steps"]) == (32, 1e-4, 500, 0.999, 10000, 1000)
        assert abs(c["mu"] - 2.3026) < 1e-3 and c["sample_steps"] == 5
    for name in ("cft_flowfree.yaml", "cft_sokoban.yaml"):
        c = load(name)
        assert (c["mode"], c["lr"], c["steps"], c["batch"], c["warmup"], c["decay"]) == ("cft", 5e-6, 1000, 32, 500, 0)
        assert (c["sft_frac"], c["p_sigma1"], c["ema"], c["fork_weight"]) == (0.25, 0.5, 0.99, 0.5)


def test_method_names():
    from videcide.training.modes import canonical_rft_mode

    assert [canonical_rft_mode(m) for m in ("cft", "decision_rft", "sample_rft", "sft")] == \
        ["cft", "decision_rft", "sample_rft", "sft"]
    with pytest.raises(ValueError):
        canonical_rft_mode("unknown")
