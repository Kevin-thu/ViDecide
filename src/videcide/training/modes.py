"""Names of the post-training methods (Tab. 5 of the paper)."""
from __future__ import annotations

# cft          Corrective Fine-tuning: the divergence segment reuses the noise of the failed rollout (+ q(t))
# decision_rft decision-level RFT: the same successful continuations with fresh noise
# sample_rft   sample-level RFT: successful whole rollouts
# sft          continued SFT on expert videos
RFT_MODES = ("cft", "decision_rft", "sample_rft", "sft")


def canonical_rft_mode(mode: str) -> str:
    """Check a method name (``rft_mode`` of the post-training configs)."""
    mode = str(mode)
    if mode not in RFT_MODES:
        raise ValueError(f"unknown rft_mode {mode!r}; expected one of {RFT_MODES}")
    return mode
