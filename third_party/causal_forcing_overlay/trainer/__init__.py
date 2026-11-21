# Modified by the ViDecide authors from Causal-Forcing (https://github.com/thu-ml/Causal-Forcing,
# commit 3e7cc99, Apache-2.0): export the diffusion trainer only. See NOTICE.
from .diffusion import Trainer as DiffusionTrainer

__all__ = ["DiffusionTrainer"]
