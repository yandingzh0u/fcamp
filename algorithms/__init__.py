from __future__ import annotations

from .chunk_ppo import SFPO as ChunkPPO
from .fpo_plus_plus import FPOPlusPlus
from .fpo import OriginalFPO
from .flowrl import FlowRL
from .fql import FQL
from .reinflow import ReinFlow
from .ppo import PPO
from .sfpo import SFPO
from .sfpo_gaussian import SFPO as SFPOGaussian

_REGISTRY = {
    "ppo": PPO,
    "fpo": OriginalFPO,
    "fpo++": FPOPlusPlus,
    "flowrl": FlowRL,
    "fql": FQL,
    "reinflow": ReinFlow,
    "sfpo": SFPO,
    "sfpo-gaussian": SFPOGaussian,
    "chunk-ppo": ChunkPPO,
}


def make_algorithm(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown algorithm '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
