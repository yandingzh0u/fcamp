from __future__ import annotations

from .fpo import FPO
from .mixgrpo import MixGRPO
from .ppo import PPO

_REGISTRY = {
    "mixgrpo": MixGRPO,
    "ppo": PPO,
    "fpo": FPO,
}


def make_algorithm(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown algorithm '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
