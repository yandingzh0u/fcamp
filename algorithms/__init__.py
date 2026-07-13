from __future__ import annotations

from .ppo import PPO
from .sfpo import SFPO
from .fcamp import FCAMP

_REGISTRY = {
    "ppo": PPO,
    "sfpo": SFPO,
    "fcamp": FCAMP,
}


def make_algorithm(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown algorithm '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
