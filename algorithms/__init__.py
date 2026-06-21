"""Algorithm registry. Maps algo_name -> Algorithm subclass."""
from __future__ import annotations

from .mixgrpo import MixGRPO

_REGISTRY = {
    "mixgrpo": MixGRPO,
}


def make_algorithm(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown algorithm '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
