from __future__ import annotations

from collections.abc import Sequence

from torch import nn


def activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation_name: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        hidden_dim = int(hidden_dim)
        if hidden_dim < 1:
            raise ValueError(f"hidden dimensions must be positive, got {hidden_dim}")
        layers.extend((nn.Linear(last_dim, hidden_dim), activation(activation_name)))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)
