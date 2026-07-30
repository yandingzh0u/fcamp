from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(input_dim, hidden_dim), _activation(activation)))
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class TaskFlowCritic(nn.Module):
    """Single-head conditional flow model for scalar task returns."""

    def __init__(
        self,
        context_dim: int,
        encoder_hidden_dims: Sequence[int],
        head_hidden_dims: Sequence[int],
        activation: str,
        *,
        flow_steps: int,
        eval_samples: int,
    ) -> None:
        super().__init__()
        encoder_dims = tuple(int(dim) for dim in encoder_hidden_dims)
        if not encoder_dims or any(dim <= 0 for dim in encoder_dims):
            raise ValueError("encoder_hidden_dims must contain positive values")
        if any(int(dim) <= 0 for dim in head_hidden_dims):
            raise ValueError("head_hidden_dims must contain positive values")
        if int(flow_steps) <= 0 or int(eval_samples) <= 0:
            raise ValueError("flow_steps and eval_samples must be positive")

        embedding_dim = encoder_dims[-1]
        self.encoder = _mlp(
            int(context_dim),
            encoder_dims[:-1],
            embedding_dim,
            activation,
        )
        self.task_head = _mlp(
            embedding_dim + 2,
            head_hidden_dims,
            1,
            activation,
        )
        self.context_dim = int(context_dim)
        self.flow_steps = int(flow_steps)
        self.eval_samples = int(eval_samples)
        self.noise_std = 1.0

    def encode(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context must have shape [batch, {self.context_dim}], "
                f"got {tuple(context.shape)}"
            )
        return self.encoder(context)

    @staticmethod
    def _expand_embedding(
        embedding: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        return (
            embedding.unsqueeze(1)
            .expand(-1, num_samples, -1)
            .reshape(-1, embedding.shape[-1])
        )

    def _velocity(
        self,
        embedding: torch.Tensor,
        value: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat(
            (embedding, value.reshape(-1, 1), time.reshape(-1, 1)),
            dim=-1,
        )
        return self.task_head(inputs).squeeze(-1)

    def _base_points(
        self,
        num_samples: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if num_samples == 1:
            return torch.zeros(
                1,
                device=reference.device,
                dtype=reference.dtype,
            )
        probabilities = (
            torch.arange(
                num_samples,
                device=reference.device,
                dtype=reference.dtype,
            )
            + 0.5
        ) / float(num_samples)
        return (
            torch.special.ndtri(probabilities).clamp(-3.0, 3.0)
            * self.noise_std
        )

    def evaluate(self, context: torch.Tensor) -> torch.Tensor:
        embedding = self.encode(context)
        batch_size = embedding.shape[0]
        value = self._base_points(
            self.eval_samples,
            embedding,
        ).repeat(batch_size)
        condition = self._expand_embedding(embedding, self.eval_samples)
        dt = 1.0 / float(self.flow_steps)
        for step in range(self.flow_steps):
            time = torch.full_like(value, float(step) * dt)
            value = value + dt * self._velocity(condition, value, time)
        return value.view(batch_size, self.eval_samples).mean(dim=1)

    def flow_matching_loss(
        self,
        context: torch.Tensor,
        target_returns: torch.Tensor,
        *,
        fm_samples: int = 1,
    ) -> torch.Tensor:
        if int(fm_samples) <= 0:
            raise ValueError("fm_samples must be positive")
        embedding = self.encode(context)
        target_returns = target_returns.reshape(-1)
        if target_returns.shape[0] != embedding.shape[0]:
            raise ValueError("target_returns batch does not match context")
        condition = self._expand_embedding(embedding, int(fm_samples))
        target = (
            target_returns.unsqueeze(1)
            .expand(-1, int(fm_samples))
            .reshape(-1)
        )
        epsilon = torch.randn_like(target) * self.noise_std
        time = torch.rand(
            target.shape[0],
            device=embedding.device,
            dtype=embedding.dtype,
        )
        interpolated = (1.0 - time) * epsilon + time * target
        target_velocity = target - epsilon
        predicted_velocity = self._velocity(
            condition,
            interpolated,
            time,
        )
        return (
            (predicted_velocity - target_velocity)
            .square()
            .view(embedding.shape[0], int(fm_samples))
            .mean(dim=1)
        )
