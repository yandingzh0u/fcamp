from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


CHANNELS = ("task", "amp")


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


class DualFlowCritic(nn.Module):
    """Shared context encoder with independent task/style flow-value heads."""

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
        embedding_dim = encoder_dims[-1]
        self.encoder = _mlp(
            context_dim, encoder_dims[:-1], embedding_dim, activation
        )
        head_input_dim = embedding_dim + 2
        self.task_head = _mlp(
            head_input_dim, head_hidden_dims, 1, activation
        )
        self.amp_head = _mlp(
            head_input_dim, head_hidden_dims, 1, activation
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
        channel: str,
    ) -> torch.Tensor:
        inputs = torch.cat(
            (embedding, value.reshape(-1, 1), time.reshape(-1, 1)),
            dim=-1,
        )
        head = self.task_head if channel == "task" else self.amp_head
        return head(inputs).squeeze(-1)

    def _base_points(
        self,
        num_samples: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if num_samples == 1:
            return torch.zeros(
                1, device=reference.device, dtype=reference.dtype
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

    def _sample_encoded(
        self,
        embedding: torch.Tensor,
        channel: str,
    ) -> torch.Tensor:
        batch_size = embedding.shape[0]
        value = self._base_points(
            self.eval_samples,
            embedding,
        ).repeat(batch_size)
        condition = self._expand_embedding(embedding, self.eval_samples)
        dt = 1.0 / float(self.flow_steps)
        for step in range(self.flow_steps):
            time = torch.full_like(value, float(step) * dt)
            value = value + dt * self._velocity(
                condition, value, time, channel
            )
        return value.view(batch_size, self.eval_samples)

    def evaluate(self, context: torch.Tensor) -> torch.Tensor:
        embedding = self.encode(context)
        samples = [
            self._sample_encoded(embedding, channel)
            for channel in CHANNELS
        ]
        return torch.stack(samples, dim=-1).mean(dim=1)

    def _flow_matching_loss_encoded(
        self,
        embedding: torch.Tensor,
        target_return: torch.Tensor,
        channel: str,
        fm_samples: int,
    ) -> torch.Tensor:
        target_return = target_return.reshape(-1)
        batch_size = embedding.shape[0]
        condition = self._expand_embedding(embedding, fm_samples)
        target = (
            target_return.unsqueeze(1)
            .expand(-1, fm_samples)
            .reshape(-1)
        )
        epsilon = torch.randn_like(target) * self.noise_std
        time = torch.rand(
            batch_size * fm_samples,
            device=embedding.device,
            dtype=embedding.dtype,
        )
        interpolated = (1.0 - time) * epsilon + time * target
        target_velocity = target - epsilon
        predicted_velocity = self._velocity(
            condition, interpolated, time, channel
        )
        return (
            (predicted_velocity - target_velocity)
            .square()
            .view(batch_size, fm_samples)
            .mean(dim=1)
        )

    def flow_matching_loss(
        self,
        context: torch.Tensor,
        target_returns: torch.Tensor,
        *,
        fm_samples: int = 1,
    ) -> torch.Tensor:
        embedding = self.encode(context)
        losses = [
            self._flow_matching_loss_encoded(
                embedding,
                target_returns[:, index],
                channel,
                fm_samples,
            )
            for index, channel in enumerate(CHANNELS)
        ]
        return torch.stack(losses, dim=-1)
