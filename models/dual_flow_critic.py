from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


CHANNELS = ("task", "amp")


def _activation(name: str) -> nn.Module:
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


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        hidden_dim = int(hidden_dim)
        if hidden_dim < 1:
            raise ValueError(f"hidden dimensions must be positive, got {hidden_dim}")
        layers.extend((nn.Linear(last_dim, hidden_dim), _activation(activation)))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


class DualFlowCritic(nn.Module):
    """Two scalar flow-value heads conditioned by one shared prefix encoder.

    ``context`` is deliberately opaque to this module. The algorithm must build
    a causal context before calling the critic (for example, privileged state,
    chunk-start state and only the *past* latent prefix). Keeping that assembly
    outside the network makes future context ablations possible without
    coupling them to the value architecture.

    The task and style heads share only ``encoder``. Their velocity networks,
    flow-matching targets and generated value samples are independent.
    Channel order in tensor APIs is always ``[task, amp]``.
    """

    def __init__(
        self,
        context_dim: int,
        encoder_hidden_dims: Sequence[int] = (512, 256),
        head_hidden_dims: Sequence[int] = (128,),
        activation: str = "elu",
        *,
        embedding_dim: int | None = None,
        flow_steps: int = 4,
        noise_std: float | Mapping[str, float] = 1.0,
        eval_samples: int = 8,
    ) -> None:
        super().__init__()
        if context_dim < 1:
            raise ValueError(f"context_dim must be positive, got {context_dim}")
        if flow_steps < 1:
            raise ValueError(f"flow_steps must be >= 1, got {flow_steps}")
        if eval_samples < 1:
            raise ValueError(f"eval_samples must be >= 1, got {eval_samples}")

        encoder_dims = tuple(int(dim) for dim in encoder_hidden_dims)
        if embedding_dim is None:
            embedding_dim = encoder_dims[-1] if encoder_dims else int(context_dim)
            encoder_body_dims = encoder_dims[:-1]
        else:
            encoder_body_dims = encoder_dims
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

        if encoder_body_dims or int(context_dim) != int(embedding_dim):
            self.encoder = _build_mlp(
                context_dim, encoder_body_dims, embedding_dim, activation
            )
        else:
            self.encoder = nn.Identity()
        head_input_dim = int(embedding_dim) + 2  # encoded context, y_t, t
        self.task_head = _build_mlp(head_input_dim, head_hidden_dims, 1, activation)
        self.amp_head = _build_mlp(head_input_dim, head_hidden_dims, 1, activation)

        if isinstance(noise_std, Mapping):
            noise_by_channel = {name: float(noise_std[name]) for name in CHANNELS}
        else:
            noise_by_channel = {name: float(noise_std) for name in CHANNELS}
        if any(std <= 0.0 for std in noise_by_channel.values()):
            raise ValueError(f"noise_std must be positive, got {noise_by_channel}")

        self.context_dim = int(context_dim)
        self.embedding_dim = int(embedding_dim)
        self.flow_steps = int(flow_steps)
        self.eval_samples = int(eval_samples)
        self.noise_std = noise_by_channel

    @staticmethod
    def _check_channel(channel: str) -> str:
        if channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}, got {channel!r}")
        return channel

    def _head(self, channel: str) -> nn.Module:
        return self.task_head if self._check_channel(channel) == "task" else self.amp_head

    def encode(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context must have shape [batch, {self.context_dim}], got {tuple(context.shape)}"
            )
        return self.encoder(context)

    @staticmethod
    def _expand_embedding(embedding: torch.Tensor, num_samples: int) -> torch.Tensor:
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
        value = value.reshape(-1, 1)
        time = time.reshape(-1, 1)
        if embedding.shape[0] != value.shape[0] or value.shape[0] != time.shape[0]:
            raise ValueError("embedding, value and time batch sizes must match")
        velocity_input = torch.cat((embedding, value, time), dim=-1)
        return self._head(channel)(velocity_input).squeeze(-1)

    def _base_points(
        self,
        channel: str,
        num_samples: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if num_samples == 1:
            return torch.zeros(1, device=device, dtype=dtype)
        probabilities = (
            torch.arange(num_samples, device=device, dtype=dtype) + 0.5
        ) / float(num_samples)
        return (
            torch.special.ndtri(probabilities).clamp(-3.0, 3.0)
            * self.noise_std[channel]
        )

    def _sample_encoded(
        self,
        embedding: torch.Tensor,
        channel: str,
        num_samples: int,
        deterministic: bool,
    ) -> torch.Tensor:
        channel = self._check_channel(channel)
        batch_size = embedding.shape[0]
        if deterministic:
            value = self._base_points(
                channel,
                num_samples,
                device=embedding.device,
                dtype=embedding.dtype,
            ).repeat(batch_size)
        else:
            value = (
                torch.randn(
                    batch_size * num_samples,
                    device=embedding.device,
                    dtype=embedding.dtype,
                )
                * self.noise_std[channel]
            )
        condition = self._expand_embedding(embedding, num_samples)
        dt = 1.0 / float(self.flow_steps)
        for step in range(self.flow_steps):
            time = torch.full_like(value, float(step) * dt)
            value = value + dt * self._velocity(condition, value, time, channel)
        return value.view(batch_size, num_samples)

    def sample(
        self,
        context: torch.Tensor,
        num_samples: int | None = None,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Return value samples with shape ``[batch, samples, 2]``."""
        num_samples = int(num_samples or self.eval_samples)
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        embedding = self.encode(context)
        samples = [
            self._sample_encoded(embedding, channel, num_samples, deterministic)
            for channel in CHANNELS
        ]
        return torch.stack(samples, dim=-1)

    def evaluate(self, context: torch.Tensor) -> torch.Tensor:
        """Return deterministic means with shape ``[batch, 2]``."""
        return self.sample(
            context, self.eval_samples, deterministic=True
        ).mean(dim=1)

    def _flow_matching_loss_encoded(
        self,
        embedding: torch.Tensor,
        target_return: torch.Tensor,
        channel: str,
        fm_samples: int,
    ) -> torch.Tensor:
        channel = self._check_channel(channel)
        target_return = target_return.reshape(-1)
        if embedding.shape[0] != target_return.shape[0]:
            raise ValueError("context and target_return batch sizes must match")
        if fm_samples < 1:
            raise ValueError(f"fm_samples must be >= 1, got {fm_samples}")

        batch_size = embedding.shape[0]
        condition = self._expand_embedding(embedding, fm_samples)
        target = target_return.unsqueeze(1).expand(-1, fm_samples).reshape(-1)
        epsilon = torch.randn_like(target) * self.noise_std[channel]
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
        """Return per-sample task/style losses with shape ``[batch, 2]``."""
        if target_returns.ndim != 2 or target_returns.shape[-1] != len(CHANNELS):
            raise ValueError(
                "target_returns must have shape [batch, 2] in [task, amp] order"
            )
        embedding = self.encode(context)
        losses = [
            self._flow_matching_loss_encoded(
                embedding, target_returns[:, index], channel, fm_samples
            )
            for index, channel in enumerate(CHANNELS)
        ]
        return torch.stack(losses, dim=-1)
