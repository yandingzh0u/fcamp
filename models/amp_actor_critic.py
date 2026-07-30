"""Standard AMP actor, value function, and diagonal Gaussian distribution.

The actor emits the mean of a normalized *absolute* action chunk directly.
There is intentionally no recurrent state, temporal integration, action
anchoring, squashing transform, or auxiliary latent transform in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import torch
from torch import nn


_LOG_TWO_PI = math.log(2.0 * math.pi)
_LOG_TWO_PI_E = math.log(2.0 * math.pi * math.e)


def _validate_positive_dimension(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate_hidden_dims(hidden_dims: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(int(dim) for dim in hidden_dims)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(
            f"hidden_dims must contain positive dimensions, got {dims}"
        )
    return dims


def _relu_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in _validate_hidden_dims(hidden_dims):
        linear = nn.Linear(previous_dim, hidden_dim)
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.ReLU()))
        previous_dim = hidden_dim
    return nn.Sequential(*layers), previous_dim


class DiagonalGaussian:
    """A diagonal Gaussian over ``[..., horizon, action_dim]`` tensors.

    ``log_prob``, ``entropy``, and ``kl_divergence`` sum only over the action
    dimension and therefore return one value per horizon offset.  Their
    ``chunk_*`` counterparts additionally sum over the horizon dimension.
    """

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor) -> None:
        if mean.ndim < 2:
            raise ValueError(
                "mean must end in [horizon, action_dim], "
                f"got shape {tuple(mean.shape)}"
            )
        if mean.shape[-2] <= 0 or mean.shape[-1] <= 0:
            raise ValueError(f"mean has an empty event shape: {tuple(mean.shape)}")
        try:
            expanded_log_std = torch.broadcast_to(log_std, mean.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"log_std shape {tuple(log_std.shape)} cannot broadcast to "
                f"mean shape {tuple(mean.shape)}"
            ) from exc
        self.mean = mean
        self.log_std = expanded_log_std

    @property
    def stddev(self) -> torch.Tensor:
        return torch.exp(self.log_std)

    @property
    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Sample direct absolute actions, optionally with supplied N(0, 1) noise."""

        if noise is None:
            noise = torch.randn_like(self.mean)
        elif noise.shape != self.mean.shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} must equal mean shape "
                f"{tuple(self.mean.shape)}"
            )
        return self.mean + self.stddev * noise

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Return exact Gaussian log probability per horizon offset."""

        if actions.shape != self.mean.shape:
            raise ValueError(
                f"actions shape {tuple(actions.shape)} must equal mean shape "
                f"{tuple(self.mean.shape)}"
            )
        standardized = (actions - self.mean) * torch.exp(-self.log_std)
        component_log_prob = (
            -0.5 * standardized.square()
            - self.log_std
            - 0.5 * _LOG_TWO_PI
        )
        return component_log_prob.sum(dim=-1)

    def chunk_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Return the joint log probability of every action in the chunk."""

        return self.log_prob(actions).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Return exact Gaussian entropy per horizon offset."""

        return (self.log_std + 0.5 * _LOG_TWO_PI_E).sum(dim=-1)

    def chunk_entropy(self) -> torch.Tensor:
        """Return joint entropy of the complete action chunk."""

        return self.entropy().sum(dim=-1)

    def kl_divergence(self, other: "DiagonalGaussian") -> torch.Tensor:
        """Return ``KL(self || other)`` per horizon offset."""

        if self.mean.shape != other.mean.shape:
            raise ValueError(
                f"Gaussian means must have identical shapes, got "
                f"{tuple(self.mean.shape)} and {tuple(other.mean.shape)}"
            )
        old_variance = torch.exp(2.0 * self.log_std)
        new_variance = torch.exp(2.0 * other.log_std)
        component_kl = (
            other.log_std
            - self.log_std
            + (
                old_variance
                + (self.mean - other.mean).square()
            )
            / (2.0 * new_variance)
            - 0.5
        )
        return component_kl.sum(dim=-1)

    def chunk_kl_divergence(
        self,
        other: "DiagonalGaussian",
    ) -> torch.Tensor:
        """Return ``KL(self || other)`` for the complete action chunk."""

        return self.kl_divergence(other).sum(dim=-1)


class GaussianActor(nn.Module):
    """MimicKit-style MLP policy for normalized absolute action chunks."""

    DEFAULT_HIDDEN_DIMS = (1024, 512)
    FIXED_STD = 0.05
    OUTPUT_INIT_SCALE = 0.01

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        horizon: int = 1,
        *,
        hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        self.observation_dim = _validate_positive_dimension(
            "observation_dim", observation_dim
        )
        self.action_dim = _validate_positive_dimension(
            "action_dim", action_dim
        )
        self.horizon = _validate_positive_dimension("horizon", horizon)
        self.hidden_dims = _validate_hidden_dims(hidden_dims)
        self.trunk, feature_dim = _relu_mlp(
            self.observation_dim,
            self.hidden_dims,
        )
        self.mean_head = nn.Linear(
            feature_dim,
            self.horizon * self.action_dim,
        )
        nn.init.uniform_(
            self.mean_head.weight,
            -self.OUTPUT_INIT_SCALE,
            self.OUTPUT_INIT_SCALE,
        )
        nn.init.zeros_(self.mean_head.bias)
        self.register_buffer(
            "log_std",
            torch.full(
                (self.horizon, self.action_dim),
                math.log(self.FIXED_STD),
            ),
        )

    def _features(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"observations must end in dimension {self.observation_dim}, "
                f"got shape {tuple(observations.shape)}"
            )
        return self.trunk(observations)

    def _reshape_head(self, values: torch.Tensor) -> torch.Tensor:
        return values.reshape(
            *values.shape[:-1],
            self.horizon,
            self.action_dim,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return the direct normalized absolute-action mean."""

        features = self._features(observations)
        return self._reshape_head(self.mean_head(features))

    def distribution(
        self,
        observations: torch.Tensor,
    ) -> DiagonalGaussian:
        features = self._features(observations)
        mean = self._reshape_head(self.mean_head(features))
        log_std = self.log_std.to(device=mean.device, dtype=mean.dtype)
        return DiagonalGaussian(mean, log_std)


class ValueMLP(nn.Module):
    """Scalar state-value MLP with the official 1024/512 ReLU topology."""

    DEFAULT_HIDDEN_DIMS = (1024, 512)

    def __init__(
        self,
        observation_dim: int,
        *,
        hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        self.observation_dim = _validate_positive_dimension(
            "observation_dim", observation_dim
        )
        self.hidden_dims = _validate_hidden_dims(hidden_dims)
        self.trunk, feature_dim = _relu_mlp(
            self.observation_dim,
            self.hidden_dims,
        )
        self.value_head = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"observations must end in dimension {self.observation_dim}, "
                f"got shape {tuple(observations.shape)}"
            )
        return self.value_head(self.trunk(observations))


class AMPActorCritic(nn.Module):
    """Container for the standard Gaussian actor and scalar value function."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        horizon: int = 1,
        *,
        actor_hidden_dims: Sequence[int] = GaussianActor.DEFAULT_HIDDEN_DIMS,
        critic_hidden_dims: Sequence[int] = ValueMLP.DEFAULT_HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        self.actor = GaussianActor(
            observation_dim,
            action_dim,
            horizon,
            hidden_dims=actor_hidden_dims,
        )
        self.critic = ValueMLP(
            observation_dim,
            hidden_dims=critic_hidden_dims,
        )

    def distribution(
        self,
        observations: torch.Tensor,
    ) -> DiagonalGaussian:
        return self.actor.distribution(observations)

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations)


__all__ = [
    "AMPActorCritic",
    "DiagonalGaussian",
    "GaussianActor",
    "ValueMLP",
]
