"""Standard MimicKit-style style discriminator and objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class StyleDiscriminator(nn.Module):
    """ReLU MLP producing an unbounded expert-vs-policy logit."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (1024, 512)) -> None:
        super().__init__()
        if input_dim <= 0 or not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("input_dim and hidden_dims must be positive")
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width in hidden_dims:
            linear = nn.Linear(previous, int(width))
            # MimicKit leaves Linear's default Kaiming-uniform(a=sqrt(5))
            # weight initialization intact and only zeros the bias.  Using the
            # larger ReLU-gain initialization here makes the input-gradient
            # penalty explode on the first adversarial update.
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.ReLU()))
            previous = int(width)
        self.trunk = nn.Sequential(*layers)
        self.logit = nn.Linear(previous, 1)
        # Match MimicKit's explicit discriminator output initialization.
        nn.init.uniform_(self.logit.weight, -1.0, 1.0)
        nn.init.zeros_(self.logit.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.input_dim:
            raise ValueError(
                f"discriminator expected final dimension {self.input_dim}, got {tuple(observation.shape)}"
            )
        return self.logit(self.trunk(observation)).squeeze(-1)

    @property
    def input_dim(self) -> int:
        first = self.trunk[0]
        assert isinstance(first, nn.Linear)
        return int(first.in_features)

    def logit_weights(self) -> torch.Tensor:
        return self.logit.weight.reshape(-1)


@dataclass
class StyleDiscriminatorLossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


def _gradient_norm_sq(logits: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
    gradient = torch.autograd.grad(
        logits,
        observations,
        grad_outputs=torch.ones_like(logits),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return gradient.reshape(gradient.shape[0], -1).square().sum(dim=-1)


def _distribution_metrics(prefix: str, logits: torch.Tensor) -> dict[str, torch.Tensor]:
    detached = logits.detach().float().reshape(-1)
    prob = torch.sigmoid(detached)
    q = torch.quantile(detached, torch.tensor([0.05, 0.5, 0.95], device=detached.device))
    return {
        f"disc/{prefix}_logit_mean": detached.mean(),
        f"disc/{prefix}_logit_std": detached.std(unbiased=False),
        f"disc/{prefix}_logit_p05": q[0],
        f"disc/{prefix}_logit_p50": q[1],
        f"disc/{prefix}_logit_p95": q[2],
        f"disc/{prefix}_prob_mean": prob.mean(),
    }


def compute_style_discriminator_loss(
    discriminator: StyleDiscriminator,
    *,
    expert_observations: torch.Tensor,
    policy_observations: torch.Tensor,
    replay_observations: torch.Tensor | None = None,
    gradient_penalty_weight: float = 10.0,
    logit_regularization_weight: float = 0.01,
) -> StyleDiscriminatorLossOutput:
    """Standard BCE + two-sided zero-centered GP + output-logit regularizer.

    This is deliberately not WGAN-GP: gradients are penalized at both real and
    fake samples, with no interpolated samples and no ``(||grad||-1)^2`` term.
    """

    if gradient_penalty_weight < 0 or logit_regularization_weight < 0:
        raise ValueError("discriminator regularization weights must be non-negative")
    expert = expert_observations.detach().requires_grad_(gradient_penalty_weight > 0)
    current = policy_observations.detach()
    fake_parts = [current]
    if replay_observations is not None and replay_observations.numel() > 0:
        fake_parts.append(replay_observations.detach())
    fake = torch.cat(fake_parts, dim=0).requires_grad_(gradient_penalty_weight > 0)

    expert_logits = discriminator(expert)
    fake_logits = discriminator(fake)
    current_logits = fake_logits[: current.shape[0]]
    replay_logits = fake_logits[current.shape[0] :]

    expert_bce = F.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits))
    fake_bce = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    bce = 0.5 * (expert_bce + fake_bce)

    if gradient_penalty_weight > 0:
        expert_gp = _gradient_norm_sq(expert_logits, expert).mean()
        fake_gp = _gradient_norm_sq(fake_logits, fake).mean()
        gradient_penalty = 0.5 * (expert_gp + fake_gp)
    else:
        zero = bce.new_zeros(())
        expert_gp = fake_gp = gradient_penalty = zero
    logit_regularization = discriminator.logit_weights().square().sum()
    total = (
        bce
        + float(gradient_penalty_weight) * gradient_penalty
        + float(logit_regularization_weight) * logit_regularization
    )

    current_bce = F.binary_cross_entropy_with_logits(current_logits, torch.zeros_like(current_logits))
    metrics: dict[str, torch.Tensor] = {
        "disc/loss": total.detach(),
        "disc/bce": bce.detach(),
        "disc/expert_bce": expert_bce.detach(),
        "disc/fake_bce": fake_bce.detach(),
        "disc/current_bce": current_bce.detach(),
        "disc/gradient_penalty": gradient_penalty.detach(),
        "disc/expert_gradient_penalty": expert_gp.detach(),
        "disc/fake_gradient_penalty": fake_gp.detach(),
        "disc/logit_regularization": logit_regularization.detach(),
        "disc/expert_accuracy": (expert_logits.detach() > 0).float().mean(),
        "disc/fake_accuracy": (fake_logits.detach() < 0).float().mean(),
        "disc/current_accuracy": (current_logits.detach() < 0).float().mean(),
    }
    metrics.update(_distribution_metrics("expert", expert_logits))
    metrics.update(_distribution_metrics("current", current_logits))
    if replay_logits.numel():
        replay_bce = F.binary_cross_entropy_with_logits(replay_logits, torch.zeros_like(replay_logits))
        metrics["disc/replay_bce"] = replay_bce.detach()
        metrics["disc/replay_accuracy"] = (replay_logits.detach() < 0).float().mean()
        metrics.update(_distribution_metrics("replay", replay_logits))
    return StyleDiscriminatorLossOutput(loss=total, metrics=metrics)
