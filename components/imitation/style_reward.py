"""Standard style discriminator reward and diagnostics."""

from __future__ import annotations

import torch


def discriminator_style_reward(
    logits: torch.Tensor,
    *,
    scale: float = 2.0,
    minimum_one_minus_prob: float = 1.0e-4,
) -> torch.Tensor:
    """Compute ``-scale*log(max(1-sigmoid(logit), eps))``."""

    if scale < 0:
        raise ValueError("style reward scale must be non-negative")
    if not 0.0 < minimum_one_minus_prob < 1.0:
        raise ValueError("minimum_one_minus_prob must lie in (0,1)")
    one_minus_prob = 1.0 - torch.sigmoid(logits)
    return -float(scale) * torch.log(torch.clamp(one_minus_prob, min=minimum_one_minus_prob))


@torch.no_grad()
def style_reward_statistics(
    logits: torch.Tensor,
    rewards: torch.Tensor | None = None,
    *,
    scale: float = 2.0,
    minimum_one_minus_prob: float = 1.0e-4,
    prefix: str = "amp_reward",
) -> dict[str, float]:
    flat_logits = logits.detach().float().reshape(-1)
    if flat_logits.numel() == 0:
        return {}
    if rewards is None:
        rewards = discriminator_style_reward(
            flat_logits,
            scale=scale,
            minimum_one_minus_prob=minimum_one_minus_prob,
        )
    flat_rewards = rewards.detach().float().reshape(-1)
    prob = torch.sigmoid(flat_logits)
    quantiles = torch.quantile(flat_rewards, torch.tensor([0.05, 0.5, 0.95], device=flat_rewards.device))
    return {
        f"{prefix}/mean": float(flat_rewards.mean().item()),
        f"{prefix}/std": float(flat_rewards.std(unbiased=False).item()),
        f"{prefix}/min": float(flat_rewards.min().item()),
        f"{prefix}/max": float(flat_rewards.max().item()),
        f"{prefix}/p05": float(quantiles[0].item()),
        f"{prefix}/p50": float(quantiles[1].item()),
        f"{prefix}/p95": float(quantiles[2].item()),
        f"{prefix}/prob_mean": float(prob.mean().item()),
        f"{prefix}/clamp_fraction": float(((1.0 - prob) <= minimum_one_minus_prob).float().mean().item()),
    }
