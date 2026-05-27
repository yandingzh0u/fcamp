from __future__ import annotations

import math

import torch


def flow_sde_transition(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    index: int,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official MixGRPO SDE transition mean/std for one flow-matching step."""
    device = model_output.device
    sigma = sigmas[index].to(device=device, dtype=model_output.dtype)
    sigma_prev = sigmas[index + 1].to(device=device, dtype=model_output.dtype)
    sigma_max = sigmas[1].to(device=device, dtype=model_output.dtype)
    dt = sigma_prev - sigma

    denominator = 1.0 - torch.where(sigma == 1, sigma_max, sigma)
    std_dev_t = torch.sqrt(torch.clamp(sigma / denominator, min=0.0)) * float(eta)
    std = std_dev_t * torch.sqrt(torch.clamp(-dt, min=0.0))
    prev_sample_mean = latents * (1.0 + std_dev_t.square() / (2.0 * sigma) * dt)
    prev_sample_mean = prev_sample_mean + model_output * (
        1.0 + std_dev_t.square() * (1.0 - sigma) / (2.0 * sigma)
    ) * dt
    return prev_sample_mean, std, latents - sigma * model_output


def flow_grpo_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    index: int,
    eta: float = 0.7,
    prev_sample: torch.Tensor | None = None,
    deterministic: bool = False,
    sample_noise: torch.Tensor | None = None,
    sample_noise_std: float = 1.0,
    horizon: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One MixGRPO SDE-ODE transition.

    The returned log_prob is **per-frame** with shape (B, horizon). The chunk_dim
    direction (= horizon * action_dim) is split into the per-frame action_dim and
    summed; horizons are kept separate so the downstream PPO ratio / KL stay
    horizon-invariant. For horizon=1 the result has shape (B, 1).
    """
    sigma = sigmas[index].to(model_output.device)
    sigma_prev = sigmas[index + 1].to(model_output.device)
    dt = sigma_prev - sigma
    prev_sample_mean, std, _ = flow_sde_transition(model_output, latents, sigmas, index, eta=eta)

    if prev_sample is None:
        if deterministic:
            prev_sample = latents + dt * model_output
        elif bool(torch.as_tensor(std).abs().max() > 1e-12):
            if sample_noise is None:
                sample_noise = torch.randn_like(model_output)
            prev_sample = prev_sample_mean + std * float(sample_noise_std) * sample_noise
        else:
            prev_sample = prev_sample_mean

    residual = prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)
    std = torch.as_tensor(std, device=model_output.device, dtype=torch.float32).clamp(min=1.0e-6)
    log_prob = -residual.square() / (2.0 * std.square())
    log_prob = log_prob - torch.log(std) - 0.5 * math.log(2.0 * math.pi)
    if log_prob.ndim == 2:
        # (B, chunk_dim) -> (B, horizon, action_dim) -> (B, horizon)
        batch_size, chunk_dim = log_prob.shape
        if chunk_dim % max(1, horizon) != 0:
            raise ValueError(
                f"chunk_dim {chunk_dim} not divisible by horizon {horizon}; cannot split per-frame log_prob"
            )
        action_dim = chunk_dim // max(1, horizon)
        log_prob = log_prob.view(batch_size, horizon, action_dim).sum(dim=-1)
    else:
        log_prob = log_prob.sum(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample, log_prob
