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
    sample_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    prev_sample_mean, std, _ = flow_sde_transition(model_output, latents, sigmas, index, eta=eta)

    if prev_sample is None:
        if bool(torch.as_tensor(std).abs().max() > 1e-12):
            if sample_noise is None:
                sample_noise = torch.randn_like(model_output)
            prev_sample = prev_sample_mean + std * sample_noise
        else:
            prev_sample = prev_sample_mean

    residual = prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)
    std = torch.as_tensor(std, device=model_output.device, dtype=torch.float32).clamp(min=1.0e-6)
    log_prob = -residual.square() / (2.0 * std.square())
    log_prob = log_prob - torch.log(std) - 0.5 * math.log(2.0 * math.pi)
    log_prob = log_prob.sum(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample, log_prob


def flow_grpo_step_per_frame(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    index: int,
    eta: float = 0.7,
    prev_sample: torch.Tensor | None = None,
    sample_noise: torch.Tensor | None = None,
    horizon: int = 1,
    action_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same SDE transition as :func:`flow_grpo_step`, but returns a per-frame
    log-prob of shape ``[batch, horizon]`` instead of a single chunk-level scalar.

    The latent chunk is interpreted as ``[horizon, action_dim]`` (row-major),
    matching :class:`FlowMatchingPolicy._action_transform`. The log-prob is
    reduced over ``action_dim`` only, so each frame keeps its own log-prob and
    can carry its own ratio / clip / KL in the actor update.
    """
    prev_sample_mean, std, _ = flow_sde_transition(model_output, latents, sigmas, index, eta=eta)

    if prev_sample is None:
        if bool(torch.as_tensor(std).abs().max() > 1e-12):
            if sample_noise is None:
                sample_noise = torch.randn_like(model_output)
            prev_sample = prev_sample_mean + std * sample_noise
        else:
            prev_sample = prev_sample_mean

    residual = prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)
    std = torch.as_tensor(std, device=model_output.device, dtype=torch.float32).clamp(min=1.0e-6)
    log_prob = -residual.square() / (2.0 * std.square())
    log_prob = log_prob - torch.log(std) - 0.5 * math.log(2.0 * math.pi)
    # latent layout is [batch, horizon * action_dim] row-major -> [batch, horizon, action_dim]
    batch = log_prob.shape[0]
    log_prob = log_prob.reshape(batch, horizon, action_dim).sum(dim=-1)
    return prev_sample, log_prob
