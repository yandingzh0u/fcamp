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
    per_frame: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One MixGRPO SDE-ODE transition.

    The returned log_prob is the transition score for one flow step.

    - per_frame=False (default): joint chunk score, shape (B,). This is the legacy path and
      is byte-identical to the original implementation (sum over every latent dimension).
    - per_frame=True: per-frame score, shape (B, horizon). The latent residual is reshaped
      to (B, horizon, action_dim) and summed ONLY over action_dim. Because the SDE transition
      noise is independent per latent dimension, the joint density is exactly the product of
      per-frame marginals, so `per_frame_logp.sum(dim=1)` equals the joint (B,) score to
      floating-point tolerance. This is the mathematical basis of Frame-Factorized h>1.
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
    if per_frame and int(horizon) > 1:
        # (B, chunk_dim) -> (B, horizon, action_dim) -> sum over action_dim -> (B, horizon)
        batch = log_prob.shape[0]
        log_prob = log_prob.reshape(batch, int(horizon), -1).sum(dim=-1)
    else:
        log_prob = log_prob.sum(dim=tuple(range(1, log_prob.ndim)))
        if per_frame:
            # horizon == 1: present as (B, 1) so callers always see a frame axis.
            log_prob = log_prob.unsqueeze(-1)
    return prev_sample, log_prob
