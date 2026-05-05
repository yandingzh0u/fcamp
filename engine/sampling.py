from __future__ import annotations

import math

import torch


def flow_grpo_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor | None = None,
    deterministic: bool = False,
    noise_level: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One CPS denoising step with per-sample log probability."""
    sigma = sigmas[index].to(model_output.device)
    sigma_prev = sigmas[index + 1].to(model_output.device)
    dt = sigma_prev - sigma

    level = 0.8 if noise_level is None else noise_level
    std_dev_t = sigma_prev * math.sin(level * math.pi / 2)

    pred_original = latents - sigma * model_output
    noise_estimate = latents + model_output * (1 - sigma)
    prev_sample_mean = pred_original * (1 - sigma_prev) + noise_estimate * torch.sqrt(
        sigma_prev**2 - std_dev_t**2
    )

    if prev_sample is None:
        prev_sample = prev_sample_mean + std_dev_t * torch.randn_like(model_output)
    if deterministic:
        prev_sample = latents + dt * model_output

    log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
    log_prob = log_prob.sum(dim=tuple(range(1, log_prob.ndim)))
    log_prob = log_prob / model_output.shape[-1]
    return prev_sample, log_prob
