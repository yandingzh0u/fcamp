from __future__ import annotations

import torch


def flow_ode_mean(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    index: int,
) -> torch.Tensor:
    sigma = sigmas[index].to(device=model_output.device, dtype=model_output.dtype)
    sigma_next = sigmas[index + 1].to(device=model_output.device, dtype=model_output.dtype)
    dt = sigma_next - sigma
    return latents + model_output * dt
