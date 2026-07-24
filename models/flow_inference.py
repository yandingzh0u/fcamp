from __future__ import annotations

import torch

from .flow_sampling import flow_ode_mean


@torch.no_grad()
def deterministic_flow_raw_targets(
    policy,
    observation: torch.Tensor,
    *,
    steps: int,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate the deterministic Flow ODE in final raw target-rate space."""
    if initial_noise is None:
        initial_noise = torch.zeros(
            observation.shape[0],
            policy.chunk_dim,
            device=observation.device,
            dtype=observation.dtype,
        )
    policy._validate_inputs(observation, initial_noise, steps)
    obs_prep = policy._prepare_observation(observation)

    latent = initial_noise
    sigma_schedule = torch.linspace(
        1.0,
        0.0,
        steps + 1,
        device=initial_noise.device,
        dtype=initial_noise.dtype,
    )
    for step_index in range(steps):
        sigma = sigma_schedule[step_index]
        time_batch = torch.full(
            (initial_noise.shape[0],),
            float(sigma.item()),
            device=initial_noise.device,
            dtype=initial_noise.dtype,
        )
        model_output = policy.velocity_field(obs_prep, latent, time_batch)
        latent = flow_ode_mean(model_output, latent, sigma_schedule, step_index)

    return policy.reshape_raw_targets(latent)
