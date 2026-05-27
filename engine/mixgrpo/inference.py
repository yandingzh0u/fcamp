from __future__ import annotations

import torch

from .sampling import flow_grpo_step


@torch.no_grad()
def deterministic_sde_ode_actions(
    policy,
    observation: torch.Tensor,
    *,
    steps: int,
    sde_eta: float = 0.7,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate actions through the zero-noise SDE mean path used by MixGRPO training."""
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
    zero_step_noise = torch.zeros_like(initial_noise)
    for step_index in range(steps):
        sigma = sigma_schedule[step_index]
        time_batch = torch.full(
            (initial_noise.shape[0],),
            float(sigma.item()),
            device=initial_noise.device,
            dtype=initial_noise.dtype,
        )
        model_output = policy.velocity_field(obs_prep, latent, time_batch)
        latent, _ = flow_grpo_step(
            model_output=model_output,
            latents=latent,
            sigmas=sigma_schedule,
            index=step_index,
            eta=sde_eta,
            deterministic=False,
            sample_noise=zero_step_noise,
            horizon=policy.horizon,
        )

    return policy._action_transform(latent).view(
        initial_noise.shape[0],
        policy.horizon,
        policy.action_dim,
    )
