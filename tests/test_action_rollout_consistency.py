from __future__ import annotations

import torch

from net.mixgrpo.flow_policy import FlowMatchingPolicy
from engine.mixgrpo.sampling import flow_grpo_step
from engine.mixgrpo.inference import deterministic_sde_ode_actions

OBS_DIM = 196
ACTION_DIM = 29


def _make_policy(horizon: int = 12, basis_count: int = 4):
    torch.manual_seed(1)
    return FlowMatchingPolicy(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        horizon=horizon,
        hidden_dims=(64,),
        action_squash_scale=5.0,
        basis_count=basis_count,
    )


def _start_action(policy, obs):
    return obs[..., -2 * policy.action_dim : -policy.action_dim]


def _start_prev_action(policy, obs):
    return obs[..., -policy.action_dim :]


def _rollout_like(policy, obs, initial_noise, sde_noise, steps, eta):
    """Mirror of trainer._sde_ode_rollout_actions (single source of truth for the transform)."""
    obs_prep = policy._prepare_observation(obs)
    sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=obs.device, dtype=obs.dtype)
    latent = initial_noise * policy.init_noise_std
    all_latents = [latent.detach()]
    step_log_probs = []
    for i in range(steps):
        t = torch.full((obs.shape[0],), float(sigma_schedule[i].item()), dtype=obs.dtype)
        model_output = policy.velocity_field(obs_prep, latent, t)
        latent, lp = flow_grpo_step(
            model_output=model_output, latents=latent, sigmas=sigma_schedule,
            index=i, eta=eta, deterministic=False, sample_noise=sde_noise[:, i],
        )
        all_latents.append(latent.detach())
        step_log_probs.append(lp)
    actions = policy._action_transform(
        latent, start_action=_start_action(policy, obs), start_prev_action=_start_prev_action(policy, obs)
    )
    return actions, torch.stack(all_latents, dim=1), torch.stack(step_log_probs, dim=1)


def _recompute_logprobs(policy, obs, latent_path, steps, eta):
    """Mirror of trainer._compute_transition_log_probs."""
    obs_prep = policy._prepare_observation(obs)
    sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=obs.device, dtype=obs.dtype)
    lps = []
    for i in range(steps):
        latent_t = latent_path[:, i]
        next_latent = latent_path[:, i + 1]
        t = torch.full((obs.shape[0],), float(sigma_schedule[i].item()), dtype=obs.dtype)
        model_output = policy.velocity_field(obs_prep, latent_t, t)
        _, lp = flow_grpo_step(
            model_output=model_output, latents=latent_t, sigmas=sigma_schedule,
            index=i, eta=eta, prev_sample=next_latent, deterministic=False,
        )
        lps.append(lp)
    return torch.stack(lps, dim=1)


def test_saved_latent_logprob_recompute_matches_rollout() -> None:
    policy = _make_policy()
    policy.eval()
    steps, eta, batch = 4, 0.7, 8
    obs = torch.randn(batch, OBS_DIM)
    initial_noise = torch.randn(batch, policy.chunk_dim)
    sde_noise = torch.randn(batch, steps, policy.chunk_dim)
    with torch.no_grad():
        _, latent_path, rollout_lp = _rollout_like(policy, obs, initial_noise, sde_noise, steps, eta)
        recomputed_lp = _recompute_logprobs(policy, obs, latent_path, steps, eta)
    assert torch.allclose(rollout_lp, recomputed_lp, atol=1e-4), (
        (rollout_lp - recomputed_lp).abs().max().item()
    )


def test_rollout_and_inference_use_same_transform() -> None:
    policy = _make_policy()
    policy.eval()
    steps, eta, batch = 4, 0.7, 6
    obs = torch.randn(batch, OBS_DIM)
    initial_noise = torch.zeros(batch, policy.chunk_dim)
    sde_noise = torch.zeros(batch, steps, policy.chunk_dim)
    with torch.no_grad():
        rollout_actions, _, _ = _rollout_like(policy, obs, initial_noise, sde_noise, steps, eta)
        infer_actions = deterministic_sde_ode_actions(
            policy, obs, steps=steps, sde_eta=eta, initial_noise=initial_noise * policy.init_noise_std,
        ).reshape(batch, -1)
    assert torch.allclose(rollout_actions, infer_actions, atol=1e-5), (
        (rollout_actions - infer_actions).abs().max().item()
    )
    # Inference actions are C0 continuous with their own boundary state (exact in action space).
    infer_chunk = infer_actions.view(batch, policy.horizon, policy.action_dim)
    assert torch.allclose(infer_chunk[:, 0], _start_action(policy, obs), atol=1e-4)
