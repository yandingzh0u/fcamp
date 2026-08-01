from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def flow_ode_mean(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    index: int,
) -> torch.Tensor:
    """Advance one Euler step of the deterministic Flow ODE."""

    sigma = sigmas[index].to(device=model_output.device, dtype=model_output.dtype)
    sigma_next = sigmas[index + 1].to(
        device=model_output.device,
        dtype=model_output.dtype,
    )
    return latents + model_output * (sigma_next - sigma)


def _activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for hidden_dim in hidden_dims:
        width = int(hidden_dim)
        if width <= 0:
            raise ValueError("hidden dimensions must be positive")
        layers.extend((nn.Linear(input_dim, width), _activation(activation)))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class FlowMatchingPolicy(nn.Module):
    """Single-step absolute-action Flow actor with learned joint CPS noise."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
        action_limit: float,
        *,
        flow_steps: int,
        cps_noise_init: float,
        cps_cov_rank: int,
    ) -> None:
        super().__init__()
        if int(obs_dim) <= 0 or int(action_dim) <= 0:
            raise ValueError("obs_dim and action_dim must be positive")
        if int(flow_steps) <= 0:
            raise ValueError("flow_steps must be positive")
        if float(action_limit) <= 0.0:
            raise ValueError("action_limit must be positive")
        if int(cps_cov_rank) < 0:
            raise ValueError("cps_cov_rank must be non-negative")
        if not 1.0e-4 < float(cps_noise_init) < 1.0 - 1.0e-4:
            raise ValueError("cps_noise_init must lie in (1e-4, 1-1e-4)")

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.action_limit = float(action_limit)
        self.flow_steps = int(flow_steps)
        self.cps_cov_rank = int(cps_cov_rank)
        self.velocity_net = _mlp(
            self.obs_dim + self.action_dim + 1,
            hidden_dims,
            self.action_dim,
            activation,
        )

        initial_diagonal = math.log(math.expm1(1.0))
        self.cps_diag_raw = nn.Parameter(
            torch.full((self.flow_steps, self.action_dim), initial_diagonal)
        )
        self.cps_lowrank_raw = nn.Parameter(
            1.0e-3
            * torch.randn(
                self.flow_steps,
                self.action_dim,
                self.cps_cov_rank,
            )
        )
        eta_epsilon = 1.0e-4
        eta_unit = (
            float(cps_noise_init) - eta_epsilon
        ) / (1.0 - 2.0 * eta_epsilon)
        self.cps_eta_raw = nn.Parameter(
            torch.tensor(math.log(eta_unit / (1.0 - eta_unit)))
        )

    def _validate_observation(self, observation: torch.Tensor) -> None:
        if observation.ndim != 2 or observation.shape[-1] != self.obs_dim:
            raise ValueError(
                f"observation must have shape [batch, {self.obs_dim}], "
                f"got {tuple(observation.shape)}"
            )

    def _validate_latent(self, latent: torch.Tensor, batch: int) -> None:
        if latent.shape != (batch, self.action_dim):
            raise ValueError(
                f"latent must have shape {(batch, self.action_dim)}, "
                f"got {tuple(latent.shape)}"
            )

    def velocity_field(
        self,
        observation: torch.Tensor,
        latent: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_observation(observation)
        self._validate_latent(latent, observation.shape[0])
        if time.shape != (observation.shape[0],):
            raise ValueError(
                f"time must have shape {(observation.shape[0],)}, "
                f"got {tuple(time.shape)}"
            )
        return self.velocity_net(
            torch.cat((observation, latent, time.unsqueeze(-1)), dim=-1)
        )

    def action_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Map an absolute pre-tanh command to the strict action domain."""

        if latent.shape[-1] != self.action_dim:
            raise ValueError(
                f"latent last dimension must be {self.action_dim}, "
                f"got {latent.shape[-1]}"
            )
        limit = self.action_limit
        return limit * torch.tanh(latent / limit)

    def eta(self) -> torch.Tensor:
        epsilon = torch.as_tensor(
            1.0e-4,
            device=self.cps_eta_raw.device,
            dtype=self.cps_eta_raw.dtype,
        )
        return epsilon + (1.0 - 2.0 * epsilon) * torch.sigmoid(
            self.cps_eta_raw
        )

    def covariance_factors(
        self,
        step_index: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return trace-normalized diagonal-plus-low-rank CPS factors."""

        if not 0 <= int(step_index) < self.flow_steps:
            raise IndexError(f"invalid flow step {step_index}")
        reference = self.cps_diag_raw
        device = reference.device if device is None else torch.device(device)
        dtype = reference.dtype if dtype is None else dtype
        diagonal = (
            F.softplus(
                self.cps_diag_raw[int(step_index)].to(
                    device=device,
                    dtype=dtype,
                )
            )
            + 1.0e-4
        )
        low_rank = self.cps_lowrank_raw[int(step_index)].to(
            device=device,
            dtype=dtype,
        )
        mean_variance = (
            diagonal.square().sum() + low_rank.square().sum()
        ) / float(self.action_dim)
        shape_scale = torch.sqrt(mean_variance.clamp_min(1.0e-12))
        diagonal = diagonal / shape_scale
        low_rank = low_rank / shape_scale
        covariance = (
            torch.diag_embed(diagonal.square())
            + low_rank @ low_rank.transpose(0, 1)
        )
        cholesky = torch.linalg.cholesky(covariance)
        log_cholesky_diagonal = torch.log(
            torch.diagonal(cholesky).clamp_min(1.0e-8)
        )
        normalized_mean_variance = (
            diagonal.square().sum() + low_rank.square().sum()
        ) / float(self.action_dim)
        return (
            diagonal,
            low_rank,
            covariance,
            cholesky,
            log_cholesky_diagonal,
            normalized_mean_variance,
        )

    def cps_step_coefficients(
        self,
        step_index: int,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= int(step_index) < self.flow_steps:
            raise IndexError(f"invalid flow step {step_index}")
        delta_sigma = torch.as_tensor(
            1.0 / float(self.flow_steps),
            device=reference.device,
            dtype=reference.dtype,
        )
        eta = self.eta().to(device=reference.device, dtype=reference.dtype)
        noise_coefficient = torch.sin(0.5 * math.pi * eta) * torch.sqrt(
            delta_sigma
        )
        preserved_coefficient = torch.sqrt(
            torch.clamp(1.0 - noise_coefficient.square(), min=1.0e-12)
        )
        return preserved_coefficient, noise_coefficient

    def _sigma_schedule(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.linspace(
            1.0,
            0.0,
            self.flow_steps + 1,
            device=reference.device,
            dtype=reference.dtype,
        )

    def _backbone_step(
        self,
        observation: torch.Tensor,
        mean_latent: torch.Tensor,
        sigma_schedule: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time = torch.full(
            (observation.shape[0],),
            float(sigma_schedule[step_index].item()),
            device=mean_latent.device,
            dtype=mean_latent.dtype,
        )
        velocity = self.velocity_field(observation, mean_latent, time)
        next_mean = flow_ode_mean(
            velocity,
            mean_latent,
            sigma_schedule,
            step_index,
        )
        preserved, noise = self.cps_step_coefficients(
            step_index,
            mean_latent,
        )
        return next_mean, preserved, noise

    def _sample_innovation(
        self,
        batch: int,
        step_index: int,
        noise_coefficient: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        diagonal, low_rank, _, _, _, _ = self.covariance_factors(
            step_index,
            device=reference.device,
            dtype=reference.dtype,
        )
        innovation = torch.randn_like(reference) * diagonal
        if self.cps_cov_rank:
            rank_noise = torch.randn(
                batch,
                self.cps_cov_rank,
                device=reference.device,
                dtype=reference.dtype,
            )
            innovation = innovation + rank_noise @ low_rank.transpose(0, 1)
        return innovation * noise_coefficient

    def _innovation_log_prob(
        self,
        innovation: torch.Tensor,
        noise_coefficient: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        if innovation.ndim != 2 or innovation.shape[-1] != self.action_dim:
            raise ValueError(
                "innovation must have shape "
                f"[batch, {self.action_dim}], got {tuple(innovation.shape)}"
            )
        _, _, _, cholesky, log_cholesky_diagonal, _ = (
            self.covariance_factors(
                step_index,
                device=innovation.device,
                dtype=innovation.dtype,
            )
        )
        standard_deviation = torch.clamp(noise_coefficient, min=1.0e-6)
        whitened = torch.linalg.solve_triangular(
            cholesky,
            (innovation / standard_deviation).transpose(0, 1),
            upper=False,
        ).transpose(0, 1)
        return (
            -0.5
            * (
                whitened.square().sum(dim=-1)
                + self.action_dim * math.log(2.0 * math.pi)
            )
            - log_cholesky_diagonal.sum()
            - self.action_dim * torch.log(standard_deviation)
        )

    def deterministic_latent(self, observation: torch.Tensor) -> torch.Tensor:
        self._validate_observation(observation)
        mean_latent = torch.zeros(
            observation.shape[0],
            self.action_dim,
            device=observation.device,
            dtype=observation.dtype,
        )
        sigma_schedule = self._sigma_schedule(observation)
        for step_index in range(self.flow_steps):
            mean_latent, _, _ = self._backbone_step(
                observation,
                mean_latent,
                sigma_schedule,
                step_index,
            )
        return mean_latent

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.action_from_latent(self.deterministic_latent(observation))

    def sample(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Sample one action and retain the exact augmented CPS path density."""

        self._validate_observation(observation)
        batch = observation.shape[0]
        mean_latent = torch.zeros(
            batch,
            self.action_dim,
            device=observation.device,
            dtype=observation.dtype,
        )
        sampled_latent = torch.zeros_like(mean_latent)
        sigma_schedule = self._sigma_schedule(observation)
        latent_path = torch.zeros(
            batch,
            self.flow_steps + 1,
            self.action_dim,
            device=observation.device,
            dtype=observation.dtype,
        )
        step_log_probabilities = torch.zeros(
            batch,
            self.flow_steps,
            device=observation.device,
            dtype=observation.dtype,
        )
        innovation_rms = torch.zeros(
            self.flow_steps,
            device=observation.device,
            dtype=observation.dtype,
        )

        for step_index in range(self.flow_steps):
            next_mean, preserved, noise = self._backbone_step(
                observation,
                mean_latent,
                sigma_schedule,
                step_index,
            )
            conditional_mean = next_mean + preserved * (
                sampled_latent - mean_latent
            )
            innovation = self._sample_innovation(
                batch,
                step_index,
                noise,
                sampled_latent,
            )
            sampled_latent = conditional_mean + innovation
            # The stored float32 path is the policy sample.  Density must be
            # evaluated for the innovation represented by that exact stored
            # path, rather than the pre-addition value that can differ by one
            # rounding unit after conditional_mean + innovation.
            represented_innovation = sampled_latent - conditional_mean
            latent_path[:, step_index + 1] = sampled_latent
            step_log_probabilities[:, step_index] = (
                self._innovation_log_prob(
                    represented_innovation,
                    noise,
                    step_index,
                )
            )
            innovation_rms[step_index] = torch.sqrt(
                represented_innovation.square().mean()
            )
            mean_latent = next_mean

        action = self.action_from_latent(sampled_latent)
        mean_action = self.action_from_latent(mean_latent)
        diagnostics = {
            "innovation_rms_per_flow_step": innovation_rms,
            "final_latent_rms": torch.sqrt(sampled_latent.square().mean()),
            "action_rms": torch.sqrt(action.square().mean()),
            "mean_action": mean_action,
        }
        return action, latent_path, step_log_probabilities, diagnostics

    def recompute_log_probs(
        self,
        observation: torch.Tensor,
        latent_path: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute exact CPS transition densities for a stored latent path."""

        self._validate_observation(observation)
        expected = (
            observation.shape[0],
            self.flow_steps + 1,
            self.action_dim,
        )
        if latent_path.shape != expected:
            raise ValueError(
                f"latent_path must have shape {expected}, "
                f"got {tuple(latent_path.shape)}"
            )
        mean_latent = torch.zeros_like(latent_path[:, 0])
        sigma_schedule = self._sigma_schedule(observation)
        step_log_probabilities: list[torch.Tensor] = []
        for step_index in range(self.flow_steps):
            sampled_latent = latent_path[:, step_index]
            next_latent = latent_path[:, step_index + 1]
            next_mean, preserved, noise = self._backbone_step(
                observation,
                mean_latent,
                sigma_schedule,
                step_index,
            )
            conditional_mean = next_mean + preserved * (
                sampled_latent - mean_latent
            )
            step_log_probabilities.append(
                self._innovation_log_prob(
                    next_latent - conditional_mean,
                    noise,
                    step_index,
                )
            )
            mean_latent = next_mean
        return torch.stack(step_log_probabilities, dim=1)
