"""Rollout-facing base for a standard absolute-action AMP Gaussian policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from components.normalization.running_stats import EmpiricalNormalization
from models.amp_actor_critic import (
    AMPActorCritic,
    DiagonalGaussian,
)


@dataclass(frozen=True)
class GaussianPolicySample:
    """Exact rollout quantities for one normalized absolute-action chunk."""

    actions: torch.Tensor
    mean: torch.Tensor
    log_std: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


@dataclass(frozen=True)
class GaussianPolicyStatistics:
    """Recomputed PPO quantities, all reported per horizon offset."""

    log_prob: torch.Tensor
    entropy: torch.Tensor
    old_to_new_kl: torch.Tensor


class AMPGaussianBase:
    """Build the fixed-std direct-action Actor and self-state value function."""

    def __init__(self, cfg, env) -> None:
        self.cfg = cfg
        self.env = env

    def build(self) -> None:
        cfg = self.cfg
        env = self.env

        self.num_act = int(env.action_dim)
        self.actor_obs_dim = int(env.observation_dim)
        self.critic_obs_dim = self.actor_obs_dim
        self.horizon_h = int(cfg.horizon)

        self.actor_critic = AMPActorCritic(
            observation_dim=self.actor_obs_dim,
            action_dim=self.num_act,
            horizon=self.horizon_h,
            actor_hidden_dims=tuple(cfg.actor_hidden_dims),
            critic_hidden_dims=tuple(cfg.critic.hidden_dims),
        ).to(env.device)
        self._policy = self.actor_critic.actor
        self.critic = self.actor_critic.critic
        self.chunk_dim = self.horizon_h * self.num_act

        self.actor_obs_normalizer = EmpiricalNormalization(
            self.actor_obs_dim,
            env.device,
        )

        self.learning_rate = float(cfg.policy_lr)
        self.critic_learning_rate = float(cfg.value_lr)
        self.actor_optimizer = torch.optim.SGD(
            self._policy.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
        )
        self.critic_optimizer = torch.optim.SGD(
            self.critic.parameters(),
            lr=self.critic_learning_rate,
            momentum=0.9,
        )

        self._policy_module = nn.ModuleDict(
            {
                "actor": self._policy,
                "actor_obs_normalizer": self.actor_obs_normalizer,
                "critic": self.critic,
            }
        )

    @property
    def policy(self) -> nn.Module:
        return self._policy_module

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.actor_optimizer

    @property
    def horizon(self) -> int:
        return self.horizon_h

    def normalize_actor_observation(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.actor_obs_normalizer(observations)

    def policy_distribution(
        self,
        normalized_observations: torch.Tensor,
    ) -> DiagonalGaussian:
        """Construct the policy density from already-normalized observations."""

        return self._policy.distribution(normalized_observations)

    def sample_normalized_action_chunk(
        self,
        normalized_observations: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> GaussianPolicySample:
        """Sample direct normalized absolute actions and exact density terms."""

        distribution = self.policy_distribution(normalized_observations)
        actions = distribution.sample(noise)
        return GaussianPolicySample(
            actions=actions,
            mean=distribution.mean,
            log_std=distribution.log_std,
            log_prob=distribution.log_prob(actions),
            entropy=distribution.entropy(),
        )

    def deterministic_normalized_action_chunk(
        self,
        normalized_observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.policy_distribution(normalized_observations).mode

    def deterministic_actions(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized absolute actions for raw actor observations."""

        normalized = self.normalize_actor_observation(observations)
        return self.deterministic_normalized_action_chunk(normalized)

    def recompute_policy_statistics(
        self,
        normalized_observations: torch.Tensor,
        sampled_actions: torch.Tensor,
        old_mean: torch.Tensor,
        old_log_std: torch.Tensor,
    ) -> GaussianPolicyStatistics:
        """Recompute exact PPO density, entropy, and ``KL(old || new)``."""

        new_distribution = self.policy_distribution(
            normalized_observations
        )
        old_distribution = DiagonalGaussian(old_mean, old_log_std)
        return GaussianPolicyStatistics(
            log_prob=new_distribution.log_prob(sampled_actions),
            entropy=new_distribution.entropy(),
            old_to_new_kl=old_distribution.kl_divergence(
                new_distribution
            ),
        )

    def value_from_normalized_observation(
        self,
        normalized_observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.critic(normalized_observations)

    def extra_checkpoint_state(self) -> dict:
        return {
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "learning_rate": float(self.learning_rate),
            "critic_learning_rate": float(self.critic_learning_rate),
            "actor_obs_normalizer": self.actor_obs_normalizer.state_dict(),
        }

    def load_extra_checkpoint_state(
        self,
        payload: dict,
        reset_optimizer: bool = False,
    ) -> None:
        if reset_optimizer:
            self.learning_rate = float(self.cfg.policy_lr)
            self.critic_learning_rate = float(self.cfg.value_lr)
        else:
            self.learning_rate = float(
                payload.get("learning_rate", self.learning_rate)
            )
            self.critic_learning_rate = float(
                payload.get(
                    "critic_learning_rate",
                    self.critic_learning_rate,
                )
            )
            critic_optimizer = payload.get("critic_optimizer")
            if critic_optimizer is not None:
                self.critic_optimizer.load_state_dict(critic_optimizer)

        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.learning_rate
        for group in self.critic_optimizer.param_groups:
            group["lr"] = self.critic_learning_rate

        normalizer_state = payload.get("actor_obs_normalizer")
        if normalizer_state is not None:
            self.actor_obs_normalizer.load_state_dict(normalizer_state)


__all__ = [
    "AMPGaussianBase",
    "GaussianPolicySample",
    "GaussianPolicyStatistics",
]
