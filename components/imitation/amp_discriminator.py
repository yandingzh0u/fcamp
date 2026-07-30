"""Official MimicKit-style AMP discriminator training contract.

The component deliberately owns only the discriminator side of AMP.  The
caller remains responsible for sampling independent current, replay, and
expert windows.  Every optimization batch must contain ``B`` samples from
each domain:

* current policy ``B``;
* replay policy ``B``;
* expert demonstration ``B``.

Current and replay are concatenated into one ``2B`` negative batch.  The BCE
objective nevertheless gives the complete negative class and expert class
equal weight.  Consequently the effective domain weights are 0.25 current,
0.25 replay, and 0.50 expert.

Normalizer updates are transactional.  ``begin_normalizer_update`` records an
equal number of current/expert raw observations and freezes the normalizer.
Discriminator optimization and reward evaluation continue to use the
previously committed statistics.  The caller explicitly commits only after
all discriminator steps with ``commit_normalizer_update``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch
from torch import nn

from components.imitation.style_reward import (
    discriminator_style_reward,
    style_reward_statistics,
)
from components.normalization.running_stats import RunningNormalizer
from models.style_discriminator import (
    StyleDiscriminator,
    StyleDiscriminatorLossOutput,
    compute_style_discriminator_loss,
)


_DEFAULT_HIDDEN_DIMS: Final[tuple[int, int]] = (1024, 512)


@dataclass(frozen=True)
class AMPRewardOutput:
    """Discriminator logits, AMP rewards, and detached scalar diagnostics."""

    logits: torch.Tensor
    rewards: torch.Tensor
    metrics: dict[str, float]


def _scalar_like(reference: torch.Tensor, value: float | int) -> torch.Tensor:
    return reference.new_tensor(float(value))


def _validate_flat_observations(
    observations: torch.Tensor,
    *,
    input_dim: int,
    domain: str,
) -> torch.Tensor:
    if observations.ndim != 2 or observations.shape[1] != input_dim:
        raise ValueError(
            f"{domain} AMP observations must have shape [B,{input_dim}], "
            f"got {tuple(observations.shape)}"
        )
    if observations.shape[0] <= 0:
        raise ValueError(f"{domain} AMP observations must contain at least one sample")
    if observations.dtype != torch.float32:
        raise TypeError(
            f"{domain} AMP observations must be float32, got {observations.dtype}"
        )
    if not bool(torch.isfinite(observations).all()):
        raise ValueError(f"{domain} AMP observations contain non-finite values")
    return observations


class AMPDiscriminator(nn.Module):
    """MimicKit AMP discriminator, optimizer, reward, and normalization lifecycle.

    Inputs are canonicalized, flattened raw motion windows.  Canonicalization
    and temporal sampling intentionally stay outside this class so the caller
    can use the exact same window pipeline for current, replay, and expert
    domains.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: tuple[int, ...] = _DEFAULT_HIDDEN_DIMS,
        learning_rate: float = 2.5e-4,
        momentum: float = 0.9,
        weight_decay: float = 1.0e-4,
        gradient_penalty_weight: float = 10.0,
        logit_regularization_weight: float = 0.01,
        reward_scale: float = 2.0,
        reward_epsilon: float = 1.0e-4,
        normalizer_clip: float = 10.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("AMP discriminator input_dim must be a positive integer")
        if not hidden_dims or any(int(width) <= 0 for width in hidden_dims):
            raise ValueError("AMP discriminator hidden_dims must be positive")
        if learning_rate <= 0.0:
            raise ValueError("AMP discriminator learning_rate must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("AMP discriminator momentum must lie in [0,1)")
        if weight_decay < 0.0:
            raise ValueError("AMP discriminator weight_decay must be non-negative")
        if gradient_penalty_weight < 0.0:
            raise ValueError(
                "AMP discriminator gradient_penalty_weight must be non-negative"
            )
        if logit_regularization_weight < 0.0:
            raise ValueError(
                "AMP discriminator logit_regularization_weight must be non-negative"
            )
        if reward_scale < 0.0:
            raise ValueError("AMP discriminator reward_scale must be non-negative")
        if not 0.0 < reward_epsilon < 1.0:
            raise ValueError("AMP discriminator reward_epsilon must lie in (0,1)")

        self.input_dim = int(input_dim)
        self.gradient_penalty_weight = float(gradient_penalty_weight)
        self.logit_regularization_weight = float(logit_regularization_weight)
        self.reward_scale = float(reward_scale)
        self.reward_epsilon = float(reward_epsilon)
        self.momentum = float(momentum)
        self.optimizer_weight_decay = float(weight_decay)

        self.discriminator = StyleDiscriminator(
            self.input_dim,
            hidden_dims=tuple(int(width) for width in hidden_dims),
        ).to(device)
        self.normalizer = RunningNormalizer(
            self.input_dim,
            device=device,
            clip=float(normalizer_clip),
        )
        self.optimizer = torch.optim.SGD(
            self.discriminator.parameters(),
            lr=float(learning_rate),
            momentum=self.momentum,
            weight_decay=self.optimizer_weight_decay,
        )
        self.register_buffer(
            "optimizer_steps",
            torch.zeros((), dtype=torch.long, device=device),
        )
        self._normalizer_update_open = False

    @property
    def device(self) -> torch.device:
        return next(self.discriminator.parameters()).device

    @property
    def normalizer_update_open(self) -> bool:
        return self._normalizer_update_open

    def _validate_batch_triplet(
        self,
        *,
        current_observations: torch.Tensor,
        replay_observations: torch.Tensor,
        expert_observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = _validate_flat_observations(
            current_observations,
            input_dim=self.input_dim,
            domain="current",
        )
        replay = _validate_flat_observations(
            replay_observations,
            input_dim=self.input_dim,
            domain="replay",
        )
        expert = _validate_flat_observations(
            expert_observations,
            input_dim=self.input_dim,
            domain="expert",
        )
        batch_size = int(current.shape[0])
        if replay.shape[0] != batch_size or expert.shape[0] != batch_size:
            raise ValueError(
                "official AMP discriminator batches require exactly B current, "
                "B replay, and B expert samples; got "
                f"{batch_size}, {replay.shape[0]}, {expert.shape[0]}"
            )
        devices = {current.device, replay.device, expert.device, self.device}
        if len(devices) != 1:
            raise ValueError(
                "AMP discriminator observations and module must share one device; "
                f"got {sorted(str(device) for device in devices)}"
            )
        return current, replay, expert

    def normalize(self, observations: torch.Tensor) -> torch.Tensor:
        """Normalize raw flat windows using committed statistics only."""

        observations = _validate_flat_observations(
            observations,
            input_dim=self.input_dim,
            domain="reward",
        )
        if observations.device != self.device:
            raise ValueError(
                f"AMP observations are on {observations.device}, discriminator is on {self.device}"
            )
        return self.normalizer.normalize(observations)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Evaluate logits from raw canonical flat windows."""

        return self.discriminator(self.normalize(observations))

    def compute_batch_loss(
        self,
        *,
        current_observations: torch.Tensor,
        replay_observations: torch.Tensor,
        expert_observations: torch.Tensor,
    ) -> StyleDiscriminatorLossOutput:
        """Build one exact AMP objective without stepping the optimizer."""

        current, replay, expert = self._validate_batch_triplet(
            current_observations=current_observations,
            replay_observations=replay_observations,
            expert_observations=expert_observations,
        )
        normalized_current = self.normalizer.normalize(current)
        normalized_replay = self.normalizer.normalize(replay)
        normalized_expert = self.normalizer.normalize(expert)
        output = compute_style_discriminator_loss(
            self.discriminator,
            expert_observations=normalized_expert,
            policy_observations=normalized_current,
            replay_observations=normalized_replay,
            gradient_penalty_weight=self.gradient_penalty_weight,
            logit_regularization_weight=self.logit_regularization_weight,
        )

        # Although the raw sample ratio is 2:1 negative:positive, MimicKit's
        # mean(fake BCE)/mean(expert BCE) split gives equal class weights.
        batch_size = int(current.shape[0])
        reference = output.loss.detach()
        output.metrics.update(
            {
                "disc/current_count": _scalar_like(reference, batch_size),
                "disc/replay_count": _scalar_like(reference, batch_size),
                "disc/expert_count": _scalar_like(reference, batch_size),
                "disc/fake_count": _scalar_like(reference, 2 * batch_size),
                "disc/total_count": _scalar_like(reference, 3 * batch_size),
                "disc/raw_negative_fraction": _scalar_like(reference, 2.0 / 3.0),
                "disc/raw_positive_fraction": _scalar_like(reference, 1.0 / 3.0),
                "disc/effective_fake_loss_weight": _scalar_like(reference, 0.5),
                "disc/effective_expert_loss_weight": _scalar_like(reference, 0.5),
                "disc/effective_current_loss_weight": _scalar_like(reference, 0.25),
                "disc/effective_replay_loss_weight": _scalar_like(reference, 0.25),
                "disc/effective_fake_gp_weight": _scalar_like(reference, 0.5),
                "disc/effective_expert_gp_weight": _scalar_like(reference, 0.5),
                "disc/gradient_penalty_weight": _scalar_like(
                    reference, self.gradient_penalty_weight
                ),
                "disc/logit_regularization_weight": _scalar_like(
                    reference, self.logit_regularization_weight
                ),
            }
        )
        for domain in ("current", "replay", "expert"):
            for quantile in ("p05", "p50", "p95"):
                logit_value = output.metrics[f"disc/{domain}_logit_{quantile}"]
                output.metrics[f"disc/{domain}_prob_{quantile}"] = torch.sigmoid(
                    logit_value
                )
        return output

    def train_batch(
        self,
        *,
        current_observations: torch.Tensor,
        replay_observations: torch.Tensor,
        expert_observations: torch.Tensor,
        micro_batch_size: int | None = None,
    ) -> dict[str, float]:
        """Apply one logical-batch SGD discriminator update.

        This method intentionally neither records nor commits normalizer
        statistics.  Reusing a minibatch across discriminator epochs therefore
        cannot accidentally count it multiple times in the running moments.

        ``micro_batch_size`` is only a memory-control knob.  Microbatch losses
        are weighted by their share of logical ``B`` and accumulated before a
        *single* optimizer step.  It therefore does not alter the official AMP
        logical batch size or discriminator-step count.
        """

        current, replay, expert = self._validate_batch_triplet(
            current_observations=current_observations,
            replay_observations=replay_observations,
            expert_observations=expert_observations,
        )
        logical_batch_size = int(current.shape[0])
        if micro_batch_size is None:
            micro_batch_size = logical_batch_size
        if (
            isinstance(micro_batch_size, bool)
            or not isinstance(micro_batch_size, int)
            or micro_batch_size <= 0
        ):
            raise ValueError(
                "AMP discriminator micro_batch_size must be a positive integer"
            )
        micro_batch_size = min(int(micro_batch_size), logical_batch_size)
        micro_batch_count = math.ceil(logical_batch_size / micro_batch_size)

        self.discriminator.train()
        self.optimizer.zero_grad(set_to_none=True)
        weighted_metrics: dict[str, float] = {}
        for start in range(0, logical_batch_size, micro_batch_size):
            stop = min(start + micro_batch_size, logical_batch_size)
            count = stop - start
            output = self.compute_batch_loss(
                current_observations=current[start:stop],
                replay_observations=replay[start:stop],
                expert_observations=expert[start:stop],
            )
            if not bool(torch.isfinite(output.loss)):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("AMP discriminator loss is non-finite")
            weight = count / logical_batch_size
            (output.loss * weight).backward()
            for key, value in output.metrics.items():
                weighted_metrics[key] = weighted_metrics.get(key, 0.0) + (
                    weight * float(value.detach().item())
                )

        reference = next(self.discriminator.parameters())
        grad_norm_sq = reference.new_zeros(())
        for parameter in self.discriminator.parameters():
            if parameter.grad is not None:
                grad_norm_sq = grad_norm_sq + parameter.grad.detach().square().sum()
        grad_norm = torch.sqrt(grad_norm_sq)
        if not bool(torch.isfinite(grad_norm)):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("AMP discriminator gradient norm is non-finite")
        # MimicKit does not clip discriminator gradients.
        self.optimizer.step()
        self.optimizer_steps.add_(1)

        metrics = weighted_metrics
        # Weighted microbatch std/quantiles are not distribution statistics.
        # Re-evaluate pre-step metrics would require retaining a second model
        # snapshot, so expose their aggregation contract explicitly.  Means,
        # BCE, accuracy, GP, and gradients are exact logical-batch quantities.
        metrics.update(
            {
                "disc/current_count": float(logical_batch_size),
                "disc/replay_count": float(logical_batch_size),
                "disc/expert_count": float(logical_batch_size),
                "disc/fake_count": float(2 * logical_batch_size),
                "disc/total_count": float(3 * logical_batch_size),
            }
        )
        parameter_group = self.optimizer.param_groups[0]
        metrics.update(
            {
                "disc/grad_norm": float(grad_norm.item()),
                "disc/lr": float(parameter_group["lr"]),
                "disc/optimizer_momentum": float(parameter_group["momentum"]),
                "disc/optimizer_weight_decay": float(parameter_group["weight_decay"]),
                "disc/optimizer_steps": float(self.optimizer_steps.item()),
                "disc/logical_batch_size": float(logical_batch_size),
                "disc/micro_batch_size": float(micro_batch_size),
                "disc/micro_batch_count": float(micro_batch_count),
                "disc/gradient_accumulation_single_step_contract": 1.0,
                "disc/distribution_quantile_micro_averaged": float(
                    micro_batch_count > 1
                ),
                "disc_norm/count_during_update": float(self.normalizer.count.item()),
                "disc_norm/pending_count_during_update": float(
                    self.normalizer.pending_count.item()
                ),
                "disc_norm/using_committed_stats": 1.0,
                "disc_norm/commit_inside_train_batch": 0.0,
            }
        )
        return metrics

    @torch.no_grad()
    def open_normalizer_update(self) -> None:
        """Open an empty transactional update for streamed observations."""
        if self._normalizer_update_open:
            raise RuntimeError("an AMP normalizer update is already open")
        if bool(self.normalizer.frozen.item()):
            raise RuntimeError("AMP normalizer is already frozen outside this component")
        self.normalizer.clear_pending()
        self.normalizer.freeze()
        self._normalizer_update_open = True

    @torch.no_grad()
    def record_normalizer_update_batch(
        self,
        *,
        current_observations: torch.Tensor,
        expert_observations: torch.Tensor,
    ) -> None:
        """Record one equal-domain pair into an open normalizer transaction."""

        if not self._normalizer_update_open:
            raise RuntimeError("no AMP normalizer update is open")
        current = _validate_flat_observations(
            current_observations,
            input_dim=self.input_dim,
            domain="normalizer current",
        )
        expert = _validate_flat_observations(
            expert_observations,
            input_dim=self.input_dim,
            domain="normalizer expert",
        )
        if current.shape[0] != expert.shape[0]:
            raise ValueError(
                "AMP normalizer requires equal current/expert sample counts; "
                f"got {current.shape[0]} and {expert.shape[0]}"
            )
        if current.device != self.device or expert.device != self.device:
            raise ValueError(
                "AMP normalizer observations and discriminator must share one device"
            )
        self.normalizer.record(current)
        self.normalizer.record(expert)

    @torch.no_grad()
    def begin_normalizer_update(
        self,
        *,
        current_observations: torch.Tensor,
        expert_observations: torch.Tensor,
    ) -> dict[str, float]:
        """Record equal current/expert samples while freezing committed moments."""

        self.open_normalizer_update()
        self.record_normalizer_update_batch(
            current_observations=current_observations,
            expert_observations=expert_observations,
        )
        return {
            "disc_norm/count_before_update": float(self.normalizer.count.item()),
            "disc_norm/pending_current_count": float(
                current_observations.shape[0]
            ),
            "disc_norm/pending_expert_count": float(
                expert_observations.shape[0]
            ),
            "disc_norm/pending_count": float(self.normalizer.pending_count.item()),
            "disc_norm/equal_domain_weight_contract": 1.0,
            "disc_norm/frozen_during_discriminator_update": 1.0,
        }

    @torch.no_grad()
    def commit_normalizer_update(self) -> bool:
        """Commit recorded moments after the caller finishes all D updates."""

        if not self._normalizer_update_open:
            raise RuntimeError("no AMP normalizer update is open")
        self.normalizer.unfreeze()
        committed = bool(self.normalizer.commit())
        self._normalizer_update_open = False
        return committed

    @torch.no_grad()
    def abort_normalizer_update(self) -> None:
        """Discard pending moments after a failed discriminator iteration."""

        if not self._normalizer_update_open:
            raise RuntimeError("no AMP normalizer update is open")
        self.normalizer.clear_pending()
        self.normalizer.unfreeze()
        self._normalizer_update_open = False

    @torch.no_grad()
    def evaluate_logits(
        self,
        observations: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Evaluate raw-window logits in bounded minibatches."""

        observations = _validate_flat_observations(
            observations,
            input_dim=self.input_dim,
            domain="reward",
        )
        if observations.device != self.device:
            raise ValueError(
                f"AMP observations are on {observations.device}, discriminator is on {self.device}"
            )
        count = int(observations.shape[0])
        if batch_size is None:
            batch_size = count
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("AMP reward batch_size must be a positive integer")

        was_training = self.discriminator.training
        self.discriminator.eval()
        logits = torch.cat(
            [
                self.discriminator(
                    self.normalizer.normalize(observations[start : start + batch_size])
                )
                for start in range(0, count, batch_size)
            ],
            dim=0,
        )
        self.discriminator.train(was_training)
        return logits

    @torch.no_grad()
    def evaluate_reward(
        self,
        observations: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> AMPRewardOutput:
        """Evaluate the exact pure-AMP style reward from committed statistics."""

        logits = self.evaluate_logits(observations, batch_size=batch_size)
        rewards = discriminator_style_reward(
            logits,
            scale=self.reward_scale,
            minimum_one_minus_prob=self.reward_epsilon,
        )
        metrics = style_reward_statistics(
            logits,
            rewards,
            scale=self.reward_scale,
            minimum_one_minus_prob=self.reward_epsilon,
            prefix="amp_reward",
        )
        flat_logits = logits.detach().float().reshape(-1)
        quantiles = torch.quantile(
            flat_logits,
            flat_logits.new_tensor([0.05, 0.5, 0.95]),
        )
        metrics.update(
            {
                "amp_reward/sample_count": float(flat_logits.numel()),
                "amp_reward/logit_mean": float(flat_logits.mean().item()),
                "amp_reward/logit_std": float(
                    flat_logits.std(unbiased=False).item()
                ),
                "amp_reward/logit_p05": float(quantiles[0].item()),
                "amp_reward/logit_p50": float(quantiles[1].item()),
                "amp_reward/logit_p95": float(quantiles[2].item()),
                "amp_reward/normalizer_count": float(self.normalizer.count.item()),
                "amp_reward/discriminator_optimizer_steps": float(
                    self.optimizer_steps.item()
                ),
                "amp_reward/formula_scale": self.reward_scale,
                "amp_reward/formula_epsilon": self.reward_epsilon,
                "amp_reward/no_dt_multiplier_contract": 1.0,
            }
        )
        return AMPRewardOutput(logits=logits, rewards=rewards, metrics=metrics)

    @torch.no_grad()
    def reward(
        self,
        observations: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Convenience wrapper returning only AMP rewards."""

        return self.evaluate_reward(observations, batch_size=batch_size).rewards


__all__ = ["AMPDiscriminator", "AMPRewardOutput"]
