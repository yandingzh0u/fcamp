from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from components.imitation.amp_discriminator import AMPDiscriminator


def _small_discriminator(**kwargs) -> AMPDiscriminator:
    return AMPDiscriminator(
        3,
        hidden_dims=(7, 5),
        learning_rate=2.5e-4,
        momentum=0.9,
        weight_decay=1.0e-4,
        gradient_penalty_weight=kwargs.pop("gradient_penalty_weight", 10.0),
        logit_regularization_weight=kwargs.pop(
            "logit_regularization_weight", 0.01
        ),
        **kwargs,
    )


def test_amp_discriminator_architecture_and_sgd_contract() -> None:
    component = AMPDiscriminator(12)
    linears = [
        module
        for module in component.discriminator.trunk
        if isinstance(module, nn.Linear)
    ]
    relus = [
        module
        for module in component.discriminator.trunk
        if isinstance(module, nn.ReLU)
    ]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (12, 1024),
        (1024, 512),
    ]
    assert len(relus) == 2
    assert isinstance(component.optimizer, torch.optim.SGD)
    group = component.optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(2.5e-4)
    assert group["momentum"] == pytest.approx(0.9)
    assert group["weight_decay"] == pytest.approx(1.0e-4)
    assert component.discriminator.logit.weight.min() >= -1.0
    assert component.discriminator.logit.weight.max() <= 1.0
    torch.testing.assert_close(
        component.discriminator.logit.bias,
        torch.zeros_like(component.discriminator.logit.bias),
    )


def test_amp_bce_is_half_fake_half_expert_despite_two_to_one_raw_ratio() -> None:
    component = _small_discriminator(
        gradient_penalty_weight=0.0,
        logit_regularization_weight=0.0,
    )
    with torch.no_grad():
        for parameter in component.discriminator.parameters():
            parameter.zero_()
        component.discriminator.logit.bias.fill_(1.25)

    batch_size = 4
    output = component.compute_batch_loss(
        current_observations=torch.zeros(batch_size, 3),
        replay_observations=torch.zeros(batch_size, 3),
        expert_observations=torch.zeros(batch_size, 3),
    )
    logit = torch.tensor(1.25)
    expected_fake = F.binary_cross_entropy_with_logits(logit, torch.tensor(0.0))
    expected_expert = F.binary_cross_entropy_with_logits(logit, torch.tensor(1.0))
    expected = 0.5 * (expected_fake + expected_expert)
    torch.testing.assert_close(output.loss, expected)
    assert output.metrics["disc/current_count"].item() == batch_size
    assert output.metrics["disc/replay_count"].item() == batch_size
    assert output.metrics["disc/expert_count"].item() == batch_size
    assert output.metrics["disc/fake_count"].item() == 2 * batch_size
    assert output.metrics["disc/raw_negative_fraction"].item() == pytest.approx(
        2.0 / 3.0
    )
    assert output.metrics["disc/effective_fake_loss_weight"].item() == 0.5
    assert output.metrics["disc/effective_expert_loss_weight"].item() == 0.5
    assert output.metrics["disc/effective_current_loss_weight"].item() == 0.25
    assert output.metrics["disc/effective_replay_loss_weight"].item() == 0.25


def test_amp_loss_has_two_sided_zero_centered_input_gradient_penalty() -> None:
    torch.manual_seed(8)
    component = _small_discriminator()
    output = component.compute_batch_loss(
        current_observations=torch.randn(5, 3),
        replay_observations=torch.randn(5, 3),
        expert_observations=torch.randn(5, 3),
    )
    assert output.metrics["disc/expert_gradient_penalty"] > 0
    assert output.metrics["disc/fake_gradient_penalty"] > 0
    expected_gp = 0.5 * (
        output.metrics["disc/expert_gradient_penalty"]
        + output.metrics["disc/fake_gradient_penalty"]
    )
    torch.testing.assert_close(output.metrics["disc/gradient_penalty"], expected_gp)
    assert output.metrics["disc/logit_regularization"] > 0
    output.loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in component.discriminator.parameters()
    )


def test_amp_normalizer_uses_old_stats_until_caller_commits() -> None:
    torch.manual_seed(4)
    component = _small_discriminator()
    current = torch.tensor([[2.0, 4.0, 6.0], [4.0, 6.0, 8.0]])
    expert = torch.tensor([[6.0, 8.0, 10.0], [8.0, 10.0, 12.0]])
    replay = torch.randn_like(current)

    before = component.normalize(current).clone()
    begin_metrics = component.begin_normalizer_update(
        current_observations=current,
        expert_observations=expert,
    )
    assert component.normalizer_update_open
    assert begin_metrics["disc_norm/pending_count"] == 4.0
    assert component.normalizer.count.item() == 0
    # Pending samples cannot affect normalization or D input statistics.
    torch.testing.assert_close(component.normalize(current), before)

    train_metrics = component.train_batch(
        current_observations=current,
        replay_observations=replay,
        expert_observations=expert,
    )
    assert train_metrics["disc_norm/count_during_update"] == 0.0
    assert train_metrics["disc_norm/pending_count_during_update"] == 4.0
    assert train_metrics["disc_norm/commit_inside_train_batch"] == 0.0
    assert component.normalizer.count.item() == 0

    assert component.commit_normalizer_update()
    assert not component.normalizer_update_open
    assert component.normalizer.count.item() == 4
    torch.testing.assert_close(
        component.normalizer.mean,
        torch.tensor([5.0, 7.0, 9.0]),
    )
    assert not torch.allclose(component.normalize(current), before)


def test_amp_normalizer_abort_discards_pending_samples() -> None:
    component = _small_discriminator()
    samples = torch.ones(2, 3)
    component.begin_normalizer_update(
        current_observations=samples,
        expert_observations=2.0 * samples,
    )
    component.abort_normalizer_update()
    assert component.normalizer.count.item() == 0
    assert component.normalizer.pending_count.item() == 0
    assert not bool(component.normalizer.frozen.item())
    assert not component.normalizer_update_open


def test_amp_reward_matches_formula_and_does_not_multiply_by_dt() -> None:
    component = _small_discriminator(reward_scale=2.0, reward_epsilon=1.0e-4)
    with torch.no_grad():
        for parameter in component.discriminator.parameters():
            parameter.zero_()
        component.discriminator.logit.bias.fill_(2.0)

    output = component.evaluate_reward(torch.zeros(5, 3), batch_size=2)
    expected = -2.0 * math.log(1.0 - torch.sigmoid(torch.tensor(2.0)).item())
    torch.testing.assert_close(output.logits, torch.full((5,), 2.0))
    torch.testing.assert_close(output.rewards, torch.full((5,), expected))
    assert output.metrics["amp_reward/mean"] == pytest.approx(expected)
    assert output.metrics["amp_reward/sample_count"] == 5.0
    assert output.metrics["amp_reward/formula_scale"] == 2.0
    assert output.metrics["amp_reward/formula_epsilon"] == 1.0e-4
    assert output.metrics["amp_reward/no_dt_multiplier_contract"] == 1.0


def test_amp_train_batch_reports_all_domains_and_never_clips_gradients() -> None:
    torch.manual_seed(12)
    component = _small_discriminator()
    metrics = component.train_batch(
        current_observations=torch.randn(3, 3),
        replay_observations=torch.randn(3, 3),
        expert_observations=torch.randn(3, 3),
    )
    for domain in ("current", "replay", "expert"):
        assert f"disc/{domain}_logit_mean" in metrics
        assert f"disc/{domain}_prob_mean" in metrics
        assert f"disc/{domain}_prob_p50" in metrics
        assert f"disc/{domain}_accuracy" in metrics
    assert metrics["disc/current_count"] == 3.0
    assert metrics["disc/replay_count"] == 3.0
    assert metrics["disc/expert_count"] == 3.0
    assert metrics["disc/optimizer_momentum"] == pytest.approx(0.9)
    assert metrics["disc/optimizer_weight_decay"] == pytest.approx(1.0e-4)
    assert metrics["disc/optimizer_steps"] == 1.0
    assert metrics["disc/gradient_accumulation_single_step_contract"] == 1.0
    assert math.isfinite(metrics["disc/grad_norm"])


def test_amp_microbatching_is_one_equivalent_logical_optimizer_step() -> None:
    torch.manual_seed(23)
    full = _small_discriminator()
    accumulated = _small_discriminator()
    accumulated.load_state_dict(full.state_dict())
    current = torch.randn(5, 3)
    replay = torch.randn(5, 3)
    expert = torch.randn(5, 3)

    full_metrics = full.train_batch(
        current_observations=current,
        replay_observations=replay,
        expert_observations=expert,
    )
    accumulated_metrics = accumulated.train_batch(
        current_observations=current,
        replay_observations=replay,
        expert_observations=expert,
        micro_batch_size=2,
    )

    assert full_metrics["disc/optimizer_steps"] == 1.0
    assert accumulated_metrics["disc/optimizer_steps"] == 1.0
    assert accumulated_metrics["disc/logical_batch_size"] == 5.0
    assert accumulated_metrics["disc/micro_batch_size"] == 2.0
    assert accumulated_metrics["disc/micro_batch_count"] == 3.0
    for full_parameter, accumulated_parameter in zip(
        full.discriminator.parameters(),
        accumulated.discriminator.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            full_parameter,
            accumulated_parameter,
            rtol=2.0e-5,
            atol=2.0e-6,
        )


def test_amp_discriminator_rejects_unequal_domain_batches() -> None:
    component = _small_discriminator()
    with pytest.raises(ValueError, match="exactly B current"):
        component.compute_batch_loss(
            current_observations=torch.zeros(3, 3),
            replay_observations=torch.zeros(4, 3),
            expert_observations=torch.zeros(3, 3),
        )
