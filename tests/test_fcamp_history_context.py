from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from components.imitation.style_reward import discriminator_style_reward
from components.imitation.window_pipeline import TemporalWindowPipeline
from components.normalization.running_stats import RunningNormalizer
from components.rollout.flow_cps_base import FlowCPSBase
from components.rollout.training_streams import (
    CURRICULUM_STREAM,
    PHASE0_STREAM,
)
from method.fcamp import FCAMP, FCAMP_CHECKPOINT_CONTRACT
from models.mlp_actor_critic import EmpiricalNormalization
from models.style_discriminator import StyleDiscriminator


class _RecordingSGD(torch.optim.SGD):
    def __init__(self, params, *, lr: float):
        super().__init__(params, lr=lr)
        self.step_lrs: list[float] = []

    def step(self, closure=None):
        self.step_lrs.append(float(self.param_groups[0]["lr"]))
        return super().step(closure)


def test_fcamp_actor_adapts_lr_from_masked_joint_path_kl_before_step() -> None:
    algo = object.__new__(FCAMP)
    algo.env = SimpleNamespace(device=torch.device("cpu"))
    algo.horizon_h = 2
    algo.chunk_dim = 2
    algo.actor_obs_dim = 1
    algo.cfg = SimpleNamespace(
        flow_steps=2,
        clip_range=0.2,
        credit=SimpleNamespace(ratio_mode="joint_path"),
        max_grad_norm=1.0,
        policy_epochs=1,
        desired_kl=0.01,
        kl_early_stop_factor=100.0,
        num_mini_batches=1,
        micro_batch_size=0,
    )
    algo._policy = torch.nn.Linear(1, 1, bias=False)
    algo.learning_rate = 3.0e-4
    algo.min_lr = 1.0e-5
    algo.max_lr = 1.0e-3
    algo.actor_optimizer = _RecordingSGD(
        algo._policy.parameters(), lr=algo.learning_rate
    )

    delta = torch.tensor(
        [
            [[0.20, 0.20], [0.20, 0.20]],
            [[0.10, 0.30], [0.10, 0.30]],
        ]
    )
    actor_obs = torch.arange(2, dtype=torch.float32).unsqueeze(-1)

    def recompute(obs, _latent_path):
        indices = obs[:, 0].long()
        return delta.index_select(0, indices) + 0.0 * algo._policy.weight.sum()

    algo._recompute_cps_path_stats = recompute
    observed: list[float] = []
    inherited_controller = FlowCPSBase._update_adaptive_learning_rates.__get__(
        algo, FCAMP
    )

    def update_lr(kl: float) -> None:
        observed.append(float(kl))
        inherited_controller(kl)

    algo._update_adaptive_learning_rates = update_lr
    valid = torch.tensor([[[True, True], [True, False]]])
    rollout = {
        "actions": torch.zeros(1, 2, 2, 1),
        "actor_obs": actor_obs.view(1, 2, 1),
        "latents": torch.zeros(1, 2, 3, 2),
        "old_log_probs": torch.zeros(1, 2, 2, 2),
        "advantages": torch.ones(1, 2, 2),
        "valid": valid,
    }

    metrics = algo._actor_update(rollout)
    mask = valid.reshape(2, 2).float()
    joint_path_delta = delta.sum(dim=1)
    expected_kl = float(
        (0.5 * joint_path_delta.square() * mask).sum().item() / mask.sum().item()
    )

    assert observed == [pytest.approx(expected_kl)]
    assert algo.actor_optimizer.step_lrs == [pytest.approx(2.0e-4)]
    assert metrics["fcamp/actor_lr_start"] == pytest.approx(3.0e-4)
    assert metrics["fcamp/actor_lr"] == pytest.approx(2.0e-4)
    assert metrics["fcamp/lr_decrease_steps"] == 1.0


def test_fcamp_owns_the_production_updater() -> None:
    assert FCAMP.update is not FlowCPSBase.update
    assert not inspect.isabstract(FCAMP)


def test_prefix_context_contains_only_policy_and_value_state() -> None:
    algo = object.__new__(FCAMP)
    algo.critic_obs_dim = 4
    algo.actor_obs_dim = 3
    algo.num_act = 2
    algo.horizon_h = 4
    algo.chunk_dim = 8
    algo.prefix_context_dim = 2 * 4 + 3 + 2 + 8 + 2 * 4

    batch = 2
    current = torch.randn(batch, 4)
    chunk_start_critic = torch.randn(batch, 4)
    chunk_start_actor = torch.randn(batch, 3)
    previous_action = torch.randn(batch, 2)
    latent = torch.randn(batch, 8)
    context = algo._prefix_context_raw(
        current,
        chunk_start_critic,
        chunk_start_actor,
        previous_action,
        latent,
        2,
    )

    expected_prefix = torch.zeros(batch, 4, 2)
    expected_prefix[:, :2] = latent.reshape(batch, 4, 2)[:, :2]
    expected_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]]).expand(batch, -1)
    expected_offset = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).expand(batch, -1)
    torch.testing.assert_close(
        context,
        torch.cat(
            (
                current,
                chunk_start_critic,
                chunk_start_actor,
                previous_action,
                expected_prefix.reshape(batch, -1),
                expected_mask,
                expected_offset,
            ),
            dim=-1,
        ),
    )
    assert context.shape[-1] == algo.prefix_context_dim


def test_amp_reward_uses_full_independent_discriminator() -> None:
    torch.manual_seed(5)
    algo = object.__new__(FCAMP)
    algo.cfg = SimpleNamespace(
        amp=SimpleNamespace(
            reward_eval_batch_size=1,
            reward_scale=2.0,
            reward_epsilon=1.0e-4,
        )
    )
    algo.imitation_history_steps = 4
    algo.imitation_frame_dim = 3
    algo.imitation_window_dim = 12
    algo.imitation_pipeline = TemporalWindowPipeline(4, 3)
    algo.discriminator = StyleDiscriminator(12, hidden_dims=(8, 5))
    algo.disc_normalizer = RunningNormalizer(12, device="cpu")
    flat = torch.randn(3, 12)
    logits, rewards = algo._evaluate_amp_reward(flat)
    expected_logits = algo.discriminator(
        algo.disc_normalizer.normalize(flat)
    )
    expected_rewards = discriminator_style_reward(
        expected_logits,
        scale=2.0,
        minimum_one_minus_prob=1.0e-4,
    )
    torch.testing.assert_close(logits, expected_logits)
    torch.testing.assert_close(rewards, expected_rewards)


def test_discriminator_update_cannot_change_deterministic_actor_action() -> None:
    torch.manual_seed(8)
    algo = object.__new__(FCAMP)
    algo.base_actor_obs_dim = 3
    algo.actor_obs_dim = 3
    algo.num_act = 1
    algo.empirical_normalization = False
    algo.discriminator = StyleDiscriminator(6, hidden_dims=(4,))
    algo.disc_normalizer = RunningNormalizer(6, device="cpu")
    algo._flow_mean_actions = lambda actor_obs, prev_action: (
        actor_obs[:, :2].reshape(-1, 2, 1) + prev_action[:, None, :]
    )
    obs = torch.tensor([[1.0, 2.0, 0.5], [3.0, 4.0, -0.25]])

    before = algo.deterministic_actions(obs)
    with torch.no_grad():
        for parameter in algo.discriminator.parameters():
            parameter.add_(torch.randn_like(parameter))
        algo.disc_normalizer.mean.add_(10.0)
        algo.disc_normalizer.variance.mul_(3.0)
    after = algo.deterministic_actions(obs)

    torch.testing.assert_close(after, before)


def test_fcamp_checkpoint_contract_is_strict_before_load() -> None:
    algo = object.__new__(FCAMP)
    algo.action_low = torch.tensor([-5.0, -5.0])
    algo.action_high = torch.tensor([5.0, 5.0])

    valid_state = {
        **FCAMP_CHECKPOINT_CONTRACT,
        "discriminator_policy_conditioning": False,
        "action_low": algo.action_low.clone(),
        "action_high": algo.action_high.clone(),
    }
    algo.validate_checkpoint_payload({"algo_state": valid_state})

    for name in FCAMP_CHECKPOINT_CONTRACT:
        missing = dict(valid_state)
        missing.pop(name)
        with pytest.raises(ValueError, match=name):
            algo.validate_checkpoint_payload({"algo_state": missing})

    for name, expected in FCAMP_CHECKPOINT_CONTRACT.items():
        mismatched = dict(valid_state)
        mismatched[name] = (
            expected + 1 if isinstance(expected, int) else f"{expected}_mismatch"
        )
        with pytest.raises(ValueError, match=name):
            algo.validate_checkpoint_payload({"algo_state": mismatched})

    for historical_schema in (8, 9, 11, 13, 14):
        historical = dict(valid_state)
        historical["fcamp_schema_version"] = historical_schema
        with pytest.raises(ValueError, match="fcamp_schema_version"):
            algo.validate_checkpoint_payload({"algo_state": historical})

    discriminator_conditioned = dict(valid_state)
    discriminator_conditioned["discriminator_policy_conditioning"] = True
    with pytest.raises(ValueError, match="discriminator-conditioned"):
        algo.validate_checkpoint_payload(
            {"algo_state": discriminator_conditioned}
        )

    wrong_action_domain = dict(valid_state)
    wrong_action_domain["action_high"] = algo.action_high + 0.01
    with pytest.raises(ValueError, match="action_high differs"):
        algo.validate_checkpoint_payload(
            {"algo_state": wrong_action_domain}
        )


def test_fcamp_validates_contract_before_restoring_base_state(monkeypatch) -> None:
    algo = object.__new__(FCAMP)
    algo.action_low = torch.tensor([-5.0])
    algo.action_high = torch.tensor([5.0])
    base_restore_calls: list[dict] = []

    def record_base_restore(self, payload, reset_optimizer=False):
        del self, reset_optimizer
        base_restore_calls.append(payload)

    monkeypatch.setattr(
        FlowCPSBase,
        "load_extra_checkpoint_state",
        record_base_restore,
    )

    with pytest.raises(ValueError, match="fcamp_schema_version"):
        algo.load_extra_checkpoint_state(
            {"fcamp_schema_version": 8},
            reset_optimizer=False,
        )
    assert base_restore_calls == []


def test_rollout_snapshot_optimizes_actor_and_critic_before_discriminator() -> None:
    algo = object.__new__(FCAMP)
    algo.disc_version = 7
    algo.disc_normalizer = SimpleNamespace(count=torch.tensor(12.0))
    algo.empirical_normalization = False
    events: list[str] = []

    def actor_update(_rollout):
        events.append("actor")
        assert algo.disc_version == 7
        return {"fcamp/actor_optimizer_steps": 1.0}

    def critic_update(_rollout):
        events.append("critic")
        assert algo.disc_version == 7
        return {"critic/optimizer_steps": 1.0}

    def disc_update(_update_idx, *, rollout):
        del rollout
        events.append("disc")
        assert algo.disc_version == 7
        algo.disc_version = 8
        return {
            "disc/input_version": 7.0,
            "disc/version": 8.0,
            "disc/update_steps": 1.0,
        }

    algo._actor_update = actor_update
    algo._critic_update = critic_update
    algo._discriminator_update = disc_update
    rollout = {
        "disc_version_used": 7,
        "disc_normalizer_count_used": 12.0,
        "amp_valid": torch.tensor([[[False, True]]]),
        "imitation_window_age": torch.tensor([[[-1, 16]]]),
    }

    result = algo._optimize_rollout_snapshot(rollout, update_idx=3)
    assert events == ["actor", "critic", "disc"]
    assert result[3]["amp/reward_disc_version"] == 7.0
    assert result[3]["amp/reward_recomputed_after_disc"] == 0.0
    assert result[3]["policy/discriminator_conditioned"] == 0.0


def test_no_current_disc_window_does_not_skip_actor_or_critic() -> None:
    algo = object.__new__(FCAMP)
    algo.disc_version = 2
    algo.disc_normalizer = SimpleNamespace(count=torch.tensor(0.0))
    algo.empirical_normalization = False
    events: list[str] = []
    algo._actor_update = lambda _rollout: events.append("actor") or {}
    algo._critic_update = lambda _rollout: events.append("critic") or {}
    algo._discriminator_update = (
        lambda _update_idx, *, rollout: events.append("disc")
        or {
            "disc/input_version": 2.0,
            "disc/version": 2.0,
            "disc/update_steps": 0.0,
        }
    )
    rollout = {
        "disc_version_used": 2,
        "disc_normalizer_count_used": 0.0,
        "amp_valid": torch.zeros(1, 1, 2, dtype=torch.bool),
        "imitation_window_age": torch.full((1, 1, 2), -1, dtype=torch.long),
    }

    algo._optimize_rollout_snapshot(rollout, update_idx=1)
    assert events == ["actor", "critic", "disc"]


def test_discriminator_trains_on_old_normalizer_then_commits_next_snapshot() -> None:
    class TwoStreamReplay:
        def sample(
            self,
            batch_size,
            *,
            stream_id,
        ):
            return (
                    torch.full(
                        (batch_size, 1, 3),
                    4.0 + float(stream_id),
                ),
                torch.full((batch_size,), stream_id + 1, dtype=torch.long),
            )

        def statistics(self, current_update):
            del current_update
            return {}

    algo = object.__new__(FCAMP)
    algo.env = SimpleNamespace(device=torch.device("cpu"))
    algo.cfg = SimpleNamespace(
        amp=SimpleNamespace(
            batch_size=2,
            epochs=1,
            max_updates_per_iteration=1,
            learning_rate=1.0e-3,
            grad_penalty=0.0,
            logit_reg=0.0,
        ),
        streams=SimpleNamespace(phase0_fraction=0.5),
    )
    algo.imitation_history_steps = 1
    algo.imitation_frame_dim = 3
    algo.imitation_window_dim = 3
    algo.imitation_pipeline = TemporalWindowPipeline(1, 3)
    algo.discriminator = StyleDiscriminator(3, hidden_dims=(4,))
    algo.disc_optimizer = torch.optim.SGD(algo.discriminator.parameters(), lr=1.0e-3)
    algo.disc_window_replay = TwoStreamReplay()
    algo.disc_version = 4
    algo.disc_normalizer = RunningNormalizer(3, device="cpu", clip=100.0)
    algo.disc_normalizer.count.fill_(1.0)
    algo.disc_normalizer.mean.fill_(1.0)
    algo.disc_normalizer.variance.fill_(1.0)
    algo._expert_flat_at_end_times = lambda end_times: torch.full(
        (end_times.numel(), 3), 9.0
    )

    seen_inputs: list[torch.Tensor] = []
    handle = algo.discriminator.register_forward_pre_hook(
        lambda _module, args: seen_inputs.append(args[0].detach().clone())
    )
    metrics = algo._discriminator_update(
        1,
        rollout={
            "current_disc_windows": torch.full((2, 3), 3.0),
            "current_disc_end_times": torch.tensor([1, 2], dtype=torch.long),
            "current_disc_stream_ids": torch.tensor(
                [PHASE0_STREAM, CURRICULUM_STREAM],
                dtype=torch.int8,
            ),
        },
    )
    handle.remove()

    # Expert is the first forward in the BCE objective. Under the committed
    # old mean/std it is (9-1)/1=8; an early pending-stat commit would differ.
    torch.testing.assert_close(seen_inputs[0], torch.full((2, 3), 8.0))
    assert metrics["disc/input_version"] == 4.0
    assert metrics["disc/version"] == 5.0
    assert metrics["disc_norm/committed_this_update"] == 1.0
    assert float(algo.disc_normalizer.count.item()) == 5.0
    assert bool(algo.disc_normalizer.frozen.item())


def test_chunked_context_normalizer_matches_one_shot_valid_moments() -> None:
    torch.manual_seed(19)
    algo = object.__new__(FCAMP)
    algo.cfg = SimpleNamespace(micro_batch_size=3)
    chunked = EmpiricalNormalization(4, "cpu")
    expected = EmpiricalNormalization(4, "cpu")
    samples = torch.randn(2, 5, 3, 4)
    valid = torch.rand(2, 5, 3) > 0.3

    algo._update_empirical_normalizer_chunked(chunked, samples, valid)
    expected._update(samples.reshape(-1, 4)[valid.reshape(-1)])

    assert int(chunked.count.item()) == int(valid.sum().item())
    torch.testing.assert_close(chunked._mean, expected._mean, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(chunked._var, expected._var, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(chunked._std, expected._std, rtol=1.0e-5, atol=1.0e-6)
