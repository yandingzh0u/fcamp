from __future__ import annotations

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from method.fixed_reward import (
    HOLOSOMA_PPO_MANIFEST,
    HOLOSOMA_STATIC_PARITY_SHA256,
    HOLOSOMA_UPSTREAM_COMMIT,
    REWARD_TERM_WEIGHTS,
    FixedRewardPPO,
)
from models.holosoma_ppo import (
    EmpiricalNormalization,
    PPOActor,
    PPOCritic,
    RolloutStorage,
)


ROOT = Path(__file__).resolve().parents[1]
HOLOSOMA_PACKAGE_ROOT = (
    ROOT.parent / "holosoma" / "src" / "holosoma"
)


def _cfg(**overrides):
    values = {
        "actor_hidden_dims": (512, 256, 128),
        "critic_hidden_dims": (512, 256, 128),
        "activation": "ELU",
        "init_noise_std": 1.0,
        "actor_learning_rate": 1.0e-3,
        "critic_learning_rate": 1.0e-3,
        "actor_weight_decay": 0.0,
        "critic_weight_decay": 0.0,
        "desired_kl": 0.01,
        "schedule": "adaptive",
        "clip_param": 0.2,
        "entropy_coef": 0.005,
        "value_loss_coef": 1.0,
        "max_grad_norm": 1.0,
        "gamma": 0.99,
        "lam": 0.95,
        "num_mini_batches": 4,
        "num_learning_epochs": 5,
        "num_steps_per_env": 24,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _algo_for_loss(obs_dim: int = 7, critic_dim: int = 9, action_dim: int = 3):
    algo = object.__new__(FixedRewardPPO)
    algo.cfg = _cfg()
    algo.env = SimpleNamespace(device=torch.device("cpu"))
    algo.actor = PPOActor(obs_dim, (16, 8), "ELU", action_dim, 1.0)
    algo.critic = PPOCritic(critic_dim, (16, 8), "ELU")
    algo.actor_learning_rate = 1.0e-3
    algo.critic_learning_rate = 1.0e-3
    algo.min_actor_learning_rate = 1.0e-5
    algo.max_actor_learning_rate = 1.0e-2
    algo.min_critic_learning_rate = 1.0e-5
    algo.max_critic_learning_rate = 1.0e-2
    algo.actor_optimizer = torch.optim.AdamW(
        algo.actor.parameters(), lr=1.0e-3, weight_decay=0.0
    )
    algo.critic_optimizer = torch.optim.AdamW(
        algo.critic.parameters(), lr=1.0e-3, weight_decay=0.0
    )
    return algo


def test_holosoma_g1_manifest_is_exact() -> None:
    assert HOLOSOMA_UPSTREAM_COMMIT == "c5c836c68f423ac4565f57801ff4ff47ea56e5ac"
    assert HOLOSOMA_PPO_MANIFEST == {
        "activation": "ELU",
        "actor_hidden_dims": [512, 256, 128],
        "actor_learning_rate": 0.001,
        "actor_weight_decay": 0.0,
        "action_clip_value": 100.0,
        "critic_hidden_dims": [512, 256, 128],
        "critic_learning_rate": 0.001,
        "critic_weight_decay": 0.0,
        "desired_kl": 0.01,
        "empirical_normalization": True,
        "entropy_coef": 0.005,
        "gamma": 0.99,
        "init_noise_std": 1.0,
        "lam": 0.95,
        "max_grad_norm": 1.0,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "num_steps_per_env": 24,
        "schedule": "adaptive",
        "use_symmetry": False,
        "value_loss_coef": 1.0,
        "clip_param": 0.2,
    }
    assert HOLOSOMA_STATIC_PARITY_SHA256 == (
        "9dddf13a81c0f094c8a6a4cbdb5cc0bef77c6a1372b10de2292a2fa70804e701"
    )


def test_actor_and_critic_are_default_initialized_elu_mlps() -> None:
    torch.manual_seed(42)
    actor = PPOActor(11, (512, 256, 128), "ELU", 29, 1.0)
    assert [type(module) for module in actor.actor_module] == [
        nn.Linear,
        nn.ELU,
        nn.Linear,
        nn.ELU,
        nn.Linear,
        nn.ELU,
        nn.Linear,
    ]
    assert torch.equal(actor.std, torch.ones(29))
    assert actor.std.requires_grad
    critic = PPOCritic(13, (512, 256, 128), "ELU")
    assert [type(module) for module in critic.critic_module] == [
        nn.Linear,
        nn.ELU,
        nn.Linear,
        nn.ELU,
        nn.Linear,
        nn.ELU,
        nn.Linear,
    ]


def test_actor_is_unsquashed_normal_and_uses_sample() -> None:
    actor = PPOActor(2, (), "ELU", 1, 1.0)
    with torch.no_grad():
        actor.actor_module[0].weight.zero_()
        actor.actor_module[0].bias.fill_(7.0)
    obs = torch.zeros(4096, 2)
    actions = actor.act(obs)
    assert actor.action_mean.unique().item() == 7.0
    assert bool((actions > 5.0).any())
    assert actor.get_actions_log_prob(actions).shape == (4096,)
    assert actor.entropy.shape == (4096,)
    source = inspect.getsource(PPOActor.act)
    assert ".sample()" in source
    assert "rsample" not in source


def test_empirical_normalization_matches_upstream_update_order() -> None:
    normalizer = EmpiricalNormalization(2, "cpu")
    first = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    output = normalizer(first)
    expected_mean = torch.tensor([[2.0, 4.0]])
    # Upstream uses delta2 after updating the mean. At count zero this leaves
    # only the population batch variance.
    expected_var = torch.tensor([[1.0, 4.0]])
    assert torch.equal(normalizer._mean, expected_mean)
    assert torch.equal(normalizer._var, expected_var)
    assert torch.allclose(output, (first - expected_mean) / (expected_var.sqrt() + 0.01))
    count = normalizer.count.clone()
    _ = normalizer(first, update=False)
    assert torch.equal(normalizer.count, count)


def test_rollout_storage_reuses_one_permutation_across_epochs() -> None:
    storage = RolloutStorage(8, 3, "cpu")
    storage.register("sample_id", (1,), torch.long)
    next_id = 0
    for _ in range(3):
        ids = torch.arange(next_id, next_id + 8).view(8, 1)
        storage.add(sample_id=ids)
        next_id += 8
    batches = list(storage.mini_batch_generator(4, 5))
    assert len(batches) == 20
    first_epoch = [batch["sample_id"] for batch in batches[:4]]
    for epoch in range(1, 5):
        for index in range(4):
            assert torch.equal(
                first_epoch[index], batches[epoch * 4 + index]["sample_id"]
            )


def test_gae_matches_holosoma_terminal_and_global_normalization() -> None:
    algo = object.__new__(FixedRewardPPO)
    algo.cfg = _cfg(gamma=0.9, lam=0.8)
    values = torch.tensor([[[0.2]], [[0.3]], [[0.4]]])
    rewards = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    dones = torch.tensor([[[False]], [[True]], [[False]]])
    last_values = torch.tensor([[0.5]])
    returns, advantages = algo._compute_returns_and_advantages(
        last_values, values, dones, rewards
    )
    expected_returns = torch.empty_like(values)
    running = 0
    for step in reversed(range(3)):
        next_value = last_values if step == 2 else values[step + 1]
        alive = 1.0 - dones[step].float()
        delta = rewards[step] + alive * 0.9 * next_value - values[step]
        running = delta + alive * 0.9 * 0.8 * running
        expected_returns[step] = running + values[step]
    raw_advantage = expected_returns - values
    expected_advantage = (raw_advantage - raw_advantage.mean()) / (
        raw_advantage.std() + 1.0e-8
    )
    assert torch.equal(returns, expected_returns)
    assert torch.equal(advantages, expected_advantage)
    assert abs(float(advantages.mean())) < 1.0e-6
    assert float(advantages.std()) == pytest.approx(1.0, abs=1.0e-6)


def test_adaptive_kl_changes_both_learning_rates_at_exact_thresholds() -> None:
    algo = _algo_for_loss()
    algo._update_learning_rate(torch.tensor(0.021))
    assert algo.actor_learning_rate == pytest.approx(1.0e-3 / 1.5)
    assert algo.critic_learning_rate == pytest.approx(1.0e-3 / 1.5)
    algo._update_learning_rate(torch.tensor(0.004))
    assert algo.actor_learning_rate == pytest.approx(1.0e-3)
    assert algo.critic_learning_rate == pytest.approx(1.0e-3)
    unchanged = algo.actor_learning_rate
    algo._update_learning_rate(torch.tensor(0.0))
    assert algo.actor_learning_rate == unchanged


def test_ppo_loss_matches_direct_reference_formula_and_backpropagates() -> None:
    torch.manual_seed(7)
    algo = _algo_for_loss()
    batch_size = 32
    actor_obs = torch.randn(batch_size, 7)
    critic_obs = torch.randn(batch_size, 9)
    with torch.no_grad():
        old_actions = algo.actor.act(actor_obs)
        old_log_prob = algo.actor.get_actions_log_prob(old_actions).unsqueeze(1)
        old_mu = algo.actor.action_mean.clone()
        old_sigma = algo.actor.action_std.clone()
        old_values = algo.critic.evaluate(critic_obs)
    returns = old_values + torch.randn_like(old_values)
    advantages = torch.randn(batch_size, 1)
    minibatch = {
        "actions": old_actions,
        "values": old_values,
        "advantages": advantages,
        "returns": returns,
        "actions_log_prob": old_log_prob,
        "action_mean": old_mu,
        "action_sigma": old_sigma,
        "actor_obs": actor_obs,
        "critic_obs": critic_obs,
    }
    result = algo._compute_ppo_loss(minibatch)
    ratio = torch.ones(batch_size)
    expected_surrogate = torch.max(
        -advantages.squeeze(1) * ratio,
        -advantages.squeeze(1) * torch.clamp(ratio, 0.8, 1.2),
    ).mean()
    expected_entropy = Normal(old_mu, old_sigma).entropy().sum(-1).mean()
    expected_value = (old_values - returns).square().mean()
    assert result["kl_mean"].item() == 0.0
    assert torch.equal(result["surrogate_loss"], expected_surrogate)
    assert torch.allclose(result["entropy_loss"], expected_entropy)
    assert torch.equal(result["value_loss"], expected_value)
    assert torch.allclose(
        result["actor_loss"], expected_surrogate - 0.005 * expected_entropy
    )
    assert torch.equal(result["critic_loss"], expected_value)
    (result["actor_loss"] + result["critic_loss"]).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in algo.actor.parameters()
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in algo.critic.parameters()
    )


def test_analytic_kl_is_old_to_new_normal() -> None:
    old_mu = torch.tensor([[0.0, 1.0]])
    old_sigma = torch.tensor([[1.0, 2.0]])
    new_mu = torch.tensor([[0.3, 0.8]])
    new_sigma = torch.tensor([[0.7, 1.4]])
    expected = kl_divergence(
        Normal(old_mu, old_sigma), Normal(new_mu, new_sigma)
    ).sum(-1).mean()
    actual = FixedRewardPPO._compute_kl_div(
        old_mu, old_sigma, new_mu, new_sigma
    )
    assert torch.equal(actual, expected)


def test_reward_contract_is_pose_only_and_dt_is_applied_once() -> None:
    assert REWARD_TERM_WEIGHTS == {
        "anchor_pos_reward": 0.5,
        "anchor_ori_reward": 0.5,
        "body_pos_reward": 2.0,
        "body_ori_reward": 2.0,
        "action_rate": -0.1,
        "joint_limit": -10.0,
        "undesired_contacts": -0.1,
    }
    source = (ROOT / "method" / "fixed_reward.py").read_text().lower()
    assert "body_lin_vel" not in source
    assert "body_ang_vel" not in source


def test_active_algorithm_has_no_flow_cps_or_state_covariance() -> None:
    active = "\n".join(
        path.read_text().lower()
        for path in (
            ROOT / "method" / "fixed_reward.py",
            ROOT / "models" / "holosoma_ppo.py",
            ROOT / "configs" / "fixed_reward_largebox.yaml",
        )
    )
    for forbidden in (
        "stateconditionedflowpolicy",
        "flow_steps",
        "low_rank",
        "woodbury",
        "latent_path",
        "cumulative",
        "gru",
        "cps_noise",
    ):
        assert forbidden not in active
    assert not (ROOT / "models" / "state_conditioned_flow_policy.py").exists()


def test_first_three_log_contract_keeps_exact_source_literals() -> None:
    line = FixedRewardPPO._parity_line()
    expected_fragments = (
        "init_noise_std=1.0",
        "num_steps_per_env=24",
        "num_learning_epochs=5",
        "num_mini_batches=4",
        "clip_param=0.2",
        "gamma=0.99",
        "lam=0.95",
        "value_loss_coef=1.0",
        "entropy_coef=0.005",
        "actor_learning_rate=0.001",
        "critic_learning_rate=0.001",
        "max_grad_norm=1.0",
        "desired_kl=0.01",
    )
    for fragment in expected_fragments:
        assert fragment in line


@pytest.mark.skipif(
    not HOLOSOMA_PACKAGE_ROOT.is_dir(),
    reason="sibling HOLOSOMA checkout is unavailable",
)
def test_actor_forward_sampling_and_logprob_are_bitwise_upstream_equal() -> None:
    sys.path.insert(0, str(HOLOSOMA_PACKAGE_ROOT))
    try:
        from holosoma.agents.modules.ppo_modules import PPOActor as UpstreamActor
        from holosoma.config_types.algo import LayerConfig, ModuleConfig
    finally:
        sys.path.pop(0)

    module_config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=["robot_action_dim"],
        layer_config=LayerConfig(
            hidden_dims=[512, 256, 128],
            activation="ELU",
        ),
    )
    torch.manual_seed(1234)
    upstream = UpstreamActor(
        {"actor_obs": 7},
        module_config,
        3,
        1.0,
        {"actor_obs": 1, "critic_obs": 1},
    )
    torch.manual_seed(1234)
    local = PPOActor(7, (512, 256, 128), "ELU", 3, 1.0)
    upstream_tensors = list(upstream.state_dict().values())
    local_tensors = list(local.state_dict().values())
    assert len(upstream_tensors) == len(local_tensors)
    assert all(
        torch.equal(left, right)
        for left, right in zip(upstream_tensors, local_tensors, strict=True)
    )

    observation = torch.randn(19, 7)
    torch.manual_seed(9876)
    upstream_action = upstream.act({"actor_obs": observation})
    torch.manual_seed(9876)
    local_action = local.act(observation)
    assert torch.equal(local_action, upstream_action)
    assert torch.equal(local.action_mean, upstream.action_mean)
    assert torch.equal(local.action_std, upstream.action_std)
    assert torch.equal(
        local.get_actions_log_prob(local_action),
        upstream.get_actions_log_prob(upstream_action),
    )
    assert torch.equal(local.entropy, upstream.entropy)


@pytest.mark.skipif(
    not HOLOSOMA_PACKAGE_ROOT.is_dir(),
    reason="sibling HOLOSOMA checkout is unavailable",
)
def test_normalizer_and_gae_are_bitwise_upstream_equal() -> None:
    sys.path.insert(0, str(HOLOSOMA_PACKAGE_ROOT))
    try:
        from holosoma.agents.ppo.ppo import (
            EmpiricalNormalization as UpstreamNormalizer,
            PPO as UpstreamPPO,
        )
    finally:
        sys.path.pop(0)

    local_normalizer = EmpiricalNormalization(5, "cpu")
    upstream_normalizer = UpstreamNormalizer(5, "cpu")
    torch.manual_seed(222)
    for _ in range(4):
        batch = torch.randn(13, 5)
        assert torch.equal(local_normalizer(batch), upstream_normalizer(batch))
        for name in ("_mean", "_var", "_std", "count"):
            assert torch.equal(
                getattr(local_normalizer, name),
                getattr(upstream_normalizer, name),
            )

    torch.manual_seed(333)
    values = torch.randn(24, 8, 1)
    rewards = torch.randn(24, 8, 1)
    dones = torch.rand(24, 8, 1) < 0.17
    last_values = torch.randn(8, 1)
    local_algo = object.__new__(FixedRewardPPO)
    local_algo.cfg = _cfg()
    upstream_algo = object.__new__(UpstreamPPO)
    upstream_algo.config = SimpleNamespace(gamma=0.99, lam=0.95)
    upstream_algo.is_multi_gpu = False
    local_returns, local_advantages = local_algo._compute_returns_and_advantages(
        last_values, values, dones, rewards
    )
    upstream_returns, upstream_advantages = (
        upstream_algo._compute_returns_and_advantages(
            last_values, values, dones, rewards
        )
    )
    assert torch.equal(local_returns, upstream_returns)
    assert torch.equal(local_advantages, upstream_advantages)
