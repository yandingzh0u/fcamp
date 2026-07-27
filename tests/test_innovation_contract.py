from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from components.normalization.running_stats import EmpiricalNormalization
from components.credit.temporal_credit import compute_dual_channel_gae
from components.rollout.flow_cps_base import FlowCPSBase
from engine.checkpoint import (
    FCAMP_CHECKPOINT_CONTRACT,
    validate_static_fcamp_checkpoint_contract,
)
from engine.config import _construct_fcamp_config, _validate_fcamp
from engine.validation_metrics import ChunkBoundaryDiagnostics
from method.fcamp import FCAMP
from models.flow_cps_policy import FlowMatchingPolicy


A, H, O = 29, 4, 7
BOUND, STEP, RMS = 5.0, 1.0, 0.05
ROOT = Path(__file__).resolve().parents[1]


class _FlowHarness(FlowCPSBase):
    def _unused(self, *args, **kwargs):
        raise NotImplementedError

    collect = deterministic_actions = evaluation_step_payload = _unused
    initial_reset = log = log_banner = update = _unused


def _flow(horizon: int = H) -> _FlowHarness:
    device = torch.device("cpu")
    algo = _FlowHarness(
        SimpleNamespace(flow_steps=4),
        SimpleNamespace(device=device),
        simulation_app=None,
    )
    algo.num_act, algo.horizon_h, algo.actor_obs_dim = A, horizon, O
    algo.chunk_dim = algo._cps_flat_dim = horizon * A
    algo.policy_action_bound, algo.innovation_step_bound = BOUND, STEP
    algo._policy = FlowMatchingPolicy(O, A, horizon, (32, 16), "elu")
    algo._policy.register_buffer("cps_raw_rms", torch.tensor(RMS), persistent=False)
    algo.actor_obs_normalizer = EmpiricalNormalization(O, device)
    return algo


def test_zero_hold_and_one_flow_call_per_ode_step() -> None:
    algo = _flow()
    calls: list[None] = []
    hook = algo._policy.velocity_net.register_forward_hook(
        lambda *_: calls.append(None)
    )
    mean = algo._flow_mean_innovation(torch.randn(9, O))
    hook.remove()
    assert len(calls) == algo.cfg.flow_steps
    assert torch.count_nonzero(mean) == 0
    assert not any(
        hasattr(algo._policy, name)
        for name in ("frame_pos_embed", "token_encoder", "causal_gru")
    )
    initial = torch.empty(9, A).uniform_(-BOUND, BOUND)
    torch.testing.assert_close(
        algo._action_chunk_from_innovations(mean, initial),
        initial[:, None].expand(-1, H, -1),
        rtol=0,
        atol=1e-6,
    )


def test_decoder_is_reblocking_invariant_and_strictly_bounded() -> None:
    torch.manual_seed(1)
    raw = torch.randn(1000, 8, A)
    initial = torch.empty(1000, A).uniform_(-4.9, 4.9)
    whole = _flow(8)._action_chunk_from_innovations(raw.flatten(1), initial)
    first = _flow(4)._action_chunk_from_innovations(raw[:, :4].flatten(1), initial)
    second = _flow(4)._action_chunk_from_innovations(
        raw[:, 4:].flatten(1), first[:, -1]
    )
    singles, state = [], initial
    one = _flow(1)
    for frame in raw.unbind(1):
        state = one._action_chunk_from_innovations(frame, state)[:, 0]
        singles.append(state)
    reblocked = torch.cat((first, second), 1)
    sequential = torch.stack(singles, 1)
    assert float((whole - reblocked).abs().max()) < 1e-6
    assert float((whole - sequential).abs().max()) < 1e-6

    extreme = _flow()._action_chunk_from_innovations(
        torch.full((2, H * A), 1e6),
        torch.full((2, A), -STEP / 2),
    )
    sequence = torch.cat((torch.full((2, 1, A), -STEP / 2), extreme), 1)
    assert sequence.abs().max() <= BOUND
    assert torch.diff(sequence, dim=1).abs().max() <= STEP + 2e-6
    assert torch.diff(sequence, dim=1)[:, 0].abs().max() == pytest.approx(STEP, abs=2e-6)


def test_iid_innovations_have_unbiased_d1_and_d2_seams() -> None:
    torch.manual_seed(2)
    algo, batch = _flow(), 2048
    action = torch.zeros(batch, A)
    previous_delta = torch.zeros_like(action)
    sums = torch.zeros(2, 2, dtype=torch.float64)
    counts = torch.zeros(2, dtype=torch.float64)
    for chunk in range(12):
        plan = algo._action_chunk_from_innovations(
            RMS * torch.randn(batch, H * A), action
        )
        delta = torch.diff(torch.cat((action[:, None], plan), 1), dim=1)
        d2 = torch.cat(((delta[:, 0] - previous_delta)[:, None], torch.diff(delta, dim=1)), 1)
        if chunk:
            for row, values in enumerate((delta, d2)):
                sums[row, 0] += values[:, 0].abs().double().sum()
                sums[row, 1] += values[:, 1:].abs().double().sum()
            counts += torch.tensor([delta[:, 0].numel(), delta[:, 1:].numel()])
        action, previous_delta = plan[:, -1], delta[:, -1]
    ratios = (sums[:, 0] / counts[0]) / (sums[:, 1] / counts[1])
    torch.testing.assert_close(ratios, torch.ones(2, dtype=torch.float64), atol=0.02, rtol=0)


def test_shared_cps_has_equal_offset_covariance() -> None:
    torch.manual_seed(3)
    algo = _flow()
    assert algo._policy.joint_cholesky_raw.numel() == A * (A + 1) // 2
    with torch.no_grad():
        algo._policy.joint_cholesky_raw.copy_(torch.linspace(-0.02, 0.02, 435))
    chol = algo._effective_joint_cholesky(device=torch.device("cpu"), dtype=torch.float32)
    target = chol @ chol.T
    assert float(torch.sqrt(chol.square().sum() / A)) == pytest.approx(RMS, abs=2e-7)
    raw, mean, _ = algo._sample_innovations(torch.zeros(8192, O))
    centered = (raw - mean).view(-1, H, A)
    for frame in centered.unbind(1):
        frame = frame - frame.mean(0)
        covariance = frame.T @ frame / (frame.shape[0] - 1)
        torch.testing.assert_close(covariance, target, atol=1.5e-4, rtol=0.12)


def test_factorized_frame_logp_sum_and_exact_kl() -> None:
    torch.manual_seed(4)
    algo, obs = _flow(), torch.randn(7, O)
    raw, mean, logp = algo._sample_innovations(obs)
    torch.testing.assert_close(
        algo._recompute_innovation_log_prob(obs, raw), logp, atol=1e-6, rtol=0
    )
    chol = algo._effective_joint_cholesky(device=raw.device, dtype=raw.dtype)
    manual = torch.distributions.MultivariateNormal(
        mean.view(-1, H, A), scale_tril=chol
    ).log_prob(raw.view(-1, H, A))
    torch.testing.assert_close(logp.sum(1), manual.sum(1), atol=5e-5, rtol=0)
    zero = algo._expected_innovation_frame_kl(mean, mean, chol, chol)
    torch.testing.assert_close(zero, torch.zeros_like(zero), atol=1e-7, rtol=0)
    shifted = mean.view(-1, H, A).clone()
    shifted[:, 2] += 0.01
    kl = algo._expected_innovation_frame_kl(mean, shifted.flatten(1), chol, chol)
    assert bool((kl[:, 2] > 0).all())
    torch.testing.assert_close(kl[:, (0, 1, 3)], torch.zeros_like(kl[:, (0, 1, 3)]), atol=1e-7, rtol=0)


def test_mid_chunk_done_freezes_inactive_action_history() -> None:
    import sys
    import types

    spec = types.ModuleType("envs.spec")
    spec.VELOCITY_RANGE = ((0.0, 0.0),) * 6
    previous_spec = sys.modules.get("envs.spec")
    sys.modules["envs.spec"] = spec
    try:
        from envs.step import MimicStepMixin
    finally:
        if previous_spec is None:
            del sys.modules["envs.spec"]
        else:
            sys.modules["envs.spec"] = previous_spec

    class Dummy(MimicStepMixin):
        def __init__(self):
            self.num_envs, self.device, self.decimation, self.render = 2, torch.device("cpu"), 0, False
            self.last_action = torch.zeros(2, 2)
            self.last_delta = torch.zeros_like(self.last_action)
            self.episode_steps = torch.zeros(2, dtype=torch.long)
            self.phase_steps = torch.zeros(2)
            self.motion_frame_delta = 1
            self.motion = SimpleNamespace(num_frames=100)
            self._last_interval_push_mask = torch.zeros(2, dtype=torch.bool)

        def _apply_action_targets(self, action): return action
        def compute_reward(self, *_): return torch.zeros(2), {}
        def compute_termination(self):
            done = torch.tensor([True, False])
            terms = {name: torch.zeros(2, dtype=torch.bool) for name in ("time_out", "motion_complete", "anchor_pos_bad", "anchor_ori_bad", "ee_body_bad")}
            terms["anchor_pos_bad"][0] = True
            return done, terms, {}
        def _record_adaptive_failures(self, *_): pass
        def _fold_adaptive_sampler(self): pass
        def _apply_interval_pushes(self, **_): return self._last_interval_push_mask.clone()
        def get_observation(self): return self.last_action.clone()
        def get_imitation_policy_frame(self): return self.last_action.clone()

    env = Dummy()
    _, _, done, _ = env.step(torch.tensor([[1., 1.], [2., 2.]]))
    assert done.tolist() == [True, False]
    frozen = (env.last_action[0].clone(), env.last_delta[0].clone(), env.episode_steps[0].clone())
    env.step(torch.tensor([[-4., -4.], [3., 3.]]), active_mask=~done)
    torch.testing.assert_close(env.last_action[0], frozen[0])
    torch.testing.assert_close(env.last_delta[0], frozen[1])
    assert torch.equal(env.episode_steps[0], frozen[2])


def test_prefix_is_causal_and_invalid_nan_cannot_enter_gae() -> None:
    critic_dim, actor_dim, action_dim = 3, 5, 2
    dummy = SimpleNamespace(
        horizon_h=H,
        num_act=action_dim,
        prefix_context_dim=2 * critic_dim + actor_dim + H * action_dim + 2 * H,
    )
    critic = torch.randn(2, critic_dim)
    actor = torch.randn(2, actor_dim)
    raw = torch.randn(2, H * action_dim)
    context = FCAMP._prefix_context_raw(dummy, critic, critic, actor, raw, 2)
    changed_future = raw.clone().view(2, H, action_dim)
    changed_future[:, 2:] += 100
    torch.testing.assert_close(
        context,
        FCAMP._prefix_context_raw(
            dummy, critic, critic, actor, changed_future.flatten(1), 2
        ),
    )
    changed_past = raw.clone().view(2, H, action_dim)
    changed_past[:, 1] += 1
    assert not torch.equal(
        context,
        FCAMP._prefix_context_raw(
            dummy, critic, critic, actor, changed_past.flatten(1), 2
        ),
    )

    rewards = torch.ones(3, 1, 2)
    values = torch.zeros_like(rewards)
    next_values = torch.zeros_like(rewards)
    rewards[-1] = torch.nan
    values[-1] = torch.nan
    next_values[1:] = torch.nan
    valid = torch.tensor([[True], [True], [False]])
    credit = compute_dual_channel_gae(
        rewards,
        values,
        next_values,
        bootstrap_mask=torch.tensor([[True], [False], [False]]),
        trace_mask=torch.tensor([[True], [False], [False]]),
        valid_mask=valid,
        gamma=0.99,
        gae_lambda=0.95,
        chunk_horizon=H,
        normalization="none",
        actor_weights=(1.0, 1.0),
    )
    assert bool(torch.isfinite(credit.advantages).all())
    assert torch.count_nonzero(credit.value_targets[-1]) == 0


def test_boundary_diagnostics_uses_only_real_post_reset_steps() -> None:
    diagnostics = ChunkBoundaryDiagnostics(horizon=H, initial_action=torch.zeros(1, 2))
    deltas = torch.tensor([[9., 9.], [1., 2.], [2., 0.], [-1., 1.], [3., 4.], [1., -2.]])
    action = torch.zeros(1, 2)
    for step, delta in enumerate(deltas):
        action += delta
        diagnostics.update(active_mask=torch.ones(1, dtype=torch.bool), chunk_offset=step % H, action=action)
    metrics = diagnostics.metrics()
    assert metrics["validation/chunk_action_delta_boundary_component_count"] == 2
    assert metrics["validation/chunk_action_delta_internal_component_count"] == 8
    expected = deltas[4].square().mean().sqrt() / deltas[[1, 2, 3, 5]].square().mean().sqrt()
    assert metrics["validation/chunk_action_delta_boundary_internal_component_rms_ratio"] == pytest.approx(float(expected))
    assert metrics["validation/chunk_action_d2_boundary_component_count"] == 2


def test_schema_config_and_source_have_no_rate_contract() -> None:
    assert FCAMP_CHECKPOINT_CONTRACT["fcamp_schema_version"] == 18
    valid = {**FCAMP_CHECKPOINT_CONTRACT, "discriminator_policy_conditioning": False}
    validate_static_fcamp_checkpoint_contract(valid)
    obsolete = {**valid, "fcamp_schema_version": 17}
    with pytest.raises(ValueError, match="fcamp_schema_version=17"):
        validate_static_fcamp_checkpoint_contract(obsolete)

    tree = yaml.safe_load((ROOT / "configs/fcamp_largebox.yaml").read_text())
    config = _construct_fcamp_config(tree["parameters"])
    _validate_fcamp(config)
    assert (config.horizon, config.innovation_step_bound, config.cps_raw_rms) == (H, STEP, RMS)
    names = {field.name for field in fields(config)}
    banned = ("command_rate", "target_rate", "rate_half_life", "cps_physical_response", "flow_mean_raw_scale", "terminal_rate_tail")
    assert all(token not in names for token in banned)
    with pytest.raises(ValueError, match="horizon == 4"):
        _validate_fcamp(replace(config, horizon=1))
    source = "\n".join(
        path.read_text(errors="ignore")
        for root in ("models", "components/rollout", "envs", "method", "engine", "configs")
        for path in (ROOT / root).rglob("*")
        if path.suffix in {".py", ".yaml"}
    )
    assert all(token not in source for token in banned)
