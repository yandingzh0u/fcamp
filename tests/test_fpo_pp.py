"""Standalone numeric tests for the FPO++ math (pure torch, no IsaacLab).

`algorithms/fpo_pp.py` transitively imports `core.logging` (IsaacLab-backed) only in `log()`.
To test the math without a simulator we load `fpo_pp.py` with the two IsaacLab-tainted imports
(`algorithms.base`, `core.logging`) stubbed out. `networks.fpo_actor` and
`networks.mlp_actor_critic` are pure torch and imported for real.

Run:  python tests/test_fpo_pp.py   (from the repo root)
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_fpo_pp_module():
    algorithms_pkg = types.ModuleType("algorithms")
    algorithms_pkg.__path__ = [str(REPO_ROOT / "algorithms")]
    base_mod = types.ModuleType("algorithms.base")

    class _Algorithm:
        def __init__(self, cfg, env, simulation_app):
            self.cfg = cfg
            self.env = env
            self.simulation_app = simulation_app

    base_mod.Algorithm = _Algorithm
    sys.modules["algorithms"] = algorithms_pkg
    sys.modules["algorithms.base"] = base_mod

    core_pkg = types.ModuleType("core")
    core_pkg.__path__ = [str(REPO_ROOT / "core")]
    logging_mod = types.ModuleType("core.logging")
    logging_mod.log_shared_tracking = lambda *a, **k: None
    logging_mod.log_shared_update_diagnostics = lambda *a, **k: None
    sys.modules["core"] = core_pkg
    sys.modules["core.logging"] = logging_mod

    spec = importlib.util.spec_from_file_location("fpo_pp_standalone", REPO_ROOT / "algorithms" / "fpo_pp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fpo = _load_fpo_pp_module()
from networks.fpo_actor import FPOActor  # noqa: E402

torch.manual_seed(0)
ATOL = 1e-5


def _actor(obs_dim=16, action_dim=3, *, actor_scale=1.0, mlp_output_scale=1.0,
           timestep_embed_dim=8, reduction="mean", sampling_steps=4):
    return FPOActor(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dims=(32, 32), activation="elu",
        actor_scale=actor_scale, mlp_output_scale=mlp_output_scale,
        timestep_embed_dim=timestep_embed_dim, cfm_loss_reduction=reduction,
        sampling_steps=sampling_steps, action_perturb_std=0.1,
    )


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  ok: {name}")


# ---------------------------------------------------------------- CFM loss (executed action)
def test_cfm_loss_matches_reference():
    a = _actor(actor_scale=2.0, mlp_output_scale=1.3)
    B, M, A = 5, 7, a.num_actions
    obs = torch.randn(B, 16)
    action = torch.randn(B, A)
    eps = torch.randn(B, M, A)
    t = torch.rand(B, M, 1) * 0.99 + 0.005

    loss, x1_pred, x0_pred = a.get_cfm_loss(obs, action, eps, t)
    check("cfm_loss shape (B, M)", loss.shape == (B, M))
    check("x1_pred shape (B, M, A)", x1_pred.shape == (B, M, A))
    check("x0_pred shape (B, M, A)", x0_pred.shape == (B, M, A))

    # Reference: explicit per-(b, m) loop of the official formulation (CFM on a / actor_scale).
    ref = torch.zeros(B, M)
    with torch.no_grad():
        scaled = action / a.actor_scale
        for b in range(B):
            for m in range(M):
                tt = t[b, m]  # (1,)
                e = eps[b, m]
                x_t = tt * e + (1.0 - tt) * scaled[b]
                embed = a._embed_timestep(tt.view(1, 1)).view(-1)
                v = a.mlp_output_scale * a.actor(torch.cat([obs[b], embed, x_t]).unsqueeze(0)).squeeze(0)
                target = e - scaled[b]
                ref[b, m] = ((v - target) ** 2).mean()
    check("cfm_loss matches explicit loop", torch.allclose(loss, ref, atol=ATOL))
    check("cfm_loss non-negative", bool((loss >= 0).all()))


def test_cfm_reduction_mean_is_sum_over_dim():
    a_mean = _actor(reduction="mean")
    a_sum = _actor(reduction="sum")
    # Share weights so only the reduction differs.
    a_sum.load_state_dict(a_mean.state_dict())
    B, M, A = 3, 4, a_mean.num_actions
    obs = torch.randn(B, 16)
    action = torch.randn(B, A)
    eps = torch.randn(B, M, A)
    t = torch.rand(B, M, 1) * 0.99 + 0.005
    lm, _, _ = a_mean.get_cfm_loss(obs, action, eps, t)
    ls, _, _ = a_sum.get_cfm_loss(obs, action, eps, t)
    check("mean == sum / action_dim", torch.allclose(lm, ls / A, atol=ATOL))


# ---------------------------------------------------------------- on-policy ratio == 1
def test_onpolicy_ratio_is_one():
    a = _actor()
    B, M, A = 6, 8, a.num_actions
    obs = torch.randn(B, 16)
    action = torch.randn(B, A)
    eps = torch.randn(B, M, A)
    t = torch.rand(B, M, 1) * 0.99 + 0.005
    old, _, _ = a.get_cfm_loss(obs, action, eps, t)
    new, _, _ = a.get_cfm_loss(obs, action, eps, t)  # same params, same (eps, t)
    ratio = fpo.fpo_pp_ratio(old.detach(), new, delta_clip=0.0)
    check("on-policy ratio == 1", torch.allclose(ratio, torch.ones_like(ratio), atol=ATOL))


def test_ratio_delta_clip_ste():
    old = torch.tensor([[0.0, 5.0, -5.0]])
    new = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    # diff = old - new = [0, 5, -5]; STE clamp upper bound +2 -> [0, 2, -5]
    ratio = fpo.fpo_pp_ratio(old, new, delta_clip=2.0)
    expect = torch.exp(torch.tensor([[0.0, 2.0, -5.0]]))
    check("delta_clip STE upper-clamps before exp", torch.allclose(ratio, expect, atol=ATOL))
    unclamped = fpo.fpo_pp_ratio(old, new.detach(), delta_clip=0.0)
    check("delta_clip<=0 disables clamp", torch.allclose(unclamped, torch.exp(old - new.detach()), atol=ATOL))


def test_clamp_ste_passes_identity_gradient():
    x = torch.tensor([5.0], requires_grad=True)
    y = fpo.clamp_ste(x, max=2.0)
    check("clamp_ste forward = clamped value", torch.allclose(y, torch.tensor([2.0]), atol=ATOL))
    y.backward()
    check("clamp_ste backward = identity grad", torch.allclose(x.grad, torch.tensor([1.0]), atol=ATOL))


# ---------------------------------------------------------------- flow integration
def test_flow_constant_velocity():
    a = _actor(mlp_output_scale=1.0, sampling_steps=20)
    B, A = 4, a.num_actions
    obs = torch.randn(B, 16)
    x0 = torch.randn(B, A)
    const = torch.randn(1, A)
    orig = a.actor.forward
    a.actor.forward = lambda inp: const.expand(inp.shape[0], -1)
    try:
        out = a._integrate_flow(obs, x0)
    finally:
        a.actor.forward = orig
    # t goes 1 -> 0, sum(dt) = -1, so x_end = x0 + const * (-1).
    check("constant velocity integrates to x0 - c", torch.allclose(out, x0 - const, atol=1e-4))


def test_flow_zero_velocity():
    a = _actor(sampling_steps=8)
    B, A = 4, a.num_actions
    obs = torch.randn(B, 16)
    x0 = torch.randn(B, A)
    orig = a.actor.forward
    a.actor.forward = lambda inp: torch.zeros(inp.shape[0], a.num_actions)
    try:
        out = a._integrate_flow(obs, x0)
    finally:
        a.actor.forward = orig
    check("zero velocity is a no-op", torch.allclose(out, x0, atol=ATOL))


# ---------------------------------------------------------------- ASPO objective
def test_aspo_selection_and_values():
    clip = 0.2
    ratio = torch.tensor([[1.5, 0.5]])
    adv_pos = torch.tensor([[2.0]])
    out_pos = fpo.aspo_objective(ratio, adv_pos, clip)
    ppo_ref = torch.minimum(ratio * adv_pos, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_pos)
    check("ASPO uses PPO clip for adv>=0", torch.allclose(out_pos, ppo_ref, atol=ATOL))

    adv_neg = torch.tensor([[-2.0]])
    out_neg = fpo.aspo_objective(ratio, adv_neg, clip)
    spo_ref = ratio * adv_neg - adv_neg.abs() / (2 * clip) * (ratio - 1.0) ** 2
    check("ASPO uses SPO for adv<0", torch.allclose(out_neg, spo_ref, atol=ATOL))

    ones = torch.ones(1, 3)
    for adv in (torch.tensor([[1.3]]), torch.tensor([[-1.3]])):
        val = fpo.aspo_objective(ones, adv, clip)
        check(f"ASPO(ratio=1, adv={adv.item()}) == adv", torch.allclose(val, adv.expand_as(val), atol=ATOL))


def test_aspo_spo_penalizes_increasing_ratio_under_negative_adv():
    clip = 0.2
    adv = torch.tensor([[-1.0]])
    low = fpo.aspo_objective(torch.tensor([[0.8]]), adv, clip)
    high = fpo.aspo_objective(torch.tensor([[1.4]]), adv, clip)
    check("SPO objective decreases as ratio rises (adv<0)", bool((high < low).all()))


# ---------------------------------------------------------------- GAE (real method)
def test_gae_lambda_one_equals_discounted_returns():
    cfg = types.SimpleNamespace(gae_lambda=1.0)
    algo = fpo.FPOPP(cfg=cfg, env=None, simulation_app=None)
    T, N = 5, 2
    gamma = 0.9
    rewards = torch.ones(T, N, 1)
    values = torch.zeros(T, N, 1)
    dones = torch.zeros(T, N, 1, dtype=torch.bool)
    last_values = torch.zeros(N, 1)
    returns, advantages = algo._compute_gae(last_values, values, dones, rewards, gamma)
    ref = torch.zeros(T, N, 1)
    acc = torch.zeros(N, 1)
    for t in reversed(range(T)):
        acc = rewards[t] + gamma * acc
        ref[t] = acc
    check("GAE(lambda=1) returns == discounted reward-to-go", torch.allclose(returns, ref, atol=1e-5))
    check("GAE advantages are normalized (mean~0)", abs(float(advantages.mean().item())) < 1e-5)
    check("GAE advantages are normalized (std~1)", abs(float(advantages.std().item()) - 1.0) < 1e-3)


def test_gae_done_blocks_bootstrap():
    cfg = types.SimpleNamespace(gae_lambda=1.0)
    algo = fpo.FPOPP(cfg=cfg, env=None, simulation_app=None)
    T, N = 3, 1
    gamma = 0.9
    rewards = torch.tensor([[[1.0]], [[1.0]], [[1.0]]])
    values = torch.zeros(T, N, 1)
    dones = torch.tensor([[[False]], [[True]], [[False]]])
    last_values = torch.zeros(N, 1)
    returns, _ = algo._compute_gae(last_values, values, dones, rewards, gamma)
    check("done truncates return at t=1", abs(float(returns[1, 0, 0]) - 1.0) < 1e-6)
    check("return at t=0 includes only up to the done", abs(float(returns[0, 0, 0]) - (1.0 + gamma)) < 1e-6)


# ---------------------------------------------------------------- gradient flow sanity
def test_cfm_loss_is_differentiable():
    a = _actor()
    B, M, A = 4, 5, a.num_actions
    obs = torch.randn(B, 16)
    action = torch.randn(B, A)
    eps = torch.randn(B, M, A)
    t = torch.rand(B, M, 1) * 0.99 + 0.005
    loss, _, _ = a.get_cfm_loss(obs, action, eps, t)
    loss.mean().backward()
    grads = [p.grad for p in a.parameters() if p.grad is not None]
    check("cfm_loss produces gradients", len(grads) > 0 and any(bool((g.abs() > 0).any()) for g in grads))


if __name__ == "__main__":
    tests = [
        test_cfm_loss_matches_reference,
        test_cfm_reduction_mean_is_sum_over_dim,
        test_onpolicy_ratio_is_one,
        test_ratio_delta_clip_ste,
        test_clamp_ste_passes_identity_gradient,
        test_flow_constant_velocity,
        test_flow_zero_velocity,
        test_aspo_selection_and_values,
        test_aspo_spo_penalizes_increasing_ratio_under_negative_adv,
        test_gae_lambda_one_equals_discounted_returns,
        test_gae_done_blocks_bootstrap,
        test_cfm_loss_is_differentiable,
    ]
    for fn in tests:
        print(f"[{fn.__name__}]")
        fn()
    print(f"\nAll {len(tests)} FPO++ numeric tests passed.")
