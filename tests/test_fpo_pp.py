"""Standalone numeric tests for the FPO++ math (pure torch, no IsaacLab).

`algorithms/fpo_pp.py` transitively imports `core.logging`, which imports the IsaacLab-backed
env config. That dependency is only used by the `log()` method, not by any of the numeric
logic. To test the math without a simulator we load `fpo_pp.py` directly with the two
IsaacLab-tainted imports (`algorithms.base`, `core.logging`) stubbed out. `networks.flow_policy`
and `networks.mlp_actor_critic` are pure torch and imported for real.

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
    """Load fpo_pp.py with IsaacLab-dependent imports stubbed, under a standalone name."""
    # Stub `algorithms` package + `algorithms.base` with a no-op Algorithm base class.
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

    # Stub `core` package + `core.logging` (only used by FPOPP.log, irrelevant to the math).
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
from networks.flow_policy import FlowMatchingPolicy  # noqa: E402

torch.manual_seed(0)
ATOL = 1e-5


def _policy(obs_dim=16, action_dim=3, horizon=4, basis_count=4):
    return FlowMatchingPolicy(
        obs_dim=obs_dim, action_dim=action_dim, horizon=horizon,
        hidden_dims=(32, 32), activation="elu", basis_count=basis_count,
        chunk_stitch_frames=0, chunk_stitch_mode="none",
    )


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  ok: {name}")


# ---------------------------------------------------------------- CFM loss
def test_cfm_loss_matches_reference():
    p = _policy()
    B, M, D = 5, 7, p.chunk_dim
    obs = torch.randn(B, 16)
    a = torch.randn(B, D)
    tau = torch.rand(B, M)
    eps = torch.randn(B, M, D)

    loss = fpo.cfm_loss(p, obs, a, tau, eps)
    check("cfm_loss shape (B, M)", loss.shape == (B, M))

    # Reference: explicit per-(b, m) loop.
    ref = torch.zeros(B, M)
    with torch.no_grad():
        for b in range(B):
            for m in range(M):
                t = tau[b, m]
                e = eps[b, m]
                a_tau = (t * a[b] + (1.0 - t) * e).unsqueeze(0)
                target = (a[b] - e)
                pred = p.velocity_field(obs[b:b + 1], a_tau, t.view(1))
                ref[b, m] = ((pred.squeeze(0) - target) ** 2).mean()
    check("cfm_loss matches explicit loop", torch.allclose(loss, ref, atol=ATOL))
    check("cfm_loss non-negative", bool((loss >= 0).all()))


def test_cfm_loss_clamp():
    p = _policy()
    B, M, D = 3, 4, p.chunk_dim
    obs = torch.randn(B, 16)
    a = torch.randn(B, D)
    tau = torch.rand(B, M)
    eps = torch.randn(B, M, D)
    raw = fpo.cfm_loss(p, obs, a, tau, eps)
    clamp = float(raw.mean().item())
    clamped = fpo.cfm_loss(p, obs, a, tau, eps, loss_clamp=clamp)
    check("loss_clamp caps the loss", bool((clamped <= clamp + ATOL).all()))
    check("loss_clamp leaves small values untouched",
          torch.allclose(torch.minimum(raw, torch.full_like(raw, clamp)), clamped, atol=ATOL))


# ---------------------------------------------------------------- on-policy ratio == 1
def test_onpolicy_ratio_is_one():
    p = _policy()
    B, M, D = 6, 8, p.chunk_dim
    obs = torch.randn(B, 16)
    a = torch.randn(B, D)
    tau = torch.rand(B, M)
    eps = torch.randn(B, M, D)
    old = fpo.cfm_loss(p, obs, a, tau, eps).detach()
    new = fpo.cfm_loss(p, obs, a, tau, eps)  # same params, same (tau, eps)
    ratio = fpo.fpo_pp_ratio(old, new, delta_clip=0.0)
    check("on-policy ratio == 1", torch.allclose(ratio, torch.ones_like(ratio), atol=ATOL))


def test_ratio_delta_clip():
    old = torch.tensor([[0.0, 5.0, -5.0]])
    new = torch.tensor([[0.0, 0.0, 0.0]])
    # diff = old - new = [0, 5, -5]; clamp to +-2 -> [0, 2, -2]
    ratio = fpo.fpo_pp_ratio(old, new, delta_clip=2.0)
    expect = torch.exp(torch.tensor([[0.0, 2.0, -2.0]]))
    check("delta_clip clamps before exp", torch.allclose(ratio, expect, atol=ATOL))
    unclamped = fpo.fpo_pp_ratio(old, new, delta_clip=0.0)
    check("delta_clip<=0 disables clamp", torch.allclose(unclamped, torch.exp(old - new), atol=ATOL))


# ---------------------------------------------------------------- Euler integration
def test_euler_constant_velocity():
    p = _policy()
    B, D = 4, p.chunk_dim
    obs = torch.randn(B, 16)
    x0 = torch.randn(B, D)
    const = torch.randn(1, D)

    # Monkeypatch the velocity field to a constant c: endpoint = x0 + c * (steps * dt) = x0 + c.
    orig = p.velocity_field
    p.velocity_field = lambda o, x, t: const.expand(x.shape[0], -1)
    try:
        out = fpo.euler_integrate(p, obs, x0, steps=50)
    finally:
        p.velocity_field = orig
    check("euler with constant velocity = x0 + c", torch.allclose(out, x0 + const, atol=1e-4))


def test_euler_zero_velocity():
    p = _policy()
    B, D = 4, p.chunk_dim
    obs = torch.randn(B, 16)
    x0 = torch.randn(B, D)
    orig = p.velocity_field
    p.velocity_field = lambda o, x, t: torch.zeros_like(x)
    try:
        out = fpo.euler_integrate(p, obs, x0, steps=10)
    finally:
        p.velocity_field = orig
    check("euler with zero velocity is a no-op", torch.allclose(out, x0, atol=ATOL))


# ---------------------------------------------------------------- ASPO objective
def test_aspo_selection_and_values():
    clip = 0.2
    ratio = torch.tensor([[1.5, 0.5]])
    # advantage >= 0 -> PPO clip branch: min(r*A, clip(r)*A)
    adv_pos = torch.tensor([[2.0]])
    out_pos = fpo.aspo_objective(ratio, adv_pos, clip)
    ppo_ref = torch.minimum(ratio * adv_pos, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_pos)
    check("ASPO uses PPO clip for adv>=0", torch.allclose(out_pos, ppo_ref, atol=ATOL))

    # advantage < 0 -> SPO branch: r*A - |A|/(2*clip) * (r-1)^2
    adv_neg = torch.tensor([[-2.0]])
    out_neg = fpo.aspo_objective(ratio, adv_neg, clip)
    spo_ref = ratio * adv_neg - adv_neg.abs() / (2 * clip) * (ratio - 1.0) ** 2
    check("ASPO uses SPO for adv<0", torch.allclose(out_neg, spo_ref, atol=ATOL))

    # at ratio==1 both branches reduce to the advantage itself.
    ones = torch.ones(1, 3)
    for adv in (torch.tensor([[1.3]]), torch.tensor([[-1.3]])):
        val = fpo.aspo_objective(ones, adv, clip)
        check(f"ASPO(ratio=1, adv={adv.item()}) == adv",
              torch.allclose(val, adv.expand_as(val), atol=ATOL))


def test_aspo_spo_penalizes_increasing_ratio_under_negative_adv():
    # For adv<0, raising the ratio should lower the objective (we want to push prob down).
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
    # With lambda=1, V=0, no dones: return[t] = sum_{k>=t} gamma^{k-t} * 1.
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
    dones = torch.tensor([[[False]], [[True]], [[False]]])  # done at t=1 cuts the bootstrap
    last_values = torch.zeros(N, 1)
    returns, _ = algo._compute_gae(last_values, values, dones, rewards, gamma)
    # t=2: 1 ; t=1: 1 (done -> no bootstrap from t=2) ; t=0: 1 + gamma*1
    check("done truncates return at t=1", abs(float(returns[1, 0, 0]) - 1.0) < 1e-6)
    check("return at t=0 includes only up to the done", abs(float(returns[0, 0, 0]) - (1.0 + gamma)) < 1e-6)


# ---------------------------------------------------------------- gradient flow sanity
def test_cfm_loss_is_differentiable():
    p = _policy()
    B, M, D = 4, 5, p.chunk_dim
    obs = torch.randn(B, 16)
    a = torch.randn(B, D)
    tau = torch.rand(B, M)
    eps = torch.randn(B, M, D)
    loss = fpo.cfm_loss(p, obs, a, tau, eps).mean()
    loss.backward()
    grads = [param.grad for param in p.parameters() if param.grad is not None]
    check("cfm_loss produces gradients", len(grads) > 0 and any(bool((g.abs() > 0).any()) for g in grads))


if __name__ == "__main__":
    tests = [
        test_cfm_loss_matches_reference,
        test_cfm_loss_clamp,
        test_onpolicy_ratio_is_one,
        test_ratio_delta_clip,
        test_euler_constant_velocity,
        test_euler_zero_velocity,
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
