from __future__ import annotations

import torch

from net.mixgrpo.flow_policy import FlowMatchingPolicy

SCALE = 5.0


def _make_policy(horizon: int = 12, basis_count: int = 4, action_dim: int = 29, scale: float = SCALE):
    torch.manual_seed(0)
    return FlowMatchingPolicy(
        obs_dim=196,
        action_dim=action_dim,
        horizon=horizon,
        hidden_dims=(64,),
        action_squash_scale=scale,
        basis_count=basis_count,
    )


def test_c0_first_frame_exact_in_action_space() -> None:
    policy = _make_policy()
    batch = 16
    coeff = torch.randn(batch, policy.chunk_dim)
    prev_action = (torch.rand(batch, policy.action_dim) * 2.0 - 1.0) * 4.0
    prev_prev = (torch.rand(batch, policy.action_dim) * 2.0 - 1.0) * 4.0
    chunk = policy._action_transform(coeff, start_action=prev_action, start_prev_action=prev_prev)
    chunk = chunk.view(batch, policy.horizon, policy.action_dim)
    # C0 is exact in action space regardless of boundary state.
    assert torch.allclose(chunk[:, 0], prev_action, atol=1e-4), (chunk[:, 0] - prev_action).abs().max().item()


def test_actions_strictly_bounded_even_near_edge_with_nonzero_velocity() -> None:
    # Near-edge boundary (prev_action ~ +4.99) with a positive latent velocity: exact action-
    # space C1 would demand >scale, which is impossible. The decoder must keep every action
    # strictly inside (-scale, scale) instead.
    policy = _make_policy()
    batch = 64
    coeff = torch.randn(batch, policy.chunk_dim) * 50.0
    prev_action = torch.full((batch, policy.action_dim), 4.99)
    prev_prev = torch.full((batch, policy.action_dim), 4.79)  # latent velocity > 0 toward the bound
    chunk = policy._action_transform(coeff, start_action=prev_action, start_prev_action=prev_prev)
    assert torch.all(chunk.abs() <= SCALE + 1e-5), chunk.abs().max().item()
    assert torch.isfinite(chunk).all()


def test_hundred_consecutive_chunks_never_exceed_bound_and_stay_finite() -> None:
    # Chain 100 chunks, each anchored on the previous chunk's last two frames. No action may
    # ever leave (-scale, scale) and everything must stay finite.
    policy = _make_policy()
    policy.eval()
    batch = 8
    prev_action = torch.zeros(batch, policy.action_dim)
    prev_prev = torch.zeros(batch, policy.action_dim)
    overall_max = 0.0
    with torch.no_grad():
        for _ in range(100):
            coeff = torch.randn(batch, policy.chunk_dim) * 10.0
            chunk = policy._action_transform(
                coeff, start_action=prev_action, start_prev_action=prev_prev
            ).view(batch, policy.horizon, policy.action_dim)
            assert torch.isfinite(chunk).all()
            assert torch.all(chunk.abs() <= SCALE + 1e-5), chunk.abs().max().item()
            # C0 continuity across every boundary.
            assert torch.allclose(chunk[:, 0], prev_action, atol=1e-4)
            overall_max = max(overall_max, float(chunk.abs().max().item()))
            prev_action = chunk[:, -1]
            prev_prev = chunk[:, -2]
    assert overall_max <= SCALE + 1e-5


def test_gradients_finite_through_transform() -> None:
    policy = _make_policy()
    coeff = torch.randn(4, policy.chunk_dim, requires_grad=True)
    prev_action = torch.full((4, policy.action_dim), 4.9)
    prev_prev = torch.full((4, policy.action_dim), 4.7)
    chunk = policy._action_transform(coeff, start_action=prev_action, start_prev_action=prev_prev)
    chunk.sum().backward()
    assert coeff.grad is not None and torch.isfinite(coeff.grad).all()


def test_displacement_basis_zero_first_two_rows_and_full_rank() -> None:
    policy = _make_policy(horizon=12, basis_count=4)
    basis = policy.displacement_basis
    assert torch.allclose(basis[0], torch.zeros_like(basis[0]), atol=1e-7)
    assert torch.allclose(basis[1], torch.zeros_like(basis[1]), atol=1e-7)
    rank = torch.linalg.matrix_rank(basis.to(torch.float64))
    assert int(rank.item()) == policy.basis_count


def test_basis_count_clamped_to_horizon_minus_two() -> None:
    policy = _make_policy(horizon=12, basis_count=999)
    assert policy.basis_count == 10
    assert torch.linalg.matrix_rank(policy.displacement_basis.to(torch.float64)).item() == 10


def test_latent_space_c1_continuity() -> None:
    # C1 is exact in LATENT (pre-tanh) space: atanh(action)/'s first increment equals the
    # boundary latent velocity. Verify away from the bound (where atanh is well-conditioned).
    policy = _make_policy()
    batch = 16
    coeff = torch.randn(batch, policy.chunk_dim)
    prev_action = (torch.rand(batch, policy.action_dim) * 2.0 - 1.0) * 2.0
    prev_prev = (torch.rand(batch, policy.action_dim) * 2.0 - 1.0) * 2.0
    chunk = policy._action_transform(coeff, start_action=prev_action, start_prev_action=prev_prev)
    chunk = chunk.view(batch, policy.horizon, policy.action_dim)
    z = torch.atanh(torch.clamp(chunk / SCALE, -1 + 1e-6, 1 - 1e-6))
    z_prev = torch.atanh(torch.clamp(prev_action / SCALE, -1 + 1e-6, 1 - 1e-6))
    z_prev2 = torch.atanh(torch.clamp(prev_prev / SCALE, -1 + 1e-6, 1 - 1e-6))
    expected_zv = z_prev - z_prev2
    latent_first_velocity = z[:, 1] - z[:, 0]
    assert torch.allclose(latent_first_velocity, expected_zv, atol=1e-3), (
        (latent_first_velocity - expected_zv).abs().max().item()
    )


def test_no_execution_time_stitch_attributes() -> None:
    policy = _make_policy()
    assert not hasattr(policy, "chunk_stitch_frames")
    assert not hasattr(policy, "_stitch_chunk_start")
    assert not hasattr(policy, "temporal_basis")


def test_decode_gain_near_unity_at_small_signal() -> None:
    # Unit-contract regression: with std=0.8 coefficients and zero boundary, the decoded action
    # magnitude must match the action-space scale (~0.5 mean, ~1.6 p95), NOT be inflated ~5x by
    # mixing action-unit displacement into the dimensionless latent before the final tanh*scale.
    policy = _make_policy()
    torch.manual_seed(0)
    coeff = torch.randn(4096, policy.chunk_dim) * 0.8
    chunk = policy._action_transform(
        coeff,
        start_action=torch.zeros(4096, policy.action_dim),
        start_prev_action=torch.zeros(4096, policy.action_dim),
    ).abs()
    assert float(chunk.mean().item()) < 1.0, chunk.mean().item()
    assert float(chunk.flatten().quantile(0.95).item()) < 2.5


def test_transform_deterministic() -> None:
    policy = _make_policy()
    coeff = torch.randn(4, policy.chunk_dim)
    start = torch.randn(4, policy.action_dim).clamp(-4.0, 4.0)
    prev = torch.randn(4, policy.action_dim).clamp(-4.0, 4.0)
    a = policy._action_transform(coeff, start_action=start, start_prev_action=prev)
    b = policy._action_transform(coeff.clone(), start_action=start.clone(), start_prev_action=prev.clone())
    assert torch.equal(a, b)
