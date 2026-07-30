"""Diagnostics for pure AMP with a configurable direct-action horizon."""

from __future__ import annotations

import math

import torch

from components.rollout.amp_contract import AMP_CHECKPOINT_CONTRACT


_MIN_LOGGED_HORIZON = 4
_MISSING = -1.0
_DOMAIN_PARITY_TOLERANCE = 1.0e-4
_DOMAIN_PARITY_SLICES = {
    "root_pos": slice(0, 3),
    "root_rot6": slice(3, 9),
    "joint_rot6": slice(9, 189),
    "key_body_pos": slice(189, 204),
    "root_lin_vel": slice(204, 207),
    "root_ang_vel": slice(207, 210),
    "joint_vel": slice(210, 239),
}
_NOISE_STATS = (
    "component_count",
    "signed_mean",
    "rms",
    "abs_mean",
    "abs_p95",
    "abs_max",
)


def _logged_horizon_slots(horizon: int) -> range:
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    return range(max(_MIN_LOGGED_HORIZON, horizon))


def _finite(
    metrics: dict,
    key: str,
    default: float = _MISSING,
) -> float:
    try:
        value = float(metrics.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _series(
    metrics: dict,
    template: str,
    horizon: int,
    precision: int,
) -> str:
    return "/".join(
        f"{_finite(metrics, template.format(index=i)):.{precision}f}"
        for i in _logged_horizon_slots(horizon)
    )


def _ensure_horizon_metrics(
    metrics: dict,
    horizon: int,
) -> None:
    """Emit finite H0-H3 slots, extending naturally when H is larger."""

    for offset in _logged_horizon_slots(horizon):
        for statistic in _NOISE_STATS:
            default = 0.0 if statistic == "component_count" else _MISSING
            metrics.setdefault(
                f"gaussian/action_noise_h{offset}/{statistic}", default
            )
        for statistic in ("kl", "ratio", "clip", "count"):
            default = 0.0 if statistic == "count" else _MISSING
            metrics.setdefault(
                f"amp_policy/offset_{offset}_{statistic}", default
            )
        metrics.setdefault(
            f"amp_policy/offset_{offset}_fixed_std", _MISSING
        )
        metrics.setdefault(
            f"amp_policy/offset_{offset}_executed_count", 0.0
        )


def amp_domain_parity_statistics(
    policy_frames: torch.Tensor,
    expert_frames: torch.Tensor,
) -> dict[str, float]:
    """Measure same-state policy/expert feature parity before D sees either."""

    if policy_frames.shape != expert_frames.shape:
        raise ValueError(
            "policy and expert AMP frames must have identical shapes, got "
            f"{tuple(policy_frames.shape)} and {tuple(expert_frames.shape)}"
        )
    if policy_frames.ndim != 2 or policy_frames.shape[-1] != 239:
        raise ValueError(
            "AMP domain-parity frames must have shape [N,239], got "
            f"{tuple(policy_frames.shape)}"
        )
    if not bool(torch.isfinite(policy_frames).all()) or not bool(
        torch.isfinite(expert_frames).all()
    ):
        raise ValueError("AMP domain-parity frames must be finite")
    difference = (policy_frames - expert_frames).float()
    metrics = {
        "full_rms": float(difference.square().mean().sqrt().item()),
        "full_max": float(difference.abs().max().item()),
    }
    for name, feature_slice in _DOMAIN_PARITY_SLICES.items():
        selected = difference[:, feature_slice]
        metrics[f"{name}_rms"] = float(
            selected.square().mean().sqrt().item()
        )
        metrics[f"{name}_max"] = float(selected.abs().max().item())
    return metrics


def _gaussian_noise_statistics(
    delta: torch.Tensor,
    executed: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float]:
    if delta.ndim != 4:
        raise ValueError(
            "Gaussian delta must have shape [decisions,envs,H,actions]"
        )
    if executed.shape != delta.shape[:-1]:
        raise ValueError("executed mask must match Gaussian delta")
    metrics: dict[str, float] = {}

    def record(name: str, values: torch.Tensor) -> None:
        flat = values.detach().float().reshape(-1)
        if flat.numel() == 0:
            for statistic in _NOISE_STATS:
                metrics[f"{name}/{statistic}"] = (
                    0.0 if statistic == "component_count" else _MISSING
                )
            return
        absolute = flat.abs()
        metrics.update(
            {
                f"{name}/component_count": float(flat.numel()),
                f"{name}/signed_mean": float(flat.mean().item()),
                f"{name}/rms": float(
                    flat.square().mean().sqrt().item()
                ),
                f"{name}/abs_mean": float(absolute.mean().item()),
                f"{name}/abs_p95": float(
                    torch.quantile(absolute, 0.95).item()
                ),
                f"{name}/abs_max": float(absolute.max().item()),
            }
        )

    expanded_mask = executed.bool().unsqueeze(-1).expand_as(delta)
    record(prefix, delta[expanded_mask])
    for offset in _logged_horizon_slots(delta.shape[2]):
        values = (
            delta[..., offset, :][executed[..., offset].bool()]
            if offset < delta.shape[2]
            else delta.new_empty((0,))
        )
        record(f"{prefix}_h{offset}", values)
    last = delta.shape[2] - 1
    first_rms = metrics[f"{prefix}_h0/rms"]
    last_rms = metrics[f"{prefix}_h{last}/rms"]
    metrics[f"{prefix}/last_to_first_rms_ratio"] = (
        float(last_rms / first_rms)
        if first_rms > 1.0e-12 and last_rms >= 0.0
        else _MISSING
    )
    return metrics


def gaussian_action_noise_statistics(
    action_delta: torch.Tensor,
    executed: torch.Tensor,
) -> dict[str, float]:
    return _gaussian_noise_statistics(
        action_delta,
        executed,
        prefix="gaussian/action_noise",
    )


class AMPDiagnosticsMixin:
    """Console reporting with no training or sampling behavior."""

    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        horizon = int(self.horizon_h)
        _ensure_horizon_metrics(metrics, horizon)
        std_h = _series(
            metrics,
            "amp_policy/offset_{index}_fixed_std",
            horizon,
            5,
        )
        noise_h = _series(
            metrics,
            "gaussian/action_noise_h{index}/rms",
            horizon,
            5,
        )
        kl_h = _series(
            metrics, "amp_policy/offset_{index}_kl", horizon, 6
        )
        clip_h = _series(
            metrics, "amp_policy/offset_{index}_clip", horizon, 4
        )
        ratio_h = _series(
            metrics, "amp_policy/offset_{index}_ratio", horizon, 4
        )
        print(
            f"[UPDATE] {update_idx}/{max_updates} "
            f"amp={_finite(metrics, 'reward/amp/mean'):.5f} "
            f"done={_finite(metrics, 'rollout/done_fraction'):.5f} "
            f"ep_len={_finite(metrics, 'train/mean_episode_length'):.2f}",
            flush=True,
        )
        print(
            "[AMP_POLICY] "
            f"loss={_finite(metrics, 'amp_policy/policy_loss'):.5f} "
            f"bound={_finite(metrics, 'amp_policy/action_bound_loss'):.5f} "
            f"kl={_finite(metrics, 'amp_policy/diagnostic_kl'):.6f} "
            f"ratio={_finite(metrics, 'amp_policy/ratio'):.4f} "
            f"clip={_finite(metrics, 'amp_policy/clip_fraction'):.4f} "
            f"grad={_finite(metrics, 'amp_policy/grad_norm'):.4f} "
            f"lr={_finite(metrics, 'amp_policy/lr'):.6f}",
            flush=True,
        )
        print(
            "[AMP_EXPLORATION] "
            f"fixed_std={_finite(metrics, 'amp_policy/fixed_normalized_std'):.5f} "
            f"std_h={std_h} noise_rms="
            f"{_finite(metrics, 'gaussian/action_noise/rms'):.5f} "
            f"noise_h={noise_h} "
            f"last_first={_finite(metrics, 'gaussian/action_noise/last_to_first_rms_ratio'):.4f} "
            f"env_clip={_finite(metrics, 'amp_policy/environment_clip_fraction'):.5f} "
            f"mean_oob={_finite(metrics, 'amp_policy/mean_out_of_bounds_fraction'):.5f}",
            flush=True,
        )
        print(
            f"[AMP_PPO_H] kl={kl_h} clip={clip_h} ratio={ratio_h}",
            flush=True,
        )
        print(
            "[RESET_SAMPLING] mode=uniform_continuous_full_motion "
            f"support=[{_finite(metrics, 'train_reset/uniform_low'):.0f},"
            f"{_finite(metrics, 'train_reset/uniform_high_exclusive'):.0f}) "
            f"phase={_finite(metrics, 'train_reset/all/phase_min'):.3f}/"
            f"{_finite(metrics, 'train_reset/all/phase_mean'):.2f}/"
            f"{_finite(metrics, 'train_reset/all/phase_max'):.3f} "
            f"count={_finite(metrics, 'train_reset/all/count', 0.0):.0f}",
            flush=True,
        )
        print(
            "[AMP_CREDIT] "
            f"raw={_finite(metrics, 'reward/amp/mean'):.5f} "
            f"chunk={_finite(metrics, 'reward/amp_credit/mean'):.5f} "
            f"adv={_finite(metrics, 'credit/amp_adv/mean'):.5f}/"
            f"{_finite(metrics, 'credit/amp_adv/std'):.5f} "
            f"actor_adv={_finite(metrics, 'credit/actor_adv/mean'):.5f}/"
            f"{_finite(metrics, 'credit/actor_adv/std'):.5f} "
            f"duration={_finite(metrics, 'amp_policy/decision_duration/mean'):.3f}",
            flush=True,
        )
        print(
            "[AMP_HISTORY] "
            f"primitive={_finite(metrics, 'rollout/primitive_count', 0.0):.0f} "
            f"decisions={_finite(metrics, 'rollout/decision_count', 0.0):.0f} "
            f"endpoint={_finite(metrics, 'rollout/endpoint_valid_count', 0.0):.0f} "
            f"coverage={_finite(metrics, 'rollout/endpoint_coverage_fraction_alive'):.5f} "
            f"warmup_gap={_finite(metrics, 'rollout/endpoint_warmup_gap_count', 0.0):.0f} "
            f"credit_gap={_finite(metrics, 'rollout/credit_gap_count', 0.0):.0f}",
            flush=True,
        )
        print(
            "[CRITIC] "
            f"loss={_finite(metrics, 'critic/loss'):.5f} "
            f"V={_finite(metrics, 'critic/value/mean'):.4f} "
            f"target={_finite(metrics, 'critic/target/mean'):.4f} "
            f"rmse={_finite(metrics, 'critic/value_rmse'):.4f} "
            f"ev={_finite(metrics, 'critic/explained_variance'):.4f} "
            f"grad={_finite(metrics, 'critic/grad_norm'):.4f}",
            flush=True,
        )
        print(
            "[DISCRIMINATOR] "
            f"train_loss={_finite(metrics, 'disc/loss'):.5f} "
            f"train_bce={_finite(metrics, 'disc/bce'):.5f} "
            f"gp={_finite(metrics, 'disc/gradient_penalty'):.5f} "
            f"post_accE={_finite(metrics, 'disc/post_expert_accuracy'):.4f} "
            f"post_accC={_finite(metrics, 'disc/post_current_accuracy'):.4f} "
            f"post_accR={_finite(metrics, 'disc/post_replay_accuracy'):.4f} "
            f"reward_clamp={_finite(metrics, 'amp_reward/clamp_fraction'):.5f}",
            flush=True,
        )
        print(
            "[DISC_TRAIN_PRE] "
            f"accE={_finite(metrics, 'disc/train_pre_expert_accuracy'):.4f} "
            f"accC={_finite(metrics, 'disc/train_pre_current_accuracy'):.4f} "
            f"accR={_finite(metrics, 'disc/train_pre_replay_accuracy'):.4f} "
            f"logitE={_finite(metrics, 'disc/train_pre_expert_logit_mean'):.4f} "
            f"logitC={_finite(metrics, 'disc/train_pre_current_logit_mean'):.4f} "
            f"logitR={_finite(metrics, 'disc/train_pre_replay_logit_mean'):.4f}",
            flush=True,
        )
        print(
            "[DISC_DOMAINS] "
            f"post_logitC={_finite(metrics, 'disc/post_current_logit_mean'):.4f} "
            f"post_logitR={_finite(metrics, 'disc/post_replay_logit_mean'):.4f} "
            f"post_logitE={_finite(metrics, 'disc/post_expert_logit_mean'):.4f} "
            f"count={_finite(metrics, 'disc/post_current_count', 0.0):.0f}/"
            f"{_finite(metrics, 'disc/post_replay_count', 0.0):.0f}/"
            f"{_finite(metrics, 'disc/post_expert_count', 0.0):.0f} "
            f"effective_w={_finite(metrics, 'disc/effective_current_loss_weight'):.2f}/"
            f"{_finite(metrics, 'disc/effective_replay_loss_weight'):.2f}/"
            f"{_finite(metrics, 'disc/effective_expert_loss_weight'):.2f}",
            flush=True,
        )
        print(
            "[DISC_ENDPOINT] "
            f"support=[{_finite(metrics, 'disc_endpoint/support_min'):.0f},"
            f"{_finite(metrics, 'disc_endpoint/support_max'):.0f}] "
            f"current={_finite(metrics, 'disc_endpoint/current_train/p05'):.1f}/"
            f"{_finite(metrics, 'disc_endpoint/current_train/p50'):.1f}/"
            f"{_finite(metrics, 'disc_endpoint/current_train/p95'):.1f} "
            f"expert={_finite(metrics, 'disc_endpoint/expert_train/p05'):.1f}/"
            f"{_finite(metrics, 'disc_endpoint/expert_train/p50'):.1f}/"
            f"{_finite(metrics, 'disc_endpoint/expert_train/p95'):.1f} "
            f"current_expert_tv={_finite(metrics, 'disc_endpoint/current_expert_tv'):.5f}",
            flush=True,
        )
        print(
            "[AMP_REPLAY] "
            f"size={_finite(metrics, 'disc_replay/size', 0.0):.0f}/"
            f"{_finite(metrics, 'disc_replay/capacity', 0.0):.0f} "
            f"fill={_finite(metrics, 'disc_replay/fill_fraction'):.5f} "
            f"inserted={_finite(metrics, 'disc/replay_inserted', 0.0):.0f}",
            flush=True,
        )
        print(
            "[AMP_SEAMS] "
            f"within_rms={_finite(metrics, 'action_delta/within_chunk_rms'):.5f} "
            f"boundary_rms={_finite(metrics, 'action_delta/chunk_boundary_rms'):.5f} "
            f"boundary_within={_finite(metrics, 'action_delta/boundary_to_within_ratio'):.5f}",
            flush=True,
        )
        print(
            "[AMP_TERMINATION] "
            f"done={_finite(metrics, 'rollout/done_fraction'):.6f} "
            f"timeout={_finite(metrics, 'rollout/timeout_fraction'):.6f} "
            f"illegal_contact={_finite(metrics, 'rollout/illegal_contact_fraction'):.6f} "
            f"numerical={_finite(metrics, 'rollout/numerical_failure_fraction'):.6f} "
            f"tracking_cf={_finite(metrics, 'rollout/tracking_counterfactual_fraction'):.6f} "
            f"reference_end_occupancy={_finite(metrics, 'rollout/reference_end_occupancy_fraction'):.6f}",
            flush=True,
        )
        print(
            "[AMP_CONTRACT] "
            f"action_violation={_finite(metrics, 'act/policy_bound_violation_max'):.3e} "
            f"history_gap={_finite(metrics, 'rollout/endpoint_warmup_gap_count', 0.0):.0f} "
            f"credit_gap={_finite(metrics, 'rollout/credit_gap_count', 0.0):.0f} "
            f"finite={_finite(metrics, 'system/parameters_finite', 0.0):.0f}",
            flush=True,
        )
        print(
            "[TIME] "
            f"collect={_finite(metrics, 'timing/collect_s', 0.0):.3f}s "
            f"actor={_finite(metrics, 'timing/actor_update_s', 0.0):.3f}s "
            f"critic={_finite(metrics, 'timing/critic_update_s', 0.0):.3f}s "
            f"disc={_finite(metrics, 'timing/disc_update_s', 0.0):.3f}s "
            f"gpu_peak={_finite(metrics, 'system/cuda_peak_allocated_gib', 0.0):.2f}GiB",
            flush=True,
        )

    def log_banner(self) -> None:
        with torch.no_grad():
            policy_frames = self.env.get_imitation_policy_frame()
            expert_frames = self.env.motion.get_amp_expert_frame_at_times(
                self.env.phase_steps
            )
            parity = amp_domain_parity_statistics(
                policy_frames,
                expert_frames,
            )
        if parity["full_max"] > _DOMAIN_PARITY_TOLERANCE:
            raise RuntimeError(
                "Policy/expert AMP feature domains disagree at the same reset "
                f"state: max={parity['full_max']:.6g} > "
                f"{_DOMAIN_PARITY_TOLERANCE:.6g}"
            )
        print(
            "[METHOD] name=amp actor=direct_absolute_fixed_gaussian "
            "prior=unconditional_temporal_discriminator reward=pure_amp "
            "critic=scalar_state_value",
            flush=True,
        )
        print(
            "[ASSETS] origin=user_local "
            f"robot_urdf={self.env.robot_asset_path} "
            f"motion={self.env.task.motion_file} "
            f"motion_frames={self.env.motion.num_frames} "
            f"motion_fps={self.env.motion.fps}",
            flush=True,
        )
        print(
            "[AMP_DOMAIN_PARITY] "
            f"full_rms={parity['full_rms']:.3e} "
            f"full_max={parity['full_max']:.3e} "
            f"key_rms={parity['key_body_pos_rms']:.3e} "
            f"root_lin_rms={parity['root_lin_vel_rms']:.3e} "
            f"root_ang_rms={parity['root_ang_vel_rms']:.3e} "
            f"joint_vel_rms={parity['joint_vel_rms']:.3e} "
            f"tolerance={_DOMAIN_PARITY_TOLERANCE:.1e} pass=1",
            flush=True,
        )
        print(
            "[ARCH] "
            f"actor_obs={self.actor_obs_dim} actor_input=self_state_only "
            f"phase=False target=False reference=False last_action=False "
            f"action_dim={self.num_act} H={self.horizon_h} "
            f"W_D={self.imitation_history_steps} "
            f"frame={self.imitation_frame_dim} "
            f"window={self.imitation_window_dim}",
            flush=True,
        )
        print(
            "[CREDIT] reward=immediate_old_discriminator "
            "reward_source=pure_amp_only dt_scale=False "
            "ratio=executed_chunk_joint_gaussian "
            "gae=variable_duration_semi_mdp",
            flush=True,
        )
        print(
            "[STYLE_PRIOR] "
            f"hidden={list(self.cfg.style_prior.hidden_dims)} "
            f"GP={self.cfg.style_prior.grad_penalty} "
            f"replay={self.cfg.style_prior.replay_size} "
            "conditioning=none history=demo_predecessor_seeded "
            "expert_sampling=independent_uniform_full_motion",
            flush=True,
        )
        print(
            "[AMP_CONTRACT] "
            f"schema={AMP_CHECKPOINT_CONTRACT['amp_schema_version']} "
            f"action={AMP_CHECKPOINT_CONTRACT['action_contract']} "
            f"observation={AMP_CHECKPOINT_CONTRACT['actor_observation_contract']} "
            f"exploration={AMP_CHECKPOINT_CONTRACT['exploration_contract']} "
            f"reset={AMP_CHECKPOINT_CONTRACT['reset_contract']} "
            f"replay={AMP_CHECKPOINT_CONTRACT['replay_contract']}",
            flush=True,
        )


__all__ = [
    "AMPDiagnosticsMixin",
    "_ensure_horizon_metrics",
    "amp_domain_parity_statistics",
    "gaussian_action_noise_statistics",
]
