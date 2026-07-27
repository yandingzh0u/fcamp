"""FCAMP metric aggregation and stable human-readable diagnostics."""

from __future__ import annotations

import time

import torch

from components.imitation.style_reward import style_reward_statistics
from components.rollout.training_streams import CURRICULUM_STREAM, PHASE0_STREAM
from engine.checkpoint import FCAMP_CHECKPOINT_CONTRACT


def masked_stats(prefix: str, values: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    flat = values.detach().float().reshape(-1)
    if mask is not None:
        flat = flat[mask.reshape(-1).bool()]
    if flat.numel() == 0:
        return {f"{prefix}/count": 0.0}
    q = torch.quantile(flat, torch.tensor([0.05, 0.5, 0.95], device=flat.device))
    return {
        f"{prefix}/count": float(flat.numel()),
        f"{prefix}/mean": float(flat.mean().item()),
        f"{prefix}/std": float(flat.std(unbiased=False).item()),
        f"{prefix}/min": float(flat.min().item()),
        f"{prefix}/max": float(flat.max().item()),
        f"{prefix}/p05": float(q[0].item()),
        f"{prefix}/p50": float(q[1].item()),
        f"{prefix}/p95": float(q[2].item()),
    }



@torch.no_grad()
def stream_rollout_metrics(rollout: dict) -> dict[str, float]:
    stream_ids = rollout["stream_ids"]
    valid = rollout["valid"]
    amp_valid = rollout["amp_valid"]
    done = rollout["done"]
    failure = rollout["failure"]
    timeout = rollout["timeout"]
    complete = rollout["motion_complete"]
    total_valid = max(int(valid.sum().item()), 1)
    total_amp_valid = max(int(amp_valid.sum().item()), 1)
    metrics: dict[str, float] = {}
    for name, stream_id in (
        ("phase0", PHASE0_STREAM),
        ("curriculum", CURRICULUM_STREAM),
    ):
        env_mask = stream_ids == stream_id
        env_count = int(env_mask.sum().item())
        transition_mask = env_mask.reshape(1, -1, 1).expand_as(valid)
        stream_valid = valid & transition_mask
        stream_amp_valid = amp_valid & transition_mask
        stream_done = done & transition_mask
        stream_failure = failure & transition_mask
        stream_timeout = timeout & transition_mask
        stream_complete = complete & transition_mask
        valid_count = int(stream_valid.sum().item())
        amp_count = int(stream_amp_valid.sum().item())
        terminal_count = int(stream_done.sum().item())
        failure_count = int(stream_failure.sum().item())
        timeout_count = int(stream_timeout.sum().item())
        complete_count = int(stream_complete.sum().item())
        possible = max(
            env_count * valid.shape[0] * valid.shape[2],
            1,
        )
        prefix = f"stream/{name}"
        metrics.update(
            {
                f"{prefix}/env_count": float(env_count),
                f"{prefix}/env_fraction": float(
                    env_count / max(stream_ids.numel(), 1)
                ),
                f"{prefix}/valid_transition_count": float(
                    valid_count
                ),
                f"{prefix}/valid_transition_fraction": float(
                    valid_count / possible
                ),
                f"{prefix}/valid_transition_share": float(
                    valid_count / total_valid
                ),
                f"{prefix}/amp_valid_window_count": float(amp_count),
                f"{prefix}/amp_valid_window_share": float(
                    amp_count / total_amp_valid
                ),
                f"{prefix}/terminal_count": float(terminal_count),
                f"{prefix}/failure_count": float(failure_count),
                f"{prefix}/timeout_count": float(timeout_count),
                f"{prefix}/motion_complete_count": float(
                    complete_count
                ),
                f"{prefix}/failure_rate": float(
                    failure_count / max(terminal_count, 1)
                ),
                f"{prefix}/completion_rate": float(
                    complete_count / max(terminal_count, 1)
                ),
            }
        )
        start_phases = rollout["collection_start_phases"][env_mask]
        if start_phases.numel() > 0:
            metrics.update(
                {
                    f"{prefix}/collection_start_mean": float(
                        start_phases.float().mean().item()
                    ),
                    f"{prefix}/collection_start_min": float(
                        start_phases.min().item()
                    ),
                    f"{prefix}/collection_start_max": float(
                        start_phases.max().item()
                    ),
                }
            )
        failure_phases = rollout["terminal_phase"][
            stream_failure
        ].float()
        if failure_phases.numel() > 0:
            quantiles = torch.quantile(
                failure_phases,
                torch.tensor(
                    [0.5, 0.95],
                    device=failure_phases.device,
                ),
            )
            metrics.update(
                {
                    f"{prefix}/failure_phase_mean": float(
                        failure_phases.mean().item()
                    ),
                    f"{prefix}/failure_phase_p50": float(
                        quantiles[0].item()
                    ),
                    f"{prefix}/failure_phase_p95": float(
                        quantiles[1].item()
                    ),
                }
            )
        else:
            for suffix in (
                "failure_phase_mean",
                "failure_phase_p50",
                "failure_phase_p95",
            ):
                metrics[f"{prefix}/{suffix}"] = -1.0
        if amp_count > 0:
            metrics[f"{prefix}/amp_logit_mean"] = float(
                rollout["amp_logits"][stream_amp_valid].mean().item()
            )
            metrics[f"{prefix}/amp_reward_mean"] = float(
                rollout["amp_reward_raw"][
                    stream_amp_valid
                ].mean().item()
            )
        else:
            metrics[f"{prefix}/amp_logit_mean"] = -1.0
            metrics[f"{prefix}/amp_reward_mean"] = -1.0
    return metrics


def collect_update_metrics(
    algorithm,
    rollout: dict,
    *,
    actor_metrics: dict[str, float],
    critic_metrics: dict[str, float],
    disc_metrics: dict[str, float],
    reward_metrics: dict[str, float],
    collect_time: float,
    actor_time: float,
    critic_time: float,
    disc_time: float,
    update_start: float,
) -> dict[str, float]:
    valid = rollout["valid"]
    amp_valid = rollout["amp_valid"]
    channel_valid = rollout["channel_valid"]
    metrics: dict[str, float] = {}
    metrics.update(actor_metrics)
    metrics.update(algorithm._cps_statistics())
    metrics.update(
        {
            "critic/prefix_normalizer_frozen": 1.0,
            "critic/prefix_normalizer_count": float(
                algorithm.prefix_context_normalizer.count.item()
            ),
        }
    )
    metrics.update(critic_metrics)
    metrics.update(disc_metrics)
    metrics.update(reward_metrics)
    metrics.update(stream_rollout_metrics(rollout))
    metrics.update(algorithm.phase0_attempts.metrics())
    metrics.update(masked_stats("reward/task", rollout["task_reward"], valid))
    metrics.update(masked_stats("reward/amp_raw", rollout["amp_reward_raw"], amp_valid))
    metrics.update(masked_stats("reward/amp_credit", rollout["amp_reward_credit"], amp_valid))
    metrics.update(masked_stats("reward/mixed", rollout["mixed_reward"], valid))
    metrics.update(masked_stats("credit/task_adv", rollout["channel_advantages"][..., 0], valid))
    metrics.update(
        masked_stats(
            "credit/amp_adv",
            rollout["channel_advantages"][..., 1],
            channel_valid[..., 1],
        )
    )
    weighted_channel_advantage = (
        float(algorithm.cfg.credit.task_weight)
        * rollout["channel_advantages"][..., 0]
        + float(algorithm.cfg.credit.amp_weight)
        * rollout["channel_advantages"][..., 1]
    )
    mixed_identity_error = (
        weighted_channel_advantage - rollout["mixed_advantage"]
    ).abs()
    metrics["credit/mixed_identity_abs_max"] = float(
        mixed_identity_error[valid].max().item()
        if bool(valid.any())
        else 0.0
    )
    metrics.update(
        masked_stats(
            "credit/mixed_adv",
            rollout["mixed_advantage"],
            valid,
        )
    )
    metrics.update(
        masked_stats(
            "credit/task_actor_component",
            rollout["actor_advantage_components"][..., 0],
            valid,
        )
    )
    metrics.update(
        masked_stats(
            "credit/amp_actor_component",
            rollout["actor_advantage_components"][..., 1],
            channel_valid[..., 1],
        )
    )
    metrics.update(masked_stats("credit/actor_adv", rollout["advantages"], valid))
    metrics.update(masked_stats("critic/task_value", rollout["values"][..., 0], valid))
    metrics.update(
        masked_stats(
            "critic/amp_value",
            rollout["values"][..., 1],
            channel_valid[..., 1],
        )
    )
    metrics.update(masked_stats("critic/task_target", rollout["value_targets"][..., 0], valid))
    metrics.update(
        masked_stats(
            "critic/amp_target",
            rollout["value_targets"][..., 1],
            channel_valid[..., 1],
        )
    )
    metrics.update(
        style_reward_statistics(
            rollout["amp_logits"][amp_valid],
            rollout["amp_reward_raw"][amp_valid],
            scale=float(algorithm.cfg.style_prior.reward_scale),
            minimum_one_minus_prob=float(
                algorithm.cfg.style_prior.reward_epsilon
            ),
            prefix="amp_reward",
        )
    )
    metrics.update(algorithm.disc_normalizer.statistics())
    metrics.update(algorithm.imitation_history.statistics())
    metrics.update(
        {
            "rollout/valid_fraction": float(valid.float().mean().item()),
            "rollout/done_fraction": float(rollout["done"].float().mean().item()),
            "rollout/failure_fraction": float(rollout["failure"].float().mean().item()),
            "rollout/timeout_fraction": float(rollout["timeout"].float().mean().item()),
            "rollout/motion_complete_fraction": float(
                rollout["motion_complete"].float().mean().item()
            ),
            "rollout/bootstrap_fraction": float(
                rollout["bootstrap_mask"].float().mean().item()
            ),
            "rollout/trace_fraction": float(rollout["trace_mask"].float().mean().item()),
            "phase/start_mean": float(
                rollout["collection_start_phases"].float().mean().item()
            ),
            "phase/start_min": float(rollout["collection_start_phases"].min().item()),
            "phase/start_max": float(rollout["collection_start_phases"].max().item()),
            "control/raw_innovation_abs_max": float(
                rollout["raw_innovation_abs_max"]
            ),
            "control/raw_innovation_tanh_saturation_fraction": float(
                rollout["raw_innovation_tanh_saturation_fraction"]
            ),
            "control/action_delta_abs_max": float(
                rollout["action_delta_abs_max"]
            ),
            "control/raw_innovation_boundary_internal_rms_ratio": float(
                rollout["raw_innovation_boundary_internal_rms_ratio"]
            ),
            "control/action_delta_boundary_internal_rms_ratio": float(
                rollout["action_delta_boundary_internal_rms_ratio"]
            ),
            "control/action_d2_boundary_internal_rms_ratio": float(
                rollout["action_d2_boundary_internal_rms_ratio"]
            ),
            "control/continuation_boundary_count": float(
                rollout["continuation_boundary_count"]
            ),
            "control/reset_boundary_count": float(
                rollout["reset_boundary_count"]
            ),
            "control/reset_action_delta_rms": float(
                rollout["reset_action_delta_rms"]
            ),
            "control/reset_action_d2_rms": float(
                rollout["reset_action_d2_rms"]
            ),
            "disc_contract/fk_alignment_abs_max": float(
                rollout["fk_alignment_abs_max"]
            ),
            "disc_contract/dirty_window_excluded_count": float(
                rollout["dirty_window_excluded_count"]
            ),
            "intervention/edge_count": float(
                rollout["intervention_edge"].sum().item()
            ),
            "intervention/amp_bootstrap_cut_count": float(
                (
                    rollout["bootstrap_mask"]
                    & ~rollout["amp_bootstrap_mask"]
                ).sum().item()
            ),
            "intervention/amp_trace_cut_count": float(
                (
                    rollout["trace_mask"] & ~rollout["amp_trace_mask"]
                ).sum().item()
            ),
            "disc_window/policy_latest_root_xy_abs_max": float(
                rollout["window_latest_root_xy_abs_max"]
            ),
            "disc_window/policy_root_xy_abs_mean": float(
                rollout["window_root_xy_abs_mean"]
            ),
            "train/mean_reward": float(
                sum(algorithm._train_reward_buffer) / len(algorithm._train_reward_buffer)
                if algorithm._train_reward_buffer
                else float("nan")
            ),
            "train/mean_episode_length": float(
                sum(algorithm._train_length_buffer) / len(algorithm._train_length_buffer)
                if algorithm._train_length_buffer
                else float("nan")
            ),
            "timing/collect_s": float(collect_time),
            "timing/actor_update_s": float(actor_time),
            "timing/critic_update_s": float(critic_time),
            "timing/disc_update_s": float(disc_time),
            "timing/update_s": float(time.perf_counter() - update_start),
            "system/cuda_peak_allocated_gib": float(
                torch.cuda.max_memory_allocated(algorithm.env.device) / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            ),
        }
    )
    algorithm._add_sampler_metrics(metrics)
    finite_parameters = all(
        bool(torch.isfinite(parameter).all())
        for module in (algorithm._policy, algorithm.critic, algorithm.discriminator)
        for parameter in module.parameters()
    )
    metrics["system/parameters_finite"] = float(finite_parameters)
    if not finite_parameters:
        raise FloatingPointError("FCAMP detected non-finite trainable parameters")
    return metrics


def log_update(algorithm, update_idx: int, max_updates: int, metrics: dict) -> None:
    print(
        f"[UPDATE] {update_idx}/{max_updates} "
        f"task={metrics.get('reward/task/mean', float('nan')):.5f} "
        f"amp={metrics.get('reward/amp_raw/mean', float('nan')):.5f} "
        f"mixed={metrics.get('reward/mixed/mean', float('nan')):.5f} "
        f"done={metrics.get('rollout/done_fraction', float('nan')):.5f} "
        f"ep_len={metrics.get('train/mean_episode_length', float('nan')):.2f}",
        flush=True,
    )
    print(
        f"[FCAMP] policy={metrics.get('fcamp/policy_loss', float('nan')):.5f} "
        f"kl={metrics.get('fcamp/kl', float('nan')):.6f} "
        f"max_kl={metrics.get('fcamp/exact_frame_kl_max', float('nan')):.6f}/"
        f"{metrics.get('fcamp/kl_acceptance_limit', float('nan')):.6f} "
        f"ratio={metrics.get('fcamp/ratio', float('nan')):.4f} "
        f"clip={metrics.get('fcamp/clip_fraction', float('nan')):.4f} "
        f"grad={metrics.get('fcamp/actor_grad_norm', float('nan')):.4f} "
        f"lr={metrics.get('fcamp/actor_lr', float('nan')):.6f} "
        f"accept={metrics.get('fcamp/actor_optimizer_steps', 0.0):.0f} "
        f"reject={metrics.get('fcamp/actor_rejected_attempts', 0.0):.0f} "
        f"skip={metrics.get('fcamp/actor_skipped_steps', 0.0):.0f}",
        flush=True,
    )
    print(
        "[FCAMP_CONTROL] "
        f"cps_raw_rms={metrics.get('policy/cps_raw_rms_achieved', float('nan')):.5f}/"
        f"{metrics.get('policy/cps_raw_rms_target', float('nan')):.5f} "
        f"shape={metrics.get('policy/cps_shape_norm', float('nan')):.4f}/"
        f"{metrics.get('policy/cps_shape_radius', float('nan')):.4f} "
        f"raw_max={metrics.get('control/raw_innovation_abs_max', float('nan')):.4f} "
        f"tanh_sat={metrics.get('control/raw_innovation_tanh_saturation_fraction', float('nan')):.6f} "
        f"delta_max={metrics.get('control/action_delta_abs_max', float('nan')):.4f}/"
        f"{float(algorithm.cfg.innovation_step_bound):.4f}",
        flush=True,
    )
    print(
        "[FCAMP_NORM] "
        f"actor_fixed={metrics.get('policy/actor_obs_normalizer_frozen', float('nan')):.0f} "
        f"actor_n={metrics.get('policy/actor_obs_normalizer_count', float('nan')):.0f} "
        f"critic_fixed={metrics.get('critic/prefix_normalizer_frozen', float('nan')):.0f} "
        f"critic_n={metrics.get('critic/prefix_normalizer_count', float('nan')):.0f} "
        f"mean_abs={metrics.get('policy/actor_obs_normalizer_mean_abs', float('nan')):.5f} "
        f"scale={metrics.get('policy/actor_obs_normalizer_scale_min', float('nan')):.5f}/"
        f"{metrics.get('policy/actor_obs_normalizer_scale_mean', float('nan')):.5f}/"
        f"{metrics.get('policy/actor_obs_normalizer_scale_max', float('nan')):.5f}",
        flush=True,
    )
    print(
        "[FCAMP_SEAM] "
        f"continuation_n={metrics.get('control/continuation_boundary_count', float('nan')):.0f} "
        f"raw_rms_ratio={metrics.get('control/raw_innovation_boundary_internal_rms_ratio', float('nan')):.5f} "
        f"d1_rms_ratio={metrics.get('control/action_delta_boundary_internal_rms_ratio', float('nan')):.5f} "
        f"d2_rms_ratio={metrics.get('control/action_d2_boundary_internal_rms_ratio', float('nan')):.5f} "
        f"reset_n={metrics.get('control/reset_boundary_count', float('nan')):.0f} "
        f"reset_d1={metrics.get('control/reset_action_delta_rms', float('nan')):.5f} "
        f"reset_d2={metrics.get('control/reset_action_d2_rms', float('nan')):.5f}",
        flush=True,
    )
    print(
        f"[SAMPLER_PHASE] "
        f"start_mean={metrics.get('phase/start_mean', float('nan')):.2f} "
        f"start_min={metrics.get('phase/start_min', float('nan')):.0f} "
        f"start_max={metrics.get('phase/start_max', float('nan')):.0f} "
        f"top_bin={metrics.get('sampler/top_bin', float('nan')):.0f} "
        f"top_prob={metrics.get('sampler/top_prob', float('nan')):.5f} "
        f"entropy={metrics.get('sampler/entropy', float('nan')):.5f} "
        f"failed_sum={metrics.get('sampler/failed_sum', float('nan')):.3f}",
        flush=True,
    )
    print(
        "[FCAMP_STREAM] "
        f"phase0_env={metrics.get('stream/phase0/env_fraction', float('nan')):.3f} "
        f"phase0_valid_share={metrics.get('stream/phase0/valid_transition_share', float('nan')):.3f} "
        f"phase0_actor_w={metrics.get('stream/phase0/actor_objective_weight', float('nan')):.3f} "
        f"phase0_critic_w={metrics.get('stream/phase0/critic_objective_weight', float('nan')):.3f} "
        f"phase0_fail_p50={metrics.get('stream/phase0/failure_phase_p50', -1.0):.1f} "
        f"curr_fail_p50={metrics.get('stream/curriculum/failure_phase_p50', -1.0):.1f} "
        f"phase0_inflight_p50={metrics.get('stream/phase0_attempt/inflight_age_p50', -1.0):.1f} "
        f"phase0_success_cum={metrics.get('stream/phase0_attempt/cumulative_success_rate', 0.0):.5f} "
        f"disc_buffer_phase0={metrics.get('disc/current_phase0_fraction', float('nan')):.3f} "
        f"disc_train_phase0={metrics.get('disc/training_phase0_fraction', float('nan')):.3f} "
        f"phase0_complete={metrics.get('stream/phase0/completion_rate', 0.0):.5f} "
        f"disc_quota={metrics.get('disc/stream_quota_available', 0.0):.0f}",
        flush=True,
    )
    print(
        f"[FCAMP_CREDIT] "
        f"task_raw_std={metrics.get('credit/task_adv/std', float('nan')):.5f} "
        f"amp_raw_std={metrics.get('credit/amp_adv/std', float('nan')):.5f} "
        f"task_actor_std={metrics.get('credit/task_actor_component/std', float('nan')):.5f} "
        f"amp_actor_std={metrics.get('credit/amp_actor_component/std', float('nan')):.5f} "
        f"mixed_identity_err={metrics.get('credit/mixed_identity_abs_max', float('nan')):.3e}",
        flush=True,
    )
    print(
        f"[DUAL_CRITIC] task_loss={metrics.get('critic/task_flow_loss', float('nan')):.5f} "
        f"amp_loss={metrics.get('critic/amp_flow_loss', float('nan')):.5f} "
        f"task_V={metrics.get('critic/task_value/mean', float('nan')):.4f} "
        f"amp_V={metrics.get('critic/amp_value/mean', float('nan')):.4f} "
        f"grad={metrics.get('critic/grad_norm', float('nan')):.4f}",
        flush=True,
    )
    print(
        f"[DISC] loss={metrics.get('disc/loss', float('nan')):.5f} "
        f"bce={metrics.get('disc/bce', float('nan')):.5f} "
        f"gp={metrics.get('disc/gradient_penalty', float('nan')):.5f} "
        f"accE={metrics.get('disc/expert_accuracy', float('nan')):.3f} "
        f"accP={metrics.get('disc/current_accuracy', float('nan')):.3f} "
        f"reward_clamp={metrics.get('amp_reward/clamp_fraction', float('nan')):.5f} "
        f"replay={metrics.get('disc_replay/size', float('nan')):.0f}",
        flush=True,
    )
    print(
        "[FCAMP_CONTRACT] "
        f"fk_max={metrics.get('disc_contract/fk_alignment_abs_max', float('nan')):.3e} "
        f"dirty_excluded={metrics.get('disc_contract/dirty_window_excluded_count', 0.0):.0f} "
        f"replay_dirty={metrics.get('disc_replay/dirty_insert_count', -1.0):.0f} "
        f"push_edges={metrics.get('intervention/edge_count', 0.0):.0f} "
        f"amp_trace_cuts={metrics.get('intervention/amp_trace_cut_count', 0.0):.0f} "
        f"replay_phase0={metrics.get('disc_replay/stream_0_size', 0.0):.0f}/"
        f"{metrics.get('disc_replay/stream_0_capacity', 0.0):.0f} "
        f"replay_curr={metrics.get('disc_replay/stream_1_size', 0.0):.0f}/"
        f"{metrics.get('disc_replay/stream_1_capacity', 0.0):.0f}",
        flush=True,
    )
    print(
        f"[TIME] collect={metrics['timing/collect_s']:.3f}s "
        f"actor={metrics['timing/actor_update_s']:.3f}s "
        f"critic={metrics['timing/critic_update_s']:.3f}s "
        f"disc={metrics['timing/disc_update_s']:.3f}s "
        f"gpu_peak={metrics['system/cuda_peak_allocated_gib']:.2f}GiB",
        flush=True,
    )
    print(
        "[FCAMP_HEALTH] "
        f"parameters_finite={metrics.get('system/parameters_finite', float('nan')):.0f} "
        f"valid={metrics.get('rollout/valid_fraction', float('nan')):.5f} "
        f"trace={metrics.get('rollout/trace_fraction', float('nan')):.5f} "
        f"bootstrap={metrics.get('rollout/bootstrap_fraction', float('nan')):.5f} "
        f"amp_valid={metrics.get('amp/valid_window_fraction', float('nan')):.5f} "
        f"history_ready={metrics.get('history/ready_fraction', float('nan')):.5f} "
        f"history_dirty={metrics.get('history/intervention_dirty_fraction', float('nan')):.5f} "
        f"disc_norm_count={metrics.get('disc_norm/count', float('nan')):.0f} "
        f"disc_norm_std_min={metrics.get('disc_norm/std_min', float('nan')):.5f} "
        f"disc_norm_std_mean={metrics.get('disc_norm/std_mean', float('nan')):.5f} "
        f"disc_norm_std_max={metrics.get('disc_norm/std_max', float('nan')):.5f}",
        flush=True,
    )


def log_banner(algorithm) -> None:
    print(
        "[METHOD] name=fcamp actor=flat_chunk_flow_cps prior=temporal_discriminator "
        "credit=causal_frame critic=shared_dual_flow",
        flush=True,
    )
    print(
        f"[ARCH] task={algorithm.env.task.name} actor_obs={algorithm.actor_obs_dim} "
        f"env_actor_obs={algorithm.base_actor_obs_dim} discriminator_in_actor=False "
        f"prefix_context={algorithm.prefix_context_dim} action_dim={algorithm.num_act} "
        f"H={algorithm.horizon_h} W_D={algorithm.imitation_history_steps} "
        f"imitation_frame={algorithm.imitation_frame_dim} imitation_window={algorithm.imitation_window_dim}",
        flush=True,
    )
    print(
        f"[CREDIT] critic_sharing=encoder heads=task,amp "
        f"credit=causal_frame advantage_norm=global "
        f"ratio_mode=factorized_frame "
        f"task_weight={algorithm.cfg.credit.task_weight} amp_weight={algorithm.cfg.credit.amp_weight} "
        f"amp_dt=True",
        flush=True,
    )
    print(
        "[CONTROL] variable=raw_action_innovation "
        "mapping=continuous_pre_squash_residual "
        "covariance=shared_joint_iid_time "
        "mean=zero_source_flat_chunk_flow "
        f"dt={algorithm.env.dt:.5f} "
        f"action_bound={float(algorithm.action_high.max().item()):.3f} "
        f"step_bound={float(algorithm.cfg.innovation_step_bound):.3f} "
        f"cps_raw_rms={float(algorithm.cfg.cps_raw_rms):.5f}",
        flush=True,
    )
    print(
        f"[STYLE_PRIOR] discriminator=standard_mlp hidden={list(algorithm.cfg.style_prior.hidden_dims)} "
        f"BCE=True GP={algorithm.cfg.style_prior.grad_penalty} replay={algorithm.cfg.style_prior.replay_size} "
        f"EMA=False policy_conditioning=False motion_end_terminal=True "
        f"replay_mode=complete_window_stratified "
        f"fcamp_schema={FCAMP_CHECKPOINT_CONTRACT['fcamp_schema_version']} "
        f"action_contract={FCAMP_CHECKPOINT_CONTRACT['action_contract']} "
        f"reset_contract={FCAMP_CHECKPOINT_CONTRACT['reset_contract']} "
        f"validation_contract={FCAMP_CHECKPOINT_CONTRACT['validation_contract']} "
        f"actor_mean_contract={FCAMP_CHECKPOINT_CONTRACT['actor_mean_contract']} "
        f"actor_norm_contract={FCAMP_CHECKPOINT_CONTRACT['actor_normalizer_contract']} "
        f"cps_metric_contract={FCAMP_CHECKPOINT_CONTRACT['cps_metric_contract']} "
        f"ppo_contract={FCAMP_CHECKPOINT_CONTRACT['ppo_contract']} "
        f"phase0_trajectory_attempt_stream={algorithm.cfg.streams.phase0_fraction:.2f}",
        flush=True,
    )
