from __future__ import annotations

import time

import torch

from components.imitation.style_reward import style_reward_statistics
from components.rollout.fcamp_contract import FCAMP_CHECKPOINT_CONTRACT
from components.rollout.training_streams import CURRICULUM_STREAM, PHASE0_STREAM


def _masked_stats(prefix: str, values: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
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



class FCAMPDiagnosticsMixin:
    @torch.no_grad()
    def _stream_rollout_metrics(self, rollout: dict) -> dict[str, float]:
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
        configured_weights = {
            "phase0": float(self.cfg.streams.phase0_fraction),
            "curriculum": 1.0
            - float(self.cfg.streams.phase0_fraction),
        }
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
                    f"{prefix}/configured_objective_weight": (
                        configured_weights[name]
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
                    f"{prefix}/sampler_failure_eligible": float(
                        stream_id == CURRICULUM_STREAM
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
                    rollout["amp_reward"][
                        stream_amp_valid
                    ].mean().item()
                )
            else:
                metrics[f"{prefix}/amp_logit_mean"] = -1.0
                metrics[f"{prefix}/amp_reward_mean"] = -1.0
        return metrics

    def update(self, rollout: dict, collect_time: float) -> dict:
        update_start = time.perf_counter()
        # Global primitive steps are used as the replay age clock.
        update_idx = self._fcamp_update_idx
        (
            actor_metrics,
            critic_metrics,
            disc_metrics,
            reward_metrics,
            actor_time,
            critic_time,
            disc_time,
        ) = self._optimize_rollout_snapshot(rollout, update_idx)

        valid = rollout["valid"]
        amp_valid = rollout["amp_valid"]
        credit_valid = rollout["credit_valid"]
        metrics: dict[str, float] = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        metrics.update(disc_metrics)
        metrics.update(reward_metrics)
        metrics.update(self._stream_rollout_metrics(rollout))
        metrics.update(self.phase0_attempts.metrics())
        metrics.update(_masked_stats("reward/amp", rollout["amp_reward"], amp_valid))
        metrics.update(
            _masked_stats(
                "reward/amp_credit",
                rollout["amp_reward_credit"],
                amp_valid,
            )
        )
        metrics.update(
            _masked_stats("credit/amp_adv", rollout["amp_advantages"], credit_valid)
        )
        metrics.update(
            _masked_stats("credit/actor_adv", rollout["advantages"], credit_valid)
        )
        metrics.update(_masked_stats("critic/value", rollout["values"], credit_valid))
        metrics.update(
            _masked_stats("critic/target", rollout["value_targets"], credit_valid)
        )
        value = rollout["values"][credit_valid].detach().float()
        target = rollout["value_targets"][credit_valid].detach().float()
        residual = target - value
        target_variance = target.var(unbiased=False)
        explained_variance = (
            0.0
            if float(target_variance.item()) <= 1.0e-12
            else float(
                (
                    1.0
                    - residual.var(unbiased=False) / target_variance
                ).item()
            )
        )
        metrics.update(
            {
                "critic/value_mae": float(residual.abs().mean().item()),
                "critic/value_rmse": float(
                    residual.square().mean().sqrt().item()
                ),
                "critic/explained_variance": explained_variance,
            }
        )
        metrics.update(
            style_reward_statistics(
                rollout["amp_logits"][amp_valid],
                rollout["amp_reward"][amp_valid],
                scale=float(self.cfg.style_prior.reward_scale),
                minimum_one_minus_prob=float(self.cfg.style_prior.reward_epsilon),
                prefix="amp_reward",
            )
        )
        for frame_idx in range(self.horizon_h):
            frame_valid = credit_valid[..., frame_idx]
            metrics.update(
                _masked_stats(
                    f"credit/frame_{frame_idx}_amp_adv",
                    rollout["amp_advantages"][..., frame_idx],
                    frame_valid,
                )
            )
            metrics.update(
                _masked_stats(
                    f"credit/frame_{frame_idx}_actor_adv",
                    rollout["advantages"][..., frame_idx],
                    frame_valid,
                )
            )
        metrics.update(self.disc_normalizer.statistics())
        metrics.update(self.imitation_history.statistics())
        valid_count = int(valid.sum().item())
        amp_valid_count = int(amp_valid.sum().item())
        intervention_count = int(rollout["intervention_edge"].sum().item())
        credit_gap_count = valid_count - amp_valid_count
        dirty_window_count = float(rollout["dirty_window_excluded_count"])
        metrics.update(
            {
                "rollout/valid_fraction": float(valid.float().mean().item()),
                "rollout/amp_valid_fraction": float(amp_valid.float().mean().item()),
                "rollout/credit_gap_count": float(credit_gap_count),
                "rollout/credit_gap_fraction_alive": float(
                    credit_gap_count / max(valid_count, 1)
                ),
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
                "act/abs_max": float(rollout["action_abs_max"]),
                "act/policy_bound_violation_max": float(
                    rollout["action_bound_violation_max"]
                ),
                "disc_contract/fk_alignment_abs_max": float(
                    rollout["fk_alignment_abs_max"]
                ),
                "disc_contract/fk_alignment_abs_mean": float(
                    rollout["fk_alignment_abs_mean"]
                ),
                "disc_contract/dirty_window_excluded_count": float(
                    dirty_window_count
                ),
                "disc_contract/replay_dirty_insert_count": float(
                    metrics.get("disc_replay/dirty_insert_count", -1.0)
                ),
                "intervention/edge_count": float(
                    intervention_count
                ),
                "disc_window/policy_latest_root_xy_abs_max": float(
                    rollout["window_latest_root_xy_abs_max"]
                ),
                "disc_window/policy_root_xy_abs_mean": float(
                    rollout["window_root_xy_abs_mean"]
                ),
                "train/mean_reward": float(
                    sum(self._train_reward_buffer) / len(self._train_reward_buffer)
                    if self._train_reward_buffer
                    else float("nan")
                ),
                "train/mean_episode_length": float(
                    sum(self._train_length_buffer) / len(self._train_length_buffer)
                    if self._train_length_buffer
                    else float("nan")
                ),
                "timing/collect_s": float(collect_time),
                "timing/actor_update_s": float(actor_time),
                "timing/critic_update_s": float(critic_time),
                "timing/disc_update_s": float(disc_time),
                "timing/update_s": float(time.perf_counter() - update_start),
                "system/primitive_steps": float(
                    self.warmup_env_transitions
                    + update_idx
                    * int(self.cfg.rollout_env_steps)
                    * self.env.num_envs
                ),
                "system/cuda_peak_allocated_gib": float(
                    torch.cuda.max_memory_allocated(self.env.device) / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                ),
            }
        )
        self._add_sampler_metrics(metrics)
        finite_parameters = all(
            bool(torch.isfinite(parameter).all())
            for module in (self._policy, self.critic, self.discriminator)
            for parameter in module.parameters()
        )
        metrics["system/parameters_finite"] = float(finite_parameters)
        if not finite_parameters:
            raise FloatingPointError("FCAMP detected non-finite trainable parameters")
        return metrics

    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        print(
            f"[UPDATE] {update_idx}/{max_updates} "
            f"amp={metrics.get('reward/amp/mean', float('nan')):.5f} "
            f"done={metrics.get('rollout/done_fraction', float('nan')):.5f} "
            f"ep_len={metrics.get('train/mean_episode_length', float('nan')):.2f}",
            flush=True,
        )
        print(
            f"[FCAMP] policy={metrics.get('fcamp/policy_loss', float('nan')):.5f} "
            f"kl={metrics.get('fcamp/kl', float('nan')):.6f} "
            f"ratio={metrics.get('fcamp/ratio', float('nan')):.4f} "
            f"clip={metrics.get('fcamp/clip_fraction', float('nan')):.4f} "
            f"grad={metrics.get('fcamp/actor_grad_norm', float('nan')):.4f} "
            f"lr={metrics.get('fcamp/actor_lr', float('nan')):.6f}",
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
            f"[AMP_CREDIT] "
            f"raw={metrics.get('reward/amp/mean', float('nan')):.5f} "
            f"credit={metrics.get('reward/amp_credit/mean', float('nan')):.5f} "
            f"amp_raw_std={metrics.get('credit/amp_adv/std', float('nan')):.5f} "
            f"actor_std={metrics.get('credit/actor_adv/std', float('nan')):.5f}",
            flush=True,
        )
        print(
            f"[CRITIC] loss={metrics.get('critic/flow_loss', float('nan')):.5f} "
            f"V={metrics.get('critic/value/mean', float('nan')):.4f} "
            f"target={metrics.get('critic/target/mean', float('nan')):.4f} "
            f"rmse={metrics.get('critic/value_rmse', float('nan')):.4f} "
            f"ev={metrics.get('critic/explained_variance', float('nan')):.4f} "
            f"grad={metrics.get('critic/grad_norm', float('nan')):.4f} "
            f"grad_clip={metrics.get('critic/grad_clip_fraction', float('nan')):.4f}",
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
            f"policy_action_violation={metrics.get('act/policy_bound_violation_max', float('nan')):.3e} "
            f"dirty_excluded={metrics.get('disc_contract/dirty_window_excluded_count', 0.0):.0f} "
            f"credit_gap={metrics.get('rollout/credit_gap_count', 0.0):.0f} "
            f"replay_dirty={metrics.get('disc_contract/replay_dirty_insert_count', -1.0):.0f} "
            f"push_edges={metrics.get('intervention/edge_count', 0.0):.0f} "
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

    def log_banner(self) -> None:
        print(
            "[METHOD] name=fcamp actor=causal_flow_cps prior=temporal_discriminator "
            "reward=pure_amp credit=causal_frame critic=scalar_flow",
            flush=True,
        )
        print(
            f"[ARCH] task={self.env.task.name} actor_obs={self.actor_obs_dim} "
            f"actor_input=self_state,last_action phase=False target=False "
            f"discriminator_in_actor=False "
            f"prefix_context={self.prefix_context_dim} action_dim={self.num_act} "
            f"H={self.horizon_h} W_D={self.imitation_history_steps} "
            f"imitation_frame={self.imitation_frame_dim} imitation_window={self.imitation_window_dim}",
            flush=True,
        )
        print(
            "[CREDIT] head=amp credit=causal_frame advantage_norm=global "
            "ratio_mode=joint_path task_weight=0 disc_weight=1 amp_dt=True",
            flush=True,
        )
        print(
            f"[STYLE_PRIOR] discriminator=standard_mlp hidden={list(self.cfg.style_prior.hidden_dims)} "
            f"BCE=True GP={self.cfg.style_prior.grad_penalty} replay={self.cfg.style_prior.replay_size} "
            f"EMA=False policy_conditioning=False motion_end_terminal=True "
            f"replay_mode=complete_window_stratified "
            f"fcamp_schema={FCAMP_CHECKPOINT_CONTRACT['fcamp_schema_version']} "
            f"action_contract={FCAMP_CHECKPOINT_CONTRACT['action_contract']} "
            f"actor_observation_contract={FCAMP_CHECKPOINT_CONTRACT['actor_observation_contract']} "
            f"reward_contract={FCAMP_CHECKPOINT_CONTRACT['reward_contract']} "
            f"critic_contract={FCAMP_CHECKPOINT_CONTRACT['critic_contract']} "
            f"reset_contract={FCAMP_CHECKPOINT_CONTRACT['reset_contract']} "
            f"expert_velocity_contract={FCAMP_CHECKPOINT_CONTRACT['expert_velocity_contract']} "
            f"validation_contract={FCAMP_CHECKPOINT_CONTRACT['validation_contract']} "
            f"ppo_contract={FCAMP_CHECKPOINT_CONTRACT['ppo_contract']} "
            f"phase0_trajectory_attempt_stream={self.cfg.streams.phase0_fraction:.2f}",
            flush=True,
        )
