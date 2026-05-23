from __future__ import annotations

from collections import deque
import time
from pathlib import Path

import torch

from net.ppo import GaussianActorCritic
from ..checkpoint import CheckpointMixin
from ..env_factory import make_mimic_env
from ..env_state import EnvStateMixin
from ..logging import LoggingMixin
from .config import OfficialPPOConfig
from ..returns import compute_gae_returns
from ..validation import ValidationMixin


class OfficialPPOTrainer(ValidationMixin, CheckpointMixin, LoggingMixin, EnvStateMixin):
    def __init__(self, simulation_app, cfg: OfficialPPOConfig):
        self.simulation_app = simulation_app
        self.cfg = cfg
        self.start_update = 1
        self.checkpoint_dir = Path(cfg.checkpoint_dir).expanduser().resolve() if cfg.checkpoint_dir else None

        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        self.env = make_mimic_env(cfg)
        if self.env.action_dim != cfg.action_dim:
            raise ValueError(f"Expected env action_dim {self.env.action_dim}, got {cfg.action_dim}")

        policy_obs_dim = cfg.policy_obs_dim if cfg.policy_obs_dim > 0 else self.env.observation_dim
        critic_obs_dim = cfg.critic_obs_dim if cfg.critic_obs_dim > 0 else self.env.critic_observation_dim
        self.policy = GaussianActorCritic(
            obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=cfg.action_dim,
            actor_hidden_dims=cfg.actor_hidden_dims,
            critic_hidden_dims=cfg.critic_hidden_dims,
            activation=cfg.activation,
            init_noise_std=cfg.init_noise_std,
        ).to(self.env.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.learning_rate = float(cfg.lr)
        self._init_train_episode_stats()
        self.current_observation = self._reset_training_envs()

        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if cfg.resume:
            self._load_checkpoint(Path(cfg.resume).expanduser().resolve())

    @property
    def critic(self):
        return self.policy.critic

    def _reset_training_envs(self) -> torch.Tensor:
        phase_indices = self.env.sample_phase_indices(self.env.num_envs, horizon=self.cfg.num_steps_per_env)
        return self.env.reset(phase_indices=phase_indices)

    def _init_train_episode_stats(self) -> None:
        self._train_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.env.device)
        self._train_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.env.device)
        self._train_reward_buffer: deque[float] = deque(maxlen=100)
        self._train_length_buffer: deque[float] = deque(maxlen=100)
        self._train_completed_episodes = 0

    def _record_train_episode_stats(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        self._train_reward_sum += rewards
        self._train_episode_length += 1.0
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return
        self._train_reward_buffer.extend(self._train_reward_sum.index_select(0, done_ids).detach().cpu().tolist())
        self._train_length_buffer.extend(self._train_episode_length.index_select(0, done_ids).detach().cpu().tolist())
        self._train_completed_episodes += int(done_ids.numel())
        self._train_reward_sum[done_ids] = 0.0
        self._train_episode_length[done_ids] = 0.0

    def _deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        action = self.policy.act_inference(obs)
        return action.view(obs.shape[0], 1, self.cfg.action_dim)

    def _compute_gae_returns(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        last_values: torch.Tensor,
        timeouts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_gae_returns(
            rewards,
            dones,
            values,
            last_values,
            gamma=float(self.cfg.discount_gamma),
            lam=float(self.cfg.gae_lambda),
            timeouts=timeouts,
            normalize_advantage=True,
        )

    def _record_first_done(
        self,
        *,
        done_mask: torch.Tensor,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        infos_list: list[dict[str, torch.Tensor]],
        step_index: int,
        first_done_step: torch.Tensor,
        first_done_phase: torch.Tensor,
        first_done_anchor_pos: torch.Tensor,
        first_done_anchor_ori: torch.Tensor,
        first_done_ee_body: torch.Tensor,
        first_done_timeout: torch.Tensor,
    ) -> None:
        done_steps = terminations | truncations
        new_done_any = (~done_mask) & done_steps.any(dim=1)
        if not bool(new_done_any.any()):
            return

        first_offsets = done_steps.to(dtype=torch.long).argmax(dim=1)
        env_ids = new_done_any.nonzero(as_tuple=False).squeeze(-1)
        first_done_step[env_ids] = int(step_index)
        if infos_list:
            if all("termination_phase_steps" in step_info for step_info in infos_list):
                phase_by_step = torch.stack(
                    [step_info["termination_phase_steps"] for step_info in infos_list],
                    dim=1,
                )
                first_done_phase[env_ids] = phase_by_step[env_ids, first_offsets[env_ids]]
            for key, target in (
                ("anchor_pos_bad", first_done_anchor_pos),
                ("anchor_ori_bad", first_done_anchor_ori),
                ("ee_body_bad", first_done_ee_body),
                ("time_out", first_done_timeout),
            ):
                if not all(key in step_info.get("done_terms", {}) for step_info in infos_list):
                    continue
                cause_by_step = torch.stack(
                    [step_info["done_terms"][key] for step_info in infos_list],
                    dim=1,
                )
                target[env_ids] = cause_by_step[env_ids, first_offsets[env_ids]].bool()
        else:
            if hasattr(self.env, "phase_steps"):
                first_done_phase[env_ids] = self.env.phase_steps[env_ids]
            first_done_ee_body[env_ids] = terminations.any(dim=1)[env_ids]
            first_done_timeout[env_ids] = truncations.any(dim=1)[env_ids]

    def _collect_rollout(self, current_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        total_envs = self.env.num_envs
        steps = int(self.cfg.num_steps_per_env)
        obs_t = current_obs
        critic_obs_t = self.env.get_critic_observation()
        collection_start_phases = self.env.phase_steps.detach().clone()
        first_done_step = torch.full((total_envs,), steps, dtype=torch.long, device=self.env.device)
        first_done_phase = torch.full((total_envs,), -1, dtype=torch.long, device=self.env.device)
        first_done_anchor_pos = torch.zeros(total_envs, dtype=torch.bool, device=self.env.device)
        first_done_anchor_ori = torch.zeros(total_envs, dtype=torch.bool, device=self.env.device)
        first_done_ee_body = torch.zeros(total_envs, dtype=torch.bool, device=self.env.device)
        first_done_timeout = torch.zeros(total_envs, dtype=torch.bool, device=self.env.device)
        ever_done = torch.zeros(total_envs, dtype=torch.bool, device=self.env.device)

        rollout_obs = []
        rollout_critic_obs = []
        rollout_actions = []
        rollout_rewards = []
        rollout_dones = []
        rollout_timeouts = []
        rollout_values = []
        rollout_log_probs = []
        rollout_means = []
        rollout_sigmas = []
        rollout_valid = []
        rollout_infos = []
        metric_rollout_info_items: list[tuple[dict[str, torch.Tensor], torch.Tensor]] = []

        action_abs_max_all = 0.0
        for step_index in range(steps):
            active_before_step = ~ever_done
            with torch.no_grad():
                sample = self.policy.act(obs_t, critic_obs_t)
            action_t = sample["actions"]
            action_abs_max_all = max(action_abs_max_all, float(action_t.abs().max().item()))
            next_obs_t, reward_t, done_t, info_t = self.env.step(
                action_t,
                auto_reset=True,
                reset_horizon=max(1, steps - step_index),
            )
            timeout_t = info_t["done_terms"]["time_out"].bool()
            termination_t = done_t & (~timeout_t)

            self._record_first_done(
                done_mask=ever_done,
                terminations=termination_t[:, None],
                truncations=timeout_t[:, None],
                infos_list=[info_t],
                step_index=step_index,
                first_done_step=first_done_step,
                first_done_phase=first_done_phase,
                first_done_anchor_pos=first_done_anchor_pos,
                first_done_anchor_ori=first_done_anchor_ori,
                first_done_ee_body=first_done_ee_body,
                first_done_timeout=first_done_timeout,
            )
            ever_done |= done_t

            rollout_obs.append(obs_t.detach())
            rollout_critic_obs.append(critic_obs_t.detach())
            rollout_actions.append(action_t.detach())
            rollout_rewards.append(reward_t.detach())
            rollout_dones.append(done_t.detach())
            rollout_timeouts.append(timeout_t.detach())
            rollout_values.append(sample["values"].detach())
            rollout_log_probs.append(sample["log_probs"].detach())
            rollout_means.append(sample["mean"].detach())
            rollout_sigmas.append(sample["sigma"].detach())
            rollout_valid.append(active_before_step.detach())
            rollout_infos.append(info_t)
            metric_rollout_info_items.append((info_t, active_before_step.detach()))
            self._record_train_episode_stats(reward_t.detach(), done_t.detach())

            obs_t = next_obs_t
            critic_obs_t = self.env.get_critic_observation()

        with torch.no_grad():
            last_values = self.policy.evaluate(critic_obs_t).detach()

        obs = torch.stack(rollout_obs, dim=1)
        critic_obs = torch.stack(rollout_critic_obs, dim=1)
        actions = torch.stack(rollout_actions, dim=1)
        rewards = torch.stack(rollout_rewards, dim=1)
        dones = torch.stack(rollout_dones, dim=1)
        timeouts = torch.stack(rollout_timeouts, dim=1)
        values = torch.stack(rollout_values, dim=1)
        log_probs = torch.stack(rollout_log_probs, dim=1)
        means = torch.stack(rollout_means, dim=1)
        sigmas = torch.stack(rollout_sigmas, dim=1)
        returns, advantages = self._compute_gae_returns(rewards, dones, values, last_values, timeouts=timeouts)
        return {
            "obs": obs,
            "critic_obs": critic_obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "timeouts": timeouts,
            "values": values,
            "old_log_probs": log_probs,
            "old_means": means,
            "old_sigmas": sigmas,
            "returns": returns,
            "advantages": advantages,
            "last_values": last_values,
            "valid_mask": torch.stack(rollout_valid, dim=1),
            "first_done_chunk": first_done_step,
            "first_done_phase": first_done_phase,
            "first_done_anchor_pos": first_done_anchor_pos,
            "first_done_anchor_ori": first_done_anchor_ori,
            "first_done_ee_body": first_done_ee_body,
            "first_done_timeout": first_done_timeout,
            "metric_infos_list": rollout_infos[:1],
            "metric_rollout_info_items": metric_rollout_info_items,
            "metric_actions": actions[:, :1, :].detach(),
            "metric_actions_first": actions[:, :1, :].detach(),
            "metric_actions_last": actions[:, -1:, :].detach(),
            "metric_chunk_return": rewards[:, 0].detach(),
            "metric_chunk_return_first": rewards[:, 0].detach(),
            "metric_chunk_return_last": rewards[:, -1].detach(),
            "metric_action_abs_max_all": action_abs_max_all,
            "collection_start_phases": collection_start_phases,
            "next_observation": obs_t.detach().clone(),
            "next_critic_observation": critic_obs_t.detach().clone(),
        }

    def _policy_mini_batch_size(self, sample_count: int) -> int:
        return max(1, sample_count // max(1, int(self.cfg.num_mini_batches)))

    def _update_adaptive_learning_rate(self, kl_mean: torch.Tensor) -> None:
        if self.cfg.schedule != "adaptive":
            return
        desired_kl = float(self.cfg.desired_kl)
        if desired_kl <= 0.0:
            return
        kl_value = float(kl_mean.item())
        if kl_value > 2.0 * desired_kl:
            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
        elif kl_value < 0.5 * desired_kl and kl_value > 0.0:
            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def _policy_update(self, data: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = data["obs"].reshape(-1, data["obs"].shape[-1])
        critic_obs = data["critic_obs"].reshape(-1, data["critic_obs"].shape[-1])
        actions = data["actions"].reshape(-1, data["actions"].shape[-1])
        old_log_probs = data["old_log_probs"].reshape(-1)
        old_values = data["values"].reshape(-1)
        returns = data["returns"].reshape(-1)
        advantages = data["advantages"].reshape(-1)
        old_means = data["old_means"].reshape(-1, data["old_means"].shape[-1])
        old_sigmas = data["old_sigmas"].reshape(-1, data["old_sigmas"].shape[-1])
        sample_count = obs.shape[0]
        mini_batch_size = self._policy_mini_batch_size(sample_count)
        permutation_count = max(mini_batch_size, (sample_count // mini_batch_size) * mini_batch_size)
        indices = torch.randperm(permutation_count, device=obs.device)

        totals = {
            "value_loss": 0.0,
            "surrogate_loss": 0.0,
            "entropy": 0.0,
            "ratio": 0.0,
            "ratio_min": float("inf"),
            "ratio_max": 0.0,
            "clip_frac": 0.0,
            "kl": 0.0,
            "logprob_delta_abs": 0.0,
            "old_log_prob": 0.0,
            "new_log_prob": 0.0,
            "grad_norm": 0.0,
        }
        update_count = 0
        probe_count = min(256, sample_count)
        with torch.no_grad():
            probe_obs = obs[:probe_count].detach().clone()
            probe_critic_obs = critic_obs[:probe_count].detach().clone()
            probe_actions = actions[:probe_count].detach().clone()
            probe_old_log_probs = old_log_probs[:probe_count].detach().clone()
            probe_action_before = self.policy.act_inference(probe_obs).detach().clone()
            params_before = [param.detach().clone() for param in self.policy.parameters()]
        for _ in range(int(self.cfg.policy_epochs)):
            for start in range(0, permutation_count, mini_batch_size):
                mb = indices[start : start + mini_batch_size]
                mb_obs = obs[mb]
                mb_critic_obs = critic_obs[mb]
                mb_actions = actions[mb]
                mb_old_log_probs = old_log_probs[mb]
                mb_old_values = old_values[mb]
                mb_returns = returns[mb]
                mb_advantages = advantages[mb]
                mb_old_means = old_means[mb]
                mb_old_sigmas = old_sigmas[mb]

                new_log_probs, value_pred, entropy, mean, sigma = self.policy.evaluate_actions(
                    mb_obs,
                    mb_critic_obs,
                    mb_actions,
                )
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma / mb_old_sigmas + 1.0e-5)
                        + (mb_old_sigmas.square() + (mb_old_means - mean).square())
                        / (2.0 * sigma.square())
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = kl.mean()
                self._update_adaptive_learning_rate(kl_mean)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                log_ratio = new_log_probs - mb_old_log_probs
                surrogate = -mb_advantages * ratio
                surrogate_clipped = -mb_advantages * torch.clamp(
                    ratio,
                    1.0 - float(self.cfg.clip_range),
                    1.0 + float(self.cfg.clip_range),
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                if self.cfg.use_clipped_value_loss:
                    value_clipped = mb_old_values + (value_pred - mb_old_values).clamp(
                        -float(self.cfg.clip_range),
                        float(self.cfg.clip_range),
                    )
                    value_losses = (value_pred - mb_returns).square()
                    value_losses_clipped = (value_clipped - mb_returns).square()
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (mb_returns - value_pred).square().mean()
                loss = (
                    surrogate_loss
                    + float(self.cfg.value_loss_coef) * value_loss
                    - float(self.cfg.entropy_coef) * entropy.mean()
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                totals["value_loss"] += float(value_loss.item())
                totals["surrogate_loss"] += float(surrogate_loss.item())
                totals["entropy"] += float(entropy.mean().item())
                totals["ratio"] += float(ratio.mean().item())
                totals["ratio_min"] = min(totals["ratio_min"], float(ratio.min().item()))
                totals["ratio_max"] = max(totals["ratio_max"], float(ratio.max().item()))
                totals["clip_frac"] += float((torch.abs(ratio - 1.0) > self.cfg.clip_range).float().mean().item())
                totals["kl"] += float(kl_mean.item())
                totals["logprob_delta_abs"] += float(log_ratio.abs().mean().item())
                totals["old_log_prob"] += float(mb_old_log_probs.mean().item())
                totals["new_log_prob"] += float(new_log_probs.mean().item())
                totals["grad_norm"] += float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                update_count += 1

        denom = max(1, update_count)
        with torch.no_grad():
            post_new_log_probs, _, _, _, _ = self.policy.evaluate_actions(
                probe_obs,
                probe_critic_obs,
                probe_actions,
            )
            post_log_ratio = post_new_log_probs - probe_old_log_probs
            post_ratio_tensor = torch.exp(post_log_ratio)
            post_ratio = float(post_ratio_tensor.mean().item())
            post_ratio_min = float(post_ratio_tensor.min().item())
            post_ratio_max = float(post_ratio_tensor.max().item())
            post_clip_frac = float((torch.abs(post_ratio_tensor - 1.0) > self.cfg.clip_range).float().mean().item())
            post_logprob_delta_abs = float(post_log_ratio.abs().mean().item())
            post_kl_loss = float((0.5 * post_log_ratio.square()).mean().item())
            probe_action_after = self.policy.act_inference(probe_obs)
            action_delta = torch.mean(torch.abs(probe_action_after - probe_action_before))
            param_delta_sq = torch.tensor(0.0, device=obs.device)
            param_count = 0
            for param, before in zip(self.policy.parameters(), params_before, strict=True):
                delta = param.detach() - before
                param_delta_sq = param_delta_sq + torch.sum(delta * delta)
                param_count += delta.numel()
            param_rms_delta = torch.sqrt(param_delta_sq / max(param_count, 1))
        policy_loss = totals["surrogate_loss"] / denom
        value_loss = totals["value_loss"] / denom
        entropy = totals["entropy"] / denom
        total_loss = policy_loss + float(self.cfg.value_loss_coef) * value_loss - float(self.cfg.entropy_coef) * entropy
        return {
            "policy/loss": total_loss,
            "policy/policy_loss": policy_loss,
            "policy/value_loss": value_loss,
            "policy/entropy": entropy,
            "policy/kl_loss": totals["kl"] / denom,
            "policy/clip_frac": totals["clip_frac"] / denom,
            "policy/ratio": totals["ratio"] / denom,
            "policy/ratio_min": totals["ratio_min"] if totals["ratio_min"] != float("inf") else 0.0,
            "policy/ratio_max": totals["ratio_max"],
            "policy/logprob_delta_abs": totals["logprob_delta_abs"] / denom,
            "policy/old_log_prob": totals["old_log_prob"] / denom,
            "policy/new_log_prob": totals["new_log_prob"] / denom,
            "policy/step_ratio": totals["ratio"] / denom,
            "policy/step_ratio_min": totals["ratio_min"] if totals["ratio_min"] != float("inf") else 0.0,
            "policy/step_ratio_max": totals["ratio_max"],
            "policy/step_clip_frac": totals["clip_frac"] / denom,
            "policy/step_logprob_delta_abs": totals["logprob_delta_abs"] / denom,
            "policy/step_kl_loss": totals["kl"] / denom,
            "policy/post_ratio": post_ratio,
            "policy/post_ratio_min": post_ratio_min,
            "policy/post_ratio_max": post_ratio_max,
            "policy/post_clip_frac": post_clip_frac,
            "policy/post_logprob_delta_abs": post_logprob_delta_abs,
            "policy/post_kl_loss": post_kl_loss,
            "policy/grad_norm": totals["grad_norm"] / denom,
            "policy/action_delta": float(action_delta.item()),
            "policy/param_rms_delta": float(param_rms_delta.item()),
            "policy/sample_count": float(sample_count),
            "policy/advantage_abs_mean": float(advantages.abs().mean().item()),
            "policy/effective_mini_batch_size": float(mini_batch_size),
            "policy/micro_batch_size": float(mini_batch_size),
            "policy/micro_batches": float(update_count),
            "policy/optimizer_steps": float(update_count),
            "policy/lr": float(self.optimizer.param_groups[0]["lr"]),
            "policy/critic_lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    def _build_metrics(
        self,
        data: dict[str, torch.Tensor],
        update_metrics: dict[str, float],
        collect_time: float,
        update_time: float,
    ) -> dict[str, float]:
        rewards = data["rewards"]
        steps = rewards.shape[1]
        raw_returns = rewards.sum(dim=1)
        valid_mask = data.get(
            "valid_mask",
            torch.ones_like(rewards, dtype=torch.bool, device=self.env.device),
        )
        valid_f = valid_mask.to(dtype=rewards.dtype)
        objective_chunk_rewards = rewards * valid_f
        objective_returns = objective_chunk_rewards.sum(dim=1)
        live_steps = valid_f.sum(dim=1)
        live_steps_flat = live_steps.flatten()
        metric_actions = data["metric_actions"]
        act_abs_tensor = metric_actions.abs()
        act_abs = act_abs_tensor.mean(dim=(0, 1))
        collection_start_phases = data["collection_start_phases"]
        first_done_step = data.get("first_done_chunk")
        first_done_phase = data.get("first_done_phase")
        first_done_anchor_pos = data.get("first_done_anchor_pos")
        first_done_anchor_ori = data.get("first_done_anchor_ori")
        first_done_ee_body = data.get("first_done_ee_body")
        first_done_timeout = data.get("first_done_timeout")
        failed_mask = first_done_phase >= 0 if first_done_phase is not None else None
        failure_phase_values = first_done_phase[failed_mask] if failed_mask is not None and bool(failed_mask.any()) else None
        failure_step_values = first_done_step[failed_mask] if first_done_step is not None and failed_mask is not None and bool(failed_mask.any()) else None
        if (
            first_done_phase is not None
            and failed_mask is not None
            and collection_start_phases is not None
            and bool(failed_mask.any())
        ):
            failure_relative_phase_values = first_done_phase[failed_mask] - collection_start_phases[failed_mask]
        else:
            failure_relative_phase_values = None
        total_live_steps = live_steps.sum().clamp(min=1.0)
        reward_per_live_step = objective_returns.sum() / total_live_steps
        raw_reward_per_step = raw_returns.sum() / max(rewards.numel(), 1)
        official_scale_reward = reward_per_live_step * self.cfg.max_episode_steps
        per_episode_objective_scale = objective_returns / live_steps.clamp(min=1.0) * self.cfg.max_episode_steps
        chunk_objective_means = objective_chunk_rewards.mean(dim=0)
        chunk_raw_means = rewards.mean(dim=0)
        mid_step_index = min(max(steps // 2, 0), steps - 1)
        metrics = {
            **update_metrics,
            "algo/name": "ppo",
            "group/reward_mean": float(official_scale_reward.item()),
            "group/reward_std": float(per_episode_objective_scale.std().item()),
            "group/reward_raw_mean": float(raw_returns.mean().item()),
            "group/reward_raw_std": float(raw_returns.std().item()),
            "group/reward_raw_min": float(raw_returns.min().item()),
            "group/reward_raw_max": float(raw_returns.max().item()),
            "group/score_reward_mean": float((objective_returns / live_steps.clamp(min=1.0)).mean().item()),
            "group/objective_reward_raw_mean": float(objective_returns.mean().item()),
            "group/objective_reward_raw_std": float(objective_returns.std().item()),
            "group/objective_reward_raw_min": float(objective_returns.min().item()),
            "group/objective_reward_raw_max": float(objective_returns.max().item()),
            "group/advantage_abs_mean": float(data["advantages"].abs().mean().item()),
            "group/grpo_advantage_abs_mean": 0.0,
            "rollout/valid_frac": float(valid_mask.float().mean().item()),
            "rollout/return_mean": float(objective_returns.mean().item()),
            "rollout/return_std": float(objective_returns.std().item()),
            "rollout/raw_return_mean": float(raw_returns.mean().item()),
            "rollout/raw_return_std": float(raw_returns.std().item()),
            "rollout/chunk_return_mean": float(data["metric_chunk_return"].mean().item()),
            "rollout/chunk_return_std": float(data["metric_chunk_return"].std().item()),
            "rollout/chunk_return_first_mean": float(data["metric_chunk_return_first"].mean().item()),
            "rollout/chunk_return_last_mean": float(data["metric_chunk_return_last"].mean().item()),
            "rollout/chunk_objective_first_mean": float(chunk_objective_means[0].item()),
            "rollout/chunk_objective_mid_mean": float(chunk_objective_means[mid_step_index].item()),
            "rollout/chunk_objective_last_mean": float(chunk_objective_means[-1].item()),
            "rollout/chunk_raw_first_mean": float(chunk_raw_means[0].item()),
            "rollout/chunk_raw_mid_mean": float(chunk_raw_means[mid_step_index].item()),
            "rollout/chunk_raw_last_mean": float(chunk_raw_means[-1].item()),
            "rollout/live_steps_mean": float(live_steps.mean().item()),
            "rollout/live_steps_min": float(live_steps.min().item()),
            "rollout/live_steps_p50": float(torch.quantile(live_steps_flat, 0.50).item()),
            "rollout/live_steps_p95": float(torch.quantile(live_steps_flat, 0.95).item()),
            "rollout/live_steps_max": float(live_steps.max().item()),
            "rollout/success_frac": (
                float((~failed_mask).float().mean().item()) if failed_mask is not None else 1.0
            ),
            "rollout/first_failure_chunk_mean": (
                float(failure_step_values.float().mean().item()) if failure_step_values is not None else float("nan")
            ),
            "rollout/first_failure_chunk_min": (
                float(failure_step_values.min().item()) if failure_step_values is not None else float("nan")
            ),
            "rollout/first_failure_chunk_max": (
                float(failure_step_values.max().item()) if failure_step_values is not None else float("nan")
            ),
            "rollout/first_failure_phase_mean": (
                float(failure_phase_values.float().mean().item()) if failure_phase_values is not None else float("nan")
            ),
            "rollout/first_failure_phase_min": (
                float(failure_phase_values.min().item()) if failure_phase_values is not None else float("nan")
            ),
            "rollout/first_failure_phase_max": (
                float(failure_phase_values.max().item()) if failure_phase_values is not None else float("nan")
            ),
            "rollout/first_failure_relative_phase_mean": (
                float(failure_relative_phase_values.float().mean().item())
                if failure_relative_phase_values is not None
                else float("nan")
            ),
            "rollout/first_failure_anchor_pos_frac": (
                float((first_done_anchor_pos & failed_mask).float().mean().item())
                if first_done_anchor_pos is not None and failed_mask is not None
                else 0.0
            ),
            "rollout/first_failure_anchor_ori_frac": (
                float((first_done_anchor_ori & failed_mask).float().mean().item())
                if first_done_anchor_ori is not None and failed_mask is not None
                else 0.0
            ),
            "rollout/first_failure_ee_body_frac": (
                float((first_done_ee_body & failed_mask).float().mean().item())
                if first_done_ee_body is not None and failed_mask is not None
                else 0.0
            ),
            "rollout/first_failure_timeout_frac": (
                float((first_done_timeout & failed_mask).float().mean().item())
                if first_done_timeout is not None and failed_mask is not None
                else 0.0
            ),
            "rollout/reward_per_live_step": float(reward_per_live_step.item()),
            "rollout/reward_per_live_second": float((reward_per_live_step / self.env.dt).item()),
            "rollout/raw_reward_per_live_step": float(raw_reward_per_step.item()),
            "rollout/max_episode_return_projection": float(official_scale_reward.item()),
            "phase/start_mean": float(collection_start_phases.float().mean().item()),
            "phase/start_min": float(collection_start_phases.min().item()),
            "phase/start_max": float(collection_start_phases.max().item()),
            "timing/collect_s": collect_time,
            "timing/update_s": update_time,
            "act/abs_mean": float(act_abs_tensor.mean().item()),
            "act/abs_max": float(act_abs_tensor.max().item()),
            "act/abs_max_all": float(data.get("metric_action_abs_max_all", 0.0)),
            "act/abs_p95": float(torch.quantile(act_abs_tensor.flatten(), 0.95).item()),
            "act/abs_p99": float(torch.quantile(act_abs_tensor.flatten(), 0.99).item()),
            "act/legs_abs": float(act_abs[[0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]].mean().item()),
            "act/waist_abs": float(act_abs[[2, 5, 8]].mean().item()),
            "act/arms_abs": float(act_abs[[11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]].mean().item()),
            "latent/final_abs_mean": 0.0,
            "latent/final_abs_max": 0.0,
            "act/first_abs_mean": float(data["metric_actions_first"].abs().mean().item()),
            "act/last_abs_mean": float(data["metric_actions_last"].abs().mean().item()),
            "act/l_wrist_roll": float(act_abs[23].item()),
            "act/r_wrist_roll": float(act_abs[24].item()),
            "act/l_wrist_pitch": float(act_abs[25].item()),
            "act/r_wrist_pitch": float(act_abs[26].item()),
            "act/l_wrist_yaw": float(act_abs[27].item()),
            "act/r_wrist_yaw": float(act_abs[28].item()),
            "act/l_elbow": float(act_abs[21].item()),
            "act/r_elbow": float(act_abs[22].item()),
        }
        if self._train_reward_buffer:
            metrics["train/mean_reward"] = float(sum(self._train_reward_buffer) / len(self._train_reward_buffer))
            metrics["train/mean_episode_length"] = float(sum(self._train_length_buffer) / len(self._train_length_buffer))
        else:
            metrics["train/mean_reward"] = float("nan")
            metrics["train/mean_episode_length"] = float("nan")
        metrics["train/recent_episode_count"] = float(len(self._train_reward_buffer))
        metrics["train/completed_episodes"] = float(self._train_completed_episodes)
        done_union: dict[str, torch.Tensor] = {}
        info_count = max(len(data["metric_infos_list"]), 1)
        for step_info in data["metric_infos_list"]:
            for key, value in step_info["reward_terms"].items():
                metrics[f"reward/{key}_mean"] = metrics.get(f"reward/{key}_mean", 0.0) + float(value.mean().item()) / info_count
            for key, value in step_info["done_terms"].items():
                done_union[key] = value.bool().clone() if key not in done_union else done_union[key] | value.bool()
        for key, union_mask in done_union.items():
            metrics[f"done/{key}_frac"] = float(union_mask.float().mean().item())

        rollout_info_items = data.get("metric_rollout_info_items", [])
        rollout_done_union: dict[str, torch.Tensor] = {}
        rollout_reward_sums: dict[str, float] = {}
        rollout_weight_sum = 0.0
        for step_info, valid_mask_for_step in rollout_info_items:
            valid_mask_f = valid_mask_for_step.float()
            valid_weight = float(valid_mask_f.sum().item())
            if valid_weight <= 0.0:
                continue
            rollout_weight_sum += valid_weight
            for key, value in step_info["reward_terms"].items():
                rollout_reward_sums[key] = rollout_reward_sums.get(key, 0.0) + float((value * valid_mask_f).sum().item())
            for key, value in step_info["done_terms"].items():
                masked_done = value.bool() & valid_mask_for_step.bool()
                if key not in rollout_done_union:
                    rollout_done_union[key] = masked_done.clone()
                else:
                    rollout_done_union[key] |= masked_done
        if rollout_weight_sum > 0.0:
            for key, value_sum in rollout_reward_sums.items():
                metrics[f"reward_rollout/{key}_mean"] = value_sum / rollout_weight_sum
            for key, union_mask in rollout_done_union.items():
                metrics[f"done_rollout/{key}_frac"] = float(union_mask.float().mean().item())

        reward_weights = {
            "joint_acc": -2.5e-7,
            "joint_torque": -1.0e-5,
            "action_rate": -1.0e-1,
            "joint_limit": -10.0,
            "anchor_pos_reward": 2.0,
            "anchor_ori_reward": 2.0,
            "body_pos_reward": 1.0,
            "body_ori_reward": 1.0,
            "body_lin_vel_reward": 1.0,
            "body_ang_vel_reward": 1.0,
            "undesired_contacts": -0.1,
        }
        weighted_positive = 0.0
        weighted_penalty = 0.0
        for reward_name, weight in reward_weights.items():
            raw_key = f"reward/{reward_name}_mean"
            if raw_key not in metrics:
                continue
            contribution = weight * metrics[raw_key] * self.env.dt
            metrics[f"reward_weighted/{reward_name}"] = contribution
            if contribution >= 0.0:
                weighted_positive += contribution
            else:
                weighted_penalty += contribution
        metrics["reward_weighted/positive"] = weighted_positive
        metrics["reward_weighted/penalty"] = weighted_penalty
        metrics["reward_weighted/total"] = weighted_positive + weighted_penalty
        return metrics

    def train(self) -> None:
        print("[INFO] Starting official-style PPO training", flush=True)
        print(
            f"[INFO] num_envs={self.cfg.num_envs} num_steps_per_env={self.cfg.num_steps_per_env} "
            f"actor_hidden_dims={list(self.cfg.actor_hidden_dims)} "
            f"critic_hidden_dims={list(self.cfg.critic_hidden_dims)} "
            f"activation={self.cfg.activation} init_noise_std={self.cfg.init_noise_std} "
            f"clip={self.cfg.clip_range} lr={self.cfg.lr} schedule={self.cfg.schedule} "
            f"desired_kl={self.cfg.desired_kl} action_clip=null",
            flush=True,
        )
        if self.checkpoint_dir is not None:
            print(f"[INFO] checkpoint_dir={self.checkpoint_dir}", flush=True)
        if self.cfg.resume:
            print(f"[INFO] resumed_from={self.cfg.resume}", flush=True)

        for update_idx in range(self.start_update, self.cfg.max_updates + 1):
            if not self.simulation_app.is_running():
                break
            t0 = time.perf_counter()
            current_obs = self.current_observation
            data = self._collect_rollout(current_obs)
            self.current_observation = data["next_observation"]
            collect_time = time.perf_counter() - t0
            t1 = time.perf_counter()
            update_metrics = self._policy_update(data)
            update_time = time.perf_counter() - t1
            metrics = self._build_metrics(data, update_metrics, collect_time, update_time)

            should_log = update_idx % self.cfg.log_every == 0
            if should_log:
                self._log_update(update_idx, metrics)

            if self.cfg.validation_every > 0 and update_idx % self.cfg.validation_every == 0:
                fixed_seed = self.cfg.validation_fixed_seed if self.cfg.validation_fixed_seed >= 0 else None
                print(
                    f"[VALIDATION_START] update={update_idx} "
                    f"max_steps={self._validation_max_steps()} "
                    f"envs={self.cfg.num_envs}",
                    flush=True,
                )
                validation_t0 = time.perf_counter()
                metrics.update(self.run_validation_rollout())
                if fixed_seed is not None:
                    fixed_metrics = self.run_validation_rollout(fixed_seed=fixed_seed)
                    for key, value in fixed_metrics.items():
                        metrics[key.replace("validation/", "val_fixed/")] = value
                metrics["timing/validation_s"] = time.perf_counter() - validation_t0
                print(
                    f"[VALIDATION_DONE] update={update_idx} "
                    f"time={metrics['timing/validation_s']:.3f}s",
                    flush=True,
                )
                self._log_validation_metrics(metrics)

            if self.checkpoint_dir is not None and (
                update_idx == self.cfg.max_updates
                or (self.cfg.save_every > 0 and update_idx % self.cfg.save_every == 0)
            ):
                self._save_checkpoint(update_idx, metrics)
            if self._target_validation_reached(metrics):
                if self.checkpoint_dir is not None:
                    self._save_checkpoint(update_idx, metrics, filename=self.cfg.success_checkpoint_name)
                print(
                    f"[SUCCESS] validation reached {self.cfg.target_validation_steps} steps; "
                    f"saved {self.cfg.success_checkpoint_name}",
                    flush=True,
                )
                break

        self.current_observation = self._reset_training_envs()
        print("[INFO] PPO training finished.", flush=True)
