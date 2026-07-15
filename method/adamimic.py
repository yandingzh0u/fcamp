from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path

import torch
from torch import nn
from muon import SingleDeviceMuonWithAuxAdam

from components.optim.kl_scheduler import adaptive_lr_from_kl
from method.base import Algorithm
from models.adamimic_policy import AdaMimicActorCritic
from models.mlp_actor_critic import EmpiricalNormalization


class AdaMimic(Algorithm):
    """AdaMimic-style adaptive-time tracking policy.

    This is a native implementation for the refactored project layout.  It keeps
    AdaMimic's two-level action distribution and optional stage-2 residual actor,
    while using the current G1MimicEnv reward.  The final policy action is
    passed to the environment as a variable reference-time step.
    """
    uses_reference_dt = True

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        self.actor_obs_dim = env.observation_dim
        self.critic_obs_dim = env.critic_observation_dim
        self.control_action_dim = env.action_dim
        self.policy_action_dim = self.control_action_dim + 1
        self.rollout_steps = int(cfg.rollout_env_steps)
        if self.rollout_steps <= 1:
            raise ValueError("AdaMimic requires rollout_env_steps > 1")

        self.empirical_normalization = bool(cfg.empirical_normalization)
        if self.empirical_normalization:
            self.actor_obs_normalizer: nn.Module = EmpiricalNormalization(self.actor_obs_dim, env.device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(self.critic_obs_dim, env.device)
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        self._model = AdaMimicActorCritic(
            actor_obs_dim=self.actor_obs_dim,
            critic_obs_dim=self.critic_obs_dim,
            control_action_dim=self.control_action_dim,
            actor_hidden_dims=tuple(cfg.actor_hidden_dims),
            critic_hidden_dims=tuple(cfg.critic_hidden_dims),
            activation=cfg.activation,
            init_noise_std=float(cfg.init_noise_std),
            infer_keyframe_time=bool(cfg.infer_keyframe_time),
            actor_time_scale_range=tuple(cfg.actor_time_scale_range),
            fixed_dt=float(cfg.fixed_dt),
            time_min_std=float(cfg.time_min_std),
            residual_delta=bool(cfg.residual_delta),
            residual_time_threshold=float(cfg.residual_time_threshold),
        ).to(env.device)
        if cfg.residual_delta and cfg.checkpoint_path:
            self._load_stage1_weights(Path(cfg.checkpoint_path))
        if cfg.residual_delta and cfg.freeze_base:
            self._model.freeze_base_actor()

        self.learning_rate = float(cfg.policy_lr)
        self.optimizer_impl = self._build_official_optimizer()
        self.min_lr = 5.0e-4
        self.max_lr = 1.0e-2
        self._policy_module = nn.ModuleDict(
            {
                "model": self._model,
                "actor_obs_normalizer": self.actor_obs_normalizer,
                "critic_obs_normalizer": self.critic_obs_normalizer,
            }
        )
        self._init_train_episode_stats()

    def _build_official_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.cfg
        weight_decay = 0.01
        if cfg.residual_delta and cfg.freeze_base:
            modules: list[nn.Module | None] = [
                self._model.actor_time,
                self._model.critic_high,
                self._model.actor_delta,
                self._model.critic_delta,
            ]
        elif cfg.residual_delta:
            modules = [
                self._model.actor,
                self._model.actor_time,
                self._model.critic_high,
                self._model.actor_delta,
                self._model.critic_delta,
            ]
        else:
            modules = [self._model.actor, self._model.critic_low]

        params: list[nn.Parameter] = [self._model.std]
        for module in modules:
            if module is not None:
                params.extend(module.parameters())
        params = [p for p in params if p.requires_grad]
        hidden_weights = [p for p in params if p.ndim >= 2]
        hidden_gains_biases = [p for p in params if p.ndim < 2]
        return SingleDeviceMuonWithAuxAdam(
            [
                {
                    "params": hidden_gains_biases,
                    "use_muon": False,
                    "lr": self.learning_rate,
                    "betas": (0.9, 0.95),
                    "eps": 1.0e-10,
                    "weight_decay": weight_decay,
                },
                {
                    "params": hidden_weights,
                    "use_muon": True,
                    "lr": self.learning_rate,
                    "weight_decay": weight_decay,
                },
            ]
        )

    @property
    def policy(self) -> nn.Module:
        return self._policy_module

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer_impl

    @property
    def horizon(self) -> int:
        return 1

    def extra_checkpoint_state(self) -> dict:
        return {
            "learning_rate": float(self.learning_rate),
            "actor_obs_normalizer": self.actor_obs_normalizer.state_dict() if self.empirical_normalization else None,
            "critic_obs_normalizer": self.critic_obs_normalizer.state_dict() if self.empirical_normalization else None,
            "adamimic_schema_version": 1,
        }

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        if reset_optimizer:
            self.learning_rate = float(self.cfg.policy_lr)
        elif payload:
            self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
        for group in self.optimizer_impl.param_groups:
            group["lr"] = self.learning_rate
        if not payload:
            return
        if self.empirical_normalization:
            if payload.get("actor_obs_normalizer") is not None:
                self.actor_obs_normalizer.load_state_dict(payload["actor_obs_normalizer"])
            if payload.get("critic_obs_normalizer") is not None:
                self.critic_obs_normalizer.load_state_dict(payload["critic_obs_normalizer"])

    def _load_stage1_weights(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"AdaMimic stage1 checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu")
        state = payload.get("policy", payload.get("model_state_dict", payload))
        if any(key.startswith("model.") for key in state):
            state = {key.removeprefix("model."): value for key, value in state.items() if key.startswith("model.")}
        current = self._model.state_dict()
        filtered = {
            key: value
            for key, value in state.items()
            if (
                key in current
                and current[key].shape == value.shape
                and not key.startswith("actor_delta.")
                and not key.startswith("critic_delta.")
                and key != "critic_high.0.weight"
                and not key.startswith("critic_high.")
            )
        }
        current.update(filtered)
        self._model.load_state_dict(current, strict=False)
        print(f"[ADAMIMIC] loaded_stage1={path} tensors={len(filtered)}", flush=True)

    def _norm_actor(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.actor_obs_normalizer(obs, update=update) if self.empirical_normalization else obs

    def _norm_critic(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.critic_obs_normalizer(obs, update=update) if self.empirical_normalization else obs

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        actor_obs = self._norm_actor(obs, update=False)
        return self._model.act_inference(actor_obs)[:, :-1]

    def deployment_actions(self, obs: torch.Tensor) -> torch.Tensor:
        actor_obs = self._norm_actor(obs, update=False)
        return self._model.act_inference(actor_obs)

    def initial_reset(self) -> torch.Tensor:
        phase0 = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.env.device)
        obs = self.env.reset(phase_indices=phase0)
        zero_action = torch.zeros(self.env.num_envs, self.env.action_dim, device=self.env.device)
        with torch.no_grad():
            obs, _, _, _ = self.env.step(
                zero_action,
                auto_reset=False,
                reference_dt=torch.full((self.env.num_envs,), float(self.cfg.fixed_dt), device=self.env.device),
            )
        if bool(self.cfg.init_at_random_ep_len) and self.env.max_episode_steps > 0:
            self.env.episode_steps = torch.randint_like(
                self.env.episode_steps,
                high=int(self.env.max_episode_steps),
            )
        self._obs = obs
        self._critic_obs = self.env.get_critic_observation()
        return obs

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        del update_idx
        return self._obs

    def _init_train_episode_stats(self) -> None:
        env = self.env
        self._train_reward_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._train_episode_length = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._train_reward_buffer: deque[float] = deque(maxlen=100)
        self._train_length_buffer: deque[float] = deque(maxlen=100)
        self._train_completed_episodes = 0

    def _record_episode_stats(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        self._train_reward_sum += rewards.to(dtype=torch.float32)
        self._train_episode_length += 1.0
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return
        self._train_reward_buffer.extend(self._train_reward_sum.index_select(0, done_ids).detach().cpu().tolist())
        self._train_length_buffer.extend(self._train_episode_length.index_select(0, done_ids).detach().cpu().tolist())
        self._train_completed_episodes += int(done_ids.numel())
        self._train_reward_sum[done_ids] = 0.0
        self._train_episode_length[done_ids] = 0.0

    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        device = env.device
        steps = self.rollout_steps
        n_envs = env.num_envs

        actor_obs_buf = torch.zeros(steps, n_envs, self.actor_obs_dim, device=device)
        next_actor_obs_buf = torch.zeros_like(actor_obs_buf)
        critic_obs_buf = torch.zeros(steps, n_envs, self.critic_obs_dim, device=device)
        next_critic_obs_buf = torch.zeros_like(critic_obs_buf)
        actions_buf = torch.zeros(steps, n_envs, self.policy_action_dim, device=device)
        values_low_buf = torch.zeros(steps, n_envs, device=device)
        values_high_buf = torch.zeros(steps, n_envs, device=device)
        logp_low_buf = torch.zeros(steps, n_envs, device=device)
        logp_high_buf = torch.zeros(steps, n_envs, device=device)
        action_mean_buf = torch.zeros_like(actions_buf)
        action_std_buf = torch.zeros_like(actions_buf)
        reward_low_buf = torch.zeros(steps, n_envs, device=device)
        reward_high_buf = torch.zeros(steps, n_envs, device=device)
        done_buf = torch.zeros(steps, n_envs, dtype=torch.bool, device=device)
        timeout_buf = torch.zeros_like(done_buf)
        failure_buf = torch.zeros_like(done_buf)
        motion_complete_buf = torch.zeros_like(done_buf)
        done_terms_union: dict[str, torch.Tensor] = {}
        first_infos: list[dict] = []
        rollout_info_items: list[tuple[dict, torch.Tensor]] = []
        start_phases = env.phase_steps.detach().clone() if hasattr(env, "phase_steps") else torch.zeros(n_envs, dtype=torch.long, device=device)
        first_done_step = torch.full((n_envs,), steps, dtype=torch.long, device=device)
        first_done_phase = torch.full((n_envs,), -1, dtype=torch.long, device=device)
        ever_done = torch.zeros(n_envs, dtype=torch.bool, device=device)
        action_abs_max = 0.0
        time_action_sum = torch.zeros((), device=device)
        time_action_sq_sum = torch.zeros((), device=device)
        time_action_count = 0
        frame_delta_sum = torch.zeros((), device=device)
        frame_delta_sq_sum = torch.zeros((), device=device)

        obs = current_obs
        critic_obs = self._critic_obs
        with torch.no_grad():
            for step_idx in range(steps):
                actor_obs_n = self._norm_actor(obs)
                critic_obs_n = self._norm_critic(critic_obs)
                action_full = self._model.act(actor_obs_n)
                low_logp, high_logp = self._model.get_actions_log_prob(action_full)
                action_time = action_full[:, -1:]
                value_low = self._model.evaluate_low(critic_obs_n, action_time)
                value_high = self._model.evaluate_high(critic_obs_n)
                control_action = action_full[:, :-1]
                next_obs, reward, done, info = env.step(
                    control_action,
                    auto_reset=True,
                    reference_dt=action_time.squeeze(-1),
                )
                next_critic_obs = env.get_critic_observation()
                next_actor_obs_n = self._norm_actor(next_obs, update=False)
                next_critic_obs_n = self._norm_critic(next_critic_obs, update=False)

                if step_idx == 0:
                    first_infos.append(info)
                actor_obs_buf[step_idx] = actor_obs_n
                next_actor_obs_buf[step_idx] = next_actor_obs_n
                critic_obs_buf[step_idx] = critic_obs_n
                next_critic_obs_buf[step_idx] = next_critic_obs_n
                actions_buf[step_idx] = action_full
                values_low_buf[step_idx] = value_low
                values_high_buf[step_idx] = value_high
                logp_low_buf[step_idx] = low_logp
                logp_high_buf[step_idx] = high_logp
                action_mean_buf[step_idx] = self._model.action_mean
                action_std_buf[step_idx] = self._model.action_std
                reward_low_buf[step_idx] = reward
                reward_high_buf[step_idx] = reward * float(self.cfg.time_reward_scale)
                done_bool = done.bool()
                done_buf[step_idx] = done_bool

                done_terms = info["done_terms"]
                timeout = done_terms["time_out"].bool()
                motion_complete = done_terms.get("motion_complete")
                motion_complete = motion_complete.bool() if torch.is_tensor(motion_complete) else torch.zeros_like(timeout)
                failure = (
                    done_terms["anchor_pos_bad"].bool()
                    | done_terms["anchor_ori_bad"].bool()
                    | done_terms["ee_body_bad"].bool()
                ) & (~timeout) & (~motion_complete)
                timeout_buf[step_idx] = done_bool & timeout
                failure_buf[step_idx] = done_bool & failure
                motion_complete_buf[step_idx] = done_bool & motion_complete
                for key, value in done_terms.items():
                    b = value.bool()
                    done_terms_union[key] = b.clone() if key not in done_terms_union else done_terms_union[key] | b

                newly_done = (~ever_done) & done_bool
                if bool(newly_done.any()):
                    ids = newly_done.nonzero(as_tuple=False).squeeze(-1)
                    first_done_step[ids] = step_idx + 1
                    phase = info.get("termination_phase_steps")
                    if torch.is_tensor(phase):
                        first_done_phase[ids] = phase.long()[ids]
                    ever_done[ids] = True

                rollout_info_items.append((info, torch.ones(n_envs, dtype=torch.bool, device=device)))
                self._record_episode_stats(reward, done_bool)
                action_abs_max = max(action_abs_max, float(control_action.abs().max().item()))
                time_action_sum += action_time.sum()
                time_action_sq_sum += (action_time * action_time).sum()
                time_action_count += int(action_time.numel())
                frame_delta = info["reference_frame_delta"].to(device=device)
                frame_delta_sum += frame_delta.sum()
                frame_delta_sq_sum += (frame_delta * frame_delta).sum()
                obs = next_obs
                critic_obs = next_critic_obs

            final_actor_obs = self._norm_actor(obs, update=False)
            final_critic_obs = self._norm_critic(critic_obs, update=False)
            final_time = self._model.act_inference(final_actor_obs)[:, -1:]
            last_value_low = self._model.evaluate_low(final_critic_obs, final_time)
            last_value_high = self._model.evaluate_high(final_critic_obs)

        self._obs = obs
        self._critic_obs = critic_obs
        returns_low, adv_low = self._compute_returns(
            rewards=reward_low_buf,
            values=values_low_buf,
            last_value=last_value_low,
            dones=done_buf,
            timeout_mask=timeout_buf,
            gamma=float(self.cfg.discount_gamma),
            timeout_bootstrap=bool(self.cfg.use_timeout_bootstrap),
        )
        returns_high, adv_high = self._compute_returns(
            rewards=reward_high_buf,
            values=values_high_buf,
            last_value=last_value_high,
            dones=done_buf,
            timeout_mask=timeout_buf,
            gamma=float(self.cfg.time_discount_gamma),
            timeout_bootstrap=False,
        )
        time_mean = time_action_sum / max(time_action_count, 1)
        time_var = time_action_sq_sum / max(time_action_count, 1) - time_mean.square()
        frame_delta_mean = frame_delta_sum / max(time_action_count, 1)
        frame_delta_var = frame_delta_sq_sum / max(time_action_count, 1) - frame_delta_mean.square()
        return {
            "actor_obs": actor_obs_buf,
            "next_actor_obs": next_actor_obs_buf,
            "critic_obs": critic_obs_buf,
            "next_critic_obs": next_critic_obs_buf,
            "actions": actions_buf,
            "values_low": values_low_buf,
            "values_high": values_high_buf,
            "returns_low": returns_low,
            "returns_high": returns_high,
            "advantages_low": self._normalize_advantages(adv_low, torch.ones_like(done_buf, dtype=torch.bool)),
            "advantages_high": self._normalize_advantages(adv_high, torch.ones_like(done_buf, dtype=torch.bool)),
            "old_logp_low": logp_low_buf,
            "old_logp_high": logp_high_buf,
            "old_mu": action_mean_buf,
            "old_sigma": action_std_buf,
            "reward_low": reward_low_buf,
            "reward_high": reward_high_buf,
            "done": done_buf,
            "timeout": timeout_buf,
            "failure": failure_buf,
            "motion_complete": motion_complete_buf,
            "done_terms_union": done_terms_union,
            "rollout_info_items": rollout_info_items,
            "first_infos": first_infos,
            "first_done_step": first_done_step,
            "first_done_phase": first_done_phase,
            "collection_start_phases": start_phases,
            "time_action_mean": float(time_mean.item()),
            "time_action_std": float(torch.sqrt(time_var.clamp(min=0.0)).item()),
            "reference_frame_delta_mean": float(frame_delta_mean.item()),
            "reference_frame_delta_std": float(torch.sqrt(frame_delta_var.clamp(min=0.0)).item()),
            "action_abs_max": action_abs_max,
            "next_observation": obs,
        }

    def _compute_returns(
        self,
        *,
        rewards: torch.Tensor,
        values: torch.Tensor,
        last_value: torch.Tensor,
        dones: torch.Tensor,
        timeout_mask: torch.Tensor,
        gamma: float,
        timeout_bootstrap: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        returns = torch.zeros_like(values)
        advantages = torch.zeros_like(values)
        gae = torch.zeros_like(last_value)
        lam = float(self.cfg.gae_lambda)
        for step_idx in range(rewards.shape[0] - 1, -1, -1):
            next_value = last_value if step_idx == rewards.shape[0] - 1 else values[step_idx + 1]
            nonterminal = (~dones[step_idx]).to(dtype=values.dtype)
            reward = rewards[step_idx]
            if timeout_bootstrap:
                reward = reward + gamma * timeout_mask[step_idx].to(dtype=values.dtype) * next_value
            delta = reward + gamma * nonterminal * next_value - values[step_idx]
            gae = delta + gamma * lam * nonterminal * gae
            advantages[step_idx] = gae
            returns[step_idx] = gae + values[step_idx]
        return returns, advantages

    @staticmethod
    def _normalize_advantages(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = advantages[mask]
        if valid.numel() == 0:
            return advantages * 0.0
        return ((advantages - valid.mean()) / (valid.std(unbiased=False) + 1.0e-8)) * mask.to(dtype=advantages.dtype)

    def _mini_batch_size(self, sample_count: int) -> int:
        return max(1, math.ceil(sample_count / max(1, int(self.cfg.num_mini_batches))))

    def _micro_batch_size(self, batch_size: int) -> int:
        if int(self.cfg.micro_batch_size) <= 0:
            return max(1, batch_size)
        return max(1, min(batch_size, int(self.cfg.micro_batch_size)))

    def _update_lr_from_kl(self, observed_kl: float) -> None:
        desired_kl = float(self.cfg.desired_kl)
        if desired_kl <= 0.0:
            return
        self.learning_rate, _ = adaptive_lr_from_kl(
            raw_kl=observed_kl,
            kl_units=1,
            target_per_step=desired_kl,
            lr=self.learning_rate,
            min_lr=self.min_lr,
            max_lr=self.max_lr,
        )
        for group in self.optimizer_impl.param_groups:
            group["lr"] = self.learning_rate

    @staticmethod
    def _gaussian_kl(
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        new_mu: torch.Tensor,
        new_sigma: torch.Tensor,
    ) -> torch.Tensor:
        return (
            torch.log(new_sigma / old_sigma + 1.0e-8)
            + (old_sigma.square() + (old_mu - new_mu).square()) / (2.0 * new_sigma.square().clamp(min=1.0e-8))
            - 0.5
        ).sum(dim=-1)

    def update(self, rollout: dict, collect_time: float) -> dict:
        start_time = time.perf_counter()
        device = self.env.device
        steps, n_envs = rollout["actions"].shape[:2]
        train_steps = steps - 1
        if train_steps <= 0:
            raise RuntimeError("AdaMimic rollout must contain at least two transitions")
        batch_size = train_steps * n_envs
        actor_obs = rollout["actor_obs"][:train_steps].reshape(batch_size, self.actor_obs_dim)
        next_actor_obs = rollout["actor_obs"][1:steps].reshape(batch_size, self.actor_obs_dim)
        critic_obs = rollout["critic_obs"][:train_steps].reshape(batch_size, self.critic_obs_dim)
        next_critic_obs = rollout["critic_obs"][1:steps].reshape(batch_size, self.critic_obs_dim)
        actions = rollout["actions"][:train_steps].reshape(batch_size, self.policy_action_dim)
        old_logp_low = rollout["old_logp_low"][:train_steps].reshape(batch_size)
        old_logp_high = rollout["old_logp_high"][:train_steps].reshape(batch_size)
        old_mu = rollout["old_mu"][:train_steps].reshape(batch_size, self.policy_action_dim)
        old_sigma = rollout["old_sigma"][:train_steps].reshape(batch_size, self.policy_action_dim)
        old_values_low = rollout["values_low"][:train_steps].reshape(batch_size)
        old_values_high = rollout["values_high"][:train_steps].reshape(batch_size)
        returns_low = rollout["returns_low"][:train_steps].reshape(batch_size)
        returns_high = rollout["returns_high"][:train_steps].reshape(batch_size)
        adv_low = rollout["advantages_low"][:train_steps].reshape(batch_size)
        adv_high = rollout["advantages_high"][:train_steps].reshape(batch_size)
        cont = (~rollout["done"][:train_steps]).to(dtype=actor_obs.dtype).reshape(batch_size, 1)

        mini_batch_size = self._mini_batch_size(batch_size)
        clip_low = 1.0 - float(self.cfg.clip_range)
        clip_high = 1.0 + float(self.cfg.clip_range)
        value_coef = float(self.cfg.value_loss_coef)
        entropy_coef = float(self.cfg.entropy_coef)
        train_time = bool(self.cfg.train_time)

        probe_count = min(128, batch_size)
        with torch.no_grad():
            probe_before = self._model.act_inference(actor_obs[:probe_count])[:, :-1]
            params_before = [p.detach().clone() for p in self._model.parameters() if p.requires_grad]

        totals = {
            "loss": 0.0,
            "policy_low": 0.0,
            "policy_high": 0.0,
            "value_low": 0.0,
            "value_high": 0.0,
            "smooth": 0.0,
            "entropy": 0.0,
            "ratio_low": 0.0,
            "ratio_high": 0.0,
            "clip_low": 0.0,
            "clip_high": 0.0,
            "kl": 0.0,
            "grad_norm": 0.0,
        }
        optimizer_steps = 0
        for _epoch in range(int(self.cfg.policy_epochs)):
            perm = torch.randperm(batch_size, device=device)
            for mb_start in range(0, batch_size, mini_batch_size):
                idx = perm[mb_start : mb_start + mini_batch_size]
                if idx.numel() == 0:
                    continue
                mb_size = int(idx.numel())
                micro = self._micro_batch_size(mb_size)
                self.optimizer_impl.zero_grad(set_to_none=True)
                mb_totals = {key: 0.0 for key in totals if key != "grad_norm"}
                for micro_start in range(0, mb_size, micro):
                    sub = idx[micro_start : min(micro_start + micro, mb_size)]
                    weight = float(sub.numel()) / float(mb_size)
                    self._model.act(actor_obs[sub])
                    logp_low, logp_high = self._model.get_actions_log_prob(actions[sub])
                    ratio_low = torch.exp(logp_low - old_logp_low[sub])
                    loss_low = torch.maximum(
                        -adv_low[sub] * ratio_low,
                        -adv_low[sub] * ratio_low.clamp(clip_low, clip_high),
                    ).mean()

                    action_time = actions[sub, -1:]
                    value_low = self._model.evaluate_low(critic_obs[sub], action_time)
                    if bool(self.cfg.use_clipped_value_loss):
                        value_low_clipped = old_values_low[sub] + (value_low - old_values_low[sub]).clamp(
                            -float(self.cfg.clip_range), float(self.cfg.clip_range)
                        )
                        value_loss_low = torch.maximum(
                            (value_low - returns_low[sub]).square(),
                            (value_low_clipped - returns_low[sub]).square(),
                        ).mean()
                    else:
                        value_loss_low = (value_low - returns_low[sub]).square().mean()

                    policy_loss = loss_low
                    value_loss = value_loss_low
                    loss_high = torch.zeros((), device=device)
                    value_loss_high = torch.zeros((), device=device)
                    ratio_high = torch.ones_like(ratio_low)
                    if train_time:
                        ratio_high = torch.exp(logp_high - old_logp_high[sub])
                        loss_high = torch.maximum(
                            -adv_high[sub] * ratio_high,
                            -adv_high[sub] * ratio_high.clamp(clip_low, clip_high),
                        ).mean()
                        value_high = self._model.evaluate_high(critic_obs[sub])
                        if bool(self.cfg.use_clipped_value_loss):
                            value_high_clipped = old_values_high[sub] + (value_high - old_values_high[sub]).clamp(
                                -float(self.cfg.clip_range), float(self.cfg.clip_range)
                            )
                            value_loss_high = torch.maximum(
                                (value_high - returns_high[sub]).square(),
                                (value_high_clipped - returns_high[sub]).square(),
                            ).mean()
                        else:
                            value_loss_high = (value_high - returns_high[sub]).square().mean()
                        policy_loss = policy_loss + loss_high
                        value_loss = value_loss + value_loss_high

                    smooth_loss = torch.zeros((), device=device)
                    if bool(self.cfg.use_smooth):
                        lower = float(self.cfg.smoothness_lower_bound)
                        upper = float(self.cfg.smoothness_upper_bound)
                        epsilon = lower / (upper - lower)
                        policy_smooth_coef = upper * epsilon
                        value_smooth_coef = float(self.cfg.value_smoothness_coef) * policy_smooth_coef
                        mix = cont[sub] * (torch.rand(sub.numel(), 1, device=device, dtype=actor_obs.dtype) - 0.5) * 2.0
                        mixed_actor_obs = actor_obs[sub] + mix * (next_actor_obs[sub] - actor_obs[sub])
                        mixed_critic_obs = critic_obs[sub] + mix * (next_critic_obs[sub] - critic_obs[sub])
                        current_mean = self._model.action_mean[:, :-1]
                        mixed_mean = self._model.act_inference(mixed_actor_obs)[:, :-1]
                        policy_smooth = (current_mean - mixed_mean).square().sum(dim=-1).mean()
                        mixed_value = self._model.evaluate_low(mixed_critic_obs, action_time)
                        value_smooth = (value_low - mixed_value).square().mean()
                        smooth_loss = policy_smooth_coef * policy_smooth + value_smooth_coef * value_smooth

                    entropy = self._model.entropy.mean()
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy + smooth_loss
                    (loss * weight).backward()

                    with torch.no_grad():
                        new_mu = self._model.action_mean
                        new_sigma = self._model.action_std
                        kl = self._gaussian_kl(old_mu[sub], old_sigma[sub], new_mu, new_sigma).mean()
                        clip_low_frac = ((ratio_low < clip_low) | (ratio_low > clip_high)).float().mean()
                        clip_high_frac = ((ratio_high < clip_low) | (ratio_high > clip_high)).float().mean()
                        mb_totals["loss"] += float(loss.item()) * weight
                        mb_totals["policy_low"] += float(loss_low.item()) * weight
                        mb_totals["policy_high"] += float(loss_high.item()) * weight
                        mb_totals["value_low"] += float(value_loss_low.item()) * weight
                        mb_totals["value_high"] += float(value_loss_high.item()) * weight
                        mb_totals["smooth"] += float(smooth_loss.item()) * weight
                        mb_totals["entropy"] += float(entropy.item()) * weight
                        mb_totals["ratio_low"] += float(ratio_low.mean().item()) * weight
                        mb_totals["ratio_high"] += float(ratio_high.mean().item()) * weight
                        mb_totals["clip_low"] += float(clip_low_frac.item()) * weight
                        mb_totals["clip_high"] += float(clip_high_frac.item()) * weight
                        mb_totals["kl"] += float(kl.item()) * weight

                self._update_lr_from_kl(mb_totals["kl"])
                grad_norm = nn.utils.clip_grad_norm_(self._model.parameters(), float(self.cfg.max_grad_norm))
                self.optimizer_impl.step()
                for key in mb_totals:
                    totals[key] += mb_totals[key]
                totals["grad_norm"] += float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                optimizer_steps += 1

        update_time = time.perf_counter() - start_time
        denom = max(optimizer_steps, 1)
        with torch.no_grad():
            probe_after = self._model.act_inference(actor_obs[:probe_count])[:, :-1]
            param_delta_sq = torch.zeros((), device=device)
            param_count = 0
            for param, before in zip([p for p in self._model.parameters() if p.requires_grad], params_before, strict=True):
                delta = param.detach() - before
                param_delta_sq += (delta * delta).sum()
                param_count += delta.numel()
            param_rms = torch.sqrt(param_delta_sq / max(param_count, 1))

        update_metrics = {
            "adamimic/loss": totals["loss"] / denom,
            "adamimic/policy_low_loss": totals["policy_low"] / denom,
            "adamimic/policy_high_loss": totals["policy_high"] / denom,
            "adamimic/value_low_loss": totals["value_low"] / denom,
            "adamimic/value_high_loss": totals["value_high"] / denom,
            "adamimic/smooth_loss": totals["smooth"] / denom,
            "adamimic/entropy": totals["entropy"] / denom,
            "adamimic/ratio_low": totals["ratio_low"] / denom,
            "adamimic/ratio_high": totals["ratio_high"] / denom,
            "adamimic/clip_low": totals["clip_low"] / denom,
            "adamimic/clip_high": totals["clip_high"] / denom,
            "adamimic/kl": totals["kl"] / denom,
            "adamimic/grad_norm": totals["grad_norm"] / denom,
            "adamimic/lr": self.learning_rate,
            "adamimic/sample_count": float(batch_size),
            "adamimic/effective_mini_batch_size": float(mini_batch_size),
            "adamimic/optimizer_steps": float(optimizer_steps),
            "policy/action_delta": float((probe_after - probe_before).abs().mean().item()),
            "policy/param_rms_delta": float(param_rms.item()),
        }
        return self._build_metrics(rollout, update_metrics, collect_time, update_time)

    def _build_metrics(self, rollout: dict, update_metrics: dict, collect_time: float, update_time: float) -> dict:
        actions = rollout["actions"][..., :-1]
        rewards = rollout["reward_low"]
        dones = rollout["done"]
        failures = rollout["failure"]
        timeouts = rollout["timeout"]
        motion_complete = rollout["motion_complete"]
        first_done_step = rollout["first_done_step"]
        failed_first = (first_done_step < self.rollout_steps) & ~rollout["timeout"].any(dim=0) & ~rollout["motion_complete"].any(dim=0)
        reward_step_mean = float(rewards.mean().item())
        returns = rewards.sum(dim=0)
        act_abs = actions.abs()
        metrics = {
            **update_metrics,
            "method/adamimic": 1.0,
            "rollout/reward_step_mean": reward_step_mean,
            "rollout/return_mean": float(returns.mean().item()),
            "rollout/return_std": float(returns.std(unbiased=False).item()),
            "rollout/done_frac": float(dones.float().mean().item()),
            "rollout/failure_frac": float(failures.float().mean().item()),
            "rollout/timeout_frac": float(timeouts.float().mean().item()),
            "rollout/motion_complete_frac": float(motion_complete.float().mean().item()),
            "rollout/success_frac": float((~failed_first).float().mean().item()),
            "rollout/first_done_step_mean": float(first_done_step.float().mean().item()),
            "phase/start_mean": float(rollout["collection_start_phases"].float().mean().item()),
            "phase/start_min": float(rollout["collection_start_phases"].min().item()),
            "phase/start_max": float(rollout["collection_start_phases"].max().item()),
            "act/abs_mean": float(act_abs.mean().item()),
            "act/abs_p95": float(torch.quantile(act_abs.flatten(), 0.95).item()),
            "act/abs_p99": float(torch.quantile(act_abs.flatten(), 0.99).item()),
            "act/abs_max": float(act_abs.max().item()),
            "act/abs_max_all": float(rollout["action_abs_max"]),
            "time_action/mean": float(rollout["time_action_mean"]),
            "time_action/std": float(rollout["time_action_std"]),
            "time_action/fixed_dt": float(self.cfg.fixed_dt),
            "time_action/range_low": float(self.cfg.actor_time_scale_range[0] + self.cfg.fixed_dt),
            "time_action/range_high": float(self.cfg.actor_time_scale_range[1] + self.cfg.fixed_dt),
            "time_action/reference_frame_delta_mean": float(rollout["reference_frame_delta_mean"]),
            "time_action/reference_frame_delta_std": float(rollout["reference_frame_delta_std"]),
            "timing/collect_s": collect_time,
            "timing/update_s": update_time,
        }
        for key, mask in rollout["done_terms_union"].items():
            metrics[f"done/{key}_frac"] = float(mask.float().mean().item())
        self._add_reward_metrics(metrics, rollout)
        self._add_action_group_metrics(metrics, actions)
        self._add_sampler_metrics(metrics)
        if self._train_reward_buffer:
            metrics["train/mean_reward"] = float(sum(self._train_reward_buffer) / len(self._train_reward_buffer))
            metrics["train/mean_episode_length"] = float(sum(self._train_length_buffer) / len(self._train_length_buffer))
        else:
            metrics["train/mean_reward"] = 0.0
            metrics["train/mean_episode_length"] = 0.0
        metrics["train/recent_episode_count"] = float(len(self._train_reward_buffer))
        metrics["train/completed_episodes"] = float(self._train_completed_episodes)
        return metrics

    def _add_reward_metrics(self, metrics: dict, rollout: dict) -> None:
        for info in rollout["first_infos"]:
            for key, value in info["reward_terms"].items():
                metrics[f"reward/{key}_mean"] = float(value.mean().item())
        reward_sums: dict[str, float] = {}
        done_sums: dict[str, float] = {}
        weight_sum = 0.0
        for info, valid in rollout["rollout_info_items"]:
            valid_f = valid.float()
            weight = float(valid_f.sum().item())
            if weight <= 0.0:
                continue
            weight_sum += weight
            for key, value in info["reward_terms"].items():
                reward_sums[key] = reward_sums.get(key, 0.0) + float((value * valid_f).sum().item())
            for key, value in info["done_terms"].items():
                done_sums[key] = done_sums.get(key, 0.0) + float((value.float() * valid_f).sum().item())
        if weight_sum > 0.0:
            for key, value in reward_sums.items():
                metrics[f"reward_rollout/{key}_mean"] = value / weight_sum
            for key, value in done_sums.items():
                metrics[f"done_rollout/{key}_frac"] = value / weight_sum

    def _add_action_group_metrics(self, metrics: dict, actions: torch.Tensor) -> None:
        act_abs = actions.abs().mean(dim=(0, 1))
        if act_abs.numel() < 29:
            return
        metrics["act/legs_abs"] = float(act_abs[:12].mean().item())
        metrics["act/waist_abs"] = float(act_abs[12:15].mean().item())
        metrics["act/arms_abs"] = float(act_abs[15:29].mean().item())

    def _add_sampler_metrics(self, metrics: dict) -> None:
        stats = self.env.adaptive_sampling_stats()
        for key in ("mode", "top_bin", "top_prob", "failed_sum", "entropy", "peak_bin", "rsi_keyframe_count"):
            metrics[f"sampler/{key}"] = float(stats.get(key, float("nan")))

    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        print(
            f"[UPDATE] {update_idx}/{max_updates} "
            f"reward_step={metrics['rollout/reward_step_mean']:.5f} "
            f"return={metrics['rollout/return_mean']:.5f} "
            f"done={metrics['rollout/done_frac']:.5f} "
            f"mean_reward={metrics.get('train/mean_reward', float('nan')):.5f} "
            f"mean_len={metrics.get('train/mean_episode_length', float('nan')):.2f}",
            flush=True,
        )
        print(
            f"[ADAMIMIC] loss={metrics['adamimic/loss']:.5f} "
            f"policy_low={metrics['adamimic/policy_low_loss']:.5f} "
            f"policy_high={metrics['adamimic/policy_high_loss']:.5f} "
            f"value_low={metrics['adamimic/value_low_loss']:.5f} "
            f"value_high={metrics['adamimic/value_high_loss']:.5f} "
            f"ratio_low={metrics['adamimic/ratio_low']:.4f} "
            f"ratio_high={metrics['adamimic/ratio_high']:.4f} "
            f"clip_low={metrics['adamimic/clip_low']:.4f} "
            f"kl={metrics['adamimic/kl']:.6f} "
            f"grad={metrics['adamimic/grad_norm']:.4f} "
            f"lr={metrics['adamimic/lr']:.6f}",
            flush=True,
        )
        print(
            f"[TIME_POLICY] mean={metrics['time_action/mean']:.5f} "
            f"std={metrics['time_action/std']:.5f} "
            f"fixed_dt={metrics['time_action/fixed_dt']:.5f} "
            f"frame_delta={metrics['time_action/reference_frame_delta_mean']:.3f}"
            f"/{metrics['time_action/reference_frame_delta_std']:.3f} "
            f"range=[{metrics['time_action/range_low']:.5f},{metrics['time_action/range_high']:.5f}] "
            f"train_time={int(bool(self.cfg.train_time))}",
            flush=True,
        )
        print(
            f"[DONE] timeout={metrics.get('done/time_out_frac', 0.0):.5f} "
            f"anchor_pos={metrics.get('done/anchor_pos_bad_frac', 0.0):.5f} "
            f"anchor_ori={metrics.get('done/anchor_ori_bad_frac', 0.0):.5f} "
            f"ee_body={metrics.get('done/ee_body_bad_frac', 0.0):.5f} "
            f"motion_complete={metrics.get('done/motion_complete_frac', 0.0):.5f}",
            flush=True,
        )
        print(
            f"[TIME] collect={metrics['timing/collect_s']:.3f}s update={metrics['timing/update_s']:.3f}s",
            flush=True,
        )

    def log_banner(self) -> None:
        print(
            f"[METHOD] name=adamimic stage={self.cfg.stage} "
            "actor=timed_track_actor prior=tracking_reward credit=gae optimizer=muon_ppo "
            "weight_decay=0.01",
            flush=True,
        )
        print(
            f"[ARCH] actor_obs={self.actor_obs_dim} critic_obs={self.critic_obs_dim} "
            f"control_action={self.control_action_dim} policy_action={self.policy_action_dim} "
            f"hidden_actor={list(self.cfg.actor_hidden_dims)} hidden_critic={list(self.cfg.critic_hidden_dims)} "
            f"residual_delta={int(bool(self.cfg.residual_delta))}",
            flush=True,
        )
        print(
            f"[ADAPTIVE_TIME] infer={int(bool(self.cfg.infer_keyframe_time))} "
            f"fixed_dt={self.cfg.fixed_dt} "
            f"scale_range={list(self.cfg.actor_time_scale_range)} "
            f"train_time={int(bool(self.cfg.train_time))}",
            flush=True,
        )


Adamimic = AdaMimic
