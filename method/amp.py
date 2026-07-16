from __future__ import annotations

import math
import time
from collections import deque

import torch
from torch import nn

from components.imitation.style_reward import discriminator_style_reward, style_reward_statistics
from components.imitation.window_pipeline import TemporalWindowPipeline
from components.normalization.running_stats import RunningNormalizer
from components.replay.sample_buffer import SampleReplayBuffer
from method.base import Algorithm
from models.style_discriminator import StyleDiscriminator, compute_style_discriminator_loss


def _activation(name: str) -> type[nn.Module]:
    return {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "mish": nn.Mish,
    }[name.lower()]


def _trunk(input_dim: int, hidden_dims: tuple[int, ...], activation: str) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    previous = int(input_dim)
    act = _activation(activation)
    for width in hidden_dims:
        linear = nn.Linear(previous, int(width))
        nn.init.zeros_(linear.bias)
        layers.extend((linear, act()))
        previous = int(width)
    return nn.Sequential(*layers), previous


class _DiagGaussian:
    def __init__(self, mean: torch.Tensor, logstd: torch.Tensor) -> None:
        self.mean = mean
        self.logstd = logstd.expand_as(mean)
        self.std = torch.exp(self.logstd)
        self.dim = mean.shape[-1]

    @property
    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        diff = (value - self.mean) / self.std
        logp = -0.5 * diff.square().sum(dim=-1)
        logp += -0.5 * self.dim * math.log(2.0 * math.pi) - self.logstd.sum(dim=-1)
        return logp

    def entropy(self) -> torch.Tensor:
        return self.logstd.sum(dim=-1) + 0.5 * self.dim * math.log(2.0 * math.pi * math.e)

    def param_reg(self) -> torch.Tensor:
        return self.mean.square().sum(dim=-1)


class _AMPActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        action_std: float,
        output_scale: float,
    ) -> None:
        super().__init__()
        self.trunk, trunk_dim = _trunk(obs_dim, hidden_dims, activation)
        self.mean = nn.Linear(trunk_dim, action_dim)
        nn.init.uniform_(self.mean.weight, -float(output_scale), float(output_scale))
        nn.init.zeros_(self.mean.bias)
        self.register_buffer("logstd", torch.full((action_dim,), math.log(float(action_std))))

    def distribution(self, obs: torch.Tensor) -> _DiagGaussian:
        return _DiagGaussian(self.mean(self.trunk(obs)), self.logstd)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.distribution(obs).mode


class _AMPCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: tuple[int, ...], activation: str) -> None:
        super().__init__()
        self.trunk, trunk_dim = _trunk(obs_dim, hidden_dims, activation)
        self.value = nn.Linear(trunk_dim, 1)
        nn.init.zeros_(self.value.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(self.trunk(obs)).squeeze(-1)


class _MimicKitDiscHistory:
    """MimicKit AMP-style per-env history seeded by demo history at reset."""

    def __init__(
        self,
        num_envs: int,
        history_len: int,
        frame_dim: int,
        *,
        device: torch.device | str,
    ) -> None:
        self.num_envs = int(num_envs)
        self.history_len = int(history_len)
        self.frame_dim = int(frame_dim)
        self.device = torch.device(device)
        self.data = torch.zeros(self.num_envs, self.history_len, self.frame_dim, device=self.device)
        self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _check(self, frames: torch.Tensor, count: int) -> torch.Tensor:
        if tuple(frames.shape) != (count, self.frame_dim):
            raise ValueError(f"imitation frame shape must be {(count, self.frame_dim)}, got {tuple(frames.shape)}")
        values = frames.detach().to(device=self.device, dtype=torch.float32)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("imitation frame contains non-finite values")
        return values

    def _check_window(self, windows: torch.Tensor, count: int) -> torch.Tensor:
        expected = (count, self.history_len, self.frame_dim)
        if tuple(windows.shape) != expected:
            raise ValueError(f"imitation history shape must be {expected}, got {tuple(windows.shape)}")
        values = windows.detach().to(device=self.device, dtype=torch.float32)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("imitation history contains non-finite values")
        return values

    @torch.no_grad()
    def reset(self, initial_window: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        ids = self._ids(env_ids)
        if ids.numel() == 0:
            return
        windows = self._check_window(initial_window, ids.numel())
        self.data[ids] = windows
        self.initialized[ids] = True
        self.age[ids] = 0

    @torch.no_grad()
    def push(self, frame: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        ids = self._ids(env_ids)
        if ids.numel() == 0:
            return
        frames = self._check(frame, ids.numel())
        if not bool(self.initialized[ids].all()):
            bad = ids[~self.initialized[ids]].detach().cpu().tolist()
            raise RuntimeError(f"AMP discriminator history reset missing for env_ids={bad}")
        current = self.data.index_select(0, ids)
        current = torch.roll(current, shifts=-1, dims=1)
        current[:, -1] = frames
        self.data[ids] = current
        self.age[ids] += 1

    def window(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        ids = self._ids(env_ids)
        if not bool(self.initialized[ids].all()):
            bad = ids[~self.initialized[ids]].detach().cpu().tolist()
            raise RuntimeError(f"AMP discriminator history requested before reset for env_ids={bad}")
        return self.data.index_select(0, ids)

    def statistics(self) -> dict[str, float]:
        return {
            "history/initialized_fraction": float(self.initialized.float().mean().item()),
            "history/age_mean": float(self.age.float().mean().item()),
            "history/age_min": float(self.age.min().item()),
            "history/age_max": float(self.age.max().item()),
        }


class AMP(Algorithm):
    """MimicKit-style AMP: PPO actor-critic plus independent style discriminator."""

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        env.termination_mode = "amp"
        env.terminate_on_motion_end = False
        self.imitation_frame_dim = int(env.imitation_frame_dim)
        self.actor_obs_dim = int(env.get_amp_policy_observation().shape[-1])
        self.critic_obs_dim = self.actor_obs_dim
        self.action_dim = int(env.action_dim)
        self.rollout_steps = int(cfg.rollout_env_steps)
        self.disc_obs_steps = int(cfg.disc_obs_steps)
        self.imitation_window_dim = self.disc_obs_steps * self.imitation_frame_dim
        self.imitation_pipeline = TemporalWindowPipeline(self.disc_obs_steps, self.imitation_frame_dim)
        self.disc_history = _MimicKitDiscHistory(
            env.num_envs,
            self.disc_obs_steps,
            self.imitation_frame_dim,
            device=env.device,
        )
        self._build_action_normalizer()

        self.empirical_normalization = bool(cfg.empirical_normalization)
        if self.empirical_normalization:
            self.obs_normalizer: nn.Module = RunningNormalizer(
                self.actor_obs_dim,
                device=env.device,
                clip=10.0,
            )
        else:
            self.obs_normalizer = nn.Identity()

        self.actor = _AMPActor(
            self.actor_obs_dim,
            self.action_dim,
            tuple(cfg.actor_hidden_dims),
            cfg.activation,
            float(cfg.action_std),
            float(cfg.actor_init_output_scale),
        ).to(env.device)
        self.critic = _AMPCritic(
            self.critic_obs_dim,
            tuple(cfg.critic_hidden_dims),
            cfg.activation,
        ).to(env.device)
        self.discriminator = StyleDiscriminator(
            self.imitation_window_dim,
            tuple(cfg.disc_hidden_dims),
        ).to(env.device)
        self.disc_normalizer = RunningNormalizer(
            self.imitation_window_dim,
            device=env.device,
            clip=float(cfg.disc_normalizer_clip),
        )
        self.disc_replay = SampleReplayBuffer(
            int(cfg.disc_buffer_size),
            self.imitation_window_dim,
            storage_dtype=torch.float32,
            pin_memory=True,
        )

        self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=float(cfg.actor_lr), momentum=0.9)
        self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=float(cfg.critic_lr), momentum=0.9)
        self.disc_optimizer = torch.optim.SGD(
            self.discriminator.parameters(),
            lr=float(cfg.disc_lr),
            momentum=0.9,
            weight_decay=float(cfg.disc_weight_decay),
        )
        self.disc_version = 0
        self._policy_module = nn.ModuleDict(
            {
                "actor": self.actor,
                "critic": self.critic,
                "obs_normalizer": self.obs_normalizer,
                "discriminator": self.discriminator,
                "disc_normalizer": self.disc_normalizer,
            }
        )
        self._init_train_episode_stats()

    def _build_action_normalizer(self) -> None:
        env = self.env
        action_space = env.get_action_space()
        low = torch.as_tensor(action_space.low, dtype=torch.float32, device=env.device)
        high = torch.as_tensor(action_space.high, dtype=torch.float32, device=env.device)
        if low.shape != (self.action_dim,) or high.shape != (self.action_dim,):
            raise ValueError(
                f"AMP action space must be {(self.action_dim,)}, got low={tuple(low.shape)} high={tuple(high.shape)}"
            )
        action_std = 0.5 * (high - low)
        if not bool(torch.isfinite(action_std).all()) or not bool((action_std > 0).all()):
            raise ValueError("AMP requires finite non-degenerate Box action bounds")
        self.action_norm_mean = 0.5 * (high + low)
        self.action_norm_std = action_std.clamp_min(1.0e-6)
        self.action_bound_low = low
        self.action_bound_high = high

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_norm_mean.to(action)) / self.action_norm_std.to(action)

    def _unnormalize_action(self, norm_action: torch.Tensor) -> torch.Tensor:
        return norm_action * self.action_norm_std.to(norm_action) + self.action_norm_mean.to(norm_action)

    def _clip_env_action(self, action: torch.Tensor) -> torch.Tensor:
        return torch.minimum(
            torch.maximum(action, self.action_bound_low.to(action)),
            self.action_bound_high.to(action),
        )

    @property
    def policy(self) -> nn.Module:
        return self._policy_module

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.actor_optimizer

    @property
    def horizon(self) -> int:
        return 1

    def extra_checkpoint_state(self) -> dict:
        return {
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),
            "disc_replay": self.disc_replay.state_dict(),
            "disc_version": int(self.disc_version),
            "amp_schema_version": 3,
        }

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        if not payload:
            return
        if not reset_optimizer:
            if "critic_optimizer" in payload:
                self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            if "disc_optimizer" in payload:
                self.disc_optimizer.load_state_dict(payload["disc_optimizer"])
        self.disc_version = int(payload.get("disc_version", self.disc_version))
        if not self.disc_replay.load_state_dict(payload.get("disc_replay")):
            print("[AMP] discriminator replay absent/incompatible; starting empty", flush=True)

    def _norm_actor(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        if not self.empirical_normalization:
            return obs
        if update:
            self.obs_normalizer.record(obs)
        return self.obs_normalizer.normalize(obs)

    def _norm_critic(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        if not self.empirical_normalization:
            return obs
        if update:
            self.obs_normalizer.record(obs)
        return self.obs_normalizer.normalize(obs)

    def _amp_observation(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.env.get_amp_policy_observation(env_ids)

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        del obs
        amp_obs = self._amp_observation()
        norm_action = self.actor(self._norm_actor(amp_obs, update=False))
        return self._clip_env_action(self._unnormalize_action(norm_action))

    def initial_reset(self) -> torch.Tensor:
        obs = self.env.reset()
        if bool(self.cfg.init_at_random_ep_len) and self.env.max_episode_steps > 0:
            self.env.episode_steps = torch.randint_like(
                self.env.episode_steps,
                high=int(self.env.max_episode_steps),
            )
        self._obs = obs
        self._reset_disc_history()
        return obs

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        del update_idx
        return self._obs

    def _reset_disc_history(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            phase_indices = self.env.phase_steps.to(dtype=torch.long)
        else:
            phase_indices = self.env.phase_steps.index_select(0, env_ids).to(dtype=torch.long)
        initial_window = self.env.get_imitation_demo_history(
            phase_indices,
            self.disc_obs_steps,
            flatten=False,
        )
        self.disc_history.reset(initial_window, env_ids=env_ids)

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

    @torch.no_grad()
    def _evaluate_amp_reward(self, flat_windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = max(1, int(self.cfg.disc_eval_batch_size))
        logits: list[torch.Tensor] = []
        self.discriminator.eval()
        for start in range(0, flat_windows.shape[0], batch_size):
            batch = flat_windows[start : start + batch_size]
            norm = self.disc_normalizer.normalize(batch)
            logits.append(self.discriminator(norm))
        all_logits = torch.cat(logits, dim=0)
        rewards = discriminator_style_reward(
            all_logits,
            scale=float(self.cfg.disc_reward_scale),
            minimum_one_minus_prob=float(self.cfg.disc_reward_epsilon),
        )
        return all_logits, rewards

    @torch.no_grad()
    def _timeout_bootstrap_value(self, info: dict, timeout: torch.Tensor) -> torch.Tensor:
        values = torch.zeros(self.env.num_envs, device=self.env.device)
        ids = timeout.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return values
        final_amp_obs = info.get("amp_policy_observation")
        if torch.is_tensor(final_amp_obs):
            critic_obs = final_amp_obs.index_select(0, ids)
        else:
            critic_obs = self._amp_observation(ids)
        values[ids] = self.critic(self._norm_critic(critic_obs, update=False))
        return values

    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        device = env.device
        steps = self.rollout_steps
        n_envs = env.num_envs
        actor_obs_buf = torch.zeros(steps, n_envs, self.actor_obs_dim, device=device)
        critic_obs_buf = torch.zeros(steps, n_envs, self.critic_obs_dim, device=device)
        actions_buf = torch.zeros(steps, n_envs, self.action_dim, device=device)
        values_buf = torch.zeros(steps, n_envs, device=device)
        logp_buf = torch.zeros(steps, n_envs, device=device)
        mean_buf = torch.zeros_like(actions_buf)
        std_buf = torch.zeros_like(actions_buf)
        task_reward_buf = torch.zeros(steps, n_envs, device=device)
        disc_reward_buf = torch.zeros_like(task_reward_buf)
        mixed_reward_buf = torch.zeros_like(task_reward_buf)
        disc_logit_buf = torch.zeros_like(task_reward_buf)
        done_buf = torch.zeros(steps, n_envs, dtype=torch.bool, device=device)
        timeout_buf = torch.zeros_like(done_buf)
        failure_buf = torch.zeros_like(done_buf)
        motion_complete_buf = torch.zeros_like(done_buf)
        timeout_value_buf = torch.zeros(steps, n_envs, device=device)
        first_done_step = torch.full((n_envs,), steps, dtype=torch.long, device=device)
        first_done_phase = torch.full((n_envs,), -1, dtype=torch.long, device=device)
        ever_done = torch.zeros(n_envs, dtype=torch.bool, device=device)
        done_terms_union: dict[str, torch.Tensor] = {}
        rollout_info_items: list[tuple[dict, torch.Tensor]] = []
        first_infos: list[dict] = []
        disc_window_chunks: list[torch.Tensor] = []
        start_phases = env.phase_steps.detach().clone()
        action_abs_max = 0.0
        obs = current_obs

        with torch.no_grad():
            for step_idx in range(steps):
                amp_obs = self._amp_observation()
                actor_obs_n = self._norm_actor(amp_obs, update=True)
                critic_obs_n = self._norm_critic(amp_obs, update=False)
                dist = self.actor.distribution(actor_obs_n)
                norm_action = dist.sample()
                action = self._unnormalize_action(norm_action)
                step_action = self._clip_env_action(action)
                logp = dist.log_prob(norm_action)
                value = self.critic(critic_obs_n)
                next_obs, task_reward, done, info = env.step(step_action, auto_reset=True)

                imitation_frame = info.get("imitation_frame")
                if imitation_frame is None:
                    imitation_frame = env.get_imitation_policy_frame()
                self.disc_history.push(imitation_frame)
                flat_windows = self.imitation_pipeline.flatten(self.disc_history.window())
                disc_logits, disc_reward = self._evaluate_amp_reward(flat_windows)
                mixed_reward = (
                    float(self.cfg.task_reward_weight) * task_reward
                    + float(self.cfg.disc_reward_weight) * disc_reward
                )

                done_bool = done.bool()
                done_terms = info["done_terms"]
                timeout = done_bool & done_terms["time_out"].bool()
                motion_complete = done_terms.get("motion_complete")
                motion_complete = (
                    motion_complete.bool()
                    if torch.is_tensor(motion_complete)
                    else torch.zeros_like(timeout)
                )
                motion_complete = done_bool & motion_complete
                failure_terms = (
                    done_terms["anchor_pos_bad"].bool()
                    | done_terms["anchor_ori_bad"].bool()
                    | done_terms["ee_body_bad"].bool()
                )
                failure = done_bool & failure_terms & (~timeout) & (~motion_complete)
                timeout_value = self._timeout_bootstrap_value(info, timeout)

                actor_obs_buf[step_idx] = amp_obs
                critic_obs_buf[step_idx] = amp_obs
                actions_buf[step_idx] = action
                values_buf[step_idx] = value
                logp_buf[step_idx] = logp
                mean_buf[step_idx] = dist.mean
                std_buf[step_idx] = dist.std
                task_reward_buf[step_idx] = task_reward
                disc_reward_buf[step_idx] = disc_reward
                mixed_reward_buf[step_idx] = mixed_reward
                disc_logit_buf[step_idx] = disc_logits
                done_buf[step_idx] = done_bool
                timeout_buf[step_idx] = timeout
                failure_buf[step_idx] = failure
                motion_complete_buf[step_idx] = motion_complete
                timeout_value_buf[step_idx] = timeout_value
                disc_window_chunks.append(flat_windows.detach().to("cpu", dtype=torch.float32))

                if step_idx == 0:
                    first_infos.append(info)
                for key, value_t in done_terms.items():
                    b = value_t.bool()
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
                self._record_episode_stats(mixed_reward, done_bool)
                action_abs_max = max(action_abs_max, float(step_action.abs().max().item()))

                reset_ids = info.get("reset_env_ids")
                if torch.is_tensor(reset_ids) and reset_ids.numel() > 0:
                    self._reset_disc_history(env_ids=reset_ids)
                obs = next_obs

            last_amp_obs = self._amp_observation()
            last_value = self.critic(self._norm_critic(last_amp_obs, update=False))

        self._obs = obs
        returns, advantages = self._compute_returns(
            rewards=mixed_reward_buf,
            values=values_buf,
            last_value=last_value,
            dones=done_buf,
            timeout_mask=timeout_buf,
            timeout_values=timeout_value_buf,
        )
        return {
            "actor_obs": actor_obs_buf,
            "critic_obs": critic_obs_buf,
            "actions": actions_buf,
            "values": values_buf,
            "returns": returns,
            "advantages": self._normalize_advantages(advantages),
            "old_logp": logp_buf,
            "old_mu": mean_buf,
            "old_sigma": std_buf,
            "task_reward": task_reward_buf,
            "disc_reward": disc_reward_buf,
            "mixed_reward": mixed_reward_buf,
            "disc_logits": disc_logit_buf,
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
            "action_abs_max": action_abs_max,
            "disc_windows": torch.cat(disc_window_chunks, dim=0),
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
        timeout_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        returns = torch.zeros_like(values)
        advantages = torch.zeros_like(values)
        gae = torch.zeros_like(last_value)
        gamma = float(self.cfg.discount_gamma)
        lam = float(self.cfg.gae_lambda)
        for step_idx in range(rewards.shape[0] - 1, -1, -1):
            next_value = last_value if step_idx == rewards.shape[0] - 1 else values[step_idx + 1]
            nonterminal = (~dones[step_idx]).to(dtype=values.dtype)
            bootstrap = nonterminal * next_value + timeout_mask[step_idx].to(values.dtype) * timeout_values[step_idx]
            delta = rewards[step_idx] + gamma * bootstrap - values[step_idx]
            gae = delta + gamma * lam * nonterminal * gae
            advantages[step_idx] = gae
            returns[step_idx] = gae + values[step_idx]
        return returns, advantages

    def _normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        mean = advantages.mean()
        std = advantages.std(unbiased=False).clamp_min(1.0e-5)
        normalized = (advantages - mean) / std
        return torch.clamp(normalized, -float(self.cfg.norm_adv_clip), float(self.cfg.norm_adv_clip))

    def _sample_indices(self, sample_count: int, batch_size: int) -> torch.Tensor:
        return torch.randint(sample_count, (int(batch_size),), device=self.env.device)

    def _mimickit_batch_size(self, multiplier: int) -> int:
        return int(math.ceil(int(multiplier) * self.env.num_envs))

    def _critic_update(self, rollout: dict) -> dict[str, float]:
        steps, n_envs = rollout["values"].shape
        sample_count = steps * n_envs
        critic_obs = rollout["critic_obs"].reshape(sample_count, self.critic_obs_dim)
        returns = rollout["returns"].reshape(sample_count)
        batch_size = self._mimickit_batch_size(int(self.cfg.critic_batch_size))
        update_steps = int(math.ceil(sample_count / batch_size)) * int(self.cfg.critic_epochs)
        total_loss = 0.0
        for _ in range(update_steps):
            idx = self._sample_indices(sample_count, batch_size)
            pred = self.critic(self._norm_critic(critic_obs[idx], update=False))
            loss = (returns[idx] - pred).square().mean()
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.critic_optimizer.step()
            total_loss += float(loss.detach().item())
        return {
            "amp/critic_loss": total_loss / max(update_steps, 1),
            "amp/critic_optimizer_steps": float(update_steps),
            "amp/critic_lr": float(self.cfg.critic_lr),
        }

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

    def _action_bound_loss(self, dist: _DiagGaussian) -> torch.Tensor:
        low_violation = torch.clamp_max(dist.mode + 1.0, 0.0)
        high_violation = torch.clamp_min(dist.mode - 1.0, 0.0)
        return (low_violation.square() + high_violation.square()).sum(dim=-1)

    def _actor_update(self, rollout: dict) -> dict[str, float]:
        steps, n_envs = rollout["old_logp"].shape
        sample_count = steps * n_envs
        actor_obs = rollout["actor_obs"].reshape(sample_count, self.actor_obs_dim)
        actions = self._normalize_action(rollout["actions"].reshape(sample_count, self.action_dim))
        old_logp = rollout["old_logp"].reshape(sample_count)
        old_mu = rollout["old_mu"].reshape(sample_count, self.action_dim)
        old_sigma = rollout["old_sigma"].reshape(sample_count, self.action_dim)
        advantages = rollout["advantages"].reshape(sample_count)
        batch_size = self._mimickit_batch_size(int(self.cfg.actor_batch_size))
        update_steps = int(math.ceil(sample_count / batch_size)) * int(self.cfg.actor_epochs)
        clip_low = 1.0 - float(self.cfg.clip_range)
        clip_high = 1.0 + float(self.cfg.clip_range)
        totals = {
            "policy": 0.0,
            "total": 0.0,
            "ratio": 0.0,
            "clip": 0.0,
            "kl": 0.0,
            "entropy": 0.0,
            "bound": 0.0,
            "reg": 0.0,
        }
        for _ in range(update_steps):
            idx = self._sample_indices(sample_count, batch_size)
            dist = self.actor.distribution(self._norm_actor(actor_obs[idx], update=False))
            logp = dist.log_prob(actions[idx])
            ratio = torch.exp(logp - old_logp[idx])
            unclipped = advantages[idx] * ratio
            clipped = advantages[idx] * torch.clamp(ratio, clip_low, clip_high)
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            loss = policy_loss
            bound_loss = torch.zeros((), device=self.env.device)
            entropy = torch.zeros((), device=self.env.device)
            reg_loss = torch.zeros((), device=self.env.device)
            if float(self.cfg.action_bound_weight) != 0.0:
                bound_loss = self._action_bound_loss(dist).mean()
                loss = loss + float(self.cfg.action_bound_weight) * bound_loss
            if float(self.cfg.action_entropy_weight) != 0.0:
                entropy = dist.entropy().mean()
                loss = loss - float(self.cfg.action_entropy_weight) * entropy
            if float(self.cfg.action_reg_weight) != 0.0:
                reg_loss = dist.param_reg().mean()
                loss = loss + float(self.cfg.action_reg_weight) * reg_loss
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.actor_optimizer.step()
            with torch.no_grad():
                kl = self._gaussian_kl(old_mu[idx], old_sigma[idx], dist.mean, dist.std).mean()
                clip_frac = ((ratio < clip_low) | (ratio > clip_high)).float().mean()
                totals["policy"] += float(policy_loss.item())
                totals["total"] += float(loss.item())
                totals["ratio"] += float(ratio.mean().item())
                totals["clip"] += float(clip_frac.item())
                totals["kl"] += float(kl.item())
                totals["entropy"] += float(entropy.item())
                totals["bound"] += float(bound_loss.item())
                totals["reg"] += float(reg_loss.item())
        denom = max(update_steps, 1)
        return {
            "amp/actor_policy_loss": totals["policy"] / denom,
            "amp/actor_loss": totals["total"] / denom,
            "amp/ratio": totals["ratio"] / denom,
            "amp/clip_fraction": totals["clip"] / denom,
            "amp/kl": totals["kl"] / denom,
            "amp/action_entropy": totals["entropy"] / denom,
            "amp/action_bound_loss": totals["bound"] / denom,
            "amp/action_reg_loss": totals["reg"] / denom,
            "amp/actor_optimizer_steps": float(update_steps),
            "amp/actor_lr": float(self.cfg.actor_lr),
        }

    @torch.no_grad()
    def _sample_demo_windows_cpu(self, count: int) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        batch_size = max(1, int(self.cfg.disc_eval_batch_size))
        for start in range(0, int(count), batch_size):
            n = min(batch_size, int(count) - start)
            raw = self.env.sample_imitation_demo_windows(n, self.disc_obs_steps, flatten=False)
            flat = self.imitation_pipeline.flatten(raw)
            chunks.append(flat.detach().to("cpu", dtype=torch.float32))
        return torch.cat(chunks, dim=0) if chunks else torch.empty((0, self.imitation_window_dim), dtype=torch.float32)

    @torch.no_grad()
    def _record_disc_normalizer(self, current_cpu: torch.Tensor, demo_cpu: torch.Tensor) -> int:
        self.disc_normalizer.clear_pending()
        batch_size = max(1, int(self.cfg.disc_eval_batch_size))
        count = int(current_cpu.shape[0])
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            self.disc_normalizer.record(current_cpu[start:end].to(device=self.env.device))
            self.disc_normalizer.record(demo_cpu[start:end].to(device=self.env.device))
        return count

    @torch.no_grad()
    def _store_disc_replay_data(self, current_cpu: torch.Tensor) -> int:
        count = int(current_cpu.shape[0])
        if count == 0:
            return 0
        if self.disc_replay.is_full:
            keep = min(count, int(self.cfg.disc_replay_samples))
        else:
            keep = count
        indices = torch.randperm(count, device="cpu")[:keep]
        self.disc_replay.push(current_cpu.index_select(0, indices))
        return int(keep)

    def _disc_update(self, current_cpu: torch.Tensor, demo_cpu: torch.Tensor) -> dict[str, float]:
        count = int(current_cpu.shape[0])
        if count == 0:
            return {"disc/update_steps": 0.0, "disc/skipped_no_current": 1.0}
        batch_size = self._mimickit_batch_size(int(self.cfg.disc_batch_size))
        update_steps = int(math.ceil(count / batch_size)) * int(self.cfg.disc_epochs)
        totals: dict[str, float] = {}
        grad_total = 0.0
        self.discriminator.train()
        for _ in range(update_steps):
            idx = torch.randint(count, (batch_size,), device="cpu")
            current = current_cpu.index_select(0, idx).to(device=self.env.device, dtype=torch.float32)
            demo = demo_cpu.index_select(0, idx).to(device=self.env.device, dtype=torch.float32)
            replay = self.disc_replay.sample(batch_size, device=self.env.device, dtype=torch.float32)
            output = compute_style_discriminator_loss(
                self.discriminator,
                expert_observations=self.disc_normalizer.normalize(demo),
                policy_observations=self.disc_normalizer.normalize(current),
                replay_observations=self.disc_normalizer.normalize(replay),
                gradient_penalty_weight=float(self.cfg.disc_grad_penalty),
                logit_regularization_weight=float(self.cfg.disc_logit_reg),
            )
            self.disc_optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            grad = nn.utils.clip_grad_norm_(self.discriminator.parameters(), float("inf"))
            self.disc_optimizer.step()
            grad_total += float(grad)
            for key, value in output.metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.item())
        self.disc_version += 1
        denom = max(update_steps, 1)
        metrics = {key: value / denom for key, value in totals.items()}
        metrics.update(
            {
                "disc/grad_norm": grad_total / denom,
                "disc/update_steps": float(update_steps),
                "disc/version": float(self.disc_version),
                "disc/lr": float(self.cfg.disc_lr),
            }
        )
        return metrics

    def update(self, rollout: dict, collect_time: float) -> dict:
        start = time.perf_counter()
        current_cpu = rollout["disc_windows"]
        demo_cpu = self._sample_demo_windows_cpu(int(current_cpu.shape[0]))
        normalizer_samples = self._record_disc_normalizer(current_cpu, demo_cpu)
        replay_added = self._store_disc_replay_data(current_cpu)
        critic_start = time.perf_counter()
        critic_metrics = self._critic_update(rollout)
        critic_time = time.perf_counter() - critic_start
        actor_start = time.perf_counter()
        actor_metrics = self._actor_update(rollout)
        actor_time = time.perf_counter() - actor_start
        disc_start = time.perf_counter()
        disc_metrics = self._disc_update(current_cpu, demo_cpu)
        disc_time = time.perf_counter() - disc_start
        obs_committed = self.obs_normalizer.commit() if self.empirical_normalization else False
        committed = self.disc_normalizer.commit()

        metrics = self._build_metrics(rollout, collect_time, time.perf_counter() - start)
        metrics.update(critic_metrics)
        metrics.update(actor_metrics)
        metrics.update(disc_metrics)
        metrics.update(self.disc_normalizer.statistics())
        if self.empirical_normalization:
            metrics.update(self.obs_normalizer.statistics(prefix="obs_norm"))
        metrics.update(self.disc_history.statistics())
        metrics.update(
            {
                key.replace("replay/", "disc_replay/"): value
                for key, value in self.disc_replay.statistics().items()
            }
        )
        metrics.update(
            {
                "disc_norm/committed_this_update": float(committed),
                "obs_norm/committed_this_update": float(obs_committed),
                "disc_norm/policy_samples_update": float(normalizer_samples),
                "disc_norm/expert_samples_update": float(normalizer_samples),
                "disc_replay/added_this_update": float(replay_added),
                "timing/collect_s": float(collect_time),
                "timing/critic_update_s": float(critic_time),
                "timing/actor_update_s": float(actor_time),
                "timing/disc_update_s": float(disc_time),
                "timing/update_s": float(time.perf_counter() - start),
                "method/amp": 1.0,
                "system/parameters_finite": float(self._parameters_finite()),
            }
        )
        if not bool(metrics["system/parameters_finite"]):
            raise FloatingPointError("AMP detected non-finite trainable parameters")
        return metrics

    def _parameters_finite(self) -> bool:
        return all(
            bool(torch.isfinite(parameter).all())
            for module in (self.actor, self.critic, self.discriminator)
            for parameter in module.parameters()
        )

    def _build_metrics(self, rollout: dict, collect_time: float, update_time: float) -> dict:
        del collect_time, update_time
        actions = rollout["actions"]
        rewards = rollout["mixed_reward"]
        done = rollout["done"]
        failures = rollout["failure"]
        timeouts = rollout["timeout"]
        motion_complete = rollout["motion_complete"]
        first_done_step = rollout["first_done_step"]
        failed_first = (first_done_step < self.rollout_steps) & ~timeouts.any(dim=0) & ~motion_complete.any(dim=0)
        returns = rewards.sum(dim=0)
        metrics = {
            "rollout/task_reward_mean": float(rollout["task_reward"].mean().item()),
            "rollout/disc_reward_mean": float(rollout["disc_reward"].mean().item()),
            "rollout/mixed_reward_mean": float(rewards.mean().item()),
            "rollout/return_mean": float(returns.mean().item()),
            "rollout/return_std": float(returns.std(unbiased=False).item()),
            "rollout/done_frac": float(done.float().mean().item()),
            "rollout/failure_frac": float(failures.float().mean().item()),
            "rollout/timeout_frac": float(timeouts.float().mean().item()),
            "rollout/motion_complete_frac": float(motion_complete.float().mean().item()),
            "rollout/success_frac": float((~failed_first).float().mean().item()),
            "rollout/first_done_step_mean": float(first_done_step.float().mean().item()),
            "phase/start_mean": float(rollout["collection_start_phases"].float().mean().item()),
            "phase/start_min": float(rollout["collection_start_phases"].min().item()),
            "phase/start_max": float(rollout["collection_start_phases"].max().item()),
            "act/abs_mean": float(actions.abs().mean().item()),
            "act/abs_p95": float(torch.quantile(actions.abs().flatten(), 0.95).item()),
            "act/abs_p99": float(torch.quantile(actions.abs().flatten(), 0.99).item()),
            "act/abs_max": float(actions.abs().max().item()),
            "act/abs_max_all": float(rollout["action_abs_max"]),
            "train/mean_reward": float(sum(self._train_reward_buffer) / len(self._train_reward_buffer)) if self._train_reward_buffer else 0.0,
            "train/mean_episode_length": float(sum(self._train_length_buffer) / len(self._train_length_buffer)) if self._train_length_buffer else 0.0,
            "train/recent_episode_count": float(len(self._train_reward_buffer)),
            "train/completed_episodes": float(self._train_completed_episodes),
        }
        metrics.update(
            style_reward_statistics(
                rollout["disc_logits"].reshape(-1),
                rollout["disc_reward"].reshape(-1),
                scale=float(self.cfg.disc_reward_scale),
                minimum_one_minus_prob=float(self.cfg.disc_reward_epsilon),
                prefix="amp_reward",
            )
        )
        for key, mask in rollout["done_terms_union"].items():
            metrics[f"done/{key}_frac"] = float(mask.float().mean().item())
        self._add_reward_metrics(metrics, rollout)
        self._add_action_group_metrics(metrics, actions)
        self._add_sampler_metrics(metrics)
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
            value = stats.get(key)
            if value is None:
                continue
            value_f = float(value)
            if math.isfinite(value_f):
                metrics[f"sampler/{key}"] = value_f

    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        print(
            f"[UPDATE] {update_idx}/{max_updates} "
            f"task={metrics.get('rollout/task_reward_mean', float('nan')):.5f} "
            f"amp={metrics.get('rollout/disc_reward_mean', float('nan')):.5f} "
            f"mixed={metrics.get('rollout/mixed_reward_mean', float('nan')):.5f} "
            f"done={metrics.get('rollout/done_frac', float('nan')):.5f} "
            f"ep_len={metrics.get('train/mean_episode_length', float('nan')):.2f}",
            flush=True,
        )
        print(
            f"[AMP_PPO] actor={metrics.get('amp/actor_policy_loss', float('nan')):.5f} "
            f"critic={metrics.get('amp/critic_loss', float('nan')):.5f} "
            f"ratio={metrics.get('amp/ratio', float('nan')):.4f} "
            f"clip={metrics.get('amp/clip_fraction', float('nan')):.4f} "
            f"kl={metrics.get('amp/kl', float('nan')):.6f} "
            f"bound={metrics.get('amp/action_bound_loss', float('nan')):.5f}",
            flush=True,
        )
        print(
            f"[AMP_DISC] loss={metrics.get('disc/loss', float('nan')):.5f} "
            f"bce={metrics.get('disc/bce', float('nan')):.5f} "
            f"gp={metrics.get('disc/gradient_penalty', float('nan')):.5f} "
            f"accE={metrics.get('disc/expert_accuracy', float('nan')):.3f} "
            f"accP={metrics.get('disc/current_accuracy', float('nan')):.3f} "
            f"rew={metrics.get('amp_reward/mean', float('nan')):.5f} "
            f"replay={metrics.get('disc_replay/size', float('nan')):.0f}",
            flush=True,
        )
        print(
            f"[TIME] collect={metrics['timing/collect_s']:.3f}s "
            f"actor={metrics['timing/actor_update_s']:.3f}s "
            f"critic={metrics['timing/critic_update_s']:.3f}s "
            f"disc={metrics['timing/disc_update_s']:.3f}s",
            flush=True,
        )

    def log_banner(self) -> None:
        print(
            "[METHOD] name=amp actor=ppo_gaussian_fixed_std prior=mimickit_discriminator "
            "credit=gae optimizer=sgd_momentum",
            flush=True,
        )
        print(
            f"[ARCH] actor_obs={self.actor_obs_dim} critic_obs={self.critic_obs_dim} "
            f"action_dim={self.action_dim} hidden_actor={list(self.cfg.actor_hidden_dims)} "
            f"hidden_critic={list(self.cfg.critic_hidden_dims)} hidden_disc={list(self.cfg.disc_hidden_dims)} "
            f"W_D={self.disc_obs_steps} imitation_window={self.imitation_window_dim}",
            flush=True,
        )
        print(
            f"[AMP_OFFICIAL] rollout={self.cfg.rollout_env_steps} "
            f"actor_epochs={self.cfg.actor_epochs} critic_epochs={self.cfg.critic_epochs} "
            f"disc_epochs={self.cfg.disc_epochs} action_std={self.cfg.action_std} "
            f"task_w={self.cfg.task_reward_weight} disc_w={self.cfg.disc_reward_weight} "
            f"pose_termination=0 motion_end_terminal=0 action=residual_env_space",
            flush=True,
        )
        print(
            f"[AMP_ACTION_SPACE] mean_abs={self.action_norm_mean.abs().mean().item():.4f} "
            f"std_mean={self.action_norm_std.mean().item():.4f} "
            f"std_min={self.action_norm_std.min().item():.4f} "
            f"std_max={self.action_norm_std.max().item():.4f}",
            flush=True,
        )


Amp = AMP
