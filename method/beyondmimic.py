from __future__ import annotations

import math
import time
from collections import deque
from importlib.metadata import PackageNotFoundError, version

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization
from tensordict import TensorDict
from torch import nn

from method.base import Algorithm, classify_mimickit_done_terms


OFFICIAL_ACTOR_OBS_DIM = 160
OFFICIAL_CRITIC_OBS_DIM = 286
OFFICIAL_ROLLOUT_STEPS = 24
BEYONDMIMIC_SCHEMA_VERSION = 3
OFFICIAL_NORMALIZER_UNTIL = 100_000_000
OFFICIAL_RSL_RL_VERSION = "2.3.3"


class Beyondmimic(Algorithm):
    """BeyondMimic's official RSL-RL actor-critic and PPO training recipe.

    The environment owns the motion command, tracking reward, termination,
    randomization, and adaptive motion sampler.  This adapter intentionally
    delegates the policy, rollout storage, timeout bootstrap, GAE, PPO update,
    observation normalizers, and adaptive learning-rate schedule to RSL-RL.
    """

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        self.rollout_steps = int(cfg.rollout_env_steps)
        if self.rollout_steps != OFFICIAL_ROLLOUT_STEPS:
            raise ValueError(
                f"BeyondMimic requires the official {OFFICIAL_ROLLOUT_STEPS}-step rollout, "
                f"got {self.rollout_steps}"
            )

        initial_observations = self._rsl_observations()
        self.actor_obs_dim = int(initial_observations["policy"].shape[-1])
        self.critic_obs_dim = int(initial_observations["critic"].shape[-1])
        self.action_dim = int(env.action_dim)
        if self.actor_obs_dim != OFFICIAL_ACTOR_OBS_DIM:
            raise ValueError(
                f"BeyondMimic policy observation must be {OFFICIAL_ACTOR_OBS_DIM}D, "
                f"got {self.actor_obs_dim}D"
            )
        if self.critic_obs_dim != OFFICIAL_CRITIC_OBS_DIM:
            raise ValueError(
                f"BeyondMimic critic observation must be {OFFICIAL_CRITIC_OBS_DIM}D, "
                f"got {self.critic_obs_dim}D"
            )
        if str(cfg.noise_std_type) != "scalar" or bool(cfg.state_dependent_std):
            raise ValueError(
                "BeyondMimic uses RSL-RL's global learnable scalar-type action standard deviation "
                "(noise_std_type='scalar', state_dependent_std=false)"
            )

        observation_groups = {"policy": ["policy"], "critic": ["critic"]}
        self.actor_critic = ActorCritic(
            initial_observations,
            observation_groups,
            self.action_dim,
            # BeyondMimic used RSL-RL 2.3.3, where the runner normalizes each
            # next observation before it becomes the following action/storage
            # input.  Keep 3.1's internal normalizers disabled so the storage
            # boundary below remains exactly action-time normalized.
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=list(cfg.actor_hidden_dims),
            critic_hidden_dims=list(cfg.critic_hidden_dims),
            activation=str(cfg.activation),
            init_noise_std=float(cfg.init_noise_std),
            noise_std_type=str(cfg.noise_std_type),
            state_dependent_std=bool(cfg.state_dependent_std),
        ).to(env.device)
        self.actor_obs_normalizer = EmpiricalNormalization(
            self.actor_obs_dim, until=OFFICIAL_NORMALIZER_UNTIL
        ).to(env.device)
        self.critic_obs_normalizer = EmpiricalNormalization(
            self.critic_obs_dim, until=OFFICIAL_NORMALIZER_UNTIL
        ).to(env.device)
        # Register the two official runner-side normalizers with the policy so
        # generic train/eval and checkpoint state handling remain complete.
        self.actor_critic.official_actor_obs_normalizer = self.actor_obs_normalizer
        self.actor_critic.official_critic_obs_normalizer = self.critic_obs_normalizer
        self.ppo = PPO(
            self.actor_critic,
            num_learning_epochs=int(cfg.num_learning_epochs),
            num_mini_batches=int(cfg.num_mini_batches),
            clip_param=float(cfg.clip_param),
            gamma=float(cfg.gamma),
            lam=float(cfg.lam),
            value_loss_coef=float(cfg.value_loss_coef),
            entropy_coef=float(cfg.entropy_coef),
            learning_rate=float(cfg.learning_rate),
            max_grad_norm=float(cfg.max_grad_norm),
            use_clipped_value_loss=bool(cfg.use_clipped_value_loss),
            schedule=str(cfg.schedule),
            desired_kl=float(cfg.desired_kl),
            device=str(env.device),
            normalize_advantage_per_mini_batch=bool(cfg.normalize_advantage_per_mini_batch),
        )
        self.ppo.init_storage(
            "rl",
            env.num_envs,
            self.rollout_steps,
            initial_observations,
            [self.action_dim],
        )
        self._init_train_episode_stats()

    def _rsl_observations(self, actor_obs: torch.Tensor | None = None) -> TensorDict:
        if actor_obs is None:
            actor_obs = self.env.get_beyondmimic_policy_observation()
        critic_obs = self.env.get_beyondmimic_critic_observation()
        expected_actor_shape = (self.env.num_envs, OFFICIAL_ACTOR_OBS_DIM)
        expected_critic_shape = (self.env.num_envs, OFFICIAL_CRITIC_OBS_DIM)
        if tuple(actor_obs.shape) != expected_actor_shape:
            raise ValueError(
                f"BeyondMimic policy observation must be {OFFICIAL_ACTOR_OBS_DIM}D "
                f"with shape {expected_actor_shape}, "
                f"got {tuple(actor_obs.shape)}"
            )
        if tuple(critic_obs.shape) != expected_critic_shape:
            raise ValueError(
                f"BeyondMimic critic observation must be {OFFICIAL_CRITIC_OBS_DIM}D "
                f"with shape {expected_critic_shape}, "
                f"got {tuple(critic_obs.shape)}"
            )
        if not bool(torch.isfinite(actor_obs).all()) or not bool(torch.isfinite(critic_obs).all()):
            raise FloatingPointError("BeyondMimic observation contains non-finite values")
        return TensorDict(
            {"policy": actor_obs, "critic": critic_obs},
            batch_size=[self.env.num_envs],
            device=self.env.device,
        )

    def _normalize_next_observations(self, observations: TensorDict) -> TensorDict:
        """Apply the official RSL-RL 2.3.3 runner normalization boundary."""
        policy_obs = observations["policy"]
        critic_obs = observations["critic"]
        self.actor_obs_normalizer.update(policy_obs)
        self.critic_obs_normalizer.update(critic_obs)
        return TensorDict(
            {
                "policy": self.actor_obs_normalizer(policy_obs),
                "critic": self.critic_obs_normalizer(critic_obs),
            },
            batch_size=[self.env.num_envs],
            device=self.env.device,
        )

    def _normalize_inference_observations(self, observations: TensorDict) -> TensorDict:
        return TensorDict(
            {
                "policy": self.actor_obs_normalizer(observations["policy"]),
                "critic": self.critic_obs_normalizer(observations["critic"]),
            },
            batch_size=[self.env.num_envs],
            device=self.env.device,
        )

    @property
    def policy(self) -> nn.Module:
        return self.actor_critic

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.ppo.optimizer

    @property
    def horizon(self) -> int:
        return 1

    def extra_checkpoint_state(self) -> dict:
        return {
            "learning_rate": float(self.ppo.learning_rate),
            "beyondmimic_schema_version": BEYONDMIMIC_SCHEMA_VERSION,
            "rsl_rl_version": self._rsl_rl_version(),
            "official_rsl_rl_version": OFFICIAL_RSL_RL_VERSION,
        }

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        saved_schema = int(payload.get("beyondmimic_schema_version", -1))
        if saved_schema != BEYONDMIMIC_SCHEMA_VERSION:
            raise ValueError(
                "BeyondMimic checkpoint schema mismatch: "
                f"expected {BEYONDMIMIC_SCHEMA_VERSION}, got {saved_schema}"
            )
        saved_official_version = str(payload.get("official_rsl_rl_version", "unknown"))
        if saved_official_version != OFFICIAL_RSL_RL_VERSION:
            raise ValueError(
                "BeyondMimic official RSL-RL provenance mismatch: "
                f"checkpoint={saved_official_version}, expected={OFFICIAL_RSL_RL_VERSION}"
            )
        saved_rsl_version = str(payload.get("rsl_rl_version", "unknown"))
        current_rsl_version = self._rsl_rl_version()
        if saved_rsl_version != current_rsl_version:
            raise ValueError(
                "BeyondMimic checkpoint RSL-RL version mismatch: "
                f"checkpoint={saved_rsl_version}, runtime={current_rsl_version}"
            )
        if reset_optimizer:
            learning_rate = float(self.cfg.learning_rate)
        else:
            learning_rate = float(
                payload.get(
                    "learning_rate",
                    self.ppo.optimizer.param_groups[0]["lr"],
                )
            )
        self.ppo.learning_rate = learning_rate
        for group in self.ppo.optimizer.param_groups:
            group["lr"] = learning_rate

    @staticmethod
    def _rsl_rl_version() -> str:
        try:
            return version("rsl-rl-lib")
        except PackageNotFoundError:
            return "unknown"

    @torch.no_grad()
    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self._rsl_observations(obs)
        return self.actor_critic.act_inference(
            self._normalize_inference_observations(raw)
        )

    def initial_reset(self) -> torch.Tensor:
        # The official IsaacLab RSL wrapper resets the environment once before
        # constructing OnPolicyRunner. G1MimicEnv's constructor already does
        # the same, so resetting again here would consume a second command,
        # reset-noise and adaptive-sampling draw.
        if bool(self.cfg.init_at_random_ep_len) and self.env.max_episode_steps > 0:
            self.env.episode_steps = torch.randint_like(
                self.env.episode_steps,
                high=int(self.env.max_episode_steps),
            )
        obs = self.env.get_observation()
        self._obs = obs
        # RSL-RL 2.3.3 intentionally sends the very first observation to the
        # policy unnormalized. Every subsequent observation is normalized at
        # the env.step boundary in collect().
        self._rsl_obs = self._rsl_observations(obs)
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
        self._train_reward_buffer.extend(
            self._train_reward_sum.index_select(0, done_ids).detach().cpu().tolist()
        )
        self._train_length_buffer.extend(
            self._train_episode_length.index_select(0, done_ids).detach().cpu().tolist()
        )
        self._train_completed_episodes += int(done_ids.numel())
        self._train_reward_sum[done_ids] = 0.0
        self._train_episode_length[done_ids] = 0.0

    def collect(self, current_obs: torch.Tensor) -> dict:
        del current_obs
        env = self.env
        device = env.device
        steps = self.rollout_steps
        n_envs = env.num_envs

        actions_buf = torch.zeros(steps, n_envs, self.action_dim, device=device)
        reward_buf = torch.zeros(steps, n_envs, device=device)
        done_buf = torch.zeros(steps, n_envs, dtype=torch.bool, device=device)
        timeout_buf = torch.zeros_like(done_buf)
        failure_buf = torch.zeros_like(done_buf)
        motion_complete_buf = torch.zeros_like(done_buf)
        motion_resample_buf = torch.zeros_like(done_buf)
        interval_push_buf = torch.zeros_like(done_buf)
        first_done_step = torch.full((n_envs,), steps, dtype=torch.long, device=device)
        first_done_phase = torch.full((n_envs,), -1, dtype=torch.long, device=device)
        ever_done = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first_failure = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first_timeout = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first_motion_complete = torch.zeros(n_envs, dtype=torch.bool, device=device)
        done_terms_union: dict[str, torch.Tensor] = {}
        rollout_infos: list[dict] = []
        start_phases = env.phase_steps.detach().clone()
        action_abs_max = 0.0
        obs = self._obs
        rsl_obs = self._rsl_obs

        with torch.inference_mode():
            for step_idx in range(steps):
                actions = self.ppo.act(rsl_obs)
                if not bool(torch.isfinite(actions).all()):
                    raise FloatingPointError("BeyondMimic sampled non-finite actions")
                next_obs, rewards, done, info = env.step(actions, auto_reset=True)
                if not bool(torch.isfinite(rewards).all()):
                    raise FloatingPointError("BeyondMimic environment returned non-finite rewards")
                # The official ObservationManager samples actor corruption
                # exactly once per environment step.  Reuse the observation
                # returned by env.step and compute only the clean critic here.
                raw_next_rsl_obs = self._rsl_observations(next_obs)
                next_rsl_obs = self._normalize_next_observations(
                    raw_next_rsl_obs
                )

                done_bool = done.bool()
                done_terms = info["done_terms"]
                _, motion_complete, failure = classify_mimickit_done_terms(done_bool, done_terms)
                # IsaacLab exposes truncation independently from termination;
                # a tracking failure on the 500th step is still bootstrapped.
                timeout = done_terms["time_out"].bool()
                motion_resample = info.get("motion_resample_mask")
                if not torch.is_tensor(motion_resample):
                    motion_resample = torch.zeros(n_envs, dtype=torch.bool, device=device)
                    motion_resample_ids = info.get("motion_wrap_env_ids")
                    if torch.is_tensor(motion_resample_ids) and motion_resample_ids.numel() > 0:
                        motion_resample[motion_resample_ids.long()] = True
                else:
                    motion_resample = motion_resample.to(device=device, dtype=torch.bool).reshape(n_envs)
                interval_push = info.get("interval_push_mask")
                if not torch.is_tensor(interval_push):
                    interval_push = torch.zeros(n_envs, dtype=torch.bool, device=device)
                else:
                    interval_push = interval_push.to(
                        device=device, dtype=torch.bool
                    ).reshape(n_envs)
                # This call intentionally preserves RSL-RL's official timeout
                # bootstrap and observation-normalizer update semantics.
                self.ppo.process_env_step(
                    next_rsl_obs,
                    rewards,
                    done_bool,
                    {"time_outs": timeout},
                )

                actions_buf[step_idx] = actions
                reward_buf[step_idx] = rewards
                done_buf[step_idx] = done_bool
                timeout_buf[step_idx] = timeout
                failure_buf[step_idx] = failure
                motion_complete_buf[step_idx] = motion_complete
                motion_resample_buf[step_idx] = motion_resample
                interval_push_buf[step_idx] = interval_push
                rollout_infos.append(info)
                self._record_episode_stats(rewards, done_bool)
                action_abs_max = max(action_abs_max, float(actions.abs().max().item()))

                for key, value in done_terms.items():
                    mask = value.bool()
                    done_terms_union[key] = (
                        mask.clone() if key not in done_terms_union else done_terms_union[key] | mask
                    )
                newly_done = (~ever_done) & done_bool
                if bool(newly_done.any()):
                    ids = newly_done.nonzero(as_tuple=False).squeeze(-1)
                    first_done_step[ids] = step_idx + 1
                    phase = info.get("termination_phase_steps")
                    if torch.is_tensor(phase):
                        first_done_phase[ids] = phase.long()[ids]
                    first_failure[ids] = failure[ids]
                    first_timeout[ids] = timeout[ids]
                    first_motion_complete[ids] = motion_complete[ids]
                    ever_done[ids] = True

                obs = next_obs
                rsl_obs = next_rsl_obs

            self.ppo.compute_returns(rsl_obs)

        self._obs = obs
        self._rsl_obs = rsl_obs
        return {
            "actions": actions_buf,
            "reward": reward_buf,
            "done": done_buf,
            "timeout": timeout_buf,
            "failure": failure_buf,
            "motion_complete": motion_complete_buf,
            "motion_resample": motion_resample_buf,
            "interval_push": interval_push_buf,
            "first_done_step": first_done_step,
            "first_done_phase": first_done_phase,
            "first_failure": first_failure,
            "first_timeout": first_timeout,
            "first_motion_complete": first_motion_complete,
            "done_terms_union": done_terms_union,
            "rollout_infos": rollout_infos,
            "collection_start_phases": start_phases,
            "action_abs_max": action_abs_max,
            "next_observation": obs,
        }

    def update(self, rollout: dict, collect_time: float) -> dict:
        start = time.perf_counter()
        learning_rate_before = float(self.ppo.learning_rate)
        losses = self.ppo.update()
        post_update_kl = self._post_update_kl()
        update_time = time.perf_counter() - start
        metrics = self._build_metrics(rollout)
        metrics.update(
            {
                "beyondmimic/value_loss": float(losses["value_function"]),
                "beyondmimic/surrogate_loss": float(losses["surrogate"]),
                "beyondmimic/entropy": float(losses["entropy"]),
                "beyondmimic/learning_rate": float(self.ppo.learning_rate),
                "beyondmimic/learning_rate_before": learning_rate_before,
                "beyondmimic/learning_rate_ratio": float(self.ppo.learning_rate)
                / max(learning_rate_before, 1.0e-12),
                "beyondmimic/post_update_kl": post_update_kl,
                "beyondmimic/noise_std_mean": float(self.actor_critic.std.detach().mean().item()),
                "beyondmimic/optimizer_steps": float(
                    int(self.cfg.num_learning_epochs) * int(self.cfg.num_mini_batches)
                ),
                "beyondmimic/last_minibatch_clipped_grad_norm": self._last_gradient_norm(),
                "timing/collect_s": float(collect_time),
                "timing/update_s": float(update_time),
                "method/beyondmimic": 1.0,
                "system/parameters_finite": float(self._parameters_finite()),
                "system/buffers_finite": float(self._buffers_finite()),
                "system/optimizer_finite": float(self._optimizer_finite()),
            }
        )
        self._add_normalizer_metrics(metrics)
        nonfinite_groups = [
            name
            for name in ("parameters", "buffers", "optimizer")
            if not bool(metrics[f"system/{name}_finite"])
        ]
        if nonfinite_groups:
            raise FloatingPointError(
                "BeyondMimic detected non-finite " + ", ".join(nonfinite_groups)
            )
        return metrics

    def _parameters_finite(self) -> bool:
        return all(bool(torch.isfinite(parameter).all()) for parameter in self.actor_critic.parameters())

    def _buffers_finite(self) -> bool:
        return all(
            not torch.is_floating_point(buffer) or bool(torch.isfinite(buffer).all())
            for buffer in self.actor_critic.buffers()
        )

    def _optimizer_finite(self) -> bool:
        for state in self.ppo.optimizer.state.values():
            for value in state.values():
                if (
                    torch.is_tensor(value)
                    and torch.is_floating_point(value)
                    and not bool(torch.isfinite(value).all())
                ):
                    return False
        return True

    @torch.no_grad()
    def _post_update_kl(self, chunk_size: int = 16384) -> float:
        """Full-rollout KL after PPO; diagnostic only, never used for updates."""
        storage = self.ppo.storage
        if storage is None:
            return 0.0
        policy_obs = storage.observations["policy"].reshape(-1, self.actor_obs_dim)
        old_mu = storage.mu.reshape(-1, self.action_dim)
        old_sigma = storage.sigma.reshape(-1, self.action_dim)
        kl_sum = torch.zeros((), dtype=torch.float64, device=self.env.device)
        count = 0
        for start in range(0, policy_obs.shape[0], chunk_size):
            stop = min(start + chunk_size, policy_obs.shape[0])
            new_mu = self.actor_critic.actor(policy_obs[start:stop])
            new_sigma = self.actor_critic.std.expand_as(new_mu)
            batch_kl = torch.sum(
                torch.log(new_sigma / old_sigma[start:stop] + 1.0e-5)
                + (
                    old_sigma[start:stop].square()
                    + (old_mu[start:stop] - new_mu).square()
                )
                / (2.0 * new_sigma.square())
                - 0.5,
                dim=-1,
            )
            kl_sum += batch_kl.double().sum()
            count += int(batch_kl.numel())
        return float((kl_sum / max(count, 1)).item())

    def _last_gradient_norm(self) -> float:
        squared_norm = 0.0
        for parameter in self.actor_critic.parameters():
            if parameter.grad is not None:
                squared_norm += float(parameter.grad.detach().float().square().sum().item())
        return math.sqrt(squared_norm)

    def _build_metrics(self, rollout: dict) -> dict[str, float]:
        rewards = rollout["reward"]
        actions = rollout["actions"]
        first_done = (
            rollout["first_failure"]
            | rollout["first_timeout"]
            | rollout["first_motion_complete"]
        )
        returns = rewards.sum(dim=0)
        motion_resample = rollout["motion_resample"]
        interval_push = rollout["interval_push"]
        metrics = {
            "rollout/task_reward_mean": float(rewards.mean().item()),
            "rollout/reward_step_mean": float(rewards.mean().item()),
            "rollout/return_mean": float(returns.mean().item()),
            "rollout/return_std": float(returns.std(unbiased=False).item()),
            "rollout/done_frac": float(rollout["done"].float().mean().item()),
            "rollout/failure_frac": float(rollout["failure"].float().mean().item()),
            "rollout/timeout_frac": float(rollout["timeout"].float().mean().item()),
            "rollout/motion_complete_frac": float(rollout["motion_complete"].float().mean().item()),
            "rollout/motion_resample_count": float(motion_resample.sum().item()),
            "rollout/motion_resample_frac": float(motion_resample.float().mean().item()),
            "rollout/motion_resample_env_frac": float(motion_resample.any(dim=0).float().mean().item()),
            "rollout/interval_push_count": float(interval_push.sum().item()),
            "rollout/interval_push_frac": float(interval_push.float().mean().item()),
            "rollout/interval_push_env_frac": float(interval_push.any(dim=0).float().mean().item()),
            "rollout/first_done_frac": float(first_done.float().mean().item()),
            "rollout/first_failure_frac": float(rollout["first_failure"].float().mean().item()),
            "rollout/first_timeout_frac": float(rollout["first_timeout"].float().mean().item()),
            "rollout/first_motion_complete_frac": float(
                rollout["first_motion_complete"].float().mean().item()
            ),
            "rollout/first_done_step_mean": float(rollout["first_done_step"].float().mean().item()),
            "phase/start_mean": float(rollout["collection_start_phases"].float().mean().item()),
            "phase/start_min": float(rollout["collection_start_phases"].min().item()),
            "phase/start_max": float(rollout["collection_start_phases"].max().item()),
            "act/abs_mean": float(actions.abs().mean().item()),
            "act/abs_p95": float(torch.quantile(actions.abs().flatten(), 0.95).item()),
            "act/abs_p99": float(torch.quantile(actions.abs().flatten(), 0.99).item()),
            "act/abs_max": float(actions.abs().max().item()),
            "act/abs_max_all": float(rollout["action_abs_max"]),
            "train/mean_reward": (
                float(sum(self._train_reward_buffer) / len(self._train_reward_buffer))
                if self._train_reward_buffer
                else 0.0
            ),
            "train/mean_episode_length": (
                float(sum(self._train_length_buffer) / len(self._train_length_buffer))
                if self._train_length_buffer
                else 0.0
            ),
            "train/recent_episode_count": float(len(self._train_reward_buffer)),
            "train/completed_episodes": float(self._train_completed_episodes),
        }
        for key, mask in rollout["done_terms_union"].items():
            metrics[f"done/{key}_frac"] = float(mask.float().mean().item())
        self._add_reward_metrics(metrics, rollout["rollout_infos"])
        self._add_action_group_metrics(metrics, actions)
        self._add_sampler_metrics(metrics)
        return metrics

    @staticmethod
    def _add_reward_metrics(metrics: dict[str, float], rollout_infos: list[dict]) -> None:
        if not rollout_infos:
            return
        first_terms = rollout_infos[0].get("reward_terms", {})
        for key, value in first_terms.items():
            if torch.is_tensor(value):
                metrics[f"reward/{key}_mean"] = float(value.float().mean().item())
        reward_sums: dict[str, float] = {}
        reward_counts: dict[str, int] = {}
        done_sums: dict[str, float] = {}
        done_counts: dict[str, int] = {}
        for info in rollout_infos:
            for key, value in info.get("reward_terms", {}).items():
                if torch.is_tensor(value):
                    reward_sums[key] = reward_sums.get(key, 0.0) + float(value.float().sum().item())
                    reward_counts[key] = reward_counts.get(key, 0) + int(value.numel())
            for key, value in info.get("done_terms", {}).items():
                if torch.is_tensor(value):
                    done_sums[key] = done_sums.get(key, 0.0) + float(value.float().sum().item())
                    done_counts[key] = done_counts.get(key, 0) + int(value.numel())
        for key, value in reward_sums.items():
            metrics[f"reward_rollout/{key}_mean"] = value / max(reward_counts[key], 1)
        for key, value in done_sums.items():
            metrics[f"done_rollout/{key}_frac"] = value / max(done_counts[key], 1)

    @staticmethod
    def _add_action_group_metrics(metrics: dict[str, float], actions: torch.Tensor) -> None:
        act_abs = actions.abs().mean(dim=(0, 1))
        if act_abs.numel() < 29:
            return
        metrics["act/legs_abs"] = float(act_abs[:12].mean().item())
        metrics["act/waist_abs"] = float(act_abs[12:15].mean().item())
        metrics["act/arms_abs"] = float(act_abs[15:29].mean().item())

    def _add_sampler_metrics(self, metrics: dict[str, float]) -> None:
        stats = self.env.adaptive_sampling_stats()
        for key, value in stats.items():
            value_f = float(value)
            if math.isfinite(value_f):
                metrics[f"sampler/{key}"] = value_f
        top_bin = metrics.get("sampler/top_bin")
        bin_count = metrics.get("sampler/bin_count")
        if top_bin is not None and bin_count is not None and bin_count > 0.0:
            metrics["sampler/top_bin_frac"] = top_bin / bin_count
        if "sampler/bin_count" not in metrics:
            bin_count = getattr(getattr(self.env, "adaptive_sampler", None), "num_bins", None)
            if bin_count is not None:
                metrics["sampler/bin_count"] = float(bin_count)

    def _add_normalizer_metrics(self, metrics: dict[str, float]) -> None:
        if not bool(self.cfg.empirical_normalization):
            metrics["normalizer/enabled"] = 0.0
            return
        actor_normalizer = self.actor_obs_normalizer
        critic_normalizer = self.critic_obs_normalizer
        metrics.update(
            {
                "normalizer/enabled": 1.0,
                "normalizer/actor_count": float(actor_normalizer.count.item()),
                "normalizer/critic_count": float(critic_normalizer.count.item()),
                "normalizer/actor_mean_abs": float(actor_normalizer.mean.abs().mean().item()),
                "normalizer/actor_std_mean": float(actor_normalizer.std.mean().item()),
                "normalizer/critic_mean_abs": float(critic_normalizer.mean.abs().mean().item()),
                "normalizer/critic_std_mean": float(critic_normalizer.std.mean().item()),
            }
        )

    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        print(
            f"[UPDATE] {update_idx}/{max_updates} "
            f"reward_step={metrics['rollout/reward_step_mean']:.5f} "
            f"return={metrics['rollout/return_mean']:.5f} "
            f"done={metrics['rollout/done_frac']:.5f} "
            f"ep_len={metrics['train/mean_episode_length']:.2f}",
            flush=True,
        )
        print(
            f"[BEYONDMIMIC_PPO] surrogate={metrics['beyondmimic/surrogate_loss']:.5f} "
            f"value={metrics['beyondmimic/value_loss']:.5f} "
            f"entropy={metrics['beyondmimic/entropy']:.5f} "
            f"std={metrics['beyondmimic/noise_std_mean']:.4f} "
            f"post_kl={metrics['beyondmimic/post_update_kl']:.5f} "
            f"grad_last={metrics['beyondmimic/last_minibatch_clipped_grad_norm']:.4f} "
            f"lr={metrics['beyondmimic/learning_rate']:.6f}",
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
            f"[PUSH] count={metrics['rollout/interval_push_count']:.0f} "
            f"env_frac={metrics['rollout/interval_push_env_frac']:.5f}",
            flush=True,
        )
        print(
            f"[TIME] collect={metrics['timing/collect_s']:.3f}s "
            f"update={metrics['timing/update_s']:.3f}s",
            flush=True,
        )

    def log_banner(self) -> None:
        print(
            "[METHOD] name=beyondmimic actor=rsl_rl.ActorCritic "
            "algorithm=rsl_rl.PPO reward=task_tracking",
            flush=True,
        )
        print(
            f"[ARCH] actor_obs={self.actor_obs_dim} critic_obs={self.critic_obs_dim} "
            f"action_dim={self.action_dim} hidden_actor={list(self.cfg.actor_hidden_dims)} "
            f"hidden_critic={list(self.cfg.critic_hidden_dims)} activation={self.cfg.activation}",
            flush=True,
        )
        print(
            f"[BEYONDMIMIC_OFFICIAL] "
            f"rollout={self.rollout_steps} epochs={self.cfg.num_learning_epochs} "
            f"mini_batches={self.cfg.num_mini_batches} gamma={self.cfg.gamma} lam={self.cfg.lam} "
            f"clip={self.cfg.clip_param} entropy={self.cfg.entropy_coef} "
            f"schedule={self.cfg.schedule} desired_kl={self.cfg.desired_kl} "
            f"normalization={int(bool(self.cfg.empirical_normalization))} "
            f"official_rsl_rl={OFFICIAL_RSL_RL_VERSION} "
            f"runtime_rsl_rl={self._rsl_rl_version()} "
            f"normalization_boundary=official_rsl_rl_2.3.3_compat "
            f"until={OFFICIAL_NORMALIZER_UNTIL}",
            flush=True,
        )
        sampler = self.env.adaptive_sampling_stats()
        print(
            f"[BEYONDMIMIC_ENV] control_hz={1.0 / self.env.dt:.1f} "
            f"physics_hz={1.0 / self.env.physics_dt:.1f} "
            f"termination=anchor_z0.25/gravity_z0.8/ee_z0.25 "
            f"motion_end=resample_command sampler_bins={int(sampler['bin_count'])} "
            f"startup_randomization={int(bool(self.env.startup_randomization_applied))} "
            f"reset_noise={int(bool(self.env.reset_noise))} "
            f"observation_noise={int(bool(self.env.observation_noise))} "
            f"interval_pushes={int(bool(self.env.interval_pushes))} "
            "push_timer=isaaclab_2.1.0_global_continuous_1_3s",
            flush=True,
        )


BeyondMimic = Beyondmimic
