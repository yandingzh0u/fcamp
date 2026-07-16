from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from components.imitation.style_reward import discriminator_style_reward, style_reward_statistics
from components.normalization.diff_stats import DiffNormalizer
from components.normalization.running_stats import RunningNormalizer
from components.replay.sample_buffer import SampleReplayBuffer
from envs.motion import ADD_TARGET_OBS_STEPS
from method.amp import AMP, _AMPActor, _AMPCritic, _MimicKitIndexSampler
from method.base import classify_mimickit_done_terms
from models.style_discriminator import StyleDiscriminator


@dataclass
class _ADDDiscLossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


def _gradient_norm_sq(logits: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
    gradient = torch.autograd.grad(
        logits,
        observations,
        grad_outputs=torch.ones_like(logits),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return gradient.reshape(gradient.shape[0], -1).square().sum(dim=-1)


def _distribution_metrics(prefix: str, logits: torch.Tensor) -> dict[str, torch.Tensor]:
    detached = logits.detach().float().reshape(-1)
    prob = torch.sigmoid(detached)
    q = torch.quantile(detached, torch.tensor([0.05, 0.5, 0.95], device=detached.device))
    return {
        f"disc/{prefix}_logit_mean": detached.mean(),
        f"disc/{prefix}_logit_std": detached.std(unbiased=False),
        f"disc/{prefix}_logit_p05": q[0],
        f"disc/{prefix}_logit_p50": q[1],
        f"disc/{prefix}_logit_p95": q[2],
        f"disc/{prefix}_prob_mean": prob.mean(),
    }


class ADD(AMP):
    """MimicKit ADD: PPO actor-critic plus discriminator over demo-policy diffs."""

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        self.actor_obs_dim = int(env.get_add_policy_observation().shape[-1])
        self.critic_obs_dim = self.actor_obs_dim
        self.action_dim = int(env.action_dim)
        self.rollout_steps = int(cfg.rollout_env_steps)
        self.disc_obs_steps = int(cfg.disc_obs_steps)
        if self.disc_obs_steps != 1:
            raise ValueError("MimicKit ADD G1 uses parameters.disc_obs_steps=1")
        self.add_disc_obs_dim = int(env.add_disc_frame_dim)
        self.add_disc_body_count = len(getattr(env, "add_disc_body_names", ()))
        self.add_disc_joint_rot_dim = self.add_disc_obs_dim - (
            3 + 6 + 3 * self.add_disc_body_count + 3 + 3 + self.action_dim
        )
        if self.add_disc_joint_rot_dim <= 0 or self.add_disc_joint_rot_dim % 6 != 0:
            raise ValueError(
                "ADD discriminator schema is inconsistent: "
                f"obs={self.add_disc_obs_dim} bodies={self.add_disc_body_count} action={self.action_dim}"
            )
        self.pos_diff = torch.zeros(self.add_disc_obs_dim, dtype=torch.float32, device=env.device)
        self._build_action_normalizer()

        self.empirical_normalization = bool(cfg.empirical_normalization)
        if self.empirical_normalization:
            self.obs_normalizer: nn.Module = RunningNormalizer(self.actor_obs_dim, device=env.device, clip=10.0)
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
            self.add_disc_obs_dim,
            tuple(cfg.disc_hidden_dims),
        ).to(env.device)
        self.disc_normalizer = DiffNormalizer(self.add_disc_obs_dim, device=env.device)
        self.disc_pair_replay = SampleReplayBuffer(
            int(cfg.disc_buffer_size),
            2 * self.add_disc_obs_dim,
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
        self.normalizer_sample_count = 0
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

    def extra_checkpoint_state(self) -> dict:
        return {
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),
            "disc_pair_replay": self.disc_pair_replay.state_dict(),
            "disc_version": int(self.disc_version),
            "normalizer_sample_count": int(self.normalizer_sample_count),
            "add_schema_version": 5,
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
        inferred_count = (
            int(self.obs_normalizer.count.item()) if self.empirical_normalization else 0
        )
        self.normalizer_sample_count = int(
            payload.get("normalizer_sample_count", inferred_count)
        )
        replay_loaded = self.disc_pair_replay.load_state_dict(payload.get("disc_pair_replay"))
        if not replay_loaded:
            print("[ADD] discriminator pair replay absent/incompatible; starting empty", flush=True)
        replay_size = int(self.disc_pair_replay.statistics()["replay/size"])
        print(
            f"[ADD_RESUME_STATE] disc_version={self.disc_version} "
            f"normalizer_samples={self.normalizer_sample_count} "
            f"obs_count={inferred_count} diff_count={int(self.disc_normalizer.count.item())} "
            f"pair_replay_size={replay_size} replay_loaded={int(replay_loaded)}",
            flush=True,
        )

    def initial_reset(self) -> torch.Tensor:
        obs = self.env.reset()
        if bool(self.cfg.init_at_random_ep_len) and self.env.max_episode_steps > 0:
            self.env.episode_steps = torch.randint_like(
                self.env.episode_steps,
                high=int(self.env.max_episode_steps),
            )
        self._obs = obs
        return obs

    def _amp_observation(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.env.get_add_policy_observation(env_ids)

    @torch.no_grad()
    def _timeout_bootstrap_value(self, info: dict, timeout: torch.Tensor) -> torch.Tensor:
        values = torch.zeros(self.env.num_envs, device=self.env.device)
        ids = timeout.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return values
        final_obs = info.get("add_policy_observation")
        if torch.is_tensor(final_obs):
            critic_obs = final_obs.index_select(0, ids)
        else:
            critic_obs = self._amp_observation(ids)
        values[ids] = self.critic(self._norm_critic(critic_obs, update=False))
        return values

    @torch.no_grad()
    def _evaluate_add_reward(self, diff_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = max(1, int(self.cfg.disc_eval_batch_size))
        logits: list[torch.Tensor] = []
        self.discriminator.eval()
        for start in range(0, diff_obs.shape[0], batch_size):
            batch = diff_obs[start : start + batch_size]
            logits.append(self.discriminator(self.disc_normalizer.normalize(batch)))
        all_logits = torch.cat(logits, dim=0)
        rewards = discriminator_style_reward(
            all_logits,
            scale=float(self.cfg.disc_reward_scale),
            minimum_one_minus_prob=float(self.cfg.disc_reward_epsilon),
        )
        return all_logits, rewards

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
        first_failure = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first_timeout = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first_motion_complete = torch.zeros(n_envs, dtype=torch.bool, device=device)
        done_terms_union: dict[str, torch.Tensor] = {}
        rollout_info_items: list[tuple[dict, torch.Tensor]] = []
        first_infos: list[dict] = []
        disc_pair_chunks: list[torch.Tensor] = []
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

                policy_disc_obs = info.get("add_policy_disc_frame")
                if policy_disc_obs is None:
                    policy_disc_obs = env.get_add_policy_disc_frame()
                demo_disc_obs = info.get("add_demo_disc_frame")
                if demo_disc_obs is None:
                    demo_disc_obs = env.get_add_demo_disc_frame(info["reference_phase_steps"])
                diff_obs = demo_disc_obs - policy_disc_obs
                disc_logits, disc_reward = self._evaluate_add_reward(diff_obs)
                mixed_reward = (
                    float(self.cfg.task_reward_weight) * task_reward
                    + float(self.cfg.disc_reward_weight) * disc_reward
                )

                done_bool = done.bool()
                done_terms = info["done_terms"]
                timeout, motion_complete, failure = classify_mimickit_done_terms(done_bool, done_terms)
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
                disc_pair_chunks.append(
                    torch.cat((policy_disc_obs, demo_disc_obs), dim=-1).detach().to("cpu", dtype=torch.float32)
                )

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
                    first_failure[ids] = failure[ids]
                    first_timeout[ids] = timeout[ids]
                    first_motion_complete[ids] = motion_complete[ids]
                    ever_done[ids] = True
                rollout_info_items.append((info, torch.ones(n_envs, dtype=torch.bool, device=device)))
                self._record_episode_stats(mixed_reward, done_bool)
                action_abs_max = max(action_abs_max, float(step_action.abs().max().item()))
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
            "first_failure": first_failure,
            "first_timeout": first_timeout,
            "first_motion_complete": first_motion_complete,
            "collection_start_phases": start_phases,
            "action_abs_max": action_abs_max,
            "disc_pairs": torch.cat(disc_pair_chunks, dim=0),
            "next_observation": obs,
        }

    def _split_pairs(self, pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pairs.ndim != 2 or pairs.shape[-1] != 2 * self.add_disc_obs_dim:
            raise ValueError(f"ADD pair tensor must have shape [B,{2 * self.add_disc_obs_dim}], got {tuple(pairs.shape)}")
        return pairs[:, : self.add_disc_obs_dim], pairs[:, self.add_disc_obs_dim :]

    @staticmethod
    def _add_ppo_metric_aliases(metrics: dict[str, float]) -> dict[str, float]:
        return {key.replace("amp/", "add/", 1): value for key, value in metrics.items() if key.startswith("amp/")}

    @torch.no_grad()
    def _record_disc_normalizer(self, policy_cpu: torch.Tensor, demo_cpu: torch.Tensor) -> int:
        self.disc_normalizer.clear_pending()
        if not self._need_normalizer_update():
            return 0
        batch_size = max(1, int(self.cfg.disc_eval_batch_size))
        count = int(policy_cpu.shape[0])
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            diff = demo_cpu[start:end].to(device=self.env.device) - policy_cpu[start:end].to(device=self.env.device)
            self.disc_normalizer.record(diff)
        return count

    @torch.no_grad()
    def _store_disc_replay_data(self, policy_cpu: torch.Tensor, demo_cpu: torch.Tensor) -> int:
        count = int(policy_cpu.shape[0])
        if count == 0:
            return 0
        if self.disc_pair_replay.is_full:
            keep = min(count, int(self.cfg.disc_replay_samples))
        else:
            keep = count
        indices = torch.randperm(count, device="cpu")[:keep]
        self.disc_pair_replay.push(torch.cat((policy_cpu.index_select(0, indices), demo_cpu.index_select(0, indices)), dim=-1))
        return int(keep)

    def _diff_component_metrics(self, diff_samples: torch.Tensor) -> dict[str, float]:
        joint_dim = self.add_disc_joint_rot_dim
        body_dim = 3 * self.add_disc_body_count
        if body_dim <= 0 or body_dim % 3 != 0:
            return {}
        layout = (
            ("root_pos", 3),
            ("root_rot", 6),
            ("joint_pos", joint_dim),
            ("body_pos", body_dim),
            ("root_lin_vel", 3),
            ("root_ang_vel", 3),
            ("joint_vel", self.action_dim),
        )
        out: dict[str, float] = {"disc_diff/body_count": float(body_dim // 3)}
        offset = 0
        for name, width in layout:
            chunk = diff_samples[:, offset : offset + width]
            out[f"disc_diff/{name}_abs_mean"] = float(chunk.abs().mean().item())
            offset += width
        return out

    def _compute_add_disc_loss(
        self,
        *,
        policy_obs: torch.Tensor,
        demo_obs: torch.Tensor,
        replay_pairs: torch.Tensor,
    ) -> _ADDDiscLossOutput:
        current_diff = demo_obs - policy_obs
        replay_policy, replay_demo = self._split_pairs(replay_pairs)
        diff_obs = torch.cat((current_diff, replay_demo - replay_policy), dim=0)

        pos_diff = self.pos_diff.clone().unsqueeze(0).requires_grad_(float(self.cfg.disc_grad_penalty) > 0.0)
        neg_diff = self.disc_normalizer.normalize(diff_obs).detach().requires_grad_(
            float(self.cfg.disc_grad_penalty) > 0.0
        )
        pos_logits = self.discriminator(pos_diff)
        neg_logits = self.discriminator(neg_diff)
        current_logits = neg_logits[: policy_obs.shape[0]]
        replay_logits = neg_logits[policy_obs.shape[0] :]

        pos_bce = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_bce = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        bce = 0.5 * (pos_bce + neg_bce)
        logit_regularization = self.discriminator.logit_weights().square().sum()
        if float(self.cfg.disc_grad_penalty) > 0.0:
            pos_gp = _gradient_norm_sq(pos_logits, pos_diff).mean()
            neg_gp = _gradient_norm_sq(neg_logits, neg_diff).mean()
            gradient_penalty = 0.5 * (pos_gp + neg_gp)
        else:
            zero = bce.new_zeros(())
            pos_gp = neg_gp = gradient_penalty = zero
        total = (
            bce
            + float(self.cfg.disc_logit_reg) * logit_regularization
            + float(self.cfg.disc_grad_penalty) * gradient_penalty
        )

        current_bce = F.binary_cross_entropy_with_logits(current_logits, torch.zeros_like(current_logits))
        replay_bce = F.binary_cross_entropy_with_logits(replay_logits, torch.zeros_like(replay_logits))
        metrics: dict[str, torch.Tensor] = {
            "disc/loss": total.detach(),
            "disc/bce": bce.detach(),
            "disc/pos_bce": pos_bce.detach(),
            "disc/neg_bce": neg_bce.detach(),
            "disc/current_bce": current_bce.detach(),
            "disc/replay_bce": replay_bce.detach(),
            "disc/gradient_penalty": gradient_penalty.detach(),
            "disc/pos_gradient_penalty": pos_gp.detach(),
            "disc/neg_gradient_penalty": neg_gp.detach(),
            "disc/logit_regularization": logit_regularization.detach(),
            "disc/pos_accuracy": (pos_logits.detach() > 0).float().mean(),
            "disc/neg_accuracy": (neg_logits.detach() < 0).float().mean(),
            "disc/current_accuracy": (current_logits.detach() < 0).float().mean(),
            "disc/replay_accuracy": (replay_logits.detach() < 0).float().mean(),
        }
        metrics.update(_distribution_metrics("pos", pos_logits))
        metrics.update(_distribution_metrics("current", current_logits))
        metrics.update(_distribution_metrics("replay", replay_logits))
        metrics.update(_distribution_metrics("neg", neg_logits))
        return _ADDDiscLossOutput(loss=total, metrics=metrics)

    def _disc_update(self, pair_cpu: torch.Tensor, sampler: _MimicKitIndexSampler) -> dict[str, float]:
        count = int(pair_cpu.shape[0])
        if count == 0:
            return {"disc/update_steps": 0.0, "disc/skipped_no_current": 1.0}
        batch_size = self._mimickit_batch_size(int(self.cfg.disc_batch_size))
        update_steps = int(math.ceil(count / batch_size)) * int(self.cfg.disc_epochs)
        totals: dict[str, float] = {}
        grad_total = 0.0
        self.discriminator.train()
        for _ in range(update_steps):
            idx = sampler.sample(batch_size).to(device="cpu")
            pair = pair_cpu.index_select(0, idx).to(device=self.env.device, dtype=torch.float32)
            policy_obs, demo_obs = self._split_pairs(pair)
            replay_pairs = self.disc_pair_replay.sample(batch_size, device=self.env.device, dtype=torch.float32)
            output = self._compute_add_disc_loss(
                policy_obs=policy_obs,
                demo_obs=demo_obs,
                replay_pairs=replay_pairs,
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
        pair_cpu = rollout["disc_pairs"]
        policy_cpu, demo_cpu = self._split_pairs(pair_cpu)
        normalizer_update_enabled = self._need_normalizer_update()
        normalizer_samples = self._record_disc_normalizer(policy_cpu, demo_cpu)
        replay_added = self._store_disc_replay_data(policy_cpu, demo_cpu)
        sampler = _MimicKitIndexSampler(int(pair_cpu.shape[0]), device=self.env.device)
        critic_start = time.perf_counter()
        critic_metrics = self._add_ppo_metric_aliases(self._critic_update(rollout, sampler))
        critic_time = time.perf_counter() - critic_start
        actor_start = time.perf_counter()
        actor_metrics = self._add_ppo_metric_aliases(self._actor_update(rollout, sampler))
        actor_time = time.perf_counter() - actor_start
        disc_start = time.perf_counter()
        disc_metrics = self._disc_update(pair_cpu, sampler)
        disc_time = time.perf_counter() - disc_start
        obs_committed = self.obs_normalizer.commit() if self.empirical_normalization else False
        committed = self.disc_normalizer.commit()
        self._advance_normalizer_sample_count(int(pair_cpu.shape[0]))

        metrics = self._build_metrics(rollout, collect_time, time.perf_counter() - start)
        metrics.update(critic_metrics)
        metrics.update(actor_metrics)
        metrics.update(disc_metrics)
        diff_samples = (demo_cpu[: min(8192, demo_cpu.shape[0])] - policy_cpu[: min(8192, policy_cpu.shape[0])]).to(
            device=self.env.device
        )
        metrics.update(self.disc_normalizer.statistics(diff_samples))
        if self.empirical_normalization:
            metrics.update(self.obs_normalizer.statistics(prefix="obs_norm"))
        metrics.update(
            {
                key.replace("replay/", "disc_pair_replay/"): value
                for key, value in self.disc_pair_replay.statistics().items()
            }
        )
        metrics.update(
            {
                "disc_diff/raw_abs_mean": float(diff_samples.abs().mean().item()),
                "disc_diff/raw_abs_p95": float(torch.quantile(diff_samples.abs().flatten(), 0.95).item()),
                "disc_diff_norm/committed_this_update": float(committed),
                "obs_norm/committed_this_update": float(obs_committed),
                "disc_diff_norm/samples_update": float(normalizer_samples),
                "normalizer/update_enabled": float(normalizer_update_enabled),
                "normalizer/sample_count": float(self.normalizer_sample_count),
                "normalizer/sample_limit": float(self.cfg.normalizer_samples),
                "disc_pair_replay/added_this_update": float(replay_added),
                "timing/collect_s": float(collect_time),
                "timing/critic_update_s": float(critic_time),
                "timing/actor_update_s": float(actor_time),
                "timing/disc_update_s": float(disc_time),
                "timing/update_s": float(time.perf_counter() - start),
                "method/add": 1.0,
                "system/parameters_finite": float(self._parameters_finite()),
            }
        )
        metrics.update(self._diff_component_metrics(diff_samples))
        if not bool(metrics["system/parameters_finite"]):
            raise FloatingPointError("ADD detected non-finite trainable parameters")
        return metrics

    def _build_metrics(self, rollout: dict, collect_time: float, update_time: float) -> dict:
        metrics = super()._build_metrics(rollout, collect_time, update_time)
        metrics = {key: value for key, value in metrics.items() if not key.startswith("amp_reward/")}
        metrics.update(
            style_reward_statistics(
                rollout["disc_logits"].reshape(-1),
                rollout["disc_reward"].reshape(-1),
                scale=float(self.cfg.disc_reward_scale),
                minimum_one_minus_prob=float(self.cfg.disc_reward_epsilon),
                prefix="add_reward",
            )
        )
        return metrics

    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        print(
            f"[UPDATE] {update_idx}/{max_updates} "
            f"task={metrics.get('rollout/task_reward_mean', float('nan')):.5f} "
            f"add={metrics.get('rollout/disc_reward_mean', float('nan')):.5f} "
            f"mixed={metrics.get('rollout/mixed_reward_mean', float('nan')):.5f} "
            f"done={metrics.get('rollout/done_frac', float('nan')):.5f} "
            f"ep_len={metrics.get('train/mean_episode_length', float('nan')):.2f}",
            flush=True,
        )
        print(
            f"[ADD_PPO] actor={metrics.get('add/actor_policy_loss', float('nan')):.5f} "
            f"critic={metrics.get('add/critic_loss', float('nan')):.5f} "
            f"ratio={metrics.get('add/ratio', float('nan')):.4f} "
            f"clip={metrics.get('add/clip_fraction', float('nan')):.4f} "
            f"kl={metrics.get('add/kl', float('nan')):.6f} "
            f"bound={metrics.get('add/action_bound_loss', float('nan')):.5f}",
            flush=True,
        )
        print(
            f"[ADD_DISC] loss={metrics.get('disc/loss', float('nan')):.5f} "
            f"bce={metrics.get('disc/bce', float('nan')):.5f} "
            f"gp={metrics.get('disc/gradient_penalty', float('nan')):.5f} "
            f"acc0={metrics.get('disc/pos_accuracy', float('nan')):.3f} "
            f"accD={metrics.get('disc/current_accuracy', float('nan')):.3f} "
            f"p0={metrics.get('disc/pos_prob_mean', float('nan')):.3f} "
            f"pD={metrics.get('disc/current_prob_mean', float('nan')):.3f} "
            f"logitD={metrics.get('disc/current_logit_mean', float('nan')):.3f} "
            f"rew={metrics.get('add_reward/mean', float('nan')):.5f} "
            f"diff={metrics.get('disc_diff/raw_abs_mean', float('nan')):.5f} "
            f"replay={metrics.get('disc_pair_replay/size', float('nan')):.0f}",
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
            "[METHOD] name=add actor=ppo_gaussian_fixed_std prior=mimickit_add_diff_discriminator "
            "credit=gae optimizer=sgd_momentum",
            flush=True,
        )
        print(
            f"[ARCH] actor_obs={self.actor_obs_dim} critic_obs={self.critic_obs_dim} "
            f"action_dim={self.action_dim} hidden_actor={list(self.cfg.actor_hidden_dims)} "
            f"hidden_critic={list(self.cfg.critic_hidden_dims)} hidden_disc={list(self.cfg.disc_hidden_dims)} "
            f"W_D={self.disc_obs_steps} add_disc_obs={self.add_disc_obs_dim}",
            flush=True,
        )
        add_body_names = getattr(self.env, "add_disc_body_names", ())
        print(
            f"[ADD_BODY_SCHEMA] count={len(add_body_names)} source=official_g1_mjcf "
            "demo_body_pos=urdf_fk "
            f"first={add_body_names[0] if add_body_names else 'none'} "
            f"last={add_body_names[-1] if add_body_names else 'none'}",
            flush=True,
        )
        print(
            f"[ADD_OFFICIAL] rollout={self.cfg.rollout_env_steps} "
            f"actor_epochs={self.cfg.actor_epochs} critic_epochs={self.cfg.critic_epochs} "
            f"disc_epochs={self.cfg.disc_epochs} action_std={self.cfg.action_std} "
            f"task_w={self.cfg.task_reward_weight} disc_w={self.cfg.disc_reward_weight} "
            f"positive=zeros negative=demo_minus_policy normalizer=mean_abs_diff",
            flush=True,
        )
        print(
            f"[ADD_NORMALIZER] samples={self.normalizer_sample_count} "
            f"limit={self.cfg.normalizer_samples} "
            f"update_enabled={int(self._need_normalizer_update())}",
            flush=True,
        )
        preview_ms = [1000.0 * self.env.dt * step for step in ADD_TARGET_OBS_STEPS]
        print(
            f"[ADD_TIME] control_hz={1.0 / self.env.dt:.1f} "
            f"physics_hz={1.0 / self.env.physics_dt:.1f} motion_fps={self.env.motion.fps:.1f} "
            f"phase_delta={self.env.motion_frame_delta:.6f} "
            f"preview_ms={preview_ms} continuous_reset=1 velocity=forward_root_link_floor",
            flush=True,
        )
        print(
            f"[ADD_ACTION_SPACE] mean_abs={self.action_norm_mean.abs().mean().item():.4f} "
            f"std_mean={self.action_norm_std.mean().item():.4f} "
            f"std_min={self.action_norm_std.min().item():.4f} "
            f"std_max={self.action_norm_std.max().item():.4f}",
            flush=True,
        )
        print(
            f"[ADD_TERMINATION] mode={getattr(self.env, 'termination_mode', 'unknown')} "
            f"motion_end_terminal={int(bool(getattr(self.env, 'terminate_on_motion_end', False)))} "
            f"task_contacts=1 pose_termination={int(getattr(self.env, 'termination_mode', '') == 'add')}",
            flush=True,
        )
        add_contact_count = int(getattr(self.env, "add_undesired_contact_body_ids").numel())
        amp_contact_count = int(getattr(self.env, "amp_undesired_contact_body_ids").numel())
        print(
            f"[ADD_CONTACT] undesired={add_contact_count} source=task_contact_bodies "
            f"matches_amp={int(add_contact_count == amp_contact_count)} "
            "source_forces=mimickit_isaaclab_ground_filter ground_kinematic=1",
            flush=True,
        )


Add = ADD
