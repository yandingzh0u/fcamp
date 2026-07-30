"""Pure adversarial motion prior training adapted to the local G1 and motion.

The H=1 path is the standard MimicKit AMP algorithm.  H>1 is a deliberately
small extension: the same MLP emits a chunk of independent *absolute*
normalized actions, and PPO treats the executed chunk (or terminal prefix) as
one variable-duration semi-MDP decision. Actor and discriminator inputs remain
reference-free, and exploration uses the fixed AMP Gaussian.
"""

from __future__ import annotations

from collections import deque
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from components.credit.amp_chunk_credit import (
    compute_amp_chunk_gae,
    normalize_and_clip_amp_advantages,
)
from components.imitation.amp_discriminator import AMPDiscriminator
from components.imitation.style_reward import style_reward_statistics
from components.imitation.temporal_history import TemporalFeatureHistory
from components.imitation.window_pipeline import TemporalWindowPipeline
from components.replay.amp_window_buffer import AMPWindowReplay
from components.rollout.amp_gaussian_base import AMPGaussianBase
from components.rollout.amp_contract import AMP_CHECKPOINT_CONTRACT
from components.rollout.amp_diagnostics import (
    AMPDiagnosticsMixin,
    _ensure_horizon_metrics,
    gaussian_action_noise_statistics,
)


def _flat_stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {
            f"{prefix}/count": 0.0,
            f"{prefix}/mean": -1.0,
            f"{prefix}/std": -1.0,
            f"{prefix}/min": -1.0,
            f"{prefix}/max": -1.0,
            f"{prefix}/p05": -1.0,
            f"{prefix}/p50": -1.0,
            f"{prefix}/p95": -1.0,
        }
    q = torch.quantile(flat, flat.new_tensor((0.05, 0.50, 0.95)))
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


def _grad_norm(parameters) -> torch.Tensor:
    parameters = tuple(parameters)
    reference = next((parameter for parameter in parameters), None)
    if reference is None:
        return torch.zeros(())
    total = reference.new_zeros(())
    for parameter in parameters:
        if parameter.grad is not None:
            total.add_(parameter.grad.detach().square().sum())
    return torch.sqrt(total)


def _endpoint_total_variation(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    support_size: int,
) -> float:
    """Discrete total-variation distance on the motion-frame support."""

    if first.numel() == 0 or second.numel() == 0:
        return -1.0
    if support_size < 1:
        raise ValueError("support_size must be positive")

    def occupancy(values: torch.Tensor) -> torch.Tensor:
        indices = (
            values.detach()
            .float()
            .round()
            .long()
            .clamp(0, support_size - 1)
            .cpu()
        )
        histogram = torch.bincount(
            indices,
            minlength=support_size,
        ).float()
        return histogram / histogram.sum()

    return float(
        0.5 * (occupancy(first) - occupancy(second)).abs().sum().item()
    )


def _action_delta_statistics(primitive: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compare primitive action changes within chunks and at chunk seams."""

    actions = primitive["actions"]
    offsets = primitive["offsets"]
    done = primitive["done"]
    if actions.shape[0] < 2:
        return {
            "action_delta/within_chunk_count": 0.0,
            "action_delta/within_chunk_rms": -1.0,
            "action_delta/within_chunk_p95": -1.0,
            "action_delta/chunk_boundary_count": 0.0,
            "action_delta/chunk_boundary_rms": -1.0,
            "action_delta/chunk_boundary_p95": -1.0,
            "action_delta/boundary_to_within_ratio": -1.0,
        }

    delta = actions[1:] - actions[:-1]
    same_episode = ~done[:-1]
    boundary = same_episode & (offsets[1:] == 0)
    within = same_episode & (offsets[1:] > 0)

    def summarize(mask: torch.Tensor, prefix: str) -> dict[str, float]:
        selected = delta[mask]
        if selected.numel() == 0:
            return {
                f"{prefix}_count": 0.0,
                f"{prefix}_rms": -1.0,
                f"{prefix}_p95": -1.0,
            }
        per_transition_rms = selected.float().square().mean(dim=-1).sqrt()
        return {
            f"{prefix}_count": float(per_transition_rms.numel()),
            f"{prefix}_rms": float(
                selected.float().square().mean().sqrt().item()
            ),
            f"{prefix}_p95": float(
                torch.quantile(per_transition_rms, 0.95).item()
            ),
        }

    metrics = {}
    metrics.update(summarize(within, "action_delta/within_chunk"))
    metrics.update(summarize(boundary, "action_delta/chunk_boundary"))
    within_rms = metrics["action_delta/within_chunk_rms"]
    boundary_rms = metrics["action_delta/chunk_boundary_rms"]
    metrics["action_delta/boundary_to_within_ratio"] = (
        boundary_rms / within_rms
        if within_rms > 1.0e-12 and boundary_rms >= 0.0
        else -1.0
    )
    return metrics


class AMP(AMPDiagnosticsMixin, AMPGaussianBase):
    """Standard unconditional pure AMP with a configurable action horizon."""

    _EXPERT_SEED_OFFSET = 0x414D50

    def build(self) -> None:
        super().build()
        cfg = self.cfg
        amp_cfg = cfg.style_prior
        env = self.env

        self.action_low = torch.full(
            (self.num_act,), -1.0, device=env.device
        )
        self.action_high = torch.full(
            (self.num_act,), 1.0, device=env.device
        )
        env.enable_strict_action_contract(self.action_low, self.action_high)

        self.imitation_history_steps = int(amp_cfg.obs_steps)
        self.imitation_frame_dim = int(env.imitation_frame_dim)
        self.imitation_window_dim = (
            self.imitation_history_steps * self.imitation_frame_dim
        )
        self.imitation_pipeline = TemporalWindowPipeline(
            self.imitation_history_steps,
            self.imitation_frame_dim,
        )
        self.imitation_history = TemporalFeatureHistory(
            env.num_envs,
            self.imitation_history_steps,
            self.imitation_frame_dim,
            device=env.device,
        )
        self.amp_discriminator = AMPDiscriminator(
            self.imitation_window_dim,
            hidden_dims=tuple(amp_cfg.hidden_dims),
            learning_rate=float(amp_cfg.learning_rate),
            weight_decay=float(amp_cfg.weight_decay),
            gradient_penalty_weight=float(amp_cfg.grad_penalty),
            logit_regularization_weight=float(amp_cfg.logit_reg),
            reward_scale=float(amp_cfg.reward_scale),
            reward_epsilon=float(amp_cfg.reward_epsilon),
            normalizer_clip=float(amp_cfg.normalizer_clip),
            device=env.device,
        )
        self.discriminator = self.amp_discriminator.discriminator
        self.disc_normalizer = self.amp_discriminator.normalizer
        self.disc_optimizer = self.amp_discriminator.optimizer
        self.disc_window_replay = AMPWindowReplay(
            int(amp_cfg.replay_size),
            self.imitation_history_steps,
            self.imitation_frame_dim,
        )

        self.expert_sampling_seed = int(
            (torch.initial_seed() + self._EXPERT_SEED_OFFSET)
            % ((1 << 63) - 1)
        )
        self.expert_sampling_generator = torch.Generator(device="cpu")
        self.expert_sampling_generator.manual_seed(
            self.expert_sampling_seed
        )
        self.replay_sampling_generator = torch.Generator(device="cpu")
        self.replay_sampling_generator.manual_seed(
            (self.expert_sampling_seed + 1) % ((1 << 63) - 1)
        )

        # Checkpoints contain the complete trainable AMP state.  Optimizer
        # states remain in the explicit checkpoint payload.
        self._policy_module = nn.ModuleDict(
            {
                "actor": self._policy,
                "actor_obs_normalizer": self.actor_obs_normalizer,
                "critic": self.critic,
                "amp_discriminator": self.amp_discriminator,
            }
        )
        self.disc_version = 0
        self._amp_update_idx = 0
        self._init_episode_statistics()

    # ------------------------------------------------------------------
    # Lifecycle and exact demonstration-history reset
    # ------------------------------------------------------------------
    def _init_episode_statistics(self) -> None:
        env = self.env
        self._episode_reward = torch.zeros(
            env.num_envs, device=env.device
        )
        self._episode_length = torch.zeros(
            env.num_envs, device=env.device
        )
        self._train_reward_buffer: deque[float] = deque(maxlen=100)
        self._train_length_buffer: deque[float] = deque(maxlen=100)

    @torch.no_grad()
    def _seed_imitation_history(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        env = self.env
        if env_ids is None:
            env_ids = torch.arange(
                env.num_envs, device=env.device, dtype=torch.long
            )
        endpoints = env.phase_steps.index_select(0, env_ids).float()
        demo_history = env.motion.get_amp_demo_windows_at_end_times(
            endpoints,
            self.imitation_history_steps,
            control_dt=float(env.dt),
            loop=False,
        )
        self.imitation_history.reset_seeded(
            demo_history,
            env_ids=env_ids,
        )

    @torch.no_grad()
    def _uniform_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        env = self.env
        if env_ids is None:
            env_ids = torch.arange(
                env.num_envs, device=env.device, dtype=torch.long
            )
        phases = env.sample_phase_indices(
            int(env_ids.numel()),
            horizon=1,
        )
        observation = env.reset_envs(
            env_ids,
            phase_indices=phases,
            root_velocity_frame="link",
        )
        self._seed_imitation_history(env_ids)
        return observation

    def initial_reset(self) -> torch.Tensor:
        self._obs = self._uniform_reset()
        return self._obs

    def reset_after_resume(self) -> torch.Tensor:
        self._obs = self._uniform_reset()
        self._init_episode_statistics()
        return self._obs

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        self._amp_update_idx = int(update_idx)
        return self._obs

    def pre_training_warmup(
        self,
        current_observation: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], int]:
        # MimicKit starts update one with identity normalizers and the initial
        # discriminator.  A discarded policy/D warm-up would change the recipe.
        return current_observation, {}, 0

    def evaluation_reset(self, phase_indices: torch.Tensor) -> torch.Tensor:
        return self.env.reset(
            phase_indices=phase_indices,
            root_velocity_frame="link",
        )

    def evaluation_step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        return self.env.step(actions)

    def snapshot_runtime_state(self):
        return None

    def restore_runtime_state(self, state) -> None:
        del state

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def _decision_storage(self) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        env = self.env
        t = int(cfg.rollout_env_steps)
        n = int(env.num_envs)
        h = int(self.horizon_h)
        a = int(self.num_act)
        o = int(self.actor_obs_dim)
        device = env.device
        return {
            "observations": torch.zeros(t, n, o, device=device),
            "actions": torch.zeros(t, n, h, a, device=device),
            "old_means": torch.zeros(t, n, h, a, device=device),
            "old_log_probs": torch.zeros(t, n, h, device=device),
            "old_entropies": torch.zeros(t, n, h, device=device),
            "action_mask": torch.zeros(
                t, n, h, dtype=torch.bool, device=device
            ),
            "values": torch.zeros(t, n, device=device),
            "discounted_rewards": torch.zeros(t, n, device=device),
            "durations": torch.zeros(
                t, n, dtype=torch.long, device=device
            ),
            "next_values": torch.zeros(t, n, device=device),
            "bootstrap_mask": torch.zeros(
                t, n, dtype=torch.bool, device=device
            ),
            "trace_mask": torch.zeros(
                t, n, dtype=torch.bool, device=device
            ),
            "valid": torch.zeros(
                t, n, dtype=torch.bool, device=device
            ),
            "timeout": torch.zeros(
                t, n, dtype=torch.bool, device=device
            ),
            "failure": torch.zeros(
                t, n, dtype=torch.bool, device=device
            ),
        }

    @torch.no_grad()
    def _sample_expert_windows(
        self,
        count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if count <= 0:
            raise ValueError("expert window count must be positive")
        env = self.env
        batch_size = max(
            1, int(self.cfg.style_prior.reward_eval_batch_size)
        )
        windows = torch.empty(
            count,
            self.imitation_history_steps,
            self.imitation_frame_dim,
            dtype=torch.float32,
            device="cpu",
        )
        endpoints = (
            torch.rand(
                count,
                generator=self.expert_sampling_generator,
                device="cpu",
            )
            * float(env.motion.num_frames - 1)
        )
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            batch_endpoints = endpoints[start:stop].to(env.device)
            batch = env.motion.get_amp_demo_windows_at_end_times(
                batch_endpoints,
                self.imitation_history_steps,
                control_dt=float(env.dt),
                loop=False,
            )
            windows[start:stop].copy_(batch.cpu())
        return windows, endpoints

    @torch.no_grad()
    def _amp_reward_from_windows(
        self,
        windows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = self.imitation_pipeline.flatten(windows)
        output = self.amp_discriminator.evaluate_reward(
            flat,
            batch_size=int(self.cfg.style_prior.reward_eval_batch_size),
        )
        return output.logits, output.rewards

    @torch.no_grad()
    def collect(self, current_obs: torch.Tensor) -> dict:
        cfg = self.cfg
        env = self.env
        device = env.device
        t_steps = int(cfg.rollout_env_steps)
        n_envs = int(env.num_envs)
        h = int(self.horizon_h)
        gamma = float(cfg.discount_gamma)
        expected_current = t_steps * n_envs

        self.discriminator.eval()
        storage = self._decision_storage()
        primitive = {
            "rewards": torch.zeros(t_steps, n_envs, device=device),
            "logits": torch.zeros(t_steps, n_envs, device=device),
            "done": torch.zeros(
                t_steps, n_envs, dtype=torch.bool, device=device
            ),
            "timeout": torch.zeros_like(
                storage["valid"], dtype=torch.bool
            ),
            "failure": torch.zeros_like(
                storage["valid"], dtype=torch.bool
            ),
            "illegal_contact": torch.zeros_like(
                storage["valid"], dtype=torch.bool
            ),
            "numerical_failure": torch.zeros_like(
                storage["valid"], dtype=torch.bool
            ),
            "tracking_counterfactual": torch.zeros_like(
                storage["valid"], dtype=torch.bool
            ),
            "motion_end_counterfactual": torch.zeros_like(
                storage["valid"], dtype=torch.bool
            ),
            "actions": torch.zeros(
                t_steps, n_envs, self.num_act, device=device
            ),
            "means": torch.zeros(
                t_steps, n_envs, self.num_act, device=device
            ),
            "offsets": torch.zeros(
                t_steps, n_envs, dtype=torch.long, device=device
            ),
            "phases": torch.zeros(t_steps, n_envs, device=device),
            "contact_force_max": torch.zeros(
                t_steps, n_envs, device=device
            ),
            "amp_window_valid": torch.zeros(
                t_steps, n_envs, dtype=torch.bool, device=device
            ),
        }
        current_windows = torch.empty(
            expected_current,
            self.imitation_history_steps,
            self.imitation_frame_dim,
            dtype=torch.float32,
            device="cpu",
        )
        current_endpoints = torch.empty(
            expected_current, dtype=torch.float32, device="cpu"
        )

        observations = current_obs
        env_ids = torch.arange(n_envs, device=device, dtype=torch.long)
        decision_count = torch.zeros(
            n_envs, dtype=torch.long, device=device
        )
        current_slot = torch.full(
            (n_envs,), -1, dtype=torch.long, device=device
        )
        offsets = torch.full(
            (n_envs,), h, dtype=torch.long, device=device
        )
        cached_actions = torch.zeros(
            n_envs, h, self.num_act, device=device
        )
        cached_means = torch.zeros_like(cached_actions)
        cached_log_probs = torch.zeros(
            n_envs, h, device=device
        )
        cached_entropies = torch.zeros_like(cached_log_probs)
        amp_normalizer_count = float(self.disc_normalizer.count.item())
        disc_version = int(self.disc_version)

        for step_index in range(t_steps):
            new_mask = offsets >= h
            new_ids = new_mask.nonzero(as_tuple=False).squeeze(-1)
            if new_ids.numel() > 0:
                raw = observations.index_select(0, new_ids)
                normalized = self.normalize_actor_observation(raw)
                sample = self.sample_normalized_action_chunk(normalized)
                values = self.value_from_normalized_observation(
                    normalized
                ).squeeze(-1)
                slots = decision_count.index_select(0, new_ids)
                if bool((slots >= t_steps).any()):
                    raise RuntimeError("AMP decision storage overflow")
                storage["observations"][slots, new_ids] = raw
                storage["actions"][slots, new_ids] = sample.actions
                storage["old_means"][slots, new_ids] = sample.mean
                storage["old_log_probs"][slots, new_ids] = (
                    sample.log_prob
                )
                storage["old_entropies"][slots, new_ids] = (
                    sample.entropy
                )
                storage["values"][slots, new_ids] = values
                cached_actions[new_ids] = sample.actions
                cached_means[new_ids] = sample.mean
                cached_log_probs[new_ids] = sample.log_prob
                cached_entropies[new_ids] = sample.entropy
                current_slot[new_ids] = slots
                offsets[new_ids] = 0
                decision_count[new_ids] += 1

            action = cached_actions[env_ids, offsets]
            mean = cached_means[env_ids, offsets]
            primitive["actions"][step_index] = action
            primitive["means"][step_index] = mean
            primitive["offsets"][step_index] = offsets
            primitive["phases"][step_index] = env.phase_steps

            post_obs, done, info = env.step(action)
            done_terms = info["done_terms"]
            numerical_failure = done_terms["numerical_failure"].bool()
            if bool((numerical_failure & ~done.bool()).any()):
                raise RuntimeError(
                    "numerical_failure must terminate the affected environment"
                )
            imitation_frame = info["imitation_frame"]
            frame_finite = torch.isfinite(imitation_frame).all(dim=-1)
            unreported_nonfinite = ~frame_finite & ~numerical_failure
            if bool(unreported_nonfinite.any()):
                bad = unreported_nonfinite.nonzero(
                    as_tuple=False
                ).squeeze(-1)
                raise FloatingPointError(
                    "non-finite AMP frame was not reported as numerical "
                    f"failure; env_ids={bad.detach().cpu().tolist()}"
                )
            window_valid = frame_finite & ~numerical_failure
            valid_ids = window_valid.nonzero(
                as_tuple=False
            ).squeeze(-1)
            logits = torch.zeros(n_envs, device=device)
            rewards = torch.zeros(n_envs, device=device)
            if valid_ids.numel() > 0:
                self.imitation_history.push(
                    imitation_frame.index_select(0, valid_ids),
                    env_ids=valid_ids,
                )
                if not bool(
                    self.imitation_history.ready[valid_ids].all()
                ):
                    raise RuntimeError(
                        "demo-seeded AMP history must be valid after every "
                        "finite post-reset policy transition"
                    )
                windows = self.imitation_history.window(valid_ids)
                valid_logits, valid_rewards = (
                    self._amp_reward_from_windows(windows)
                )
                logits.index_copy_(0, valid_ids, valid_logits)
                rewards.index_copy_(0, valid_ids, valid_rewards)

            flat_start = step_index * n_envs
            if valid_ids.numel() > 0:
                flat_ids = valid_ids.detach().cpu() + flat_start
                current_windows.index_copy_(
                    0,
                    flat_ids,
                    windows.detach().cpu(),
                )
                current_endpoints.index_copy_(
                    0,
                    flat_ids,
                    info["imitation_frame_phase_steps"]
                    .index_select(0, valid_ids)
                    .float()
                    .cpu(),
                )

            timeout = done & done_terms["time_out"].bool()
            failure = done & done_terms["physical_failure"].bool()
            slot = current_slot
            offset = offsets
            storage["action_mask"][slot, env_ids, offset] = True
            storage["discounted_rewards"][slot, env_ids] += (
                rewards * torch.pow(rewards.new_tensor(gamma), offset)
            )
            storage["durations"][slot, env_ids] = offset + 1

            finish = done | (offset + 1 >= h) | (
                step_index + 1 == t_steps
            )
            finish_ids = finish.nonzero(as_tuple=False).squeeze(-1)
            if finish_ids.numel() > 0:
                finish_slots = slot.index_select(0, finish_ids)
                finish_failure = failure.index_select(0, finish_ids)
                finish_done = done.index_select(0, finish_ids)
                terminal_value = torch.zeros(
                    finish_ids.numel(),
                    dtype=storage["next_values"].dtype,
                    device=device,
                )
                bootstrap_local_ids = (
                    ~finish_failure
                ).nonzero(as_tuple=False).squeeze(-1)
                if bootstrap_local_ids.numel() > 0:
                    bootstrap_env_ids = finish_ids.index_select(
                        0, bootstrap_local_ids
                    )
                    terminal_obs = post_obs.index_select(
                        0, bootstrap_env_ids
                    )
                    bootstrap_values = (
                        self.value_from_normalized_observation(
                            self.normalize_actor_observation(terminal_obs)
                        )
                        .squeeze(-1)
                    )
                    terminal_value.index_copy_(
                        0, bootstrap_local_ids, bootstrap_values
                    )
                storage["next_values"][
                    finish_slots, finish_ids
                ] = terminal_value
                storage["bootstrap_mask"][
                    finish_slots, finish_ids
                ] = ~finish_failure
                storage["trace_mask"][
                    finish_slots, finish_ids
                ] = ~finish_done
                storage["valid"][finish_slots, finish_ids] = True
                storage["timeout"][
                    finish_slots, finish_ids
                ] = timeout.index_select(0, finish_ids)
                storage["failure"][
                    finish_slots, finish_ids
                ] = finish_failure

            primitive["rewards"][step_index] = rewards
            primitive["logits"][step_index] = logits
            primitive["done"][step_index] = done
            primitive["timeout"][step_index] = timeout
            primitive["failure"][step_index] = failure
            primitive["illegal_contact"][step_index] = (
                done_terms["illegal_contact"].bool()
            )
            primitive["numerical_failure"][step_index] = (
                numerical_failure
            )
            primitive["amp_window_valid"][step_index] = window_valid
            primitive["tracking_counterfactual"][step_index] = (
                done_terms["tracking_failure"].bool()
            )
            primitive["motion_end_counterfactual"][step_index] = (
                done_terms["motion_complete"].bool()
            )
            primitive["contact_force_max"][step_index] = info[
                "debug_terms"
            ]["contact_force_max"]
            self._record_episode_statistics(rewards, done)

            observations = post_obs
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                reset_obs = self._uniform_reset(done_ids)
                observations.index_copy_(0, done_ids, reset_obs)
            offsets = torch.where(finish, h, offsets + 1)

        expected_decisions = (
            torch.arange(t_steps, device=device)[:, None]
            < decision_count[None, :]
        )
        if not torch.equal(expected_decisions, storage["valid"]):
            raise RuntimeError(
                "AMP variable-duration decision packing is not contiguous"
            )
        if int(storage["action_mask"].sum().item()) != expected_current:
            raise RuntimeError(
                "every primitive AMP action must belong to exactly one decision"
            )
        self._obs = observations
        flat_window_valid = (
            primitive["amp_window_valid"].reshape(-1).detach().cpu()
        )
        return {
            **storage,
            "primitive": primitive,
            "current_windows": current_windows[flat_window_valid],
            "current_endpoints": current_endpoints[flat_window_valid],
            "decision_count": decision_count,
            "next_observation": observations,
            "disc_version": disc_version,
            "disc_normalizer_count": amp_normalizer_count,
        }

    def _record_episode_statistics(
        self,
        rewards: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        self._episode_reward += rewards.float()
        self._episode_length += 1.0
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return
        self._train_reward_buffer.extend(
            self._episode_reward[done_ids].detach().cpu().tolist()
        )
        self._train_length_buffer.extend(
            self._episode_length[done_ids].detach().cpu().tolist()
        )
        self._episode_reward[done_ids] = 0.0
        self._episode_length[done_ids] = 0.0

    # ------------------------------------------------------------------
    # Variable-duration credit
    # ------------------------------------------------------------------
    def _compute_credit(self, rollout: dict) -> None:
        valid = rollout["valid"]
        credit = compute_amp_chunk_gae(
            rollout["discounted_rewards"],
            rollout["values"],
            rollout["next_values"],
            rollout["durations"],
            rollout["bootstrap_mask"],
            rollout["trace_mask"],
            valid,
            gamma=float(self.cfg.discount_gamma),
            gae_lambda=float(self.cfg.gae_lambda),
        )
        rollout["advantages"] = credit.advantages
        rollout["actor_advantages"] = (
            normalize_and_clip_amp_advantages(
                credit.advantages,
                valid,
                clip=float(self.cfg.advantage_clip),
            )
        )
        rollout["value_targets"] = credit.value_targets

    # ------------------------------------------------------------------
    # PPO and value updates
    # ------------------------------------------------------------------
    def _logical_batches(
        self,
        count: int,
        multiplier: int,
    ):
        logical = max(1, int(multiplier) * int(self.env.num_envs))
        order = torch.randperm(count, device=self.env.device)
        for start in range(0, count, logical):
            yield order[start : start + logical]

    def _critic_update(self, rollout: dict) -> dict[str, float]:
        valid = rollout["valid"]
        obs = rollout["observations"][valid]
        targets = rollout["value_targets"][valid].detach()
        micro = max(1, int(self.cfg.micro_batch_size))
        totals = {"loss": 0.0, "grad": 0.0, "steps": 0.0}
        for _ in range(int(self.cfg.critic_epochs)):
            for indices in self._logical_batches(
                int(obs.shape[0]), int(self.cfg.critic_batch_size)
            ):
                self.critic_optimizer.zero_grad(set_to_none=True)
                logical_count = int(indices.numel())
                loss_sum = 0.0
                for start in range(0, logical_count, micro):
                    sub = indices[start : start + micro]
                    normalized = self.normalize_actor_observation(obs[sub])
                    prediction = self.value_from_normalized_observation(
                        normalized
                    ).squeeze(-1)
                    loss = F.mse_loss(
                        prediction, targets[sub], reduction="sum"
                    )
                    (loss / logical_count).backward()
                    loss_sum += float(loss.detach().item())
                grad = _grad_norm(self.critic.parameters())
                if not bool(torch.isfinite(grad)):
                    raise FloatingPointError(
                        "AMP critic gradient is non-finite"
                    )
                self.critic_optimizer.step()
                totals["loss"] += loss_sum / logical_count
                totals["grad"] += float(grad.item())
                totals["steps"] += 1.0

        with torch.no_grad():
            prediction = self.value_from_normalized_observation(
                self.normalize_actor_observation(obs)
            ).squeeze(-1)
            error = targets - prediction
            target_var = targets.var(unbiased=False)
            explained = (
                1.0 - error.var(unbiased=False) / target_var
                if float(target_var.item()) > 1.0e-12
                else target_var.new_zeros(())
            )
        denominator = max(1.0, totals["steps"])
        return {
            "critic/loss": totals["loss"] / denominator,
            "critic/grad_norm": totals["grad"] / denominator,
            "critic/optimizer_steps": totals["steps"],
            "critic/value/mean": float(prediction.mean().item()),
            "critic/value/std": float(
                prediction.std(unbiased=False).item()
            ),
            "critic/target/mean": float(targets.mean().item()),
            "critic/target/std": float(
                targets.std(unbiased=False).item()
            ),
            "critic/value_rmse": float(
                error.square().mean().sqrt().item()
            ),
            "critic/explained_variance": float(explained.item()),
            "critic/grad_clip_fraction": 0.0,
            "critic/direct_mse_contract": 1.0,
        }

    def _actor_update(self, rollout: dict) -> dict[str, float]:
        valid = rollout["valid"]
        obs = rollout["observations"][valid]
        actions = rollout["actions"][valid]
        old_means = rollout["old_means"][valid]
        old_log_probs_h = rollout["old_log_probs"][valid]
        action_mask = rollout["action_mask"][valid]
        advantages = rollout["actor_advantages"][valid].detach()
        old_joint_log_prob = (
            old_log_probs_h * action_mask.float()
        ).sum(dim=-1)
        micro = max(1, int(self.cfg.micro_batch_size))
        clip = float(self.cfg.clip_range)
        totals = {
            "policy": 0.0,
            "bound": 0.0,
            "entropy": 0.0,
            "ratio": 0.0,
            "clip": 0.0,
            "kl": 0.0,
            "sample_kl": 0.0,
            "grad": 0.0,
            "steps": 0.0,
        }
        h_kl = torch.zeros(self.horizon_h, device=self.env.device)
        h_ratio = torch.zeros_like(h_kl)
        h_clip = torch.zeros_like(h_kl)
        h_count = torch.zeros_like(h_kl)

        for _ in range(int(self.cfg.policy_epochs)):
            for indices in self._logical_batches(
                int(obs.shape[0]), int(self.cfg.actor_batch_size)
            ):
                logical_count = int(indices.numel())
                self.actor_optimizer.zero_grad(set_to_none=True)
                batch_totals = {key: 0.0 for key in totals if key not in ("grad", "steps")}
                for start in range(0, logical_count, micro):
                    sub = indices[start : start + micro]
                    normalized = self.normalize_actor_observation(obs[sub])
                    stats = self.recompute_policy_statistics(
                        normalized,
                        actions[sub],
                        old_means[sub],
                        self._policy.log_std,
                    )
                    mask = action_mask[sub].float()
                    new_joint = (stats.log_prob * mask).sum(dim=-1)
                    log_ratio = new_joint - old_joint_log_prob[sub]
                    ratio = torch.exp(log_ratio)
                    surrogate = torch.minimum(
                        advantages[sub] * ratio,
                        advantages[sub]
                        * torch.clamp(
                            ratio, 1.0 - clip, 1.0 + clip
                        ),
                    )
                    distribution = self.policy_distribution(normalized)
                    bound = (
                        torch.clamp(
                            distribution.mean.abs() - 1.0, min=0.0
                        )
                        .square()
                        .sum(dim=(-1, -2))
                        .mean()
                    )
                    entropy = (
                        stats.entropy * mask
                    ).sum(dim=-1).mean()
                    policy_loss = -surrogate.mean()
                    loss = (
                        policy_loss
                        + float(self.cfg.action_bound_weight) * bound
                    )
                    (loss * (sub.numel() / logical_count)).backward()

                    with torch.no_grad():
                        clipped = (
                            (ratio < 1.0 - clip)
                            | (ratio > 1.0 + clip)
                        ).float()
                        analytic_kl = (
                            stats.old_to_new_kl * mask
                        ).sum(dim=-1)
                        batch_totals["policy"] += (
                            float(policy_loss.item())
                            * sub.numel()
                            / logical_count
                        )
                        batch_totals["bound"] += (
                            float(bound.item())
                            * sub.numel()
                            / logical_count
                        )
                        batch_totals["entropy"] += (
                            float(entropy.item())
                            * sub.numel()
                            / logical_count
                        )
                        batch_totals["ratio"] += (
                            float(ratio.mean().item())
                            * sub.numel()
                            / logical_count
                        )
                        batch_totals["clip"] += (
                            float(clipped.mean().item())
                            * sub.numel()
                            / logical_count
                        )
                        batch_totals["kl"] += (
                            float(analytic_kl.mean().item())
                            * sub.numel()
                            / logical_count
                        )
                        batch_totals["sample_kl"] += (
                            float((0.5 * log_ratio.square()).mean().item())
                            * sub.numel()
                            / logical_count
                        )
                        for offset in range(self.horizon_h):
                            active = action_mask[sub, offset]
                            if bool(active.any()):
                                count = active.float().sum()
                                per_ratio = torch.exp(
                                    stats.log_prob[:, offset]
                                    - old_log_probs_h[sub, offset]
                                )
                                h_kl[offset] += (
                                    stats.old_to_new_kl[:, offset][active].sum()
                                )
                                h_ratio[offset] += per_ratio[active].sum()
                                h_clip[offset] += (
                                    (
                                        (per_ratio[active] < 1.0 - clip)
                                        | (per_ratio[active] > 1.0 + clip)
                                    )
                                    .float()
                                    .sum()
                                )
                                h_count[offset] += count

                grad = _grad_norm(self._policy.parameters())
                if not bool(torch.isfinite(grad)):
                    raise FloatingPointError(
                        "AMP actor gradient is non-finite"
                    )
                self.actor_optimizer.step()
                for key, value in batch_totals.items():
                    totals[key] += value
                totals["grad"] += float(grad.item())
                totals["steps"] += 1.0

        denominator = max(1.0, totals["steps"])
        metrics = {
            "amp_policy/policy_loss": totals["policy"] / denominator,
            "amp_policy/action_bound_loss": totals["bound"] / denominator,
            "amp_policy/entropy": totals["entropy"] / denominator,
            "amp_policy/ratio": totals["ratio"] / denominator,
            "amp_policy/clip_fraction": totals["clip"] / denominator,
            "amp_policy/diagnostic_kl": totals["kl"] / denominator,
            "amp_policy/sample_kl": totals["sample_kl"] / denominator,
            "amp_policy/grad_norm": totals["grad"] / denominator,
            "amp_policy/optimizer_steps": totals["steps"],
            "amp_policy/lr": float(self.actor_optimizer.param_groups[0]["lr"]),
            "amp_policy/direct_absolute_action_contract": 1.0,
            "amp_policy/chunk_joint_ratio_contract": 1.0,
        }
        for offset in range(self.horizon_h):
            count = float(h_count[offset].item())
            metrics[f"amp_policy/offset_{offset}_kl"] = (
                float(h_kl[offset].item()) / count if count else -1.0
            )
            metrics[f"amp_policy/offset_{offset}_ratio"] = (
                float(h_ratio[offset].item()) / count if count else -1.0
            )
            metrics[f"amp_policy/offset_{offset}_clip"] = (
                float(h_clip[offset].item()) / count if count else -1.0
            )
            metrics[f"amp_policy/offset_{offset}_count"] = count
        return metrics

    # ------------------------------------------------------------------
    # Discriminator, replay, and normalizer ordering
    # ------------------------------------------------------------------
    def _windows_to_flat(
        self,
        windows: torch.Tensor,
    ) -> torch.Tensor:
        return self.imitation_pipeline.flatten(
            windows.to(
                device=self.env.device,
                dtype=torch.float32,
                non_blocking=True,
            )
        )

    @torch.no_grad()
    def _evaluate_discriminator_domain(
        self,
        windows: torch.Tensor,
        *,
        domain: str,
        expert: bool,
    ) -> dict[str, float]:
        """Evaluate the committed post-update discriminator on a fixed pool."""

        count = int(windows.shape[0])
        if count <= 0:
            raise ValueError("discriminator evaluation pool cannot be empty")
        batch_size = max(
            1, int(self.cfg.style_prior.reward_eval_batch_size)
        )
        logits_cpu: list[torch.Tensor] = []
        self.discriminator.eval()
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            logits_cpu.append(
                self.amp_discriminator(
                    self._windows_to_flat(windows[start:stop])
                )
                .detach()
                .float()
                .cpu()
            )
        logits = torch.cat(logits_cpu)
        probability = torch.sigmoid(logits)
        quantiles = torch.quantile(
            logits,
            logits.new_tensor((0.05, 0.50, 0.95)),
        )
        correct = logits > 0.0 if expert else logits < 0.0
        target = torch.ones_like(logits) if expert else torch.zeros_like(
            logits
        )
        return {
            f"disc/post_{domain}_count": float(count),
            f"disc/post_{domain}_accuracy": float(
                correct.float().mean().item()
            ),
            f"disc/post_{domain}_bce": float(
                F.binary_cross_entropy_with_logits(logits, target).item()
            ),
            f"disc/post_{domain}_logit_mean": float(logits.mean().item()),
            f"disc/post_{domain}_logit_std": float(
                logits.std(unbiased=False).item()
            ),
            f"disc/post_{domain}_logit_p05": float(quantiles[0].item()),
            f"disc/post_{domain}_logit_p50": float(quantiles[1].item()),
            f"disc/post_{domain}_logit_p95": float(quantiles[2].item()),
            f"disc/post_{domain}_prob_mean": float(
                probability.mean().item()
            ),
        }

    def _discriminator_update(
        self,
        rollout: dict,
        expert_windows: torch.Tensor,
    ) -> dict[str, float]:
        amp_cfg = self.cfg.style_prior
        current_windows = rollout["current_windows"]
        count = int(current_windows.shape[0])
        if int(expert_windows.shape[0]) != count:
            raise RuntimeError(
                "AMP current and expert pools must have equal size"
            )
        logical_batch = int(amp_cfg.batch_size) * int(
            self.env.num_envs
        )
        if count < logical_batch:
            raise RuntimeError(
                "AMP has too few finite current windows for one "
                f"discriminator batch: {count} < {logical_batch}"
            )

        normalizer_metrics = {
            "disc_norm/count_before_update": float(
                self.disc_normalizer.count.item()
            ),
            "disc_norm/pending_current_count": float(count),
            "disc_norm/pending_expert_count": float(count),
            "disc_norm/equal_domain_weight_contract": 1.0,
            "disc_norm/frozen_during_discriminator_update": 1.0,
        }
        self.amp_discriminator.open_normalizer_update()
        staging = max(1, int(amp_cfg.reward_eval_batch_size))
        try:
            for start in range(0, count, staging):
                stop = min(start + staging, count)
                self.amp_discriminator.record_normalizer_update_batch(
                    current_observations=self._windows_to_flat(
                        current_windows[start:stop]
                    ),
                    expert_observations=self._windows_to_flat(
                        expert_windows[start:stop]
                    ),
                )

            steps_per_epoch = count // logical_batch
            metric_sums: dict[str, float] = {}
            optimizer_steps = 0
            for _ in range(int(amp_cfg.epochs)):
                permutation = torch.randperm(count, device="cpu")
                for batch_index in range(steps_per_epoch):
                    indices = permutation[
                        batch_index
                        * logical_batch : (batch_index + 1)
                        * logical_batch
                    ]
                    current_raw = current_windows.index_select(0, indices)
                    expert_raw = expert_windows.index_select(0, indices)
                    replay_raw, _ = self.disc_window_replay.sample(
                        logical_batch,
                        generator=self.replay_sampling_generator,
                    )
                    batch_metrics = self.amp_discriminator.train_batch(
                        current_observations=self._windows_to_flat(
                            current_raw
                        ),
                        replay_observations=self._windows_to_flat(
                            replay_raw
                        ),
                        expert_observations=self._windows_to_flat(
                            expert_raw
                        ),
                        micro_batch_size=int(amp_cfg.micro_batch_size),
                    )
                    for key, value in batch_metrics.items():
                        metric_sums[key] = (
                            metric_sums.get(key, 0.0) + float(value)
                        )
                    optimizer_steps += 1
            if optimizer_steps <= 0:
                raise RuntimeError(
                    "AMP discriminator pool is smaller than one logical batch"
                )
            self.amp_discriminator.commit_normalizer_update()
        except BaseException:
            if self.amp_discriminator.normalizer_update_open:
                self.amp_discriminator.abort_normalizer_update()
            raise

        metrics = {
            key: value / optimizer_steps
            for key, value in metric_sums.items()
        }
        metrics.update(normalizer_metrics)
        metrics.update(
            self.disc_normalizer.statistics(prefix="disc_norm")
        )
        metrics.update(
            {
                "disc/update_steps": float(optimizer_steps),
                "disc/optimizer_steps_total": float(
                    self.amp_discriminator.optimizer_steps.item()
                ),
                "disc/version_before": float(self.disc_version),
                "disc/version_after": float(self.disc_version + 1),
                "disc_norm/committed_after_training": 1.0,
                "disc_norm/count_after_update": float(
                    self.disc_normalizer.count.item()
                ),
            }
        )
        for name in (
            "loss",
            "bce",
            "expert_accuracy",
            "current_accuracy",
            "replay_accuracy",
            "expert_logit_mean",
            "current_logit_mean",
            "replay_logit_mean",
        ):
            source = f"disc/{name}"
            if source in metrics:
                metrics[f"disc/train_pre_{name}"] = metrics[source]

        diagnostic_generator = torch.Generator(device="cpu")
        diagnostic_generator.manual_seed(
            (
                self.expert_sampling_seed
                + 2
                + int(self._amp_update_idx)
            )
            % ((1 << 63) - 1)
        )
        replay_evaluation, _ = self.disc_window_replay.sample(
            count,
            generator=diagnostic_generator,
        )
        metrics.update(
            self._evaluate_discriminator_domain(
                current_windows,
                domain="current",
                expert=False,
            )
        )
        metrics.update(
            self._evaluate_discriminator_domain(
                replay_evaluation,
                domain="replay",
                expert=False,
            )
        )
        metrics.update(
            self._evaluate_discriminator_domain(
                expert_windows,
                domain="expert",
                expert=True,
            )
        )
        metrics["disc/post_fake_accuracy"] = 0.5 * (
            metrics["disc/post_current_accuracy"]
            + metrics["disc/post_replay_accuracy"]
        )
        metrics["disc/post_evaluation_uses_committed_normalizer"] = 1.0
        self.disc_version += 1
        return metrics

    # ------------------------------------------------------------------
    # Complete update and diagnostics
    # ------------------------------------------------------------------
    def update(self, rollout: dict, collect_time: float) -> dict[str, float]:
        update_start = time.perf_counter()
        self._compute_credit(rollout)

        critic_start = time.perf_counter()
        critic_metrics = self._critic_update(rollout)
        critic_time = time.perf_counter() - critic_start

        actor_start = time.perf_counter()
        actor_metrics = self._actor_update(rollout)
        actor_time = time.perf_counter() - actor_start

        expert_windows, expert_endpoints = self._sample_expert_windows(
            int(rollout["current_windows"].shape[0])
        )
        inserted = self.disc_window_replay.update_from_rollout(
            rollout["current_windows"],
            end_times=rollout["current_endpoints"],
            replay_samples=int(self.cfg.style_prior.replay_samples),
            generator=self.replay_sampling_generator,
        )
        disc_start = time.perf_counter()
        disc_metrics = self._discriminator_update(
            rollout, expert_windows
        )
        disc_time = time.perf_counter() - disc_start

        # MimicKit advances observation statistics only after critic, actor,
        # discriminator, and this iteration's rewards all used old statistics.
        valid_obs = rollout["observations"][rollout["valid"]]
        self.actor_obs_normalizer._update(valid_obs)

        primitive = rollout["primitive"]
        valid = rollout["valid"]
        action_mask = rollout["action_mask"]
        sampled = rollout["actions"]
        means = rollout["old_means"]
        noise = sampled - means
        metrics: dict[str, float] = {}
        metrics.update(critic_metrics)
        metrics.update(actor_metrics)
        metrics.update(disc_metrics)
        metrics.update(
            gaussian_action_noise_statistics(noise, action_mask)
        )
        metrics.update(_action_delta_statistics(primitive))
        metrics.update(
            _flat_stats("reward/amp", primitive["rewards"])
        )
        amp_valid = primitive["amp_window_valid"]
        metrics.update(
            _flat_stats("amp_reward", primitive["rewards"][amp_valid])
        )
        metrics.update(
            style_reward_statistics(
                primitive["logits"][amp_valid],
                primitive["rewards"][amp_valid],
                scale=float(self.cfg.style_prior.reward_scale),
                minimum_one_minus_prob=float(
                    self.cfg.style_prior.reward_epsilon
                ),
                prefix="amp_reward",
            )
        )
        metrics.update(
            _flat_stats("credit/amp_adv", rollout["advantages"][valid])
        )
        metrics.update(
            _flat_stats(
                "credit/actor_adv",
                rollout["actor_advantages"][valid],
            )
        )
        metrics.update(
            _flat_stats(
                "reward/amp_credit",
                rollout["discounted_rewards"][valid],
            )
        )
        metrics.update(
            _flat_stats(
                "amp_policy/decision_duration",
                rollout["durations"][valid].float(),
            )
        )

        fixed_std = self._policy.log_std.exp().detach()
        metrics.update(
            {
                "amp_policy/fixed_normalized_std": float(
                    fixed_std.mean().item()
                ),
                "amp_policy/std_trainable": 0.0,
                "amp_policy/environment_clip_fraction": float(
                    (primitive["actions"].abs() > 1.0)
                    .float()
                    .mean()
                    .item()
                ),
                "amp_policy/mean_out_of_bounds_fraction": float(
                    (primitive["means"].abs() > 1.0)
                    .float()
                    .mean()
                    .item()
                ),
                "amp_policy/normalized_sample_rms": float(
                    primitive["actions"].square().mean().sqrt().item()
                ),
                "amp_policy/normalized_mean_rms": float(
                    primitive["means"].square().mean().sqrt().item()
                ),
                "rollout/decision_valid_count": float(valid.sum().item()),
                "rollout/endpoint_valid_count": float(
                    amp_valid.sum().item()
                ),
                "rollout/endpoint_coverage_fraction_alive": float(
                    amp_valid.float().mean().item()
                ),
                "rollout/endpoint_warmup_gap_count": 0.0,
                "rollout/warmup_action_credited_count": 0.0,
                "rollout/warmup_with_future_amp_count": 0.0,
                "rollout/action_credit_valid_count": float(
                    action_mask.sum().item()
                ),
                "rollout/action_credit_coverage_fraction_alive": 1.0,
                "rollout/credit_gap_count": 0.0,
                "rollout/reward_without_endpoint_count": 0.0,
                "rollout/numerical_window_excluded_count": float(
                    (~amp_valid).sum().item()
                ),
                "rollout/decision_count": float(valid.sum().item()),
                "rollout/primitive_count": float(
                    primitive["rewards"].numel()
                ),
                "rollout/done_fraction": float(
                    primitive["done"].float().mean().item()
                ),
                "rollout/failure_fraction": float(
                    primitive["failure"].float().mean().item()
                ),
                "rollout/timeout_fraction": float(
                    primitive["timeout"].float().mean().item()
                ),
                "rollout/illegal_contact_fraction": float(
                    primitive["illegal_contact"].float().mean().item()
                ),
                "rollout/numerical_failure_fraction": float(
                    primitive["numerical_failure"].float().mean().item()
                ),
                "rollout/tracking_counterfactual_fraction": float(
                    primitive["tracking_counterfactual"]
                    .float()
                    .mean()
                    .item()
                ),
                "rollout/reference_end_occupancy_fraction": float(
                    primitive["motion_end_counterfactual"]
                    .float()
                    .mean()
                    .item()
                ),
                "rollout/bootstrap_fraction": float(
                    rollout["bootstrap_mask"][valid].float().mean().item()
                ),
                "rollout/trace_fraction": float(
                    rollout["trace_mask"][valid].float().mean().item()
                ),
                "act/abs_max": float(
                    primitive["actions"].abs().max().item()
                ),
                "act/policy_bound_violation_max": float(
                    torch.clamp(
                        primitive["actions"].abs() - 1.0, min=0.0
                    )
                    .max()
                    .item()
                ),
                "disc_contract/numerical_window_excluded_count": float(
                    (~amp_valid).sum().item()
                ),
                "disc/current_pool_size": float(
                    rollout["current_windows"].shape[0]
                ),
                "disc/expert_pool_size": float(expert_windows.shape[0]),
                "disc/replay_inserted": float(inserted),
                "disc/reward_discriminator_version": float(
                    rollout["disc_version"]
                ),
                "disc/reward_normalizer_count": float(
                    rollout["disc_normalizer_count"]
                ),
                "disc/expert_independent_sampling_contract": 1.0,
                "disc/expert_full_motion_sampling_contract": 1.0,
                "disc/current_replay_global_uniform_contract": 1.0,
                "disc/phase_conditioning": 0.0,
                "disc/reference_conditioning": 0.0,
                "train/mean_reward": float(
                    sum(self._train_reward_buffer)
                    / len(self._train_reward_buffer)
                    if self._train_reward_buffer
                    else primitive["rewards"].mean().item()
                ),
                "train/mean_episode_length": float(
                    sum(self._train_length_buffer)
                    / len(self._train_length_buffer)
                    if self._train_length_buffer
                    else self._episode_length.mean().item()
                ),
                "timing/collect_s": float(collect_time),
                "timing/actor_update_s": float(actor_time),
                "timing/critic_update_s": float(critic_time),
                "timing/disc_update_s": float(disc_time),
                "timing/update_s": float(time.perf_counter() - update_start),
                "system/primitive_steps": float(
                    self._amp_update_idx
                    * int(self.cfg.rollout_env_steps)
                    * int(self.env.num_envs)
                ),
                "system/cuda_peak_allocated_gib": float(
                    torch.cuda.max_memory_allocated(self.env.device)
                    / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                ),
                "amp_contract/horizon": float(self.horizon_h),
                "amp_contract/history_steps": float(
                    self.imitation_history_steps
                ),
                "amp_contract/frame_dim": float(
                    self.imitation_frame_dim
                ),
                "amp_contract/window_dim": float(
                    self.imitation_window_dim
                ),
                "amp_contract/actor_observation_dim": float(
                    self.actor_obs_dim
                ),
                "amp_contract/action_dim": float(self.num_act),
                "amp_contract/pure_amp": 1.0,
                "amp_contract/no_reference_terminal": 1.0,
                "amp_contract/no_motion_end_terminal": 1.0,
                "amp_contract/demo_seeded_history": 1.0,
                "amp_contract/no_reward_dt_scaling": 1.0,
            }
        )
        for offset in range(self.horizon_h):
            metrics[f"amp_policy/offset_{offset}_fixed_std"] = float(
                fixed_std[offset].mean().item()
            )
            offset_count = float(
                action_mask[..., offset].sum().item()
            )
            metrics[f"amp_policy/offset_{offset}_executed_count"] = (
                offset_count
            )

        replay_metrics = self.disc_window_replay.statistics()
        for key, value in replay_metrics.items():
            metrics[f"disc_{key}"] = value
            metrics[f"disc_replay/{key.removeprefix('replay/')}"] = value
        metrics["disc_replay/size"] = float(
            len(self.disc_window_replay)
        )
        metrics["disc_replay/capacity"] = float(
            self.disc_window_replay.capacity
        )
        metrics.update(
            _flat_stats(
                "disc_endpoint/current_train",
                rollout["current_endpoints"].float(),
            )
        )
        metrics.update(
            _flat_stats(
                "disc_endpoint/expert_train",
                expert_endpoints.float(),
            )
        )
        metrics["disc_endpoint/support_min"] = 0.0
        metrics["disc_endpoint/support_max"] = float(
            self.env.motion.num_frames - 1
        )
        metrics["disc_endpoint/current_train_sample_count"] = float(
            rollout["current_endpoints"].numel()
        )
        metrics["disc_endpoint/expert_train_sample_count"] = float(
            expert_endpoints.numel()
        )
        metrics["disc_endpoint/current_expert_tv"] = (
            _endpoint_total_variation(
                rollout["current_endpoints"],
                expert_endpoints,
                support_size=self.env.motion.num_frames,
            )
        )

        finite = all(
            bool(torch.isfinite(parameter).all())
            for module in (
                self._policy,
                self.critic,
                self.discriminator,
            )
            for parameter in module.parameters()
        )
        metrics["system/parameters_finite"] = float(finite)
        if not finite:
            raise FloatingPointError(
                "AMP detected non-finite trainable parameters"
            )
        _ensure_horizon_metrics(metrics, self.horizon_h)
        return metrics

    # ------------------------------------------------------------------
    # Checkpoint contract
    # ------------------------------------------------------------------
    def validate_checkpoint_payload(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            raise ValueError("AMP checkpoint payload must be a dictionary")
        state = payload.get("algo_state")
        if not isinstance(state, dict):
            raise ValueError(
                "Checkpoint has no compatible AMP state; start fresh"
            )
        for key, expected in AMP_CHECKPOINT_CONTRACT.items():
            if type(state.get(key)) is not type(expected) or state.get(
                key
            ) != expected:
                raise ValueError(
                    f"AMP checkpoint {key}={state.get(key)!r}, "
                    f"required {expected!r}; start a fresh run"
                )
        if int(state.get("horizon", -1)) != self.horizon_h:
            raise ValueError("AMP checkpoint horizon differs from config")
        if int(state.get("imitation_history_steps", -1)) != (
            self.imitation_history_steps
        ):
            raise ValueError(
                "AMP checkpoint discriminator history differs"
            )

    def extra_checkpoint_state(self) -> dict:
        state = super().extra_checkpoint_state()
        state.update(
            {
                **AMP_CHECKPOINT_CONTRACT,
                "horizon": int(self.horizon_h),
                "imitation_history_steps": int(
                    self.imitation_history_steps
                ),
                "imitation_frame_dim": int(self.imitation_frame_dim),
                "disc_optimizer": self.disc_optimizer.state_dict(),
                "disc_version": int(self.disc_version),
                "disc_window_replay": self.disc_window_replay.state_dict(),
                "expert_sampling_seed": int(self.expert_sampling_seed),
                "expert_sampling_generator_state": (
                    self.expert_sampling_generator.get_state().clone()
                ),
                "replay_sampling_generator_state": (
                    self.replay_sampling_generator.get_state().clone()
                ),
                "action_low": self.action_low.detach().cpu(),
                "action_high": self.action_high.detach().cpu(),
            }
        )
        return state

    def load_extra_checkpoint_state(
        self,
        payload: dict,
        reset_optimizer: bool = False,
    ) -> None:
        self._validate_extra_checkpoint_state(payload)
        super().load_extra_checkpoint_state(
            payload, reset_optimizer=reset_optimizer
        )
        if not reset_optimizer:
            self.disc_optimizer.load_state_dict(
                payload["disc_optimizer"]
            )
        self.disc_version = int(payload["disc_version"])
        self.disc_window_replay.load_state_dict(
            payload["disc_window_replay"]
        )
        self.expert_sampling_generator.set_state(
            payload["expert_sampling_generator_state"].cpu()
        )
        self.replay_sampling_generator.set_state(
            payload["replay_sampling_generator_state"].cpu()
        )

    def _validate_extra_checkpoint_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise ValueError("AMP checkpoint algorithm state is missing")
        for key, expected in AMP_CHECKPOINT_CONTRACT.items():
            if state.get(key) != expected:
                raise ValueError(
                    f"incompatible AMP checkpoint contract: {key}"
                )

__all__ = ["AMP"]
