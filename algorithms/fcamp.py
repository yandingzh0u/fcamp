"""FC-AMP: causal Flow-chunk policy optimization with standard AMP.

The discriminator remains the standard independent MimicKit-style classifier.
The policy contribution is the coupling of primitive AMP rewards to causal
frame conditionals, using a shared prefix encoder with separate task/AMP Flow
value heads.
"""

from __future__ import annotations

import math
import time

import torch
from torch import nn

from algorithms.causal_credit import (
    compute_dual_channel_gae,
    resolve_terminal_masks,
    with_chunk_shared_actor_credit,
)
from algorithms.sfpo import SFPO
from amp import (
    AMPReplayBuffer,
    AMPRunningNormalizer,
    CausalAMPHistory,
    amp_reward_from_logits,
    amp_reward_statistics,
)
from networks.amp_discriminator import (
    AMPDiscriminator,
    compute_amp_discriminator_loss,
)
from networks.mlp_actor_critic import EmpiricalNormalization
from networks.prefix_flow_critic import SharedTrunkDualFlowCritic


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


class FCAMP(SFPO):
    """Full H=4 causal Flow-CPS policy with W=16 standard AMP."""

    def build(self) -> None:
        super().build()
        cfg = self.cfg
        amp_cfg = cfg.amp
        critic_cfg = cfg.critics
        env = self.env

        # A temporal discriminator must never observe a silent motion teleport.
        env.terminate_on_motion_end = True

        # Context = current privileged state, chunk-start privileged and actor
        # states, previous action, padded causal latent prefix, prefix mask and
        # offset one-hot. It never includes z_j or a future latent.
        self.prefix_context_dim = (
            2 * self.critic_obs_dim
            + self.actor_obs_dim
            + self.num_act
            + self.chunk_dim
            + 2 * self.horizon_h
        )
        self.critic = SharedTrunkDualFlowCritic(
            context_dim=self.prefix_context_dim,
            encoder_hidden_dims=critic_cfg.encoder_hidden_dims,
            head_hidden_dims=critic_cfg.head_hidden_dims,
            activation=cfg.activation,
            flow_steps=self.flow_critic_steps,
            eval_samples=self.flow_critic_samples,
        ).to(env.device)
        self.prefix_context_normalizer = EmpiricalNormalization(
            self.prefix_context_dim, env.device
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.critic_learning_rate,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=float(cfg.critic_weight_decay),
        )

        self.amp_history_steps = int(amp_cfg.obs_steps)
        self.amp_frame_dim = int(env.amp_frame_dim)
        self.amp_window_dim = self.amp_history_steps * self.amp_frame_dim
        self.discriminator = AMPDiscriminator(
            self.amp_window_dim, tuple(amp_cfg.hidden_dims)
        ).to(env.device)
        self.disc_normalizer = AMPRunningNormalizer(
            self.amp_window_dim,
            device=env.device,
            clip=float(amp_cfg.normalizer_clip),
        )
        self.amp_history = CausalAMPHistory(
            env.num_envs,
            self.amp_history_steps,
            self.amp_frame_dim,
            device=env.device,
        )
        replay_dtype = torch.float32
        # Replay is intentionally CPU-backed and kept FP32 so current/replay/demo
        # discriminator domains do not differ by storage precision.
        self.current_disc_buffer = AMPReplayBuffer(
            int(amp_cfg.current_buffer_size),
            self.amp_window_dim,
            storage_dtype=replay_dtype,
        )
        self.disc_replay = AMPReplayBuffer(
            int(amp_cfg.replay_size),
            self.amp_window_dim,
            storage_dtype=replay_dtype,
        )
        disc_parameters = [p for p in self.discriminator.parameters() if p.requires_grad]
        optimizer_name = amp_cfg.optimizer.lower()
        optimizer_kwargs = {
            "lr": float(amp_cfg.learning_rate),
            "weight_decay": float(amp_cfg.weight_decay),
        }
        if optimizer_name == "sgd":
            # MimicKit's MPOptimizer uses momentum=0.9 and no gradient clip for
            # the standard G1 AMP discriminator.
            self.disc_optimizer = torch.optim.SGD(
                disc_parameters, momentum=0.9, **optimizer_kwargs
            )
        elif optimizer_name == "adam":
            self.disc_optimizer = torch.optim.Adam(disc_parameters, **optimizer_kwargs)
        else:
            self.disc_optimizer = torch.optim.AdamW(disc_parameters, **optimizer_kwargs)
        self.disc_version = 0

        # Everything needed for a training checkpoint is registered, while
        # deployment still calls deterministic_actions() on the actor only.
        self._policy_module = nn.ModuleDict(
            {
                "actor": self._policy,
                "actor_obs_normalizer": self.actor_obs_normalizer,
                "critic": self.critic,
                "prefix_context_normalizer": self.prefix_context_normalizer,
                "discriminator": self.discriminator,
                "disc_normalizer": self.disc_normalizer,
            }
        )

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def extra_checkpoint_state(self) -> dict:
        payload = super().extra_checkpoint_state()
        payload.update(
            {
                "disc_optimizer": self.disc_optimizer.state_dict(),
                "disc_version": int(self.disc_version),
                "disc_replay": self.disc_replay.state_dict(),
                "fcamp_schema_version": 1,
                "amp_history_steps": self.amp_history_steps,
                "amp_frame_dim": self.amp_frame_dim,
                "prefix_context_dim": self.prefix_context_dim,
            }
        )
        return payload

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        super().load_extra_checkpoint_state(payload, reset_optimizer=reset_optimizer)
        if not payload:
            return
        if not reset_optimizer and "disc_optimizer" in payload:
            self.disc_optimizer.load_state_dict(payload["disc_optimizer"])
        self.disc_version = int(payload.get("disc_version", self.disc_version))
        if not self.disc_replay.load_state_dict(payload.get("disc_replay")):
            print("[FCAMP] discriminator replay absent/incompatible; starting empty", flush=True)

    # ------------------------------------------------------------------ #
    # Causal prefix contexts and AMP reward
    # ------------------------------------------------------------------ #
    def _prefix_context_raw(
        self,
        current_critic_obs: torch.Tensor,
        chunk_start_critic_obs: torch.Tensor,
        chunk_start_actor_obs: torch.Tensor,
        previous_action: torch.Tensor,
        final_latent: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        batch = current_critic_obs.shape[0]
        latent = final_latent.reshape(batch, self.horizon_h, self.num_act)
        prefix = torch.zeros_like(latent)
        if offset > 0:
            prefix[:, :offset] = latent[:, :offset]
        prefix_mask = torch.zeros(
            batch, self.horizon_h, device=latent.device, dtype=latent.dtype
        )
        if offset > 0:
            prefix_mask[:, :offset] = 1.0
        offset_onehot = torch.zeros_like(prefix_mask)
        offset_onehot[:, int(offset)] = 1.0
        context = torch.cat(
            [
                current_critic_obs,
                chunk_start_critic_obs,
                chunk_start_actor_obs,
                previous_action,
                prefix.reshape(batch, -1),
                prefix_mask,
                offset_onehot,
            ],
            dim=-1,
        )
        if context.shape[-1] != self.prefix_context_dim:
            raise RuntimeError(
                f"FCAMP prefix context dim {context.shape[-1]} != {self.prefix_context_dim}"
            )
        return context

    @torch.no_grad()
    def _evaluate_prefix_values(self, contexts: torch.Tensor) -> torch.Tensor:
        flat = contexts.reshape(-1, self.prefix_context_dim)
        batch_size = max(1, int(self.cfg.micro_batch_size))
        outputs = []
        for start in range(0, flat.shape[0], batch_size):
            outputs.append(self.critic.evaluate(flat[start : start + batch_size]))
        return torch.cat(outputs, dim=0).reshape(*contexts.shape[:-1], 2)

    @torch.no_grad()
    def _evaluate_amp_reward(self, raw_windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = max(1, int(self.cfg.amp.reward_eval_batch_size))
        logits = []
        for start in range(0, raw_windows.shape[0], batch_size):
            normalized = self.disc_normalizer.normalize(raw_windows[start : start + batch_size])
            logits.append(self.discriminator(normalized))
        all_logits = torch.cat(logits, dim=0)
        rewards = amp_reward_from_logits(
            all_logits,
            scale=float(self.cfg.amp.reward_scale),
            minimum_one_minus_prob=float(self.cfg.amp.reward_epsilon),
        )
        return all_logits, rewards

    def _reset_amp_history(
        self,
        phase_indices: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        del phase_indices
        initial_frame = self.env.get_amp_policy_frame(env_ids)
        self.amp_history.reset(initial_frame, env_ids=env_ids)

    def initial_reset(self) -> torch.Tensor:
        obs = super().initial_reset()
        self._reset_amp_history(self.env.phase_steps.clone())
        return obs

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        # Trainer owns the canonical update clock, including after resume.
        self._fcamp_update_idx = int(update_idx)
        return super().reset_for_update(update_idx)

    # ------------------------------------------------------------------ #
    # Rollout and causal credit
    # ------------------------------------------------------------------ #
    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        device = env.device
        n_envs = env.num_envs
        chunks = self._chunks_per_update()
        h = self.horizon_h
        flow_steps = int(self.cfg.flow_steps)
        total_primitive_steps = chunks * h

        # Rollout only stores legal policy windows. AMP rewards are recomputed
        # after this same batch has updated the discriminator.
        self.discriminator.eval()
        self.current_disc_buffer.clear()

        actor_obs_buf = torch.zeros(chunks, n_envs, self.actor_obs_dim, device=device)
        actor_obs_raw_buf = torch.zeros_like(actor_obs_buf)
        actions_buf = torch.zeros(chunks, n_envs, h, self.num_act, device=device)
        latent_path_buf = torch.zeros(
            chunks, n_envs, flow_steps + 1, self.chunk_dim, device=device
        )
        old_log_probs_buf = torch.zeros(chunks, n_envs, flow_steps, h, device=device)
        prev_action_buf = torch.zeros(chunks, n_envs, self.num_act, device=device)
        context_buf = torch.zeros(
            chunks, n_envs, h, self.prefix_context_dim, device=device
        )
        next_context_buf = torch.zeros_like(context_buf)
        context_raw_buf = torch.zeros_like(context_buf)
        next_context_raw_buf = torch.zeros_like(context_buf)
        task_reward_buf = torch.zeros(chunks, n_envs, h, device=device)
        amp_reward_raw_buf = torch.zeros_like(task_reward_buf)
        amp_reward_credit_buf = torch.zeros_like(task_reward_buf)
        mixed_reward_buf = torch.zeros_like(task_reward_buf)
        amp_logit_buf = torch.zeros_like(task_reward_buf)
        valid_buf = torch.zeros(chunks, n_envs, h, dtype=torch.bool, device=device)
        amp_valid_buf = torch.zeros_like(valid_buf)
        amp_age_buf = torch.full(
            (chunks, n_envs, h), -1, dtype=torch.long, device=device
        )
        amp_window_chunks: list[torch.Tensor] = []
        amp_window_index_chunks: list[torch.Tensor] = []
        done_buf = torch.zeros_like(valid_buf)
        failure_buf = torch.zeros_like(valid_buf)
        timeout_buf = torch.zeros_like(valid_buf)
        motion_complete_buf = torch.zeros_like(valid_buf)
        bootstrap_buf = torch.zeros_like(valid_buf)
        trace_buf = torch.zeros_like(valid_buf)

        obs = current_obs
        critic_obs = self._critic_obs
        rollout_info_items: list[tuple[dict, torch.Tensor]] = []
        current_keep_per_step = max(
            1, math.ceil(int(self.cfg.amp.current_buffer_size) / total_primitive_steps)
        )
        task_weight = float(self.cfg.credit.task_weight)
        amp_weight = float(self.cfg.credit.amp_weight)
        amp_dt_scale = float(env.dt) if self.cfg.credit.integrate_amp_reward_dt else 1.0
        action_abs_max = 0.0
        disc_norm_policy_samples = 0
        window_latest_root_xy_abs_max = torch.zeros((), device=device)
        window_root_xy_abs_sum = torch.zeros((), device=device)
        window_root_xy_count = 0

        with torch.no_grad():
            for chunk_idx in range(chunks):
                chunk_actor_raw = obs.clone()
                chunk_critic_raw = critic_obs.clone()
                actor_obs_n = self._norm_actor(chunk_actor_raw, update=False)
                previous_action = chunk_actor_raw[..., -self.num_act :].detach()
                final_latent, latent_path, old_log_probs, _ = self._sample_cps_path(actor_obs_n)
                action_chunk = self._policy._action_transform(
                    final_latent, prev_action=previous_action
                ).view(n_envs, h, self.num_act)
                action_abs_max = max(action_abs_max, float(action_chunk.abs().max().item()))

                actor_obs_buf[chunk_idx] = actor_obs_n
                actor_obs_raw_buf[chunk_idx] = chunk_actor_raw
                actions_buf[chunk_idx] = action_chunk
                latent_path_buf[chunk_idx] = latent_path
                old_log_probs_buf[chunk_idx] = old_log_probs
                prev_action_buf[chunk_idx] = previous_action

                alive = torch.ones(n_envs, dtype=torch.bool, device=device)
                for frame_idx in range(h):
                    alive_before = alive.clone()
                    current_context_raw = self._prefix_context_raw(
                        critic_obs,
                        chunk_critic_raw,
                        chunk_actor_raw,
                        previous_action,
                        final_latent,
                        frame_idx,
                    )
                    action_t = action_chunk[:, frame_idx]
                    action_t = torch.where(
                        alive_before.unsqueeze(-1), action_t, torch.zeros_like(action_t)
                    )
                    next_obs, task_reward, done, info = env.step(action_t, auto_reset=False)
                    next_critic_obs = env.get_critic_observation()

                    # Standard AMP uses post-action windows. Reset contributes
                    # one real frame at age=0; legal D windows start at ages
                    # 1..W, so early post-reset frames are sample-masked.
                    amp_frame = info.get("amp_frame")
                    if amp_frame is None:
                        amp_frame = env.get_amp_policy_frame()
                    self.amp_history.push(amp_frame)
                    active_float = alive_before.to(dtype=task_reward.dtype)
                    ready_mask = self.amp_history.ready & alive_before
                    raw_windows = None
                    if bool(ready_mask.any()):
                        ready_ids = ready_mask.nonzero(as_tuple=False).squeeze(-1)
                        raw_windows = self.amp_history.flatten(
                            ready_ids, canonicalize_root=True
                        )
                        amp_valid_buf[chunk_idx, ready_ids, frame_idx] = True
                        flat_indices = (
                            (chunk_idx * n_envs + ready_ids) * h + frame_idx
                        ).detach().to(device="cpu", dtype=torch.long)
                        amp_window_chunks.append(raw_windows.detach().to("cpu", dtype=torch.float32))
                        amp_window_index_chunks.append(flat_indices)
                        amp_age_buf[chunk_idx, ready_ids, frame_idx] = self.amp_history.ages[
                            ready_ids
                        ]
                        window_view = raw_windows.view(
                            ready_ids.numel(), self.amp_history_steps, self.amp_frame_dim
                        )
                        window_latest_root_xy_abs_max = torch.maximum(
                            window_latest_root_xy_abs_max,
                            window_view[:, -1, :2].abs().max(),
                        )
                        window_root_xy_abs_sum += window_view[..., :2].abs().sum()
                        window_root_xy_count += window_view.shape[0] * window_view.shape[1] * 2
                    amp_reward_raw = torch.zeros_like(task_reward)
                    amp_logits = torch.zeros_like(task_reward)
                    amp_reward_credit = amp_reward_raw * amp_dt_scale
                    mixed_reward = task_weight * task_reward * active_float

                    task_reward_buf[chunk_idx, :, frame_idx] = task_reward * active_float
                    amp_reward_raw_buf[chunk_idx, :, frame_idx] = amp_reward_raw * active_float
                    amp_reward_credit_buf[chunk_idx, :, frame_idx] = amp_reward_credit * active_float
                    mixed_reward_buf[chunk_idx, :, frame_idx] = mixed_reward
                    amp_logit_buf[chunk_idx, :, frame_idx] = amp_logits * active_float
                    valid_buf[chunk_idx, :, frame_idx] = alive_before

                    timeouts = info["done_terms"]["time_out"].bool()
                    motion_complete = info["done_terms"].get("motion_complete")
                    motion_complete = (
                        motion_complete.bool()
                        if torch.is_tensor(motion_complete)
                        else torch.zeros_like(timeouts)
                    )
                    failures = (
                        info["done_terms"]["anchor_pos_bad"].bool()
                        | info["done_terms"]["anchor_ori_bad"].bool()
                        | info["done_terms"]["ee_body_bad"].bool()
                    )
                    new_done = alive_before & done.bool()
                    # Tracking failure has precedence if terminal causes overlap;
                    # it must never bootstrap merely because the time limit also
                    # fired on the same primitive step.
                    new_failure, new_timeout, new_motion_complete = (
                        resolve_terminal_masks(
                            new_done,
                            timeouts,
                            motion_complete,
                            failures,
                        )
                    )
                    done_buf[chunk_idx, :, frame_idx] = new_done
                    failure_buf[chunk_idx, :, frame_idx] = new_failure
                    timeout_buf[chunk_idx, :, frame_idx] = new_timeout
                    motion_complete_buf[chunk_idx, :, frame_idx] = new_motion_complete
                    # Timeout bootstraps from the terminal observation but never
                    # connects its GAE trace to the reset state.
                    bootstrap_buf[chunk_idx, :, frame_idx] = (
                        alive_before & ~new_failure & ~new_motion_complete
                    )
                    trace_buf[chunk_idx, :, frame_idx] = alive_before & ~new_done

                    if frame_idx < h - 1:
                        next_context_raw = self._prefix_context_raw(
                            next_critic_obs,
                            chunk_critic_raw,
                            chunk_actor_raw,
                            previous_action,
                            final_latent,
                            frame_idx + 1,
                        )
                    else:
                        zero_latent = torch.zeros_like(final_latent)
                        next_context_raw = self._prefix_context_raw(
                            next_critic_obs,
                            next_critic_obs,
                            next_obs,
                            action_t,
                            zero_latent,
                            0,
                        )
                    context_raw_buf[chunk_idx, :, frame_idx] = current_context_raw
                    next_context_raw_buf[chunk_idx, :, frame_idx] = next_context_raw
                    context_buf[chunk_idx, :, frame_idx] = self.prefix_context_normalizer(
                        current_context_raw, update=False
                    )
                    next_context_buf[chunk_idx, :, frame_idx] = self.prefix_context_normalizer(
                        next_context_raw, update=False
                    )

                    valid_ids = ready_mask.nonzero(as_tuple=False).squeeze(-1)
                    if valid_ids.numel() > 0:
                        keep = min(current_keep_per_step, int(valid_ids.numel()))
                        select_ids = valid_ids[
                            torch.randperm(valid_ids.numel(), device=device)[:keep]
                        ]
                        selected_windows = self.amp_history.flatten(
                            select_ids, canonicalize_root=True
                        )
                        self.current_disc_buffer.push(
                            selected_windows,
                            step=chunk_idx * h + frame_idx,
                        )
                        self.disc_normalizer.record(selected_windows)
                        disc_norm_policy_samples += int(selected_windows.shape[0])

                    self._record_train_episode_stats(
                        mixed_reward,
                        new_done,
                        step_counts=active_float,
                    )
                    rollout_info_items.append((info, alive_before.detach()))
                    alive = alive_before & ~done.bool()
                    obs = next_obs
                    critic_obs = next_critic_obs

                chunk_done = done_buf[chunk_idx].any(dim=-1)
                if bool(chunk_done.any()):
                    reset_ids = chunk_done.nonzero(as_tuple=False).squeeze(-1)
                    reset_phases = env.sample_phase_indices(
                        reset_ids.numel(), horizon=max(1, h)
                    )
                    reset_obs = env.reset_envs(reset_ids, phase_indices=reset_phases)
                    obs[reset_ids] = reset_obs
                    critic_obs = env.get_critic_observation()
                    self._reset_amp_history(env_ids=reset_ids)

            values = self._evaluate_prefix_values(context_buf)
            next_values = self._evaluate_prefix_values(next_context_buf)

            # [chunk,env,frame] -> chronological [time,env].
            def chronological(value: torch.Tensor) -> torch.Tensor:
                dims = list(range(value.ndim))
                order = [0, 2, 1] + dims[3:]
                return value.permute(*order).reshape(chunks * h, n_envs, *value.shape[3:])

            def chunk_layout(value: torch.Tensor) -> torch.Tensor:
                tail = value.shape[2:]
                return value.reshape(chunks, h, n_envs, *tail).permute(
                    0, 2, 1, *range(3, 3 + len(tail))
                )

        self._obs = obs
        self._critic_obs = critic_obs
        rollout = {
            "actor_obs": actor_obs_buf,
            "actor_obs_raw": actor_obs_raw_buf,
            "actions": actions_buf,
            "latents": latent_path_buf,
            "old_log_probs": old_log_probs_buf,
            "prev_action": prev_action_buf,
            "contexts": context_buf,
            "contexts_raw": context_raw_buf,
            "next_contexts_raw": next_context_raw_buf,
            "values": values,
            "next_values": next_values,
            "valid": valid_buf,
            "amp_valid": amp_valid_buf,
            "amp_windows": (
                torch.cat(amp_window_chunks, dim=0)
                if amp_window_chunks
                else torch.empty((0, self.amp_window_dim), dtype=torch.float32)
            ),
            "amp_window_indices": (
                torch.cat(amp_window_index_chunks, dim=0)
                if amp_window_index_chunks
                else torch.empty((0,), dtype=torch.long)
            ),
            "amp_window_age": amp_age_buf,
            "done": done_buf,
            "failure": failure_buf,
            "timeout": timeout_buf,
            "motion_complete": motion_complete_buf,
            "bootstrap_mask": bootstrap_buf,
            "trace_mask": trace_buf,
            "task_reward": task_reward_buf,
            "amp_reward_raw": amp_reward_raw_buf,
            "amp_reward_credit": amp_reward_credit_buf,
            "mixed_reward": mixed_reward_buf,
            "amp_logits": amp_logit_buf,
            "rollout_info_items": rollout_info_items,
            "action_abs_max": action_abs_max,
            "disc_norm_policy_samples": disc_norm_policy_samples,
            "window_latest_root_xy_abs_max": float(
                window_latest_root_xy_abs_max.item()
            ),
            "window_root_xy_abs_mean": float(
                (window_root_xy_abs_sum / max(window_root_xy_count, 1)).item()
            ),
            "next_observation": obs,
            "train_step_indices": self._train_step_indices(device),
        }
        self._assign_credit(rollout)
        return rollout

    @torch.no_grad()
    def _assign_credit(self, rollout: dict) -> None:
        chunks, n_envs, h = rollout["valid"].shape

        def chronological(value: torch.Tensor) -> torch.Tensor:
            dims = list(range(value.ndim))
            order = [0, 2, 1] + dims[3:]
            return value.permute(*order).reshape(chunks * h, n_envs, *value.shape[3:])

        def chunk_layout(value: torch.Tensor) -> torch.Tensor:
            tail = value.shape[2:]
            return value.reshape(chunks, h, n_envs, *tail).permute(
                0, 2, 1, *range(3, 3 + len(tail))
            )

        norm_name = str(self.cfg.credit.advantage_normalization)
        norm_map = {
            "per_channel_per_offset": "per_offset",
            "per_channel_global": "global",
            "per_offset": "per_offset",
            "global": "global",
            "none": "none",
        }
        credit_normalization = norm_map[norm_name]
        task_weight = float(self.cfg.credit.task_weight)
        amp_weight = float(self.cfg.credit.amp_weight)
        rewards_time = torch.stack(
            [
                chronological(rollout["task_reward"]),
                chronological(rollout["amp_reward_credit"]),
            ],
            dim=-1,
        )
        credit = compute_dual_channel_gae(
            rewards_time,
            chronological(rollout["values"]),
            chronological(rollout["next_values"]),
            chronological(rollout["bootstrap_mask"]),
            chronological(rollout["trace_mask"]),
            chronological(rollout["valid"]),
            gamma=float(self.cfg.discount_gamma),
            gae_lambda=float(self.cfg.gae_lambda),
            chunk_horizon=h,
            normalization=credit_normalization,
            actor_weights=(task_weight, amp_weight),
        )
        if self.cfg.credit.mode == "chunk_shared":
            credit = with_chunk_shared_actor_credit(
                credit,
                chronological(rollout["valid"]),
                chunk_horizon=h,
                normalization=credit_normalization,
                actor_weights=(task_weight, amp_weight),
            )
        rollout["advantages"] = chunk_layout(credit.actor_advantage)
        rollout["channel_advantages"] = chunk_layout(credit.advantages)
        rollout["normalized_channel_advantages"] = chunk_layout(
            credit.actor_advantage_components
        )
        rollout["mixed_advantage"] = chunk_layout(credit.mixed_advantage)
        rollout["value_targets"] = chunk_layout(credit.value_targets)
        rollout["td_errors"] = chunk_layout(credit.td_errors)

    @torch.no_grad()
    def _recompute_amp_rewards(self, rollout: dict) -> dict[str, float]:
        valid = rollout["amp_valid"]
        rollout["amp_logits"].zero_()
        rollout["amp_reward_raw"].zero_()
        rollout["amp_reward_credit"].zero_()
        windows_cpu = rollout["amp_windows"]
        indices_cpu = rollout["amp_window_indices"]
        if windows_cpu.shape[0] != indices_cpu.shape[0]:
            raise RuntimeError("AMP rollout windows and indices are misaligned")
        if windows_cpu.shape[0] > 0:
            flat_logits = rollout["amp_logits"].reshape(-1)
            flat_raw = rollout["amp_reward_raw"].reshape(-1)
            flat_credit = rollout["amp_reward_credit"].reshape(-1)
            amp_dt_scale = float(self.env.dt) if self.cfg.credit.integrate_amp_reward_dt else 1.0
            batch_size = max(1, int(self.cfg.amp.reward_eval_batch_size))
            for start in range(0, windows_cpu.shape[0], batch_size):
                end = min(start + batch_size, windows_cpu.shape[0])
                raw = windows_cpu[start:end].to(device=self.env.device, non_blocking=False)
                logits, rewards = self._evaluate_amp_reward(raw)
                idx = indices_cpu[start:end].to(device=self.env.device, non_blocking=False)
                flat_logits.index_copy_(0, idx, logits.to(flat_logits.dtype))
                flat_raw.index_copy_(0, idx, rewards.to(flat_raw.dtype))
                flat_credit.index_copy_(0, idx, (rewards * amp_dt_scale).to(flat_credit.dtype))
        active = rollout["valid"].to(dtype=rollout["task_reward"].dtype)
        rollout["mixed_reward"] = (
            float(self.cfg.credit.task_weight) * rollout["task_reward"]
            + float(self.cfg.credit.amp_weight) * rollout["amp_reward_credit"]
        ) * active
        self._assign_credit(rollout)
        return {
            "amp/valid_window_fraction": float(valid.float().mean().item()),
            "amp/valid_window_count": float(valid.sum().item()),
            "amp/age0_in_reward_count": float(
                ((rollout["amp_window_age"] == 0) & valid).sum().item()
            ),
        }

    # ------------------------------------------------------------------ #
    # Optimizers
    # ------------------------------------------------------------------ #
    def _actor_update(self, rollout: dict) -> dict[str, float]:
        device = self.env.device
        chunks, n_envs = rollout["actions"].shape[:2]
        batch_size = chunks * n_envs
        h = self.horizon_h
        flow_steps = int(self.cfg.flow_steps)
        actor_obs = rollout["actor_obs"].reshape(batch_size, self.actor_obs_dim)
        latent_path = rollout["latents"].reshape(
            batch_size, flow_steps + 1, self.chunk_dim
        )
        old_log_probs = rollout["old_log_probs"].reshape(batch_size, flow_steps, h)
        advantages = rollout["advantages"].reshape(batch_size, h)
        valid = rollout["valid"].reshape(batch_size, h)
        mini_batch_size = self._policy_mini_batch_size(batch_size)
        micro_batch_size = self._policy_micro_batch_size(mini_batch_size)
        clip_low = 1.0 - float(self.cfg.clip_range)
        clip_high = 1.0 + float(self.cfg.clip_range)
        if self.cfg.credit.ratio_mode != "joint_path":
            raise ValueError("FCAMP only supports credit.ratio_mode=joint_path")

        totals = {
            "policy_loss": 0.0,
            "kl": 0.0,
            "per_factor_kl": 0.0,
            "full_chunk_path_kl": 0.0,
            "ratio": 0.0,
            "clip": 0.0,
            "grad_norm": 0.0,
            "joint_log_ratio_abs_max": 0.0,
        }
        frame_kl = torch.zeros(h, device=device)
        frame_ratio = torch.zeros(h, device=device)
        frame_clip = torch.zeros(h, device=device)
        frame_count = torch.zeros(h, device=device)
        steps = 0
        early_stop_epoch = int(self.cfg.policy_epochs)

        for epoch in range(int(self.cfg.policy_epochs)):
            epoch_kl_sum = 0.0
            epoch_steps = 0
            permutation = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mini_batch_size):
                idx = permutation[start : start + mini_batch_size]
                if idx.numel() == 0:
                    continue
                self.actor_optimizer.zero_grad(set_to_none=True)
                mb_kl = 0.0
                mb_weight_total = 0.0
                for micro_start in range(0, idx.numel(), micro_batch_size):
                    sub = idx[micro_start : micro_start + micro_batch_size]
                    weight = float(sub.numel()) / float(idx.numel())
                    new_log_probs = self._recompute_cps_path_stats(
                        actor_obs[sub], latent_path[sub]
                    )
                    delta = new_log_probs - old_log_probs[sub]
                    adv = advantages[sub]
                    mask = valid[sub].to(dtype=delta.dtype)
                    frame_joint_log_ratio = delta.sum(dim=1)
                    log_ratio = frame_joint_log_ratio
                    ratio = torch.exp(log_ratio)
                    objective_adv = adv
                    objective_mask = mask
                    mask_sum = objective_mask.sum().clamp(min=1.0)
                    unclipped = -objective_adv * ratio
                    clipped = -objective_adv * torch.clamp(ratio, clip_low, clip_high)
                    policy_loss = (
                        torch.maximum(unclipped, clipped) * objective_mask
                    ).sum() / mask_sum
                    (policy_loss * weight).backward()

                    with torch.no_grad():
                        kl = 0.5 * log_ratio.square()
                        kl_mean = float((kl * objective_mask).sum().item() / mask_sum.item())
                        factor_mask = mask.unsqueeze(1).expand_as(delta)
                        factor_sum = factor_mask.sum().clamp(min=1.0)
                        per_factor_kl = float(
                            (0.5 * delta.square() * factor_mask).sum().item()
                            / factor_sum.item()
                        )
                        full_chunk_log_ratio = (frame_joint_log_ratio * mask).sum(dim=1)
                        chunk_active = (mask.sum(dim=1) > 0).to(delta.dtype)
                        chunk_count = chunk_active.sum().clamp(min=1.0)
                        full_chunk_kl = float(
                            (0.5 * full_chunk_log_ratio.square() * chunk_active).sum().item()
                            / chunk_count.item()
                        )
                        clipped_flag = ((ratio < clip_low) | (ratio > clip_high)).to(ratio.dtype)
                        totals["policy_loss"] += float(policy_loss.item()) * weight
                        totals["kl"] += kl_mean * weight
                        totals["per_factor_kl"] += per_factor_kl * weight
                        totals["full_chunk_path_kl"] += full_chunk_kl * weight
                        totals["ratio"] += float(
                            (ratio * objective_mask).sum().item() / mask_sum.item()
                        ) * weight
                        totals["clip"] += float(
                            (clipped_flag * objective_mask).sum().item() / mask_sum.item()
                        ) * weight
                        totals["joint_log_ratio_abs_max"] = max(
                            totals["joint_log_ratio_abs_max"],
                            float(frame_joint_log_ratio.abs().max().item()),
                        )
                        frame_kl += (0.5 * frame_joint_log_ratio.square() * mask).sum(dim=0)
                        frame_ratio += (torch.exp(frame_joint_log_ratio) * mask).sum(dim=0)
                        frame_clip += (
                            ((torch.exp(frame_joint_log_ratio) < clip_low) | (torch.exp(frame_joint_log_ratio) > clip_high))
                            .to(mask.dtype)
                            * mask
                        ).sum(dim=0)
                        frame_count += mask.sum(dim=0)
                    mb_kl += kl_mean * weight
                    mb_weight_total += weight
                grad_norm = nn.utils.clip_grad_norm_(
                    self._policy.parameters(), float(self.cfg.max_grad_norm)
                )
                self.actor_optimizer.step()
                totals["grad_norm"] += float(grad_norm)
                steps += 1
                epoch_kl_sum += mb_kl / max(mb_weight_total, 1e-8)
                epoch_steps += 1
            if (
                float(self.cfg.desired_kl) > 0
                and epoch_steps > 0
                and epoch_kl_sum / epoch_steps
                > float(self.cfg.kl_early_stop_factor) * float(self.cfg.desired_kl)
            ):
                early_stop_epoch = epoch + 1
                break

        denom = max(steps, 1)
        frame_count = frame_count.clamp(min=1.0)
        metrics = {
            "fcamp/policy_loss": totals["policy_loss"] / denom,
            "fcamp/kl": totals["kl"] / denom,
            "fcamp/per_factor_kl": totals["per_factor_kl"] / denom,
            "fcamp/full_chunk_path_kl": totals["full_chunk_path_kl"] / denom,
            "fcamp/ratio": totals["ratio"] / denom,
            "fcamp/clip_fraction": totals["clip"] / denom,
            "fcamp/actor_grad_norm": totals["grad_norm"] / denom,
            "fcamp/actor_lr": float(self.learning_rate),
            "fcamp/actor_optimizer_steps": float(steps),
            "fcamp/actor_early_stop_epoch": float(early_stop_epoch),
            "fcamp/joint_log_ratio_abs_max": totals["joint_log_ratio_abs_max"],
            "fcamp/ratio_mode": 2.0,
        }
        for frame_idx in range(h):
            metrics[f"fcamp/frame_{frame_idx}_kl"] = float(
                (frame_kl[frame_idx] / frame_count[frame_idx]).item()
            )
            metrics[f"fcamp/frame_{frame_idx}_ratio"] = float(
                (frame_ratio[frame_idx] / frame_count[frame_idx]).item()
            )
            metrics[f"fcamp/frame_{frame_idx}_clip"] = float(
                (frame_clip[frame_idx] / frame_count[frame_idx]).item()
            )
        return metrics

    def _critic_update(self, rollout: dict) -> dict[str, float]:
        contexts = rollout["contexts"].reshape(-1, self.prefix_context_dim)
        targets = rollout["value_targets"].reshape(-1, 2)
        valid = rollout["valid"].reshape(-1)
        valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            raise RuntimeError("FCAMP rollout has no valid critic samples")
        batch_size = int(valid_idx.numel())
        mini_batch_size = self._policy_mini_batch_size(batch_size)
        micro_batch_size = self._policy_micro_batch_size(mini_batch_size)
        task_weight = float(self.cfg.critics.task_loss_weight)
        amp_weight = float(self.cfg.critics.amp_loss_weight)
        totals = {"task": 0.0, "amp": 0.0, "grad": 0.0}
        steps = 0
        for _ in range(int(self.cfg.policy_epochs)):
            perm = valid_idx[torch.randperm(batch_size, device=valid_idx.device)]
            for start in range(0, batch_size, mini_batch_size):
                idx = perm[start : start + mini_batch_size]
                self.critic_optimizer.zero_grad(set_to_none=True)
                for micro_start in range(0, idx.numel(), micro_batch_size):
                    sub = idx[micro_start : micro_start + micro_batch_size]
                    weight = float(sub.numel()) / float(idx.numel())
                    loss_channels = self.critic.flow_matching_loss(
                        contexts[sub],
                        targets[sub],
                        fm_samples=self.flow_critic_fm_samples,
                    )
                    task_loss = loss_channels[:, 0].mean()
                    amp_loss = loss_channels[:, 1].mean()
                    loss = task_weight * task_loss + amp_weight * amp_loss
                    (loss * weight).backward()
                    totals["task"] += float(task_loss.item()) * weight
                    totals["amp"] += float(amp_loss.item()) * weight
                grad = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), float(self.cfg.max_grad_norm)
                )
                self.critic_optimizer.step()
                totals["grad"] += float(grad)
                steps += 1
        denom = max(steps, 1)
        return {
            "critic/task_flow_loss": totals["task"] / denom,
            "critic/amp_flow_loss": totals["amp"] / denom,
            "critic/grad_norm": totals["grad"] / denom,
            "critic/lr": float(self.critic_learning_rate),
            "critic/optimizer_steps": float(steps),
        }

    def _store_current_in_replay(self, update_idx: int) -> None:
        if len(self.current_disc_buffer) == 0:
            return
        if self.disc_replay.is_full:
            count = min(int(self.cfg.amp.replay_samples), len(self.current_disc_buffer))
            windows = self.current_disc_buffer.sample(count)
        else:
            windows = self.current_disc_buffer.get_all(dtype=torch.float32)
        self.disc_replay.push(windows, step=update_idx)

    def _record_expert_normalizer_samples(self, num_samples: int) -> None:
        """Record a demo count exactly matched to rollout policy windows."""

        remaining = int(num_samples)
        batch_size = int(self.cfg.amp.batch_size)
        while remaining > 0:
            count = min(remaining, batch_size)
            expert_raw = self.env.sample_amp_demo_windows(
                count, self.amp_history_steps, flatten=True
            )
            self.disc_normalizer.record(expert_raw)
            remaining -= count

    def _discriminator_update(
        self,
        update_idx: int,
        *,
        expert_normalizer_samples: int,
    ) -> dict[str, float]:
        if len(self.current_disc_buffer) == 0:
            return {
                "disc/update_steps": 0.0,
                "disc/version": float(self.disc_version),
                "disc/skipped_no_current": 1.0,
            }
        self._store_current_in_replay(update_idx)
        batch_size = int(self.cfg.amp.batch_size)
        possible_steps = math.ceil(len(self.current_disc_buffer) / batch_size) * int(
            self.cfg.amp.epochs
        )
        update_steps = min(possible_steps, int(self.cfg.amp.max_updates_per_iteration))
        self._record_expert_normalizer_samples(expert_normalizer_samples)
        self.disc_normalizer.unfreeze()
        committed = self.disc_normalizer.commit()
        self.disc_normalizer.freeze()
        totals: dict[str, float] = {}
        grad_total = 0.0
        canonical_root_xy_max = {"current": 0.0, "replay": 0.0, "expert": 0.0}
        self.discriminator.train()
        for disc_step in range(update_steps):
            current_raw = self.current_disc_buffer.sample(
                batch_size, device=self.env.device, dtype=torch.float32
            )
            replay_raw = self.disc_replay.sample(
                batch_size, device=self.env.device, dtype=torch.float32
            )
            expert_raw = self.env.sample_amp_demo_windows(
                batch_size, self.amp_history_steps, flatten=True
            )
            if disc_step == 0:
                for name, raw in (
                    ("current", current_raw),
                    ("replay", replay_raw),
                    ("expert", expert_raw),
                ):
                    window = raw.view(-1, self.amp_history_steps, self.amp_frame_dim)
                    canonical_root_xy_max[name] = float(
                        window[:, -1, :2].abs().max().item()
                    )
            current = self.disc_normalizer.normalize(current_raw)
            replay = self.disc_normalizer.normalize(replay_raw)
            expert = self.disc_normalizer.normalize(expert_raw)
            output = compute_amp_discriminator_loss(
                self.discriminator,
                expert_observations=expert,
                policy_observations=current,
                replay_observations=replay,
                gradient_penalty_weight=float(self.cfg.amp.grad_penalty),
                logit_regularization_weight=float(self.cfg.amp.logit_reg),
            )
            self.disc_optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            # Record the true norm without modifying gradients; standard
            # MimicKit AMP does not configure discriminator grad clipping.
            grad = nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), float("inf")
            )
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
                "disc/lr": float(self.cfg.amp.learning_rate),
                "disc/update_steps": float(update_steps),
                "disc/version": float(self.disc_version),
                "disc_norm/committed_this_update": float(committed),
                "disc_norm/policy_samples_update": float(expert_normalizer_samples),
                "disc_norm/expert_samples_update": float(expert_normalizer_samples),
                "disc_window/current_latest_root_xy_abs_max": canonical_root_xy_max["current"],
                "disc_window/replay_latest_root_xy_abs_max": canonical_root_xy_max["replay"],
                "disc_window/expert_latest_root_xy_abs_max": canonical_root_xy_max["expert"],
            }
        )
        current_metrics = {
            key.replace("replay/", "disc_current/"): value
            for key, value in self.current_disc_buffer.statistics().items()
        }
        metrics.update(current_metrics)
        metrics.update(
            {
                key.replace("replay/", "disc_replay/"): value
                for key, value in self.disc_replay.statistics(current_step=update_idx).items()
            }
        )
        return metrics

    def update(self, rollout: dict, collect_time: float) -> dict:
        update_start = time.perf_counter()
        disc_start = time.perf_counter()
        # Global primitive steps are used as the replay age clock.
        update_idx = int(getattr(self, "_fcamp_update_idx", 1))
        disc_metrics = self._discriminator_update(
            update_idx,
            expert_normalizer_samples=int(rollout["disc_norm_policy_samples"]),
        )
        disc_time = time.perf_counter() - disc_start
        reward_metrics = self._recompute_amp_rewards(rollout)

        actor_start = time.perf_counter()
        actor_metrics = self._actor_update(rollout)
        actor_time = time.perf_counter() - actor_start
        critic_start = time.perf_counter()
        critic_metrics = self._critic_update(rollout)
        critic_time = time.perf_counter() - critic_start

        # Commit actor/prefix normalizers only after every optimizer has used
        # the rollout snapshot.
        if self.empirical_normalization:
            with torch.no_grad():
                self.actor_obs_normalizer(
                    rollout["actor_obs_raw"].reshape(-1, self.actor_obs_dim),
                    update=True,
                )
                context_valid = rollout["valid"].reshape(-1)
                self.prefix_context_normalizer(
                    rollout["contexts_raw"].reshape(-1, self.prefix_context_dim)[context_valid],
                    update=True,
                )
                self.prefix_context_normalizer(
                    rollout["next_contexts_raw"].reshape(-1, self.prefix_context_dim)[context_valid],
                    update=True,
                )
        self.disc_normalizer.unfreeze()

        valid = rollout["valid"]
        amp_valid = rollout["amp_valid"]
        metrics: dict[str, float] = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        metrics.update(disc_metrics)
        metrics.update(reward_metrics)
        metrics.update(_masked_stats("reward/task", rollout["task_reward"], valid))
        metrics.update(_masked_stats("reward/amp_raw", rollout["amp_reward_raw"], amp_valid))
        metrics.update(_masked_stats("reward/amp_credit", rollout["amp_reward_credit"], amp_valid))
        metrics.update(_masked_stats("reward/mixed", rollout["mixed_reward"], valid))
        metrics.update(_masked_stats("credit/task_adv", rollout["channel_advantages"][..., 0], valid))
        metrics.update(_masked_stats("credit/amp_adv", rollout["channel_advantages"][..., 1], valid))
        metrics.update(_masked_stats("credit/actor_adv", rollout["advantages"], valid))
        metrics.update(_masked_stats("critic/task_value", rollout["values"][..., 0], valid))
        metrics.update(_masked_stats("critic/amp_value", rollout["values"][..., 1], valid))
        metrics.update(_masked_stats("critic/task_target", rollout["value_targets"][..., 0], valid))
        metrics.update(_masked_stats("critic/amp_target", rollout["value_targets"][..., 1], valid))
        metrics.update(
            amp_reward_statistics(
                rollout["amp_logits"][amp_valid],
                rollout["amp_reward_raw"][amp_valid],
                scale=float(self.cfg.amp.reward_scale),
                minimum_one_minus_prob=float(self.cfg.amp.reward_epsilon),
                prefix="amp_reward",
            )
        )
        for frame_idx in range(self.horizon_h):
            frame_valid = valid[..., frame_idx]
            metrics.update(
                _masked_stats(
                    f"credit/frame_{frame_idx}_task_adv",
                    rollout["channel_advantages"][..., frame_idx, 0],
                    frame_valid,
                )
            )
            metrics.update(
                _masked_stats(
                    f"credit/frame_{frame_idx}_amp_adv",
                    rollout["channel_advantages"][..., frame_idx, 1],
                    frame_valid,
                )
            )
        metrics.update(self.disc_normalizer.statistics())
        metrics.update(self.amp_history.statistics())
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
                "act/abs_max": float(rollout["action_abs_max"]),
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
                    update_idx * int(self.cfg.rollout_env_steps) * self.env.num_envs
                ),
                "system/cuda_peak_allocated_gib": float(
                    torch.cuda.max_memory_allocated(self.env.device) / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                ),
            }
        )
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
            f"ratio={metrics.get('fcamp/ratio', float('nan')):.4f} "
            f"clip={metrics.get('fcamp/clip_fraction', float('nan')):.4f} "
            f"grad={metrics.get('fcamp/actor_grad_norm', float('nan')):.4f} "
            f"lr={metrics.get('fcamp/actor_lr', float('nan')):.6f}",
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
            f"[TIME] collect={metrics['timing/collect_s']:.3f}s "
            f"actor={metrics['timing/actor_update_s']:.3f}s "
            f"critic={metrics['timing/critic_update_s']:.3f}s "
            f"disc={metrics['timing/disc_update_s']:.3f}s "
            f"gpu_peak={metrics['system/cuda_peak_allocated_gib']:.2f}GiB",
            flush=True,
        )

    def log_banner(self) -> None:
        print(
            "[INFO] Starting FC-AMP: causal Flow-CPS H=4 + standard AMP W=16 "
            "+ causal frame credit + shared-trunk dual Flow heads",
            flush=True,
        )
        print(
            f"[INFO] task={self.env.task.name} actor_obs={self.actor_obs_dim} "
            f"prefix_context={self.prefix_context_dim} action_dim={self.num_act} "
            f"H={self.horizon_h} W_D={self.amp_history_steps} "
            f"amp_frame={self.amp_frame_dim} amp_window={self.amp_window_dim}",
            flush=True,
        )
        print(
            f"[INFO] critic_sharing=encoder heads=task,amp "
            f"credit={self.cfg.credit.mode} advantage_norm={self.cfg.credit.advantage_normalization} "
            f"ratio_mode={self.cfg.credit.ratio_mode} "
            f"task_weight={self.cfg.credit.task_weight} amp_weight={self.cfg.credit.amp_weight} "
            f"amp_dt={self.cfg.credit.integrate_amp_reward_dt}",
            flush=True,
        )
        print(
            f"[INFO] discriminator=standard_mlp hidden={list(self.cfg.amp.hidden_dims)} "
            f"BCE=True GP={self.cfg.amp.grad_penalty} replay={self.cfg.amp.replay_size} "
            f"EMA=False independent_trunk=True motion_end_terminal=True "
            f"fcamp_schema=1",
            flush=True,
        )
