"""FC-AMP: causal Flow-chunk policy optimization with temporal discriminator prior.

The discriminator remains an independent MimicKit-style reward model.  Its
online representation is never part of the actor or critic observation; only
the scalar temporal style reward is coupled to the corresponding causal action
conditional.
"""

from __future__ import annotations

import math
import time

import torch
from torch import nn

from components.credit.temporal_credit import (
    compute_dual_channel_gae,
    normalize_actor_mixture,
    resolve_terminal_masks,
    with_chunk_shared_actor_credit,
)
from components.rollout.flow_cps_base import FlowCPSBase
from components.rollout.training_streams import (
    CURRICULUM_STREAM,
    PHASE0_STREAM,
    Phase0AttemptTracker,
    Phase0CurriculumStreams,
)
from components.imitation.style_reward import (
    discriminator_style_reward,
    style_reward_statistics,
)
from components.imitation.temporal_history import TemporalFeatureHistory
from components.imitation.window_pipeline import TemporalWindowPipeline
from components.normalization.running_stats import RunningNormalizer
from components.replay.frame_trajectory import FrameTrajectoryReplay
from models.style_discriminator import (
    StyleDiscriminator,
    compute_style_discriminator_loss,
)
from models.mlp_actor_critic import EmpiricalNormalization
from models.dual_flow_critic import SharedTrunkDualFlowCritic


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


class FCAMP(FlowCPSBase):
    """Full H=4 causal Flow-CPS policy with W=16 temporal discriminator prior."""

    def build(self) -> None:
        cfg = self.cfg
        amp_cfg = cfg.amp
        critic_cfg = cfg.critics
        env = self.env

        self.imitation_history_steps = int(amp_cfg.obs_steps)
        super().build()

        self.training_streams = Phase0CurriculumStreams.create(
            env.num_envs,
            phase0_fraction=float(cfg.streams.phase0_fraction),
            phase0_start=int(env.motion_start_phase),
            device=env.device,
        )
        self.phase0_attempts = Phase0AttemptTracker(
            self.training_streams.stream_ids
        )
        # Failure phases produced after accumulated phase-0 trajectory error do
        # not describe a pristine reset state. Only the curriculum stream may
        # update the predecessor-reset sampler.
        env.set_adaptive_failure_eligibility(
            self.training_streams.curriculum_mask
        )

        # A temporal discriminator must never observe a silent motion teleport.
        env.terminate_on_motion_end = True

        # Context = current privileged state, chunk-start privileged and actor
        # states, previous action, padded causal latent prefix, prefix mask and
        # offset one-hot.  The online discriminator state is deliberately absent.
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

        self.imitation_frame_dim = int(env.imitation_frame_dim)
        self.imitation_window_dim = self.imitation_history_steps * self.imitation_frame_dim
        self.imitation_pipeline = TemporalWindowPipeline(
            self.imitation_history_steps,
            self.imitation_frame_dim,
        )
        self.discriminator = StyleDiscriminator(
            self.imitation_window_dim, tuple(amp_cfg.hidden_dims)
        ).to(env.device)
        self.disc_normalizer = RunningNormalizer(
            self.imitation_window_dim,
            device=env.device,
            clip=float(amp_cfg.normalizer_clip),
        )
        self.imitation_history = TemporalFeatureHistory(
            env.num_envs,
            self.imitation_history_steps,
            self.imitation_frame_dim,
            device=env.device,
        )
        # Replay is CPU-backed FP32 frame replay: every post-action frame is
        # stored once, and discriminator windows are reconstructed only through
        # continuous predecessor chains.
        self.disc_frame_replay = FrameTrajectoryReplay(
            int(amp_cfg.replay_size),
            self.imitation_frame_dim,
            num_envs=env.num_envs,
        )
        disc_parameters = [p for p in self.discriminator.parameters() if p.requires_grad]
        optimizer_name = amp_cfg.optimizer.lower()
        optimizer_kwargs = {
            "lr": float(amp_cfg.learning_rate),
            "weight_decay": float(amp_cfg.weight_decay),
        }
        if optimizer_name == "sgd":
            # MimicKit's MPOptimizer uses momentum=0.9 and no gradient clip for
            # the standard G1 style discriminator.
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
    def validate_checkpoint_payload(self, payload: dict) -> None:
        state = payload.get("algo_state")
        if not isinstance(state, dict) or int(
            state.get("fcamp_schema_version", 0)
        ) != 6:
            raise ValueError(
                "FCAMP checkpoint predates the reward-only discriminator "
                "architecture; start a fresh run."
            )
        if bool(state.get("discriminator_policy_conditioning", True)):
            raise ValueError(
                "FCAMP checkpoints with discriminator-conditioned policies "
                "cannot be resumed."
            )

    def extra_checkpoint_state(self) -> dict:
        payload = super().extra_checkpoint_state()
        payload.update(
            {
                "disc_optimizer": self.disc_optimizer.state_dict(),
                "disc_version": int(self.disc_version),
                "disc_frame_replay": self.disc_frame_replay.state_dict(),
                "fcamp_schema_version": 6,
                "imitation_history_steps": self.imitation_history_steps,
                "imitation_frame_dim": self.imitation_frame_dim,
                "prefix_context_dim": self.prefix_context_dim,
                "discriminator_policy_conditioning": False,
                "phase0_stream_fraction": float(
                    self.training_streams.phase0_fraction
                ),
                "phase0_stream_count": int(
                    self.training_streams.phase0_ids.numel()
                ),
                "stream_ids": self.training_streams.stream_ids.detach().cpu(),
                "phase0_attempt_tracker": self.phase0_attempts.state_dict(),
            }
        )
        return payload

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        super().load_extra_checkpoint_state(payload, reset_optimizer=reset_optimizer)
        if not payload:
            return
        schema = int(payload.get("fcamp_schema_version", 0))
        if schema != 6:
            raise ValueError(
                "FCAMP checkpoint predates the reward-only discriminator "
                "architecture; start a fresh run."
            )
        if bool(payload.get("discriminator_policy_conditioning", True)):
            raise ValueError(
                "FCAMP checkpoints with discriminator-conditioned policies "
                "cannot be resumed."
            )
        saved_stream_ids = payload.get("stream_ids")
        if not torch.is_tensor(saved_stream_ids) or not torch.equal(
            saved_stream_ids.to(dtype=torch.int8, device="cpu"),
            self.training_streams.stream_ids.detach().to("cpu"),
        ):
            raise ValueError("FCAMP checkpoint training-stream assignment differs")
        if int(payload.get("phase0_stream_count", -1)) != int(
            self.training_streams.phase0_ids.numel()
        ):
            raise ValueError("FCAMP checkpoint phase0 stream count differs")
        self.phase0_attempts.load_state_dict(
            payload.get("phase0_attempt_tracker")
        )
        if not reset_optimizer and "disc_optimizer" in payload:
            self.disc_optimizer.load_state_dict(payload["disc_optimizer"])
        self.disc_version = int(payload.get("disc_version", self.disc_version))
        if not self.disc_frame_replay.load_state_dict(payload.get("disc_frame_replay")):
            print("[FCAMP] discriminator frame replay absent/incompatible; starting empty", flush=True)

    # ------------------------------------------------------------------ #
    # Causal prefix contexts and style reward
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
    def _update_empirical_normalizer_chunked(
        self,
        normalizer: EmpiricalNormalization,
        samples: torch.Tensor,
        valid: torch.Tensor | None = None,
        stream_labels: torch.Tensor | None = None,
    ) -> None:
        """Update running moments without materializing a full normalized copy.

        FCAMP keeps primitive prefix contexts for the two value heads.  At the
        production 8192-env shape, boolean-selecting and then forwarding the
        complete ``[6*8192*4, context_dim]`` tensor creates an unnecessary
        gigabyte-scale output.  Streaming the exact same empirical-moment
        update bounds temporary memory independently of the rollout size.
        """

        flat = samples.reshape(-1, samples.shape[-1])
        flat_valid = None if valid is None else valid.reshape(-1).bool()
        if flat_valid is not None and flat_valid.shape[0] != flat.shape[0]:
            raise ValueError("normalizer samples and validity mask are misaligned")
        selected_indices: torch.Tensor | None = None
        if stream_labels is not None:
            flat_streams = stream_labels.reshape(-1).to(
                device=flat.device,
                dtype=torch.int8,
            )
            if flat_streams.shape[0] != flat.shape[0]:
                raise ValueError(
                    "normalizer samples and stream labels are misaligned"
                )
            eligible_indices = (
                torch.arange(flat.shape[0], device=flat.device)
                if flat_valid is None
                else flat_valid.nonzero(as_tuple=False).squeeze(-1)
            )
            eligible_streams = flat_streams.index_select(
                0,
                eligible_indices,
            )
            specs = self._stream_specs(eligible_streams)
            balanced_total = int(
                min(
                    indices.numel() / objective_weight
                    for _, _, objective_weight, indices in specs
                )
            )
            selections: list[torch.Tensor] = []
            for _, _, objective_weight, local_indices in specs:
                quota = min(
                    int(local_indices.numel()),
                    int(round(balanced_total * objective_weight)),
                )
                order = torch.randperm(
                    local_indices.numel(),
                    device=flat.device,
                )[:quota]
                selected_local = local_indices.index_select(0, order)
                selections.append(
                    eligible_indices.index_select(0, selected_local)
                )
            selected_indices = torch.cat(selections, dim=0)
        batch_size = min(max(1, int(self.cfg.micro_batch_size)), 1024)
        sample_count = (
            int(selected_indices.numel())
            if selected_indices is not None
            else int(flat.shape[0])
        )
        for start in range(0, sample_count, batch_size):
            if selected_indices is None:
                batch = flat[start : start + batch_size]
                if flat_valid is not None:
                    batch = batch[
                        flat_valid[start : start + batch_size]
                    ]
            else:
                batch = flat.index_select(
                    0,
                    selected_indices[start : start + batch_size],
                )
            if batch.shape[0] > 0:
                normalizer._update(batch)

    @torch.no_grad()
    def _evaluate_amp_reward(
        self,
        flat_windows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the independent temporal discriminator in bounded batches."""

        if flat_windows.ndim != 2 or flat_windows.shape[-1] != self.imitation_window_dim:
            raise ValueError("FCAMP discriminator windows have an incompatible shape")
        batch_size = max(1, int(self.cfg.amp.reward_eval_batch_size))
        logits: list[torch.Tensor] = []
        self.discriminator.eval()
        for start in range(0, flat_windows.shape[0], batch_size):
            batch = flat_windows[start : start + batch_size]
            normalized = self.imitation_pipeline.normalize_flat(
                batch,
                self.disc_normalizer,
            )
            logits.append(self.discriminator(normalized))
        all_logits = torch.cat(logits, dim=0)
        rewards = discriminator_style_reward(
            all_logits,
            scale=float(self.cfg.amp.reward_scale),
            minimum_one_minus_prob=float(self.cfg.amp.reward_epsilon),
        )
        return all_logits, rewards

    def _reset_imitation_history(
        self,
        phase_indices: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        del phase_indices
        initial_frame = self.env.get_imitation_policy_frame(env_ids)
        self.imitation_history.reset(initial_frame, env_ids=env_ids)

    def _reset_training_streams(
        self,
        *,
        randomize_curriculum_episode_age: bool,
    ) -> torch.Tensor:
        env = self.env
        self.phase0_attempts.interrupt_inflight()
        env_ids = torch.arange(
            env.num_envs,
            device=env.device,
            dtype=torch.long,
        )
        phases, reset_streams = self.training_streams.reset_phases(
            env_ids,
            lambda count: env.sample_phase_indices(
                count,
                horizon=max(1, self.horizon_h),
            ),
        )
        obs = env.reset(
            phase_indices=phases,
            reset_stream_ids=reset_streams,
        )
        env.episode_steps[self.training_streams.phase0_ids] = 0
        if (
            randomize_curriculum_episode_age
            and bool(self.cfg.init_at_random_ep_len)
            and self.max_episode_steps > 0
            and self.training_streams.curriculum_ids.numel() > 0
        ):
            curriculum_ids = self.training_streams.curriculum_ids
            random_age = torch.randint(
                0,
                int(self.max_episode_steps),
                (curriculum_ids.numel(),),
                device=env.device,
                dtype=env.episode_steps.dtype,
            )
            env.episode_steps.index_copy_(0, curriculum_ids, random_age)
        self._obs = obs
        self._critic_obs = env.get_critic_observation()
        self._reset_imitation_history(env.phase_steps.clone())
        self.phase0_attempts.start(self.training_streams.phase0_ids)
        return obs

    def initial_reset(self) -> torch.Tensor:
        return self._reset_training_streams(
            randomize_curriculum_episode_age=True,
        )

    def reset_after_resume(self) -> torch.Tensor:
        # Checkpointer invokes this after sampler and RNG restoration. Resume is
        # an explicit new episode boundary because simulator state is not stored.
        return self._reset_training_streams(
            randomize_curriculum_episode_age=False,
        )

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.base_actor_obs_dim:
            raise ValueError(
                f"FCAMP expected raw actor observation dim {self.base_actor_obs_dim}, "
                f"got {tuple(obs.shape)}"
            )
        return super().deterministic_actions(obs)

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        # Trainer owns the canonical update clock, including after resume.
        self._fcamp_update_idx = int(update_idx)
        self.phase0_attempts.begin_update()
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

        # One rollout is generated, history-conditioned and rewarded by one
        # immutable committed discriminator snapshot.  As in the local
        # MimicKit AMP implementation, policy/value optimization consumes these
        # rewards before the discriminator or its normalizer is advanced.
        self.discriminator.eval()
        self.disc_normalizer.freeze()
        rollout_disc_version = int(self.disc_version)
        rollout_disc_normalizer_count = float(self.disc_normalizer.count.item())

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
        current_disc_window_chunks: list[torch.Tensor] = []
        current_disc_end_time_chunks: list[torch.Tensor] = []
        current_disc_stream_chunks: list[torch.Tensor] = []
        done_buf = torch.zeros_like(valid_buf)
        failure_buf = torch.zeros_like(valid_buf)
        timeout_buf = torch.zeros_like(valid_buf)
        motion_complete_buf = torch.zeros_like(valid_buf)
        terminal_phase_buf = torch.full(
            (chunks, n_envs, h),
            -1.0,
            dtype=torch.float32,
            device=device,
        )
        bootstrap_buf = torch.zeros_like(valid_buf)
        trace_buf = torch.zeros_like(valid_buf)

        obs = current_obs
        critic_obs = self._critic_obs
        collection_start_phases = env.phase_steps.detach().clone()
        rollout_info_items: list[tuple[dict, torch.Tensor]] = []
        current_keep_per_step = max(
            1, math.ceil(int(self.cfg.amp.current_buffer_size) / total_primitive_steps)
        )
        task_weight = float(self.cfg.credit.task_weight)
        amp_weight = float(self.cfg.credit.amp_weight)
        amp_dt_scale = float(env.dt) if self.cfg.credit.integrate_amp_reward_dt else 1.0
        action_abs_max = 0.0
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

                    # Temporal discriminator prior uses post-action windows. Reset contributes
                    # one real frame at age=0; legal D windows start at ages
                    # 1..W, so early post-reset frames are sample-masked.
                    imitation_frame = info.get("imitation_frame")
                    if imitation_frame is None:
                        imitation_frame = env.get_imitation_policy_frame()
                    self.imitation_history.push(imitation_frame)
                    alive_ids = alive_before.nonzero(as_tuple=False).squeeze(-1)
                    if alive_ids.numel() > 0:
                        self.disc_frame_replay.push_frames(
                            imitation_frame.index_select(0, alive_ids),
                            env_ids=alive_ids,
                            episode_ids=env.episode_ids.index_select(0, alive_ids),
                            reference_times=info["imitation_frame_phase_steps"].index_select(
                                0, alive_ids
                            ),
                            ages=self.imitation_history.ages.index_select(0, alive_ids),
                            update=int(getattr(self, "_fcamp_update_idx", 0)),
                            stream=self.training_streams.stream_ids.index_select(
                                0, alive_ids
                            ),
                        )
                    active_float = alive_before.to(dtype=task_reward.dtype)
                    amp_reward_raw = torch.zeros_like(task_reward)
                    amp_logits = torch.zeros_like(task_reward)
                    ready_mask = self.imitation_history.ready & alive_before
                    if bool(ready_mask.any()):
                        ready_ids = ready_mask.nonzero(as_tuple=False).squeeze(-1)
                        raw_window_frames = self.imitation_history.window(ready_ids)
                        raw_windows = self.imitation_pipeline.flatten(raw_window_frames)
                        ready_logits, ready_rewards = self._evaluate_amp_reward(
                            raw_windows
                        )
                        amp_logits.index_copy_(0, ready_ids, ready_logits)
                        amp_reward_raw.index_copy_(0, ready_ids, ready_rewards)
                        amp_valid_buf[chunk_idx, ready_ids, frame_idx] = True
                        amp_age_buf[chunk_idx, ready_ids, frame_idx] = self.imitation_history.ages[
                            ready_ids
                        ]
                        window_view = raw_windows.view(
                            ready_ids.numel(), self.imitation_history_steps, self.imitation_frame_dim
                        )
                        window_latest_root_xy_abs_max = torch.maximum(
                            window_latest_root_xy_abs_max,
                            window_view[:, -1, :2].abs().max(),
                        )
                        window_root_xy_abs_sum += window_view[..., :2].abs().sum()
                        window_root_xy_count += window_view.shape[0] * window_view.shape[1] * 2
                    amp_reward_credit = amp_reward_raw * amp_dt_scale
                    mixed_reward = (
                        task_weight * task_reward + amp_weight * amp_reward_credit
                    ) * active_float

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
                    self.phase0_attempts.observe_step(
                        alive_before,
                        new_done,
                        new_failure,
                        new_timeout,
                        new_motion_complete,
                    )
                    termination_phases = info.get("termination_phase_steps")
                    if torch.is_tensor(termination_phases) and bool(new_done.any()):
                        terminal_phase_buf[chunk_idx, new_done, frame_idx] = (
                            termination_phases[new_done].to(dtype=torch.float32)
                        )
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
                        select_ids, selected_streams = (
                            self.training_streams.balanced_sample(
                                valid_ids,
                                current_keep_per_step,
                            )
                        )
                        selected_windows = self.imitation_pipeline.flatten(
                            self.imitation_history.window(select_ids)
                        )
                        current_disc_window_chunks.append(
                            selected_windows.detach().to("cpu", dtype=torch.float32)
                        )
                        current_disc_end_time_chunks.append(
                            info["imitation_frame_phase_steps"]
                            .index_select(0, select_ids)
                            .detach()
                            .to("cpu", dtype=torch.long)
                        )
                        current_disc_stream_chunks.append(
                            selected_streams.detach().to(
                                "cpu",
                                dtype=torch.int8,
                            )
                        )
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
                    reset_phases, reset_streams = (
                        self.training_streams.reset_phases(
                            reset_ids,
                            lambda count: env.sample_phase_indices(
                                count,
                                horizon=max(1, h),
                            ),
                        )
                    )
                    reset_obs = env.reset_envs(
                        reset_ids,
                        phase_indices=reset_phases,
                        reset_stream_ids=reset_streams,
                    )
                    obs[reset_ids] = reset_obs
                    critic_obs = env.get_critic_observation()
                    self._reset_imitation_history(env_ids=reset_ids)
                    self.phase0_attempts.start(reset_ids)

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
            "current_disc_windows": (
                torch.cat(current_disc_window_chunks, dim=0)
                if current_disc_window_chunks
                else torch.empty((0, self.imitation_window_dim), dtype=torch.float32)
            ),
            "current_disc_end_times": (
                torch.cat(current_disc_end_time_chunks, dim=0)
                if current_disc_end_time_chunks
                else torch.empty((0,), dtype=torch.long)
            ),
            "current_disc_stream_ids": (
                torch.cat(current_disc_stream_chunks, dim=0)
                if current_disc_stream_chunks
                else torch.empty((0,), dtype=torch.int8)
            ),
            "imitation_window_age": amp_age_buf,
            "done": done_buf,
            "failure": failure_buf,
            "timeout": timeout_buf,
            "motion_complete": motion_complete_buf,
            "terminal_phase": terminal_phase_buf,
            "bootstrap_mask": bootstrap_buf,
            "trace_mask": trace_buf,
            "task_reward": task_reward_buf,
            "amp_reward_raw": amp_reward_raw_buf,
            "amp_reward_credit": amp_reward_credit_buf,
            "mixed_reward": mixed_reward_buf,
            "amp_logits": amp_logit_buf,
            "disc_version_used": rollout_disc_version,
            "disc_normalizer_count_used": rollout_disc_normalizer_count,
            "rollout_info_items": rollout_info_items,
            "stream_ids": self.training_streams.stream_ids,
            "collection_start_phases": collection_start_phases,
            "action_abs_max": action_abs_max,
            "window_latest_root_xy_abs_max": float(
                window_latest_root_xy_abs_max.item()
            ),
            "window_root_xy_abs_mean": float(
                (window_root_xy_abs_sum / max(window_root_xy_count, 1)).item()
            ),
            "next_observation": obs,
            "train_step_indices": self._train_step_indices(device),
        }
        if int(self.disc_version) != rollout_disc_version:
            raise RuntimeError("FCAMP discriminator changed while collecting one rollout")
        if float(self.disc_normalizer.count.item()) != rollout_disc_normalizer_count:
            raise RuntimeError("FCAMP discriminator normalizer changed while collecting one rollout")
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

        if str(self.cfg.credit.advantage_normalization) != "global":
            raise ValueError(
                "FCAMP requires one global normalization of the weighted "
                "actor advantage"
            )
        task_weight = float(self.cfg.credit.task_weight)
        amp_weight = float(self.cfg.credit.amp_weight)
        rewards_time = torch.stack(
            [
                chronological(rollout["task_reward"]),
                chronological(rollout["amp_reward_credit"]),
            ],
            dim=-1,
        )
        values_time = chronological(rollout["values"])
        next_values_time = chronological(rollout["next_values"])
        bootstrap_time = chronological(rollout["bootstrap_mask"])
        trace_time = chronological(rollout["trace_mask"])
        valid_time = chronological(rollout["valid"])
        # Dual GAE is vectorized over independent environments.  Compute it once
        # without normalization so task/style critic targets remain raw.
        credit = compute_dual_channel_gae(
            rewards_time,
            values_time,
            next_values_time,
            bootstrap_time,
            trace_time,
            valid_time,
            gamma=float(self.cfg.discount_gamma),
            gae_lambda=float(self.cfg.gae_lambda),
            chunk_horizon=h,
            normalization="none",
            actor_weights=(task_weight, amp_weight),
        )
        if self.cfg.credit.mode == "chunk_shared":
            credit = with_chunk_shared_actor_credit(
                credit,
                valid_time,
                chunk_horizon=h,
                normalization="none",
                actor_weights=(task_weight, amp_weight),
            )

        # Match the exact actor objective q_s * mean_s(loss): every valid sample
        # in stream s carries q_s / N_s normalization mass.  This is one global
        # scalar transform, not one transform per channel or per stream.
        normalization_weights = torch.zeros_like(
            valid_time,
            dtype=rewards_time.dtype,
        )
        for _, _, objective_weight, env_ids in self._stream_specs(
            self.training_streams.stream_ids
        ):
            stream_valid = valid_time.index_select(1, env_ids)
            valid_count = int(stream_valid.sum().item())
            if valid_count <= 0:
                raise RuntimeError("FCAMP stream has no valid actor samples")
            stream_weights = (
                stream_valid.to(dtype=rewards_time.dtype)
                * (float(objective_weight) / float(valid_count))
            )
            normalization_weights.index_copy_(1, env_ids, stream_weights)
        credit = normalize_actor_mixture(
            credit,
            valid_time,
            normalization_weights,
        )

        rollout["advantages"] = chunk_layout(credit.actor_advantage)
        rollout["channel_advantages"] = chunk_layout(credit.advantages)
        rollout["actor_advantage_components"] = chunk_layout(
            credit.actor_advantage_components
        )
        rollout["mixed_advantage"] = chunk_layout(credit.mixed_advantage)
        rollout["value_targets"] = chunk_layout(credit.value_targets)
        rollout["td_errors"] = chunk_layout(credit.td_errors)

    # ------------------------------------------------------------------ #
    # Optimizers
    # ------------------------------------------------------------------ #
    def _stream_specs(
        self,
        labels: torch.Tensor,
    ) -> list[tuple[str, int, float, torch.Tensor]]:
        configured_phase0 = float(
            getattr(
                getattr(self.cfg, "streams", None),
                "phase0_fraction",
                1.0,
            )
        )
        configured = (
            ("phase0", PHASE0_STREAM, configured_phase0),
            ("curriculum", CURRICULUM_STREAM, 1.0 - configured_phase0),
        )
        active: list[tuple[str, int, float, torch.Tensor]] = []
        for name, stream_id, weight in configured:
            indices = (labels == stream_id).nonzero(
                as_tuple=False
            ).squeeze(-1)
            if indices.numel() > 0 and weight > 0.0:
                active.append((name, stream_id, weight, indices))
        weight_sum = sum(item[2] for item in active)
        if not active or weight_sum <= 0.0:
            raise RuntimeError("FCAMP has no active training stream")
        return [
            (name, stream_id, weight / weight_sum, indices)
            for name, stream_id, weight, indices in active
        ]

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
        env_stream_ids = rollout.get("stream_ids")
        if env_stream_ids is None:
            env_stream_ids = torch.full(
                (n_envs,),
                PHASE0_STREAM,
                dtype=torch.int8,
                device=device,
            )
        stream_labels = env_stream_ids.reshape(1, n_envs).expand(
            chunks, n_envs
        ).reshape(-1)
        stream_specs = self._stream_specs(stream_labels)
        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        micro_batch_size = self._policy_micro_batch_size(
            self._policy_mini_batch_size(batch_size)
        )
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
        metric_names = (
            "policy_loss",
            "kl",
            "per_factor_kl",
            "full_chunk_path_kl",
            "ratio",
            "clip",
        )
        stream_totals = {
            name: {key: 0.0 for key in metric_names}
            for name, _, _, _ in stream_specs
        }
        stream_steps = {name: 0 for name, _, _, _ in stream_specs}
        frame_totals = {
            name: {
                "kl": torch.zeros(h, device=device),
                "ratio": torch.zeros(h, device=device),
                "clip": torch.zeros(h, device=device),
                "count": torch.zeros(h, device=device),
            }
            for name, _, _, _ in stream_specs
        }
        steps = 0
        early_stop_epoch = int(self.cfg.policy_epochs)
        actor_lr_start = float(self.learning_rate)
        lr_decrease_steps = 0
        lr_increase_steps = 0
        lr_hold_steps = 0

        for epoch in range(int(self.cfg.policy_epochs)):
            epoch_kl_sum = 0.0
            epoch_steps = 0
            stream_splits: dict[str, tuple[float, tuple[torch.Tensor, ...]]] = {}
            for name, _, objective_weight, indices in stream_specs:
                shuffled = indices.index_select(
                    0,
                    torch.randperm(indices.numel(), device=device),
                )
                stream_splits[name] = (
                    objective_weight,
                    torch.tensor_split(shuffled, num_mini_batches),
                )
            for mini_batch_index in range(num_mini_batches):
                parts = [
                    (
                        name,
                        objective_weight,
                        splits[mini_batch_index],
                    )
                    for name, (
                        objective_weight,
                        splits,
                    ) in stream_splits.items()
                    if splits[mini_batch_index].numel() > 0
                ]
                if not parts:
                    continue
                self.actor_optimizer.zero_grad(set_to_none=True)
                combined_metrics = {key: 0.0 for key in metric_names}
                for name, objective_weight, idx in parts:
                    valid_denominator = float(
                        valid.index_select(0, idx).sum().item()
                    )
                    if valid_denominator <= 0.0:
                        raise RuntimeError(
                            f"FCAMP {name} actor minibatch has no valid frames"
                        )
                    stream_sums = {
                        "policy_loss": 0.0,
                        "kl": 0.0,
                        "per_factor_kl": 0.0,
                        "full_chunk_path_kl": 0.0,
                        "ratio": 0.0,
                        "clip": 0.0,
                    }
                    factor_denominator = valid_denominator * flow_steps
                    chunk_denominator = float(
                        (
                            valid.index_select(0, idx).sum(dim=1) > 0
                        ).sum().item()
                    )
                    for micro_start in range(
                        0, idx.numel(), micro_batch_size
                    ):
                        sub = idx[
                            micro_start : micro_start + micro_batch_size
                        ]
                        new_log_probs = self._recompute_cps_path_stats(
                            actor_obs[sub],
                            latent_path[sub],
                        )
                        delta = new_log_probs - old_log_probs[sub]
                        adv = advantages[sub]
                        mask = valid[sub].to(dtype=delta.dtype)
                        log_ratio = delta.sum(dim=1)
                        ratio = torch.exp(log_ratio)
                        unclipped = -adv * ratio
                        clipped = -adv * torch.clamp(
                            ratio,
                            clip_low,
                            clip_high,
                        )
                        policy_sum = (
                            torch.maximum(unclipped, clipped) * mask
                        ).sum()
                        (
                            policy_sum
                            * (objective_weight / valid_denominator)
                        ).backward()

                        with torch.no_grad():
                            kl = 0.5 * log_ratio.square()
                            clipped_flag = (
                                (ratio < clip_low) | (ratio > clip_high)
                            ).to(ratio.dtype)
                            factor_mask = mask.unsqueeze(1).expand_as(delta)
                            full_chunk_log_ratio = (
                                log_ratio * mask
                            ).sum(dim=1)
                            chunk_active = (
                                mask.sum(dim=1) > 0
                            ).to(delta.dtype)
                            stream_sums["policy_loss"] += float(
                                policy_sum.item()
                            )
                            stream_sums["kl"] += float(
                                (kl * mask).sum().item()
                            )
                            stream_sums["per_factor_kl"] += float(
                                (
                                    0.5
                                    * delta.square()
                                    * factor_mask
                                ).sum().item()
                            )
                            stream_sums[
                                "full_chunk_path_kl"
                            ] += float(
                                (
                                    0.5
                                    * full_chunk_log_ratio.square()
                                    * chunk_active
                                ).sum().item()
                            )
                            stream_sums["ratio"] += float(
                                (ratio * mask).sum().item()
                            )
                            stream_sums["clip"] += float(
                                (clipped_flag * mask).sum().item()
                            )
                            totals["joint_log_ratio_abs_max"] = max(
                                totals["joint_log_ratio_abs_max"],
                                float(log_ratio.abs().max().item()),
                            )
                            frame_totals[name]["kl"] += (
                                kl * mask
                            ).sum(dim=0)
                            frame_totals[name]["ratio"] += (
                                ratio * mask
                            ).sum(dim=0)
                            frame_totals[name]["clip"] += (
                                clipped_flag * mask
                            ).sum(dim=0)
                            frame_totals[name]["count"] += mask.sum(
                                dim=0
                            )
                    stream_means = {
                        "policy_loss": (
                            stream_sums["policy_loss"]
                            / valid_denominator
                        ),
                        "kl": stream_sums["kl"] / valid_denominator,
                        "per_factor_kl": (
                            stream_sums["per_factor_kl"]
                            / max(factor_denominator, 1.0)
                        ),
                        "full_chunk_path_kl": (
                            stream_sums["full_chunk_path_kl"]
                            / max(chunk_denominator, 1.0)
                        ),
                        "ratio": stream_sums["ratio"] / valid_denominator,
                        "clip": stream_sums["clip"] / valid_denominator,
                    }
                    for key, value in stream_means.items():
                        combined_metrics[key] += objective_weight * value
                        stream_totals[name][key] += value
                    stream_steps[name] += 1

                observed_mb_kl = combined_metrics["kl"]
                previous_lr = float(self.learning_rate)
                self._update_adaptive_learning_rates(observed_mb_kl)
                if self.learning_rate < previous_lr:
                    lr_decrease_steps += 1
                elif self.learning_rate > previous_lr:
                    lr_increase_steps += 1
                else:
                    lr_hold_steps += 1
                grad_norm = nn.utils.clip_grad_norm_(
                    self._policy.parameters(), float(self.cfg.max_grad_norm)
                )
                self.actor_optimizer.step()
                for key in metric_names:
                    totals[key] += combined_metrics[key]
                totals["grad_norm"] += float(grad_norm)
                steps += 1
                epoch_kl_sum += observed_mb_kl
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
        metrics = {
            "fcamp/policy_loss": totals["policy_loss"] / denom,
            "fcamp/kl": totals["kl"] / denom,
            "fcamp/per_factor_kl": totals["per_factor_kl"] / denom,
            "fcamp/full_chunk_path_kl": totals["full_chunk_path_kl"] / denom,
            "fcamp/ratio": totals["ratio"] / denom,
            "fcamp/clip_fraction": totals["clip"] / denom,
            "fcamp/actor_grad_norm": totals["grad_norm"] / denom,
            "fcamp/actor_lr_start": actor_lr_start,
            "fcamp/actor_lr": float(self.learning_rate),
            "fcamp/kl_target_per_step": float(self.cfg.desired_kl),
            "fcamp/kl_units": float(self.kl_units),
            "fcamp/lr_decrease_steps": float(lr_decrease_steps),
            "fcamp/lr_increase_steps": float(lr_increase_steps),
            "fcamp/lr_hold_steps": float(lr_hold_steps),
            "fcamp/actor_optimizer_steps": float(steps),
            "fcamp/actor_early_stop_epoch": float(early_stop_epoch),
            "fcamp/joint_log_ratio_abs_max": totals["joint_log_ratio_abs_max"],
            "fcamp/ratio_mode": 2.0,
        }
        objective_weights = {
            name: weight for name, _, weight, _ in stream_specs
        }
        for name, _, objective_weight, _ in stream_specs:
            stream_denom = max(stream_steps[name], 1)
            metrics[
                f"stream/{name}/actor_objective_weight"
            ] = objective_weight
            for key in metric_names:
                metrics[f"stream/{name}/actor_{key}"] = (
                    stream_totals[name][key] / stream_denom
                )
        for frame_idx in range(h):
            frame_values = {"kl": 0.0, "ratio": 0.0, "clip": 0.0}
            for name in objective_weights:
                count = float(
                    frame_totals[name]["count"][frame_idx].item()
                )
                if count <= 0.0:
                    continue
                for key in frame_values:
                    frame_values[key] += objective_weights[name] * float(
                        (
                            frame_totals[name][key][frame_idx]
                            / count
                        ).item()
                    )
            metrics[f"fcamp/frame_{frame_idx}_kl"] = float(
                frame_values["kl"]
            )
            metrics[f"fcamp/frame_{frame_idx}_ratio"] = float(
                frame_values["ratio"]
            )
            metrics[f"fcamp/frame_{frame_idx}_clip"] = float(
                frame_values["clip"]
            )
        return metrics

    def _critic_update(self, rollout: dict) -> dict[str, float]:
        chunks, n_envs, horizon = rollout["valid"].shape
        contexts = rollout["contexts"].reshape(-1, self.prefix_context_dim)
        targets = rollout["value_targets"].reshape(-1, 2)
        valid = rollout["valid"].reshape(-1)
        env_stream_ids = rollout.get("stream_ids")
        if env_stream_ids is None:
            env_stream_ids = torch.full(
                (n_envs,),
                PHASE0_STREAM,
                dtype=torch.int8,
                device=valid.device,
            )
        stream_labels = env_stream_ids.reshape(1, n_envs, 1).expand(
            chunks,
            n_envs,
            horizon,
        ).reshape(-1)
        valid_labels = stream_labels[valid]
        valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            raise RuntimeError("FCAMP rollout has no valid critic samples")
        local_specs = self._stream_specs(valid_labels)
        stream_specs = [
            (
                name,
                stream_id,
                objective_weight,
                valid_idx.index_select(0, local_indices),
            )
            for name, stream_id, objective_weight, local_indices in local_specs
        ]
        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        micro_batch_size = self._policy_micro_batch_size(
            self._policy_mini_batch_size(int(valid_idx.numel()))
        )
        task_weight = float(self.cfg.critics.task_loss_weight)
        amp_weight = float(self.cfg.critics.amp_loss_weight)
        totals = {"task": 0.0, "amp": 0.0, "grad": 0.0}
        stream_totals = {
            name: {"task": 0.0, "amp": 0.0}
            for name, _, _, _ in stream_specs
        }
        stream_steps = {name: 0 for name, _, _, _ in stream_specs}
        steps = 0
        for _ in range(int(self.cfg.policy_epochs)):
            stream_splits: dict[str, tuple[float, tuple[torch.Tensor, ...]]] = {}
            for name, _, objective_weight, indices in stream_specs:
                shuffled = indices.index_select(
                    0,
                    torch.randperm(indices.numel(), device=indices.device),
                )
                stream_splits[name] = (
                    objective_weight,
                    torch.tensor_split(shuffled, num_mini_batches),
                )
            for mini_batch_index in range(num_mini_batches):
                parts = [
                    (
                        name,
                        objective_weight,
                        splits[mini_batch_index],
                    )
                    for name, (
                        objective_weight,
                        splits,
                    ) in stream_splits.items()
                    if splits[mini_batch_index].numel() > 0
                ]
                if not parts:
                    continue
                self.critic_optimizer.zero_grad(set_to_none=True)
                combined_task = 0.0
                combined_amp = 0.0
                for name, objective_weight, idx in parts:
                    denominator = float(idx.numel())
                    task_sum = 0.0
                    amp_sum = 0.0
                    for micro_start in range(
                        0,
                        idx.numel(),
                        micro_batch_size,
                    ):
                        sub = idx[
                            micro_start : micro_start + micro_batch_size
                        ]
                        loss_channels = self.critic.flow_matching_loss(
                            contexts[sub],
                            targets[sub],
                            fm_samples=self.flow_critic_fm_samples,
                        )
                        task_loss_sum = loss_channels[:, 0].sum()
                        amp_loss_sum = loss_channels[:, 1].sum()
                        loss = objective_weight * (
                            task_weight * task_loss_sum
                            + amp_weight * amp_loss_sum
                        ) / denominator
                        loss.backward()
                        task_sum += float(task_loss_sum.item())
                        amp_sum += float(amp_loss_sum.item())
                    task_mean = task_sum / denominator
                    amp_mean = amp_sum / denominator
                    combined_task += objective_weight * task_mean
                    combined_amp += objective_weight * amp_mean
                    stream_totals[name]["task"] += task_mean
                    stream_totals[name]["amp"] += amp_mean
                    stream_steps[name] += 1
                grad = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), float(self.cfg.max_grad_norm)
                )
                self.critic_optimizer.step()
                totals["task"] += combined_task
                totals["amp"] += combined_amp
                totals["grad"] += float(grad)
                steps += 1
        denom = max(steps, 1)
        metrics = {
            "critic/task_flow_loss": totals["task"] / denom,
            "critic/amp_flow_loss": totals["amp"] / denom,
            "critic/grad_norm": totals["grad"] / denom,
            "critic/lr": float(self.critic_learning_rate),
            "critic/optimizer_steps": float(steps),
        }
        for name, _, objective_weight, _ in stream_specs:
            stream_denom = max(stream_steps[name], 1)
            metrics[
                f"stream/{name}/critic_objective_weight"
            ] = objective_weight
            metrics[f"stream/{name}/critic_task_flow_loss"] = (
                stream_totals[name]["task"] / stream_denom
            )
            metrics[f"stream/{name}/critic_amp_flow_loss"] = (
                stream_totals[name]["amp"] / stream_denom
            )
        return metrics

    def _sample_cpu_flat_with_end_times(
        self,
        windows_cpu: torch.Tensor,
        end_times_cpu: torch.Tensor,
        batch_size: int,
        *,
        stream_ids_cpu: torch.Tensor | None = None,
        stream_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if windows_cpu.ndim != 2 or windows_cpu.shape[1] != self.imitation_window_dim:
            raise ValueError("current discriminator windows are malformed")
        if windows_cpu.shape[0] != end_times_cpu.shape[0]:
            raise RuntimeError("current discriminator windows/end-times are misaligned")
        candidates = torch.arange(windows_cpu.shape[0], device="cpu")
        if stream_id is not None:
            if (
                stream_ids_cpu is None
                or stream_ids_cpu.shape != end_times_cpu.shape
            ):
                raise RuntimeError(
                    "current discriminator stream labels are misaligned"
                )
            candidates = candidates[
                stream_ids_cpu.to(device="cpu", dtype=torch.int8)
                == int(stream_id)
            ]
        if candidates.numel() == 0:
            raise RuntimeError("cannot sample empty current discriminator windows")
        draw = torch.randint(
            candidates.numel(),
            (int(batch_size),),
            device="cpu",
        )
        indices = candidates.index_select(0, draw)
        raw = windows_cpu.index_select(0, indices).to(
            device=self.env.device,
            dtype=torch.float32,
            non_blocking=False,
        )
        ends = end_times_cpu.index_select(0, indices)
        return raw, ends

    def _disc_stream_quotas(
        self,
        batch_size: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        phase0 = int(
            round(
                int(batch_size)
                * float(self.cfg.streams.phase0_fraction)
            )
        )
        phase0 = min(max(1, phase0), int(batch_size) - 1)
        return (
            (PHASE0_STREAM, phase0),
            (CURRICULUM_STREAM, int(batch_size) - phase0),
        )

    def _sample_balanced_current_windows(
        self,
        windows_cpu: torch.Tensor,
        end_times_cpu: torch.Tensor,
        stream_ids_cpu: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_parts: list[torch.Tensor] = []
        end_parts: list[torch.Tensor] = []
        for stream_id, count in self._disc_stream_quotas(batch_size):
            raw, ends = self._sample_cpu_flat_with_end_times(
                windows_cpu,
                end_times_cpu,
                count,
                stream_ids_cpu=stream_ids_cpu,
                stream_id=stream_id,
            )
            raw_parts.append(raw)
            end_parts.append(ends)
        return torch.cat(raw_parts, dim=0), torch.cat(end_parts, dim=0)

    def _sample_balanced_replay_windows(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame_parts: list[torch.Tensor] = []
        end_parts: list[torch.Tensor] = []
        for stream_id, count in self._disc_stream_quotas(batch_size):
            frames, ends = self.disc_frame_replay.sample_windows(
                count,
                self.imitation_history_steps,
                stream_id=stream_id,
            )
            if frames.shape[0] != count:
                raise RuntimeError(
                    f"replay stream {stream_id} has no legal discriminator window"
                )
            frame_parts.append(frames)
            end_parts.append(ends)
        return torch.cat(frame_parts, dim=0), torch.cat(end_parts, dim=0)

    def _expert_flat_at_end_times(self, end_times_cpu: torch.Tensor) -> torch.Tensor:
        raw = self.env.get_imitation_demo_windows_at_end_indices(
            end_times_cpu.to(device=self.env.device, dtype=torch.long),
            self.imitation_history_steps,
            flatten=False,
        )
        return self.imitation_pipeline.flatten(raw)

    @torch.no_grad()
    def _record_matched_disc_normalizer(
        self,
        current_windows_cpu: torch.Tensor,
        current_end_times_cpu: torch.Tensor,
        current_stream_ids_cpu: torch.Tensor,
        sample_count: int,
    ) -> int:
        """Record an exact fixed-mixture policy/expert moment update."""

        if current_windows_cpu.shape[0] == 0 or sample_count <= 0:
            return 0
        self.disc_normalizer.clear_pending()
        batch_size = max(1, int(self.cfg.amp.batch_size))
        recorded = 0
        while recorded < int(sample_count):
            count = min(batch_size, int(sample_count) - recorded)
            if count < 2:
                break
            current, end_times = self._sample_balanced_current_windows(
                current_windows_cpu,
                current_end_times_cpu,
                current_stream_ids_cpu,
                count,
            )
            expert = self._expert_flat_at_end_times(end_times)
            self.disc_normalizer.record(current)
            self.disc_normalizer.record(expert)
            recorded += int(current.shape[0])
        return recorded

    def _discriminator_update(
        self,
        update_idx: int,
        *,
        rollout: dict,
    ) -> dict[str, float]:
        current_windows_cpu = rollout["current_disc_windows"]
        current_end_times_cpu = rollout["current_disc_end_times"]
        current_stream_ids_cpu = rollout["current_disc_stream_ids"]
        current_count = int(current_windows_cpu.shape[0])
        input_version = int(self.disc_version)
        if not (
            current_end_times_cpu.shape == current_stream_ids_cpu.shape
            and current_end_times_cpu.shape[0] == current_count
        ):
            raise RuntimeError(
                "FCAMP current discriminator windows have misaligned metadata"
            )
        current_phase0_count = int(
            (current_stream_ids_cpu == PHASE0_STREAM).sum().item()
        )
        current_curriculum_count = int(
            (current_stream_ids_cpu == CURRICULUM_STREAM).sum().item()
        )
        if current_count == 0:
            self.disc_normalizer.freeze()
            return {
                "disc/update_steps": 0.0,
                "disc/version": float(self.disc_version),
                "disc/input_version": float(input_version),
                "disc/skipped_no_current": 1.0,
                "disc/current_count": 0.0,
                "disc_norm/committed_this_update": 0.0,
                "disc_norm/policy_samples_update": 0.0,
                "disc_norm/expert_samples_update": 0.0,
                "disc/stream_quota_available": 0.0,
            }
        batch_size = int(self.cfg.amp.batch_size)
        if current_phase0_count == 0 or current_curriculum_count == 0:
            self.disc_normalizer.freeze()
            return {
                "disc/update_steps": 0.0,
                "disc/version": float(self.disc_version),
                "disc/input_version": float(input_version),
                "disc/skipped_missing_stream_quota": 1.0,
                "disc/stream_quota_available": 0.0,
                "disc/current_count": float(current_count),
                "disc/current_phase0_count": float(current_phase0_count),
                "disc/current_curriculum_count": float(
                    current_curriculum_count
                ),
                "disc_norm/committed_this_update": 0.0,
                "disc_norm/policy_samples_update": 0.0,
                "disc_norm/expert_samples_update": 0.0,
            }
        try:
            first_replay_window_frames, first_replay_ends = (
                self._sample_balanced_replay_windows(batch_size)
            )
        except RuntimeError:
            self.disc_normalizer.freeze()
            metrics = {
                "disc/update_steps": 0.0,
                "disc/version": float(self.disc_version),
                "disc/input_version": float(input_version),
                "disc/skipped_missing_stream_quota": 1.0,
                "disc/stream_quota_available": 0.0,
                "disc/current_count": float(current_count),
                "disc/current_phase0_count": float(current_phase0_count),
                "disc/current_curriculum_count": float(
                    current_curriculum_count
                ),
                "disc_norm/committed_this_update": 0.0,
                "disc_norm/policy_samples_update": 0.0,
                "disc_norm/expert_samples_update": 0.0,
            }
            metrics.update(
                {
                    key.replace("replay/", "disc_replay/"): value
                    for key, value in self.disc_frame_replay.statistics(
                        current_step=update_idx
                    ).items()
                }
            )
            return metrics
        possible_steps = math.ceil(current_count / batch_size) * int(
            self.cfg.amp.epochs
        )
        update_steps = min(possible_steps, int(self.cfg.amp.max_updates_per_iteration))
        # Record pending statistics now, but train D on the same committed
        # normalization snapshot that produced this rollout's rewards/history.
        # Commit only after D_old has been consumed, matching method/amp.py.
        self.disc_normalizer.freeze()
        normalizer_samples = self._record_matched_disc_normalizer(
            current_windows_cpu,
            current_end_times_cpu,
            current_stream_ids_cpu,
            min(current_count, batch_size * update_steps),
        )
        totals: dict[str, float] = {}
        grad_total = 0.0
        canonical_root_xy_max = {"current": 0.0, "replay": 0.0, "expert": 0.0}
        endpoint_abs_diff_total = 0.0
        self.discriminator.train()
        for disc_step in range(update_steps):
            current_raw, current_ends = self._sample_balanced_current_windows(
                current_windows_cpu,
                current_end_times_cpu,
                current_stream_ids_cpu,
                batch_size,
            )
            if disc_step == 0:
                replay_window_frames = first_replay_window_frames
                replay_ends = first_replay_ends
            else:
                replay_window_frames, replay_ends = (
                    self._sample_balanced_replay_windows(batch_size)
                )
            replay_raw = self.imitation_pipeline.flatten(
                replay_window_frames
            ).to(
                device=self.env.device,
                dtype=torch.float32,
                non_blocking=False,
            )
            fake_ends = torch.cat((current_ends, replay_ends.to(dtype=torch.long)), dim=0)
            expert_indices = torch.randint(fake_ends.shape[0], (batch_size,), device="cpu")
            expert_end_times = fake_ends.index_select(0, expert_indices)
            expert_raw = self._expert_flat_at_end_times(expert_end_times).to(
                device=self.env.device,
                dtype=torch.float32,
            )
            endpoint_abs_diff_total += float(
                (expert_end_times.float() - current_ends.float()).abs().mean().item()
            )
            if disc_step == 0:
                for name, raw in (
                    ("current", current_raw),
                    ("replay", replay_raw),
                    ("expert", expert_raw),
                ):
                    window = raw.view(-1, self.imitation_history_steps, self.imitation_frame_dim)
                    canonical_root_xy_max[name] = float(
                        window[:, -1, :2].abs().max().item()
                    )
            current = self.imitation_pipeline.normalize_flat(current_raw, self.disc_normalizer)
            replay = self.imitation_pipeline.normalize_flat(replay_raw, self.disc_normalizer)
            expert = self.imitation_pipeline.normalize_flat(expert_raw, self.disc_normalizer)
            output = compute_style_discriminator_loss(
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
        self.disc_normalizer.unfreeze()
        committed = self.disc_normalizer.commit()
        self.disc_normalizer.freeze()
        denom = max(update_steps, 1)
        metrics = {key: value / denom for key, value in totals.items()}
        metrics.update(
            {
                "disc/grad_norm": grad_total / denom,
                "disc/lr": float(self.cfg.amp.learning_rate),
                "disc/update_steps": float(update_steps),
                "disc/version": float(self.disc_version),
                "disc/input_version": float(input_version),
                "disc_norm/committed_this_update": float(committed),
                "disc_norm/policy_samples_update": float(normalizer_samples),
                "disc_norm/expert_samples_update": float(normalizer_samples),
                "disc/current_count": float(current_count),
                "disc/current_phase0_count": float(current_phase0_count),
                "disc/current_curriculum_count": float(
                    current_curriculum_count
                ),
                "disc/current_phase0_fraction": float(
                    current_phase0_count / max(current_count, 1)
                ),
                "disc/training_phase0_fraction": float(
                    self.cfg.streams.phase0_fraction
                ),
                "disc/replay_phase0_fraction": float(
                    self.cfg.streams.phase0_fraction
                ),
                "disc/stream_quota_available": 1.0,
                "disc/current_unique_endpoint_count": float(
                    torch.unique(current_end_times_cpu).numel()
                ),
                "disc/replay_fallback_current_count": 0.0,
                "disc/expert_endpoint_abs_diff_mean": endpoint_abs_diff_total / denom,
                "disc_window/current_latest_root_xy_abs_max": canonical_root_xy_max["current"],
                "disc_window/replay_latest_root_xy_abs_max": canonical_root_xy_max["replay"],
                "disc_window/expert_latest_root_xy_abs_max": canonical_root_xy_max["expert"],
            }
        )
        metrics.update(
            {
                key.replace("replay/", "disc_replay/"): value
                for key, value in self.disc_frame_replay.statistics(current_step=update_idx).items()
            }
        )
        return metrics

    def _optimize_rollout_snapshot(
        self,
        rollout: dict,
        update_idx: int,
    ) -> tuple[dict, dict, dict, dict, float, float, float]:
        """Optimize actor/value on D_old rewards, then advance D for next rollout."""

        rollout_disc_version = int(rollout["disc_version_used"])
        if rollout_disc_version != int(self.disc_version):
            raise RuntimeError(
                "FCAMP rollout discriminator snapshot is stale: "
                f"rollout={rollout_disc_version} current={self.disc_version}"
            )
        normalizer_count_used = float(rollout["disc_normalizer_count_used"])
        if normalizer_count_used != float(self.disc_normalizer.count.item()):
            raise RuntimeError("FCAMP rollout discriminator normalizer snapshot is stale")

        amp_valid = rollout["amp_valid"]
        reward_metrics = {
            "amp/valid_window_fraction": float(amp_valid.float().mean().item()),
            "amp/valid_window_count": float(amp_valid.sum().item()),
            "amp/age0_in_reward_count": float(
                ((rollout["imitation_window_age"] == 0) & amp_valid).sum().item()
            ),
            "amp/reward_disc_version": float(rollout_disc_version),
            "amp/reward_recomputed_after_disc": 0.0,
            "disc_norm/count_used_for_rollout": normalizer_count_used,
            "policy/discriminator_conditioned": 0.0,
        }

        actor_start = time.perf_counter()
        actor_metrics = self._actor_update(rollout)
        actor_time = time.perf_counter() - actor_start
        critic_start = time.perf_counter()
        critic_metrics = self._critic_update(rollout)
        critic_time = time.perf_counter() - critic_start

        # Commit actor/prefix normalizers only after both optimizers have used
        # the rollout snapshot. Stream the exact moment update at 8192 envs.
        if self.empirical_normalization:
            with torch.no_grad():
                chunks, n_envs, horizon = rollout["valid"].shape
                chunk_streams = rollout["stream_ids"].reshape(
                    1, n_envs
                ).expand(chunks, n_envs)
                frame_streams = chunk_streams.unsqueeze(-1).expand(
                    chunks,
                    n_envs,
                    horizon,
                )
                self._update_empirical_normalizer_chunked(
                    self.actor_obs_normalizer,
                    rollout["actor_obs_raw"],
                    stream_labels=chunk_streams,
                )
                self._update_empirical_normalizer_chunked(
                    self.prefix_context_normalizer,
                    rollout["contexts_raw"],
                    rollout["valid"],
                    stream_labels=frame_streams,
                )
                self._update_empirical_normalizer_chunked(
                    self.prefix_context_normalizer,
                    rollout["next_contexts_raw"],
                    rollout["valid"],
                    stream_labels=frame_streams,
                )

        disc_start = time.perf_counter()
        disc_metrics = self._discriminator_update(update_idx, rollout=rollout)
        disc_time = time.perf_counter() - disc_start
        if int(disc_metrics["disc/input_version"]) != rollout_disc_version:
            raise RuntimeError("FCAMP discriminator advanced before actor/value optimization")
        return (
            actor_metrics,
            critic_metrics,
            disc_metrics,
            reward_metrics,
            actor_time,
            critic_time,
            disc_time,
        )

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
                    rollout["amp_reward_raw"][
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
        update_idx = int(getattr(self, "_fcamp_update_idx", 1))
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
        metrics: dict[str, float] = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        metrics.update(disc_metrics)
        metrics.update(reward_metrics)
        metrics.update(self._stream_rollout_metrics(rollout))
        metrics.update(self.phase0_attempts.metrics())
        metrics.update(_masked_stats("reward/task", rollout["task_reward"], valid))
        metrics.update(_masked_stats("reward/amp_raw", rollout["amp_reward_raw"], amp_valid))
        metrics.update(_masked_stats("reward/amp_credit", rollout["amp_reward_credit"], amp_valid))
        metrics.update(_masked_stats("reward/mixed", rollout["mixed_reward"], valid))
        metrics.update(_masked_stats("credit/task_adv", rollout["channel_advantages"][..., 0], valid))
        metrics.update(_masked_stats("credit/amp_adv", rollout["channel_advantages"][..., 1], valid))
        weighted_channel_advantage = (
            float(self.cfg.credit.task_weight)
            * rollout["channel_advantages"][..., 0]
            + float(self.cfg.credit.amp_weight)
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
            _masked_stats(
                "credit/mixed_adv",
                rollout["mixed_advantage"],
                valid,
            )
        )
        metrics.update(
            _masked_stats(
                "credit/task_actor_component",
                rollout["actor_advantage_components"][..., 0],
                valid,
            )
        )
        metrics.update(
            _masked_stats(
                "credit/amp_actor_component",
                rollout["actor_advantage_components"][..., 1],
                valid,
            )
        )
        metrics.update(_masked_stats("credit/actor_adv", rollout["advantages"], valid))
        metrics.update(_masked_stats("critic/task_value", rollout["values"][..., 0], valid))
        metrics.update(_masked_stats("critic/amp_value", rollout["values"][..., 1], valid))
        metrics.update(_masked_stats("critic/task_target", rollout["value_targets"][..., 0], valid))
        metrics.update(_masked_stats("critic/amp_target", rollout["value_targets"][..., 1], valid))
        metrics.update(
            style_reward_statistics(
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
            metrics.update(
                _masked_stats(
                    f"credit/frame_{frame_idx}_task_actor_component",
                    rollout["actor_advantage_components"][..., frame_idx, 0],
                    frame_valid,
                )
            )
            metrics.update(
                _masked_stats(
                    f"credit/frame_{frame_idx}_amp_actor_component",
                    rollout["actor_advantage_components"][..., frame_idx, 1],
                    frame_valid,
                )
            )
        metrics.update(self.disc_normalizer.statistics())
        metrics.update(self.imitation_history.statistics())
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
            "credit=causal_frame critic=shared_dual_flow",
            flush=True,
        )
        print(
            f"[ARCH] task={self.env.task.name} actor_obs={self.actor_obs_dim} "
            f"env_actor_obs={self.base_actor_obs_dim} discriminator_in_actor=False "
            f"prefix_context={self.prefix_context_dim} action_dim={self.num_act} "
            f"H={self.horizon_h} W_D={self.imitation_history_steps} "
            f"imitation_frame={self.imitation_frame_dim} imitation_window={self.imitation_window_dim}",
            flush=True,
        )
        print(
            f"[CREDIT] critic_sharing=encoder heads=task,amp "
            f"credit={self.cfg.credit.mode} advantage_norm={self.cfg.credit.advantage_normalization} "
            f"ratio_mode={self.cfg.credit.ratio_mode} "
            f"task_weight={self.cfg.credit.task_weight} amp_weight={self.cfg.credit.amp_weight} "
            f"amp_dt={self.cfg.credit.integrate_amp_reward_dt}",
            flush=True,
        )
        print(
            f"[STYLE_PRIOR] discriminator=standard_mlp hidden={list(self.cfg.amp.hidden_dims)} "
            f"BCE=True GP={self.cfg.amp.grad_penalty} replay={self.cfg.amp.replay_size} "
            f"EMA=False policy_conditioning=False motion_end_terminal=True "
            f"replay_mode=frame_trajectory fcamp_schema=6 "
            f"phase0_trajectory_attempt_stream={self.cfg.streams.phase0_fraction:.2f}",
            flush=True,
        )
