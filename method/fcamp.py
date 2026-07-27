"""FC-AMP: Flow-chunk policy optimization with temporal discriminator prior.

The discriminator remains an independent MimicKit-style reward model. Its
online representation is never part of the actor or critic observation.
"""

from __future__ import annotations

import copy
import math
import time

import torch
from torch import nn

from components.credit.temporal_credit import (
    compute_dual_channel_gae,
    normalize_actor_mixture,
    resolve_terminal_masks,
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
)
from components.imitation.temporal_history import TemporalFeatureHistory
from components.imitation.window_pipeline import TemporalWindowPipeline
from components.normalization.running_stats import (
    EmpiricalNormalization,
    RunningNormalizer,
)
from components.replay.fcamp_window_buffer import FCAMPWindowReplay
from components.rollout.fcamp_diagnostics import (
    collect_update_metrics,
    log_banner,
    log_update,
)
from models.style_discriminator import (
    StyleDiscriminator,
    compute_style_discriminator_loss,
)
from models.dual_flow_critic import DualFlowCritic
from engine.checkpoint import (
    FCAMP_CHECKPOINT_CONTRACT,
    validate_static_fcamp_checkpoint_contract,
)


class FCAMP(FlowCPSBase):
    """One-shot H=4 innovation Flow-CPS with a temporal discriminator prior."""

    def build(self) -> None:
        cfg = self.cfg
        style_cfg = cfg.style_prior
        critic_cfg = cfg.critics
        env = self.env

        self.imitation_history_steps = int(style_cfg.obs_steps)
        super().build()

        # The environment owns the immutable normalized PD-command domain.
        self.action_low = env.action_low
        self.action_high = env.action_high

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

        # A primitive value is conditioned on the state that selected the
        # open-loop chunk and only the innovations already executed.  The
        # current innovation is deliberately excluded so this remains a valid
        # action-independent policy-gradient baseline.
        self.prefix_context_dim = (
            2 * self.critic_obs_dim
            + self.actor_obs_dim
            + self.chunk_dim
            + 2 * self.horizon_h
        )
        self.flow_critic_steps = int(cfg.flow_steps)
        self.flow_critic_samples = 4
        self.flow_critic_fm_samples = 1
        self.critic_learning_rate = float(cfg.value_lr)
        if self.critic_learning_rate <= 0.0:
            raise ValueError(f"value_lr must be > 0, got {cfg.value_lr}")
        self.critic = DualFlowCritic(
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
            self.imitation_window_dim, tuple(style_cfg.hidden_dims)
        ).to(env.device)
        self.disc_normalizer = RunningNormalizer(
            self.imitation_window_dim,
            device=env.device,
            clip=float(style_cfg.normalizer_clip),
        )
        self.imitation_history = TemporalFeatureHistory(
            env.num_envs,
            self.imitation_history_steps,
            self.imitation_frame_dim,
            device=env.device,
        )
        replay_capacity = int(style_cfg.replay_size)
        replay_replace = int(style_cfg.replay_samples)
        phase0_fraction = float(cfg.streams.phase0_fraction)
        phase0_capacity = int(round(replay_capacity * phase0_fraction))
        phase0_replace = int(round(replay_replace * phase0_fraction))
        self.replay_stream_capacities = {
            PHASE0_STREAM: phase0_capacity,
            CURRICULUM_STREAM: replay_capacity - phase0_capacity,
        }
        self.replay_replacement_quotas = {
            PHASE0_STREAM: phase0_replace,
            CURRICULUM_STREAM: replay_replace - phase0_replace,
        }
        current_capacity = int(style_cfg.current_buffer_size)
        current_phase0_capacity = int(round(current_capacity * phase0_fraction))
        self.current_stream_capacities = {
            PHASE0_STREAM: current_phase0_capacity,
            CURRICULUM_STREAM: current_capacity - current_phase0_capacity,
        }
        # One replay item is one lossless chronological raw [W,F] window.
        # Stream partitions make the configured 10/90 contract invariant to
        # episode survival and curriculum drift.
        self.disc_window_replay = FCAMPWindowReplay(
            self.replay_stream_capacities,
            self.imitation_history_steps,
            self.imitation_frame_dim,
            # 200k x 16 x 233 FP32 is ~2.78 GiB.  Page-locking that entire
            # resident store is unsafe and unnecessary; only sampled batches
            # cross to CUDA.
            pin_memory=False,
        )
        disc_parameters = [p for p in self.discriminator.parameters() if p.requires_grad]
        optimizer_name = style_cfg.optimizer.lower()
        optimizer_kwargs = {
            "lr": float(style_cfg.learning_rate),
            "weight_decay": float(style_cfg.weight_decay),
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
        self.warmup_env_transitions = 0
        self._has_chunk_predecessor = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

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
    def _validate_checkpoint_contract(self, state: dict) -> None:
        validate_static_fcamp_checkpoint_contract(state)
        self._validate_checkpoint_action_contract(state)

    def _validate_checkpoint_action_contract(self, state: dict) -> None:
        tensor_contract = (
            ("action_low", self.action_low),
            ("action_high", self.action_high),
        )
        for name, expected in tensor_contract:
            saved = state.get(name)
            if (
                not torch.is_tensor(saved)
                or saved.shape != expected.shape
                or not torch.equal(
                    saved.detach().to(device="cpu"),
                    expected.detach().to(device="cpu"),
                )
            ):
                raise ValueError(
                    f"FCAMP checkpoint {name} differs from the action domain"
                )
        scalar_contract = (
            ("innovation_step_bound", float(self.cfg.innovation_step_bound)),
            ("cps_raw_rms", float(self.cfg.cps_raw_rms)),
        )
        for name, expected in scalar_contract:
            saved = state.get(name)
            if type(saved) is not float or saved != expected:
                raise ValueError(
                    f"FCAMP checkpoint {name}={saved!r} differs from "
                    f"the action contract value {expected!r}"
                )

    def validate_checkpoint_payload(self, payload: dict) -> None:
        state = payload.get("algo_state") if isinstance(payload, dict) else None
        self._validate_checkpoint_contract(state)

    def extra_checkpoint_state(self) -> dict:
        payload = super().extra_checkpoint_state()
        payload.update(
            {
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "critic_learning_rate": float(self.critic_learning_rate),
                "disc_optimizer": self.disc_optimizer.state_dict(),
                "disc_version": int(self.disc_version),
                "warmup_env_transitions": int(self.warmup_env_transitions),
                "action_low": self.action_low.detach().cpu(),
                "action_high": self.action_high.detach().cpu(),
                "innovation_step_bound": float(
                    self.cfg.innovation_step_bound
                ),
                "cps_raw_rms": float(self.cfg.cps_raw_rms),
                "disc_window_replay": self.disc_window_replay.state_dict(),
                **FCAMP_CHECKPOINT_CONTRACT,
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
        if payload:
            self._validate_checkpoint_contract(payload)
        super().load_extra_checkpoint_state(payload, reset_optimizer=reset_optimizer)
        if not payload:
            return
        if reset_optimizer:
            self.critic_learning_rate = float(self.cfg.value_lr)
        else:
            self.critic_learning_rate = float(
                payload.get(
                    "critic_learning_rate",
                    self.critic_learning_rate,
                )
            )
            self.critic_optimizer.load_state_dict(
                payload["critic_optimizer"]
            )
        for group in self.critic_optimizer.param_groups:
            group["lr"] = self.critic_learning_rate
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
        self.warmup_env_transitions = int(
            payload.get("warmup_env_transitions", 0)
        )
        if self.warmup_env_transitions < 0:
            raise ValueError("FCAMP checkpoint has invalid warmup transition count")
        replay_state = payload.get("disc_window_replay")
        if replay_state is None:
            raise ValueError("FCAMP checkpoint is missing complete-window replay state")
        self.disc_window_replay.load_state_dict(replay_state)

    # ------------------------------------------------------------------ #
    # Primitive-step value contexts and style reward
    # ------------------------------------------------------------------ #
    def _prefix_context_raw(
        self,
        current_critic_obs: torch.Tensor,
        chunk_start_critic_obs: torch.Tensor,
        chunk_start_actor_obs: torch.Tensor,
        raw_innovation: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        batch = current_critic_obs.shape[0]
        innovations = raw_innovation.reshape(
            batch,
            self.horizon_h,
            self.num_act,
        )
        prefix = torch.zeros_like(innovations)
        if offset:
            prefix[:, :offset] = innovations[:, :offset]
        offset_onehot = torch.zeros(
            batch,
            self.horizon_h,
            device=current_critic_obs.device,
            dtype=current_critic_obs.dtype,
        )
        offset_onehot[:, int(offset)] = 1.0
        prefix_mask = torch.zeros_like(offset_onehot)
        if offset:
            prefix_mask[:, :offset] = 1.0
        context = torch.cat(
            [
                current_critic_obs,
                chunk_start_critic_obs,
                chunk_start_actor_obs,
                prefix.flatten(start_dim=1),
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
        *,
        stream_labels: torch.Tensor,
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
        sample_count = int(selected_indices.numel())
        for start in range(0, sample_count, batch_size):
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
        batch_size = max(1, int(self.cfg.style_prior.reward_eval_batch_size))
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
            scale=float(self.cfg.style_prior.reward_scale),
            minimum_one_minus_prob=float(self.cfg.style_prior.reward_epsilon),
        )
        return all_logits, rewards

    def _reset_imitation_history(
        self,
        phase_indices: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        phases = phase_indices.to(device=self.env.device)
        seed = self.env.motion.get_fcamp_demo_history(
            phases,
            self.imitation_history_steps,
            flatten=False,
        )
        self.imitation_history.reset_seeded(seed, env_ids=env_ids)

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
            root_velocity_frame="link",
        )
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
            env.set_episode_age(curriculum_ids, random_age)
        self._obs = obs
        self._critic_obs = env.get_critic_observation()
        self._has_chunk_predecessor.zero_()
        self._reset_imitation_history(phases)
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

    def pre_training_warmup(
        self,
        current_observation: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], int]:
        """Fit immutable actor normalization and prime D on one discarded rollout.

        The complete-window replay is retained because those interactions count
        toward the training budget; adaptive curriculum evidence and episode
        accounting are restored so the discarded rollout cannot steer update 1.
        """

        sampler_state = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in self.env.adaptive_sampler.state_dict().items()
        }
        failure_recorded = self.env._failure_recorded.clone()
        self._fcamp_update_idx = 0
        rollout = self.collect(current_observation, fit_normalizers=True)
        disc_metrics = self._discriminator_update(
            0,
            rollout=rollout,
            commit_normalizer_before_training=True,
        )
        if disc_metrics["disc/update_steps"] <= 0.0:
            raise RuntimeError("FCAMP discriminator warm-up produced no optimizer step")
        if disc_metrics["disc_norm/committed_before_training"] != 1.0:
            raise RuntimeError(
                "FCAMP discriminator warm-up did not commit matched normalization first"
            )

        if not self.env.adaptive_sampler.load_state_dict(sampler_state):
            raise RuntimeError("FCAMP could not restore sampler after discarded warm-up")
        self.env._failure_recorded.copy_(failure_recorded)
        self.phase0_attempts = Phase0AttemptTracker(
            self.training_streams.stream_ids
        )
        self._init_train_episode_stats()
        next_observation = self._reset_training_streams(
            randomize_curriculum_episode_age=True,
        )
        transitions = int(self.env.num_envs) * int(self.cfg.rollout_env_steps)
        self.warmup_env_transitions = transitions
        metrics = {
            **disc_metrics,
            "discarded_rollouts": 1.0,
            "actor_optimizer_steps": 0.0,
            "critic_optimizer_steps": 0.0,
            "actor_obs_normalizer_count": float(
                self.actor_obs_normalizer.count.item()
            ),
            "prefix_context_normalizer_count": float(
                self.prefix_context_normalizer.count.item()
            ),
            "env_transitions": float(transitions),
            "amp_valid_window_count": float(rollout["amp_valid"].sum().item()),
            "replay_dirty_insert_count": float(
                self.disc_window_replay.statistics(
                    current_update=0
                )["replay/dirty_insert_count"]
            ),
        }
        del rollout
        return next_observation, metrics, transitions

    def deployment_chunk(self, obs: torch.Tensor) -> torch.Tensor:
        """Return four deterministic bounded absolute PD actions."""

        if obs.shape[-1] != self.base_actor_obs_dim:
            raise ValueError(
                f"FCAMP expected raw actor observation dim {self.base_actor_obs_dim}, "
                f"got {tuple(obs.shape)}"
            )
        observed_last_action = obs[..., -self.num_act :]
        anchor_error = (
            observed_last_action - self.env.last_action
        ).abs().max()
        if float(anchor_error.item()) > 1.0e-6:
            raise RuntimeError(
                "FCAMP deployment observation does not contain the exact "
                "environment-owned last_action"
            )
        actor_obs = self._norm_actor(obs)
        mean_innovation = self._flow_mean_innovation(actor_obs)
        return self._action_chunk_from_innovations(
            mean_innovation,
            observed_last_action,
        )

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the four-frame absolute action plan."""
        return self.deployment_chunk(obs)

    def evaluation_step_payload(
        self,
        frame_payload: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        return self.env.step(frame_payload, active_mask=active_mask)

    def evaluation_reset(self, phase_indices: torch.Tensor) -> torch.Tensor:
        """Reset FCAMP evaluation in the same root-link domain as training."""

        return self.env.reset(
            phase_indices=phase_indices,
            root_velocity_frame="link",
        )

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        # Trainer owns the canonical update clock, including after resume.
        self._fcamp_update_idx = int(update_idx)
        self.phase0_attempts.begin_update()
        return super().reset_for_update(update_idx)

    # ------------------------------------------------------------------ #
    # Rollout and causal credit
    # ------------------------------------------------------------------ #
    def collect(
        self,
        current_obs: torch.Tensor,
        *,
        fit_normalizers: bool = False,
    ) -> dict:
        env = self.env
        device = env.device
        n_envs = env.num_envs
        chunks = self._chunks_per_update()
        h = self.horizon_h
        # One rollout is generated, history-conditioned and rewarded by one
        # immutable committed discriminator snapshot.  As in the local
        # MimicKit AMP implementation, policy/value optimization consumes these
        # rewards before the discriminator or its normalizer is advanced.
        self.discriminator.eval()
        self.disc_normalizer.freeze()
        rollout_disc_version = int(self.disc_version)
        rollout_disc_normalizer_count = float(self.disc_normalizer.count.item())

        actor_obs_buf = torch.zeros(chunks, n_envs, self.actor_obs_dim, device=device)
        raw_innovation_buf = torch.zeros(
            chunks, n_envs, self.chunk_dim, device=device
        )
        old_mean_innovation_buf = torch.zeros_like(raw_innovation_buf)
        old_log_probs_buf = torch.zeros(chunks, n_envs, h, device=device)
        old_joint_cholesky = self._effective_joint_cholesky(
            device=device,
            dtype=raw_innovation_buf.dtype,
        )
        old_joint_cholesky = old_joint_cholesky.detach().clone()
        context_buf = torch.zeros(
            chunks, n_envs, h, self.prefix_context_dim, device=device
        )
        value_buf = torch.zeros(chunks, n_envs, h, 2, device=device)
        next_value_buf = torch.zeros_like(value_buf)
        task_reward_buf = torch.zeros(chunks, n_envs, h, device=device)
        amp_reward_raw_buf = torch.zeros_like(task_reward_buf)
        amp_reward_credit_buf = torch.zeros_like(task_reward_buf)
        mixed_reward_buf = torch.zeros_like(task_reward_buf)
        amp_logit_buf = torch.zeros_like(task_reward_buf)
        valid_buf = torch.zeros(chunks, n_envs, h, dtype=torch.bool, device=device)
        amp_valid_buf = torch.zeros_like(valid_buf)
        intervention_edge_buf = torch.zeros_like(valid_buf)
        amp_bootstrap_buf = torch.zeros_like(valid_buf)
        amp_trace_buf = torch.zeros_like(valid_buf)
        amp_age_buf = torch.full(
            (chunks, n_envs, h), -1, dtype=torch.long, device=device
        )
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
        task_weight = float(self.cfg.credit.task_weight)
        amp_weight = float(self.cfg.credit.amp_weight)
        amp_dt_scale = float(env.dt)
        raw_innovation_abs_max = 0.0
        action_delta_abs_max = 0.0
        action_contract_error_max = 0.0
        tanh_saturated_count = 0
        raw_innovation_active_count = 0
        continuation_chunk = self._has_chunk_predecessor.clone()
        seam_square_sums = torch.zeros(2, 3, device=device)
        seam_component_counts = torch.zeros(2, device=device)
        reset_square_sums = torch.zeros(2, device=device)
        reset_component_count = torch.zeros((), device=device)
        fk_alignment_abs_max = 0.0
        geometry_checked = False
        dirty_window_excluded_count = 0
        window_latest_root_xy_abs_max = torch.zeros((), device=device)
        window_root_xy_abs_sum = torch.zeros((), device=device)
        window_root_xy_count = 0
        replay_update_idx = int(self._fcamp_update_idx)
        self.disc_window_replay.begin_update(
            replay_update_idx,
            replacement_quotas=self.replay_replacement_quotas,
        )
        # Current-D data uses the same whole-rollout reservoir semantics.  A
        # per-primitive 10/90 rounding would make short-lived streams or small
        # diagnostics silently disappear and biases selection by step order.
        current_window_buffer = FCAMPWindowReplay(
            self.current_stream_capacities,
            self.imitation_history_steps,
            self.imitation_frame_dim,
            pin_memory=False,
        )
        current_window_buffer.begin_update(
            0,
            replacement_quotas={PHASE0_STREAM: 0, CURRICULUM_STREAM: 0},
        )
        with torch.no_grad():
            for chunk_idx in range(chunks):
                chunk_actor_raw = obs.clone()
                chunk_critic_raw = critic_obs.clone()
                actor_obs_n = self._norm_actor(chunk_actor_raw)
                previous_action = env.last_action.detach().clone()
                previous_delta = env.last_delta.detach().clone()
                observed_previous_action = chunk_actor_raw[..., -self.num_act :]
                previous_action_error = (
                    observed_previous_action - previous_action
                ).abs().max()
                if float(previous_action_error.item()) > 1.0e-6:
                    raise RuntimeError(
                        "FCAMP actor observation does not contain the exact "
                        "environment-owned last_action"
                    )
                raw_innovation, old_mean_innovation, old_log_probs = self._sample_innovations(
                    actor_obs_n
                )
                raw_innovation_chunk = raw_innovation.view(n_envs, h, self.num_act)
                action_chunk = self._action_chunk_from_innovations(
                    raw_innovation,
                    previous_action,
                )

                actor_obs_buf[chunk_idx] = actor_obs_n
                raw_innovation_buf[chunk_idx] = raw_innovation
                old_mean_innovation_buf[chunk_idx] = old_mean_innovation
                old_log_probs_buf[chunk_idx] = old_log_probs
                if fit_normalizers:
                    self._update_empirical_normalizer_chunked(
                        self.actor_obs_normalizer,
                        chunk_actor_raw,
                        stream_labels=self.training_streams.stream_ids,
                    )

                alive = torch.ones(n_envs, dtype=torch.bool, device=device)
                for frame_idx in range(h):
                    alive_before = alive.clone()
                    current_context_raw = self._prefix_context_raw(
                        critic_obs,
                        chunk_critic_raw,
                        chunk_actor_raw,
                        raw_innovation,
                        frame_idx,
                    )
                    raw_innovation_t = raw_innovation_chunk[:, frame_idx]
                    planned_action = action_chunk[:, frame_idx]
                    executed_action = torch.where(
                        alive_before.unsqueeze(-1),
                        planned_action,
                        env.last_action,
                    )
                    next_obs, task_reward, done, info = env.step(
                        executed_action,
                        active_mask=alive_before,
                    )
                    next_critic_obs = env.get_critic_observation()
                    applied_action = info["applied_action"]
                    applied_delta = info["applied_delta"]
                    if not torch.is_tensor(applied_action) or not torch.is_tensor(
                        applied_delta
                    ):
                        raise RuntimeError(
                            "FCAMP environment did not report applied action/delta"
                        )
                    if bool(alive_before.any()):
                        active_raw_innovation = raw_innovation_t[alive_before]
                        active_action = applied_action[alive_before]
                        active_delta = applied_delta[alive_before]
                        action_d2 = applied_delta - previous_delta
                        active_d2 = action_d2[alive_before]
                        raw_innovation_abs_max = max(
                            raw_innovation_abs_max,
                            float(active_raw_innovation.abs().max().item()),
                        )
                        if frame_idx == 0:
                            continuation_boundary = alive_before & continuation_chunk
                            reset_boundary = alive_before & ~continuation_chunk
                            if bool(continuation_boundary.any()):
                                seam_square_sums[0] += torch.stack(
                                    (
                                        raw_innovation_t[
                                            continuation_boundary
                                        ].square().sum(),
                                        applied_delta[
                                            continuation_boundary
                                        ].square().sum(),
                                        action_d2[
                                            continuation_boundary
                                        ].square().sum(),
                                    )
                                )
                                seam_component_counts[0] += (
                                    continuation_boundary.sum() * self.num_act
                                )
                            if bool(reset_boundary.any()):
                                reset_square_sums += torch.stack(
                                    (
                                        applied_delta[
                                            reset_boundary
                                        ].square().sum(),
                                        action_d2[
                                            reset_boundary
                                        ].square().sum(),
                                    )
                                )
                                reset_component_count += (
                                    reset_boundary.sum() * self.num_act
                                )
                        else:
                            seam_square_sums[1] += torch.stack(
                                (
                                    active_raw_innovation.square().sum(),
                                    active_delta.square().sum(),
                                    active_d2.square().sum(),
                                )
                            )
                            seam_component_counts[1] += (
                                alive_before.sum() * self.num_act
                            )
                        tanh_saturated_count += int(
                            (torch.tanh(active_raw_innovation).abs() >= 0.99).sum().item()
                        )
                        raw_innovation_active_count += int(active_raw_innovation.numel())
                        action_contract_error_max = max(
                            action_contract_error_max,
                            float(
                                (
                                    planned_action[alive_before]
                                    - active_action
                                )
                                .abs()
                                .max()
                                .item()
                            ),
                            float(
                                (
                                    active_delta.abs()
                                    - float(self.cfg.innovation_step_bound)
                                )
                                .clamp_min(0.0)
                                .max()
                                .item()
                            ),
                        )
                        action_delta_abs_max = max(
                            action_delta_abs_max,
                            float(active_delta.abs().max().item()),
                        )
                    previous_delta = torch.where(
                        alive_before.unsqueeze(-1),
                        applied_delta,
                        previous_delta,
                    )

                    # Reset installs a complete phase-matched demo predecessor
                    # history.  This first post-action frame immediately forms
                    # the same fixed-W endpoint window used everywhere else.
                    imitation_frame = info["imitation_frame"]
                    alive_ids = alive_before.nonzero(as_tuple=False).squeeze(-1)
                    intervention_edges = info["intervention_edge_mask"]
                    if not torch.is_tensor(intervention_edges):
                        raise RuntimeError(
                            "FCAMP environment did not report intervention_edge_mask"
                        )
                    intervention_edges = intervention_edges.bool()
                    intervention_edge_buf[chunk_idx, :, frame_idx] = (
                        intervention_edges & alive_before
                    )
                    if alive_ids.numel() > 0:
                        self.imitation_history.push(
                            imitation_frame.index_select(0, alive_ids),
                            alive_ids,
                            intervention_after=intervention_edges.index_select(
                                0, alive_ids
                            ),
                        )
                    if not geometry_checked:
                        # ``imitation_frame`` was captured before an interval
                        # push while the returned simulator state is post-push.
                        # Compare only untouched states so this is an exact
                        # same-state geometry/feature-domain invariant.
                        alignment_ids = (
                            alive_before & ~intervention_edges
                        ).nonzero(as_tuple=False).squeeze(-1)
                        if alignment_ids.numel() > 0:
                            aligned_frame = env.get_fcamp_fk_aligned_policy_frame(
                                alignment_ids
                            )
                            alignment_error = (
                                imitation_frame.index_select(0, alignment_ids)
                                - aligned_frame
                            ).abs()
                            fk_alignment_abs_max = float(
                                alignment_error.max().item()
                            )
                            geometry_checked = True
                            if fk_alignment_abs_max > 1.0e-4:
                                raise RuntimeError(
                                    "FCAMP expert/runtime 233-D geometry contract failed: "
                                    f"max_abs={fk_alignment_abs_max:.6g}"
                                )
                    active_float = alive_before.to(dtype=task_reward.dtype)
                    amp_reward_raw = torch.zeros_like(task_reward)
                    amp_logits = torch.zeros_like(task_reward)
                    dirty_ready = (
                        self.imitation_history.ready
                        & ~self.imitation_history.causal_ready
                        & alive_before
                    )
                    dirty_window_excluded_count += int(dirty_ready.sum().item())
                    ready_mask = self.imitation_history.causal_ready & alive_before
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
                        ready_streams = self.training_streams.stream_ids.index_select(
                            0, ready_ids
                        )
                        ready_end_times = info[
                            "imitation_frame_phase_steps"
                        ].index_select(0, ready_ids)
                        rounded_end_times = ready_end_times.float().round().to(
                            dtype=torch.long
                        )
                        if not bool(
                            torch.allclose(
                                ready_end_times.to(dtype=torch.float32),
                                rounded_end_times.float(),
                                rtol=0.0,
                                atol=1.0e-4,
                            )
                        ):
                            raise RuntimeError(
                                "FCAMP complete-window replay requires integer reference endpoints"
                            )
                        for stream_id in (PHASE0_STREAM, CURRICULUM_STREAM):
                            local = ready_streams == stream_id
                            if bool(local.any()):
                                self.disc_window_replay.offer(
                                    raw_window_frames[local],
                                    end_times=rounded_end_times[local],
                                    stream_id=stream_id,
                                    dirty=torch.zeros(
                                        int(local.sum().item()),
                                        dtype=torch.bool,
                                        device=device,
                                    ),
                                )
                                current_window_buffer.offer(
                                    raw_window_frames[local],
                                    end_times=rounded_end_times[local],
                                    stream_id=stream_id,
                                    dirty=torch.zeros(
                                        int(local.sum().item()),
                                        dtype=torch.bool,
                                        device=device,
                                    ),
                                )
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
                    motion_complete = info["done_terms"]["motion_complete"].bool()
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
                    termination_phases = info["termination_phase_steps"]
                    if bool(new_done.any()):
                        terminal_phase_buf[chunk_idx, new_done, frame_idx] = (
                            termination_phases[new_done].to(dtype=torch.float32)
                        )
                    # Timeout bootstraps from the terminal observation but never
                    # connects its GAE trace to the reset state.
                    bootstrap_buf[chunk_idx, :, frame_idx] = (
                        alive_before & ~new_failure & ~new_motion_complete
                    )
                    trace_buf[chunk_idx, :, frame_idx] = alive_before & ~new_done
                    # An interval push occurs after this endpoint was captured.
                    # Task learning retains the domain-randomized transition;
                    # style credit cannot bootstrap or trace across that
                    # exogenous edge.
                    amp_bootstrap_buf[chunk_idx, :, frame_idx] = (
                        bootstrap_buf[chunk_idx, :, frame_idx]
                        & ~intervention_edges
                    )
                    amp_trace_buf[chunk_idx, :, frame_idx] = (
                        trace_buf[chunk_idx, :, frame_idx]
                        & ~intervention_edges
                    )

                    if frame_idx < h - 1:
                        next_context_raw = self._prefix_context_raw(
                            next_critic_obs,
                            chunk_critic_raw,
                            chunk_actor_raw,
                            raw_innovation,
                            frame_idx + 1,
                        )
                    else:
                        next_context_raw = self._prefix_context_raw(
                            next_critic_obs,
                            next_critic_obs,
                            next_obs,
                            raw_innovation,
                            0,
                        )
                    current_context = self.prefix_context_normalizer(
                        current_context_raw, update=False
                    )
                    next_context = self.prefix_context_normalizer(
                        next_context_raw, update=False
                    )
                    context_buf[chunk_idx, :, frame_idx] = current_context
                    pair_values = self._evaluate_prefix_values(
                        torch.stack((current_context, next_context), dim=1)
                    )
                    value_buf[chunk_idx, :, frame_idx] = pair_values[:, 0]
                    next_value_buf[chunk_idx, :, frame_idx] = pair_values[:, 1]
                    if fit_normalizers:
                        pair_valid = alive_before.unsqueeze(-1).expand(-1, 2)
                        pair_streams = self.training_streams.stream_ids.unsqueeze(
                            -1
                        ).expand(-1, 2)
                        self._update_empirical_normalizer_chunked(
                            self.prefix_context_normalizer,
                            torch.stack(
                                (current_context_raw, next_context_raw),
                                dim=1,
                            ),
                            pair_valid,
                            stream_labels=pair_streams,
                        )

                    self._record_train_episode_stats(
                        mixed_reward,
                        new_done,
                        step_counts=active_float,
                    )
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
                        root_velocity_frame="link",
                    )
                    obs[reset_ids] = reset_obs
                    critic_obs = env.get_critic_observation()
                    self._reset_imitation_history(
                        phase_indices=reset_phases,
                        env_ids=reset_ids,
                    )
                    self.phase0_attempts.start(reset_ids)
                continuation_chunk = ~chunk_done

        if not geometry_checked:
            self.disc_window_replay.abort_update()
            current_window_buffer.abort_update()
            raise RuntimeError("FCAMP could not perform a same-state FK alignment check")
        if bool((intervention_edge_buf & done_buf).any()):
            self.disc_window_replay.abort_update()
            current_window_buffer.abort_update()
            raise RuntimeError("terminal FCAMP transitions must never receive an interval push")
        if action_contract_error_max > 1.0e-6:
            self.disc_window_replay.abort_update()
            current_window_buffer.abort_update()
            raise RuntimeError(
                "FCAMP planned/applied innovation contract was violated"
            )
        self.disc_window_replay.commit_update()
        current_window_buffer.commit_update()
        current_disc_windows: list[torch.Tensor] = []
        current_disc_end_times: list[torch.Tensor] = []
        current_disc_stream_ids: list[torch.Tensor] = []
        for stream_id in (PHASE0_STREAM, CURRICULUM_STREAM):
            count = current_window_buffer.size(stream_id)
            if count <= 0:
                raise RuntimeError(
                    f"FCAMP current-D stream {stream_id} has no clean complete window"
                )
            raw, ends = current_window_buffer.sample(
                count,
                stream_id=stream_id,
                replacement=False,
            )
            current_disc_windows.append(self.imitation_pipeline.flatten(raw))
            current_disc_end_times.append(ends)
            current_disc_stream_ids.append(
                torch.full((count,), stream_id, dtype=torch.int8)
            )
        self._obs = obs
        self._critic_obs = critic_obs
        self._has_chunk_predecessor.copy_(continuation_chunk)
        seam_rms = torch.sqrt(
            seam_square_sums
            / seam_component_counts.clamp_min(1.0).unsqueeze(-1)
        )
        seam_ratio = torch.where(
            seam_rms[1] > 1.0e-12,
            seam_rms[0] / seam_rms[1],
            torch.full_like(seam_rms[0], -1.0),
        )
        reset_rms = torch.sqrt(
            reset_square_sums / reset_component_count.clamp_min(1.0)
        )
        rollout = {
            "actor_obs": actor_obs_buf,
            "raw_innovation": raw_innovation_buf,
            "old_mean_innovation": old_mean_innovation_buf,
            "old_joint_cholesky": old_joint_cholesky,
            "old_log_probs": old_log_probs_buf,
            "contexts": context_buf,
            "values": value_buf,
            "next_values": next_value_buf,
            "valid": valid_buf,
            "amp_valid": amp_valid_buf,
            "intervention_edge": intervention_edge_buf,
            "current_disc_windows": (
                torch.cat(current_disc_windows, dim=0)
            ),
            "current_disc_end_times": (
                torch.cat(current_disc_end_times, dim=0)
            ),
            "current_disc_stream_ids": (
                torch.cat(current_disc_stream_ids, dim=0)
            ),
            "imitation_window_age": amp_age_buf,
            "done": done_buf,
            "failure": failure_buf,
            "timeout": timeout_buf,
            "motion_complete": motion_complete_buf,
            "terminal_phase": terminal_phase_buf,
            "bootstrap_mask": bootstrap_buf,
            "trace_mask": trace_buf,
            "amp_bootstrap_mask": amp_bootstrap_buf,
            "amp_trace_mask": amp_trace_buf,
            "task_reward": task_reward_buf,
            "amp_reward_raw": amp_reward_raw_buf,
            "amp_reward_credit": amp_reward_credit_buf,
            "mixed_reward": mixed_reward_buf,
            "amp_logits": amp_logit_buf,
            "disc_version_used": rollout_disc_version,
            "disc_normalizer_count_used": rollout_disc_normalizer_count,
            "stream_ids": self.training_streams.stream_ids,
            "collection_start_phases": collection_start_phases,
            "raw_innovation_abs_max": raw_innovation_abs_max,
            "raw_innovation_tanh_saturation_fraction": float(
                tanh_saturated_count / max(raw_innovation_active_count, 1)
            ),
            "action_delta_abs_max": action_delta_abs_max,
            "raw_innovation_boundary_internal_rms_ratio": float(
                seam_ratio[0].item()
            ),
            "action_delta_boundary_internal_rms_ratio": float(
                seam_ratio[1].item()
            ),
            "action_d2_boundary_internal_rms_ratio": float(
                seam_ratio[2].item()
            ),
            "continuation_boundary_count": float(
                (seam_component_counts[0] / self.num_act).item()
            ),
            "reset_boundary_count": float(
                (reset_component_count / self.num_act).item()
            ),
            "reset_action_delta_rms": float(reset_rms[0].item()),
            "reset_action_d2_rms": float(reset_rms[1].item()),
            "fk_alignment_abs_max": fk_alignment_abs_max,
            "dirty_window_excluded_count": float(dirty_window_excluded_count),
            "window_latest_root_xy_abs_max": float(
                window_latest_root_xy_abs_max.item()
            ),
            "window_root_xy_abs_mean": float(
                (window_root_xy_abs_sum / max(window_root_xy_count, 1)).item()
            ),
            "next_observation": obs,
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
        amp_valid_time = chronological(rollout["amp_valid"])
        channel_valid_time = torch.stack(
            (valid_time, amp_valid_time), dim=-1
        )
        channel_bootstrap_time = torch.stack(
            (
                bootstrap_time,
                chronological(rollout["amp_bootstrap_mask"]),
            ),
            dim=-1,
        )
        channel_trace_time = torch.stack(
            (
                trace_time,
                chronological(rollout["amp_trace_mask"]),
            ),
            dim=-1,
        )
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
            channel_valid_mask=channel_valid_time,
            channel_bootstrap_mask=channel_bootstrap_time,
            channel_trace_mask=channel_trace_time,
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
        rollout["channel_valid"] = chunk_layout(credit.channel_valid_mask)

    # ------------------------------------------------------------------ #
    # Optimizers
    # ------------------------------------------------------------------ #
    def _stream_specs(
        self,
        labels: torch.Tensor,
    ) -> list[tuple[str, int, float, torch.Tensor]]:
        phase0 = (labels == PHASE0_STREAM).nonzero(
            as_tuple=False
        ).squeeze(-1)
        curriculum = (labels == CURRICULUM_STREAM).nonzero(
            as_tuple=False
        ).squeeze(-1)
        if phase0.numel() == 0 or curriculum.numel() == 0:
            raise RuntimeError("FCAMP requires both training streams")
        phase0_weight = float(self.cfg.streams.phase0_fraction)
        return [
            ("phase0", PHASE0_STREAM, phase0_weight, phase0),
            (
                "curriculum",
                CURRICULUM_STREAM,
                1.0 - phase0_weight,
                curriculum,
            ),
        ]

    def _set_actor_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate = float(learning_rate)
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.learning_rate

    @torch.no_grad()
    def _exact_rollout_frame_kl(
        self,
        actor_obs: torch.Tensor,
        old_mean_innovation: torch.Tensor,
        old_joint_cholesky: torch.Tensor,
        valid: torch.Tensor,
        micro_batch_size: int,
    ) -> torch.Tensor:
        """Evaluate exact old-rollout||current-policy frame KL."""
        new_joint_cholesky = self._effective_joint_cholesky(
            device=actor_obs.device,
            dtype=actor_obs.dtype,
        )
        sums = torch.zeros(
            self.horizon_h,
            device=actor_obs.device,
            dtype=torch.float64,
        )
        counts = valid.sum(dim=0).to(dtype=torch.float64)
        if bool((counts <= 0).any()):
            raise RuntimeError(
                "FCAMP KL acceptance minibatch has a frame with no valid sample"
            )
        for start in range(0, actor_obs.shape[0], micro_batch_size):
            stop = min(start + micro_batch_size, actor_obs.shape[0])
            new_mean_z = self._flow_mean_innovation(actor_obs[start:stop])
            frame_kl = self._expected_innovation_frame_kl(
                old_mean_innovation[start:stop],
                new_mean_z,
                old_joint_cholesky,
                new_joint_cholesky,
            )
            sums += (
                frame_kl.to(dtype=torch.float64)
                * valid[start:stop].to(dtype=torch.float64)
            ).sum(dim=0)
        return (sums / counts).to(dtype=actor_obs.dtype)

    def _atomic_actor_step(
        self,
        *,
        actor_obs: torch.Tensor,
        old_mean_innovation: torch.Tensor,
        old_joint_cholesky: torch.Tensor,
        valid: torch.Tensor,
        micro_batch_size: int,
    ) -> tuple[bool, torch.Tensor, int]:
        """Commit one Adam step only if every raw-innovation frame KL is safe."""
        policy_state = copy.deepcopy(self._policy.state_dict())
        optimizer_state = copy.deepcopy(self.actor_optimizer.state_dict())
        max_attempts = 3
        rejected_attempts = 0
        limit = (
            float(self.cfg.kl_acceptance_factor)
            * float(self.cfg.desired_kl)
        )
        attempted_lr = float(self.learning_rate)
        last_frame_kl = torch.full(
            (self.horizon_h,),
            float("inf"),
            device=actor_obs.device,
            dtype=actor_obs.dtype,
        )

        for attempt in range(max_attempts):
            self._set_actor_learning_rate(attempted_lr)
            self.actor_optimizer.step()
            last_frame_kl = self._exact_rollout_frame_kl(
                actor_obs,
                old_mean_innovation,
                old_joint_cholesky,
                valid,
                micro_batch_size,
            )
            finite = bool(torch.isfinite(last_frame_kl).all())
            within_limit = (
                finite
                and float(last_frame_kl.max().item()) <= limit
            )
            if within_limit:
                return True, last_frame_kl, rejected_attempts

            rejected_attempts += 1
            self._policy.load_state_dict(policy_state)
            # Optimizer.load_state_dict may retain references to the supplied
            # state tensors.  A fresh deep copy is required on every retry;
            # otherwise the next tentative Adam step mutates the rollback
            # snapshot itself (including moments and the step counter).
            self.actor_optimizer.load_state_dict(
                copy.deepcopy(optimizer_state)
            )
            reduced_lr = max(float(self.min_lr), attempted_lr * 0.5)
            self._set_actor_learning_rate(reduced_lr)
            if attempt + 1 >= max_attempts or reduced_lr >= attempted_lr:
                break
            attempted_lr = reduced_lr

        return False, last_frame_kl, rejected_attempts

    def _actor_update(self, rollout: dict) -> dict[str, float]:
        chunks, n_envs = rollout["raw_innovation"].shape[:2]
        batch_size = chunks * n_envs
        h = self.horizon_h
        actor_obs = rollout["actor_obs"].reshape(batch_size, self.actor_obs_dim)
        raw_innovation = rollout["raw_innovation"].reshape(batch_size, self.chunk_dim)
        old_mean_innovation = rollout["old_mean_innovation"].reshape(
            batch_size,
            self.chunk_dim,
        )
        old_joint_cholesky = rollout["old_joint_cholesky"]
        old_log_probs = rollout["old_log_probs"].reshape(batch_size, h)
        advantages = rollout["advantages"].reshape(batch_size, h)
        valid = rollout["valid"].reshape(batch_size, h)
        stream_labels = rollout["stream_ids"].reshape(1, n_envs).expand(
            chunks, n_envs
        ).reshape(-1)
        stream_specs = self._stream_specs(stream_labels)
        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        micro_batch_size = self._policy_micro_batch_size(
            self._policy_mini_batch_size(batch_size)
        )
        clip_low = 1.0 - float(self.cfg.clip_range)
        clip_high = 1.0 + float(self.cfg.clip_range)
        totals = {
            "policy_loss": 0.0,
            "exact_kl": 0.0,
            "ratio": 0.0,
            "clip": 0.0,
            "grad_norm": 0.0,
        }
        steps = 0
        rejected_attempts = 0
        skipped_steps = 0
        exact_frame_kl_max = 0.0

        for epoch in range(int(self.cfg.policy_epochs)):
            stream_splits: dict[str, tuple[float, tuple[torch.Tensor, ...]]] = {}
            for name, _, objective_weight, indices in stream_specs:
                shuffled = indices.index_select(
                    0,
                    torch.randperm(indices.numel(), device=actor_obs.device),
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
                combined_metrics = {
                    "policy_loss": 0.0,
                    "ratio": 0.0,
                    "clip": 0.0,
                }
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
                        "ratio": 0.0,
                        "clip": 0.0,
                    }
                    for micro_start in range(
                        0, idx.numel(), micro_batch_size
                    ):
                        sub = idx[
                            micro_start : micro_start + micro_batch_size
                        ]
                        new_log_probs = self._recompute_innovation_log_prob(
                            actor_obs[sub],
                            raw_innovation[sub],
                        )
                        delta = new_log_probs - old_log_probs[sub]
                        adv = advantages[sub]
                        mask = valid[sub].to(dtype=delta.dtype)
                        # Each frame owns one raw innovation factor. PPO acts
                        # on those factors; the bounded action recurrence is a
                        # fixed deterministic environment transform.
                        log_ratio = delta
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
                            clipped_flag = (
                                (ratio < clip_low) | (ratio > clip_high)
                            ).to(ratio.dtype)
                            stream_sums["policy_loss"] += float(
                                policy_sum.item()
                            )
                            stream_sums["ratio"] += float(
                                (ratio * mask).sum().item()
                            )
                            stream_sums["clip"] += float(
                                (clipped_flag * mask).sum().item()
                            )
                    stream_means = {
                        key: value / valid_denominator
                        for key, value in stream_sums.items()
                    }
                    for key, value in stream_means.items():
                        combined_metrics[key] += objective_weight * value

                grad_norm = nn.utils.clip_grad_norm_(
                    self._policy.parameters(), float(self.cfg.max_grad_norm)
                )
                acceptance_indices = torch.cat(
                    [idx for _, _, idx in parts],
                    dim=0,
                )
                accepted, exact_frame_kl, rejected = self._atomic_actor_step(
                    actor_obs=actor_obs.index_select(
                        0,
                        acceptance_indices,
                    ),
                    old_mean_innovation=old_mean_innovation.index_select(
                        0,
                        acceptance_indices,
                    ),
                    old_joint_cholesky=old_joint_cholesky,
                    valid=valid.index_select(
                        0,
                        acceptance_indices,
                    ),
                    micro_batch_size=micro_batch_size,
                )
                rejected_attempts += rejected
                if not accepted:
                    skipped_steps += 1
                    continue
                for key in ("policy_loss", "ratio", "clip"):
                    totals[key] += combined_metrics[key]
                totals["exact_kl"] += float(
                    exact_frame_kl.mean().item()
                )
                totals["grad_norm"] += float(grad_norm)
                exact_frame_kl_max = max(
                    exact_frame_kl_max,
                    float(exact_frame_kl.max().item()),
                )
                steps += 1

        denom = max(steps, 1)
        metrics = {
            "fcamp/policy_loss": totals["policy_loss"] / denom,
            "fcamp/kl": totals["exact_kl"] / denom,
            "fcamp/ratio": totals["ratio"] / denom,
            "fcamp/clip_fraction": totals["clip"] / denom,
            "fcamp/actor_grad_norm": totals["grad_norm"] / denom,
            "fcamp/actor_lr": float(self.learning_rate),
            "fcamp/actor_optimizer_steps": float(steps),
            "fcamp/actor_rejected_attempts": float(rejected_attempts),
            "fcamp/actor_skipped_steps": float(skipped_steps),
            "fcamp/exact_frame_kl_max": float(exact_frame_kl_max),
            "fcamp/kl_acceptance_limit": (
                float(self.cfg.kl_acceptance_factor)
                * float(self.cfg.desired_kl)
            ),
        }
        for name, _, objective_weight, _ in stream_specs:
            metrics[
                f"stream/{name}/actor_objective_weight"
            ] = objective_weight
        return metrics

    def _critic_update(self, rollout: dict) -> dict[str, float]:
        chunks, n_envs, horizon = rollout["valid"].shape
        contexts = rollout["contexts"].reshape(-1, self.prefix_context_dim)
        targets = rollout["value_targets"].reshape(-1, 2)
        valid = rollout["valid"].reshape(-1)
        channel_valid = rollout["channel_valid"].reshape(-1, 2).bool()
        if not torch.equal(channel_valid[:, 0], valid.bool()):
            raise RuntimeError("FCAMP task critic validity diverged from rollout validity")
        amp_valid = channel_valid[:, 1]
        env_stream_ids = rollout["stream_ids"]
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
        stream_specs = []
        for name, stream_id, objective_weight, local_indices in local_specs:
            task_indices = valid_idx.index_select(0, local_indices)
            amp_indices = task_indices[amp_valid.index_select(0, task_indices)]
            if amp_indices.numel() == 0:
                raise RuntimeError(
                    f"FCAMP {name} stream has no clean AMP critic targets"
                )
            stream_specs.append(
                (
                    name,
                    stream_id,
                    objective_weight,
                    task_indices,
                    amp_indices,
                )
            )
        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        micro_batch_size = self._policy_micro_batch_size(
            self._policy_mini_batch_size(int(valid_idx.numel()))
        )
        task_weight = float(self.cfg.critics.task_loss_weight)
        amp_weight = float(self.cfg.critics.amp_loss_weight)
        totals = {"task": 0.0, "amp": 0.0, "grad": 0.0}
        stream_totals = {
            name: {"task": 0.0, "amp": 0.0}
            for name, _, _, _, _ in stream_specs
        }
        stream_task_steps = {name: 0 for name, _, _, _, _ in stream_specs}
        stream_amp_steps = {name: 0 for name, _, _, _, _ in stream_specs}
        steps = 0
        for _ in range(int(self.cfg.policy_epochs)):
            stream_splits: dict[
                str,
                tuple[
                    float,
                    tuple[torch.Tensor, ...],
                    tuple[torch.Tensor, ...],
                ],
            ] = {}
            for name, _, objective_weight, task_indices, amp_indices in stream_specs:
                shuffled_task = task_indices.index_select(
                    0,
                    torch.randperm(
                        task_indices.numel(), device=task_indices.device
                    ),
                )
                shuffled_amp = amp_indices.index_select(
                    0,
                    torch.randperm(
                        amp_indices.numel(), device=amp_indices.device
                    ),
                )
                stream_splits[name] = (
                    objective_weight,
                    torch.tensor_split(shuffled_task, num_mini_batches),
                    torch.tensor_split(shuffled_amp, num_mini_batches),
                )
            for mini_batch_index in range(num_mini_batches):
                parts = [
                    (
                        name,
                        objective_weight,
                        task_splits[mini_batch_index],
                        amp_splits[mini_batch_index],
                    )
                    for name, (
                        objective_weight,
                        task_splits,
                        amp_splits,
                    ) in stream_splits.items()
                    if task_splits[mini_batch_index].numel() > 0
                ]
                if not parts:
                    continue
                self.critic_optimizer.zero_grad(set_to_none=True)
                combined_task = 0.0
                combined_amp = 0.0
                for name, objective_weight, task_idx, amp_idx in parts:
                    task_denominator = float(task_idx.numel())
                    task_sum = 0.0
                    amp_sum = 0.0
                    for micro_start in range(
                        0,
                        task_idx.numel(),
                        micro_batch_size,
                    ):
                        sub = task_idx[
                            micro_start : micro_start + micro_batch_size
                        ]
                        loss_channels = self.critic.flow_matching_loss(
                            contexts[sub],
                            targets[sub],
                            fm_samples=self.flow_critic_fm_samples,
                        )
                        task_loss_sum = loss_channels[:, 0].sum()
                        (
                            objective_weight
                            * task_weight
                            * task_loss_sum
                            / task_denominator
                        ).backward()
                        task_sum += float(task_loss_sum.item())
                    task_mean = task_sum / task_denominator
                    stream_task_steps[name] += 1
                    amp_mean = 0.0
                    if amp_idx.numel() > 0:
                        amp_denominator = float(amp_idx.numel())
                        for micro_start in range(
                            0,
                            amp_idx.numel(),
                            micro_batch_size,
                        ):
                            sub = amp_idx[
                                micro_start : micro_start + micro_batch_size
                            ]
                            loss_channels = self.critic.flow_matching_loss(
                                contexts[sub],
                                targets[sub],
                                fm_samples=self.flow_critic_fm_samples,
                            )
                            amp_loss_sum = loss_channels[:, 1].sum()
                            (
                                objective_weight
                                * amp_weight
                                * amp_loss_sum
                                / amp_denominator
                            ).backward()
                            amp_sum += float(amp_loss_sum.item())
                        amp_mean = amp_sum / amp_denominator
                        stream_amp_steps[name] += 1
                    combined_task += objective_weight * task_mean
                    combined_amp += objective_weight * amp_mean
                    stream_totals[name]["task"] += task_mean
                    stream_totals[name]["amp"] += amp_mean
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
            "critic/task_valid_count": float(valid.sum().item()),
            "critic/amp_valid_count": float(amp_valid.sum().item()),
        }
        for name, _, objective_weight, _, _ in stream_specs:
            metrics[
                f"stream/{name}/critic_objective_weight"
            ] = objective_weight
            metrics[f"stream/{name}/critic_task_flow_loss"] = (
                stream_totals[name]["task"]
                / max(stream_task_steps[name], 1)
            )
            metrics[f"stream/{name}/critic_amp_flow_loss"] = (
                stream_totals[name]["amp"]
                / max(stream_amp_steps[name], 1)
            )
        return metrics

    def _sample_cpu_flat_with_end_times(
        self,
        windows_cpu: torch.Tensor,
        end_times_cpu: torch.Tensor,
        batch_size: int,
        *,
        stream_ids_cpu: torch.Tensor,
        stream_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if windows_cpu.ndim != 2 or windows_cpu.shape[1] != self.imitation_window_dim:
            raise ValueError("current discriminator windows are malformed")
        if windows_cpu.shape[0] != end_times_cpu.shape[0]:
            raise RuntimeError("current discriminator windows/end-times are misaligned")
        if stream_ids_cpu.shape != end_times_cpu.shape:
            raise RuntimeError(
                "current discriminator stream labels are misaligned"
            )
        candidates = torch.arange(windows_cpu.shape[0], device="cpu")
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
            frames, ends = self.disc_window_replay.sample(
                count,
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
        raw = self.env.motion.get_fcamp_demo_windows_at_end_indices(
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
        batch_size = max(1, int(self.cfg.style_prior.batch_size))
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
        commit_normalizer_before_training: bool = False,
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
            raise RuntimeError("FCAMP current discriminator window set is empty")
        batch_size = int(self.cfg.style_prior.batch_size)
        if current_phase0_count == 0 or current_curriculum_count == 0:
            raise RuntimeError(
                "FCAMP current discriminator windows require both streams"
            )
        first_replay_window_frames, first_replay_ends = (
            self._sample_balanced_replay_windows(batch_size)
        )
        possible_steps = math.ceil(current_count / batch_size) * int(
            self.cfg.style_prior.epochs
        )
        update_steps = min(
            possible_steps,
            int(self.cfg.style_prior.max_updates_per_iteration),
        )
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
        committed_before_training = False
        if commit_normalizer_before_training:
            # The first discriminator must not see raw, unscaled 3728-D input.
            # Commit matched expert/policy moments first; the rollout itself is
            # discarded, so no actor can consume its random-D reward snapshot.
            self.disc_normalizer.unfreeze()
            committed_before_training = bool(self.disc_normalizer.commit())
            self.disc_normalizer.freeze()
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
                gradient_penalty_weight=float(
                    self.cfg.style_prior.grad_penalty
                ),
                logit_regularization_weight=float(
                    self.cfg.style_prior.logit_reg
                ),
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
        if commit_normalizer_before_training:
            committed = committed_before_training
        else:
            self.disc_normalizer.unfreeze()
            committed = self.disc_normalizer.commit()
            self.disc_normalizer.freeze()
        denom = max(update_steps, 1)
        metrics = {key: value / denom for key, value in totals.items()}
        metrics.update(
            {
                "disc/grad_norm": grad_total / denom,
                "disc/lr": float(self.cfg.style_prior.learning_rate),
                "disc/update_steps": float(update_steps),
                "disc/version": float(self.disc_version),
                "disc/input_version": float(input_version),
                "disc_norm/committed_this_update": float(committed),
                "disc_norm/committed_before_training": float(
                    commit_normalizer_before_training
                    and committed_before_training
                ),
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
                for key, value in self.disc_window_replay.statistics(
                    current_update=update_idx
                ).items()
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

    def update(self, rollout: dict, collect_time: float) -> dict:
        update_start = time.perf_counter()
        # Global primitive steps are used as the replay age clock.
        update_idx = int(self._fcamp_update_idx)
        (
            actor_metrics,
            critic_metrics,
            disc_metrics,
            reward_metrics,
            actor_time,
            critic_time,
            disc_time,
        ) = self._optimize_rollout_snapshot(rollout, update_idx)
        return collect_update_metrics(
            self,
            rollout,
            actor_metrics=actor_metrics,
            critic_metrics=critic_metrics,
            disc_metrics=disc_metrics,
            reward_metrics=reward_metrics,
            collect_time=collect_time,
            actor_time=actor_time,
            critic_time=critic_time,
            disc_time=disc_time,
            update_start=update_start,
        )

    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        log_update(self, update_idx, max_updates, metrics)

    def log_banner(self) -> None:
        log_banner(self)
