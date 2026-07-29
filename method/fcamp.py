"""FC-AMP: causal Flow-chunk policy optimization with temporal discriminator prior.

The discriminator remains an independent MimicKit-style reward model.  Its
online representation is never part of the actor or critic observation; only
the scalar temporal style reward is coupled to the corresponding causal action
conditional.
"""

from __future__ import annotations

import time

import torch
from torch import nn

from components.credit.temporal_credit import (
    compute_amp_gae,
    normalize_amp_advantage,
    resolve_terminal_masks,
)
from components.rollout.flow_cps_base import FlowCPSBase
from components.rollout.fcamp_contract import FCAMP_CHECKPOINT_CONTRACT
from components.rollout.fcamp_diagnostics import FCAMPDiagnosticsMixin
from components.imitation.fcamp_discriminator import FCAMPDiscriminatorMixin
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
from components.normalization.running_stats import EmpiricalNormalization, RunningNormalizer
from components.replay.fcamp_window_buffer import FCAMPWindowReplay
from models.style_discriminator import (
    StyleDiscriminator,
)
from models.flow_critic import FlowCritic




class FCAMP(FCAMPDiagnosticsMixin, FCAMPDiscriminatorMixin, FlowCPSBase):
    """Full H=4 causal Flow-CPS policy with W=16 temporal discriminator prior."""

    def build(self) -> None:
        cfg = self.cfg
        amp_cfg = cfg.style_prior
        critic_cfg = cfg.critic
        env = self.env

        self.imitation_history_steps = int(amp_cfg.obs_steps)
        super().build()

        # FCAMP actions are normalized PD target commands.  Their algorithmic
        # domain is defined solely by action_squash_scale, never by URDF joint
        # position metadata or startup-randomized default poses.
        action_limit = float(cfg.action_squash_scale)
        self.action_low = torch.full(
            (self.num_act,), -action_limit, device=env.device
        )
        self.action_high = torch.full(
            (self.num_act,), action_limit, device=env.device
        )
        env.enable_strict_action_contract(self.action_low, self.action_high)
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
        self.critic = FlowCritic(
            context_dim=self.prefix_context_dim,
            encoder_hidden_dims=critic_cfg.encoder_hidden_dims,
            head_hidden_dims=critic_cfg.head_hidden_dims,
            activation=cfg.activation,
            flow_steps=int(cfg.flow_steps),
            eval_samples=self.FLOW_CRITIC_SAMPLES,
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
        # Expert sampling owns an isolated RNG stream. Reading initial_seed()
        # does not advance the global CPU RNG, so expert draws cannot perturb
        # current/replay minibatch sampling or any other policy-side randomness.
        self.expert_sampling_seed = int(
            (
                int(torch.initial_seed())
                + int(self._EXPERT_SAMPLING_SEED_OFFSET)
            )
            % ((1 << 63) - 1)
        )
        self.expert_sampling_generator = torch.Generator(device="cpu")
        self.expert_sampling_generator.manual_seed(
            self.expert_sampling_seed
        )
        self.imitation_history = TemporalFeatureHistory(
            env.num_envs,
            self.imitation_history_steps,
            self.imitation_frame_dim,
            device=env.device,
        )
        replay_capacity = int(amp_cfg.replay_size)
        replay_replace = int(amp_cfg.replay_samples)
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
        current_capacity = int(amp_cfg.current_buffer_size)
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
        )
        disc_parameters = [p for p in self.discriminator.parameters() if p.requires_grad]
        self.disc_optimizer = torch.optim.SGD(
            disc_parameters,
            momentum=0.9,
            lr=float(amp_cfg.learning_rate),
            weight_decay=float(amp_cfg.weight_decay),
        )
        self.disc_version = 0
        self.warmup_env_transitions = 0

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
        if not isinstance(state, dict):
            raise ValueError(
                "FCAMP checkpoint is missing its algorithm contract; start a fresh run."
            )
        for name, expected in FCAMP_CHECKPOINT_CONTRACT.items():
            saved = state.get(name)
            if type(saved) is not type(expected) or saved != expected:
                raise ValueError(
                    f"FCAMP checkpoint {name}={saved!r} differs from "
                    f"the required contract {expected!r}; start a fresh run."
                )
        if bool(state.get("discriminator_policy_conditioning", True)):
            raise ValueError(
                "FCAMP checkpoints with discriminator-conditioned policies "
                "cannot be resumed."
            )
        if state.get("expert_sampling_seed") != self.expert_sampling_seed:
            raise ValueError(
                "FCAMP checkpoint expert sampling seed differs from this run"
            )
        expert_generator_state = state.get("expert_sampling_generator_state")
        if (
            not torch.is_tensor(expert_generator_state)
            or expert_generator_state.device.type != "cpu"
            or expert_generator_state.dtype != torch.uint8
            or expert_generator_state.ndim != 1
            or expert_generator_state.numel() == 0
        ):
            raise ValueError(
                "FCAMP checkpoint has an invalid expert sampling generator state"
            )
        self._validate_checkpoint_action_domain(state)

    def _validate_checkpoint_action_domain(self, state: dict) -> None:
        for name, expected in (
            ("action_low", self.action_low),
            ("action_high", self.action_high),
        ):
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
                    f"FCAMP checkpoint {name} differs from the policy command domain"
                )

    def validate_checkpoint_payload(self, payload: dict) -> None:
        state = payload.get("algo_state") if isinstance(payload, dict) else None
        self._validate_checkpoint_contract(state)

    def extra_checkpoint_state(self) -> dict:
        payload = super().extra_checkpoint_state()
        payload.update(
            {
                "disc_optimizer": self.disc_optimizer.state_dict(),
                "disc_version": int(self.disc_version),
                "warmup_env_transitions": int(self.warmup_env_transitions),
                "expert_sampling_seed": int(self.expert_sampling_seed),
                "expert_sampling_generator_state": (
                    self.expert_sampling_generator.get_state().clone()
                ),
                "action_low": self.action_low.detach().cpu(),
                "action_high": self.action_high.detach().cpu(),
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
        self.expert_sampling_generator.set_state(
            payload["expert_sampling_generator_state"].to(device="cpu")
        )
        replay_state = payload.get("disc_window_replay")
        if replay_state is None:
            raise ValueError("FCAMP checkpoint is missing complete-window replay state")
        self.disc_window_replay.load_state_dict(replay_state)

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
        return torch.cat(outputs, dim=0).reshape(*contexts.shape[:-1])

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
        flat_streams = stream_labels.reshape(-1).to(
            device=flat.device,
            dtype=torch.int8,
        )
        eligible_indices = (
            torch.arange(flat.shape[0], device=flat.device)
            if flat_valid is None
            else flat_valid.nonzero(as_tuple=False).squeeze(-1)
        )
        specs = self._stream_specs(
            flat_streams.index_select(0, eligible_indices)
        )
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
            selections.append(
                eligible_indices.index_select(
                    0,
                    local_indices.index_select(0, order),
                )
            )
        selected_indices = torch.cat(selections, dim=0)
        batch_size = min(max(1, int(self.cfg.micro_batch_size)), 1024)
        for start in range(0, selected_indices.numel(), batch_size):
            normalizer._update(
                flat.index_select(
                    0,
                    selected_indices[start : start + batch_size],
                )
            )

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
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Start policy history from the actual post-reset simulator state.

        Demonstration predecessors are valid expert data, but they are not
        transitions produced by the policy.  Policy-side discriminator windows
        therefore become eligible only after W-1 real simulator transitions.
        """

        reset_frame = self.env.get_imitation_policy_frame(
            env_ids=env_ids,
        )
        self.imitation_history.reset_from_frame(
            reset_frame,
            env_ids=env_ids,
        )

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
            and env.max_episode_steps > 0
            and self.training_streams.curriculum_ids.numel() > 0
        ):
            curriculum_ids = self.training_streams.curriculum_ids
            random_age = torch.randint(
                0,
                int(env.max_episode_steps),
                (curriculum_ids.numel(),),
                device=env.device,
                dtype=env.episode_steps.dtype,
            )
            env.set_episode_age(curriculum_ids, random_age)
        self._obs = obs
        self._critic_obs = env.get_critic_observation()
        self._reset_imitation_history()
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
        """Prime normalization and D on one discarded policy rollout.

        Actor, critic, and their normalizers are deliberately untouched.
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
        rollout = self.collect(current_observation)
        disc_metrics = self._discriminator_update(
            0,
            rollout=rollout,
            commit_normalizer_before_training=True,
        )
        if disc_metrics.get("disc/update_steps", 0.0) <= 0.0:
            raise RuntimeError("FCAMP discriminator warm-up produced no optimizer step")
        if disc_metrics.get("disc_norm/committed_before_training", 0.0) != 1.0:
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
            "env_transitions": float(transitions),
            "amp_valid_window_count": float(rollout["amp_valid"].sum().item()),
            "replay_dirty_insert_count": float(
                self.disc_window_replay.statistics(current_update=0).get(
                    "replay/dirty_insert_count", -1.0
                )
            ),
        }
        del rollout
        return next_observation, metrics, transitions

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
        return self._obs

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
        latent_path_buf = torch.zeros(
            chunks, n_envs, flow_steps + 1, self.chunk_dim, device=device
        )
        old_log_probs_buf = torch.zeros(chunks, n_envs, flow_steps, h, device=device)
        context_buf = torch.zeros(
            chunks, n_envs, h, self.prefix_context_dim, device=device
        )
        next_context_buf = torch.zeros_like(context_buf)
        context_raw_buf = torch.zeros_like(context_buf)
        next_context_raw_buf = torch.zeros_like(context_buf)
        amp_reward_buf = torch.zeros(chunks, n_envs, h, device=device)
        amp_logit_buf = torch.zeros_like(amp_reward_buf)
        valid_buf = torch.zeros(chunks, n_envs, h, dtype=torch.bool, device=device)
        amp_valid_buf = torch.zeros_like(valid_buf)
        intervention_edge_buf = torch.zeros_like(valid_buf)
        bootstrap_buf = torch.zeros_like(valid_buf)
        trace_buf = torch.zeros_like(valid_buf)
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
        obs = current_obs
        critic_obs = self._critic_obs
        collection_start_phases = env.phase_steps.detach().clone()
        action_abs_max = 0.0
        action_bound_violation_max = 0.0
        fk_alignment_abs_max = 0.0
        fk_alignment_abs_sum = 0.0
        fk_alignment_count = 0
        geometry_checked = False
        dirty_window_excluded_count = 0
        window_latest_root_xy_abs_max = torch.zeros((), device=device)
        window_root_xy_abs_sum = torch.zeros((), device=device)
        window_root_xy_count = 0
        replay_update_idx = self._fcamp_update_idx
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
        )
        current_window_buffer.begin_update(
            0,
            replacement_quotas={PHASE0_STREAM: 0, CURRICULUM_STREAM: 0},
        )
        with torch.no_grad():
            for chunk_idx in range(chunks):
                chunk_actor_raw = obs.clone()
                chunk_critic_raw = critic_obs.clone()
                actor_obs_n = self.actor_obs_normalizer(chunk_actor_raw)
                previous_action = chunk_actor_raw[..., -self.num_act :].detach()
                final_latent, latent_path, old_log_probs, _ = self._sample_cps_path(actor_obs_n)
                action_chunk = self._policy._action_transform(
                    final_latent, prev_action=previous_action
                ).view(n_envs, h, self.num_act)
                action_abs_max = max(action_abs_max, float(action_chunk.abs().max().item()))
                below = (self.action_low - action_chunk).clamp_min(0.0)
                above = (action_chunk - self.action_high).clamp_min(0.0)
                action_bound_violation_max = max(
                    action_bound_violation_max,
                    float(torch.maximum(below, above).max().item()),
                )

                actor_obs_buf[chunk_idx] = actor_obs_n
                actor_obs_raw_buf[chunk_idx] = chunk_actor_raw
                latent_path_buf[chunk_idx] = latent_path
                old_log_probs_buf[chunk_idx] = old_log_probs

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
                    next_obs, done, info = env.step(action_t)
                    next_critic_obs = env.get_critic_observation()

                    # The reset state is the oldest policy-side history frame.
                    # A discriminator endpoint becomes legal only after W-1
                    # actual post-action frames have completed the window.
                    imitation_frame = info["imitation_frame"]
                    alive_ids = alive_before.nonzero(as_tuple=False).squeeze(-1)
                    intervention_edges = info["intervention_edge_mask"].bool()
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
                            fk_alignment_abs_sum = float(
                                alignment_error.sum().item()
                            )
                            fk_alignment_count = int(alignment_error.numel())
                            geometry_checked = True
                            if fk_alignment_abs_max > 1.0e-4:
                                raise RuntimeError(
                                    "FCAMP expert/runtime 233-D geometry contract failed: "
                                    f"max_abs={fk_alignment_abs_max:.6g}"
                                )
                    active_float = alive_before.to(dtype=amp_reward_buf.dtype)
                    amp_reward = torch.zeros(n_envs, device=device)
                    amp_logits = torch.zeros_like(amp_reward)
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
                        amp_reward.index_copy_(0, ready_ids, ready_rewards)
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
                    amp_reward *= active_float
                    amp_reward_buf[chunk_idx, :, frame_idx] = amp_reward
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
                    # An interval push occurs after this endpoint was captured.
                    # AMP credit cannot bootstrap or trace across that exogenous
                    # edge.
                    bootstrap_buf[chunk_idx, :, frame_idx] = (
                        alive_before
                        & ~new_failure
                        & ~new_motion_complete
                        & ~intervention_edges
                    )
                    trace_buf[chunk_idx, :, frame_idx] = (
                        alive_before
                        & ~new_done
                        & ~intervention_edges
                    )

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
                    context_buf[chunk_idx, :, frame_idx] = (
                        self.prefix_context_normalizer(current_context_raw)
                    )
                    next_context_buf[chunk_idx, :, frame_idx] = (
                        self.prefix_context_normalizer(next_context_raw)
                    )

                    self._record_train_episode_stats(
                        amp_reward,
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
                        env_ids=reset_ids,
                    )
                    self.phase0_attempts.start(reset_ids)

            values = self._evaluate_prefix_values(context_buf)
            next_values = self._evaluate_prefix_values(next_context_buf)

        if not geometry_checked:
            self.disc_window_replay.abort_update()
            current_window_buffer.abort_update()
            raise RuntimeError("FCAMP could not perform a same-state FK alignment check")
        if bool((intervention_edge_buf & done_buf).any()):
            self.disc_window_replay.abort_update()
            current_window_buffer.abort_update()
            raise RuntimeError("terminal FCAMP transitions must never receive an interval push")
        if action_bound_violation_max > 1.0e-6:
            self.disc_window_replay.abort_update()
            current_window_buffer.abort_update()
            raise RuntimeError(
                "FCAMP actor emitted an action outside its configured policy command domain"
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
        rollout = {
            "actor_obs": actor_obs_buf,
            "actor_obs_raw": actor_obs_raw_buf,
            "latents": latent_path_buf,
            "old_log_probs": old_log_probs_buf,
            "contexts": context_buf,
            "contexts_raw": context_raw_buf,
            "next_contexts_raw": next_context_raw_buf,
            "values": values,
            "next_values": next_values,
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
            "amp_reward": amp_reward_buf,
            "amp_logits": amp_logit_buf,
            "disc_version_used": rollout_disc_version,
            "disc_normalizer_count_used": rollout_disc_normalizer_count,
            "stream_ids": self.training_streams.stream_ids,
            "collection_start_phases": collection_start_phases,
            "action_abs_max": action_abs_max,
            "action_bound_violation_max": action_bound_violation_max,
            "fk_alignment_abs_max": fk_alignment_abs_max,
            "fk_alignment_abs_mean": (
                fk_alignment_abs_sum / max(fk_alignment_count, 1)
            ),
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

        rollout["amp_reward_credit"] = (
            rollout["amp_reward"] * float(self.env.dt)
        )
        rewards_time = chronological(rollout["amp_reward_credit"])
        values_time = chronological(rollout["values"])
        next_values_time = chronological(rollout["next_values"])
        bootstrap_time = chronological(rollout["bootstrap_mask"])
        trace_time = chronological(rollout["trace_mask"])
        action_valid_time = chronological(rollout["valid"]).bool()
        endpoint_valid_time = chronological(rollout["amp_valid"]).bool()
        if bool((endpoint_valid_time & ~action_valid_time).any()):
            raise RuntimeError(
                "FCAMP discriminator endpoint validity must be a subset of "
                "real alive policy actions"
            )
        if bool(
            (
                (rewards_time != 0.0)
                & ~endpoint_valid_time
            ).any()
        ):
            raise RuntimeError(
                "FCAMP produced non-zero AMP reward without a legal "
                "discriminator endpoint"
            )
        # Endpoint validity gates D evaluation and reward production only.
        # Every real alive action remains part of PPO/critic credit; ordinary
        # GAE carries later legal AMP rewards backward through the W-1 warm-up
        # actions, while terminal/intervention masks remain the only trace cuts.
        valid_time = action_valid_time
        credit = compute_amp_gae(
            rewards_time,
            values_time,
            next_values_time,
            bootstrap_time,
            trace_time,
            valid_time,
            gamma=float(self.cfg.discount_gamma),
            gae_lambda=float(self.cfg.gae_lambda),
        )
        if not torch.equal(credit.valid_mask, action_valid_time):
            raise RuntimeError(
                "FCAMP action-credit validity diverged from alive policy actions"
            )
        # Diagnostic only: distinguish "included in PPO/critic" from "this
        # finite rollout contains a later non-zero AMP reward reachable through
        # the uncut GAE trace".  The latter may be false near rollout tails and
        # for episodes that terminate before their first W-frame endpoint.
        delayed_amp_reachable_time = torch.zeros_like(action_valid_time)
        reachable_later = torch.zeros(
            n_envs,
            dtype=torch.bool,
            device=action_valid_time.device,
        )
        for time_index in range(rewards_time.shape[0] - 1, -1, -1):
            has_amp_reward = rewards_time[time_index] != 0.0
            reachable_now = action_valid_time[time_index] & (
                has_amp_reward
                | (trace_time[time_index].bool() & reachable_later)
            )
            delayed_amp_reachable_time[time_index] = reachable_now
            reachable_later = reachable_now
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
        credit = normalize_amp_advantage(
            credit,
            valid_time,
            normalization_weights,
        )

        rollout["advantages"] = chunk_layout(credit.actor_advantage)
        rollout["amp_advantages"] = chunk_layout(credit.advantages)
        rollout["value_targets"] = chunk_layout(credit.value_targets)
        rollout["credit_valid"] = chunk_layout(credit.valid_mask)
        rollout["delayed_amp_reachable"] = chunk_layout(
            delayed_amp_reachable_time
        )

    # ------------------------------------------------------------------ #
    # Optimizers
    # ------------------------------------------------------------------ #
    def _stream_specs(
        self,
        labels: torch.Tensor,
    ) -> list[tuple[str, int, float, torch.Tensor]]:
        configured_phase0 = float(self.cfg.streams.phase0_fraction)
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
        chunks, n_envs = rollout["valid"].shape[:2]
        batch_size = chunks * n_envs
        h = self.horizon_h
        flow_steps = int(self.cfg.flow_steps)
        actor_obs = rollout["actor_obs"].reshape(batch_size, self.actor_obs_dim)
        latent_path = rollout["latents"].reshape(
            batch_size, flow_steps + 1, self.chunk_dim
        )
        old_log_probs = rollout["old_log_probs"].reshape(batch_size, flow_steps, h)
        advantages = rollout["advantages"].reshape(batch_size, h)
        valid = rollout["credit_valid"].reshape(batch_size, h)
        env_stream_ids = rollout["stream_ids"]
        stream_labels = env_stream_ids.reshape(1, n_envs).expand(
            chunks, n_envs
        ).reshape(-1)
        eligible = valid.any(dim=1).nonzero(as_tuple=False).squeeze(-1)
        if eligible.numel() == 0:
            raise RuntimeError("FCAMP rollout has no clean AMP actor samples")
        stream_specs = [
            (
                name,
                stream_id,
                objective_weight,
                eligible.index_select(0, local_indices),
            )
            for name, stream_id, objective_weight, local_indices in self._stream_specs(
                stream_labels.index_select(0, eligible)
            )
        ]
        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        micro_batch_size = self._policy_micro_batch_size(
            self._policy_mini_batch_size(int(eligible.numel()))
        )
        clip_low = 1.0 - float(self.cfg.clip_range)
        clip_high = 1.0 + float(self.cfg.clip_range)
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
        chunks, n_envs, horizon = rollout["credit_valid"].shape
        contexts = rollout["contexts"].reshape(-1, self.prefix_context_dim)
        targets = rollout["value_targets"].reshape(-1)
        valid = rollout["credit_valid"].reshape(-1).bool()
        stream_labels = rollout["stream_ids"].reshape(1, n_envs, 1).expand(
            chunks,
            n_envs,
            horizon,
        ).reshape(-1)
        valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            raise RuntimeError("FCAMP rollout has no clean AMP critic samples")
        stream_specs = [
            (
                name,
                stream_id,
                objective_weight,
                valid_idx.index_select(0, local_indices),
            )
            for name, stream_id, objective_weight, local_indices in self._stream_specs(
                stream_labels.index_select(0, valid_idx)
            )
        ]
        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        micro_batch_size = self._policy_micro_batch_size(
            self._policy_mini_batch_size(int(valid_idx.numel()))
        )
        totals = {"loss": 0.0, "grad": 0.0, "grad_clipped": 0.0}
        stream_totals = {name: 0.0 for name, _, _, _ in stream_specs}
        stream_steps = {name: 0 for name, _, _, _ in stream_specs}
        steps = 0
        for _ in range(int(self.cfg.policy_epochs)):
            stream_splits = {}
            for name, _, objective_weight, indices in stream_specs:
                shuffled = indices.index_select(
                    0,
                    torch.randperm(
                        indices.numel(), device=indices.device
                    ),
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
                    for name, (objective_weight, splits) in stream_splits.items()
                    if splits[mini_batch_index].numel() > 0
                ]
                if not parts:
                    continue
                self.critic_optimizer.zero_grad(set_to_none=True)
                combined_loss = 0.0
                for name, objective_weight, indices in parts:
                    denominator = float(indices.numel())
                    loss_sum_value = 0.0
                    for micro_start in range(
                        0,
                        indices.numel(),
                        micro_batch_size,
                    ):
                        sub = indices[
                            micro_start : micro_start + micro_batch_size
                        ]
                        loss_sum = self.critic.flow_matching_loss(
                            contexts[sub],
                            targets[sub],
                            fm_samples=self.FLOW_CRITIC_FM_SAMPLES,
                        ).sum()
                        (objective_weight * loss_sum / denominator).backward()
                        loss_sum_value += float(loss_sum.item())
                    mean_loss = loss_sum_value / denominator
                    combined_loss += objective_weight * mean_loss
                    stream_totals[name] += mean_loss
                    stream_steps[name] += 1
                grad = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), float(self.cfg.max_grad_norm)
                )
                self.critic_optimizer.step()
                grad_value = float(grad)
                totals["loss"] += combined_loss
                totals["grad"] += grad_value
                totals["grad_clipped"] += float(
                    grad_value > float(self.cfg.max_grad_norm)
                )
                steps += 1
        denom = max(steps, 1)
        metrics = {
            "critic/flow_loss": totals["loss"] / denom,
            "critic/grad_norm": totals["grad"] / denom,
            "critic/grad_clip_fraction": totals["grad_clipped"] / denom,
            "critic/lr": float(self.critic_learning_rate),
            "critic/optimizer_steps": float(steps),
            "critic/valid_count": float(valid.sum().item()),
        }
        for name, _, objective_weight, _ in stream_specs:
            metrics[
                f"stream/{name}/critic_objective_weight"
            ] = objective_weight
            metrics[f"stream/{name}/critic_flow_loss"] = (
                stream_totals[name] / max(stream_steps[name], 1)
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
                rollout["credit_valid"],
                stream_labels=frame_streams,
            )
            self._update_empirical_normalizer_chunked(
                self.prefix_context_normalizer,
                rollout["next_contexts_raw"],
                rollout["credit_valid"],
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
