from __future__ import annotations

import torch

from .spec import VELOCITY_RANGE


class MimicStepMixin:
    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        applied_actions = self._apply_action_targets(actions)
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
        if self.render:
            self._render_step_index += 1
            if self._render_step_index % self.render_every == 0:
                self.sim.render()

        phase_start_steps = self.phase_steps.clone()
        reference_frame_delta = torch.full_like(
            phase_start_steps,
            float(self.motion_frame_delta),
            dtype=torch.float32,
        )
        next_phase_steps = phase_start_steps + reference_frame_delta.to(dtype=phase_start_steps.dtype)
        self.episode_steps += 1
        self._motion_end_mask = next_phase_steps >= (
            self.motion.num_frames - 1
        )
        reference_phase_steps = torch.clamp(
            next_phase_steps, max=self.motion.num_frames - 1
        )
        self.phase_steps = reference_phase_steps

        termination_phase_steps = reference_phase_steps.clone()
        done, done_terms, debug_terms = self.compute_termination()
        self.last_action.copy_(applied_actions)
        imitation_frame = self.get_imitation_policy_frame()

        tracking_failure = done_terms["anchor_pos_bad"] | done_terms["anchor_ori_bad"] | done_terms["ee_body_bad"]
        self._record_adaptive_failures(tracking_failure, termination_phase_steps)

        self._fold_adaptive_sampler()

        # External impulses are interventions on the edge leading to the next
        # returned observation. Never mutate an already terminal transition.
        intervention_edge_mask = self._apply_interval_pushes(
            eligible_mask=~done.bool()
        )
        observation = self.get_observation()
        info = {
            "done_terms": done_terms,
            "debug_terms": debug_terms,
            "phase_start_steps": phase_start_steps,
            "reference_phase_steps": reference_phase_steps,
            "termination_phase_steps": termination_phase_steps,
            "imitation_frame_phase_steps": reference_phase_steps,
            "reference_frame_delta": reference_frame_delta,
            "interval_push_mask": self._last_interval_push_mask.clone(),
            "intervention_edge_mask": intervention_edge_mask,
            "imitation_frame": imitation_frame,
        }
        return observation, done, info

    def _apply_interval_pushes(
        self,
        eligible_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._last_interval_push_mask.zero_()
        if not self.interval_pushes:
            return self._last_interval_push_mask.clone()
        if eligible_mask is None:
            eligible_mask = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        if eligible_mask.shape != (self.num_envs,):
            raise ValueError(
                f"eligible_mask must have shape {(self.num_envs,)}, "
                f"got {tuple(eligible_mask.shape)}"
            )
        eligible_mask = eligible_mask.to(device=self.device, dtype=torch.bool)
        due_env_ids = torch.where(self.episode_steps >= self.next_push_step)[0]
        if due_env_ids.numel() > 0:
            due_env_ids = due_env_ids[
                eligible_mask.index_select(0, due_env_ids)
            ]
        if due_env_ids.numel() == 0:
            return self._last_interval_push_mask.clone()
        self._last_interval_push_mask[due_env_ids] = True
        # record each env's first-push episode-step for validation diagnostics.
        first_timers = due_env_ids[self.first_push_step[due_env_ids] < 0]
        if first_timers.numel() > 0:
            self.first_push_step[first_timers] = self.episode_steps[first_timers]

        velocity_range = torch.tensor(VELOCITY_RANGE, dtype=torch.float32, device=self.device)
        low = velocity_range[:, 0].unsqueeze(0)
        high = velocity_range[:, 1].unsqueeze(0)
        velocity_delta = low + (high - low) * torch.rand((due_env_ids.numel(), 6), device=self.device)
        root_velocity = (
            self.get_mimic_root_velocity_w().index_select(0, due_env_ids)
            + velocity_delta
        )
        self.write_mimic_root_velocity_to_sim(root_velocity, due_env_ids)

        min_interval, max_interval = self.push_interval_step_range
        self.next_push_step[due_env_ids] = self.episode_steps[due_env_ids] + torch.randint(
            min_interval,
            max_interval + 1,
            (due_env_ids.numel(),),
            dtype=torch.long,
            device=self.device,
        )
        return self._last_interval_push_mask.clone()

    def _reset_interval_push_schedule(self, env_ids: torch.Tensor) -> None:
        self.first_push_step[env_ids] = -1
        min_push, max_push = self.push_interval_step_range
        self.next_push_step[env_ids] = (
            self.episode_steps.index_select(0, env_ids)
            + torch.randint(
                min_push,
                max_push + 1,
                (env_ids.numel(),),
                dtype=torch.long,
                device=self.device,
            )
        )

    def set_episode_age(
        self,
        env_ids: torch.Tensor,
        episode_age: torch.Tensor,
        *,
        reschedule_interval_pushes: bool = True,
    ) -> None:
        """Set synthetic episode ages without leaving event clocks stale."""

        if env_ids.ndim != 1 or episode_age.shape != env_ids.shape:
            raise ValueError(
                "env_ids and episode_age must be matching 1-D tensors"
            )
        if torch.is_floating_point(episode_age):
            raise ValueError("episode_age must use an integer dtype")
        episode_age = episode_age.to(
            device=self.device, dtype=self.episode_steps.dtype
        )
        if bool((episode_age < 0).any()):
            raise ValueError("episode_age must be non-negative")
        self.episode_steps.index_copy_(0, env_ids, episode_age)
        if reschedule_interval_pushes:
            self._reset_interval_push_schedule(env_ids)
