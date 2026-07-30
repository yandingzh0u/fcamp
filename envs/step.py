from __future__ import annotations

import torch


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

        observation = self.get_observation()
        info = {
            "done_terms": done_terms,
            "debug_terms": debug_terms,
            "phase_start_steps": phase_start_steps,
            "reference_phase_steps": reference_phase_steps,
            "termination_phase_steps": termination_phase_steps,
            "imitation_frame_phase_steps": reference_phase_steps,
            "reference_frame_delta": reference_frame_delta,
            "imitation_frame": imitation_frame,
        }
        return observation, done, info
