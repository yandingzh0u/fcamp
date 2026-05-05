from __future__ import annotations

import torch

from .config import PUSH_INTERVAL_STEP_RANGE, VELOCITY_RANGE


class MimicStepMixin:
    def step(
        self,
        action_offsets: torch.Tensor,
        auto_reset: bool = False,
        reset_horizon: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        previous_action = self.last_action.clone()
        self._apply_action_targets(action_offsets)
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
        if self.cfg.render:
            self.sim.render()

        self.episode_steps += 1

        termination_phase_steps = self.phase_steps.clone()
        reward, reward_terms = self.compute_reward(action_offsets, previous_action)
        done, done_terms, debug_terms = self.compute_termination()
        terminal_observation = None

        if auto_reset:
            failure_mask = done & (~done_terms["time_out"])
            failed_env_ids = failure_mask.nonzero(as_tuple=False).squeeze(-1)
            self._record_adaptive_motion_failures(failed_env_ids, termination_phase_steps)

            if bool(done.any()):
                terminal_observation = self.get_observation().clone()
                env_ids = done.nonzero(as_tuple=False).squeeze(-1)
                reset_phases = self.sample_phase_indices(env_ids.numel(), horizon=max(1, reset_horizon))
                self.reset_envs(env_ids, phase_indices=reset_phases)

        self.last_action = action_offsets.clone()
        if auto_reset and bool(done.any()):
            self.last_action[done] = 0.0
        self.phase_steps += 1
        self._resample_finished_motions()
        self._update_adaptive_motion_sampling()
        self._apply_interval_pushes()
        observation = self.get_observation()
        info = {
            "phase_steps": self.phase_steps.clone(),
            "reward_terms": reward_terms,
            "done_terms": done_terms,
            "debug_terms": debug_terms,
            "termination_phase_steps": termination_phase_steps,
        }
        if terminal_observation is not None:
            info["final_observation"] = terminal_observation
            info["_final_observation"] = done.clone()
        return observation, reward, done, info

    def chunk_step(
        self,
        chunk_actions: torch.Tensor,
        auto_reset: bool = True,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]]]:
        if chunk_actions.ndim != 3:
            raise ValueError(
                f"chunk_actions must have shape (num_envs, chunk_size, action_dim), got {tuple(chunk_actions.shape)}"
            )
        if chunk_actions.shape[0] != self.num_envs or chunk_actions.shape[2] != self.action_dim:
            raise ValueError(
                f"Expected chunk_actions shape ({self.num_envs}, chunk_size, {self.action_dim}), "
                f"got {tuple(chunk_actions.shape)}"
            )

        chunk_size = int(chunk_actions.shape[1])
        obs_list: list[torch.Tensor] = []
        infos_list: list[dict[str, torch.Tensor]] = []
        chunk_rewards: list[torch.Tensor] = []
        raw_chunk_terminations: list[torch.Tensor] = []
        raw_chunk_truncations: list[torch.Tensor] = []
        done_in_chunk = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reset_phase_cache = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        reset_obs_cache = torch.empty(self.num_envs, self.observation_dim, device=self.device)

        for step_idx in range(chunk_size):
            obs, reward, done, info = self.step(chunk_actions[:, step_idx, :], auto_reset=False)
            truncations = info["done_terms"]["time_out"].clone()
            terminations = done & (~truncations)

            if bool(done_in_chunk.any()):
                reward = reward.clone()
                reward[done_in_chunk] = 0.0
                terminations = terminations.clone()
                truncations = truncations.clone()
                terminations[done_in_chunk] = False
                truncations[done_in_chunk] = False
                for key in info.get("reward_terms", {}):
                    info["reward_terms"][key] = info["reward_terms"][key].clone()
                    info["reward_terms"][key][done_in_chunk] = 0.0
                for key in info.get("done_terms", {}):
                    info["done_terms"][key] = info["done_terms"][key].clone()
                    info["done_terms"][key][done_in_chunk] = False
                if auto_reset:
                    obs = obs.clone()
                    obs[done_in_chunk] = reset_obs_cache[done_in_chunk]

            new_done = (~done_in_chunk) & done
            new_failures = new_done & (~truncations)
            if bool(new_failures.any()):
                failed_env_ids = new_failures.nonzero(as_tuple=False).squeeze(-1)
                self._record_adaptive_motion_failures(failed_env_ids, info["termination_phase_steps"])
                self._update_adaptive_motion_sampling()
            if auto_reset and bool(new_done.any()):
                env_ids = new_done.nonzero(as_tuple=False).squeeze(-1)
                reset_phases = self.sample_phase_indices(env_ids.numel(), horizon=chunk_size)
                reset_obs = self.reset_envs(env_ids, phase_indices=reset_phases)
                reset_phase_cache[env_ids] = reset_phases
                reset_obs_cache[env_ids] = reset_obs
                obs = obs.clone()
                obs[env_ids] = reset_obs

            done_in_chunk |= new_done
            obs_list.append(obs)
            infos_list.append(info)
            chunk_rewards.append(reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards_tensor = torch.stack(chunk_rewards, dim=1)
        raw_chunk_terminations_tensor = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations_tensor = torch.stack(raw_chunk_truncations, dim=1)

        past_terminations = raw_chunk_terminations_tensor.any(dim=1)
        past_truncations = raw_chunk_truncations_tensor.any(dim=1)
        past_dones = past_terminations | past_truncations

        if bool(past_dones.any()) and auto_reset:
            env_ids = past_dones.nonzero(as_tuple=False).squeeze(-1)
            reset_phases = reset_phase_cache.index_select(0, env_ids)
            if bool(torch.any(reset_phases < 0)):
                fallback_reset_phases = self.sample_phase_indices(env_ids.numel(), horizon=chunk_size)
                reset_phases = torch.where(reset_phases < 0, fallback_reset_phases, reset_phases)
            reset_obs = self.reset_envs(env_ids, phase_indices=reset_phases)
            obs_list[-1] = obs_list[-1].clone()
            obs_list[-1][env_ids] = reset_obs

        if auto_reset:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations_tensor)
            chunk_truncations = torch.zeros_like(raw_chunk_truncations_tensor)
            chunk_terminations[:, -1] = past_terminations
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations_tensor
            chunk_truncations = raw_chunk_truncations_tensor

        return (
            obs_list,
            chunk_rewards_tensor,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _apply_interval_pushes(self) -> None:
        if not hasattr(self, "next_push_step"):
            return
        due_env_ids = torch.where(self.episode_steps >= self.next_push_step)[0]
        if due_env_ids.numel() == 0:
            return

        velocity_range = torch.tensor(VELOCITY_RANGE, dtype=torch.float32, device=self.device)
        low = velocity_range[:, 0].unsqueeze(0)
        high = velocity_range[:, 1].unsqueeze(0)
        velocity_delta = low + (high - low) * torch.rand((due_env_ids.numel(), 6), device=self.device)
        root_velocity = self.robot.data.root_vel_w.index_select(0, due_env_ids) + velocity_delta
        self.robot.write_root_velocity_to_sim(root_velocity, env_ids=due_env_ids)

        min_interval, max_interval = PUSH_INTERVAL_STEP_RANGE
        self.next_push_step[due_env_ids] = self.episode_steps[due_env_ids] + torch.randint(
            min_interval,
            max_interval + 1,
            (due_env_ids.numel(),),
            dtype=torch.long,
            device=self.device,
        )
