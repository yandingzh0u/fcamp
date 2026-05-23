from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path

import torch

from net.mixgrpo import FlowMatchingPolicy
from ..checkpoint import CheckpointMixin
from .config import MixGRPOConfig
from ..env_factory import make_mimic_env
from ..env_state import EnvStateMixin
from .inference import deterministic_sde_ode_actions
from ..logging import LoggingMixin
from ..returns import compute_gae_returns
from .sampling import flow_grpo_step
from ..validation import ValidationMixin


class MixGRPOTrainer(ValidationMixin, CheckpointMixin, LoggingMixin, EnvStateMixin):
    def __init__(self, simulation_app, cfg: MixGRPOConfig):
        self.simulation_app = simulation_app
        self.cfg = cfg
        self.start_update = 1
        if cfg.horizon != 1:
            raise ValueError(f"Official-style training requires horizon=1, got {cfg.horizon}")
        if float(cfg.value_loss_coef) != 0.0:
            raise ValueError("MixGRPO is critic-free; value_loss_coef must be 0.")
        self.chunk_dim = cfg.horizon * cfg.action_dim
        self.checkpoint_dir = Path(cfg.checkpoint_dir).expanduser().resolve() if cfg.checkpoint_dir else None
        self._debug_probe_update = 0
        self._debug_probe_sample_printed = False

        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        self.env = make_mimic_env(cfg)
        if self.env.action_dim != cfg.action_dim:
            raise ValueError(f"Expected env action_dim {self.env.action_dim}, got {cfg.action_dim}")
        if cfg.num_generations < 1:
            raise ValueError(f"num_generations must be >= 1, got {cfg.num_generations}")
        if self.env.num_envs % cfg.num_generations != 0:
            raise ValueError(
                f"num_envs ({self.env.num_envs}) must be divisible by num_generations ({cfg.num_generations})"
            )
        self.num_grpo_groups = self.env.num_envs // cfg.num_generations

        policy_obs_dim = cfg.policy_obs_dim if cfg.policy_obs_dim > 0 else self.env.observation_dim
        self.policy = FlowMatchingPolicy(
            obs_dim=policy_obs_dim,
            action_dim=cfg.action_dim,
            horizon=cfg.horizon,
            hidden_dims=cfg.actor_hidden_dims,
            activation=cfg.activation,
            init_noise_std=cfg.init_noise_std,
            action_squash_scale=cfg.action_squash_scale,
        ).to(self.env.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=cfg.policy_lr,
            betas=(0.9, 0.999),
            eps=1.0e-8,
        )
        self.learning_rate = float(cfg.policy_lr)
        self._init_train_episode_stats()
        self.current_observation = self._reset_training_envs()

        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if cfg.resume:
            self._load_checkpoint(Path(cfg.resume).expanduser().resolve())

    def _debug_enabled(self) -> bool:
        if not self.cfg.debug_probe:
            return False
        every = max(1, int(self.cfg.debug_probe_every))
        return self._debug_probe_update > 0 and self._debug_probe_update % every == 0

    def _peek(self, tensor: torch.Tensor | None, name: str, max_values: int = 5) -> str:
        if tensor is None:
            return f"[DEBUG] {name:<20} | EMPTY: None"
        if tensor.numel() == 0:
            return f"[DEBUG] {name:<20} | shape={list(tensor.shape)} | EMPTY"
        tensor_detached = tensor.detach()
        finite_mask = torch.isfinite(tensor_detached)
        finite_frac = finite_mask.float().mean().item()
        safe_tensor = tensor_detached[finite_mask] if bool(finite_mask.any()) else tensor_detached.reshape(-1)
        t_min = float(safe_tensor.min().item())
        t_max = float(safe_tensor.max().item())
        t_mean = float(safe_tensor.float().mean().item())
        t_std = float(safe_tensor.float().std(unbiased=False).item()) if safe_tensor.numel() > 1 else 0.0
        flat = tensor_detached.reshape(tensor_detached.shape[0], -1)
        sample_index = min(flat.shape[0] - 1, (self._debug_probe_update * 9973) % flat.shape[0])
        sample = flat[sample_index, : min(max_values, flat.shape[1])].detach().cpu().tolist()
        sample_str = ", ".join(f"{float(value):.4f}" for value in sample)
        return (
            f"[DEBUG] {name:<20} | shape={list(tensor_detached.shape)} "
            f"| min={t_min:>9.4f} max={t_max:>9.4f} mean={t_mean:>9.4f} std={t_std:>9.4f} "
            f"finite={finite_frac * 100:>6.2f}% | sample_5=[{sample_str}]"
        )

    def _debug_print_collection(self, group_data: dict[str, torch.Tensor]) -> None:
        if not self._debug_enabled():
            return
        valid_mask = group_data["valid_mask"]
        print("\n" + "=" * 72, flush=True)
        print(f"[DEBUG STAGE 1] Environment & Collection | update={self._debug_probe_update}", flush=True)
        print(self._peek(group_data["obs"], "obs_raw"), flush=True)
        print(self._peek(group_data["rewards"], "rollout_rewards"), flush=True)
        print(self._peek(group_data["chunk_rewards"], "chunk_rewards"), flush=True)
        if "raw_chunk_rewards" in group_data:
            print(self._peek(group_data["raw_chunk_rewards"], "raw_chunk_rewards"), flush=True)
        print(self._peek(group_data["actions"], "sample_actions"), flush=True)
        print(self._peek(group_data["old_log_probs"], "old_log_probs"), flush=True)
        print(self._peek(group_data["latents"][..., -1, :], "final_latents"), flush=True)
        alive_ratio = valid_mask.float().mean().item()
        per_chunk_alive = valid_mask.float().mean(dim=(0, 1))
        per_group_return = group_data["rewards"].mean(dim=0)
        live_chunks = valid_mask.float().sum(dim=-1)
        first_done_phase = group_data.get("first_done_phase")
        first_done_chunk = group_data.get("first_done_chunk")
        collection_start_phases = group_data.get("collection_start_phases")
        if collection_start_phases is not None:
            print(
                "[DEBUG] start_phase="
                f"min={int(collection_start_phases.min().item())} "
                f"mean={float(collection_start_phases.float().mean().item()):.1f} "
                f"max={int(collection_start_phases.max().item())}",
                flush=True,
            )
        print(f"[DEBUG] survival_valid_frac={alive_ratio * 100:.2f}%", flush=True)
        print(
            "[DEBUG] per_chunk_alive_first8="
            + ", ".join(f"{float(value):.3f}" for value in per_chunk_alive[:8])
            + " | last="
            + f"{float(per_chunk_alive[-1]):.3f}",
            flush=True,
        )
        print(
            "[DEBUG] live_chunks="
            f"min={float(live_chunks.min().item()):.1f} "
            f"mean={float(live_chunks.mean().item()):.1f} "
            f"p95={float(torch.quantile(live_chunks.flatten(), 0.95).item()):.1f} "
            f"max={float(live_chunks.max().item()):.1f}",
            flush=True,
        )
        if first_done_phase is not None and first_done_chunk is not None:
            failed = first_done_phase >= 0
            if bool(failed.any()):
                print(
                    "[DEBUG] first_failure="
                    f"chunk_mean={float(first_done_chunk[failed].float().mean().item()):.1f} "
                    f"chunk_min={int(first_done_chunk[failed].min().item())} "
                    f"chunk_max={int(first_done_chunk[failed].max().item())} "
                    f"phase_mean={float(first_done_phase[failed].float().mean().item()):.1f} "
                    f"phase_min={int(first_done_phase[failed].min().item())} "
                    f"phase_max={int(first_done_phase[failed].max().item())}",
                    flush=True,
                )
        print(
            "[DEBUG] per_group_return="
            + ", ".join(f"{float(value):.5f}" for value in per_group_return),
            flush=True,
        )
        if alive_ratio < 0.05:
            print("[DEBUG WARNING] valid_mask is below 5%; almost no usable rollout data.", flush=True)

    def _debug_print_advantages(
        self,
        advantages: torch.Tensor,
        adv_flat: torch.Tensor,
        update_mask: torch.Tensor,
        group_data: dict[str, torch.Tensor],
    ) -> None:
        if not self._debug_enabled():
            return
        print("\n[DEBUG STAGE 2] GRPO Advantages", flush=True)
        print(self._peek(advantages, "adv_raw"), flush=True)
        print(self._peek(adv_flat, "adv_flat"), flush=True)
        valid_adv = advantages[group_data["valid_mask"]]
        print(self._peek(valid_adv, "adv_valid_raw"), flush=True)
        first_life_valid = group_data["valid_mask"].reshape(-1)
        print(
            f"[DEBUG] update_sample_count={int(update_mask.sum().item())}/{update_mask.numel()} "
            f"first_life_valid_count={int(first_life_valid.sum().item())}/{first_life_valid.numel()}",
            flush=True,
        )
        if adv_flat.numel() == 0 or adv_flat.abs().max().item() < 1e-4:
            print("[DEBUG WARNING] advantages are nearly zero; group exploration may be indistinguishable.", flush=True)
        same_group_std = group_data["rewards"].std(dim=1, unbiased=False)
        print(self._peek(same_group_std, "chunk_reward_std_g"), flush=True)

    def _train_step_indices(self, device: torch.device | str) -> torch.Tensor:
        return torch.arange(int(self.cfg.flow_steps), device=device, dtype=torch.long)

    def _replicate_group_reset_state(
        self,
        group_count: int,
        generation_count: int,
        group_ids: torch.Tensor | None = None,
    ) -> None:
        if generation_count <= 1:
            return
        total_envs = group_count * generation_count
        if total_envs != self.env.num_envs:
            raise ValueError(
                f"group_count * generation_count must equal num_envs, got {group_count} * {generation_count} != {self.env.num_envs}"
            )

        device = self.env.device
        if group_ids is None:
            env_ids = torch.arange(total_envs, device=device, dtype=torch.long)
            target_env_ids = env_ids[(env_ids % generation_count) != 0]
            source_for_target = (target_env_ids // generation_count) * generation_count
        else:
            group_ids = group_ids.to(device=device, dtype=torch.long).reshape(-1)
            if group_ids.numel() == 0:
                return
            branch_offsets = torch.arange(1, generation_count, device=device, dtype=torch.long)
            target_env_ids = (group_ids[:, None] * generation_count + branch_offsets[None, :]).reshape(-1)
            source_for_target = (group_ids * generation_count).repeat_interleave(generation_count - 1)
        if target_env_ids.numel() == 0:
            return
        root_state = self.env.robot.data.root_state_w.index_select(0, source_for_target)
        joint_pos = self.env.robot.data.joint_pos.index_select(0, source_for_target)
        joint_vel = self.env.robot.data.joint_vel.index_select(0, source_for_target)
        root_pos_local = root_state[:, :3] - self.env.scene.env_origins.index_select(0, source_for_target)

        self.env.default_root_state[target_env_ids] = self.env.default_root_state.index_select(0, source_for_target)
        self.env.default_joint_pos[target_env_ids] = self.env.default_joint_pos.index_select(0, source_for_target)
        self.env.default_joint_vel[target_env_ids] = self.env.default_joint_vel.index_select(0, source_for_target)
        self.env.default_action_joint_pos[target_env_ids] = self.env.default_action_joint_pos.index_select(
            0,
            source_for_target,
        )
        self.env.default_action_joint_vel[target_env_ids] = self.env.default_action_joint_vel.index_select(
            0,
            source_for_target,
        )

        target_env_ids_cpu = target_env_ids.detach().cpu()
        source_for_target_cpu = source_for_target.detach().cpu()
        try:
            coms = self.env.robot.root_physx_view.get_coms().clone()
            coms[target_env_ids_cpu] = coms[source_for_target_cpu]
            self.env.robot.root_physx_view.set_coms(coms, target_env_ids_cpu)
        except Exception as exc:
            print(f"[WARN] Failed to replicate group torso COM randomization: {exc}", flush=True)
        try:
            materials = self.env.robot.root_physx_view.get_material_properties().clone()
            materials[target_env_ids_cpu] = materials[source_for_target_cpu]
            self.env.robot.root_physx_view.set_material_properties(materials, target_env_ids_cpu)
        except Exception as exc:
            print(f"[WARN] Failed to replicate group material randomization: {exc}", flush=True)

        self.env.scene.reset(env_ids=target_env_ids)
        self.env._write_robot_state(
            root_pos=root_pos_local,
            root_quat=root_state[:, 3:7],
            root_lin_vel=root_state[:, 7:10],
            root_ang_vel=root_state[:, 10:13],
            joint_pos=joint_pos[:, self.env.action_joint_ids],
            joint_vel=joint_vel[:, self.env.action_joint_ids],
            env_ids=target_env_ids,
        )
        self.env.phase_steps[target_env_ids] = self.env.phase_steps.index_select(0, source_for_target)
        self.env.episode_steps[target_env_ids] = self.env.episode_steps.index_select(0, source_for_target)
        self.env.last_action[target_env_ids] = self.env.last_action.index_select(0, source_for_target)
        self.env.next_push_step[target_env_ids] = self.env.next_push_step.index_select(0, source_for_target)
        self.env.scene.update(self.env.physics_dt)

    def _compute_group_relative_advantages(self, rewards: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if rewards.ndim != 2:
            raise ValueError(f"rewards must have shape (groups, generations), got {tuple(rewards.shape)}")
        if valid_mask.shape != rewards.shape:
            raise ValueError("valid_mask must match rewards")
        rewards_float = rewards.to(torch.float32)
        valid_float = valid_mask.to(dtype=torch.float32)
        counts = valid_float.sum(dim=1, keepdim=True).clamp(min=1.0)
        means = (rewards_float * valid_float).sum(dim=1, keepdim=True) / counts
        centered = (rewards_float - means) * valid_float
        denom = (counts - 1.0).clamp(min=1.0)
        variances = centered.square().sum(dim=1, keepdim=True) / denom
        stds = torch.sqrt(variances + 1e-8)
        advantages = (rewards_float - means) / stds
        return torch.where(valid_mask, advantages.to(dtype=rewards.dtype), torch.zeros_like(rewards))

    def _init_train_episode_stats(self) -> None:
        self._train_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.env.device)
        self._train_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.env.device)
        self._train_reward_buffer: deque[float] = deque(maxlen=100)
        self._train_length_buffer: deque[float] = deque(maxlen=100)
        self._train_completed_episodes = 0

    def _record_train_episode_stats(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        if not hasattr(self, "_train_reward_sum"):
            self._init_train_episode_stats()
        self._train_reward_sum += rewards.to(dtype=torch.float32)
        self._train_episode_length += 1.0
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return
        self._train_reward_buffer.extend(self._train_reward_sum.index_select(0, done_ids).detach().cpu().tolist())
        self._train_length_buffer.extend(self._train_episode_length.index_select(0, done_ids).detach().cpu().tolist())
        self._train_completed_episodes += int(done_ids.numel())
        self._train_reward_sum[done_ids] = 0.0
        self._train_episode_length[done_ids] = 0.0

    def _sde_sample_with_logprobs(
        self,
        obs: torch.Tensor,
        noise: torch.Tensor,
        sde_noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        train_step_indices = self._train_step_indices(obs.device)
        actions, latent_path, step_log_probs = self._sde_ode_rollout_actions(
            obs,
            initial_noise=noise,
            sde_noise=sde_noise,
        )
        if self._debug_enabled() and not self._debug_probe_sample_printed:
            self._debug_probe_sample_printed = True
            print("\n[DEBUG STAGE 3] Flow Action Generation", flush=True)
            print(self._peek(actions, "sample_actions"), flush=True)
            print(self._peek(latent_path[:, -1], "final_latents"), flush=True)
            print(self._peek(step_log_probs, "sde_old_log_probs"), flush=True)
            print(
                "[DEBUG] sde_train_steps="
                + ",".join(str(int(index)) for index in train_step_indices.detach().cpu().tolist()),
                flush=True,
            )
            print(f"[DEBUG] action_abs_mean={actions.abs().mean().item():.5f}", flush=True)
            if actions.abs().mean().item() < 0.01:
                print("[DEBUG WARNING] action magnitude is below 0.01; policy may be nearly idle.", flush=True)
        return {
            "actions": actions.view(obs.shape[0], self.policy.horizon, self.policy.action_dim),
            "all_latents": latent_path.detach(),
            "log_probs": step_log_probs.detach(),
            "train_step_indices": train_step_indices.detach().clone(),
            "sigma_schedule": torch.linspace(1.0, 0.0, self.cfg.flow_steps + 1, device=obs.device, dtype=obs.dtype),
        }

    def _sample_policy_with_logprobs(
        self,
        obs: torch.Tensor,
        noise: torch.Tensor,
        sde_noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self._sde_sample_with_logprobs(obs, noise, sde_noise=sde_noise)

    def _sde_ode_mean_actions(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.zeros(obs.shape[0], self.chunk_dim, device=obs.device, dtype=obs.dtype)
        self.policy._validate_inputs(obs, noise, self.cfg.flow_steps)
        obs_prep = self.policy._prepare_observation(obs)
        batch_size = obs.shape[0]
        steps = self.cfg.flow_steps
        sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=obs.device, dtype=obs.dtype)
        latent = noise
        all_latents = [latent.detach()]
        zero_step_noise = torch.zeros_like(latent)

        for step_index in range(steps):
            sigma = sigma_schedule[step_index]
            timestep_batch = torch.full(
                (batch_size,),
                float(sigma.item()),
                device=obs.device,
                dtype=obs.dtype,
            )

            model_output = self.policy.velocity_field(obs_prep, latent, timestep_batch)
            latent, _ = flow_grpo_step(
                model_output=model_output,
                latents=latent,
                sigmas=sigma_schedule,
                index=step_index,
                eta=float(self.cfg.sde_eta),
                deterministic=False,
                sample_noise=zero_step_noise,
            )
            all_latents.append(latent.detach())
        mean_actions = self.policy._action_transform(latent)
        return mean_actions, torch.stack(all_latents, dim=1)

    def _sde_ode_rollout_actions(
        self,
        obs: torch.Tensor,
        *,
        initial_noise: torch.Tensor,
        sde_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.policy._validate_inputs(obs, initial_noise, self.cfg.flow_steps)
        obs_prep = self.policy._prepare_observation(obs)
        batch_size = obs.shape[0]
        steps = int(self.cfg.flow_steps)
        sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=obs.device, dtype=obs.dtype)
        latent = initial_noise.to(device=obs.device, dtype=obs.dtype) * float(self.cfg.init_noise_std)
        all_latents = [latent.detach()]
        if sde_noise is None:
            sde_noise = torch.randn(batch_size, steps, self.chunk_dim, device=obs.device, dtype=obs.dtype)
        elif sde_noise.shape != (batch_size, steps, self.chunk_dim):
            raise ValueError(
                f"sde_noise must have shape {(batch_size, steps, self.chunk_dim)}, got {tuple(sde_noise.shape)}"
            )
        step_log_probs = []

        for step_index in range(steps):
            sigma = sigma_schedule[step_index]
            timestep_batch = torch.full(
                (batch_size,),
                float(sigma.item()),
                device=obs.device,
                dtype=obs.dtype,
            )

            model_output = self.policy.velocity_field(obs_prep, latent, timestep_batch)
            latent, log_prob = flow_grpo_step(
                model_output=model_output,
                latents=latent,
                sigmas=sigma_schedule,
                index=step_index,
                eta=float(self.cfg.sde_eta),
                deterministic=False,
                sample_noise=sde_noise[:, step_index],
            )
            all_latents.append(latent.detach())
            step_log_probs.append(log_prob)

        if not step_log_probs:
            raise RuntimeError("SDE-ODE rollout produced no trainable transition log-probs.")
        actions = self.policy._action_transform(latent)
        return actions, torch.stack(all_latents, dim=1), torch.stack(step_log_probs, dim=1)

    def _mean_actions(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._sde_ode_mean_actions(obs)

    def _deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        initial_noise = None
        if getattr(self.cfg, "eval_initial_noise", "random") == "random":
            initial_noise = torch.randn(obs.shape[0], self.chunk_dim, device=obs.device, dtype=obs.dtype)
        return deterministic_sde_ode_actions(
            self.policy,
            obs,
            steps=self.cfg.flow_steps,
            sde_eta=float(self.cfg.sde_eta),
            initial_noise=initial_noise,
        )

    def _compute_transition_log_probs(
        self,
        obs: torch.Tensor,
        latent_path: torch.Tensor,
        step_indices: torch.Tensor,
    ) -> torch.Tensor:
        if latent_path.ndim != 3:
            raise ValueError(f"latent_path must have shape (batch, steps + 1, dim), got {tuple(latent_path.shape)}")
        sample_count, path_steps, latent_dim = latent_path.shape
        steps = int(self.cfg.flow_steps)
        if obs.shape[0] != sample_count:
            raise ValueError("obs and latent_path batch sizes must match")
        if path_steps != steps + 1 or latent_dim != self.chunk_dim:
            raise ValueError(
                f"latent_path must have shape {(sample_count, steps + 1, self.chunk_dim)}, got {tuple(latent_path.shape)}"
            )
        if step_indices.ndim != 1 or step_indices.numel() == 0:
            raise ValueError("step_indices must be a non-empty 1-D tensor")

        obs_prep = self.policy._prepare_observation(obs)
        sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=obs.device, dtype=obs.dtype)
        log_probs = []
        for step_tensor in step_indices.to(device=obs.device, dtype=torch.long):
            step_index = int(step_tensor.item())
            if step_index < 0 or step_index >= steps:
                raise ValueError(f"MixGRPO step index {step_index} is outside [0, {steps})")
            latent_t = latent_path[:, step_index].to(dtype=obs.dtype)
            next_latent = latent_path[:, step_index + 1].to(dtype=obs.dtype)
            sigma = sigma_schedule[step_index]
            timestep_batch = torch.full(
                (sample_count,),
                float(sigma.item()),
                device=obs.device,
                dtype=obs.dtype,
            )
            model_output = self.policy.velocity_field(obs_prep, latent_t, timestep_batch)
            _, log_prob = flow_grpo_step(
                model_output=model_output,
                latents=latent_t,
                sigmas=sigma_schedule,
                index=step_index,
                eta=float(self.cfg.sde_eta),
                prev_sample=next_latent,
                deterministic=False,
            )
            log_probs.append(log_prob)
        return torch.stack(log_probs, dim=1)

    def _record_first_done(
        self,
        *,
        done_mask: torch.Tensor,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        infos_list: list[dict[str, torch.Tensor]],
        chunk_index: int,
        first_done_chunk: torch.Tensor,
        first_done_phase: torch.Tensor,
        first_done_anchor_pos: torch.Tensor,
        first_done_anchor_ori: torch.Tensor,
        first_done_ee_body: torch.Tensor,
        first_done_timeout: torch.Tensor,
    ) -> None:
        done_steps = terminations | truncations
        new_done_any = (~done_mask) & done_steps.any(dim=1)
        if not bool(new_done_any.any()):
            return

        first_offsets = done_steps.to(dtype=torch.long).argmax(dim=1)
        env_ids = new_done_any.nonzero(as_tuple=False).squeeze(-1)
        first_done_chunk[env_ids] = int(chunk_index)
        if infos_list:
            if all("termination_phase_steps" in step_info for step_info in infos_list):
                phase_by_step = torch.stack(
                    [step_info["termination_phase_steps"] for step_info in infos_list],
                    dim=1,
                )
                first_done_phase[env_ids] = phase_by_step[env_ids, first_offsets[env_ids]]
            for key, target in (
                ("anchor_pos_bad", first_done_anchor_pos),
                ("anchor_ori_bad", first_done_anchor_ori),
                ("ee_body_bad", first_done_ee_body),
                ("time_out", first_done_timeout),
            ):
                if not all(key in step_info.get("done_terms", {}) for step_info in infos_list):
                    continue
                cause_by_step = torch.stack(
                    [step_info["done_terms"][key] for step_info in infos_list],
                    dim=1,
                )
                target[env_ids] = cause_by_step[env_ids, first_offsets[env_ids]].bool()
        else:
            if hasattr(self.env, "phase_steps"):
                first_done_phase[env_ids] = self.env.phase_steps[env_ids]
            first_done_ee_body[env_ids] = terminations.any(dim=1)[env_ids]
            first_done_timeout[env_ids] = truncations.any(dim=1)[env_ids]

    def _collect_rollout(self, current_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        total_envs = self.env.num_envs
        generation_count = int(getattr(self.cfg, "num_generations", 1))
        group_count = total_envs // generation_count
        chunks_per_rollout = self._chunks_per_grpo_update()
        collection_start_phases = (
            self.env.phase_steps.detach().clone()
            if hasattr(self.env, "phase_steps")
            else torch.zeros(total_envs, dtype=torch.long, device=self.env.device)
        )
        obs_t = current_obs
        critic_obs_t = torch.zeros(total_envs, 0, device=self.env.device)
        sigma_schedule = None
        metric_action_abs_max_all = 0.0

        first_done_chunk = torch.full(
            (total_envs,),
            chunks_per_rollout,
            dtype=torch.long,
            device=self.env.device,
        )
        first_done_phase = torch.full((total_envs,), -1, dtype=torch.long, device=self.env.device)
        first_done_anchor_pos = torch.zeros(total_envs, dtype=torch.bool, device=self.env.device)
        first_done_anchor_ori = torch.zeros(total_envs, dtype=torch.bool, device=self.env.device)
        first_done_ee_body = torch.zeros(total_envs, dtype=torch.bool, device=self.env.device)
        first_done_timeout = torch.zeros(total_envs, dtype=torch.bool, device=self.env.device)
        ever_done = torch.zeros(total_envs, dtype=torch.bool, device=self.env.device)

        rollout_obs = []
        rollout_critic_obs = []
        rollout_latents = []
        rollout_old_log_probs = []
        rollout_train_step_indices = None
        rollout_actions = []
        rollout_rewards = []
        rollout_dones = []
        rollout_timeouts = []
        rollout_valid = []
        rollout_values = []
        rollout_infos = []
        metric_rollout_info_items: list[tuple[dict[str, torch.Tensor], torch.Tensor]] = []

        metric_chunk_return_first = None
        metric_chunk_return_last = None
        metric_actions_first = None
        metric_actions_last = None

        for chunk_index in range(chunks_per_rollout):
            active_before_step = ~ever_done
            if bool(getattr(self.cfg, "init_same_noise", True)):
                noise = torch.randn(group_count, self.chunk_dim, device=self.env.device).repeat_interleave(
                    generation_count,
                    dim=0,
                )
            else:
                noise = torch.randn(total_envs, self.chunk_dim, device=self.env.device)
            sde_noise = torch.randn(
                total_envs,
                int(self.cfg.flow_steps),
                self.chunk_dim,
                device=self.env.device,
                dtype=obs_t.dtype,
            )
            with torch.no_grad():
                sample = self._sample_policy_with_logprobs(obs_t, noise, sde_noise=sde_noise)
                value_t = torch.zeros(total_envs, device=self.env.device)

            sigma_schedule = sample["sigma_schedule"]
            rollout_train_step_indices = sample["train_step_indices"]
            action_chunk = sample["actions"]
            expected_action_shape = (total_envs, 1, self.cfg.action_dim)
            if action_chunk.shape != expected_action_shape:
                raise RuntimeError(
                    f"policy produced action_chunk shape {tuple(action_chunk.shape)}, expected {expected_action_shape}"
                )
            metric_action_abs_max_all = max(metric_action_abs_max_all, float(action_chunk.abs().max().item()))
            action_t = action_chunk[:, 0, :]
            next_obs_t, reward_t, done_t, info_t = self.env.step(
                action_t,
                auto_reset=True,
                reset_horizon=max(1, chunks_per_rollout - chunk_index),
            )
            timeout_t = info_t["done_terms"]["time_out"].bool()
            termination_t = done_t & (~timeout_t)

            self._record_first_done(
                done_mask=ever_done,
                terminations=termination_t[:, None],
                truncations=timeout_t[:, None],
                infos_list=[info_t],
                chunk_index=chunk_index,
                first_done_chunk=first_done_chunk,
                first_done_phase=first_done_phase,
                first_done_anchor_pos=first_done_anchor_pos,
                first_done_anchor_ori=first_done_anchor_ori,
                first_done_ee_body=first_done_ee_body,
                first_done_timeout=first_done_timeout,
            )
            ever_done |= done_t

            if chunk_index == 0:
                metric_chunk_return_first = reward_t.detach()
                metric_actions_first = action_chunk.detach().clone()
            if chunk_index == chunks_per_rollout - 1:
                metric_chunk_return_last = reward_t.detach()
                metric_actions_last = action_chunk.detach().clone()

            rollout_obs.append(obs_t)
            rollout_critic_obs.append(critic_obs_t)
            rollout_latents.append(sample["all_latents"])
            rollout_old_log_probs.append(sample["log_probs"].detach())
            rollout_actions.append(action_chunk.detach())
            rollout_rewards.append(reward_t.detach())
            rollout_dones.append(done_t.detach())
            rollout_timeouts.append(timeout_t.detach())
            rollout_valid.append(active_before_step.detach())
            rollout_values.append(value_t.detach())
            rollout_infos.append(info_t)
            metric_rollout_info_items.append((info_t, active_before_step.detach()))
            self._record_train_episode_stats(reward_t.detach(), done_t.detach())

            obs_t = next_obs_t
            critic_obs_t = torch.zeros(total_envs, 0, device=self.env.device)

        last_values = torch.zeros(total_envs, device=self.env.device)
        tail_steps = max(0, int(getattr(self.cfg, "tail_bootstrap_steps", 0)))
        tail_alive_at_end = (~ever_done).clone()
        if tail_steps > 0 and bool(tail_alive_at_end.any()):
            last_values = self._compute_tail_bootstrap(
                obs_t,
                tail_alive_at_end,
                tail_steps=tail_steps,
                gamma=float(self.cfg.discount_gamma),
                terminal_penalty=float(self.cfg.terminal_penalty),
            )

        chunk_rewards = torch.stack(rollout_rewards, dim=1)
        valid_steps = torch.stack(rollout_valid, dim=1)
        objective_chunk_rewards = chunk_rewards * valid_steps.to(dtype=chunk_rewards.dtype)
        score_denominator = valid_steps.to(dtype=chunk_rewards.dtype).sum(dim=1).clamp(min=1.0)
        score_rewards = objective_chunk_rewards.sum(dim=1) / score_denominator
        actions = torch.stack(rollout_actions, dim=1)
        latents = torch.stack(rollout_latents, dim=1)
        old_log_probs = torch.stack(rollout_old_log_probs, dim=1)
        if rollout_train_step_indices is None:
            rollout_train_step_indices = self._train_step_indices(self.env.device)
        valid_mask = valid_steps.view(group_count, generation_count, chunks_per_rollout)
        metric_chunk_return = (
            metric_chunk_return_first
            if metric_chunk_return_first is not None
            else torch.zeros(total_envs, device=self.env.device)
        )
        metric_actions = (
            metric_actions_first
            if metric_actions_first is not None
            else torch.zeros(total_envs, 1, self.cfg.action_dim, device=self.env.device)
        )
        return {
            "obs": torch.stack(rollout_obs, dim=1).view(
                group_count,
                generation_count,
                chunks_per_rollout,
                -1,
            ),
            "critic_obs": torch.stack(rollout_critic_obs, dim=1).view(
                group_count,
                generation_count,
                chunks_per_rollout,
                -1,
            ),
            "rewards": objective_chunk_rewards.sum(dim=1).view(group_count, generation_count),
            "raw_rewards": chunk_rewards.sum(dim=1).view(group_count, generation_count),
            "score_rewards": score_rewards.view(group_count, generation_count),
            "chunk_rewards": objective_chunk_rewards.view(group_count, generation_count, chunks_per_rollout),
            "raw_chunk_rewards": chunk_rewards.view(group_count, generation_count, chunks_per_rollout),
            "train_chunk_rewards": chunk_rewards.view(group_count, generation_count, chunks_per_rollout),
            "dones": torch.stack(rollout_dones, dim=1).view(group_count, generation_count, chunks_per_rollout),
            "timeouts": torch.stack(rollout_timeouts, dim=1).view(group_count, generation_count, chunks_per_rollout),
            "values": torch.stack(rollout_values, dim=1).view(group_count, generation_count, chunks_per_rollout),
            "last_values": last_values,
            "latents": latents.view(
                group_count,
                generation_count,
                chunks_per_rollout,
                self.cfg.flow_steps + 1,
                self.chunk_dim,
            ),
            "old_log_probs": old_log_probs.view(
                group_count,
                generation_count,
                chunks_per_rollout,
                -1,
            ),
            "train_step_indices": rollout_train_step_indices,
            "actions": actions.view(group_count, generation_count, chunks_per_rollout, 1, self.cfg.action_dim),
            "valid_mask": valid_mask,
            "first_done_chunk": first_done_chunk.view(group_count, generation_count),
            "first_done_phase": first_done_phase.view(group_count, generation_count),
            "first_done_anchor_pos": first_done_anchor_pos.view(group_count, generation_count),
            "first_done_anchor_ori": first_done_anchor_ori.view(group_count, generation_count),
            "first_done_ee_body": first_done_ee_body.view(group_count, generation_count),
            "first_done_timeout": first_done_timeout.view(group_count, generation_count),
            "collection_start_phases": collection_start_phases.view(group_count, generation_count),
            "sigma_schedule": sigma_schedule,
            "metric_chunk_return": metric_chunk_return,
            "metric_actions": metric_actions,
            "metric_infos_list": rollout_infos[:1],
            "metric_chunk_return_first": metric_chunk_return_first,
            "metric_chunk_return_last": metric_chunk_return_last,
            "metric_actions_first": metric_actions_first,
            "metric_actions_last": metric_actions_last,
            "metric_action_abs_max_all": metric_action_abs_max_all,
            "metric_rollout_info_items": metric_rollout_info_items,
            "next_observation": obs_t.detach().clone(),
            "next_critic_observation": critic_obs_t.detach().clone(),
        }

    def _compute_tail_bootstrap(
        self,
        obs_start: torch.Tensor,
        alive_mask: torch.Tensor,
        *,
        tail_steps: int,
        gamma: float,
        terminal_penalty: float,
    ) -> torch.Tensor:
        """Roll out the deterministic policy for `tail_steps` steps to estimate a
        Monte-Carlo tail bootstrap value for each env.

        The result is used as `last_values` for GAE / RTG so failures occurring shortly
        after the GRPO window can still flow gradient back to in-window actions, without
        introducing a learned critic.

        - alive_mask: shape (E,), envs that survived the main rollout. Dead envs already
          paid their terminal_penalty in-window, so we return 0 for them.
        - On any tail-step termination the running discounted sum stops accumulating and
          a `gamma**k * terminal_penalty` is subtracted (matching in-window penalty bookkeeping).
        - `auto_reset=False` so the env state is *not* respawned during the tail; tail
          metrics are kept independent. The env state is left dirty afterwards because
          the next training update calls `_resample_group_starts` which resets all envs.
        - Computation is done in inference_mode; no gradients flow into the policy.
        """
        if tail_steps <= 0:
            return torch.zeros(obs_start.shape[0], device=obs_start.device, dtype=obs_start.dtype)

        was_training = self.policy.training
        self.policy.eval()
        try:
            with torch.no_grad():
                obs_t = obs_start
                tail_return = torch.zeros(obs_t.shape[0], device=obs_t.device, dtype=obs_t.dtype)
                still_alive = alive_mask.clone()
                if not bool(still_alive.any()):
                    return tail_return
                horizon = int(self.cfg.horizon)
                cached_chunk: torch.Tensor | None = None
                chunk_index = horizon
                discount = torch.ones(obs_t.shape[0], device=obs_t.device, dtype=obs_t.dtype)
                for step_idx in range(tail_steps):
                    if cached_chunk is None or chunk_index >= horizon:
                        cached_chunk = self._deterministic_actions(obs_t)
                        chunk_index = 0
                    action = cached_chunk[:, chunk_index, :]
                    # zero out actions for envs already dead in the tail to avoid
                    # giving them physical commands; the env will keep stepping but
                    # we will mask their reward contribution below.
                    if bool((~still_alive).any()):
                        action = torch.where(still_alive.unsqueeze(-1), action, torch.zeros_like(action))
                    chunk_index += 1
                    obs_t, reward, step_done, _info = self.env.step(action, auto_reset=False)
                    contrib_mask = still_alive.to(dtype=tail_return.dtype)
                    tail_return = tail_return + discount * reward.to(dtype=tail_return.dtype) * contrib_mask
                    new_done = still_alive & step_done
                    if bool(new_done.any()):
                        tail_return = tail_return - discount * float(terminal_penalty) * new_done.to(
                            dtype=tail_return.dtype
                        )
                    still_alive = still_alive & ~step_done
                    if not bool(still_alive.any()):
                        break
                    discount = discount * float(gamma)
        finally:
            if was_training:
                self.policy.train()
        # zero out envs that were not alive at the start of the tail; their final
        # in-window penalty already reflects the death event.
        tail_return = tail_return * alive_mask.to(dtype=tail_return.dtype)
        return tail_return.detach()

    def _compute_gae_returns(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        last_values: torch.Tensor,
        timeouts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_gae_returns(
            rewards,
            dones,
            values,
            last_values,
            gamma=float(self.cfg.discount_gamma),
            lam=float(self.cfg.gae_lambda),
            timeouts=timeouts,
            normalize_advantage=True,
        )

    def _compute_path_log_probs_and_kl(
        self,
        obs: torch.Tensor,
        latent_path: torch.Tensor,
        step_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_count = obs.shape[0]
        if latent_path.shape[0] != sample_count:
            raise ValueError("obs and latent_path batch sizes must match")
        step_log_probs = self._compute_transition_log_probs(obs, latent_path, step_indices)
        return step_log_probs, torch.zeros_like(step_log_probs)

    def _policy_update(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        latent_path: torch.Tensor,
        old_log_probs: torch.Tensor,
        train_step_indices: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
    ) -> dict[str, float]:
        sample_count = obs.shape[0]
        if actions.ndim != 2:
            raise ValueError("MixGRPO PPO update expects one action sample per env step.")
        if latent_path.ndim != 3:
            raise ValueError("MixGRPO PPO update expects one SDE-ODE latent path per env step.")
        if old_log_probs.ndim != 2:
            raise ValueError("old_log_probs must contain per-SDE-step transition scores.")
        if old_log_probs.shape != (sample_count, train_step_indices.numel()):
            raise ValueError(
                f"old_log_probs must have shape {(sample_count, train_step_indices.numel())}, got {tuple(old_log_probs.shape)}"
            )
        if critic_obs.shape[0] != sample_count or returns.shape != (sample_count,) or old_values.shape != (sample_count,):
            raise ValueError("critic_obs, returns, and old_values must match the flattened sample count.")
        if sample_count == 0:
            return {
                "policy/loss": 0.0,
                "policy/policy_loss": 0.0,
                "policy/value_loss": 0.0,
                "policy/entropy": 0.0,
                "policy/clip_frac": 0.0,
                "policy/ratio": 1.0,
                "policy/ratio_min": 1.0,
                "policy/ratio_max": 1.0,
                "policy/logprob_delta_abs": 0.0,
                "policy/old_log_prob": 0.0,
                "policy/new_log_prob": 0.0,
                "policy/kl_loss": 0.0,
                "policy/step_ratio": 1.0,
                "policy/step_ratio_min": 1.0,
                "policy/step_ratio_max": 1.0,
                "policy/step_clip_frac": 0.0,
                "policy/step_logprob_delta_abs": 0.0,
                "policy/step_kl_loss": 0.0,
                "policy/post_ratio": 1.0,
                "policy/post_ratio_min": 1.0,
                "policy/post_ratio_max": 1.0,
                "policy/post_clip_frac": 0.0,
                "policy/post_logprob_delta_abs": 0.0,
                "policy/post_kl_loss": 0.0,
                "policy/grad_norm": 0.0,
                "policy/action_delta": 0.0,
                "policy/param_rms_delta": 0.0,
                "policy/sample_count": 0.0,
                "policy/advantage_abs_mean": 0.0,
                "policy/effective_mini_batch_size": 0.0,
                "policy/optimizer_steps": 0.0,
                "policy/lr": float(self.optimizer.param_groups[0]["lr"]),
                "policy/sde_train_steps": 0.0,
            }
        mini_batch_size = self._policy_mini_batch_size(sample_count)
        advantage_abs_mean = float(advantages.detach().abs().mean().item())
        train_value = float(self.cfg.value_loss_coef) != 0.0
        probe_count = min(128, sample_count)
        probe_obs = obs[:probe_count]
        with torch.no_grad():
            probe_action_before = self._deterministic_actions(probe_obs)
            params_before = [param.detach().clone() for param in self.policy.parameters()]

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "pre_clip_frac": 0.0,
            "pre_ratio": 0.0,
            "pre_ratio_min": float("inf"),
            "pre_ratio_max": 0.0,
            "step_clip_frac": 0.0,
            "step_ratio": 0.0,
            "step_ratio_min": float("inf"),
            "step_ratio_max": 0.0,
            "step_logprob_delta_abs": 0.0,
            "step_kl_loss": 0.0,
            "logprob_delta_abs": 0.0,
            "old_log_prob": 0.0,
            "new_log_prob": 0.0,
            "kl_loss": 0.0,
            "grad_norm": 0.0,
        }
        mini_batch_update_count = 0
        grad_update_count = 0
        micro_batch_count = 0

        permutation_count = max(mini_batch_size, (sample_count // mini_batch_size) * mini_batch_size)
        perm = torch.randperm(permutation_count, device=obs.device)
        for _ in range(self.cfg.policy_epochs):
            for start in range(0, permutation_count, mini_batch_size):
                mb = perm[start : start + mini_batch_size]
                mb_obs = obs[mb]
                mb_actions = actions[mb]
                mb_latent_path = latent_path[mb]
                mb_critic_obs = critic_obs[mb]
                mb_old_log_probs = old_log_probs[mb]
                mb_adv = advantages[mb]
                mb_returns = returns[mb]
                mb_old_values = old_values[mb]
                debug_first_policy_batch = self._debug_enabled() and start == 0
                micro_batch_size = self._policy_micro_batch_size(mb.numel())

                self.optimizer.zero_grad(set_to_none=True)
                mb_size = max(1, mb.numel())
                mb_policy_loss_value = 0.0
                mb_value_loss_value = 0.0
                mb_entropy_value = 0.0
                mb_kl_loss_value = 0.0
                mb_clip_weighted = 0.0
                mb_ratio_weighted = 0.0
                mb_logprob_delta_weighted = 0.0
                mb_old_logprob_weighted = 0.0
                mb_new_logprob_weighted = 0.0
                mb_kl_weighted = 0.0
                mb_ratio_min = float("inf")
                mb_ratio_max = 0.0
                debug_new_log_probs = None
                debug_log_ratio = None
                debug_ratio = None

                for micro_start in range(0, mb_size, micro_batch_size):
                    micro_end = min(micro_start + micro_batch_size, mb_size)
                    weight = (micro_end - micro_start) / mb_size
                    micro_obs = mb_obs[micro_start:micro_end]
                    micro_critic_obs = mb_critic_obs[micro_start:micro_end]
                    micro_actions = mb_actions[micro_start:micro_end]
                    micro_latent_path = mb_latent_path[micro_start:micro_end]
                    micro_old_log_probs = mb_old_log_probs[micro_start:micro_end]
                    micro_adv = mb_adv[micro_start:micro_end]
                    micro_returns = mb_returns[micro_start:micro_end]
                    micro_old_values = mb_old_values[micro_start:micro_end]

                    new_log_probs, _ = self._compute_path_log_probs_and_kl(
                        micro_obs,
                        micro_latent_path,
                        train_step_indices,
                    )
                    log_ratio = new_log_probs - micro_old_log_probs
                    ratio = torch.exp(log_ratio)
                    micro_adv_steps = torch.clamp(
                        micro_adv,
                        -float(self.cfg.adv_clip_max),
                        float(self.cfg.adv_clip_max),
                    ).unsqueeze(-1)
                    unclipped_loss = -micro_adv_steps * ratio
                    clipped_loss = -micro_adv_steps * torch.clamp(
                        ratio,
                        1.0 - self.cfg.clip_range,
                        1.0 + self.cfg.clip_range,
                    )
                    micro_policy_loss = torch.maximum(unclipped_loss, clipped_loss).mean()
                    if train_value:
                        value_pred = self.critic(micro_critic_obs)
                        if self.cfg.use_clipped_value_loss:
                            value_clipped = micro_old_values + (value_pred - micro_old_values).clamp(
                                -self.cfg.clip_range,
                                self.cfg.clip_range,
                            )
                            value_losses = (value_pred - micro_returns).square()
                            value_losses_clipped = (value_clipped - micro_returns).square()
                            micro_value_loss = torch.maximum(value_losses, value_losses_clipped).mean()
                        else:
                            micro_value_loss = (micro_returns - value_pred).square().mean()
                    else:
                        micro_value_loss = torch.zeros((), device=obs.device, dtype=micro_policy_loss.dtype)
                    micro_entropy = torch.zeros((), device=obs.device, dtype=micro_policy_loss.dtype)
                    # KL between new and old action distribution; tracked as a metric and used
                    # by adaptive_kl learning-rate scheduling, NOT added to the loss.
                    micro_kl_loss = 0.5 * log_ratio.square().mean()
                    micro_loss = (
                        micro_policy_loss
                        + float(self.cfg.value_loss_coef) * micro_value_loss
                        - float(self.cfg.entropy_coef) * micro_entropy
                    )
                    (micro_loss * weight).backward()

                    with torch.no_grad():
                        mb_policy_loss_value += float(micro_policy_loss.item() * weight)
                        mb_value_loss_value += float(micro_value_loss.item() * weight)
                        mb_entropy_value += float(micro_entropy.item() * weight)
                        mb_kl_loss_value += float(micro_kl_loss.item() * weight)
                        mb_clip_weighted += (
                            torch.abs(ratio - 1.0) > self.cfg.clip_range
                        ).float().mean().item() * weight
                        mb_ratio_weighted += float(ratio.mean().item() * weight)
                        mb_logprob_delta_weighted += float(log_ratio.abs().mean().item() * weight)
                        mb_old_logprob_weighted += float(micro_old_log_probs.mean().item() * weight)
                        mb_new_logprob_weighted += float(new_log_probs.mean().item() * weight)
                        mb_kl_weighted += float(micro_kl_loss.item() * weight)
                        mb_ratio_min = min(mb_ratio_min, float(ratio.min().item()))
                        mb_ratio_max = max(mb_ratio_max, float(ratio.max().item()))
                        if debug_first_policy_batch and debug_new_log_probs is None:
                            debug_new_log_probs = new_log_probs.detach()
                            debug_log_ratio = log_ratio.detach()
                            debug_ratio = ratio.detach()
                    micro_batch_count += 1

                if debug_first_policy_batch:
                    print("\n[DEBUG STAGE 4A] MixGRPO Step-LogProb Update | epoch_first_batch", flush=True)
                    print(self._peek(mb_actions, "mb_actions"), flush=True)
                    print(self._peek(mb_latent_path[:, :-1], "mb_latents_t"), flush=True)
                    print(self._peek(mb_latent_path[:, 1:], "mb_next_latents"), flush=True)
                    print(self._peek(mb_old_log_probs, "old_log_probs"), flush=True)
                    print(self._peek(debug_new_log_probs, "new_log_probs"), flush=True)
                    print(self._peek(debug_log_ratio, "step_log_ratio"), flush=True)
                    print(self._peek(debug_ratio, "step_ratio"), flush=True)
                    print(self._peek(mb_adv, "mb_adv"), flush=True)
                    print(self._peek(mb_returns, "mb_returns"), flush=True)
                    print(self._peek(mb_old_values, "mb_old_values"), flush=True)
                    print(
                        f"[DEBUG] mb_policy_loss={mb_policy_loss_value:.6f} "
                        f"mb_value_loss={mb_value_loss_value:.6f} "
                        f"mb_kl_loss={mb_kl_loss_value:.6f}",
                        flush=True,
                    )

                self._update_adaptive_learning_rate(mb_kl_loss_value)
                clip_params = list(self.policy.parameters())
                if train_value:
                    clip_params += list(self.critic.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, self.cfg.max_grad_norm)
                self.optimizer.step()

                totals["policy_loss"] += mb_policy_loss_value
                totals["value_loss"] += mb_value_loss_value
                totals["entropy"] += mb_entropy_value
                totals["kl_loss"] += mb_kl_loss_value
                totals["pre_clip_frac"] += mb_clip_weighted
                totals["pre_ratio"] += mb_ratio_weighted
                totals["pre_ratio_min"] = min(totals["pre_ratio_min"], mb_ratio_min)
                totals["pre_ratio_max"] = max(totals["pre_ratio_max"], mb_ratio_max)
                totals["logprob_delta_abs"] += mb_logprob_delta_weighted
                totals["old_log_prob"] += mb_old_logprob_weighted
                totals["new_log_prob"] += mb_new_logprob_weighted
                totals["grad_norm"] += float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                mini_batch_update_count += 1
                grad_update_count += 1

                with torch.no_grad():
                    debug_step_log_ratio = None
                    debug_step_ratio = None
                    for micro_start in range(0, mb_size, micro_batch_size):
                        micro_end = min(micro_start + micro_batch_size, mb_size)
                        weight = (micro_end - micro_start) / mb_size
                        step_new_log_probs, _ = self._compute_path_log_probs_and_kl(
                            mb_obs[micro_start:micro_end],
                            mb_latent_path[micro_start:micro_end],
                            train_step_indices,
                        )
                        step_log_ratio = step_new_log_probs - mb_old_log_probs[micro_start:micro_end]
                        step_ratio = torch.exp(step_log_ratio)
                        step_kl_loss = 0.5 * step_log_ratio.square().mean()
                        totals["step_clip_frac"] += (
                            torch.abs(step_ratio - 1.0) > self.cfg.clip_range
                        ).float().mean().item() * weight
                        totals["step_ratio"] += float(step_ratio.mean().item() * weight)
                        totals["step_ratio_min"] = min(totals["step_ratio_min"], float(step_ratio.min().item()))
                        totals["step_ratio_max"] = max(totals["step_ratio_max"], float(step_ratio.max().item()))
                        totals["step_logprob_delta_abs"] += float(step_log_ratio.abs().mean().item() * weight)
                        totals["step_kl_loss"] += float(step_kl_loss.item() * weight)
                        if debug_first_policy_batch and debug_step_log_ratio is None:
                            debug_step_log_ratio = step_log_ratio.detach()
                            debug_step_ratio = step_ratio.detach()
                if debug_first_policy_batch:
                    print(self._peek(debug_step_log_ratio, "step_log_ratio"), flush=True)
                    print(self._peek(debug_step_ratio, "step_ratio"), flush=True)

        with torch.no_grad():
            ratio_probe_count = min(256, sample_count)
            post_new_log_probs, _ = self._compute_path_log_probs_and_kl(
                obs[:ratio_probe_count],
                latent_path[:ratio_probe_count],
                train_step_indices,
            )
            post_log_ratio = post_new_log_probs - old_log_probs[:ratio_probe_count]
            post_ratio_tensor = torch.exp(post_log_ratio)
            post_ratio = float(post_ratio_tensor.mean().item())
            post_ratio_min = float(post_ratio_tensor.min().item())
            post_ratio_max = float(post_ratio_tensor.max().item())
            post_logprob_delta_abs = float(post_log_ratio.abs().mean().item())
            post_kl_loss = float((0.5 * post_log_ratio.square()).mean().item())
            post_clip_frac = float((torch.abs(post_ratio_tensor - 1.0) > self.cfg.clip_range).float().mean().item())

        update_denom = max(mini_batch_update_count, 1)
        grad_denom = max(grad_update_count, 1)
        with torch.no_grad():
            probe_action_after = self._deterministic_actions(probe_obs)
            action_delta = torch.mean(torch.abs(probe_action_after - probe_action_before))

            param_delta_sq = torch.tensor(0.0, device=obs.device)
            param_count = 0
            for param, before in zip(self.policy.parameters(), params_before, strict=True):
                delta = param.detach() - before
                param_delta_sq = param_delta_sq + torch.sum(delta * delta)
                param_count += delta.numel()
            param_rms_delta = torch.sqrt(param_delta_sq / max(param_count, 1))

        policy_loss = totals["policy_loss"] / update_denom
        entropy = totals["entropy"] / update_denom
        kl_loss = totals["kl_loss"] / update_denom
        value_loss = totals["value_loss"] / update_denom
        total_loss = policy_loss + float(self.cfg.value_loss_coef) * value_loss - float(self.cfg.entropy_coef) * entropy
        current_policy_lr = float(self.optimizer.param_groups[0]["lr"])
        current_critic_lr = 0.0

        return {
            "policy/loss": total_loss,
            "policy/policy_loss": policy_loss,
            "policy/value_loss": value_loss,
            "policy/entropy": entropy,
            "policy/kl_loss": kl_loss,
            "policy/clip_frac": totals["pre_clip_frac"] / update_denom,
            "policy/ratio": totals["pre_ratio"] / update_denom,
            "policy/ratio_min": totals["pre_ratio_min"] if totals["pre_ratio_min"] != float("inf") else 0.0,
            "policy/ratio_max": totals["pre_ratio_max"],
            "policy/logprob_delta_abs": totals["logprob_delta_abs"] / update_denom,
            "policy/old_log_prob": totals["old_log_prob"] / update_denom,
            "policy/new_log_prob": totals["new_log_prob"] / update_denom,
            "policy/step_ratio": totals["step_ratio"] / update_denom,
            "policy/step_ratio_min": totals["step_ratio_min"] if totals["step_ratio_min"] != float("inf") else 0.0,
            "policy/step_ratio_max": totals["step_ratio_max"],
            "policy/step_clip_frac": totals["step_clip_frac"] / update_denom,
            "policy/step_logprob_delta_abs": totals["step_logprob_delta_abs"] / update_denom,
            "policy/step_kl_loss": totals["step_kl_loss"] / update_denom,
            "policy/post_ratio": post_ratio,
            "policy/post_ratio_min": post_ratio_min,
            "policy/post_ratio_max": post_ratio_max,
            "policy/post_clip_frac": post_clip_frac,
            "policy/post_logprob_delta_abs": post_logprob_delta_abs,
            "policy/post_kl_loss": post_kl_loss,
            "policy/grad_norm": totals["grad_norm"] / grad_denom,
            "policy/action_delta": float(action_delta.item()),
            "policy/param_rms_delta": float(param_rms_delta.item()),
            "policy/sample_count": float(sample_count),
            "policy/advantage_abs_mean": advantage_abs_mean,
            "policy/effective_mini_batch_size": float(mini_batch_size),
            "policy/micro_batch_size": float(self._policy_micro_batch_size(mini_batch_size)),
            "policy/micro_batches": float(micro_batch_count),
            "policy/optimizer_steps": float(grad_update_count),
            "policy/lr": current_policy_lr,
            "policy/critic_lr": current_critic_lr,
            "policy/sde_train_steps": float(train_step_indices.numel()),
        }

    def _update_adaptive_learning_rate(self, observed_kl: float) -> None:
        desired_kl = float(self.cfg.desired_kl)
        if desired_kl <= 0.0:
            return
        kl_value = float(observed_kl.item() if torch.is_tensor(observed_kl) else observed_kl)
        if not math.isfinite(kl_value) or kl_value <= 0.0:
            return
        if kl_value > 2.0 * desired_kl:
            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
        elif kl_value < 0.5 * desired_kl:
            self.learning_rate = min(float(self.cfg.policy_lr), self.learning_rate * 1.5)
        self.optimizer.param_groups[0]["lr"] = self.learning_rate

    def _policy_mini_batch_size(self, sample_count: int) -> int:
        if self.cfg.mini_batch_size > 0:
            return max(1, min(sample_count, int(self.cfg.mini_batch_size)))
        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        return max(1, sample_count // num_mini_batches)

    def _policy_micro_batch_size(self, batch_size: int) -> int:
        if self.cfg.micro_batch_size <= 0:
            return max(1, batch_size)
        return max(1, min(batch_size, int(self.cfg.micro_batch_size)))

    def _training_rollout_horizon(self) -> int:
        return max(1, self.cfg.horizon * self._chunks_per_grpo_update())

    def _rollout_segments_per_update(self) -> int:
        return max(1, int(getattr(self.cfg, "rollout_segments_per_update", 1)))

    def _chunks_per_grpo_update(self) -> int:
        return max(1, int(self.cfg.chunks_per_rollout) * self._rollout_segments_per_update())

    def _training_anchor_phases(self) -> torch.Tensor:
        sample_phase_indices = getattr(self.env, "sample_phase_indices", None)
        if callable(sample_phase_indices):
            anchors = sample_phase_indices(self.num_grpo_groups, self._training_rollout_horizon())
            return anchors.to(device=self.env.device, dtype=torch.long)
        # Fallback for harnessed envs that lack sample_phase_indices: use motion_start_phase.
        min_phase = int(getattr(self.env, "motion_start_phase", self.cfg.motion_start_phase))
        return torch.full((self.num_grpo_groups,), min_phase, dtype=torch.long, device=self.env.device)

    def _resample_group_starts(self) -> torch.Tensor:
        """Per-update random anchor sampling for every group, with full intra-group sync.

        Required by GRPO: all envs inside one group must start from an *identical* state so
        the group-relative advantage compares only the effect of SDE-noise variation. This
        helper (a) samples a fresh per-group anchor phase using the env's adaptive sampler,
        (b) resets every env to its group's anchor, and (c) replicates source state to the
        rest of the group's branches so that root_state / joint_pos / domain randomization
        match exactly inside each group.
        """
        generation_count = int(self.cfg.num_generations)
        phase_indices = self._training_anchor_phases()
        reset_phases = phase_indices.repeat_interleave(generation_count)
        env_ids = torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        self.env.reset_envs(env_ids, phase_indices=reset_phases)
        self._replicate_group_reset_state(self.num_grpo_groups, generation_count)
        return self.env.get_observation()

    def _reset_training_envs(self) -> torch.Tensor:
        phase_indices = self._training_anchor_phases()
        generation_count = int(self.cfg.num_generations)
        reset_phases = phase_indices.repeat_interleave(generation_count)
        self.env.reset(phase_indices=reset_phases)
        self._replicate_group_reset_state(self.num_grpo_groups, generation_count)
        return self.env.get_observation()

    def train(self) -> None:
        print("[INFO] Starting MixGRPO training", flush=True)
        print(f"[INFO] motion_file={self.cfg.motion_file}", flush=True)
        print(
            f"[INFO] obs_dim={self.env.observation_dim} "
            f"actor_type=mix_sde_ode "
            f"policy_horizon={self.cfg.horizon} "
            f"action_dim={self.cfg.action_dim} "
            f"single_action_mode=True "
            f"rollout_steps={self._chunks_per_grpo_update()} "
                f"reset_noise={self.cfg.reset_noise} interval_pushes={self.cfg.interval_pushes} "
                f"num_envs={self.cfg.num_envs} "
                f"chunks_per_rollout={self.cfg.chunks_per_rollout} "
                f"rollout_segments_per_update={self._rollout_segments_per_update()} "
                f"grpo_update_steps={self._chunks_per_grpo_update()} "
                f"tail_bootstrap_steps={int(getattr(self.cfg, 'tail_bootstrap_steps', 0))} "
                f"terminal_penalty={self.cfg.terminal_penalty} "
                f"num_generations={self.cfg.num_generations} "
                f"grpo_groups={self.num_grpo_groups} "
                f"init_noise_std={self.cfg.init_noise_std} "
                f"init_same_noise={self.cfg.init_same_noise} "
                f"eval_initial_noise={self.cfg.eval_initial_noise} "
                f"sde_eta={self.cfg.sde_eta} "
                f"flow_steps={self.cfg.flow_steps} "
                f"actor_hidden_dims={list(self.cfg.actor_hidden_dims)} "
                f"activation={self.cfg.activation} "
                f"action_squash_scale={self.cfg.action_squash_scale}",
            flush=True,
        )
        print(
            f"[INFO] policy_epochs={self.cfg.policy_epochs} "
            f"clip_range={self.cfg.clip_range} "
            f"adv_clip_max={self.cfg.adv_clip_max} "
                f"desired_kl={self.cfg.desired_kl} "
            f"gamma={self.cfg.discount_gamma} "
            f"gae_lambda={self.cfg.gae_lambda} "
            f"entropy_coef={self.cfg.entropy_coef} "
            f"value_loss_coef={self.cfg.value_loss_coef} "
            f"use_clipped_value_loss={self.cfg.use_clipped_value_loss} "
            f"num_mini_batches={self.cfg.num_mini_batches} "
            f"mini_batch_override={self.cfg.mini_batch_size} "
            f"micro_batch={self.cfg.micro_batch_size} "
            f"policy_lr={self.cfg.policy_lr} "
            f"critic_lr=0.0",
            flush=True,
        )
        print(
            "[INFO] critic_free=True actor_std_trainable=False "
            "exploration=four_step_sde_noise first_generation_zero_sde_noise=False",
            flush=True,
        )
        if self.checkpoint_dir is not None:
            print(f"[INFO] checkpoint_dir={self.checkpoint_dir}", flush=True)
        if self.cfg.resume:
            print(f"[INFO] resumed_from={self.cfg.resume}", flush=True)

        for update_idx in range(self.start_update, self.cfg.max_updates + 1):
            if not self.simulation_app.is_running():
                break

            t0 = time.perf_counter()
            current_obs = self._resample_group_starts()
            self.current_observation = current_obs
            self._debug_probe_update = update_idx
            self._debug_probe_sample_printed = False
            group_data = self._collect_rollout(current_obs)
            self.current_observation = group_data["next_observation"]
            self._debug_print_collection(group_data)
            collect_time = time.perf_counter() - t0
            env_count = self.num_grpo_groups
            rollout_branch_count = int(self.cfg.num_generations)
            chunks = int(group_data["chunk_rewards"].shape[-1])
            train_chunk_rewards = group_data.get("train_chunk_rewards", group_data["chunk_rewards"])
            returns_per_step, _ppo_advantages_per_step = self._compute_gae_returns(
                train_chunk_rewards.reshape(self.cfg.num_envs, chunks),
                group_data["dones"].reshape(self.cfg.num_envs, chunks),
                group_data["values"].reshape(self.cfg.num_envs, chunks),
                group_data["last_values"],
                timeouts=group_data["timeouts"].reshape(self.cfg.num_envs, chunks),
            )
            returns_per_chunk = returns_per_step.view(env_count, rollout_branch_count, chunks)
            # Per-chunk reward-to-go is the GRPO score signal. Survival is encouraged via a
            # *local* terminal_penalty subtracted at the death chunk only — RTG then folds it
            # back through gamma into earlier chunks, but it is no longer broadcast as a flat
            # global penalty to every chunk (which would let chunk-0's advantage encode the
            # branch's eventual death and pollute early credit assignment).
            live_chunks = group_data["valid_mask"].float().sum(dim=-1)
            early_stop_penalty = (
                (float(chunks) - live_chunks)
                / max(float(chunks), 1.0)
                * float(self.cfg.terminal_penalty)
            )
            sample_rewards = group_data["rewards"] - early_stop_penalty  # metric only
            train_chunk_rewards_g = train_chunk_rewards.view(env_count, rollout_branch_count, chunks)
            valid_mask_g = group_data["valid_mask"]
            valid_float = valid_mask_g.to(dtype=train_chunk_rewards_g.dtype)
            first_life_rewards = train_chunk_rewards_g * valid_float
            # Add a one-shot terminal penalty at the death chunk (the chunk indexed by
            # first_done_chunk for branches that actually died). Branches that survive the
            # whole rollout have first_done_chunk == chunks_per_rollout and contribute no
            # penalty.
            first_done_chunk_g = group_data["first_done_chunk"].to(device=first_life_rewards.device)
            died_mask = first_done_chunk_g < chunks
            if bool(died_mask.any()):
                death_idx = first_done_chunk_g.clamp(max=chunks - 1)
                death_one_hot = torch.nn.functional.one_hot(death_idx, num_classes=chunks).to(
                    dtype=first_life_rewards.dtype,
                )
                first_life_rewards = first_life_rewards - (
                    died_mask.to(dtype=first_life_rewards.dtype).unsqueeze(-1)
                    * death_one_hot
                    * float(self.cfg.terminal_penalty)
                )
            reward_to_go = torch.zeros_like(first_life_rewards)
            # Seed running_return with the per-env tail bootstrap value for branches
            # still alive at the end of the rollout. last_values is already zero for
            # dead branches inside _compute_tail_bootstrap; mask again for safety.
            tail_bootstrap_g = group_data["last_values"].view(env_count, rollout_branch_count).to(
                device=first_life_rewards.device,
                dtype=first_life_rewards.dtype,
            )
            alive_at_end_g = (~died_mask).to(dtype=first_life_rewards.dtype)
            running_return = tail_bootstrap_g * alive_at_end_g
            gamma = float(self.cfg.discount_gamma)
            for chunk_idx in reversed(range(chunks)):
                running_return = first_life_rewards[:, :, chunk_idx] + gamma * running_return
                reward_to_go[:, :, chunk_idx] = running_return
                running_return = running_return * valid_float[:, :, chunk_idx]
            chunk_scores = reward_to_go
            # Use only branches that are still alive at this chunk to compute the group
            # baseline. Including dead branches (which carry zero RTG after their death
            # chunk) progressively pushes the mean toward zero and inflates std for the
            # surviving branches, drowning their advantage in noise where credit matters
            # most (recovery / stabilization actions late in the rollout).
            group_baseline_mask = valid_mask_g
            advantages_by_chunk = self._compute_group_relative_advantages(
                chunk_scores.permute(0, 2, 1).reshape(env_count * chunks, rollout_branch_count),
                group_baseline_mask.permute(0, 2, 1).reshape(env_count * chunks, rollout_branch_count),
            )
            advantages_per_chunk = advantages_by_chunk.view(env_count, chunks, rollout_branch_count).permute(0, 2, 1)
            valid_flat = group_data["valid_mask"].reshape(env_count * rollout_branch_count * chunks)
            update_flat = valid_flat

            obs_flat = group_data["obs"].reshape(env_count * rollout_branch_count * chunks, -1)[update_flat]
            critic_obs_flat = group_data["critic_obs"].reshape(env_count * rollout_branch_count * chunks, -1)[
                update_flat
            ]

            grpo_adv_flat = advantages_per_chunk.reshape(env_count * rollout_branch_count * chunks)[valid_flat]
            adv_flat = advantages_per_chunk.reshape(env_count * rollout_branch_count * chunks)[update_flat]
            self._debug_print_advantages(advantages_per_chunk, adv_flat, update_flat, group_data)
            t1 = time.perf_counter()
            update_metrics = self._policy_update(
                obs_flat,
                critic_obs_flat,
                group_data["actions"].reshape(env_count * rollout_branch_count * chunks, self.chunk_dim)[update_flat],
                group_data["latents"].reshape(
                    env_count * rollout_branch_count * chunks,
                    self.cfg.flow_steps + 1,
                    self.chunk_dim,
                )[update_flat],
                group_data["old_log_probs"].reshape(
                    env_count * rollout_branch_count * chunks,
                    group_data["old_log_probs"].shape[-1],
                )[update_flat],
                group_data["train_step_indices"],
                adv_flat,
                returns_per_chunk.reshape(env_count * rollout_branch_count * chunks)[update_flat],
                group_data["values"].reshape(env_count * rollout_branch_count * chunks)[update_flat],
            )
            update_time = time.perf_counter() - t1
            metrics = self._build_metrics(group_data, advantages_per_chunk, update_metrics, collect_time, update_time)
            if grpo_adv_flat.numel() > 0:
                metrics["group/grpo_advantage_abs_mean"] = float(grpo_adv_flat.abs().mean().item())
            else:
                metrics["group/grpo_advantage_abs_mean"] = 0.0
            metrics["group/grpo_score_mean"] = float(sample_rewards.mean().item())
            metrics["group/early_stop_penalty_mean"] = float(early_stop_penalty.mean().item())
            valid_reward_to_go = reward_to_go[valid_mask_g]
            metrics["group/reward_to_go_mean"] = (
                float(valid_reward_to_go.mean().item()) if valid_reward_to_go.numel() > 0 else 0.0
            )

            should_log = update_idx % self.cfg.log_every == 0
            if should_log:
                self._log_update(update_idx, metrics)

            # Validation can be much slower than one training update on 4096 envs, so
            # print the training metrics first and then announce the validation pass.
            if self.cfg.validation_every > 0 and update_idx % self.cfg.validation_every == 0:
                fixed_seed = self.cfg.validation_fixed_seed if self.cfg.validation_fixed_seed >= 0 else None
                print(
                    f"[VALIDATION_START] update={update_idx} "
                    f"max_steps={self._validation_max_steps()} "
                    f"envs={self.cfg.num_envs} "
                    f"fixed_seed={fixed_seed if fixed_seed is not None else 'disabled'}",
                    flush=True,
                )
                validation_t0 = time.perf_counter()
                metrics.update(self.run_validation_rollout())
                if fixed_seed is not None:
                    fixed_metrics = self.run_validation_rollout(fixed_seed=fixed_seed)
                    for key, value in fixed_metrics.items():
                        metrics[key.replace("validation/", "val_fixed/")] = value
                metrics["timing/validation_s"] = time.perf_counter() - validation_t0
                print(
                    f"[VALIDATION_DONE] update={update_idx} "
                    f"time={metrics['timing/validation_s']:.3f}s",
                    flush=True,
                )
                if should_log:
                    self._log_validation_metrics(metrics)

            if self.checkpoint_dir is not None and (
                update_idx == self.cfg.max_updates
                or (self.cfg.save_every > 0 and update_idx % self.cfg.save_every == 0)
            ):
                self._save_checkpoint(update_idx, metrics)
            if self._target_validation_reached(metrics):
                if self.checkpoint_dir is not None:
                    self._save_checkpoint(update_idx, metrics, filename=self.cfg.success_checkpoint_name)
                print(
                    f"[SUCCESS] validation reached {self.cfg.target_validation_steps} steps; "
                    f"saved {self.cfg.success_checkpoint_name}",
                    flush=True,
                )
                break

        self.current_observation = self._reset_training_envs()
        print("[INFO] Training finished.", flush=True)

    def _build_metrics(
        self,
        group_data: dict[str, torch.Tensor],
        advantages: torch.Tensor,
        update_metrics: dict[str, float],
        collect_time: float,
        update_time: float,
    ) -> dict[str, float]:
        group_rewards = group_data["rewards"]
        raw_group_rewards = group_data.get("raw_rewards", group_rewards)
        score_group_rewards = group_data.get("score_rewards")
        metric_chunk_return = group_data["metric_chunk_return"]
        metric_chunk_return_first = group_data.get("metric_chunk_return_first")
        metric_chunk_return_last = group_data.get("metric_chunk_return_last")
        metric_actions = group_data["metric_actions"]
        act_abs_tensor = metric_actions.abs()

        metric_actions_first = group_data.get("metric_actions_first")
        metric_actions_last = group_data.get("metric_actions_last")
        act_first_mean = float(metric_actions_first.abs().mean().item()) if metric_actions_first is not None else 0.0
        act_last_mean = float(metric_actions_last.abs().mean().item()) if metric_actions_last is not None else 0.0
        chunk_first_mean = (
            float(metric_chunk_return_first.mean().item())
            if metric_chunk_return_first is not None
            else float(metric_chunk_return.mean().item())
        )
        chunk_last_mean = (
            float(metric_chunk_return_last.mean().item())
            if metric_chunk_return_last is not None
            else float("nan")
        )

        valid_mask = group_data["valid_mask"]
        valid_advantages = advantages[valid_mask]
        act_abs = act_abs_tensor.mean(dim=(0, 1))
        final_latents = group_data["latents"][..., -1, :]
        valid_final_latents = final_latents[valid_mask]
        live_steps = valid_mask.float().sum(dim=-1) * self.cfg.horizon
        live_steps_flat = live_steps.flatten()
        first_done_chunk = group_data.get("first_done_chunk")
        first_done_phase = group_data.get("first_done_phase")
        first_done_anchor_pos = group_data.get("first_done_anchor_pos")
        first_done_anchor_ori = group_data.get("first_done_anchor_ori")
        first_done_ee_body = group_data.get("first_done_ee_body")
        first_done_timeout = group_data.get("first_done_timeout")
        collection_start_phases = group_data.get("collection_start_phases")
        failed_mask = first_done_phase >= 0 if first_done_phase is not None else None
        failure_phase_values = first_done_phase[failed_mask] if failed_mask is not None and bool(failed_mask.any()) else None
        failure_chunk_values = first_done_chunk[failed_mask] if failed_mask is not None and bool(failed_mask.any()) else None
        if (
            first_done_phase is not None
            and failed_mask is not None
            and collection_start_phases is not None
            and bool(failed_mask.any())
        ):
            if collection_start_phases.shape == first_done_phase.shape:
                start_phase_by_group = collection_start_phases
            else:
                start_phase_by_group = collection_start_phases.reshape_as(first_done_phase)
            failure_relative_phase_values = first_done_phase[failed_mask] - start_phase_by_group[failed_mask]
        else:
            failure_relative_phase_values = None
        total_live_steps = live_steps.sum().clamp(min=1.0)
        reward_per_live_step = group_rewards.sum() / total_live_steps
        raw_reward_per_live_step = raw_group_rewards.sum() / total_live_steps
        official_scale_reward = reward_per_live_step * self.cfg.max_episode_steps
        chunk_rewards_for_metrics = group_data.get("chunk_rewards", advantages)
        chunk_objective_means = chunk_rewards_for_metrics.mean(dim=(0, 1))
        chunk_raw_means = group_data.get("raw_chunk_rewards", chunk_rewards_for_metrics).mean(dim=(0, 1))
        chunk_count_for_metrics = int(chunk_objective_means.shape[0])
        mid_chunk_index = min(max(chunk_count_for_metrics // 2, 0), chunk_count_for_metrics - 1)
        metrics = {
            **update_metrics,
            "algo/name": "mixgrpo",
            "group/reward_mean": float(official_scale_reward.item()),
            "group/reward_raw_mean": float(raw_group_rewards.mean().item()),
            "group/reward_raw_std": float(raw_group_rewards.std().item()),
            "group/reward_raw_min": float(raw_group_rewards.min().item()),
            "group/reward_raw_max": float(raw_group_rewards.max().item()),
            "group/score_reward_mean": (
                float(score_group_rewards.mean().item()) if score_group_rewards is not None else float("nan")
            ),
            "group/score_reward_std": (
                float(score_group_rewards.std().item()) if score_group_rewards is not None else float("nan")
            ),
            "group/objective_reward_raw_mean": float(group_rewards.mean().item()),
            "group/objective_reward_raw_std": float(group_rewards.std().item()),
            "group/objective_reward_raw_min": float(group_rewards.min().item()),
            "group/objective_reward_raw_max": float(group_rewards.max().item()),
            "group/reward_std": float(
                (
                    group_rewards
                    / live_steps.clamp(min=1.0)
                    * self.cfg.max_episode_steps
                ).std().item()
            ),
            "group/reward_min": float(
                (
                    group_rewards
                    / live_steps.clamp(min=1.0)
                    * self.cfg.max_episode_steps
                ).min().item()
            ),
            "group/reward_max": float(
                (
                    group_rewards
                    / live_steps.clamp(min=1.0)
                    * self.cfg.max_episode_steps
                ).max().item()
            ),
            "group/advantage_abs_mean": (
                float(valid_advantages.abs().mean().item()) if valid_advantages.numel() > 0 else 0.0
            ),
            "rollout/valid_frac": float(valid_mask.float().mean().item()),
            "rollout/return_mean": float(group_rewards.mean().item()),
            "rollout/return_std": float(group_rewards.std().item()),
            "rollout/raw_return_mean": float(raw_group_rewards.mean().item()),
            "rollout/raw_return_std": float(raw_group_rewards.std().item()),
            "rollout/chunk_return_mean": float(metric_chunk_return.mean().item()),
            "rollout/chunk_return_std": float(metric_chunk_return.std().item()),
            "rollout/chunk_return_first_mean": chunk_first_mean,
            "rollout/chunk_return_last_mean": chunk_last_mean,
            "rollout/chunk_objective_first_mean": float(chunk_objective_means[0].item()),
            "rollout/chunk_objective_mid_mean": float(chunk_objective_means[mid_chunk_index].item()),
            "rollout/chunk_objective_last_mean": float(chunk_objective_means[-1].item()),
            "rollout/chunk_raw_first_mean": float(chunk_raw_means[0].item()),
            "rollout/chunk_raw_mid_mean": float(chunk_raw_means[mid_chunk_index].item()),
            "rollout/chunk_raw_last_mean": float(chunk_raw_means[-1].item()),
            "rollout/live_steps_mean": float(live_steps.mean().item()),
            "rollout/live_steps_min": float(live_steps_flat.min().item()),
            "rollout/live_steps_p50": float(torch.quantile(live_steps_flat, 0.50).item()),
            "rollout/live_steps_p95": float(torch.quantile(live_steps_flat, 0.95).item()),
            "rollout/live_steps_max": float(live_steps_flat.max().item()),
            "rollout/success_frac": (
                float((~failed_mask).float().mean().item()) if failed_mask is not None else 0.0
            ),
            "rollout/first_failure_chunk_mean": (
                float(failure_chunk_values.float().mean().item()) if failure_chunk_values is not None else float("nan")
            ),
            "rollout/first_failure_chunk_min": (
                float(failure_chunk_values.min().item()) if failure_chunk_values is not None else float("nan")
            ),
            "rollout/first_failure_chunk_max": (
                float(failure_chunk_values.max().item()) if failure_chunk_values is not None else float("nan")
            ),
            "rollout/first_failure_phase_mean": (
                float(failure_phase_values.float().mean().item()) if failure_phase_values is not None else float("nan")
            ),
            "rollout/first_failure_phase_min": (
                float(failure_phase_values.min().item()) if failure_phase_values is not None else float("nan")
            ),
            "rollout/first_failure_phase_max": (
                float(failure_phase_values.max().item()) if failure_phase_values is not None else float("nan")
            ),
            "rollout/first_failure_relative_phase_mean": (
                float(failure_relative_phase_values.float().mean().item())
                if failure_relative_phase_values is not None
                else float("nan")
            ),
            "rollout/first_failure_anchor_pos_frac": (
                float((first_done_anchor_pos & failed_mask).float().mean().item())
                if first_done_anchor_pos is not None and failed_mask is not None
                else 0.0
            ),
            "rollout/first_failure_anchor_ori_frac": (
                float((first_done_anchor_ori & failed_mask).float().mean().item())
                if first_done_anchor_ori is not None and failed_mask is not None
                else 0.0
            ),
            "rollout/first_failure_ee_body_frac": (
                float((first_done_ee_body & failed_mask).float().mean().item())
                if first_done_ee_body is not None and failed_mask is not None
                else 0.0
            ),
            "rollout/first_failure_timeout_frac": (
                float((first_done_timeout & failed_mask).float().mean().item())
                if first_done_timeout is not None and failed_mask is not None
                else 0.0
            ),
            "rollout/reward_per_live_step": float(reward_per_live_step.item()),
            "rollout/reward_per_live_second": float((reward_per_live_step / self.env.dt).item()),
            "rollout/raw_reward_per_live_step": float(raw_reward_per_live_step.item()),
            "rollout/max_episode_return_projection": float(official_scale_reward.item()),
            "phase/start_mean": (
                float(collection_start_phases.float().mean().item())
                if collection_start_phases is not None
                else float("nan")
            ),
            "phase/start_min": (
                float(collection_start_phases.min().item()) if collection_start_phases is not None else float("nan")
            ),
            "phase/start_max": (
                float(collection_start_phases.max().item()) if collection_start_phases is not None else float("nan")
            ),
            "timing/collect_s": collect_time,
            "timing/update_s": update_time,
            "act/abs_mean": float(act_abs_tensor.mean().item()),
            "act/abs_max": float(act_abs_tensor.max().item()),
            "act/abs_max_all": float(group_data.get("metric_action_abs_max_all", 0.0)),
            "act/abs_p95": float(torch.quantile(act_abs_tensor.flatten(), 0.95).item()),
            "act/abs_p99": float(torch.quantile(act_abs_tensor.flatten(), 0.99).item()),
            "act/legs_abs": float(act_abs[[0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]].mean().item()),
            "act/waist_abs": float(act_abs[[2, 5, 8]].mean().item()),
            "act/arms_abs": float(act_abs[[11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]].mean().item()),
            "latent/final_abs_mean": (
                float(valid_final_latents.abs().mean().item()) if valid_final_latents.numel() > 0 else 0.0
            ),
            "latent/final_abs_max": (
                float(valid_final_latents.abs().max().item()) if valid_final_latents.numel() > 0 else 0.0
            ),
            "act/l_shoulder_pitch": float(act_abs[11].item()),
            "act/r_shoulder_pitch": float(act_abs[12].item()),
            "act/l_shoulder_roll": float(act_abs[15].item()),
            "act/r_shoulder_roll": float(act_abs[16].item()),
            "act/l_shoulder_yaw": float(act_abs[19].item()),
            "act/r_shoulder_yaw": float(act_abs[20].item()),
            "act/l_elbow": float(act_abs[21].item()),
            "act/r_elbow": float(act_abs[22].item()),
            "act/l_wrist_roll": float(act_abs[23].item()),
            "act/r_wrist_roll": float(act_abs[24].item()),
            "act/l_wrist_pitch": float(act_abs[25].item()),
            "act/r_wrist_pitch": float(act_abs[26].item()),
            "act/l_wrist_yaw": float(act_abs[27].item()),
            "act/r_wrist_yaw": float(act_abs[28].item()),
            "act/first_abs_mean": act_first_mean,
            "act/last_abs_mean": act_last_mean,
        }
        done_union: dict[str, torch.Tensor] = {}
        metric_infos_list = group_data["metric_infos_list"]
        info_count = max(len(metric_infos_list), 1)
        for step_info in metric_infos_list:
            for key, value in step_info["reward_terms"].items():
                metric_key = f"reward/{key}_mean"
                metrics[metric_key] = metrics.get(metric_key, 0.0) + float(value.mean().item()) / info_count
            for key, value in step_info["done_terms"].items():
                if key not in done_union:
                    done_union[key] = value.bool().clone()
                else:
                    done_union[key] |= value.bool()
        for key, union_mask in done_union.items():
            metrics[f"done/{key}_frac"] = float(union_mask.float().mean().item())
        rollout_info_items = group_data.get("metric_rollout_info_items", [])
        rollout_done_union: dict[str, torch.Tensor] = {}
        rollout_reward_sums: dict[str, float] = {}
        rollout_weight_sum = 0.0
        for step_info, valid_mask_for_step in rollout_info_items:
            valid_mask_f = valid_mask_for_step.float()
            valid_weight = float(valid_mask_f.sum().item())
            if valid_weight <= 0.0:
                continue
            rollout_weight_sum += valid_weight
            for key, value in step_info["reward_terms"].items():
                rollout_reward_sums[key] = rollout_reward_sums.get(key, 0.0) + float((value * valid_mask_f).sum().item())
            for key, value in step_info["done_terms"].items():
                masked_done = value.bool() & valid_mask_for_step.bool()
                if key not in rollout_done_union:
                    rollout_done_union[key] = masked_done.clone()
                else:
                    rollout_done_union[key] |= masked_done
        if rollout_weight_sum > 0.0:
            for key, value_sum in rollout_reward_sums.items():
                metrics[f"reward_rollout/{key}_mean"] = value_sum / rollout_weight_sum
            for key, union_mask in rollout_done_union.items():
                metrics[f"done_rollout/{key}_frac"] = float(union_mask.float().mean().item())
        train_reward_buffer = getattr(self, "_train_reward_buffer", ())
        train_length_buffer = getattr(self, "_train_length_buffer", ())
        if train_reward_buffer:
            metrics["train/mean_reward"] = float(sum(train_reward_buffer) / len(train_reward_buffer))
            metrics["train/mean_episode_length"] = float(sum(train_length_buffer) / len(train_length_buffer))
        else:
            metrics["train/mean_reward"] = float("nan")
            metrics["train/mean_episode_length"] = float("nan")
        metrics["train/recent_episode_count"] = float(len(train_reward_buffer))
        metrics["train/completed_episodes"] = float(getattr(self, "_train_completed_episodes", 0))
        reward_weights = {
            "joint_acc": -2.5e-7,
            "joint_torque": -1.0e-5,
            "action_rate": -1.0e-1,
            "joint_limit": -10.0,
            "anchor_pos_reward": 2.0,
            "anchor_ori_reward": 2.0,
            "body_pos_reward": 1.0,
            "body_ori_reward": 1.0,
            "body_lin_vel_reward": 1.0,
            "body_ang_vel_reward": 1.0,
            "undesired_contacts": -0.1,
        }
        weighted_positive = 0.0
        weighted_penalty = 0.0
        for reward_name, weight in reward_weights.items():
            raw_key = f"reward/{reward_name}_mean"
            if raw_key not in metrics:
                continue
            contribution = weight * metrics[raw_key] * self.env.dt
            metrics[f"reward_weighted/{reward_name}"] = contribution
            if contribution >= 0.0:
                weighted_positive += contribution
            else:
                weighted_penalty += contribution
        metrics["reward_weighted/positive"] = weighted_positive
        metrics["reward_weighted/penalty"] = weighted_penalty
        metrics["reward_weighted/total"] = weighted_positive + weighted_penalty
        last_values = group_data.get("last_values")
        if last_values is not None and last_values.numel() > 0:
            metrics["group/tail_return_mean"] = float(last_values.mean().item())
            metrics["group/tail_return_alive_frac"] = float(
                (last_values != 0.0).to(dtype=torch.float32).mean().item()
            )
        return metrics
