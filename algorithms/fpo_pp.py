"""FPO++ (Flow Policy Optimization++) algorithm plugin -- official-aligned single-step actor.

Faithful port of the amazon-far/fpo-control robot implementation, specifically the G1
whole-body motion-tracking configuration (`G1FlatMotionTrackingFlowPPORunnerCfg`):

  PPO single-step rollout  +  single-step Flow actor  +  CFM-loss ratio  +  ASPO trust region
  +  adaptive-KL learning rate.

Differences from the earlier draft (the root causes this rewrite fixes):

  * The actor has NO tanh squash. The executed action is LINEAR in the flow endpoint,
    a = actor_scale * x_t (+ a small `action_perturb_std` Gaussian during training). The CFM
    loss -- and therefore the FPO ratio rho_i = exp(l_old_i - l_new_i) -- is computed on the
    *executed* action `a` itself (scaled back by a / actor_scale). Previously the CFM loss
    lived on the pre-tanh latent `z` while the advantage belonged to a = squash(z); since the
    FPO ratio is an approximate (not true) likelihood ratio, the tanh Jacobian does not cancel,
    mis-aligning the gradient. See networks/fpo_actor.py.
  * num_steps_per_env = 48 (official tracking) so a rollout spans past the average death horizon
    and the per-step GAE (gamma=0.99) carries the death signal (no explicit terminal penalty).
  * Adaptive learning rate driven by the flow-endpoint KL drift
    kl = mean((x1_pred_new - x1_pred_old)^2), targeting desired_kl, exactly as the official code.
  * Official update knobs: ASPO, symmetric CFM-loss clamp, negative-advantage CFM clamp,
    straight-through (STE) clamp on the log-ratio, symmetric advantage clamp, UNCLIPPED value loss.

The flow time convention follows the official code: t=1 is noise, t=0 is the action.

`fpo_pp_ratio`, `aspo_objective`, `clamp_ste` are module-level pure functions so the FPO math
is unit-testable without IsaacLab; the flow + CFM math lives in `networks/fpo_actor.FPOActor`.
"""
from __future__ import annotations

from collections import deque

import torch
from torch import nn

from algorithms.base import Algorithm
from core.logging import log_shared_tracking, log_shared_update_diagnostics
from networks.fpo_actor import FPOActor
from networks.mlp_actor_critic import Critic, EmpiricalNormalization


# ---------------------------------------------------------------------------- FPO objective math
def clamp_ste(x: torch.Tensor, *, min: float | None = None, max: float | None = None) -> torch.Tensor:
    """Straight-through clamp: forward uses the clamped value, backward passes identity gradient.

    Matches the official `clamp_ste` used on the log-ratio: clamping bounds the *value* fed to
    exp() (numerical safety) without killing the gradient when a sample sits at the bound.
    """
    clamped = x.clamp(min=min, max=max)
    return x + (clamped - x).detach()


def fpo_pp_ratio(old_cfm: torch.Tensor, new_cfm: torch.Tensor, delta_clip: float) -> torch.Tensor:
    """Per-sample FPO++ ratio rho_i = exp(l_old_i - l_new_i) (paper Eq. 10).

    The difference is STE-clamped to <= delta_clip (the official `cfm_diff_clamp_max`) before
    exp() for numerical stability. delta_clip <= 0 disables the clamp.
    """
    diff = old_cfm - new_cfm
    if delta_clip > 0.0:
        diff = clamp_ste(diff, max=float(delta_clip))
    return torch.exp(diff)


def aspo_objective(ratio: torch.Tensor, advantage: torch.Tensor, clip: float) -> torch.Tensor:
    """Asymmetric SPO objective (paper Eq. 11/12), to be MAXIMIZED.

    advantage >= 0 -> PPO clip: min(r*A, clip(r, 1-e, 1+e)*A)
    advantage <  0 -> SPO:      r*A - |A|/(2e) * (r - 1)^2
    `advantage` broadcasts against `ratio` (e.g. ratio (B, M), advantage (B, 1)).
    """
    if clip <= 0.0:
        raise ValueError(f"ASPO/PPO clip must be > 0, got {clip}")
    ppo = torch.minimum(ratio * advantage, torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage)
    spo = ratio * advantage - advantage.abs() / (2.0 * clip) * (ratio - 1.0) ** 2
    return torch.where(advantage >= 0.0, ppo, spo)


class FPOPP(Algorithm):
    name = "fpo_pp"

    # ------------------------------------------------------------------ build
    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        if env.action_dim != cfg.action_dim:
            raise ValueError(f"Expected env action_dim {env.action_dim}, got {cfg.action_dim}")
        # FPO++ is a SINGLE-STEP actor-critic (one flow action per env step). The flow acts
        # directly in the env action space; there is no temporal action chunk. horizon != 1 is
        # rejected -- that was the earlier chunk-based draft.
        if int(cfg.horizon) != 1:
            raise ValueError(
                f"FPO++ requires horizon=1 (single-step flow actor), got horizon={cfg.horizon}. "
                "Set algo.horizon=1 in the config."
            )
        self.num_act = int(cfg.action_dim)
        self.horizon = 1
        self.num_steps_per_env = max(1, int(cfg.num_steps_per_env))
        self.actor_obs_dim = env.observation_dim
        self.critic_obs_dim = env.critic_observation_dim
        device = env.device

        self.flow_steps = max(1, int(cfg.flow_steps))
        self.actor = FPOActor(
            obs_dim=self.actor_obs_dim,
            action_dim=self.num_act,
            hidden_dims=tuple(cfg.actor_hidden_dims),
            activation=cfg.activation,
            actor_scale=float(cfg.actor_scale),
            mlp_output_scale=float(cfg.mlp_output_scale),
            timestep_embed_dim=int(cfg.timestep_embed_dim),
            cfm_loss_reduction=str(cfg.cfm_loss_reduction),
            sampling_steps=self.flow_steps,
            action_perturb_std=float(cfg.action_perturb_std),
            cfm_loss_t_inverse_cdf_beta=float(cfg.cfm_loss_t_inverse_cdf_beta),
        ).to(device)
        self.actor.train()
        self.critic = Critic(self.critic_obs_dim, tuple(cfg.actor_hidden_dims), cfg.activation).to(device)
        # chunk_dim == action_dim; kept for log/metric parity with the harness.
        self.chunk_dim = self.num_act

        self.empirical_normalization = bool(cfg.empirical_normalization)
        if self.empirical_normalization:
            self.actor_obs_normalizer: nn.Module = EmpiricalNormalization(self.actor_obs_dim, device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(self.critic_obs_dim, device)
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        # ONE AdamW object (so the harness checkpoints a single optimizer) but TWO param groups:
        #   group "actor"  -> adaptive-KL learning rate (moves at runtime)
        #   group "critic" -> fixed value_lr
        # Crucially the actor and critic gradients are CLIPPED SEPARATELY in update() (see below),
        # so the value head's large early gradient -- inflated by the terminal_penalty death credit
        # -- can no longer dominate a shared global grad-norm and squash the actor's tiny gradient.
        self.learning_rate = float(cfg.policy_lr)
        _value_lr = float(getattr(cfg, "value_lr", 0.0) or 0.0)
        self.critic_learning_rate = _value_lr if _value_lr > 0.0 else self.learning_rate
        self._optimizer = torch.optim.AdamW(
            [
                {"params": list(self.actor.parameters()), "lr": self.learning_rate, "name": "actor"},
                {"params": list(self.critic.parameters()), "lr": self.critic_learning_rate, "name": "critic"},
            ],
            betas=(0.9, 0.999),
            weight_decay=float(cfg.weight_decay),
        )

        self.num_mc = max(1, int(cfg.fpo_num_mc))
        # Official update knobs (G1 tracking values shown in comments).
        self.cfm_diff_clamp_max = float(cfg.fpo_delta_clip)        # STE clamp on log-ratio (3.0)
        self.cfm_loss_clamp = float(cfg.fpo_cfm_loss_clamp)        # symmetric CFM clamp (3.0)
        self.cfm_loss_clamp_neg_adv = bool(cfg.cfm_loss_clamp_neg_adv)          # True
        self.cfm_loss_clamp_neg_adv_max = float(cfg.cfm_loss_clamp_neg_adv_max)  # 20.0
        self.adv_clamp = float(cfg.fpo_adv_clamp)                 # symmetric advantage clamp (5.0)
        self.schedule = str(cfg.schedule)                        # "adaptive"
        self.desired_kl = float(cfg.desired_kl)                  # 1e-4
        self.trust_region_mode = str(cfg.trust_region_mode)      # "aspo"
        self.num_micro_batches = max(1, int(cfg.num_micro_batches))  # gradient-accum microbatches
        self.storage_action_noise_std = float(cfg.storage_action_noise_std)  # 0.0
        # Residual-innovation parametrization (the fix for the residual action space). The flow
        # generates an INNOVATION u_t; the residual sent to the env is a low-pass AR(1) filter
        #     r_t = residual_rho * r_{t-1} + residual_innov_scale * u_t .
        # The flow models pi(u_t | s_t), so the CFM loss / FPO ratio is computed on u_t (the
        # innovation), NOT on the executed residual r_t. r_{t-1} is the previous EXECUTED residual,
        # which the env stores in `last_action` (reset to 0 on episode reset, included in the obs,
        # and snapshotted/restored around validation) -- so no extra rollout state is needed.
        # residual_rho == 0 and residual_innov_scale == 1 recovers the old "flow emits the full
        # residual every step" behaviour.
        self.residual_rho = float(cfg.residual_innov_rho)
        self.residual_innov_scale = float(cfg.residual_innov_scale)
        # Survival objective: non-timeout deaths get -terminal_penalty in their reward so the
        # failure enters GAE directly (the crawl task has no env-level termination reward and a
        # rollout often does not see the death, so bootstrap truncation alone is too weak). This
        # mirrors MixGRPO's terminal_penalty. Set 0 to disable (pure official tracking behaviour).
        self.terminal_penalty = float(cfg.terminal_penalty)
        self.lr_min = 1e-5
        self.lr_max = 1e-2

        self.max_episode_steps = int(getattr(env.task_cfg, "max_episode_steps", -1))
        self._init_train_episode_stats()
        self._policy_module = nn.ModuleDict({"actor": self.actor, "critic": self.critic})

    @property
    def policy(self) -> nn.Module:
        return self._policy_module

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def extra_checkpoint_state(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "actor_obs_normalizer": self.actor_obs_normalizer.state_dict() if self.empirical_normalization else None,
            "critic_obs_normalizer": self.critic_obs_normalizer.state_dict() if self.empirical_normalization else None,
        }

    def load_extra_checkpoint_state(self, payload: dict) -> None:
        if not payload:
            return
        self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
        for group in self._optimizer.param_groups:
            if group.get("name") == "critic":
                continue  # critic LR is fixed (value_lr), only the actor group is adaptive
            group["lr"] = self.learning_rate
        if self.empirical_normalization:
            if payload.get("actor_obs_normalizer") is not None:
                self.actor_obs_normalizer.load_state_dict(payload["actor_obs_normalizer"])
            if payload.get("critic_obs_normalizer") is not None:
                self.critic_obs_normalizer.load_state_dict(payload["critic_obs_normalizer"])

    # ------------------------------------------------------------------ episode stats
    def _init_train_episode_stats(self) -> None:
        env = self.env
        self._train_reward_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._train_episode_length = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._train_reward_buffer: deque[float] = deque(maxlen=100)
        self._train_length_buffer: deque[float] = deque(maxlen=100)
        self._train_completed_episodes = 0

    def _record_episode_stats(self, rewards, dones) -> None:
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

    # ------------------------------------------------------------------ obs helpers
    def _norm_actor(self, obs, update=True):
        return self.actor_obs_normalizer(obs, update=update) if self.empirical_normalization else obs

    def _norm_critic(self, obs, update=True):
        return self.critic_obs_normalizer(obs, update=update) if self.empirical_normalization else obs

    # ------------------------------------------------------------------ resets
    def initial_reset(self) -> torch.Tensor:
        obs = self.env.reset()
        if bool(self.cfg.init_at_random_ep_len) and self.max_episode_steps > 0:
            self.env.episode_steps = torch.randint_like(self.env.episode_steps, high=int(self.max_episode_steps))
        self._obs = obs
        self._critic_obs = self.env.get_critic_observation()
        return obs

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        # FPO++ rolls continuously (auto_reset inside step); nothing to re-sample per update.
        return self._obs

    # ------------------------------------------------------------------ rollout
    def collect(self, current_obs: torch.Tensor) -> dict:
        """Single-step on-policy rollout (mirrors the official FPO.act / process_env_step).

        Per env step: evaluate the critic, sample one flow INNOVATION u = actor_scale*Euler(noise)
        (+ action_perturb), form the executed residual r = residual_rho*r_prev +
        residual_innov_scale*u, draw M Monte-Carlo (eps, t) pairs and cache the old CFM loss and
        flow-endpoint x1_pred (the ratio + KL references) of the INNOVATION u, then
        env.step(r, auto_reset=True). Timeouts use the value bootstrap; done truncation happens in GAE.
        """
        env = self.env
        device = env.device
        N = env.num_envs
        M = self.num_mc
        A = self.num_act
        T = self.num_steps_per_env
        gamma = float(self.cfg.discount_gamma)

        actor_obs_buf = torch.zeros(T, N, self.actor_obs_dim, device=device)
        critic_obs_buf = torch.zeros(T, N, self.critic_obs_dim, device=device)
        actions_buf = torch.zeros(T, N, A, device=device)       # innovations u_t (CFM coordinates)
        residual_buf = torch.zeros(T, N, A, device=device)      # executed residuals r_t (env input)
        cfm_t_buf = torch.zeros(T, N, M, 1, device=device)
        cfm_eps_buf = torch.zeros(T, N, M, A, device=device)
        old_cfm_buf = torch.zeros(T, N, M, device=device)
        x1_pred_buf = torch.zeros(T, N, M, A, device=device)
        values_buf = torch.zeros(T, N, 1, device=device)
        rewards_buf = torch.zeros(T, N, 1, device=device)
        dones_buf = torch.zeros(T, N, 1, dtype=torch.bool, device=device)

        obs = self._obs
        critic_obs = self._critic_obs
        done_terms_union: dict[str, torch.Tensor] = {}
        rollout_info_items: list[tuple] = []
        first_chunk_infos: list[dict] = []
        action_abs_max = 0.0

        # First-failure bookkeeping across the rollout (death phase / cause), for diagnostics and
        # adaptive motion sampling. Per env: the first termination in this rollout window.
        ever_done = torch.zeros(N, dtype=torch.bool, device=device)
        first_done_step = torch.full((N,), T, dtype=torch.long, device=device)
        first_done_phase = torch.full((N,), -1, dtype=torch.long, device=device)
        first_done_ee_body = torch.zeros(N, dtype=torch.bool, device=device)
        first_done_anchor_pos = torch.zeros(N, dtype=torch.bool, device=device)
        first_done_anchor_ori = torch.zeros(N, dtype=torch.bool, device=device)
        first_done_timeout = torch.zeros(N, dtype=torch.bool, device=device)
        start_phase = None
        _ps = getattr(env, "phase_steps", None)
        if torch.is_tensor(_ps):
            start_phase = _ps.detach().clone().long()

        with torch.no_grad():
            for t in range(T):
                actor_obs_n = self._norm_actor(obs)
                critic_obs_n = self._norm_critic(critic_obs)
                value = self.critic.evaluate(critic_obs_n).detach()

                # Flow generates the INNOVATION u_t (linear scale + action_perturb, both on u_t).
                innovation = self.actor.act(actor_obs_n).detach()  # (N, A)
                if self.storage_action_noise_std > 0.0:
                    innovation = innovation + self.storage_action_noise_std * torch.randn_like(innovation)
                # Executed residual via the AR(1) low-pass filter. r_{t-1} is the env's last
                # executed residual (`last_action`), already reset to 0 on episode reset.
                prev_residual = env.last_action
                residual = self.residual_rho * prev_residual + self.residual_innov_scale * innovation
                action_abs_max = max(action_abs_max, float(residual.abs().max().item()))

                # CFM loss / FPO references live in INNOVATION coordinates (the variable the flow
                # models): u_t -> r_t is a fixed affine map whose constant Jacobian cancels in the
                # old/new ratio.
                cfm_eps = torch.randn(N, M, A, device=device)
                cfm_t = self.actor.sample_cfm_timesteps(N, M, device=device)
                old_cfm, x1_pred, _ = self.actor.get_cfm_loss(actor_obs_n, innovation, cfm_eps, cfm_t)
                old_cfm = old_cfm.detach()
                x1_pred = x1_pred.detach()

                next_obs, reward, done, info = env.step(residual, auto_reset=True)
                next_critic_obs = env.get_critic_observation()

                done_b = done.bool()
                time_outs = info["done_terms"]["time_out"]
                time_outs_b = time_outs.bool()
                # Survival objective: a non-timeout death is a bad trajectory. Subtract
                # terminal_penalty from its reward so the death enters GAE directly. Timeouts are
                # truncations, NOT failures, and keep the value bootstrap below instead.
                failure = done_b & ~time_outs_b
                if self.terminal_penalty != 0.0 and bool(failure.any()):
                    reward = reward - self.terminal_penalty * failure.to(reward.dtype)

                # Record the first termination per env (phase + cause) for [FIRST_FAILURE] /
                # adaptive phase sampling.
                newly_done = (~ever_done) & done_b
                if bool(newly_done.any()):
                    ids = newly_done.nonzero(as_tuple=False).squeeze(-1)
                    first_done_step[ids] = t
                    first_done_timeout[ids] = time_outs_b[ids]
                    dterms = info["done_terms"]
                    if "ee_body_bad" in dterms:
                        first_done_ee_body[ids] = dterms["ee_body_bad"].bool()[ids]
                    if "anchor_pos_bad" in dterms:
                        first_done_anchor_pos[ids] = dterms["anchor_pos_bad"].bool()[ids]
                    if "anchor_ori_bad" in dterms:
                        first_done_anchor_ori[ids] = dterms["anchor_ori_bad"].bool()[ids]
                    tps = info.get("termination_phase_steps")
                    if torch.is_tensor(tps):
                        first_done_phase[ids] = tps.long().to(device)[ids]
                    ever_done[ids] = True

                # Timeout value bootstrap: add gamma * V(final_state) for envs that timed out.
                final_rewards = torch.zeros_like(reward)
                if bool(time_outs.any()) and "final_critic_observation" in info:
                    fco = self._norm_critic(info["final_critic_observation"], update=False)
                    final_values = self.critic.evaluate(fco).detach().squeeze(1)
                    final_rewards = final_rewards + gamma * final_values * time_outs.to(dtype=reward.dtype)

                actor_obs_buf[t] = actor_obs_n
                critic_obs_buf[t] = critic_obs_n
                actions_buf[t] = innovation
                residual_buf[t] = residual
                cfm_t_buf[t] = cfm_t
                cfm_eps_buf[t] = cfm_eps
                old_cfm_buf[t] = old_cfm
                x1_pred_buf[t] = x1_pred
                values_buf[t] = value
                rewards_buf[t] = (reward + final_rewards).view(-1, 1)
                dones_buf[t] = done.view(-1, 1)

                self._record_episode_stats(reward, done)
                if t == 0:
                    first_chunk_infos.append(info)
                # Every single-step transition is a valid sample (the action was applied before
                # auto_reset), so terminal transitions must count in the [DONE_ROLLOUT] diagnostics.
                valid = torch.ones_like(done, dtype=torch.bool)
                rollout_info_items.append((info, valid.detach()))
                for key, val in info["done_terms"].items():
                    b = val.bool()
                    done_terms_union[key] = b.clone() if key not in done_terms_union else (done_terms_union[key] | b)

                obs = next_obs
                critic_obs = next_critic_obs

            last_critic_obs = self._norm_critic(critic_obs, update=False)
            last_values = self.critic.evaluate(last_critic_obs).detach()
            returns, advantages = self._compute_gae(last_values, values_buf, dones_buf, rewards_buf, gamma)

        self._obs = obs
        self._critic_obs = critic_obs
        # Adaptive phase-sampler feedback: reinforce the motion phases the robot actually dies at
        # so the most failure-prone interval is oversampled next rollout (MixGRPO parity).
        self._update_adaptive_motion_sampler(first_done_phase, first_done_timeout, start_phase, T)
        return {
            "actor_obs": actor_obs_buf, "critic_obs": critic_obs_buf, "actions": actions_buf,
            "residuals": residual_buf,
            "cfm_t": cfm_t_buf, "cfm_eps": cfm_eps_buf, "old_cfm": old_cfm_buf, "x1_pred": x1_pred_buf,
            "values": values_buf, "returns": returns, "advantages": advantages,
            "rewards": rewards_buf, "dones": dones_buf,
            "done_terms_union": done_terms_union, "rollout_info_items": rollout_info_items,
            "first_chunk_infos": first_chunk_infos, "action_abs_max": action_abs_max,
            "first_done_step": first_done_step, "first_done_phase": first_done_phase,
            "first_done_ee_body": first_done_ee_body, "first_done_anchor_pos": first_done_anchor_pos,
            "first_done_anchor_ori": first_done_anchor_ori, "first_done_timeout": first_done_timeout,
            "next_observation": obs,
        }

    def _update_adaptive_motion_sampler(self, first_done_phase, first_done_timeout, start_phase, rollout_steps) -> None:
        env = self.env
        update_sampler = getattr(env, "update_adaptive_motion_statistics", None)
        if not callable(update_sampler):
            return
        num_frames = int(getattr(getattr(env, "motion", None), "num_frames", 0) or 0)
        if num_frames <= 0 or start_phase is None:
            return
        died = first_done_phase >= 0
        sampler_failed = died & (~first_done_timeout)
        death_phase = first_done_phase.clamp(min=0, max=num_frames - 1)
        survivor_phase = (start_phase + int(rollout_steps)).clamp(min=0, max=num_frames - 1)
        sampler_phases = torch.where(died, death_phase, survivor_phase)
        update_sampler(sampler_phases, sampler_failed, rollout_steps=int(rollout_steps))

    def _compute_gae(self, last_values, values, dones, rewards, gamma):
        lam = float(self.cfg.gae_lambda)
        advantage = torch.zeros_like(values[0])
        returns = torch.zeros_like(values)
        T = returns.shape[0]
        for step in reversed(range(T)):
            next_values = last_values if step == T - 1 else values[step + 1]
            next_nonterminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_nonterminal * gamma * next_values - values[step]
            advantage = delta + next_nonterminal * gamma * lam * advantage
            returns[step] = advantage + values[step]
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    # ------------------------------------------------------------------ update
    def update(self, rollout: dict, collect_time: float) -> dict:
        import time as _time
        device = self.env.device
        T, N = rollout["actions"].shape[0], rollout["actions"].shape[1]
        B = T * N
        M = self.num_mc
        A = self.num_act

        actor_obs = rollout["actor_obs"].reshape(B, self.actor_obs_dim)
        critic_obs = rollout["critic_obs"].reshape(B, self.critic_obs_dim)
        actions = rollout["actions"].reshape(B, A)  # innovations u_t -- the CFM/ratio variable
        cfm_t = rollout["cfm_t"].reshape(B, M, 1)
        cfm_eps = rollout["cfm_eps"].reshape(B, M, A)
        old_cfm = rollout["old_cfm"].reshape(B, M)
        old_x1_pred = rollout["x1_pred"].reshape(B, M, A)
        returns = rollout["returns"].reshape(B, 1)
        advantages = rollout["advantages"].reshape(B, 1)

        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        mini_batch_size = max(1, B // num_mini_batches)
        epochs = int(self.cfg.num_learning_epochs)
        clip = float(self.cfg.clip_range)
        value_coef = float(self.cfg.value_loss_coef)

        totals = {
            "actor_loss": 0.0, "value_loss": 0.0, "ratio": 0.0, "ratio_min": float("inf"),
            "ratio_max": 0.0, "clip_frac": 0.0, "cfm_new": 0.0, "cfm_old": 0.0, "grad_norm": 0.0,
            "grad_norm_critic": 0.0, "kl": 0.0,
        }
        num_updates = 0

        # Probe greedy (zero-sampling) action drift across the whole update.
        probe_count = min(128, N)
        with torch.no_grad():
            probe_obs_n = rollout["actor_obs"][0, :probe_count]
            probe_before = self.actor.act_inference(probe_obs_n, eval_mode="zero")
            params_before = [p.detach().clone() for p in self.actor.parameters()]

        t1 = _time.perf_counter()
        for _ in range(epochs):
            perm = torch.randperm(B, device=device)
            for mb in range(num_mini_batches):
                idx = perm[mb * mini_batch_size:(mb + 1) * mini_batch_size]
                mb_size = idx.numel()
                if mb_size == 0:
                    continue
                # Symmetric advantage clamp (official advantage_clamp), per logical minibatch.
                mb_adv_full = advantages[idx].clamp(-self.adv_clamp, self.adv_clamp)

                self._optimizer.zero_grad(set_to_none=True)
                micro_chunks = torch.chunk(torch.arange(mb_size, device=device), self.num_micro_batches)
                agg = {k: 0.0 for k in ("actor_loss", "value_loss", "ratio", "clip_frac", "cfm_new", "cfm_old", "kl")}
                agg_ratio_min = float("inf")
                agg_ratio_max = 0.0
                for sub in micro_chunks:
                    if sub.numel() == 0:
                        continue
                    weight = float(sub.numel()) / float(mb_size)
                    li = idx[sub]
                    new_cfm, x1_pred, _ = self.actor.get_cfm_loss(
                        actor_obs[li], actions[li], cfm_eps[li], cfm_t[li]
                    )
                    value = self.critic.evaluate(critic_obs[li])
                    mb_adv = mb_adv_full[sub]
                    mb_old_cfm = old_cfm[li]

                    # Symmetric CFM-loss clamp (numerical safety on both old and new).
                    if self.cfm_loss_clamp > 0.0:
                        mb_old_cfm = mb_old_cfm.clamp(max=self.cfm_loss_clamp)
                        new_cfm = new_cfm.clamp(max=self.cfm_loss_clamp)
                    # Negative-advantage CFM clamp: cap the new CFM where the action is bad,
                    # preventing extreme ratios when the policy aggressively avoids it.
                    if self.cfm_loss_clamp_neg_adv:
                        new_cfm = torch.where(
                            mb_adv < 0, new_cfm.clamp(max=self.cfm_loss_clamp_neg_adv_max), new_cfm
                        )

                    ratio = fpo_pp_ratio(mb_old_cfm, new_cfm, self.cfm_diff_clamp_max)  # (mb, M)
                    surrogate = aspo_objective(ratio, mb_adv, clip)                     # (mb, M)
                    actor_loss = -surrogate.mean()

                    # Unclipped value loss (official use_clipped_value_loss=False).
                    value_loss = (value - returns[li]).pow(2).mean()

                    loss = actor_loss + value_coef * value_loss
                    (loss * weight).backward()

                    with torch.no_grad():
                        agg["actor_loss"] += float(actor_loss.item()) * weight
                        agg["value_loss"] += float(value_loss.item()) * weight
                        agg["ratio"] += float(ratio.mean().item()) * weight
                        agg["clip_frac"] += float((torch.abs(ratio - 1.0) > clip).float().mean().item()) * weight
                        agg["cfm_new"] += float(new_cfm.mean().item()) * weight
                        agg["cfm_old"] += float(mb_old_cfm.mean().item()) * weight
                        agg_ratio_min = min(agg_ratio_min, float(ratio.min().item()))
                        agg_ratio_max = max(agg_ratio_max, float(ratio.max().item()))
                        # KL drift of the flow endpoint (official adaptive-LR signal).
                        kl_micro = ((x1_pred.detach() - old_x1_pred[li]) ** 2).mean()
                        agg["kl"] += float(kl_micro.item()) * weight

                # Adaptive learning-rate schedule from the aggregated KL (official rule). Only the
                # ACTOR group's LR is adapted; the critic keeps its fixed value_lr.
                if self.schedule == "adaptive":
                    kl_mean = agg["kl"]
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(self.lr_min, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(self.lr_max, self.learning_rate * 1.5)
                    for group in self._optimizer.param_groups:
                        if group.get("name") != "critic":
                            group["lr"] = self.learning_rate

                # Clip actor and critic gradients SEPARATELY so the value head's large gradient
                # cannot rescale (squash) the actor's gradient through a shared global norm.
                grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                grad_norm_critic = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self._optimizer.step()

                totals["actor_loss"] += agg["actor_loss"]
                totals["value_loss"] += agg["value_loss"]
                totals["ratio"] += agg["ratio"]
                totals["ratio_min"] = min(totals["ratio_min"], agg_ratio_min)
                totals["ratio_max"] = max(totals["ratio_max"], agg_ratio_max)
                totals["clip_frac"] += agg["clip_frac"]
                totals["cfm_new"] += agg["cfm_new"]
                totals["cfm_old"] += agg["cfm_old"]
                totals["kl"] += agg["kl"]
                totals["grad_norm"] += float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                totals["grad_norm_critic"] += float(grad_norm_critic.item() if torch.is_tensor(grad_norm_critic) else grad_norm_critic)
                num_updates += 1

        update_time = _time.perf_counter() - t1
        denom = max(num_updates, 1)
        with torch.no_grad():
            probe_after = self.actor.act_inference(probe_obs_n, eval_mode="zero")
            action_delta = float(torch.mean(torch.abs(probe_after - probe_before)).item())
            param_delta_sq = torch.zeros((), device=device)
            param_count = 0
            for p, before in zip(self.actor.parameters(), params_before, strict=True):
                d = p.detach() - before
                param_delta_sq = param_delta_sq + torch.sum(d * d)
                param_count += d.numel()
            param_rms_delta = float(torch.sqrt(param_delta_sq / max(param_count, 1)).item())

        agg_out = {
            "actor_loss": totals["actor_loss"] / denom,
            "value_loss": totals["value_loss"] / denom,
            "ratio": totals["ratio"] / denom,
            "ratio_min": totals["ratio_min"] if totals["ratio_min"] != float("inf") else 0.0,
            "ratio_max": totals["ratio_max"],
            "clip_frac": totals["clip_frac"] / denom,
            "cfm_new": totals["cfm_new"] / denom,
            "cfm_old": totals["cfm_old"] / denom,
            "grad_norm": totals["grad_norm"] / denom,
            "grad_norm_critic": totals["grad_norm_critic"] / denom,
            "kl": totals["kl"] / denom,
            "action_delta": action_delta,
            "param_rms_delta": param_rms_delta,
            "mini_batch_size": float(mini_batch_size),
        }
        return self._build_metrics(rollout, agg_out, collect_time, update_time)

    # ------------------------------------------------------------------ inference
    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Zero-sampling inference. Returns (N, 1, action_dim): the validation harness indexes
        [:, frame, :] and re-queries the policy every step (horizon == 1).

        Applies the same residual-innovation filter as training: the flow produces the innovation
        u_t and the executed residual is r_t = residual_rho*r_{t-1} + residual_innov_scale*u_t.
        r_{t-1} is read from the env's last executed residual (`last_action`), which the validation
        harness advances each step and resets on episode reset, so the AR(1) state stays consistent
        without extra bookkeeping."""
        actor_obs_n = self._norm_actor(obs, update=False)
        innovation = self.actor.act_inference(actor_obs_n, eval_mode="zero")
        prev_residual = self.env.last_action
        residual = self.residual_rho * prev_residual + self.residual_innov_scale * innovation
        return residual.unsqueeze(1)

    # ------------------------------------------------------------------ metrics + logging
    def _add_first_failure_metrics(self, metrics: dict, rollout: dict) -> None:
        """Populate rollout/first_failure_* (the shared [FIRST_FAILURE] line). `chunk` == step
        index here (single-step rollout). Deaths that recorded no phase still count as failures."""
        T = self.num_steps_per_env
        fds = rollout.get("first_done_step")
        if fds is None:
            for k in ("first_failure_chunk_mean", "first_failure_chunk_min", "first_failure_chunk_max",
                      "first_failure_phase_mean", "first_failure_phase_min", "first_failure_phase_max"):
                metrics[f"rollout/{k}"] = float("nan")
            return
        died = fds < T
        timeout = rollout["first_done_timeout"]
        if bool(died.any()):
            steps = fds[died].float()
            metrics["rollout/first_failure_chunk_mean"] = float(steps.mean().item())
            metrics["rollout/first_failure_chunk_min"] = float(steps.min().item())
            metrics["rollout/first_failure_chunk_max"] = float(steps.max().item())
        else:
            for k in ("first_failure_chunk_mean", "first_failure_chunk_min", "first_failure_chunk_max"):
                metrics[f"rollout/{k}"] = float("nan")
        phase = rollout["first_done_phase"]
        valid_phase = phase[phase >= 0]
        if valid_phase.numel() > 0:
            metrics["rollout/first_failure_phase_mean"] = float(valid_phase.float().mean().item())
            metrics["rollout/first_failure_phase_min"] = float(valid_phase.min().item())
            metrics["rollout/first_failure_phase_max"] = float(valid_phase.max().item())
        else:
            for k in ("first_failure_phase_mean", "first_failure_phase_min", "first_failure_phase_max"):
                metrics[f"rollout/{k}"] = float("nan")
        metrics["rollout/first_failure_ee_body_frac"] = float((rollout["first_done_ee_body"] & died).float().mean().item())
        metrics["rollout/first_failure_anchor_pos_frac"] = float((rollout["first_done_anchor_pos"] & died).float().mean().item())
        metrics["rollout/first_failure_anchor_ori_frac"] = float((rollout["first_done_anchor_ori"] & died).float().mean().item())
        metrics["rollout/first_failure_timeout_frac"] = float((timeout & died).float().mean().item())
        metrics["rollout/failure_frac"] = float((died & (~timeout)).float().mean().item())
        metrics["fpo/terminal_penalty"] = float(self.terminal_penalty)

    def _build_metrics(self, rollout, agg, collect_time, update_time) -> dict:
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        metrics = {
            "fpo/actor_loss": agg["actor_loss"],
            "fpo/value_loss": agg["value_loss"],
            "fpo/ratio": agg["ratio"],
            "fpo/ratio_min": agg["ratio_min"],
            "fpo/ratio_max": agg["ratio_max"],
            "fpo/clip_frac": agg["clip_frac"],
            "fpo/cfm_new": agg["cfm_new"],
            "fpo/cfm_old": agg["cfm_old"],
            "fpo/grad_norm": agg["grad_norm"],
            "fpo/grad_norm_critic": agg["grad_norm_critic"],
            "fpo/kl": agg["kl"],
            "fpo/lr": self.learning_rate,
            "fpo/critic_lr": self.critic_learning_rate,
            "rollout/reward_step_mean": float(rewards.mean().item()),
            "rollout/done_frac": float(dones.float().mean().item()),
            "act/abs_max_all": float(rollout.get("action_abs_max", 0.0)),
            "timing/collect_s": collect_time,
            "timing/update_s": update_time,
            "policy/action_delta": agg["action_delta"],
            "policy/param_rms_delta": agg["param_rms_delta"],
            "policy/effective_mini_batch_size": agg["mini_batch_size"],
        }
        for key, mask in rollout["done_terms_union"].items():
            metrics[f"done/{key}_frac"] = float(mask.float().mean().item())
        # alive-weighted rollout reward terms + done causes (shared [TRACK_ROLLOUT]/[DONE_ROLLOUT]).
        rollout_reward_sums: dict[str, float] = {}
        rollout_done_union: dict[str, torch.Tensor] = {}
        weight_sum = 0.0
        for info, valid in rollout["rollout_info_items"]:
            vf = valid.float()
            w = float(vf.sum().item())
            if w <= 0.0:
                continue
            weight_sum += w
            for k, v in info["reward_terms"].items():
                rollout_reward_sums[k] = rollout_reward_sums.get(k, 0.0) + float((v * vf).sum().item())
            for k, v in info["done_terms"].items():
                md = v.bool() & valid.bool()
                rollout_done_union[k] = md.clone() if k not in rollout_done_union else (rollout_done_union[k] | md)
        if weight_sum > 0.0:
            for k, s in rollout_reward_sums.items():
                metrics[f"reward_rollout/{k}_mean"] = s / weight_sum
            for k, m in rollout_done_union.items():
                metrics[f"done_rollout/{k}_frac"] = float(m.float().mean().item())
        for info in rollout["first_chunk_infos"]:
            for k, v in info["reward_terms"].items():
                metrics[f"reward/{k}_mean"] = float(v.mean().item())
        if self._train_reward_buffer:
            metrics["train/mean_reward"] = float(sum(self._train_reward_buffer) / len(self._train_reward_buffer))
            metrics["train/mean_episode_length"] = float(sum(self._train_length_buffer) / len(self._train_length_buffer))
        else:
            metrics["train/mean_reward"] = float("nan")
            metrics["train/mean_episode_length"] = float("nan")
        metrics["train/recent_episode_count"] = float(len(self._train_reward_buffer))
        metrics["train/completed_episodes"] = float(self._train_completed_episodes)
        self._add_first_failure_metrics(metrics, rollout)
        # Generic action magnitude summary so the shared [ACT_SUMMARY] line is not all-nan.
        with torch.no_grad():
            # Magnitude metrics report the EXECUTED residual r_t (what the env actually applies).
            acts = rollout["residuals"].reshape(-1, self.num_act)
            metrics["latent/final_abs_mean"] = float(acts.abs().mean().item())
            metrics["latent/final_abs_max"] = float(acts.abs().max().item())
            greedy = self.actor.act_inference(rollout["actor_obs"][0], eval_mode="zero")  # (N, A) innovation
            a_abs = greedy.abs()
            flat = a_abs.reshape(-1)
            metrics["act/abs_mean"] = float(flat.mean().item())
            metrics["act/abs_p95"] = float(torch.quantile(flat, 0.95).item())
            metrics["act/abs_p99"] = float(torch.quantile(flat, 0.99).item())
            metrics["act/abs_max"] = float(flat.max().item())
            metrics["act/first_abs_mean"] = float(a_abs.mean().item())
            metrics["act/last_abs_mean"] = float(a_abs.mean().item())
        return metrics

    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        print(
            f"[UPDATE] {update_idx}/{max_updates} "
            f"reward_step={metrics['rollout/reward_step_mean']:.5f} "
            f"done_frac={metrics['rollout/done_frac']:.5f} "
            f"mean_reward={metrics.get('train/mean_reward', float('nan')):.5f} "
            f"mean_len={metrics.get('train/mean_episode_length', float('nan')):.2f}",
            flush=True,
        )
        print(
            f"[FPO++] actor_loss={metrics['fpo/actor_loss']:.5f} value_loss={metrics['fpo/value_loss']:.5f} "
            f"ratio={metrics['fpo/ratio']:.4f} [{metrics['fpo/ratio_min']:.3f},{metrics['fpo/ratio_max']:.3f}] "
            f"clip_frac={metrics['fpo/clip_frac']:.4f} cfm_old={metrics['fpo/cfm_old']:.4f} "
            f"cfm_new={metrics['fpo/cfm_new']:.4f} kl={metrics['fpo/kl']:.6f} "
            f"grad={metrics['fpo/grad_norm']:.4f} grad_c={metrics.get('fpo/grad_norm_critic', float('nan')):.4f} "
            f"lr={metrics['fpo/lr']:.6f} critic_lr={metrics.get('fpo/critic_lr', float('nan')):.6f}",
            flush=True,
        )
        log_shared_update_diagnostics(metrics, failure_label="FIRST_FAILURE", index_name="step")
        log_shared_tracking(metrics)

    def log_banner(self) -> None:
        cfg = self.cfg
        env = self.env
        print("[INFO] Starting FPO++ training", flush=True)
        print(f"[INFO] motion_file={env.task_cfg.motion_file}", flush=True)
        print(
            f"[INFO] algo=fpo_pp actor_obs_dim={self.actor_obs_dim} critic_obs_dim={self.critic_obs_dim} "
            f"action_dim={self.num_act} horizon={self.horizon} "
            f"chunk_dim={self.chunk_dim} num_envs={env.num_envs} num_steps_per_env={self.num_steps_per_env} "
            f"flow_steps={self.flow_steps} num_mc={self.num_mc} actor_scale={self.actor.actor_scale} "
            f"action_perturb_std={self.actor.action_perturb_std} timestep_embed_dim={self.actor.timestep_embed_dim} "
            f"residual_rho={self.residual_rho} residual_innov_scale={self.residual_innov_scale} "
            f"cfm_reduction={self.actor.cfm_loss_reduction} clip={cfg.clip_range} "
            f"cfm_diff_clamp_max={self.cfm_diff_clamp_max} cfm_loss_clamp={self.cfm_loss_clamp} "
            f"adv_clamp={self.adv_clamp} schedule={self.schedule} desired_kl={self.desired_kl} "
            f"trust_region={self.trust_region_mode} num_micro_batches={self.num_micro_batches} "
            f"terminal_penalty={self.terminal_penalty} "
            f"num_learning_epochs={cfg.num_learning_epochs} num_mini_batches={cfg.num_mini_batches} "
            f"gamma={cfg.discount_gamma} lam={cfg.gae_lambda} value_loss_coef={cfg.value_loss_coef} "
            f"lr={cfg.policy_lr} critic_lr={self.critic_learning_rate} weight_decay={cfg.weight_decay} "
            f"empirical_normalization={cfg.empirical_normalization} "
            f"actor_hidden_dims={list(cfg.actor_hidden_dims)} activation={cfg.activation}",
            flush=True,
        )
