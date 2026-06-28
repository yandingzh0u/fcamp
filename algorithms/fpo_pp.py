"""FPO++ (Flow Policy Optimization++) algorithm plugin.

Paper-faithful implementation of FPO++ from "Flow Policy Gradients for Robot Control"
(arXiv:2602.02481). FPO++ trains a flow-matching policy with a PPO-style actor-critic, but
replaces the action log-likelihood ratio with a *conditional flow matching (CFM) loss ratio*:

    rho_i = exp( L_CFM_old_i - L_CFM_new_i )                         (paper Eq. 3 / 10)

evaluated per Monte-Carlo (tau_i, eps_i) pair (per-sample ratio, paper Eq. 10), and combines a
PPO clip (positive advantages) with the SPO trust region (negative advantages) into the
Asymmetric SPO (ASPO) objective (paper Eq. 11/12/13).

This is intentionally distinct from `algorithms/mixgrpo.py`, which uses an SDE-trajectory
transition log_prob ratio. FPO++ does NOT use likelihoods at all.

Key pieces (all in flow / chunk-coefficient space of the existing FlowMatchingPolicy):
  * Deterministic Euler integration of the learned velocity field for exploration / inference.
    Training draws the initial noise eps ~ N(0, I); evaluation uses eps = 0 ("zero-sampling",
    paper Sec. III-D).
  * CFM loss with linear interpolation a^tau = tau*a + (1-tau)*eps and velocity target a - eps
    (paper Eq. 5/6/8).
  * A standard PPO critic + chunk-level GAE + clipped value loss (the paper's motion-tracking
    appendix uses a critic, GAE, minibatches, and learning epochs).
  * Chunk rollout (execute H frames per action chunk with alive-masking), reused from the
    MixGRPO collection pattern, but with continuous rolling (auto_reset) like PPO instead of
    GRPO group resets.

The flow / CFM math lives in module-level pure functions (`euler_integrate`, `cfm_loss`,
`fpo_pp_ratio`, `aspo_objective`) so it can be unit-tested without IsaacLab.

Note (paper-faithfulness vs. trajectory parametrization): the CFM loss is computed in the
flow's native space, which is the policy's `chunk_dim = basis_count * action_dim` coefficient
space. With `basis_count == horizon` the temporal basis is the identity, so this is exactly the
per-frame action-chunk space the paper operates in. With `basis_count < horizon` the CFM ratio
is taken in the lower-dim coefficient/latent manifold ("latent FPO++"), which is no longer the
strict paper formulation.
"""
from __future__ import annotations

from collections import deque

import torch
from torch import nn

from algorithms.base import Algorithm
from core.logging import log_shared_tracking, log_shared_update_diagnostics
from networks.flow_policy import FlowMatchingPolicy
from networks.mlp_actor_critic import Critic, EmpiricalNormalization


# ---------------------------------------------------------------------------- flow / CFM math
def euler_integrate(policy: FlowMatchingPolicy, observation: torch.Tensor, x0: torch.Tensor, steps: int) -> torch.Tensor:
    """Deterministic forward Euler integration of the learned velocity field.

    Flow convention (paper Eq. 5/6): time tau in [0, 1], tau=0 -> noise, tau=1 -> action;
    a^tau = tau*a + (1-tau)*eps, so d a^tau / d tau = a - eps. We integrate tau: 0 -> 1 with
    x_{k+1} = x_k + v_theta(x_k, tau_k; o) * dt, dt = 1/steps. x0 is the initial noise eps
    (eps ~ N(0, I) for exploration, eps = 0 for zero-sampling).

    Returns the flow endpoint a (the chunk-coefficient latent), shape == x0.shape.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    batch = x0.shape[0]
    x = x0
    dt = 1.0 / steps
    for k in range(steps):
        tau = torch.full((batch,), k * dt, device=x0.device, dtype=x0.dtype)
        velocity = policy.velocity_field(observation, x, tau)
        x = x + velocity * dt
    return x


def cfm_loss(
    policy: FlowMatchingPolicy,
    observation: torch.Tensor,
    action_latent: torch.Tensor,
    tau: torch.Tensor,
    eps: torch.Tensor,
    *,
    loss_clamp: float = 0.0,
) -> torch.Tensor:
    """Per-Monte-Carlo-sample conditional flow matching loss (paper Eq. 8).

        a^tau_i  = tau_i * a + (1 - tau_i) * eps_i
        target_i = a - eps_i
        l_i      = mean_d ( v_theta(a^tau_i, tau_i; o) - target_i )^2     # mean over chunk dims

    IMPORTANT (ratio scale): the CFM loss is the MEAN squared error over the chunk dimension
    (D = basis_count * action_dim), NOT the raw sum. The FPO++ ratio is exp(l_old - l_new); a
    plain sum over D (=232..348 here) makes the exponent scale with dimensionality, so tiny
    per-dim velocity changes blow the ratio up to e^6+ (observed ratio_max ~785 / clip_frac ~0.9
    in the fpo_pp_30 run). Averaging over D keeps the exponent dimension-invariant and O(1), so
    delta_clip / clip_range act on a sane scale. This matches the standard flow-matching MSE
    convention; the paper's ||.||^2 notation is the per-sample objective, implemented as the mean.

    Args:
        observation:   (B, obs_dim) policy observation (already normalized by the caller).
        action_latent: (B, D) flow endpoint a (chunk-coefficient latent).
        tau:           (B, M) flow steps in [0, 1].
        eps:           (B, M, D) noises ~ N(0, I).
        loss_clamp:    if > 0, clamp each CFM loss to [0, loss_clamp] before it is used in the
                       ratio difference (paper App. C.23: clamping CFM losses aids stability).

    Returns:
        (B, M) per-sample mean-squared CFM losses.
    """
    B, M = tau.shape
    D = action_latent.shape[-1]
    if eps.shape != (B, M, D):
        raise ValueError(f"eps must be {(B, M, D)}, got {tuple(eps.shape)}")
    obs_rep = observation.unsqueeze(1).expand(B, M, observation.shape[-1]).reshape(B * M, observation.shape[-1])
    a = action_latent.unsqueeze(1).expand(B, M, D)
    tau_e = tau.unsqueeze(-1)
    a_tau = (tau_e * a + (1.0 - tau_e) * eps).reshape(B * M, D)
    target = (a - eps).reshape(B * M, D)
    tau_flat = tau.reshape(B * M)
    pred = policy.velocity_field(obs_rep, a_tau, tau_flat)
    loss = ((pred - target) ** 2).mean(-1).reshape(B, M)
    if loss_clamp > 0.0:
        loss = loss.clamp(min=0.0, max=float(loss_clamp))
    return loss


def fpo_pp_ratio(old_cfm: torch.Tensor, new_cfm: torch.Tensor, delta_clip: float) -> torch.Tensor:
    """Per-sample FPO++ ratio (paper Eq. 10): rho_i = exp(l_old_i - l_new_i).

    The difference is clamped before exponentiation (paper App. C.23) for numerical stability.
    delta_clip <= 0 disables the clamp.
    """
    diff = old_cfm - new_cfm
    if delta_clip > 0.0:
        diff = diff.clamp(min=-float(delta_clip), max=float(delta_clip))
    return torch.exp(diff)


def aspo_objective(ratio: torch.Tensor, advantage: torch.Tensor, clip: float) -> torch.Tensor:
    """Asymmetric SPO objective (paper Eq. 11/12).

    PPO clip for advantage >= 0, SPO trust region for advantage < 0. `advantage` broadcasts
    against `ratio` (e.g. ratio (B, M), advantage (B, 1)). Returns the per-element objective
    psi_ASPO(rho, A) (to be maximized).
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
        self.num_act = int(cfg.action_dim)
        self.horizon = max(1, int(cfg.horizon))
        self.actor_obs_dim = env.observation_dim
        self.critic_obs_dim = env.critic_observation_dim
        device = env.device

        self.actor = FlowMatchingPolicy(
            obs_dim=self.actor_obs_dim,
            action_dim=self.num_act,
            horizon=self.horizon,
            hidden_dims=tuple(cfg.actor_hidden_dims),
            activation=cfg.activation,
            init_noise_std=cfg.init_noise_std,
            action_squash_scale=cfg.action_squash_scale,
            basis_count=cfg.basis_count,
            chunk_stitch_frames=cfg.chunk_stitch_frames,
            chunk_stitch_mode=cfg.chunk_stitch_mode,
        ).to(device)
        self.critic = Critic(self.critic_obs_dim, tuple(cfg.actor_hidden_dims), cfg.activation).to(device)

        self.chunk_dim = self.actor.chunk_dim

        self.empirical_normalization = bool(cfg.empirical_normalization)
        if self.empirical_normalization:
            self.actor_obs_normalizer: nn.Module = EmpiricalNormalization(self.actor_obs_dim, device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(self.critic_obs_dim, device)
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        # Single AdamW over actor + critic (paper Table A.2: one LR, AdamW betas (0.9, 0.95)).
        self.learning_rate = float(cfg.policy_lr)
        self._optimizer = torch.optim.AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=float(cfg.weight_decay),
        )

        self.num_mc = max(1, int(cfg.fpo_num_mc))
        self.flow_steps = max(1, int(cfg.flow_steps))
        self.delta_clip = float(cfg.fpo_delta_clip)
        self.cfm_loss_clamp = float(cfg.fpo_cfm_loss_clamp)
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

    def _record_episode_stats(self, rewards, dones, step_counts) -> None:
        self._train_reward_sum += rewards.to(dtype=torch.float32)
        self._train_episode_length += step_counts.to(dtype=torch.float32)
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

    # ------------------------------------------------------------------ flow sampling
    def _sample_action_chunk(self, actor_obs_n, raw_obs, x0):
        """Euler-integrate the flow to the endpoint latent and transform to an action chunk.

        Returns (action_chunk (N, H, A), action_latent (N, chunk_dim)).
        """
        latent = euler_integrate(self.actor, actor_obs_n, x0, self.flow_steps)
        start_action = raw_obs[..., -self.num_act:]
        action_flat = self.actor._action_transform(latent, start_action=start_action)
        action_chunk = action_flat.view(actor_obs_n.shape[0], self.horizon, self.num_act)
        return action_chunk, latent

    # ------------------------------------------------------------------ resets
    def initial_reset(self) -> torch.Tensor:
        obs = self.env.reset()
        if bool(self.cfg.init_at_random_ep_len) and self.max_episode_steps > 0:
            self.env.episode_steps = torch.randint_like(self.env.episode_steps, high=int(self.max_episode_steps))
        self._obs = obs
        self._critic_obs = self.env.get_critic_observation()
        return obs

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        # FPO++ rolls continuously (auto_reset inside the chunk); nothing to re-sample per update.
        return self._obs

    # ------------------------------------------------------------------ rollout
    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        device = env.device
        N = env.num_envs
        H = self.horizon
        M = self.num_mc
        D = self.chunk_dim
        T = self._chunks_per_rollout()
        gamma = float(self.cfg.discount_gamma)

        raw_obs_buf = torch.zeros(T, N, self.actor_obs_dim, device=device)
        actor_obs_buf = torch.zeros(T, N, self.actor_obs_dim, device=device)
        critic_obs_buf = torch.zeros(T, N, self.critic_obs_dim, device=device)
        latent_buf = torch.zeros(T, N, D, device=device)
        tau_buf = torch.zeros(T, N, M, device=device)
        eps_buf = torch.zeros(T, N, M, D, device=device)
        old_cfm_buf = torch.zeros(T, N, M, device=device)
        values_buf = torch.zeros(T, N, 1, device=device)
        rewards_buf = torch.zeros(T, N, 1, device=device)
        dones_buf = torch.zeros(T, N, 1, dtype=torch.bool, device=device)

        obs = self._obs
        critic_obs = self._critic_obs
        done_terms_union: dict[str, torch.Tensor] = {}
        rollout_info_items: list[tuple] = []
        first_chunk_infos: list[dict] = []
        action_abs_max = 0.0

        with torch.no_grad():
            for t in range(T):
                chunk_start_obs = obs
                actor_obs_n = self._norm_actor(obs)
                critic_obs_n = self._norm_critic(critic_obs)
                value = self.critic.evaluate(critic_obs_n).detach()

                x0 = torch.randn(N, D, device=device)
                action_chunk, latent = self._sample_action_chunk(actor_obs_n, obs, x0)
                action_abs_max = max(action_abs_max, float(action_chunk.abs().max().item()))

                tau = torch.rand(N, M, device=device)
                eps = torch.randn(N, M, D, device=device)
                old_cfm = cfm_loss(self.actor, actor_obs_n, latent, tau, eps, loss_clamp=self.cfm_loss_clamp).detach()

                # ---- execute H frames (alive-masked); auto_reset only on the last frame ----
                chunk_reward = torch.zeros(N, device=device, dtype=actor_obs_n.dtype)
                chunk_done = torch.zeros(N, dtype=torch.bool, device=device)
                chunk_timeout = torch.zeros(N, dtype=torch.bool, device=device)
                chunk_live = torch.zeros(N, device=device, dtype=actor_obs_n.dtype)
                timeout_bootstrap_value = torch.zeros(N, device=device, dtype=actor_obs_n.dtype)
                timeout_bootstrap_discount = torch.zeros(N, device=device, dtype=actor_obs_n.dtype)
                alive = torch.ones(N, dtype=torch.bool, device=device)
                for f in range(H):
                    alive_before = alive.clone()
                    action_f = action_chunk[:, f, :]
                    if bool((~alive_before).any()):
                        action_f = torch.where(alive_before.unsqueeze(-1), action_f, torch.zeros_like(action_f))
                    last_frame = f == H - 1
                    next_obs, reward, done, info = env.step(
                        action_f,
                        auto_reset=last_frame,
                        reset_horizon=max(1, (T - t - 1) * H + (H - f)),
                    )
                    if t == 0 and f == 0:
                        first_chunk_infos.append(info)
                    rollout_info_items.append((info, alive_before.detach()))
                    contrib = alive_before.to(dtype=chunk_reward.dtype)
                    chunk_reward = chunk_reward + (gamma ** f) * reward.to(dtype=chunk_reward.dtype) * contrib
                    chunk_live = chunk_live + contrib
                    timeout_f = info["done_terms"]["time_out"].bool()
                    new_done = alive_before & done
                    new_timeout = new_done & timeout_f
                    chunk_done = chunk_done | new_done
                    chunk_timeout = chunk_timeout | new_timeout
                    if bool(new_timeout.any()):
                        if last_frame and "final_critic_observation" in info:
                            timeout_critic_obs = info["final_critic_observation"]
                        else:
                            timeout_critic_obs = env.get_critic_observation()
                        timeout_critic_obs = self._norm_critic(timeout_critic_obs, update=False)
                        timeout_values = self.critic.evaluate(timeout_critic_obs).detach().squeeze(1)
                        timeout_bootstrap_value[new_timeout] = timeout_values[new_timeout]
                        timeout_bootstrap_discount[new_timeout] = float(gamma ** (f + 1))
                    for key, val in info["done_terms"].items():
                        b = val.bool()
                        done_terms_union[key] = b.clone() if key not in done_terms_union else (done_terms_union[key] | b)
                    alive = alive & ~done
                    obs = next_obs
                    critic_obs = env.get_critic_observation()

                # Timeout value bootstrap (truncation, not failure): add gamma^k * V(final state).
                if bool(chunk_timeout.any()):
                    chunk_reward = chunk_reward + timeout_bootstrap_discount * timeout_bootstrap_value

                raw_obs_buf[t] = chunk_start_obs
                actor_obs_buf[t] = actor_obs_n
                critic_obs_buf[t] = critic_obs_n
                latent_buf[t] = latent
                tau_buf[t] = tau
                eps_buf[t] = eps
                old_cfm_buf[t] = old_cfm
                values_buf[t] = value
                rewards_buf[t] = chunk_reward.view(-1, 1)
                dones_buf[t] = chunk_done.view(-1, 1)
                self._record_episode_stats(chunk_reward, chunk_done, chunk_live)

            last_critic_obs = self._norm_critic(critic_obs, update=False)
            last_values = self.critic.evaluate(last_critic_obs).detach()
            chunk_gamma = gamma ** H
            returns, advantages = self._compute_gae(last_values, values_buf, dones_buf, rewards_buf, chunk_gamma)

        self._obs = obs
        self._critic_obs = critic_obs
        return {
            "raw_obs": raw_obs_buf, "actor_obs": actor_obs_buf, "critic_obs": critic_obs_buf, "latent": latent_buf,
            "tau": tau_buf, "eps": eps_buf, "old_cfm": old_cfm_buf, "values": values_buf,
            "returns": returns, "advantages": advantages, "rewards": rewards_buf, "dones": dones_buf,
            "done_terms_union": done_terms_union, "rollout_info_items": rollout_info_items,
            "first_chunk_infos": first_chunk_infos, "action_abs_max": action_abs_max,
            "next_observation": obs,
        }

    def _chunks_per_rollout(self) -> int:
        rollout_env_steps = int(self.cfg.rollout_env_steps)
        if rollout_env_steps > 0:
            if rollout_env_steps % self.horizon != 0:
                raise ValueError(
                    f"rollout_env_steps ({rollout_env_steps}) must be divisible by horizon ({self.horizon})."
                )
            return max(1, rollout_env_steps // self.horizon)
        return max(1, int(self.cfg.chunks_per_rollout))

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
        T, N = rollout["latent"].shape[0], rollout["latent"].shape[1]
        B = T * N
        M = self.num_mc
        D = self.chunk_dim

        actor_obs = rollout["actor_obs"].reshape(B, self.actor_obs_dim)
        critic_obs = rollout["critic_obs"].reshape(B, self.critic_obs_dim)
        latent = rollout["latent"].reshape(B, D)
        tau = rollout["tau"].reshape(B, M)
        eps = rollout["eps"].reshape(B, M, D)
        old_cfm = rollout["old_cfm"].reshape(B, M)
        old_values = rollout["values"].reshape(B, 1)
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
        }
        num_updates = 0

        # Probe action drift before the update (zero-sampling greedy chunk).
        probe_count = min(128, N)
        with torch.no_grad():
            probe_obs = rollout["raw_obs"][0, :probe_count]
            probe_before = self._zero_sample_chunk(probe_obs)
            params_before = [p.detach().clone() for p in self.actor.parameters()]

        t1 = _time.perf_counter()
        for _ in range(epochs):
            perm = torch.randperm(B, device=device)
            for mb in range(num_mini_batches):
                idx = perm[mb * mini_batch_size:(mb + 1) * mini_batch_size]
                mb_actor_obs = actor_obs[idx]
                mb_latent = latent[idx]
                mb_tau = tau[idx]
                mb_eps = eps[idx]
                mb_old_cfm = old_cfm[idx]
                mb_adv = advantages[idx]
                mb_old_values = old_values[idx]
                mb_returns = returns[idx]
                mb_critic_obs = critic_obs[idx]

                new_cfm = cfm_loss(self.actor, mb_actor_obs, mb_latent, mb_tau, mb_eps, loss_clamp=self.cfm_loss_clamp)
                ratio = fpo_pp_ratio(mb_old_cfm, new_cfm, self.delta_clip)  # (mb, M)
                objective = aspo_objective(ratio, mb_adv, clip)             # (mb, M)
                actor_loss = -objective.mean()

                value = self.critic.evaluate(mb_critic_obs)
                value_clipped = mb_old_values + (value - mb_old_values).clamp(-clip, clip)
                value_loss = torch.max((value - mb_returns) ** 2, (value_clipped - mb_returns) ** 2).mean()

                loss = actor_loss + value_coef * value_loss
                self._optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), self.cfg.max_grad_norm
                )
                self._optimizer.step()

                with torch.no_grad():
                    totals["actor_loss"] += float(actor_loss.item())
                    totals["value_loss"] += float(value_loss.item())
                    totals["ratio"] += float(ratio.mean().item())
                    totals["ratio_min"] = min(totals["ratio_min"], float(ratio.min().item()))
                    totals["ratio_max"] = max(totals["ratio_max"], float(ratio.max().item()))
                    totals["clip_frac"] += float((torch.abs(ratio - 1.0) > clip).float().mean().item())
                    totals["cfm_new"] += float(new_cfm.mean().item())
                    totals["cfm_old"] += float(mb_old_cfm.mean().item())
                    totals["grad_norm"] += float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                num_updates += 1

        update_time = _time.perf_counter() - t1
        denom = max(num_updates, 1)
        with torch.no_grad():
            probe_after = self._zero_sample_chunk(probe_obs)
            action_delta = float(torch.mean(torch.abs(probe_after - probe_before)).item())
            param_delta_sq = torch.zeros((), device=device)
            param_count = 0
            for p, before in zip(self.actor.parameters(), params_before, strict=True):
                d = p.detach() - before
                param_delta_sq = param_delta_sq + torch.sum(d * d)
                param_count += d.numel()
            param_rms_delta = float(torch.sqrt(param_delta_sq / max(param_count, 1)).item())

        agg = {
            "actor_loss": totals["actor_loss"] / denom,
            "value_loss": totals["value_loss"] / denom,
            "ratio": totals["ratio"] / denom,
            "ratio_min": totals["ratio_min"] if totals["ratio_min"] != float("inf") else 0.0,
            "ratio_max": totals["ratio_max"],
            "clip_frac": totals["clip_frac"] / denom,
            "cfm_new": totals["cfm_new"] / denom,
            "cfm_old": totals["cfm_old"] / denom,
            "grad_norm": totals["grad_norm"] / denom,
            "action_delta": action_delta,
            "param_rms_delta": param_rms_delta,
            "mini_batch_size": float(mini_batch_size),
        }
        return self._build_metrics(rollout, agg, collect_time, update_time)

    # ------------------------------------------------------------------ inference
    def _zero_sample_chunk(self, obs: torch.Tensor) -> torch.Tensor:
        """Zero-sampling greedy action chunk (paper Sec. III-D): integrate flow from eps = 0."""
        actor_obs_n = self._norm_actor(obs, update=False)
        x0 = torch.zeros(obs.shape[0], self.chunk_dim, device=obs.device, dtype=obs.dtype)
        action_chunk, _ = self._sample_action_chunk(actor_obs_n, obs, x0)
        return action_chunk

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        return self._zero_sample_chunk(obs)

    # ------------------------------------------------------------------ metrics + logging
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
            "fpo/lr": self.learning_rate,
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
        for k in ("first_failure_chunk_mean", "first_failure_chunk_min", "first_failure_chunk_max",
                  "first_failure_phase_mean", "first_failure_phase_min", "first_failure_phase_max"):
            metrics[f"rollout/{k}"] = float("nan")
        # Generic action / latent magnitude summary so the shared [ACT_SUMMARY] line is not all-nan.
        # (Joint-specific [TRAIN_ACT]/[TRAIN_BODY]/[REWARD_WEIGHTED] keys need env body mappings the
        # MixGRPO path fills; FPO++ leaves those to the shared default.)
        with torch.no_grad():
            lat = rollout["latent"].reshape(-1, self.chunk_dim)
            metrics["latent/final_abs_mean"] = float(lat.abs().mean().item())
            metrics["latent/final_abs_max"] = float(lat.abs().max().item())
            greedy = self._zero_sample_chunk(rollout["raw_obs"][0])  # (N, H, A)
            a_abs = greedy.abs()
            flat = a_abs.reshape(-1)
            metrics["act/abs_mean"] = float(flat.mean().item())
            metrics["act/abs_p95"] = float(torch.quantile(flat, 0.95).item())
            metrics["act/abs_p99"] = float(torch.quantile(flat, 0.99).item())
            metrics["act/abs_max"] = float(flat.max().item())
            metrics["act/first_abs_mean"] = float(a_abs[:, 0].mean().item())
            metrics["act/last_abs_mean"] = float(a_abs[:, -1].mean().item())
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
            f"cfm_new={metrics['fpo/cfm_new']:.4f} grad={metrics['fpo/grad_norm']:.4f} lr={metrics['fpo/lr']:.6f}",
            flush=True,
        )
        log_shared_update_diagnostics(metrics, failure_label="FIRST_FAILURE", index_name="chunk")
        log_shared_tracking(metrics)

    def log_banner(self) -> None:
        cfg = self.cfg
        env = self.env
        print("[INFO] Starting FPO++ training", flush=True)
        print(f"[INFO] motion_file={env.task_cfg.motion_file}", flush=True)
        print(
            f"[INFO] algo=fpo_pp actor_obs_dim={self.actor_obs_dim} critic_obs_dim={self.critic_obs_dim} "
            f"action_dim={self.num_act} horizon={self.horizon} basis_count={self.actor.basis_count} "
            f"chunk_dim={self.chunk_dim} num_envs={env.num_envs} chunks_per_rollout={self._chunks_per_rollout()} "
            f"flow_steps={self.flow_steps} num_mc={self.num_mc} clip={cfg.clip_range} "
            f"delta_clip={self.delta_clip} cfm_loss_clamp={self.cfm_loss_clamp} "
            f"num_learning_epochs={cfg.num_learning_epochs} num_mini_batches={cfg.num_mini_batches} "
            f"gamma={cfg.discount_gamma} lam={cfg.gae_lambda} value_loss_coef={cfg.value_loss_coef} "
            f"lr={cfg.policy_lr} weight_decay={cfg.weight_decay} "
            f"empirical_normalization={cfg.empirical_normalization} "
            f"actor_hidden_dims={list(cfg.actor_hidden_dims)} activation={cfg.activation}",
            flush=True,
        )
