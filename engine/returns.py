from __future__ import annotations

import torch


def compute_gae_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    *,
    gamma: float,
    lam: float,
    timeouts: torch.Tensor | None = None,
    normalize_advantage: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape (env, steps), got {tuple(rewards.shape)}")
    if dones.shape != rewards.shape or values.shape != rewards.shape:
        raise ValueError("dones and values must match rewards shape")
    if last_values.shape != (rewards.shape[0],):
        raise ValueError(f"last_values must have shape {(rewards.shape[0],)}, got {tuple(last_values.shape)}")

    rewards_for_gae = rewards
    if timeouts is not None:
        if timeouts.shape != rewards.shape:
            raise ValueError(f"timeouts must have shape {tuple(rewards.shape)}, got {tuple(timeouts.shape)}")
        rewards_for_gae = rewards_for_gae + gamma * values * timeouts.to(dtype=rewards.dtype)

    returns = torch.empty_like(rewards)
    advantage = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    for step_index in range(rewards.shape[1] - 1, -1, -1):
        next_values = last_values if step_index == rewards.shape[1] - 1 else values[:, step_index + 1]
        next_is_not_terminal = 1.0 - dones[:, step_index].to(dtype=rewards.dtype)
        delta = rewards_for_gae[:, step_index] + gamma * next_is_not_terminal * next_values - values[:, step_index]
        advantage = delta + gamma * lam * next_is_not_terminal * advantage
        returns[:, step_index] = advantage + values[:, step_index]

    advantages = returns - values
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


def compute_frame_level_reward_to_go(
    reward_frame: torch.Tensor,
    done_frame: torch.Tensor,
    alive_frame: torch.Tensor,
    last_values: torch.Tensor,
    *,
    gamma: float,
    terminal_penalty: float,
) -> torch.Tensor:
    """Frame-level reward-to-go for Frame-Factorized h>1 (design S4).

    Expands the rollout to the real time axis T = chunks * horizon and rolls a per-frame
    discounted return, applying the one-shot terminal penalty at the exact death frame and
    seeding the tail bootstrap for branches that never died.

    Shapes (E = grpo groups, G = generations):
      reward_frame : (E, G, chunks, horizon)  raw (undiscounted) per-frame reward
      done_frame   : (E, G, chunks, horizon)  bool, alive_before_frame & done_t
      alive_frame  : (E, G, chunks, horizon)  float/bool, alive entering the frame
      last_values  : (E*G,) or (E, G)         tail bootstrap value (0 for dead branches)
    Returns:
      reward_to_go : (E, G, chunks, horizon)

    h=1 reduces exactly to the chunk-level RTG: T == chunks, gamma == chunk_gamma, and the
    chunk-level death discount gamma**death_frame * chunk_gamma**death_chunk collapses to
    gamma**death_t, which is what this backward recursion produces.
    """
    if reward_frame.ndim != 4:
        raise ValueError(f"reward_frame must be (E, G, chunks, horizon), got {tuple(reward_frame.shape)}")
    E, G, chunks, horizon = reward_frame.shape
    T = chunks * horizon
    dtype = reward_frame.dtype
    device = reward_frame.device

    reward_T = reward_frame.reshape(E, G, T)
    done_T = done_frame.reshape(E, G, T).to(dtype=torch.bool)
    alive_T = alive_frame.reshape(E, G, T).to(dtype=dtype)

    first_life = reward_T * alive_T

    # Death frame = first True along T. died = any done in window (incl. timeout, to match
    # the chunk-level first_done_chunk < chunks convention exactly).
    died = done_T.any(dim=-1)
    # argmax on a bool tensor returns the first True index (0 if none, but masked by died).
    death_t = done_T.to(dtype=torch.int64).argmax(dim=-1)
    if bool(died.any()):
        penalty = torch.zeros(E, G, T, device=device, dtype=dtype)
        died_idx = died.nonzero(as_tuple=False)
        for e, g in died_idx.tolist():
            penalty[e, g, int(death_t[e, g].item())] = float(terminal_penalty)
        first_life = first_life - penalty * died.unsqueeze(-1).to(dtype=dtype)

    last_values = last_values.reshape(E, G).to(device=device, dtype=dtype)
    alive_at_end = (~died).to(dtype=dtype)
    running = last_values * alive_at_end
    reward_to_go = torch.zeros(E, G, T, device=device, dtype=dtype)
    for t in range(T - 1, -1, -1):
        running = first_life[:, :, t] + float(gamma) * running
        reward_to_go[:, :, t] = running
        running = running * alive_T[:, :, t]
    return reward_to_go.reshape(E, G, chunks, horizon)
