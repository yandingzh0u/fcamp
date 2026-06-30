"""Algorithm plugin interface.

The core trainer is algorithm-agnostic: it owns the env, the update loop, checkpoint /
validation / logging scheduling, and nothing else. Everything that differs between
algorithms (network, action sampling, rollout collection, credit assignment, policy update,
whether there is a critic, whether rollouts use GRPO groups) lives behind this interface.

A concrete algorithm (MixGRPO today; PPO / FPO later) subclasses Algorithm and implements:

  build(env)            -> construct networks + optimizer given the built env
  reset_for_update()    -> reset/sample env starts for the next on-policy batch, return obs
  collect(obs)          -> roll the env, return an opaque rollout dict
  update(rollout)       -> run the policy update, return a metrics dict
  state_dict/load_state_dict -> checkpoint payload for the algorithm's own state
  log(update_idx, metrics)   -> algorithm-specific console logging

The core trainer never reaches into algorithm internals; it only calls these methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Algorithm(ABC):
    # Whether collect() returns the next starting observation under "next_observation"
    # so the loop can avoid an extra reset. Informational; the loop calls reset_for_update.
    name: str = "base"

    def __init__(self, cfg, env, simulation_app):
        self.cfg = cfg            # AlgoCfg
        self.env = env
        self.simulation_app = simulation_app

    @abstractmethod
    def build(self) -> None:
        """Construct networks, optimizer, and any algorithm state. Called once after the env
        exists."""

    @abstractmethod
    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        """Reset envs for the next on-policy batch and return the starting observation."""

    @abstractmethod
    def initial_reset(self) -> torch.Tensor:
        """One-time plain reset at trainer construction (before any rollout)."""

    @abstractmethod
    def collect(self, obs: torch.Tensor) -> dict:
        """Collect one on-policy batch. Returns an opaque rollout dict (algorithm-defined)."""

    @abstractmethod
    def update(self, rollout: dict, collect_time: float) -> dict:
        """Run the policy update from a collected rollout. Returns a flat metrics dict."""

    @abstractmethod
    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        """Algorithm-specific console logging for one update."""

    @abstractmethod
    def log_banner(self) -> None:
        """Print the one-time [INFO] startup banner describing this run's configuration."""

    # ---- networks for checkpoint / validation (subclasses set these in build) ----
    @property
    @abstractmethod
    def policy(self) -> torch.nn.Module: ...

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Optimizer: ...

    # ---- checkpoint hooks (algorithm-specific extra state) ----
    def extra_checkpoint_state(self) -> dict:
        return {}

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        pass

    @abstractmethod
    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Greedy/eval action chunk (B, horizon, action_dim) for validation/playback."""
