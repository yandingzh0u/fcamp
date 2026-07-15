from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Algorithm(ABC):
    def __init__(self, cfg, env, simulation_app):
        self.cfg = cfg
        self.env = env
        self.simulation_app = simulation_app

    @abstractmethod
    def build(self) -> None:
        ...

    @abstractmethod
    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        ...

    @abstractmethod
    def initial_reset(self) -> torch.Tensor:
        ...

    @abstractmethod
    def collect(self, obs: torch.Tensor) -> dict:
        ...

    @abstractmethod
    def update(self, rollout: dict, collect_time: float) -> dict:
        ...

    @abstractmethod
    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        ...

    @abstractmethod
    def log_banner(self) -> None:
        ...


    @property
    @abstractmethod
    def policy(self) -> torch.nn.Module: ...

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Optimizer: ...

    @property
    @abstractmethod
    def horizon(self) -> int: ...


    @abstractmethod
    def extra_checkpoint_state(self) -> dict:
        ...

    @abstractmethod
    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        ...

    @abstractmethod
    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        ...

    def deployment_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action payload used by validation and playback."""
        return self.deterministic_actions(obs)
