from __future__ import annotations

from abc import ABC, abstractmethod

import torch


def classify_mimickit_done_terms(
    done: torch.Tensor,
    done_terms: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply MimicKit's TIME -> SUCC -> FAIL overwrite precedence."""
    done = done.bool()
    failure = done & (
        done_terms["anchor_pos_bad"].bool()
        | done_terms["anchor_ori_bad"].bool()
        | done_terms["ee_body_bad"].bool()
    )
    motion_complete_term = done_terms.get("motion_complete")
    motion_complete = (
        done & motion_complete_term.bool() & ~failure
        if torch.is_tensor(motion_complete_term)
        else torch.zeros_like(done)
    )
    timeout = done & done_terms["time_out"].bool() & ~motion_complete & ~failure
    return timeout, motion_complete, failure


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

    def reset_after_resume(self) -> torch.Tensor | None:
        """Optionally rebuild method-owned rollout state after checkpoint restore.

        The checkpointer invokes this only after the adaptive sampler and random
        number generator states have been restored.  Returning an observation
        replaces the trainer's current observation; returning ``None`` keeps the
        observation produced by :meth:`initial_reset`.
        """
        return None

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

    def evaluation_reset(self, phase_indices: torch.Tensor) -> torch.Tensor:
        """Reset for validation/playback and return this method's policy observation."""
        return self.env.reset(phase_indices=phase_indices)

    def evaluation_step(
        self,
        actions: torch.Tensor,
        reference_dt: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance one clean evaluator transition.

        Methods with private observation state (AdaMimic history, for example)
        override this while retaining the shared evaluator reward/termination.
        """
        return self.env.step(
            actions,
            auto_reset=False,
            reference_dt=reference_dt,
        )

    def snapshot_runtime_state(self):
        """Return method-owned rollout state that validation must restore."""
        return None

    def restore_runtime_state(self, state) -> None:
        """Restore state returned by :meth:`snapshot_runtime_state`."""
        del state
