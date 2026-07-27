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

    def pre_training_warmup(
        self,
        current_observation: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], int]:
        """Optionally run work before the first formal training update.

        The returned tuple contains the observation from which formal training
        should continue, method-specific metrics, and the exact number of
        environment transitions consumed by the warm-up.  The trainer accounts
        those transitions and the elapsed wall time without advancing the formal
        update index.  Checkpoint resumes deliberately skip this hook.
        """
        return current_observation, {}, 0

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

    def validate_checkpoint_payload(self, payload: dict) -> None:
        """Reject a checkpoint whose method contract is not deployable.

        Stateful methods override this hook and validate their schema before
        any parameters are loaded.  The no-op default keeps stateless methods
        source-compatible with the shared trainer and player.
        """
        del payload

    def deployment_chunk(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the deterministic payload cached for one policy chunk.

        The payload is deliberately *not* required to be an absolute action.
        For example, FCAMP returns raw target-rate coordinates here and decodes
        exactly one row at each call to :meth:`evaluation_step_payload`.
        """
        return self.deterministic_actions(obs)

    def validated_deployment_chunk(self, obs: torch.Tensor) -> torch.Tensor:
        """Return a shape-checked ``[N,H,D]`` deployment payload."""
        payload = self.deployment_chunk(obs)
        if payload.dim() == 2:
            payload = payload.unsqueeze(1)
        if payload.dim() != 3:
            raise ValueError(
                "deployment_chunk must return [N,D] or [N,H,D], got "
                f"shape={tuple(payload.shape)}"
            )
        expected_prefix = (self.env.num_envs, self.horizon)
        if payload.shape[:2] != expected_prefix:
            raise ValueError(
                "deployment_chunk batch/time shape must be "
                f"{expected_prefix}, got {tuple(payload.shape[:2])}"
            )
        return payload

    def split_deployment_frame(
        self,
        frame_payload: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Separate an optional legacy reference-time channel."""
        if not bool(getattr(self, "uses_reference_dt", False)):
            return frame_payload, None
        expected_dim = int(self.env.action_dim) + 1
        if frame_payload.shape[-1] != expected_dim:
            raise ValueError(
                f"{self.__class__.__name__} uses reference_dt but returned "
                f"payload dim {frame_payload.shape[-1]}, expected {expected_dim}"
            )
        return frame_payload[..., : self.env.action_dim], frame_payload[..., -1]

    def require_applied_action(self, info: dict) -> torch.Tensor:
        """Return the normalized command actually sent by the environment."""
        action = info.get("applied_action")
        if not torch.is_tensor(action):
            raise RuntimeError(
                "evaluation_step_payload must return info['applied_action']; "
                "raw policy coordinates are not execution diagnostics"
            )
        if action.shape != self.env.last_action.shape:
            raise RuntimeError(
                "info['applied_action'] must have shape "
                f"{tuple(self.env.last_action.shape)}, got {tuple(action.shape)}"
            )
        return action

    def evaluation_reset(self, phase_indices: torch.Tensor) -> torch.Tensor:
        """Reset for validation/playback and return this method's policy observation."""
        return self.env.reset(phase_indices=phase_indices)

    def evaluation_step(
        self,
        actions: torch.Tensor,
        reference_dt: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance one clean evaluator transition.

        Methods with private observation state override this while retaining
        the shared evaluator reward/termination.
        """
        return self.env.step(
            actions,
            auto_reset=False,
            reference_dt=reference_dt,
        )

    def evaluation_step_payload(
        self,
        frame_payload: torch.Tensor,
        reference_dt: torch.Tensor | None,
        *,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Decode and execute one cached deployment frame.

        The default implementation treats ``frame_payload`` as an absolute
        environment action.  Stateful methods override this method so the
        payload is decoded against the environment's current carried state at
        the instant it is executed.  Inactive environments hold their last
        applied command; callers must never manufacture a zero action.
        """
        actions = frame_payload
        if active_mask is not None:
            if active_mask.dtype != torch.bool or active_mask.shape != (self.env.num_envs,):
                raise ValueError(
                    "active_mask must be bool [num_envs], got "
                    f"dtype={active_mask.dtype} shape={tuple(active_mask.shape)}"
                )
            if actions.shape != self.env.last_action.shape:
                raise ValueError(
                    "The default deployment step only accepts absolute actions "
                    f"with shape {tuple(self.env.last_action.shape)}, got "
                    f"{tuple(actions.shape)}"
                )
            actions = torch.where(
                active_mask.unsqueeze(-1),
                actions,
                self.env.last_action,
            )
            if reference_dt is not None:
                reference_dt = torch.where(
                    active_mask,
                    reference_dt,
                    torch.full_like(reference_dt, float(self.env.dt)),
                )
        return self.evaluation_step(actions, reference_dt)

    def snapshot_runtime_state(self):
        """Return method-owned rollout state that validation must restore."""
        return None

    def restore_runtime_state(self, state) -> None:
        """Restore state returned by :meth:`snapshot_runtime_state`."""
        del state
