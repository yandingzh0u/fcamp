from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .noise_bank import CollectorMode, OVERLAP_ELIGIBLE_MODES

PRIMARY_OVERLAP_MODES = OVERLAP_ELIGIBLE_MODES


@dataclass(frozen=True, slots=True)
class ActionRecord:
    mean: torch.Tensor
    std: torch.Tensor
    common_epsilon: torch.Tensor
    sampled: torch.Tensor
    applied: torch.Tensor


def policy_mean_and_std(algo: Any, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the checkpoint mean/std without mutating normalizer statistics."""

    normalized = algo.actor_obs_normalizer(observation, update=False)
    mean = algo.actor.act_inference(normalized)
    std_parameter = algo.actor.std
    std = std_parameter.unsqueeze(0).expand_as(mean)
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise FloatingPointError("policy mean/std contains NaN or Inf")
    if bool((std < 0.0).any()):
        raise ValueError("policy std must be non-negative")
    return mean, std


def construct_action_record(
    mean: torch.Tensor,
    native_std: torch.Tensor,
    common_epsilon: torch.Tensor,
    *,
    mode: CollectorMode | str,
    common_sigma: float = 0.0,
    action_low: torch.Tensor | None = None,
    action_high: torch.Tensor | None = None,
) -> ActionRecord:
    mode = CollectorMode(mode)
    if mean.shape != native_std.shape or mean.shape != common_epsilon.shape:
        raise ValueError("mean, std, and epsilon must have identical shapes")
    if not bool(
        torch.isfinite(mean).all()
        and torch.isfinite(native_std).all()
        and torch.isfinite(common_epsilon).all()
    ):
        raise FloatingPointError("action inputs contain NaN or Inf")
    if common_sigma < 0.0:
        raise ValueError("common_sigma must be non-negative")

    if mode in (CollectorMode.CLEAN_MEAN, CollectorMode.CONTROLLED_ENVIRONMENT):
        sampled = mean
    elif mode is CollectorMode.COMMON_ACTION_NOISE:
        sampled = mean + float(common_sigma) * common_epsilon
    else:
        # Using a common epsilon exposes only checkpoint std scale while
        # removing an irrelevant difference in random draws across policies.
        sampled = mean + native_std * common_epsilon

    if action_low is None and action_high is None:
        applied = sampled
    elif action_low is None or action_high is None:
        raise ValueError("both action bounds must be supplied")
    else:
        low = torch.as_tensor(action_low, device=sampled.device, dtype=sampled.dtype)
        high = torch.as_tensor(action_high, device=sampled.device, dtype=sampled.dtype)
        applied = torch.maximum(torch.minimum(sampled, high), low)
    return ActionRecord(
        mean=mean.detach().clone(),
        std=native_std.detach().clone(),
        common_epsilon=common_epsilon.detach().clone(),
        sampled=sampled.detach().clone(),
        applied=applied.detach().clone(),
    )


def collector_environment_flags(mode: CollectorMode | str) -> dict[str, bool]:
    mode = CollectorMode(mode)
    if mode is CollectorMode.CLEAN_MEAN:
        return {
            "observation_noise": False,
            "reset_noise": False,
            "interval_pushes": False,
        }
    return {
        "observation_noise": True,
        "reset_noise": True,
        "interval_pushes": True,
    }


class TensorTreeAccumulator:
    """Accumulate aligned per-step tensor trees and stack on the CPU."""

    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []

    @staticmethod
    def _clone_tree(value: Any) -> Any:
        if torch.is_tensor(value):
            if not bool(torch.isfinite(value).all()) and value.is_floating_point():
                raise FloatingPointError("rollout record contains NaN or Inf")
            return value.detach().cpu().clone()
        if isinstance(value, Mapping):
            return {str(key): TensorTreeAccumulator._clone_tree(nested) for key, nested in value.items()}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        raise TypeError(f"unsupported rollout record value {type(value).__name__}")

    def append(self, record: Mapping[str, Any]) -> None:
        self._steps.append(self._clone_tree(dict(record)))

    @staticmethod
    def _stack(values: list[Any]) -> Any:
        first = values[0]
        if torch.is_tensor(first):
            if any(not torch.is_tensor(value) or value.shape != first.shape for value in values):
                raise ValueError("rollout tensor shape changed across steps")
            return torch.stack(values)
        if isinstance(first, dict):
            keys = set(first)
            if any(not isinstance(value, dict) or set(value) != keys for value in values):
                raise ValueError("rollout record keys changed across steps")
            return {
                key: TensorTreeAccumulator._stack([value[key] for value in values])
                for key in sorted(keys)
            }
        if any(value != first for value in values):
            return list(values)
        return first

    def finalize(self) -> dict[str, Any]:
        if not self._steps:
            raise ValueError("cannot finalize an empty rollout")
        return self._stack(self._steps)


REQUIRED_ROLLOUT_SECTIONS = frozenset(
    {"metadata", "trajectory", "observation", "state", "action", "reference", "outcome"}
)


def validate_rollout_tree(tree: Mapping[str, Any]) -> None:
    missing = REQUIRED_ROLLOUT_SECTIONS - set(tree)
    unknown = set(tree) - REQUIRED_ROLLOUT_SECTIONS
    if missing or unknown:
        raise ValueError(
            f"rollout sections mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    metadata = tree["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("rollout metadata must be a mapping")
    mode = CollectorMode(str(metadata.get("collector_mode")))
    if bool(metadata.get("eligible_for_primary_overlap", mode in PRIMARY_OVERLAP_MODES)) != (
        mode in PRIMARY_OVERLAP_MODES
    ):
        raise ValueError("collector overlap eligibility contradicts collector mode")


def save_rollout_shard(tree: Mapping[str, Any], path: str | Path) -> Path:
    validate_rollout_tree(tree)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(tree), temporary)
    temporary.replace(destination)
    return destination


def deterministic_action_equivalence(
    algo: Any,
    observation: torch.Tensor,
    *,
    atol: float = 0.0,
) -> dict[str, float | bool]:
    expected = algo.deterministic_action(observation)
    mean, _ = policy_mean_and_std(algo, observation)
    maximum_error = float(torch.max(torch.abs(expected - mean)).item())
    return {
        "equivalent": bool(maximum_error <= float(atol)),
        "max_abs_error": maximum_error,
        "atol": float(atol),
    }


def same_state_branch_identity(
    restore: Callable[[], None],
    branch: Callable[[], Mapping[str, torch.Tensor]],
) -> dict[str, float | bool]:
    """Run an identical branch twice and fail if snapshot replay drifts."""

    restore()
    first = {key: value.detach().cpu().clone() for key, value in branch().items()}
    restore()
    second = {key: value.detach().cpu().clone() for key, value in branch().items()}
    if set(first) != set(second):
        raise ValueError("branch record keys changed")
    maximum = 0.0
    for key in first:
        if first[key].shape != second[key].shape:
            raise ValueError(f"branch tensor shape changed for {key}")
        maximum = max(maximum, float(torch.max(torch.abs(first[key] - second[key])).item()))
    return {"identical": maximum == 0.0, "max_abs_error": maximum}
