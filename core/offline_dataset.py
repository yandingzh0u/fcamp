from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


FORMAT_NAME = "mimic_offline_transitions_v1"
TENSOR_KEYS = ("observations", "actions", "rewards", "next_observations", "masks")


def _as_float_matrix(value: Any, name: str, width: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if tensor.ndim == 1 and width == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(f"Offline dataset {name} must have shape [N, {width}], got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Offline dataset {name} contains non-finite values")
    return tensor.contiguous()


def validate_offline_transition_payload(
    payload: dict[str, Any],
    *,
    obs_dim: int,
    action_dim: int,
    action_limit: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if payload.get("format") != FORMAT_NAME:
        raise ValueError(
            f"Unsupported offline dataset format {payload.get('format')!r}; expected {FORMAT_NAME!r}"
        )
    missing = set(TENSOR_KEYS) - payload.keys()
    if missing:
        raise KeyError(f"Offline dataset missing tensors: {sorted(missing)}")

    tensors = {
        "observations": _as_float_matrix(payload["observations"], "observations", obs_dim),
        "actions": _as_float_matrix(payload["actions"], "actions", action_dim),
        "rewards": _as_float_matrix(payload["rewards"], "rewards", 1),
        "next_observations": _as_float_matrix(
            payload["next_observations"], "next_observations", obs_dim
        ),
        "masks": _as_float_matrix(payload["masks"], "masks", 1),
    }
    sizes = {tensor.shape[0] for tensor in tensors.values()}
    if len(sizes) != 1:
        raise ValueError(
            "Offline dataset tensors have different transition counts: "
            + ", ".join(f"{name}={tensor.shape[0]}" for name, tensor in tensors.items())
        )
    transition_count = sizes.pop()
    if transition_count < 1:
        raise ValueError("Offline dataset must contain at least one transition")
    if bool(((tensors["masks"] < 0.0) | (tensors["masks"] > 1.0)).any()):
        raise ValueError("Offline dataset masks must be in [0, 1]")
    observed_action_limit = float(tensors["actions"].abs().max().item())
    if observed_action_limit > float(action_limit) + 1e-5:
        raise ValueError(
            f"Offline actions reach {observed_action_limit:.6f}, above FQL action limit {action_limit:.6f}"
        )

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("Offline dataset metadata must be a dictionary")
    metadata = dict(metadata)
    metadata["transition_count"] = int(transition_count)
    metadata["observed_action_limit"] = observed_action_limit
    return tensors, metadata


def load_offline_transition_dataset(
    path: str | Path,
    *,
    obs_dim: int,
    action_dim: int,
    action_limit: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Offline dataset not found: {dataset_path}")
    payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Offline dataset must contain a dictionary, got {type(payload).__name__}")
    tensors, metadata = validate_offline_transition_payload(
        payload,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_limit=action_limit,
    )
    metadata["path"] = str(dataset_path)
    return tensors, metadata


def save_offline_transition_dataset(
    path: str | Path,
    *,
    observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_observations: torch.Tensor,
    masks: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT_NAME,
        "metadata": dict(metadata or {}),
        "observations": observations.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        "actions": actions.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        "rewards": rewards.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        "next_observations": next_observations.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous(),
        "masks": masks.detach().to(device="cpu", dtype=torch.float32).contiguous(),
    }
    torch.save(payload, output_path)
    return output_path
