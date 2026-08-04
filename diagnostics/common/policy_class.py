from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def action_prediction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    normalization_scale: np.ndarray | None = None,
) -> dict[str, object]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction/target must share shape [N,A]")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise FloatingPointError("action arrays contain NaN or Inf")
    residual = prediction - target
    rmse_joint = np.sqrt(np.mean(np.square(residual), axis=0))
    if normalization_scale is None:
        scale = np.std(target, axis=0)
    else:
        scale = np.asarray(normalization_scale, dtype=np.float64).reshape(-1)
        if scale.shape != (target.shape[1],):
            raise ValueError("normalization_scale has wrong action dimension")
    scale = np.maximum(scale, 1.0e-6)
    nrmse_joint = rmse_joint / scale
    target_centered = target - target.mean(axis=0, keepdims=True)
    denominator = np.square(target_centered).sum(axis=0)
    r2_joint = 1.0 - np.square(residual).sum(axis=0) / np.maximum(denominator, 1.0e-12)
    return {
        "nrmse": float(np.sqrt(np.mean(np.square(residual))) / np.sqrt(np.mean(np.square(scale)))),
        "nrmse_per_joint": nrmse_joint.tolist(),
        "r2": float(1.0 - np.square(residual).sum() / max(float(np.square(target_centered).sum()), 1.0e-12)),
        "r2_per_joint": r2_joint.tolist(),
        "sample_count": int(target.shape[0]),
    }


def build_history_windows(
    observations: np.ndarray,
    actions: np.ndarray,
    trajectory_ids: Sequence[str | int],
    *,
    history: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    trajectory_ids = np.asarray(trajectory_ids)
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("observations/actions must have shape [N,D]")
    if observations.shape[0] != actions.shape[0] or trajectory_ids.shape != (actions.shape[0],):
        raise ValueError("history inputs are not aligned")
    if int(history) < 1:
        raise ValueError("history must be positive")
    windows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    end_indices: list[int] = []
    for end in range(int(history) - 1, observations.shape[0]):
        start = end + 1 - int(history)
        if np.all(trajectory_ids[start : end + 1] == trajectory_ids[end]):
            windows.append(observations[start : end + 1])
            targets.append(actions[end])
            end_indices.append(end)
    if not windows:
        raise ValueError("no valid within-trajectory history windows")
    return np.stack(windows), np.stack(targets), np.asarray(end_indices, dtype=np.int64)


def knn_action_aliasing(
    features: np.ndarray,
    actions: np.ndarray,
    phases: np.ndarray,
    contact_modes: np.ndarray,
    *,
    k: int,
    action_delta: float,
    phase_period: float,
) -> dict[str, float]:
    features = np.asarray(features, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64).reshape(-1)
    contact_modes = np.asarray(contact_modes).reshape(-1)
    count = features.shape[0]
    if features.ndim != 2 or actions.ndim != 2:
        raise ValueError("features/actions must have shape [N,D]")
    if actions.shape[0] != count or phases.shape != (count,) or contact_modes.shape != (count,):
        raise ValueError("aliasing inputs are not aligned")
    if count <= int(k) or int(k) < 1:
        raise ValueError("k must be smaller than sample count")
    if float(phase_period) <= 0.0:
        raise ValueError("phase_period must be positive")
    if not all(np.isfinite(value).all() for value in (features, actions, phases)):
        raise FloatingPointError("aliasing inputs contain NaN or Inf")

    feature_scale = np.std(features, axis=0)
    normalized = (features - np.mean(features, axis=0)) / np.maximum(feature_scale, 1.0e-6)
    indices = NearestNeighbors(n_neighbors=int(k) + 1).fit(normalized).kneighbors(
        normalized, return_distance=False
    )[:, 1:]
    neighbor_actions = actions[indices]
    action_distance = np.linalg.norm(neighbor_actions - actions[:, None, :], axis=-1) / np.sqrt(actions.shape[1])
    phase_delta = np.abs(phases[indices] - phases[:, None])
    phase_delta = np.minimum(phase_delta, float(phase_period) - np.minimum(phase_delta, float(phase_period)))
    contact_mismatch = contact_modes[indices] != contact_modes[:, None]
    conditional_variance = np.mean(np.var(neighbor_actions, axis=1))
    return {
        "sample_count": float(count),
        "k": float(k),
        "action_alias_fraction": float(np.mean(action_distance > float(action_delta))),
        "neighbor_action_distance_mean": float(action_distance.mean()),
        "neighbor_phase_delta_mean": float(phase_delta.mean()),
        "neighbor_contact_mismatch": float(contact_mismatch.mean()),
        "teacher_action_conditional_variance": float(conditional_variance),
    }


class MLPBehaviorClone(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = int(input_dim)
        for width in hidden:
            layers.extend((nn.Linear(current, int(width)), nn.ELU()))
            current = int(width)
        layers.append(nn.Linear(current, int(action_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 3:
            observation = observation[:, -1]
        return self.network(observation)


class GRUBehaviorClone(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.recurrent = nn.GRU(int(input_dim), int(hidden_dim), batch_first=True)
        self.output = nn.Linear(int(hidden_dim), int(action_dim))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 3:
            raise ValueError("GRU behavior clone expects [B,H,D]")
        encoded, _ = self.recurrent(observation)
        return self.output(encoded[:, -1])


@dataclass(frozen=True, slots=True)
class BCTrainingConfig:
    epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.0
    patience: int = 8


def train_behavior_clone(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    seed: int,
    config: BCTrainingConfig = BCTrainingConfig(),
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, float]]:
    torch.manual_seed(int(seed))
    device = torch.device(device)
    model = model.to(device)
    train_dataset = TensorDataset(
        torch.as_tensor(train_x, dtype=torch.float32),
        torch.as_tensor(train_y, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        generator=generator,
    )
    validation_inputs = torch.as_tensor(validation_x, dtype=torch.float32, device=device)
    validation_targets = torch.as_tensor(validation_y, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    completed = 0
    for epoch in range(int(config.epochs)):
        model.train()
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss = torch.mean(torch.square(model(inputs) - targets))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("behavior-clone loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                torch.mean(torch.square(model(validation_inputs) - validation_targets)).item()
            )
        completed = epoch + 1
        if validation_loss < best_loss - 1.0e-10:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(config.patience):
                break
    if best_state is None:
        raise RuntimeError("behavior clone never produced a finite checkpoint")
    model.load_state_dict(best_state)
    return model, {"validation_mse": best_loss, "epochs": float(completed)}
