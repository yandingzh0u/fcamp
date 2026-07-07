from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(last, int(hidden)))
        layers.append(_activation(activation))
        last = int(hidden)
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class FlowChunkValueCritic(nn.Module):
    """Flow-matching value critic for SFPO chunk-start values.

    This is a drop-in replacement for the scalar MLP value function: it keeps
    the same ``evaluate(obs)`` interface, but represents the value sample with
    a scalar flow. SFPO still decides the target semantics outside this module
    (for example f366's cross-chunk GAE target). Training uses conditional
    flow matching between Gaussian noise and that scalar value target:

        y_t = (1 - t) * eps + t * y_target
        v_target = y_target - eps

    There is no action-conditioned Q head, distilled critic, target critic, or
    auxiliary objective here.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        *,
        flow_steps: int = 4,
        noise_std: float = 1.0,
        eval_samples: int = 8,
    ):
        super().__init__()
        if flow_steps < 1:
            raise ValueError(f"flow_steps must be >= 1, got {flow_steps}")
        if noise_std <= 0.0:
            raise ValueError(f"noise_std must be > 0, got {noise_std}")
        if eval_samples < 1:
            raise ValueError(f"eval_samples must be >= 1, got {eval_samples}")
        self.obs_dim = int(obs_dim)
        self.flow_steps = int(flow_steps)
        self.noise_std = float(noise_std)
        self.eval_samples = int(eval_samples)
        self.velocity = _build_mlp(self.obs_dim + 2, tuple(hidden_dims), 1, activation)

    @staticmethod
    def _expand_condition(condition: torch.Tensor, num_samples: int) -> torch.Tensor:
        return condition.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, condition.shape[-1])

    def _velocity(self, condition: torch.Tensor, value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        if time.ndim == 1:
            time = time.unsqueeze(-1)
        return self.velocity(torch.cat([condition, value, time], dim=-1)).squeeze(-1)

    def _base_points(self, num_samples: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if num_samples == 1:
            return torch.zeros(1, device=device, dtype=dtype)
        probs = (torch.arange(num_samples, device=device, dtype=dtype) + 0.5) / float(num_samples)
        return torch.special.ndtri(probs).clamp(-3.0, 3.0) * self.noise_std

    def sample_v(
        self,
        obs: torch.Tensor,
        num_samples: int | None = None,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        num = int(num_samples or self.eval_samples)
        batch = obs.shape[0]
        if deterministic:
            z = self._base_points(num, device=obs.device, dtype=obs.dtype).repeat(batch)
        else:
            z = torch.randn(batch * num, device=obs.device, dtype=obs.dtype) * self.noise_std
        cond = self._expand_condition(obs, num)
        dt = 1.0 / float(self.flow_steps)
        for step in range(self.flow_steps):
            t = torch.full_like(z, float(step) * dt)
            z = z + dt * self._velocity(cond, z, t)
        return z.view(batch, num)

    def mean_v(self, obs: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
        return self.sample_v(obs, num_samples=num_samples, deterministic=True).mean(dim=1, keepdim=True)

    def evaluate(self, obs: torch.Tensor) -> torch.Tensor:
        """Compatibility shim for code paths that expect a scalar V critic."""
        return self.mean_v(obs)

    def flow_matching_loss_v(
        self,
        obs: torch.Tensor,
        target_return: torch.Tensor,
        *,
        fm_samples: int = 1,
    ) -> torch.Tensor:
        if target_return.ndim != 1:
            target_return = target_return.reshape(-1)
        if obs.shape[0] != target_return.shape[0]:
            raise ValueError("obs and target_return batch sizes must match")
        if fm_samples < 1:
            raise ValueError(f"fm_samples must be >= 1, got {fm_samples}")

        batch = obs.shape[0]
        cond = self._expand_condition(obs, fm_samples)
        target = target_return.unsqueeze(1).expand(-1, fm_samples).reshape(-1)
        eps = torch.randn_like(target) * self.noise_std
        t = torch.rand(batch * fm_samples, device=obs.device, dtype=obs.dtype)
        y_t = (1.0 - t) * eps + t * target
        target_velocity = target - eps
        pred_velocity = self._velocity(cond, y_t, t)
        return (pred_velocity - target_velocity).square().view(batch, fm_samples).mean(dim=1)
