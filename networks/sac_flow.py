from __future__ import annotations

import math

import torch
from torch import nn


class BatchRenorm1d(nn.Module):
    """PyTorch port of the BatchRenorm block used by SAC Flow's CrossQ code."""

    def __init__(
        self,
        features: int,
        momentum: float = 0.99,
        epsilon: float = 1.0e-3,
        warmup_steps: int = 100_000,
    ) -> None:
        super().__init__()
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.warmup_steps = int(warmup_steps)
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_var", torch.ones(features))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    def forward(self, inputs: torch.Tensor, *, update_stats: bool) -> torch.Tensor:
        if not update_stats:
            normalized = (inputs - self.running_mean) / torch.sqrt(
                self.running_var + self.epsilon
            )
            return normalized * self.weight + self.bias

        mean = inputs.mean(dim=0)
        variance = inputs.var(dim=0, unbiased=False)
        custom_mean = mean
        custom_variance = variance
        if int(self.steps.item()) >= self.warmup_steps:
            std = torch.sqrt(variance + self.epsilon)
            running_std = torch.sqrt(self.running_var + self.epsilon)
            ratio = (std / running_std).detach().clamp(1.0 / 3.0, 3.0)
            drift = ((mean - self.running_mean) / running_std).detach().clamp(-5.0, 5.0)
            custom_variance = variance / ratio.square()
            custom_mean = mean - drift * torch.sqrt(variance + self.epsilon) / ratio
        with torch.no_grad():
            self.running_mean.mul_(self.momentum).add_(mean, alpha=1.0 - self.momentum)
            self.running_var.mul_(self.momentum).add_(variance, alpha=1.0 - self.momentum)
            self.steps.add_(1)
        normalized = (inputs - custom_mean) / torch.sqrt(custom_variance + self.epsilon)
        return normalized * self.weight + self.bias


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 4 or embedding_dim % 2:
            raise ValueError("SAC Flow time embedding must be even and >= 4")
        half = embedding_dim // 2
        frequency = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / (half - 1)
        )
        self.register_buffer("frequency", frequency)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 1:
            time = time.unsqueeze(-1)
        angles = time * self.frequency.unsqueeze(0)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


def _mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    *,
    activation: str = "silu",
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        nonlinearity: nn.Module = nn.SiLU() if activation == "silu" else nn.ReLU()
        layers.extend((nn.Linear(current_dim, hidden_dim), nonlinearity))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class SACFlowActor(nn.Module):
    """Official SAC Flow-G actor with a stochastic likelihood-bearing flow path."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        flow_steps: int,
        action_scale: float,
        time_embed_dim: int = 32,
        log_std_hidden_dims: tuple[int, ...] = (512, 512),
        use_batch_renorm: bool = True,
        batch_norm_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        if flow_steps < 1:
            raise ValueError("SAC Flow flow_steps must be positive")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.flow_steps = int(flow_steps)
        self.action_scale = float(action_scale)
        self.log_std_min = -5.0
        self.log_std_max = 2.0
        self.use_batch_renorm = bool(use_batch_renorm)

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, 2 * time_embed_dim),
            nn.SiLU(),
            nn.Linear(2 * time_embed_dim, time_embed_dim),
        )
        flow_input_dim = self.obs_dim + self.action_dim + time_embed_dim
        self.gate_net = _mlp(flow_input_dim, (hidden_dim,), self.action_dim)
        self.candidate_net = _mlp(flow_input_dim, (hidden_dim,), self.action_dim)
        self.log_std_net = _mlp(
            self.obs_dim, log_std_hidden_dims, self.action_dim, activation="relu"
        )
        self.input_norm = BatchRenorm1d(
            self.obs_dim + self.action_dim, momentum=batch_norm_momentum
        )

        gate_output = self.gate_net[-1]
        assert isinstance(gate_output, nn.Linear)
        nn.init.zeros_(gate_output.weight)
        nn.init.constant_(gate_output.bias, 5.0)

    def flow_step(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        time: torch.Tensor,
        *,
        update_stats: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_action = torch.cat((observation, action), dim=-1)
        if self.use_batch_renorm:
            state_action = self.input_norm(state_action, update_stats=update_stats)
        time_embedding = self.time_embedding(time)
        inputs = torch.cat((state_action, time_embedding), dim=-1)
        gate = torch.sigmoid(self.gate_net(inputs))
        candidate = self.candidate_net(inputs)
        velocity = gate * (candidate - action)
        raw_log_std = torch.tanh(self.log_std_net(observation))
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (raw_log_std + 1.0)
        return velocity, log_std, gate

    def sample(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
        update_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = observation.shape[0]
        if deterministic:
            action = torch.zeros(
                batch_size,
                self.action_dim,
                device=observation.device,
                dtype=observation.dtype,
            )
        else:
            action = torch.randn(
                batch_size,
                self.action_dim,
                device=observation.device,
                dtype=observation.dtype,
            )
        base_distribution = torch.distributions.Normal(torch.zeros_like(action), torch.ones_like(action))
        total_log_prob = base_distribution.log_prob(action).sum(-1, keepdim=True)
        dt = 1.0 / self.flow_steps
        std_values: list[torch.Tensor] = []
        gate_values: list[torch.Tensor] = []
        for step in range(self.flow_steps):
            time = torch.full(
                (batch_size, 1),
                step * dt,
                device=observation.device,
                dtype=observation.dtype,
            )
            velocity, log_std, gate = self.flow_step(
                observation, action, time, update_stats=update_stats
            )
            std = log_std.exp()
            mean_next = action + dt * velocity
            noise = torch.zeros_like(action) if deterministic else torch.randn_like(action)
            action = mean_next + std * noise
            transition = torch.distributions.Normal(mean_next, std)
            total_log_prob = total_log_prob + transition.log_prob(action).sum(-1, keepdim=True)
            std_values.append(std)
            gate_values.append(gate)

        squashed = torch.tanh(action)
        environment_action = self.action_scale * squashed
        correction = torch.log(
            self.action_scale * (1.0 - squashed.square()) + 1.0e-6
        ).sum(-1, keepdim=True)
        total_log_prob = total_log_prob - correction
        return environment_action, total_log_prob, {
            "path_std": torch.stack(std_values, dim=1),
            "gate": torch.stack(gate_values, dim=1),
            "terminal_latent": action,
        }


class SACFlowQBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        use_batch_renorm: bool,
        batch_norm_momentum: float,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("SAC Flow critic requires at least one hidden layer")
        self.use_batch_renorm = bool(use_batch_renorm)
        dimensions = (input_dim, *hidden_dims)
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(len(hidden_dims))
        )
        self.norms = nn.ModuleList(
            BatchRenorm1d(dimension, momentum=batch_norm_momentum)
            for dimension in dimensions
        )
        self.output = nn.Linear(hidden_dims[-1], 1)

    def forward(self, inputs: torch.Tensor, *, update_stats: bool) -> torch.Tensor:
        if self.use_batch_renorm:
            inputs = self.norms[0](inputs, update_stats=update_stats)
        hidden = inputs
        for index, layer in enumerate(self.layers):
            hidden = torch.relu(layer(hidden))
            if self.use_batch_renorm:
                hidden = self.norms[index + 1](hidden, update_stats=update_stats)
        return self.output(hidden)


class SACFlowTwinQ(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        use_batch_renorm: bool = True,
        batch_norm_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        input_dim = int(obs_dim) + int(action_dim)
        self.q1 = SACFlowQBranch(
            input_dim, hidden_dims, use_batch_renorm, batch_norm_momentum
        )
        self.q2 = SACFlowQBranch(
            input_dim, hidden_dims, use_batch_renorm, batch_norm_momentum
        )

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        *,
        update_stats: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((observation, action), dim=-1)
        return (
            self.q1(inputs, update_stats=update_stats),
            self.q2(inputs, update_stats=update_stats),
        )
