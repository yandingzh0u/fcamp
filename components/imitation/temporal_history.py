"""Per-environment causal imitation history with explicit episode age."""

from __future__ import annotations

import torch

from components.imitation.motion_features import canonicalize_imitation_window


class TemporalFeatureHistory:
    """A vectorized per-environment raw-frame ring, returned oldest-to-newest.

    Reset stores one real simulator frame at age 0. That frame can be a
    predecessor, but it is never legal discriminator input. The first legal
    window is exactly ages 1..W.
    """

    def __init__(
        self,
        num_envs: int,
        history_len: int = 16,
        feature_dim: int = 233,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_envs <= 0 or history_len <= 0 or feature_dim <= 0:
            raise ValueError("num_envs, history_len and feature_dim must be positive")
        self.num_envs = int(num_envs)
        self.history_len = int(history_len)
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)
        self.dtype = dtype
        self._data = torch.zeros(
            (self.num_envs, self.history_len, self.feature_dim),
            device=self.device,
            dtype=dtype,
        )
        if dtype != torch.float32:
            raise ValueError("imitation history must store lossless float32 frames")
        self._slot_ages = torch.full(
            (self.num_envs, self.history_len),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        # Cursor is the slot overwritten by the next post-action push.
        self._cursor = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._initialized = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._age = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)

    @property
    def initialized(self) -> torch.Tensor:
        return self._initialized.clone()

    @property
    def push_count(self) -> torch.Tensor:
        return torch.clamp(self._age, min=0).clone()

    @property
    def ages(self) -> torch.Tensor:
        return self._age.clone()

    @property
    def ready(self) -> torch.Tensor:
        return self._initialized & (self._age >= self.history_len)

    def _ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.ndim != 1:
            raise ValueError(f"env_ids must be 1-D, got {tuple(ids.shape)}")
        if ids.numel() and (bool((ids < 0).any()) or bool((ids >= self.num_envs).any())):
            raise IndexError("env_ids contains an out-of-range environment")
        if ids.numel() != torch.unique(ids).numel():
            raise ValueError("env_ids must not contain duplicates")
        return ids

    def _check_frames(self, name: str, frames: torch.Tensor, count: int) -> torch.Tensor:
        expected = (count, self.feature_dim)
        if tuple(frames.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(frames.shape)}")
        if frames.dtype != torch.float32:
            raise TypeError(f"{name} must be float32, got {frames.dtype}")
        values = frames.detach().to(device=self.device, dtype=self.dtype)
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"{name} contains non-finite imitation features")
        return values

    @torch.no_grad()
    def reset(self, initial_frame: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Reset selected rings with one simulator frame ``[M,F]`` at age 0."""

        ids = self._ids(env_ids)
        frames = self._check_frames("initial_frame", initial_frame, ids.numel())
        if ids.numel() == 0:
            return
        self._data[ids] = 0.0
        self._slot_ages[ids] = -1
        self._data[ids, 0] = frames
        self._slot_ages[ids, 0] = 0
        self._cursor[ids] = 1 % self.history_len
        self._initialized[ids] = True
        self._age[ids] = 0

    @torch.no_grad()
    def push(self, frame: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Append post-action frames for selected environments."""

        ids = self._ids(env_ids)
        frames = self._check_frames("frame", frame, ids.numel())
        if ids.numel() == 0:
            return
        if not bool(self._initialized[ids].all()):
            bad = ids[~self._initialized[ids]].detach().cpu().tolist()
            raise RuntimeError(f"imitation history must be reset before push; uninitialized={bad}")
        slots = self._cursor[ids]
        next_age = self._age[ids] + 1
        self._data[ids, slots] = frames
        self._slot_ages[ids, slots] = next_age
        self._cursor[ids] = torch.remainder(slots + 1, self.history_len)
        self._age[ids] = next_age

    def window_ages(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        ids = self._ids(env_ids)
        if not bool(self._initialized[ids].all()):
            bad = ids[~self._initialized[ids]].detach().cpu().tolist()
            raise RuntimeError(f"requested uninitialized imitation histories: {bad}")
        if not bool(self.ready[ids].all()):
            bad = ids[~self.ready[ids]].detach().cpu().tolist()
            ages = self._age[ids][~self.ready[ids]].detach().cpu().tolist()
            raise RuntimeError(f"imitation history not ready: env_ids={bad}, ages={ages}")
        offsets = torch.arange(self.history_len, device=self.device)
        chronological = torch.remainder(self._cursor[ids, None] + offsets[None, :], self.history_len)
        slot_ages = torch.gather(self._slot_ages[ids], 1, chronological)
        expected = self._age[ids, None] - (self.history_len - 1) + offsets[None, :]
        if not torch.equal(slot_ages, expected) or bool((slot_ages <= 0).any()):
            raise RuntimeError("corrupt imitation history: legal window ages must be positive and contiguous")
        return slot_ages

    def window(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return legal chronological windows with shape ``[M,W,F]``."""

        ids = self._ids(env_ids)
        self.window_ages(ids)
        offsets = torch.arange(self.history_len, device=self.device)
        chronological = torch.remainder(self._cursor[ids, None] + offsets[None, :], self.history_len)
        return torch.gather(
            self._data[ids],
            1,
            chronological[..., None].expand(-1, -1, self.feature_dim),
        )

    def flatten(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        canonicalize_root: bool = False,
    ) -> torch.Tensor:
        windows = self.window(env_ids)
        if canonicalize_root:
            windows = canonicalize_imitation_window(windows)
        return windows.reshape(windows.shape[0], self.history_len * self.feature_dim)

    def statistics(self) -> dict[str, float]:
        initialized = self._initialized
        ready = self.ready
        metrics = {
            "history/initialized_fraction": float(self._initialized.float().mean().item()),
            "history/ready_fraction": float(ready.float().mean().item()),
            "history/ready_count": float(ready.sum().item()),
            "history/age0_count": float((initialized & (self._age == 0)).sum().item()),
            "history/age0_in_legal_window_count": 0.0,
        }
        if bool(initialized.any()):
            ages = self._age[initialized]
            metrics.update(
                {
                    "history/age_mean": float(ages.float().mean().item()),
                    "history/age_min": float(ages.min().item()),
                    "history/age_max": float(ages.max().item()),
                    "history/push_count_mean": float(ages.float().mean().item()),
                    "history/push_count_max": float(ages.max().item()),
                }
            )
        else:
            metrics.update(
                {
                    "history/age_mean": 0.0,
                    "history/age_min": -1.0,
                    "history/age_max": -1.0,
                    "history/push_count_mean": 0.0,
                    "history/push_count_max": 0.0,
                }
            )
        return metrics
