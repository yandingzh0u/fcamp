"""Demo-seeded chronological history for the AMP discriminator."""

from __future__ import annotations

import torch


class TemporalFeatureHistory:
    """Per-environment ring containing one causal AMP window.

    Reset installs ``W`` chronological demonstration frames ending at the
    sampled reset state. The first real policy transition overwrites the oldest
    predecessor, so the discriminator can reward that transition immediately.
    """

    def __init__(
        self,
        num_envs: int,
        history_len: int,
        feature_dim: int,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_envs < 1 or history_len < 1 or feature_dim < 1:
            raise ValueError(
                "num_envs, history_len, and feature_dim must be positive"
            )
        self.num_envs = int(num_envs)
        self.history_len = int(history_len)
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)
        self._data = torch.zeros(
            self.num_envs,
            self.history_len,
            self.feature_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self._slot_ages = torch.full(
            (self.num_envs, self.history_len),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        self._cursor = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self._age = torch.full(
            (self.num_envs,),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        self._initialized = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )

    def _ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(
                self.num_envs,
                device=self.device,
                dtype=torch.long,
            )
        ids = torch.as_tensor(
            env_ids,
            device=self.device,
            dtype=torch.long,
        )
        if ids.ndim != 1:
            raise ValueError("env_ids must be one-dimensional")
        if ids.numel() and (
            bool((ids < 0).any()) or bool((ids >= self.num_envs).any())
        ):
            raise IndexError("env_ids contains an out-of-range environment")
        if ids.numel() != torch.unique(ids).numel():
            raise ValueError("env_ids must not contain duplicates")
        return ids

    @staticmethod
    def _finite_float32(
        name: str,
        values: torch.Tensor,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(values) or tuple(values.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if values.dtype != torch.float32:
            raise TypeError(f"{name} must use float32")
        result = values.detach().to(device=device)
        if not bool(torch.isfinite(result).all()):
            raise ValueError(f"{name} contains non-finite AMP features")
        return result

    @property
    def ages(self) -> torch.Tensor:
        return self._age.clone()

    @property
    def ready(self) -> torch.Tensor:
        return self._initialized & (self._age >= 1)

    @torch.no_grad()
    def reset_seeded(
        self,
        seed_windows: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        ids = self._ids(env_ids)
        windows = self._finite_float32(
            "seed_windows",
            seed_windows,
            (ids.numel(), self.history_len, self.feature_dim),
            self.device,
        )
        if ids.numel() == 0:
            return
        self._data[ids] = windows
        self._slot_ages[ids] = torch.arange(
            1 - self.history_len,
            1,
            device=self.device,
            dtype=torch.long,
        )
        self._cursor[ids] = 0
        self._age[ids] = 0
        self._initialized[ids] = True

    @torch.no_grad()
    def push(
        self,
        frames: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        ids = self._ids(env_ids)
        values = self._finite_float32(
            "frames",
            frames,
            (ids.numel(), self.feature_dim),
            self.device,
        )
        if ids.numel() == 0:
            return
        if not bool(self._initialized[ids].all()):
            bad = ids[~self._initialized[ids]].detach().cpu().tolist()
            raise RuntimeError(
                f"AMP history must be seeded before push; env_ids={bad}"
            )
        slots = self._cursor[ids]
        next_age = self._age[ids] + 1
        self._data[ids, slots] = values
        self._slot_ages[ids, slots] = next_age
        self._cursor[ids] = torch.remainder(
            slots + 1,
            self.history_len,
        )
        self._age[ids] = next_age

    def _chronological_slots(self, ids: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.history_len, device=self.device)
        return torch.remainder(
            self._cursor[ids, None] + offsets[None, :],
            self.history_len,
        )

    def window_ages(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ids = self._ids(env_ids)
        if not bool(self.ready[ids].all()):
            bad = ids[~self.ready[ids]].detach().cpu().tolist()
            raise RuntimeError(f"AMP history is not ready; env_ids={bad}")
        slots = self._chronological_slots(ids)
        ages = torch.gather(self._slot_ages[ids], 1, slots)
        expected = (
            self._age[ids, None]
            - (self.history_len - 1)
            + torch.arange(self.history_len, device=self.device)[None, :]
        )
        if not torch.equal(ages, expected):
            raise RuntimeError("AMP history ages are not chronological")
        return ages

    def window(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ids = self._ids(env_ids)
        self.window_ages(ids)
        slots = self._chronological_slots(ids)
        return torch.gather(
            self._data[ids],
            1,
            slots[..., None].expand(-1, -1, self.feature_dim),
        )


__all__ = ["TemporalFeatureHistory"]
