"""Per-environment causal ring history for AMP state features."""

from __future__ import annotations

import torch

from amp.features import canonicalize_amp_window


class CausalAMPHistory:
    """A vectorized per-environment ring, returned oldest-to-newest.

    Each environment owns its cursor, so arbitrary subsets can reset or advance
    without corrupting another environment's temporal ordering.  Episode reset
    requires a *complete expert history*; padding a single state repeatedly is
    intentionally not supported because it gives the discriminator a reset cue.
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
        # Cursor is the slot overwritten by the next push, hence also the oldest
        # element when the history is initialized.
        self._cursor = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._initialized = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._push_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    @property
    def initialized(self) -> torch.Tensor:
        return self._initialized.clone()

    @property
    def push_count(self) -> torch.Tensor:
        return self._push_count.clone()

    def _ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        ids = env_ids.to(device=self.device, dtype=torch.long)
        if ids.ndim != 1:
            raise ValueError(f"env_ids must be 1-D, got {tuple(ids.shape)}")
        if ids.numel() and (bool((ids < 0).any()) or bool((ids >= self.num_envs).any())):
            raise IndexError("env_ids contains an out-of-range environment")
        return ids

    @torch.no_grad()
    def reset(self, expert_history: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Fill selected rings using chronological expert frames ``[M,W,F]``."""

        ids = self._ids(env_ids)
        expected = (ids.numel(), self.history_len, self.feature_dim)
        if tuple(expert_history.shape) != expected:
            raise ValueError(f"expert_history must have shape {expected}, got {tuple(expert_history.shape)}")
        history = expert_history.to(device=self.device, dtype=self.dtype)
        self._data[ids] = history
        self._cursor[ids] = 0
        self._initialized[ids] = True
        self._push_count[ids] = 0

    @torch.no_grad()
    def push(self, frame: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Append post-action frames for selected environments."""

        ids = self._ids(env_ids)
        expected = (ids.numel(), self.feature_dim)
        if tuple(frame.shape) != expected:
            raise ValueError(f"frame must have shape {expected}, got {tuple(frame.shape)}")
        if ids.numel() == 0:
            return
        if not bool(self._initialized[ids].all()):
            bad = ids[~self._initialized[ids]].detach().cpu().tolist()
            raise RuntimeError(f"AMP history must be reset with expert history before push; uninitialized={bad}")
        self._data[ids, self._cursor[ids]] = frame.to(device=self.device, dtype=self.dtype)
        self._cursor[ids] = torch.remainder(self._cursor[ids] + 1, self.history_len)
        self._push_count[ids] += 1

    def window(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return chronological windows with shape ``[M,W,F]``."""

        ids = self._ids(env_ids)
        if not bool(self._initialized[ids].all()):
            bad = ids[~self._initialized[ids]].detach().cpu().tolist()
            raise RuntimeError(f"requested uninitialized AMP histories: {bad}")
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
            windows = canonicalize_amp_window(windows)
        return windows.reshape(windows.shape[0], self.history_len * self.feature_dim)

    def statistics(self) -> dict[str, float]:
        return {
            "history/initialized_fraction": float(self._initialized.float().mean().item()),
            "history/push_count_mean": float(self._push_count.float().mean().item()),
            "history/push_count_max": float(self._push_count.max().item()),
        }
