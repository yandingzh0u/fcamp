"""Per-environment causal imitation history with explicit episode age."""

from __future__ import annotations

import torch

from components.imitation.motion_features import canonicalize_imitation_window


class TemporalFeatureHistory:
    """A vectorized per-environment raw-frame ring, returned oldest-to-newest.

    ``reset_seeded`` installs a complete chronological demonstration history.
    The seed itself is never emitted as a policy endpoint; the first
    post-action frame replaces its oldest predecessor and immediately produces
    a fixed-width window.  Incoming frame edges also carry an explicit causal
    flag so windows crossing an exogenous intervention fail closed.
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
        # For the frame stored in slot s, this says whether the edge from its
        # chronological predecessor into s is policy-causal.  The oldest
        # frame's incoming edge lies outside a returned window and is ignored.
        self._incoming_edge_clean = torch.ones(
            (self.num_envs, self.history_len),
            device=self.device,
            dtype=torch.bool,
        )
        # A push is applied after the current frame is captured.  It therefore
        # dirties the incoming edge of the next appended frame.
        self._pending_intervention = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
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
        return self._initialized & (self._age >= 1)

    @property
    def pending_intervention(self) -> torch.Tensor:
        return self._pending_intervention.clone()

    def _ordered_incoming_edge_clean(self, ids: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.history_len, device=self.device)
        chronological = torch.remainder(
            self._cursor[ids, None] + offsets[None, :], self.history_len
        )
        return torch.gather(self._incoming_edge_clean[ids], 1, chronological)

    @property
    def causal_ready(self) -> torch.Tensor:
        """Ready endpoints whose W-1 internal edges contain no intervention."""

        ready = self.ready
        result = torch.zeros_like(ready)
        ids = ready.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return result
        ordered_edges = self._ordered_incoming_edge_clean(ids)
        result[ids] = ordered_edges[:, 1:].all(dim=1)
        return result

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
        if not torch.is_tensor(frames):
            raise TypeError(f"{name} must be a torch.Tensor")
        expected = (count, self.feature_dim)
        if tuple(frames.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(frames.shape)}")
        if frames.dtype != torch.float32:
            raise TypeError(f"{name} must be float32, got {frames.dtype}")
        values = frames.detach().to(device=self.device, dtype=self.dtype)
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"{name} contains non-finite imitation features")
        return values

    def _check_windows(self, name: str, windows: torch.Tensor, count: int) -> torch.Tensor:
        if not torch.is_tensor(windows):
            raise TypeError(f"{name} must be a torch.Tensor")
        expected = (count, self.history_len, self.feature_dim)
        if tuple(windows.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(windows.shape)}")
        if windows.dtype != torch.float32:
            raise TypeError(f"{name} must be float32, got {windows.dtype}")
        values = windows.detach().to(device=self.device, dtype=self.dtype)
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"{name} contains non-finite imitation features")
        return values

    @torch.no_grad()
    def reset_seeded(
        self,
        seed_windows: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Reset from complete demo windows ``[M,W,F]`` oldest-to-newest."""

        ids = self._ids(env_ids)
        windows = self._check_windows("seed_windows", seed_windows, ids.numel())
        if ids.numel() == 0:
            return
        self._data[ids] = windows
        self._slot_ages[ids] = torch.arange(
            1 - self.history_len, 1, device=self.device, dtype=torch.long
        )
        self._incoming_edge_clean[ids] = True
        self._pending_intervention[ids] = False
        # Slot zero is the oldest demo predecessor and is replaced first.
        self._cursor[ids] = 0
        self._initialized[ids] = True
        self._age[ids] = 0

    @torch.no_grad()
    def push(
        self,
        frame: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        *,
        intervention_after: torch.Tensor | None = None,
    ) -> None:
        """Append frames and mark interventions occurring after each frame."""

        ids = self._ids(env_ids)
        frames = self._check_frames("frame", frame, ids.numel())
        if intervention_after is None:
            interventions = torch.zeros(ids.numel(), device=self.device, dtype=torch.bool)
        else:
            if not torch.is_tensor(intervention_after):
                raise TypeError("intervention_after must be a torch.Tensor")
            expected = (ids.numel(),)
            if tuple(intervention_after.shape) != expected:
                raise ValueError(
                    f"intervention_after must have shape {expected}, got {tuple(intervention_after.shape)}"
                )
            if intervention_after.dtype != torch.bool:
                raise TypeError("intervention_after must use bool dtype")
            interventions = intervention_after.detach().to(device=self.device)
        if ids.numel() == 0:
            return
        if not bool(self._initialized[ids].all()):
            bad = ids[~self._initialized[ids]].detach().cpu().tolist()
            raise RuntimeError(f"imitation history must be reset before push; uninitialized={bad}")
        slots = self._cursor[ids]
        next_age = self._age[ids] + 1
        self._data[ids, slots] = frames
        self._slot_ages[ids, slots] = next_age
        self._incoming_edge_clean[ids, slots] = ~self._pending_intervention[ids]
        self._cursor[ids] = torch.remainder(slots + 1, self.history_len)
        self._age[ids] = next_age
        self._pending_intervention[ids] = interventions

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
        if not torch.equal(slot_ages, expected):
            raise RuntimeError("corrupt imitation history: legal window ages must be contiguous")
        return slot_ages

    def window(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return clean chronological windows with shape ``[M,W,F]``."""

        ids = self._ids(env_ids)
        self.window_ages(ids)
        clean = self.causal_ready
        if not bool(clean[ids].all()):
            bad = ids[~clean[ids]].detach().cpu().tolist()
            raise RuntimeError(
                "imitation history window crosses an external intervention: "
                f"env_ids={bad}"
            )
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
        causal_ready = self.causal_ready
        dirty = ready & ~causal_ready
        seed_frames_remaining = torch.where(
            initialized,
            torch.clamp(self.history_len - self._age, min=0, max=self.history_len),
            torch.zeros_like(self._age),
        )
        metrics = {
            "history/initialized_fraction": float(self._initialized.float().mean().item()),
            "history/ready_fraction": float(ready.float().mean().item()),
            "history/ready_count": float(ready.sum().item()),
            "history/causal_ready_fraction": float(causal_ready.float().mean().item()),
            "history/causal_ready_count": float(causal_ready.sum().item()),
            "history/intervention_dirty_fraction": float(dirty.float().mean().item()),
            "history/intervention_dirty_count": float(dirty.sum().item()),
            "history/pending_intervention_count": float(self._pending_intervention.sum().item()),
            "history/age0_count": float((initialized & (self._age == 0)).sum().item()),
            "history/age0_in_legal_window_count": 0.0,
            "history/seeded_fraction": float(initialized.float().mean().item()),
            "history/seeded_count": float(initialized.sum().item()),
            "history/seed_frames_remaining_mean": float(seed_frames_remaining.float().mean().item()),
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
