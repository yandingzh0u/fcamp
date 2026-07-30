"""Single global circular replay buffer used by standard AMP."""

from __future__ import annotations

import torch


class AMPReplayStateError(ValueError):
    """Raised when a checkpoint violates the AMP replay contract."""


class AMPWindowReplay:
    """CPU replay of complete chronological discriminator windows.

    MimicKit keeps one unconditional policy replay ring.  Before the ring is
    full, every current policy window is inserted.  Once full, callers insert
    only a random ``replay_samples`` subset from the newest rollout.
    """

    SCHEMA_VERSION = 2
    STATE_KIND = "amp_global_circular_window_replay"

    def __init__(self, capacity: int, history_len: int, frame_dim: int) -> None:
        if capacity <= 0 or history_len <= 0 or frame_dim <= 0:
            raise ValueError(
                "capacity, history_len and frame_dim must be positive"
            )
        self.capacity = int(capacity)
        self.history_len = int(history_len)
        self.frame_dim = int(frame_dim)
        self.data = torch.empty(
            (self.capacity, self.history_len, self.frame_dim),
            dtype=torch.float32,
            device="cpu",
        )
        # Endpoints are diagnostic metadata only; they never affect sampling.
        self.end_time = torch.full(
            (self.capacity,), -1.0, dtype=torch.float32, device="cpu"
        )
        self.head = 0
        self.size = 0
        self.total_inserted = 0
        self.total_offered = 0
        # MimicKit samples replay through a shuffled capacity-sized index
        # stream. When the replay is not full, indices are reduced modulo the
        # current sample count; sampling is not independent randint.
        self._sample_order = torch.empty(
            0,
            dtype=torch.long,
            device="cpu",
        )
        self._sample_cursor = 0

    def __len__(self) -> int:
        return self.size

    @property
    def is_full(self) -> bool:
        return self.size == self.capacity

    @torch.no_grad()
    def clear(self) -> None:
        self.head = 0
        self.size = 0
        self.total_inserted = 0
        self.total_offered = 0
        self.end_time.fill_(-1)
        self._sample_order = torch.empty(
            0,
            dtype=torch.long,
            device="cpu",
        )
        self._sample_cursor = 0

    def _reset_sample_order(
        self,
        generator: torch.Generator | None,
    ) -> None:
        self._sample_order = torch.randperm(
            self.capacity,
            generator=generator,
            device="cpu",
        )
        self._sample_cursor = 0

    def _sample_indices(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if batch_size > self.capacity:
            raise ValueError(
                "AMP replay batch_size cannot exceed replay capacity"
            )
        if self._sample_order.numel() != self.capacity:
            self._reset_sample_order(generator)
        stop = self._sample_cursor + int(batch_size)
        if stop <= self.capacity:
            indices = self._sample_order[self._sample_cursor : stop]
            self._sample_cursor = stop
        else:
            tail = self._sample_order[self._sample_cursor :]
            remainder = int(batch_size) - int(tail.numel())
            self._reset_sample_order(generator)
            indices = torch.cat(
                (tail, self._sample_order[:remainder]),
                dim=0,
            )
            self._sample_cursor = remainder
        return torch.remainder(indices, self.size)

    def _validate(
        self,
        windows: torch.Tensor,
        end_times: torch.Tensor,
    ) -> int:
        if not torch.is_tensor(windows) or windows.ndim != 3:
            raise TypeError("windows must be a [B,W,F] tensor")
        expected = (self.history_len, self.frame_dim)
        if tuple(windows.shape[1:]) != expected:
            raise ValueError(
                f"windows must have trailing shape {expected}, "
                f"got {tuple(windows.shape[1:])}"
            )
        if windows.dtype != torch.float32:
            raise TypeError("windows must be float32")
        count = int(windows.shape[0])
        if tuple(end_times.shape) != (count,):
            raise ValueError("end_times must match the window batch")
        if windows.numel() and not bool(torch.isfinite(windows).all()):
            raise ValueError("windows contain non-finite values")
        if end_times.numel() and not bool(torch.isfinite(end_times.float()).all()):
            raise ValueError("end_times contain non-finite values")
        return count

    @torch.no_grad()
    def push(
        self,
        windows: torch.Tensor,
        *,
        end_times: torch.Tensor,
    ) -> None:
        """Append windows in circular FIFO order."""

        count = self._validate(windows, end_times)
        self.total_offered += count
        if count == 0:
            return
        cpu_windows = windows.detach().to(device="cpu", dtype=torch.float32)
        cpu_ends = end_times.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        if count > self.capacity:
            cpu_windows = cpu_windows[-self.capacity :]
            cpu_ends = cpu_ends[-self.capacity :]
            count = self.capacity

        first = min(count, self.capacity - self.head)
        self.data[self.head : self.head + first].copy_(cpu_windows[:first])
        self.end_time[self.head : self.head + first].copy_(cpu_ends[:first])
        remainder = count - first
        if remainder:
            self.data[:remainder].copy_(cpu_windows[first:])
            self.end_time[:remainder].copy_(cpu_ends[first:])
        self.head = (self.head + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        self.total_inserted += count

    @torch.no_grad()
    def update_from_rollout(
        self,
        windows: torch.Tensor,
        *,
        end_times: torch.Tensor,
        replay_samples: int,
        generator: torch.Generator | None = None,
    ) -> int:
        """Apply MimicKit's fill-all/replace-random-subset replay policy."""

        count = self._validate(windows, end_times)
        if replay_samples <= 0:
            raise ValueError("replay_samples must be positive")
        insert_count = count if not self.is_full else min(count, replay_samples)
        if insert_count == 0:
            return 0
        if generator is not None and str(generator.device) != "cpu":
            raise ValueError("AMP replay generator must be a CPU generator")
        indices = torch.randperm(
            count, generator=generator, device="cpu"
        )[:insert_count]
        source_windows = windows.detach().to(device="cpu", dtype=torch.float32)
        source_ends = end_times.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        self.push(
            source_windows.index_select(0, indices),
            end_times=source_ends.index_select(0, indices),
        )
        return insert_count

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size == 0:
            raise RuntimeError("cannot sample an empty AMP replay")
        if generator is not None and str(generator.device) != "cpu":
            raise ValueError("AMP replay generator must be a CPU generator")
        indices = self._sample_indices(
            int(batch_size),
            generator=generator,
        )
        windows = self.data.index_select(0, indices)
        ends = self.end_time.index_select(0, indices)
        if device is not None:
            windows = windows.to(device=device, dtype=torch.float32)
            ends = ends.to(device=device, dtype=torch.float32)
        return windows, ends

    def statistics(self) -> dict[str, float]:
        return {
            "replay/size": float(self.size),
            "replay/capacity": float(self.capacity),
            "replay/fill_fraction": float(self.size / self.capacity),
            "replay/head": float(self.head),
            "replay/total_offered": float(self.total_offered),
            "replay/total_inserted": float(self.total_inserted),
            "replay/stream_count": 1.0,
            "replay/global_unconditional_contract": 1.0,
            "replay/permutation_sampling_contract": 1.0,
            "replay/sample_cursor": float(self._sample_cursor),
        }

    def state_dict(self) -> dict:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "state_kind": self.STATE_KIND,
            "capacity": self.capacity,
            "history_len": self.history_len,
            "frame_dim": self.frame_dim,
            "head": self.head,
            "size": self.size,
            "total_inserted": self.total_inserted,
            "total_offered": self.total_offered,
            "data": self.data[: self.size].clone(),
            "end_time": self.end_time[: self.size].clone(),
            "sample_order": self._sample_order.clone(),
            "sample_cursor": self._sample_cursor,
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        expected = {
            "schema_version": self.SCHEMA_VERSION,
            "state_kind": self.STATE_KIND,
            "capacity": self.capacity,
            "history_len": self.history_len,
            "frame_dim": self.frame_dim,
        }
        if not isinstance(state, dict):
            raise AMPReplayStateError("AMP replay state must be a dictionary")
        for key, value in expected.items():
            if state.get(key) != value:
                raise AMPReplayStateError(
                    f"AMP replay {key}={state.get(key)!r}, expected {value!r}"
                )
        size = int(state.get("size", -1))
        head = int(state.get("head", -1))
        data = state.get("data")
        ends = state.get("end_time")
        sample_order = state.get("sample_order")
        sample_cursor = int(state.get("sample_cursor", -1))
        if not 0 <= size <= self.capacity or not 0 <= head < self.capacity:
            raise AMPReplayStateError("AMP replay size/head is invalid")
        if (
            not torch.is_tensor(data)
            or tuple(data.shape)
            != (size, self.history_len, self.frame_dim)
            or data.dtype != torch.float32
            or not bool(torch.isfinite(data).all())
        ):
            raise AMPReplayStateError("AMP replay data tensor is invalid")
        if (
            not torch.is_tensor(ends)
            or tuple(ends.shape) != (size,)
            or ends.dtype != torch.float32
            or not bool(torch.isfinite(ends).all())
        ):
            raise AMPReplayStateError("AMP replay endpoint tensor is invalid")
        if (
            not torch.is_tensor(sample_order)
            or sample_order.dtype != torch.long
            or tuple(sample_order.shape) not in ((0,), (self.capacity,))
        ):
            raise AMPReplayStateError(
                "AMP replay sample-order tensor is invalid"
            )
        if sample_order.numel() == self.capacity:
            if not torch.equal(
                torch.sort(sample_order.cpu()).values,
                torch.arange(self.capacity, dtype=torch.long),
            ):
                raise AMPReplayStateError(
                    "AMP replay sample-order is not a permutation"
                )
            if not 0 <= sample_cursor <= self.capacity:
                raise AMPReplayStateError(
                    "AMP replay sample cursor is invalid"
                )
        elif sample_cursor != 0:
            raise AMPReplayStateError(
                "empty AMP replay sample order requires cursor zero"
            )
        self.clear()
        self.data[:size].copy_(data.to(device="cpu"))
        self.end_time[:size].copy_(ends.to(device="cpu"))
        self.size = size
        self.head = head
        self.total_inserted = int(state.get("total_inserted", size))
        self.total_offered = int(state.get("total_offered", size))
        self._sample_order = sample_order.detach().to(
            device="cpu",
            dtype=torch.long,
        ).clone()
        self._sample_cursor = sample_cursor
