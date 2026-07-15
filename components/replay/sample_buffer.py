"""Memory-bounded CPU replay for policy imitation windows."""

from __future__ import annotations

import torch


class SampleReplayBuffer:
    """Raw chronological imitation windows stored in a CPU ring.

    FP16 storage and pinned memory are configurable. Samples are converted to
    the training dtype only while being transferred to the discriminator device.
    """

    def __init__(
        self,
        capacity: int,
        window_dim: int,
        *,
        storage_dtype: torch.dtype = torch.float16,
        pin_memory: bool = True,
    ) -> None:
        if capacity <= 0 or window_dim <= 0:
            raise ValueError("capacity and window_dim must be positive")
        if storage_dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError("frame replay storage must use a floating-point dtype")
        self.capacity = int(capacity)
        self.window_dim = int(window_dim)
        self.storage_dtype = storage_dtype
        # CPU-only torch builds cannot allocate pinned memory.
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self._data = torch.empty(
            (self.capacity, self.window_dim),
            dtype=storage_dtype,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self._insert_step = torch.full((self.capacity,), -1, dtype=torch.int64)
        self._size = 0
        self._cursor = 0
        self._total_inserted = 0
        self._push_step = 0

    def __len__(self) -> int:
        return self._size

    @torch.no_grad()
    def clear(self) -> None:
        """Discard samples without reallocating the (potentially large) ring."""
        self._size = 0
        self._cursor = 0
        self._total_inserted = 0
        self._push_step = 0
        self._insert_step.fill_(-1)

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    @torch.no_grad()
    def push(self, windows: torch.Tensor, *, step: int | None = None) -> None:
        if windows.ndim != 2 or windows.shape[1] != self.window_dim:
            raise ValueError(
                f"windows must have shape [B,{self.window_dim}], got {tuple(windows.shape)}"
            )
        if windows.shape[0] == 0:
            return
        cpu = windows.detach().to(device="cpu", dtype=self.storage_dtype)
        if cpu.shape[0] > self.capacity:
            cpu = cpu[-self.capacity :]
        n = int(cpu.shape[0])
        insert_step = self._push_step if step is None else int(step)
        first = min(n, self.capacity - self._cursor)
        self._data[self._cursor : self._cursor + first].copy_(cpu[:first])
        self._insert_step[self._cursor : self._cursor + first] = insert_step
        rest = n - first
        if rest:
            self._data[:rest].copy_(cpu[first:])
            self._insert_step[:rest] = insert_step
        self._cursor = (self._cursor + n) % self.capacity
        self._size = min(self.capacity, self._size + n)
        self._total_inserted += n
        self._push_step = max(self._push_step + 1, insert_step + 1)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self._size == 0:
            raise RuntimeError("cannot sample an empty frame replay buffer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        idx = torch.randint(self._size, (int(batch_size),), generator=generator, device="cpu")
        batch = self._data.index_select(0, idx)
        if device is None:
            return batch.to(dtype=dtype)
        target = torch.device(device)
        return batch.to(device=target, dtype=dtype, non_blocking=self.pin_memory and target.type == "cuda")

    @torch.no_grad()
    def get_all(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        data = self._data[: self._size]
        if device is None:
            return data.to(dtype=dtype)
        target = torch.device(device)
        return data.to(
            device=target,
            dtype=dtype,
            non_blocking=self.pin_memory and target.type == "cuda",
        )

    def statistics(self, *, current_step: int | None = None) -> dict[str, float]:
        memory_bytes = self.capacity * self.window_dim * torch.tensor([], dtype=self.storage_dtype).element_size()
        metrics = {
            "replay/size": float(self._size),
            "replay/capacity": float(self.capacity),
            "replay/fill_fraction": float(self._size / self.capacity),
            "replay/total_inserted": float(self._total_inserted),
            "replay/replacement_count": float(max(0, self._total_inserted - self.capacity)),
            "replay/storage_gib": float(memory_bytes / (1024**3)),
            "replay/pinned": float(self.pin_memory),
        }
        if self._size:
            now = self._push_step if current_step is None else int(current_step)
            ages = now - self._insert_step[: self._size]
            metrics.update(
                {
                    "replay/age_mean": float(ages.float().mean().item()),
                    "replay/age_max": float(ages.max().item()),
                }
            )
        return metrics

    def state_dict(self) -> dict:
        # Only initialized storage is serialized. This keeps early checkpoints
        # compact while preserving exact adversarial replay on resume.
        return {
            "capacity": self.capacity,
            "window_dim": self.window_dim,
            "storage_dtype": str(self.storage_dtype),
            "data": self._data[: self._size].clone(),
            "insert_step": self._insert_step[: self._size].clone(),
            "size": self._size,
            "cursor": self._cursor,
            "total_inserted": self._total_inserted,
            "push_step": self._push_step,
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict | None) -> bool:
        if not state:
            return False
        if int(state.get("capacity", -1)) != self.capacity or int(state.get("window_dim", -1)) != self.window_dim:
            return False
        size = int(state.get("size", 0))
        data = state.get("data")
        insert_step = state.get("insert_step")
        if size < 0 or size > self.capacity or data is None or tuple(data.shape) != (size, self.window_dim):
            return False
        self.clear()
        self._data[:size].copy_(data.to(dtype=self.storage_dtype, device="cpu"))
        if insert_step is not None and tuple(insert_step.shape) == (size,):
            self._insert_step[:size].copy_(insert_step.to(dtype=torch.int64, device="cpu"))
        self._size = size
        self._cursor = int(state.get("cursor", size % self.capacity)) % self.capacity
        self._total_inserted = int(state.get("total_inserted", size))
        self._push_step = int(state.get("push_step", 0))
        return True
