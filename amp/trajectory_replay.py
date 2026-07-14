"""Frame-level AMP trajectory replay with explicit continuity metadata."""

from __future__ import annotations

import torch


class AMPFrameTrajectoryReplay:
    """CPU replay that stores each post-action AMP frame once.

    Windows are reconstructed only from predecessor chains that keep episode,
    environment, reference time and AMP age contiguous. Reset age-0 frames are
    deliberately not inserted; the first legal discriminator endpoint has age
    ``history_len`` and reconstructs ages ``1..history_len``.
    """

    def __init__(
        self,
        capacity: int,
        frame_dim: int,
        *,
        num_envs: int,
        pin_memory: bool = True,
    ) -> None:
        if capacity <= 0 or frame_dim <= 0 or num_envs <= 0:
            raise ValueError("capacity, frame_dim and num_envs must be positive")
        self.capacity = int(capacity)
        self.frame_dim = int(frame_dim)
        self.num_envs = int(num_envs)
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self._frames = torch.empty(
            (self.capacity, self.frame_dim),
            dtype=torch.float32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self._valid = torch.zeros(self.capacity, dtype=torch.bool)
        self._eligible = torch.zeros(self.capacity, dtype=torch.bool)
        self._env_id = torch.full((self.capacity,), -1, dtype=torch.long)
        self._episode_id = torch.full((self.capacity,), -1, dtype=torch.long)
        self._reference_time = torch.full((self.capacity,), -1, dtype=torch.long)
        self._age = torch.full((self.capacity,), -1, dtype=torch.long)
        self._update = torch.full((self.capacity,), -1, dtype=torch.long)
        self._predecessor = torch.full((self.capacity,), -1, dtype=torch.long)
        self._last_slot_by_env = torch.full((self.num_envs,), -1, dtype=torch.long)
        self._last_episode_by_env = torch.full((self.num_envs,), -1, dtype=torch.long)
        self._last_time_by_env = torch.full((self.num_envs,), -1, dtype=torch.long)
        self._last_age_by_env = torch.full((self.num_envs,), -1, dtype=torch.long)
        self._cursor = 0
        self._size = 0
        self._total_inserted = 0
        self._last_rejected_sample_count = 0

    def __len__(self) -> int:
        return self._size

    @torch.no_grad()
    def clear(self) -> None:
        self._valid.zero_()
        self._eligible.zero_()
        self._env_id.fill_(-1)
        self._episode_id.fill_(-1)
        self._reference_time.fill_(-1)
        self._age.fill_(-1)
        self._update.fill_(-1)
        self._predecessor.fill_(-1)
        self._last_slot_by_env.fill_(-1)
        self._last_episode_by_env.fill_(-1)
        self._last_time_by_env.fill_(-1)
        self._last_age_by_env.fill_(-1)
        self._cursor = 0
        self._size = 0
        self._total_inserted = 0
        self._last_rejected_sample_count = 0

    @torch.no_grad()
    def push_frames(
        self,
        frames: torch.Tensor,
        *,
        env_ids: torch.Tensor,
        episode_ids: torch.Tensor,
        reference_times: torch.Tensor,
        ages: torch.Tensor,
        update: int,
        stream: int = 0,
    ) -> None:
        del stream
        if frames.ndim != 2 or frames.shape[1] != self.frame_dim:
            raise ValueError(
                f"frames must have shape [B,{self.frame_dim}], got {tuple(frames.shape)}"
            )
        count = int(frames.shape[0])
        if count == 0:
            return
        if count > self.capacity:
            frames = frames[-self.capacity :]
            env_ids = env_ids[-self.capacity :]
            episode_ids = episode_ids[-self.capacity :]
            reference_times = reference_times[-self.capacity :]
            ages = ages[-self.capacity :]
            count = self.capacity
        cpu_frames = frames.detach().to(device="cpu", dtype=torch.float32)
        cpu_env = env_ids.detach().to(device="cpu", dtype=torch.long)
        cpu_episode = episode_ids.detach().to(device="cpu", dtype=torch.long)
        cpu_time = reference_times.detach().to(device="cpu", dtype=torch.long)
        cpu_age = ages.detach().to(device="cpu", dtype=torch.long)
        if not bool(torch.isfinite(cpu_frames).all()):
            raise ValueError("AMP replay frames contain non-finite values")
        if bool((cpu_env < 0).any()) or bool((cpu_env >= self.num_envs).any()):
            raise IndexError("env_ids contain an out-of-range environment")
        slots = torch.remainder(
            torch.arange(count, dtype=torch.long) + int(self._cursor),
            self.capacity,
        )
        prev = self._last_slot_by_env.index_select(0, cpu_env)
        prev_valid = (prev >= 0) & self._valid[prev.clamp_min(0)]
        contiguous = (
            prev_valid
            & (self._last_episode_by_env.index_select(0, cpu_env) == cpu_episode)
            & (self._last_time_by_env.index_select(0, cpu_env) + 1 == cpu_time)
            & (self._last_age_by_env.index_select(0, cpu_env) + 1 == cpu_age)
        )
        predecessor = torch.where(contiguous, prev, torch.full_like(prev, -1))

        self._frames.index_copy_(0, slots, cpu_frames)
        self._valid[slots] = True
        self._eligible[slots] = cpu_age > 0
        self._env_id[slots] = cpu_env
        self._episode_id[slots] = cpu_episode
        self._reference_time[slots] = cpu_time
        self._age[slots] = cpu_age
        self._update[slots] = int(update)
        self._predecessor[slots] = predecessor
        self._last_slot_by_env[cpu_env] = slots
        self._last_episode_by_env[cpu_env] = cpu_episode
        self._last_time_by_env[cpu_env] = cpu_time
        self._last_age_by_env[cpu_env] = cpu_age
        self._cursor = (self._cursor + count) % self.capacity
        self._size = min(self.capacity, self._size + count)
        self._total_inserted += count

    def _chain_indices(self, endpoint: int, history_len: int) -> torch.Tensor | None:
        if history_len <= 0:
            return None
        if endpoint < 0 or endpoint >= self.capacity or not bool(self._valid[endpoint].item()):
            return None
        indices: list[int] = []
        slot = int(endpoint)
        end_env = int(self._env_id[endpoint].item())
        end_episode = int(self._episode_id[endpoint].item())
        end_time = int(self._reference_time[endpoint].item())
        end_age = int(self._age[endpoint].item())
        for offset in range(history_len):
            if slot < 0 or not bool(self._valid[slot].item()):
                return None
            if (
                int(self._env_id[slot].item()) != end_env
                or int(self._episode_id[slot].item()) != end_episode
                or int(self._reference_time[slot].item()) != end_time - offset
                or int(self._age[slot].item()) != end_age - offset
                or int(self._age[slot].item()) <= 0
            ):
                return None
            indices.append(slot)
            slot = int(self._predecessor[slot].item())
        indices.reverse()
        return torch.tensor(indices, dtype=torch.long)

    def _chain_indices_batch(
        self,
        endpoints: torch.Tensor,
        history_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if endpoints.ndim != 1:
            raise ValueError("endpoints must be 1-D")
        if endpoints.numel() == 0:
            return torch.empty((0, history_len), dtype=torch.long), torch.empty(0, dtype=torch.bool)
        endpoints = endpoints.to(device="cpu", dtype=torch.long)
        end_env = self._env_id.index_select(0, endpoints)
        end_episode = self._episode_id.index_select(0, endpoints)
        end_time = self._reference_time.index_select(0, endpoints)
        end_age = self._age.index_select(0, endpoints)
        slots = endpoints
        columns: list[torch.Tensor] = []
        valid = torch.ones(endpoints.shape[0], dtype=torch.bool)
        for offset in range(int(history_len)):
            safe = slots.clamp_min(0)
            slot_valid = (
                (slots >= 0)
                & self._valid.index_select(0, safe)
                & (self._env_id.index_select(0, safe) == end_env)
                & (self._episode_id.index_select(0, safe) == end_episode)
                & (self._reference_time.index_select(0, safe) == end_time - offset)
                & (self._age.index_select(0, safe) == end_age - offset)
                & (self._age.index_select(0, safe) > 0)
            )
            valid &= slot_valid
            columns.append(safe)
            slots = self._predecessor.index_select(0, safe)
        chronological = torch.stack(list(reversed(columns)), dim=1)
        return chronological[valid], valid

    @torch.no_grad()
    def sample_windows(
        self,
        batch_size: int,
        history_len: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        candidates = torch.where(self._valid & self._eligible & (self._age >= int(history_len)))[0]
        if candidates.numel() == 0:
            self._last_rejected_sample_count = 0
            return (
                torch.empty((0, history_len, self.frame_dim), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
            )
        chain_batches: list[torch.Tensor] = []
        end_batches: list[torch.Tensor] = []
        rejected = 0
        attempts = 0
        max_attempts = max(int(batch_size) * 8, 64)
        collected = 0
        while collected < int(batch_size) and attempts < max_attempts:
            need = int(batch_size) - collected
            draw_count = min(max(need * 2, need), max_attempts - attempts)
            draw = candidates[
                torch.randint(
                    candidates.numel(),
                    (draw_count,),
                    generator=generator,
                    device="cpu",
                )
            ]
            chains, valid = self._chain_indices_batch(draw, int(history_len))
            rejected += int(draw_count - chains.shape[0])
            if chains.shape[0] > 0:
                take = min(need, int(chains.shape[0]))
                chain_batches.append(chains[:take])
                end_batches.append(self._reference_time.index_select(0, draw[valid][:take]))
                collected += take
            attempts += int(draw_count)
        self._last_rejected_sample_count = rejected
        if not chain_batches:
            return (
                torch.empty((0, history_len, self.frame_dim), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
            )
        chains = torch.cat(chain_batches, dim=0)
        windows = self._frames.index_select(0, chains.reshape(-1)).reshape(
            chains.shape[0],
            int(history_len),
            self.frame_dim,
        )
        return windows, torch.cat(end_batches, dim=0)

    def statistics(self, *, current_step: int | None = None) -> dict[str, float]:
        del current_step
        memory_bytes = self.capacity * (
            self.frame_dim * 4 + 7 * torch.tensor([], dtype=torch.long).element_size() + 2
        )
        valid = self._valid
        eligible = valid & self._eligible
        metrics = {
            "replay/size": float(valid.sum().item()),
            "replay/capacity": float(self.capacity),
            "replay/fill_fraction": float(valid.float().mean().item()),
            "replay/eligible_endpoints": float(eligible.sum().item()),
            "replay/total_inserted": float(self._total_inserted),
            "replay/replacement_count": float(max(0, self._total_inserted - self.capacity)),
            "replay/storage_gib": float(memory_bytes / (1024**3)),
            "replay/pinned": float(self.pin_memory),
            "replay/last_rejected_sample_count": float(self._last_rejected_sample_count),
        }
        if bool(valid.any()):
            ages = self._age[valid]
            metrics.update(
                {
                    "replay/frame_age_mean": float(ages.float().mean().item()),
                    "replay/frame_age_max": float(ages.max().item()),
                    "replay/age0_count": float((ages == 0).sum().item()),
                    "replay/episode_count": float(torch.unique(self._episode_id[valid]).numel()),
                }
            )
        return metrics

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "frame_dim": self.frame_dim,
            "num_envs": self.num_envs,
            "frames": self._frames.clone(),
            "valid": self._valid.clone(),
            "eligible": self._eligible.clone(),
            "env_id": self._env_id.clone(),
            "episode_id": self._episode_id.clone(),
            "reference_time": self._reference_time.clone(),
            "age": self._age.clone(),
            "update": self._update.clone(),
            "predecessor": self._predecessor.clone(),
            "last_slot_by_env": self._last_slot_by_env.clone(),
            "last_episode_by_env": self._last_episode_by_env.clone(),
            "last_time_by_env": self._last_time_by_env.clone(),
            "last_age_by_env": self._last_age_by_env.clone(),
            "cursor": self._cursor,
            "size": self._size,
            "total_inserted": self._total_inserted,
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict | None) -> bool:
        if not state:
            return False
        if (
            int(state.get("capacity", -1)) != self.capacity
            or int(state.get("frame_dim", -1)) != self.frame_dim
            or int(state.get("num_envs", -1)) != self.num_envs
        ):
            return False
        try:
            self._frames.copy_(state["frames"].to(dtype=torch.float32, device="cpu"))
            self._valid.copy_(state["valid"].to(dtype=torch.bool, device="cpu"))
            self._eligible.copy_(state["eligible"].to(dtype=torch.bool, device="cpu"))
            self._env_id.copy_(state["env_id"].to(dtype=torch.long, device="cpu"))
            self._episode_id.copy_(state["episode_id"].to(dtype=torch.long, device="cpu"))
            self._reference_time.copy_(state["reference_time"].to(dtype=torch.long, device="cpu"))
            self._age.copy_(state["age"].to(dtype=torch.long, device="cpu"))
            self._update.copy_(state["update"].to(dtype=torch.long, device="cpu"))
            self._predecessor.copy_(state["predecessor"].to(dtype=torch.long, device="cpu"))
            self._last_slot_by_env.copy_(state["last_slot_by_env"].to(dtype=torch.long, device="cpu"))
            self._last_episode_by_env.copy_(state["last_episode_by_env"].to(dtype=torch.long, device="cpu"))
            self._last_time_by_env.copy_(state["last_time_by_env"].to(dtype=torch.long, device="cpu"))
            self._last_age_by_env.copy_(state["last_age_by_env"].to(dtype=torch.long, device="cpu"))
        except (KeyError, RuntimeError):
            return False
        self._cursor = int(state.get("cursor", 0)) % self.capacity
        self._size = int(state.get("size", int(self._valid.sum().item())))
        self._total_inserted = int(state.get("total_inserted", self._size))
        self._last_rejected_sample_count = 0
        return True
