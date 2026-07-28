"""Window-native replay with fixed per-stream storage contracts for FCAMP.

The replay unit is one raw, chronological imitation window.  Candidate windows
are offered over an explicit update transaction so a bounded, uniform reservoir
can be selected across the *whole* rollout instead of taking an order-biased
prefix.  Empty slots are filled first.  A partition that was already full at
``begin_update`` replaces exactly its configured quota of uniformly selected
victims at ``commit_update``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


class FCAMPReplayStateError(ValueError):
    """Raised when a checkpoint violates the window replay schema."""


@dataclass
class _Partition:
    capacity: int
    data: torch.Tensor
    end_time: torch.Tensor
    insert_update: torch.Tensor
    size: int = 0
    offered_count: int = 0
    inserted_count: int = 0
    replaced_count: int = 0
    dropped_count: int = 0
    dirty_insert_count: int = 0
    dirty_rejected_count: int = 0


@dataclass
class _StreamStage:
    mode: str
    target: int
    data: torch.Tensor
    end_time: torch.Tensor
    seen: int = 0
    count: int = 0


@dataclass
class _UpdateStage:
    update: int
    generator: torch.Generator | None
    streams: dict[int, _StreamStage]
    poisoned: bool = False


class FCAMPWindowReplay:
    """CPU FP32 replay of complete raw ``[history, feature]`` windows.

    Parameters
    ----------
    stream_capacities:
        Fixed, independent capacities keyed by stream id.  Capacity is measured
        in complete windows, never frames.
    history_len / frame_dim:
        The exact raw window schema.  Flattened or prefix windows are rejected.

    Notes
    -----
    ``dirty`` is a mandatory argument to :meth:`offer`.  A batch containing any
    intervention-contaminated window poisons the transaction and raises before
    it can become live replay data.  Callers must abort that transaction.
    """

    SCHEMA_VERSION = 1
    STATE_KIND = "fcamp_complete_window_replay"

    def __init__(
        self,
        stream_capacities: Mapping[int, int],
        history_len: int,
        frame_dim: int,
    ) -> None:
        if isinstance(history_len, bool) or int(history_len) <= 0:
            raise ValueError("history_len must be a positive integer")
        if isinstance(frame_dim, bool) or int(frame_dim) <= 0:
            raise ValueError("frame_dim must be a positive integer")
        if not isinstance(stream_capacities, Mapping) or not stream_capacities:
            raise ValueError("stream_capacities must be a non-empty mapping")

        capacities: dict[int, int] = {}
        for raw_stream, raw_capacity in stream_capacities.items():
            if isinstance(raw_stream, bool) or not isinstance(raw_stream, int):
                raise TypeError("stream ids must be integers")
            if isinstance(raw_capacity, bool) or not isinstance(raw_capacity, int):
                raise TypeError("stream capacities must be integers")
            if raw_capacity <= 0:
                raise ValueError("every stream capacity must be positive")
            capacities[int(raw_stream)] = int(raw_capacity)
        if len(capacities) != len(stream_capacities):
            raise ValueError("stream ids must be unique after integer normalization")

        self.history_len = int(history_len)
        self.frame_dim = int(frame_dim)
        self.stream_capacities = dict(sorted(capacities.items()))
        self.capacity = int(sum(self.stream_capacities.values()))
        self._partitions: dict[int, _Partition] = {}
        for stream_id, capacity in self.stream_capacities.items():
            self._partitions[stream_id] = _Partition(
                capacity=capacity,
                data=torch.empty(
                    (capacity, self.history_len, self.frame_dim),
                    dtype=torch.float32,
                    device="cpu",
                ),
                end_time=torch.full((capacity,), -1, dtype=torch.long),
                insert_update=torch.full((capacity,), -1, dtype=torch.long),
            )
        self._latest_update = -1
        self._active: _UpdateStage | None = None

    def __len__(self) -> int:
        return sum(partition.size for partition in self._partitions.values())

    def size(self, stream_id: int) -> int:
        return self._partition(stream_id).size

    @torch.no_grad()
    def clear(self) -> None:
        """Clear all committed data and any uncommitted update transaction."""

        self._active = None
        self._latest_update = -1
        for partition in self._partitions.values():
            partition.size = 0
            partition.offered_count = 0
            partition.inserted_count = 0
            partition.replaced_count = 0
            partition.dropped_count = 0
            partition.dirty_insert_count = 0
            partition.dirty_rejected_count = 0
            partition.end_time.fill_(-1)
            partition.insert_update.fill_(-1)

    @torch.no_grad()
    def begin_update(
        self,
        update: int,
        *,
        replacement_quotas: Mapping[int, int],
        generator: torch.Generator | None = None,
    ) -> None:
        """Begin one atomic replay update.

        ``replacement_quotas`` must name every configured stream.  It is used
        only for partitions that are already full at the start of this update;
        non-full partitions uniformly fill as many empty slots as possible.
        """

        if self._active is not None:
            raise RuntimeError("a replay update transaction is already active")
        if isinstance(update, bool) or not isinstance(update, int) or update < 0:
            raise ValueError("update must be a non-negative integer")
        if update <= self._latest_update:
            raise ValueError(
                f"update must advance monotonically beyond {self._latest_update}, got {update}"
            )
        if not isinstance(replacement_quotas, Mapping):
            raise TypeError("replacement_quotas must be a mapping")
        if set(replacement_quotas) != set(self.stream_capacities):
            raise ValueError("replacement_quotas must contain exactly the configured streams")
        if generator is not None and str(generator.device) != "cpu":
            raise ValueError("the replay transaction generator must be a CPU generator")

        stages: dict[int, _StreamStage] = {}
        for stream_id, partition in self._partitions.items():
            raw_quota = replacement_quotas[stream_id]
            if isinstance(raw_quota, bool) or not isinstance(raw_quota, int):
                raise TypeError("replacement quotas must be integers")
            quota = int(raw_quota)
            if quota < 0 or quota > partition.capacity:
                raise ValueError(
                    f"replacement quota for stream {stream_id} must be in "
                    f"[0,{partition.capacity}]"
                )
            if partition.size == partition.capacity:
                mode = "replace"
                target = quota
                stage_data = torch.empty(
                    (target, self.history_len, self.frame_dim),
                    dtype=torch.float32,
                    device="cpu",
                )
                stage_end_time = torch.empty((target,), dtype=torch.long)
            else:
                mode = "fill"
                target = partition.capacity - partition.size
                # Uncommitted fill candidates live only in the currently invalid
                # tail.  This avoids a second multi-gigabyte first-fill buffer.
                stage_data = partition.data[partition.size : partition.capacity]
                stage_end_time = partition.end_time[partition.size : partition.capacity]
            stages[stream_id] = _StreamStage(
                mode=mode,
                target=target,
                data=stage_data,
                end_time=stage_end_time,
            )
        self._active = _UpdateStage(
            update=int(update),
            generator=generator,
            streams=stages,
        )

    @torch.no_grad()
    def offer(
        self,
        windows: torch.Tensor,
        *,
        end_times: torch.Tensor,
        stream_id: int,
        dirty: torch.Tensor,
    ) -> None:
        """Offer one batch to the current update's uniform stream reservoir.

        ``windows`` must be raw chronological ``[B,W,F]`` FP data.  ``end_times``
        is the reference endpoint used to draw a phase-matched expert window.
        Any true value in ``dirty`` rejects the entire batch and poisons the
        transaction, guaranteeing that dirty data is never partially inserted.
        """

        active = self._require_active()
        if active.poisoned:
            raise RuntimeError("the active replay transaction is poisoned; abort it")
        partition = self._partition(stream_id)
        stage = active.streams[int(stream_id)]

        try:
            count = self._validate_offer(windows, end_times=end_times, dirty=dirty)
        except (TypeError, ValueError):
            active.poisoned = True
            raise
        if count == 0:
            return
        dirty_count = int(dirty.sum().item())
        if dirty_count:
            partition.dirty_rejected_count += dirty_count
            active.poisoned = True
            raise ValueError(
                f"stream {stream_id} offered {dirty_count} dirty FCAMP windows; "
                "the whole replay transaction must be aborted"
            )

        cpu_windows = windows.detach().to(device="cpu", dtype=torch.float32)
        cpu_end_times = end_times.detach().to(device="cpu", dtype=torch.long)
        self._reservoir_offer(
            stage,
            cpu_windows,
            cpu_end_times,
            generator=active.generator,
        )

    @torch.no_grad()
    def commit_update(self) -> None:
        """Atomically commit all stream reservoirs."""

        active = self._require_active()
        if active.poisoned:
            raise RuntimeError("cannot commit a poisoned replay transaction; abort it")

        # Validate every stream before mutating any committed partition.
        for stream_id, stage in active.streams.items():
            if stage.mode == "replace" and stage.count != stage.target:
                raise RuntimeError(
                    f"stream {stream_id} offered only {stage.seen} clean windows, "
                    f"but exact replacement quota {stage.target} is required"
                )

        for stream_id, stage in active.streams.items():
            partition = self._partitions[stream_id]
            partition.offered_count += stage.seen
            if stage.mode == "fill":
                start = partition.size
                stop = start + stage.count
                partition.insert_update[start:stop] = active.update
                partition.size = stop
                partition.inserted_count += stage.count
                partition.dropped_count += stage.seen - stage.count
            else:
                replaced = stage.target
                if replaced:
                    victims = torch.randperm(
                        partition.capacity,
                        generator=active.generator,
                        device="cpu",
                    )[:replaced]
                    partition.data.index_copy_(0, victims, stage.data[:replaced])
                    partition.end_time.index_copy_(
                        0, victims, stage.end_time[:replaced]
                    )
                    partition.insert_update[victims] = active.update
                partition.replaced_count += replaced
                partition.dropped_count += stage.seen - replaced

        self._latest_update = active.update
        self._active = None

    @torch.no_grad()
    def abort_update(self) -> None:
        """Discard all candidates staged by the active update."""

        if self._active is None:
            raise RuntimeError("there is no active replay update to abort")
        self._active = None

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        *,
        stream_id: int,
        generator: torch.Generator | None = None,
        replacement: bool = True,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniformly sample complete windows and their reference end times."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(replacement, bool):
            raise TypeError("replacement must be bool")
        if generator is not None and str(generator.device) != "cpu":
            raise ValueError("the replay sampling generator must be a CPU generator")
        partition = self._partition(stream_id)
        if partition.size == 0:
            raise RuntimeError(f"cannot sample empty replay stream {stream_id}")
        if not replacement and batch_size > partition.size:
            raise ValueError("batch_size exceeds stream size without replacement")
        if replacement:
            indices = torch.randint(
                partition.size,
                (batch_size,),
                generator=generator,
                device="cpu",
            )
        else:
            indices = torch.randperm(
                partition.size,
                generator=generator,
                device="cpu",
            )[:batch_size]
        windows = partition.data.index_select(0, indices)
        end_times = partition.end_time.index_select(0, indices)
        if device is not None:
            target = torch.device(device)
            windows = windows.to(
                device=target,
                dtype=torch.float32,
                non_blocking=False,
            )
            end_times = end_times.to(
                device=target,
                dtype=torch.long,
                non_blocking=False,
            )
        return windows, end_times

    def statistics(
        self,
        *,
        current_update: int | None = None,
        current_step: int | None = None,
    ) -> dict[str, float]:
        """Return global and per-stream lifecycle/integrity metrics."""

        if current_update is not None and current_step is not None:
            raise ValueError("specify only one of current_update and current_step")
        now = current_update if current_update is not None else current_step
        if now is None:
            now = self._latest_update
        if isinstance(now, bool) or not isinstance(now, int):
            raise TypeError("current update must be an integer")

        total_size = len(self)
        total_offered = sum(p.offered_count for p in self._partitions.values())
        total_inserted = sum(p.inserted_count for p in self._partitions.values())
        total_replaced = sum(p.replaced_count for p in self._partitions.values())
        total_dropped = sum(p.dropped_count for p in self._partitions.values())
        dirty_inserted = sum(p.dirty_insert_count for p in self._partitions.values())
        dirty_rejected = sum(p.dirty_rejected_count for p in self._partitions.values())
        element_bytes = torch.tensor([], dtype=torch.float32).element_size()
        long_bytes = torch.tensor([], dtype=torch.long).element_size()
        storage_bytes = self.capacity * (
            self.history_len * self.frame_dim * element_bytes + 2 * long_bytes
        )
        metrics = {
            "replay/size": float(total_size),
            "replay/capacity": float(self.capacity),
            "replay/fill_fraction": float(total_size / self.capacity),
            "replay/total_offered": float(total_offered),
            "replay/total_inserted": float(total_inserted),
            "replay/total_replaced": float(total_replaced),
            "replay/replacement_count": float(total_replaced),
            "replay/total_dropped": float(total_dropped),
            "replay/dirty_insert_count": float(dirty_inserted),
            "replay/dirty_rejected_count": float(dirty_rejected),
            "replay/storage_gib": float(storage_bytes / (1024**3)),
            "replay/pinned": 0.0,
            "replay/latest_update": float(self._latest_update),
            "replay/update_active": float(self._active is not None),
        }

        all_ages: list[torch.Tensor] = []
        for stream_id, partition in self._partitions.items():
            prefix = f"replay/stream_{stream_id}"
            metrics.update(
                {
                    f"{prefix}_size": float(partition.size),
                    f"{prefix}_capacity": float(partition.capacity),
                    f"{prefix}_fill_fraction": float(
                        partition.size / partition.capacity
                    ),
                    f"{prefix}_offered": float(partition.offered_count),
                    f"{prefix}_inserted": float(partition.inserted_count),
                    f"{prefix}_replaced": float(partition.replaced_count),
                    f"{prefix}_dropped": float(partition.dropped_count),
                    f"{prefix}_dirty_insert_count": float(
                        partition.dirty_insert_count
                    ),
                    f"{prefix}_dirty_rejected_count": float(
                        partition.dirty_rejected_count
                    ),
                }
            )
            if partition.size:
                insert_updates = partition.insert_update[: partition.size]
                if now < int(insert_updates.max().item()):
                    raise ValueError(
                        f"current update {now} predates stored stream {stream_id} data"
                    )
                ages = int(now) - insert_updates
                all_ages.append(ages)
                metrics.update(self._age_metrics(prefix, ages))
        if all_ages:
            metrics.update(
                self._age_metrics("replay/", torch.cat(all_ages, dim=0))
            )
        return metrics

    def state_dict(self) -> dict:
        """Serialize only committed storage under a strict, versioned schema."""

        if self._active is not None:
            raise RuntimeError("cannot checkpoint an active replay update transaction")
        partitions: dict[int, dict] = {}
        for stream_id, partition in self._partitions.items():
            partitions[stream_id] = {
                "data": partition.data[: partition.size].clone(),
                "end_time": partition.end_time[: partition.size].clone(),
                "insert_update": partition.insert_update[: partition.size].clone(),
                "size": partition.size,
                "offered_count": partition.offered_count,
                "inserted_count": partition.inserted_count,
                "replaced_count": partition.replaced_count,
                "dropped_count": partition.dropped_count,
                "dirty_insert_count": partition.dirty_insert_count,
                "dirty_rejected_count": partition.dirty_rejected_count,
            }
        return {
            "schema_version": self.SCHEMA_VERSION,
            "kind": self.STATE_KIND,
            "history_len": self.history_len,
            "frame_dim": self.frame_dim,
            "storage_dtype": "float32",
            "stream_capacities": dict(self.stream_capacities),
            "latest_update": self._latest_update,
            "partitions": partitions,
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> bool:
        """Load an exact schema match; legacy or malformed state is rejected."""

        top_keys = {
            "schema_version",
            "kind",
            "history_len",
            "frame_dim",
            "storage_dtype",
            "stream_capacities",
            "latest_update",
            "partitions",
        }
        if not isinstance(state, dict) or set(state) != top_keys:
            raise FCAMPReplayStateError("window replay state has incompatible top-level keys")
        if state["schema_version"] != self.SCHEMA_VERSION:
            raise FCAMPReplayStateError("window replay schema version mismatch")
        if state["kind"] != self.STATE_KIND:
            raise FCAMPReplayStateError("window replay state kind mismatch")
        if state["history_len"] != self.history_len or state["frame_dim"] != self.frame_dim:
            raise FCAMPReplayStateError("window replay shape schema mismatch")
        if state["storage_dtype"] != "float32":
            raise FCAMPReplayStateError("window replay storage must be float32")
        if state["stream_capacities"] != self.stream_capacities:
            raise FCAMPReplayStateError("window replay stream capacities mismatch")
        latest_update = state["latest_update"]
        if isinstance(latest_update, bool) or not isinstance(latest_update, int):
            raise FCAMPReplayStateError("latest_update must be an integer")
        if latest_update < -1:
            raise FCAMPReplayStateError("latest_update cannot be below -1")
        raw_partitions = state["partitions"]
        if not isinstance(raw_partitions, dict) or set(raw_partitions) != set(self._partitions):
            raise FCAMPReplayStateError("window replay partition ids mismatch")

        partition_keys = {
            "data",
            "end_time",
            "insert_update",
            "size",
            "offered_count",
            "inserted_count",
            "replaced_count",
            "dropped_count",
            "dirty_insert_count",
            "dirty_rejected_count",
        }
        validated: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]] = {}
        for stream_id, partition in self._partitions.items():
            raw = raw_partitions[stream_id]
            if not isinstance(raw, dict) or set(raw) != partition_keys:
                raise FCAMPReplayStateError(
                    f"window replay stream {stream_id} has incompatible keys"
                )
            integer_fields = (
                "size",
                "offered_count",
                "inserted_count",
                "replaced_count",
                "dropped_count",
                "dirty_insert_count",
                "dirty_rejected_count",
            )
            for name in integer_fields:
                value = raw[name]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise FCAMPReplayStateError(
                        f"window replay stream {stream_id} has invalid {name}"
                    )
            size = raw["size"]
            if size > partition.capacity:
                raise FCAMPReplayStateError("window replay partition exceeds capacity")
            data = raw["data"]
            end_time = raw["end_time"]
            insert_update = raw["insert_update"]
            if (
                not torch.is_tensor(data)
                or data.dtype != torch.float32
                or tuple(data.shape) != (size, self.history_len, self.frame_dim)
                or not bool(torch.isfinite(data).all())
            ):
                raise FCAMPReplayStateError("window replay data tensor is invalid")
            if (
                not torch.is_tensor(end_time)
                or end_time.dtype != torch.long
                or tuple(end_time.shape) != (size,)
                or bool((end_time < 0).any())
            ):
                raise FCAMPReplayStateError("window replay end_time tensor is invalid")
            if (
                not torch.is_tensor(insert_update)
                or insert_update.dtype != torch.long
                or tuple(insert_update.shape) != (size,)
                or bool((insert_update < 0).any())
                or (size > 0 and int(insert_update.max().item()) > latest_update)
            ):
                raise FCAMPReplayStateError("window replay insert_update tensor is invalid")
            if raw["dirty_insert_count"] != 0:
                raise FCAMPReplayStateError("checkpoint contains dirty replay insertions")
            if raw["inserted_count"] != size:
                raise FCAMPReplayStateError("inserted_count must equal committed partition size")
            if raw["offered_count"] != (
                raw["inserted_count"] + raw["replaced_count"] + raw["dropped_count"]
            ):
                raise FCAMPReplayStateError("window replay lifecycle counters do not conserve")
            validated[stream_id] = (data, end_time, insert_update, raw)

        # Validation is complete: mutate only after the entire state is known good.
        self.clear()
        self._latest_update = latest_update
        for stream_id, (data, end_time, insert_update, raw) in validated.items():
            partition = self._partitions[stream_id]
            size = raw["size"]
            partition.data[:size].copy_(data.to(device="cpu"))
            partition.end_time[:size].copy_(end_time.to(device="cpu"))
            partition.insert_update[:size].copy_(insert_update.to(device="cpu"))
            partition.size = size
            partition.offered_count = raw["offered_count"]
            partition.inserted_count = raw["inserted_count"]
            partition.replaced_count = raw["replaced_count"]
            partition.dropped_count = raw["dropped_count"]
            partition.dirty_insert_count = 0
            partition.dirty_rejected_count = raw["dirty_rejected_count"]
        return True

    def _partition(self, stream_id: int) -> _Partition:
        if isinstance(stream_id, bool) or not isinstance(stream_id, int):
            raise TypeError("stream_id must be an integer")
        try:
            return self._partitions[int(stream_id)]
        except KeyError as exc:
            raise KeyError(f"unknown replay stream {stream_id}") from exc

    def _require_active(self) -> _UpdateStage:
        if self._active is None:
            raise RuntimeError("begin_update must be called before offering or committing")
        return self._active

    def _validate_offer(
        self,
        windows: torch.Tensor,
        *,
        end_times: torch.Tensor,
        dirty: torch.Tensor,
    ) -> int:
        if not torch.is_tensor(windows) or windows.ndim != 3 or tuple(windows.shape[1:]) != (
            self.history_len,
            self.frame_dim,
        ):
            raise ValueError(
                f"windows must have raw shape [B,{self.history_len},{self.frame_dim}]"
            )
        if not torch.is_floating_point(windows) or torch.is_complex(windows):
            raise TypeError("FCAMP replay windows must use a real floating-point dtype")
        count = int(windows.shape[0])
        if not torch.is_tensor(end_times) or end_times.ndim != 1 or end_times.shape[0] != count:
            raise ValueError(f"end_times must have shape [{count}]")
        if (
            end_times.dtype == torch.bool
            or torch.is_floating_point(end_times)
            or torch.is_complex(end_times)
        ):
            raise TypeError("end_times must use an integer dtype")
        if not torch.is_tensor(dirty) or dirty.dtype != torch.bool or dirty.ndim != 1:
            raise TypeError("dirty must be a 1-D bool tensor")
        if dirty.shape[0] != count:
            raise ValueError(f"dirty must have shape [{count}]")
        if not bool(torch.isfinite(windows).all()):
            raise ValueError("FCAMP replay windows contain non-finite values")
        if bool((end_times < 0).any()):
            raise ValueError("FCAMP replay end_times must be non-negative")
        return count

    @staticmethod
    def _reservoir_offer(
        stage: _StreamStage,
        windows: torch.Tensor,
        end_times: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> None:
        """Algorithm R, vectorized by retaining the last write per victim slot."""

        count = int(windows.shape[0])
        if count == 0:
            return
        if stage.target == 0:
            stage.seen += count
            return

        source_offset = 0
        direct = min(count, stage.target - stage.count)
        if direct:
            stage.data[stage.count : stage.count + direct].copy_(
                windows[:direct]
            )
            stage.end_time[stage.count : stage.count + direct].copy_(
                end_times[:direct]
            )
            stage.count += direct
            stage.seen += direct
            source_offset = direct

        remaining = count - source_offset
        if remaining:
            # For the item at global zero-based stream position t, Algorithm R
            # chooses j uniformly from [0,t].  It survives iff j < reservoir K.
            positions = torch.arange(
                stage.seen,
                stage.seen + remaining,
                dtype=torch.float64,
                device="cpu",
            )
            draws = torch.rand(
                remaining,
                dtype=torch.float64,
                generator=generator,
                device="cpu",
            )
            slots = torch.floor(draws * (positions + 1.0)).to(dtype=torch.long)
            accepted = slots < stage.target
            if bool(accepted.any()):
                accepted_slots = slots[accepted]
                source_indices = (
                    torch.arange(source_offset, count, dtype=torch.long)[accepted]
                )
                # Several later candidates can address the same reservoir slot.
                # Sequential Algorithm R leaves the final such candidate there.
                final_source = torch.full(
                    (stage.target,), -1, dtype=torch.long, device="cpu"
                )
                final_source.scatter_reduce_(
                    0,
                    accepted_slots,
                    source_indices,
                    reduce="amax",
                    include_self=True,
                )
                victims = torch.where(final_source >= 0)[0]
                sources = final_source.index_select(0, victims)
                stage.data.index_copy_(
                    0, victims, windows.index_select(0, sources)
                )
                stage.end_time.index_copy_(
                    0, victims, end_times.index_select(0, sources)
                )
            stage.seen += remaining

    @staticmethod
    def _age_metrics(prefix: str, ages: torch.Tensor) -> dict[str, float]:
        values = ages.to(dtype=torch.float32)
        separator = "" if prefix.endswith("/") else "_"
        return {
            f"{prefix}{separator}residence_update_age_mean": float(values.mean().item()),
            f"{prefix}{separator}residence_update_age_max": float(values.max().item()),
            f"{prefix}{separator}residence_update_age_p50": float(
                torch.quantile(values, 0.50).item()
            ),
            f"{prefix}{separator}residence_update_age_p95": float(
                torch.quantile(values, 0.95).item()
            ),
        }
