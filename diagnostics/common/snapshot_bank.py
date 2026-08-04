"""Serializable simulator-snapshot identities and branch provenance.

This module stores opaque state tensors; it intentionally knows nothing about
Isaac Lab.  Environment-specific capture/restore code lives in the collector.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import torch


class SnapshotBankError(ValueError):
    pass


def _clone_tensors(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cloned: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        if not name or not torch.is_tensor(value):
            raise SnapshotBankError("snapshot tensor fields require nonempty names and tensors")
        if not bool(torch.isfinite(value).all()):
            raise SnapshotBankError(f"snapshot tensor {name!r} is non-finite")
        cloned[name] = value.detach().to("cpu").clone()
    return cloned


def _snapshot_rng_seed(bank_seed: int, snapshot_id: str) -> int:
    digest = hashlib.blake2b(
        f"{int(bank_seed)}\0{snapshot_id}".encode("utf-8"),
        digest_size=8,
        person=b"fcamp-snap",
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    phase: float
    rng_seed: int
    state: Mapping[str, torch.Tensor]
    reset_randomization: Mapping[str, torch.Tensor]
    physics_randomization: Mapping[str, torch.Tensor]

    def clone(self) -> "Snapshot":
        return Snapshot(
            snapshot_id=self.snapshot_id,
            phase=self.phase,
            rng_seed=self.rng_seed,
            state=_clone_tensors(self.state),
            reset_randomization=_clone_tensors(self.reset_randomization),
            physics_randomization=_clone_tensors(self.physics_randomization),
        )


@dataclass(frozen=True)
class SnapshotBranch:
    snapshot_id: str
    branch_id: str
    checkpoint_id: str
    rng_seed: int


class SnapshotBank:
    VERSION = 1

    def __init__(self, snapshots: Sequence[Snapshot], *, bank_seed: int) -> None:
        if not snapshots:
            raise SnapshotBankError("snapshot bank must not be empty")
        by_id: dict[str, Snapshot] = {}
        for snapshot in snapshots:
            if not snapshot.snapshot_id:
                raise SnapshotBankError("snapshot_id must not be empty")
            if snapshot.snapshot_id in by_id:
                raise SnapshotBankError(f"duplicate snapshot_id: {snapshot.snapshot_id}")
            if not torch.isfinite(torch.tensor(snapshot.phase)):
                raise SnapshotBankError(f"snapshot {snapshot.snapshot_id} has non-finite phase")
            by_id[snapshot.snapshot_id] = snapshot.clone()
        self._snapshots = by_id
        self.bank_seed = int(bank_seed)

    @classmethod
    def from_batched_tensors(
        cls,
        *,
        snapshot_ids: Sequence[str],
        phase: torch.Tensor,
        state: Mapping[str, torch.Tensor],
        reset_randomization: Mapping[str, torch.Tensor] | None = None,
        physics_randomization: Mapping[str, torch.Tensor] | None = None,
        bank_seed: int,
    ) -> "SnapshotBank":
        count = len(snapshot_ids)
        if phase.ndim != 1 or phase.shape[0] != count:
            raise SnapshotBankError("phase must have shape [num_snapshots]")
        collections = {
            "state": state,
            "reset_randomization": reset_randomization or {},
            "physics_randomization": physics_randomization or {},
        }
        for collection_name, fields in collections.items():
            for name, value in fields.items():
                if not torch.is_tensor(value) or value.ndim < 1 or value.shape[0] != count:
                    raise SnapshotBankError(
                        f"{collection_name}.{name} must have leading dimension {count}"
                    )
        snapshots = []
        for index, raw_id in enumerate(snapshot_ids):
            snapshot_id = str(raw_id)
            snapshots.append(
                Snapshot(
                    snapshot_id=snapshot_id,
                    phase=float(phase[index].item()),
                    rng_seed=_snapshot_rng_seed(bank_seed, snapshot_id),
                    state={name: value[index] for name, value in state.items()},
                    reset_randomization={
                        name: value[index]
                        for name, value in (reset_randomization or {}).items()
                    },
                    physics_randomization={
                        name: value[index]
                        for name, value in (physics_randomization or {}).items()
                    },
                )
            )
        return cls(snapshots, bank_seed=bank_seed)

    def __len__(self) -> int:
        return len(self._snapshots)

    @property
    def snapshot_ids(self) -> tuple[str, ...]:
        return tuple(self._snapshots)

    def get(self, snapshot_id: str) -> Snapshot:
        try:
            return self._snapshots[snapshot_id].clone()
        except KeyError as error:
            raise KeyError(f"unknown snapshot_id: {snapshot_id}") from error

    def branch(
        self,
        snapshot_id: str,
        *,
        branch_id: str,
        checkpoint_id: str,
    ) -> SnapshotBranch:
        if not branch_id or not checkpoint_id:
            raise SnapshotBankError("branch_id and checkpoint_id must not be empty")
        snapshot = self.get(snapshot_id)
        return SnapshotBranch(
            snapshot_id=snapshot.snapshot_id,
            branch_id=branch_id,
            checkpoint_id=checkpoint_id,
            rng_seed=snapshot.rng_seed,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.VERSION,
            "bank_seed": self.bank_seed,
            "snapshots": [
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "phase": snapshot.phase,
                    "rng_seed": snapshot.rng_seed,
                    "state": _clone_tensors(snapshot.state),
                    "reset_randomization": _clone_tensors(snapshot.reset_randomization),
                    "physics_randomization": _clone_tensors(snapshot.physics_randomization),
                }
                for snapshot in self._snapshots.values()
            ],
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "SnapshotBank":
        if payload.get("version") != cls.VERSION:
            raise SnapshotBankError("incompatible snapshot-bank version")
        raw_snapshots = payload.get("snapshots")
        if not isinstance(raw_snapshots, list):
            raise SnapshotBankError("snapshot-bank payload has no snapshot list")
        snapshots: list[Snapshot] = []
        for raw in raw_snapshots:
            if not isinstance(raw, Mapping):
                raise SnapshotBankError("invalid snapshot record")
            snapshots.append(
                Snapshot(
                    snapshot_id=str(raw["snapshot_id"]),
                    phase=float(raw["phase"]),
                    rng_seed=int(raw["rng_seed"]),
                    state=_clone_tensors(raw["state"]),
                    reset_randomization=_clone_tensors(raw["reset_randomization"]),
                    physics_randomization=_clone_tensors(raw["physics_randomization"]),
                )
            )
        bank = cls(snapshots, bank_seed=int(payload["bank_seed"]))
        for snapshot in bank._snapshots.values():
            expected = _snapshot_rng_seed(bank.bank_seed, snapshot.snapshot_id)
            if snapshot.rng_seed != expected:
                raise SnapshotBankError(
                    f"snapshot {snapshot.snapshot_id} has an incompatible RNG seed"
                )
        return bank

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "SnapshotBank":
        return cls.from_state_dict(torch.load(Path(path), map_location="cpu", weights_only=True))
