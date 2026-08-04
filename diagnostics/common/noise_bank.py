"""Order-independent common-random-number banks for rollout comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Iterable, Sequence

import torch


class NoiseProtocolError(ValueError):
    pass


class CollectorMode(str, Enum):
    CLEAN_MEAN = "clean_mean"
    CONTROLLED_ENVIRONMENT = "controlled_environment"
    COMMON_ACTION_NOISE = "common_action_noise"
    NATIVE_STOCHASTIC = "native_stochastic"


OVERLAP_ELIGIBLE_MODES = frozenset(
    {
        CollectorMode.CLEAN_MEAN,
        CollectorMode.CONTROLLED_ENVIRONMENT,
        CollectorMode.COMMON_ACTION_NOISE,
    }
)


def parse_collector_mode(mode: CollectorMode | str) -> CollectorMode:
    try:
        return mode if isinstance(mode, CollectorMode) else CollectorMode(mode)
    except ValueError as error:
        raise NoiseProtocolError(f"unknown collector mode: {mode!r}") from error


def is_overlap_eligible(mode: CollectorMode | str) -> bool:
    return parse_collector_mode(mode) in OVERLAP_ELIGIBLE_MODES


def require_overlap_eligible(modes: Iterable[CollectorMode | str]) -> None:
    parsed = tuple(parse_collector_mode(mode) for mode in modes)
    invalid = sorted({mode.value for mode in parsed if mode not in OVERLAP_ELIGIBLE_MODES})
    if invalid:
        raise NoiseProtocolError(
            "native/checkpoint-specific stochasticity is excluded from effective-overlap "
            f"analysis; invalid modes={invalid}"
        )


def overlap_eligible_indices(modes: Sequence[CollectorMode | str]) -> tuple[int, ...]:
    return tuple(index for index, mode in enumerate(modes) if is_overlap_eligible(mode))


def _stable_seed(bank_seed: int, snapshot_id: str, stream: str) -> int:
    digest = hashlib.blake2b(
        f"{int(bank_seed)}\0{snapshot_id}\0{stream}".encode("utf-8"),
        digest_size=8,
        person=b"fcamp-rng",
    ).digest()
    # Torch accepts signed 64-bit seeds.  Masking also keeps serialization
    # stable across Python versions and platforms.
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


@dataclass(frozen=True)
class NoiseBank:
    seed: int

    def _sample_per_snapshot(
        self,
        snapshot_ids: Sequence[str],
        shape_tail: tuple[int, ...],
        *,
        stream: str,
        distribution: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not snapshot_ids:
            raise NoiseProtocolError("snapshot_ids must not be empty")
        if not stream:
            raise NoiseProtocolError("noise stream must not be empty")
        if not shape_tail or any(not isinstance(size, int) or size <= 0 for size in shape_tail):
            raise NoiseProtocolError("noise shape must contain positive dimensions")
        rows: list[torch.Tensor] = []
        for raw_snapshot_id in snapshot_ids:
            snapshot_id = str(raw_snapshot_id)
            if not snapshot_id:
                raise NoiseProtocolError("snapshot_id must not be empty")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_stable_seed(self.seed, snapshot_id, stream))
            if distribution == "normal":
                row = torch.randn(shape_tail, generator=generator, dtype=dtype)
            elif distribution == "uniform":
                row = torch.rand(shape_tail, generator=generator, dtype=dtype)
            else:
                raise NoiseProtocolError(f"unknown distribution: {distribution}")
            rows.append(row)
        return torch.stack(rows, dim=0)

    def common_action_epsilon(
        self,
        snapshot_ids: Sequence[str],
        *,
        horizon: int,
        action_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        epsilon = self._sample_per_snapshot(
            snapshot_ids,
            (horizon, action_dim),
            stream="common_action_epsilon",
            distribution="normal",
            dtype=dtype,
        )
        return epsilon.to(device=device)

    def controlled_uniform(
        self,
        snapshot_ids: Sequence[str],
        *,
        horizon: int,
        width: int,
        stream: str,
        low: float = 0.0,
        high: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        if not float(high) > float(low):
            raise NoiseProtocolError("uniform high must be greater than low")
        unit = self._sample_per_snapshot(
            snapshot_ids,
            (horizon, width),
            stream=f"controlled_environment:{stream}",
            distribution="uniform",
            dtype=dtype,
        )
        return (float(low) + (float(high) - float(low)) * unit).to(device=device)

    def state_dict(self) -> dict[str, object]:
        return {"version": 1, "seed": int(self.seed), "algorithm": "blake2b+torch_generator"}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "NoiseBank":
        if state.get("version") != 1 or "seed" not in state:
            raise NoiseProtocolError("incompatible noise-bank state")
        return cls(seed=int(state["seed"]))
