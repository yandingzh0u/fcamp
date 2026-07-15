from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


class MetricsLogger:
    """Durable structured metrics with an optional TensorBoard mirror.

    Console logs are useful while watching a run, but JSONL is the source of
    truth for debugging and ablations.  Non-finite metrics are encoded as null
    and counted explicitly, so one NaN cannot silently corrupt downstream
    plotting scripts.
    """

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "metrics.jsonl"
        self.validation_path = self.log_dir / "validation_summary.csv"
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        self._tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        except Exception as exc:
            print(f"[METRICS] TensorBoard disabled: {exc}", flush=True)
        print(f"[METRICS] jsonl={self.path} tensorboard={self._tb is not None}", flush=True)

    def write(self, update_idx: int, metrics: dict[str, Any]) -> None:
        clean: dict[str, float | None] = {}
        nonfinite: list[str] = []
        for key, value in sorted(metrics.items()):
            scalar = _finite_float(value)
            clean[key] = scalar
            if scalar is None:
                nonfinite.append(key)
            elif self._tb is not None:
                self._tb.add_scalar(key, scalar, update_idx)
        record = {
            "update": int(update_idx),
            "nonfinite_count": len(nonfinite),
            "nonfinite_keys": nonfinite,
            "metrics": clean,
        }
        self._handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._handle.flush()
        if self._tb is not None:
            self._tb.add_scalar("system/nonfinite_metric_count", len(nonfinite), update_idx)
            self._tb.flush()
        if nonfinite:
            print(f"[METRICS_WARN] update={update_idx} nonfinite={nonfinite}", flush=True)

    def write_validation_summary(self, update_idx: int, metrics: dict[str, Any]) -> None:
        if "validation/steps_mean" not in metrics:
            return
        columns = [
            "update",
            "validation/fixed_seed",
            "validation/protocol_version",
            "validation/steps_mean",
            "validation/steps_min",
            "validation/steps_p50",
            "validation/steps_p95",
            "validation/steps_max",
            "validation/return_mean",
            "validation/done_frac",
            "validation/motion_complete_frac",
            "validation/ee_body_bad_frac",
            "validation/push_applied_frac",
            "validation/died_before_push_frac",
            "validation/first_push_step_mean",
        ]
        write_header = not self.validation_path.exists() or self.validation_path.stat().st_size == 0
        with self.validation_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if write_header:
                writer.writeheader()
            row = {"update": int(update_idx)}
            for key in columns[1:]:
                value = _finite_float(metrics.get(key))
                row[key] = "" if value is None else value
            writer.writerow(row)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
        self._handle.close()
