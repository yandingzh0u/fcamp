from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


class MetricsLogger:
    """Durable metrics that fail fast before persisting a non-finite scalar."""

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
        if nonfinite:
            raise FloatingPointError(
                f"Non-finite or non-scalar metrics at update {update_idx}: "
                f"{nonfinite}"
            )
        tensorboard_step = int(
            clean.get("samples/env_transitions_total", float(update_idx))
        )
        if self._tb is not None:
            for key, scalar in clean.items():
                assert scalar is not None
                self._tb.add_scalar(key, scalar, tensorboard_step)
        record = {
            "update": int(update_idx),
            "nonfinite_count": 0,
            "nonfinite_keys": [],
            "metrics": clean,
        }
        self._handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._handle.flush()
        if self._tb is not None:
            self._tb.add_scalar(
                "system/nonfinite_metric_count",
                0,
                tensorboard_step,
            )
            self._tb.flush()

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
            "validation/survival_seconds_mean",
            "validation/survival_seconds_p50",
            "validation/survival_seconds_p95",
            "validation/reference_progress_mean",
            "validation/reference_progress_p50",
            "validation/reference_progress_p95",
            "validation/return_mean",
            "validation/done_frac",
            "validation/motion_complete_frac",
            "validation/failure_frac",
            "validation/time_out_frac",
            "validation/censored_frac",
            "validation/ee_body_bad_frac",
            "validation/push_applied_frac",
            "validation/died_before_push_frac",
            "validation/first_push_step_mean",
            "validation/reset_root_pos_err",
            "validation/reset_root_ori_deg",
            "validation/reset_anchor_pos_err",
            "validation/reset_anchor_ori_deg",
            "validation/reset_joint_pos_err",
            "validation/reset_joint_vel_err",
            "validation/reset_body_pos_err",
            "validation/reset_body_ori_deg",
            "validation/action_target_ref_now_abs",
            "validation/action_target_ref_next_abs",
            "samples/env_transitions_total",
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
