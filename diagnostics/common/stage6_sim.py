"""Real Isaac/PhysX collector for diagnostic 62 only.

This module creates no PPO updates and no training labels from reference
errors.  It forks the same snapshot bank across candidate reference segments
and records only measured closed-loop event outcomes.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from .canonical_collection import (
    CanonicalCollectionProtocol,
    assert_startup_randomization_matches,
    close_collection_trainer,
    collector_environment,
    make_collection_trainer,
    read_dense_checkpoint_records,
    restore_snapshot_bank,
    switch_policy_state,
)
from .family_screeners import ExecutabilityDataset, save_executability_dataset
from .manifest import DependencyUnavailable, ProtocolError, read_json, sha256_file
from .noise_bank import CollectorMode, NoiseBank
from .rollout_collector import policy_mean_and_std
from .snapshot_bank import SnapshotBank


def _reference_segment_features(
    env: Any,
    start_phase: torch.Tensor,
    *,
    time_scale: float,
    horizon: int,
) -> torch.Tensor:
    fractions = torch.tensor((0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0), device=env.device)
    offsets = fractions * float(horizon) * float(env.motion_frame_delta) * float(time_scale)
    phases = start_phase[:, None] + offsets[None, :]
    flat = phases.reshape(-1)
    frame = env.motion.get_frame(flat)
    anchor = frame["anchor_pos_w"]
    body_local = frame["body_pos_w"] - anchor[:, None, :]
    per_frame = torch.cat(
        (
            frame["joint_pos"],
            frame["joint_vel"],
            frame["root_lin_vel_w"],
            frame["root_ang_vel_w"],
            body_local.reshape(flat.shape[0], -1),
        ),
        dim=-1,
    )
    return per_frame.reshape(start_phase.shape[0], -1)


def _hash_trace(tensors: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().contiguous().cpu().numpy()
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _motion_delta_owner(env: Any) -> tuple[type, property]:
    for owner in type(env).mro():
        descriptor = owner.__dict__.get("motion_frame_delta")
        if isinstance(descriptor, property):
            return owner, descriptor
    raise ProtocolError("environment has no motion_frame_delta property")


@contextmanager
def _scaled_motion_clock(env: Any, scale: float) -> Iterator[None]:
    if float(scale) <= 0.0 or not np.isfinite(scale):
        raise ProtocolError("candidate time scale must be finite and positive")
    owner, original = _motion_delta_owner(env)
    base = float(env.motion_frame_delta)
    setattr(owner, "motion_frame_delta", property(lambda _self: base * float(scale)))
    try:
        yield
    finally:
        setattr(owner, "motion_frame_delta", original)


def _candidate_start(env: Any, base_phase: torch.Tensor, offset: int, scale: float, horizon: int) -> torch.Tensor:
    maximum = float(env.motion_end_phase) - (
        float(horizon) * float(env.motion_frame_delta) * float(scale)
    ) - 1.0e-4
    if maximum < float(env.motion_start_phase):
        raise DependencyUnavailable("motion is shorter than the frozen executability segment")
    return torch.clamp(
        base_phase.to(dtype=torch.float32) + float(offset),
        min=float(env.motion_start_phase),
        max=maximum,
    )


def _run_candidate(
    trainer: Any,
    bank: SnapshotBank,
    *,
    noise_bank: NoiseBank,
    offset: int,
    time_scale: float,
    horizon: int,
) -> dict[str, Any]:
    env = trainer.env
    restore_snapshot_bank(env, bank, mode=CollectorMode.CONTROLLED_ENVIRONMENT)
    base_phase = env.phase_steps.clone()
    start_phase = _candidate_start(env, base_phase, int(offset), float(time_scale), int(horizon))
    env.phase_steps.copy_(start_phase)
    snapshot_ids = bank.snapshot_ids
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    failure = torch.zeros_like(active)
    tracking_loss = torch.zeros_like(active)
    joint_limit = torch.zeros_like(active)
    undesired_contact = torch.zeros_like(active)
    trace: list[torch.Tensor] = []
    with _scaled_motion_clock(env, float(time_scale)):
        with collector_environment(
            env,
            noise_bank,
            snapshot_ids,
            mode=CollectorMode.CONTROLLED_ENVIRONMENT,
            horizon=int(horizon),
        ) as randomness:
            randomness.begin_observation(0)
            observation = env.get_observation()
            reference_features = _reference_segment_features(
                env, start_phase, time_scale=1.0, horizon=int(horizon)
            )
            normalized_phase = start_phase / max(float(env.motion.num_frames - 1), 1.0)
            feature = torch.cat(
                (
                    observation,
                    reference_features,
                    normalized_phase[:, None],
                    torch.full((env.num_envs, 1), float(time_scale), device=env.device),
                ),
                dim=-1,
            )
            if not bool(torch.isfinite(feature).all()):
                raise ProtocolError("executability classifier feature contains NaN or Inf")
            for step in range(int(horizon)):
                before = active.clone()
                mean, _ = policy_mean_and_std(trainer.algo, observation)
                randomness.begin_observation(step + 1)
                observation, _, done, info = env.step(mean)
                done_terms = info["done_terms"]
                tracking = (
                    done_terms["anchor_pos_bad"]
                    | done_terms["anchor_ori_bad"]
                    | done_terms["ee_body_bad"]
                )
                reward_terms = info["reward_terms"]
                failure |= before & tracking.bool()
                tracking_loss |= before & tracking.bool()
                joint_limit |= before & (reward_terms["joint_limit"] > 0.0)
                undesired_contact |= before & (reward_terms["undesired_contacts"] > 0.0)
                active &= ~done.bool()
                # Environments auto-reset after ``done``.  Hash only the
                # original branch through its first terminal transition; any
                # post-terminal state is a newly sampled episode and is not a
                # same-snapshot counterfactual sample.
                trace.extend(
                    (
                        before,
                        mean[before],
                        env.robot.data.root_link_pose_w[before],
                        env.robot.data.joint_pos[before][:, env.action_joint_ids],
                        done[before].bool(),
                    )
                )
    segment_complete = active
    success = segment_complete & ~failure & ~joint_limit & ~undesired_contact
    return {
        "features": feature.detach().cpu(),
        "success": success.detach().cpu(),
        "segment_complete": segment_complete.detach().cpu(),
        "failure": failure.detach().cpu(),
        "tracking_loss": tracking_loss.detach().cpu(),
        "joint_limit_event": joint_limit.detach().cpu(),
        "undesired_contact_event": undesired_contact.detach().cpu(),
        "start_phase": start_phase.detach().cpu(),
        "trace_sha256": _hash_trace(trace),
    }


def run_closed_loop_executability_real(
    *,
    simulation_app: Any,
    repo_root: Path,
    output_dir: Path,
    spec: Mapping[str, Any],
    result_path: Path,
) -> bool:
    protocol = spec.get("analysis_protocols", {}).get("family_screeners", {}).get(
        "closed_loop_executability"
    )
    if not isinstance(protocol, Mapping):
        raise ProtocolError("closed-loop executability protocol is not frozen")
    if protocol.get("label_source") != "closed_loop_PhysX_outcomes_only":
        raise ProtocolError("executability labels are no longer PhysX outcomes only")
    offsets = tuple(int(value) for value in protocol["candidate_phase_offsets"])
    scales = tuple(float(value) for value in protocol["candidate_time_scales"])
    horizon = int(protocol["segment_horizon_control_steps"])
    if offsets != (-64, -32, -16, 0, 16, 32, 64) or scales != (0.75, 1.0, 1.25) or horizon != 50:
        raise ProtocolError("diag_62 candidate grid changed")
    collection = CanonicalCollectionProtocol.from_spec(spec)
    if int(protocol["snapshot_count"]) != collection.num_snapshots:
        raise ProtocolError("diag_62 snapshot count differs from canonical bank")
    manifest_path = output_dir / "manifest.json"
    snapshot_path = output_dir / "snapshot_bank.pt"
    snapshot_status_path = output_dir / "snapshot_bank.status.json"
    dense_path = output_dir / "checkpoints" / "dense_checkpoint_inventory.csv"
    if not all(path.is_file() for path in (manifest_path, snapshot_path, snapshot_status_path, dense_path)):
        raise DependencyUnavailable("diag_62 real collector lacks manifest/snapshot/final-checkpoint evidence")
    manifest = read_json(manifest_path)
    snapshot_status = read_json(snapshot_status_path)
    if snapshot_status.get("status") != "PASS" or sha256_file(snapshot_path) != snapshot_status.get("evidence", {}).get("snapshot_bank_sha256"):
        raise ProtocolError("snapshot bank is not the validated diag_11 entity")
    bank = SnapshotBank.load(snapshot_path)
    checkpoints = read_dense_checkpoint_records(dense_path)
    update = int(protocol["teacher_checkpoint_update"])
    matches = [record for record in checkpoints if record.update == update]
    if len(matches) != 1:
        raise DependencyUnavailable(f"diag_62 needs exactly one physical u{update} checkpoint")
    checkpoint = matches[0]
    if "A_mix" in str(checkpoint.path) or "FCAMP" in str(checkpoint.path):
        raise ProtocolError("retired A_mix/FCAMP checkpoint entered diag_62")
    trainer = None
    try:
        trainer = make_collection_trainer(
            simulation_app,
            repo_root=repo_root,
            protocol=collection,
            runtime_dir=output_dir / "runtime" / "diag_62",
        )
        assert_startup_randomization_matches(trainer.env, bank)
        switch_policy_state(
            trainer,
            checkpoint,
            expected_platform={
                "dataset_sha256": str(manifest["motion_sha256"]),
                "robot_asset_sha256": str(manifest["robot_asset_sha256"]),
                "action_schema_sha256": str(manifest["action_schema_sha256"]),
            },
        )
        noise_bank = NoiseBank(seed=collection.collector_seed)
        first = _run_candidate(
            trainer, bank, noise_bank=noise_bank,
            offset=offsets[0], time_scale=scales[0], horizon=horizon,
        )
        replay = _run_candidate(
            trainer, bank, noise_bank=NoiseBank(seed=collection.collector_seed),
            offset=offsets[0], time_scale=scales[0], horizon=horizon,
        )
        if first["trace_sha256"] != replay["trace_sha256"]:
            raise ProtocolError("same-snapshot/shared-noise PhysX replay is not exact")
        for name in (
            "features", "success", "segment_complete", "failure", "tracking_loss",
            "joint_limit_event", "undesired_contact_event", "start_phase",
        ):
            if not torch.equal(first[name], replay[name]):
                raise ProtocolError(f"same-snapshot replay differs in {name}")

        results = []
        for offset in offsets:
            for scale in scales:
                if offset == offsets[0] and scale == scales[0]:
                    value = first
                else:
                    value = _run_candidate(
                        trainer, bank,
                        noise_bank=NoiseBank(seed=collection.collector_seed),
                        offset=offset, time_scale=scale, horizon=horizon,
                    )
                results.append((offset, scale, value))
                print(
                    f"[diag_62_collect] offset={offset:+d} scale={scale:.2f} "
                    f"success={float(value['success'].float().mean()):.4f}",
                    flush=True,
                )
        features = []
        labels: dict[str, list[np.ndarray]] = {
            name: [] for name in (
                "success", "segment_complete", "failure", "tracking_loss",
                "joint_limit_event", "undesired_contact_event",
            )
        }
        snapshot_ids: list[str] = []
        segment_ids: list[str] = []
        phases: list[np.ndarray] = []
        requested_offsets: list[np.ndarray] = []
        time_scales: list[np.ndarray] = []
        for offset, scale, value in results:
            segment_id = f"offset_{offset:+d}_scale_{scale:.2f}"
            features.append(value["features"].numpy())
            for name in labels:
                labels[name].append(value[name].numpy().astype(bool))
            snapshot_ids.extend(bank.snapshot_ids)
            segment_ids.extend([segment_id] * len(bank))
            phases.append(value["start_phase"].numpy())
            requested_offsets.append(np.full(len(bank), offset, dtype=np.int64))
            time_scales.append(np.full(len(bank), scale, dtype=np.float64))
        phase = np.concatenate(phases).astype(np.float64)
        phase_normalized = phase / max(float(trainer.env.motion.num_frames - 1), 1.0)
        phase_bins = np.minimum(15, np.floor(phase_normalized * 16).astype(np.int64))
        dataset = ExecutabilityDataset(
            features=np.concatenate(features).astype(np.float32),
            success=np.concatenate(labels["success"]),
            segment_complete=np.concatenate(labels["segment_complete"]),
            failure=np.concatenate(labels["failure"]),
            tracking_loss=np.concatenate(labels["tracking_loss"]),
            joint_limit_event=np.concatenate(labels["joint_limit_event"]),
            undesired_contact_event=np.concatenate(labels["undesired_contact_event"]),
            snapshot_ids=np.asarray(snapshot_ids),
            segment_ids=np.asarray(segment_ids),
            phase_bins=phase_bins,
            phase=phase_normalized,
            requested_phase_offset=np.concatenate(requested_offsets),
            time_scale=np.concatenate(time_scales),
            metadata={
                "schema": "largebox_policy_relative_executability_v1",
                "real_physx_outcomes": True,
                "same_snapshot_replay_verified": True,
                "same_snapshot_replay_trace_sha256": first["trace_sha256"],
                "shared_environment_randomness": True,
                "noise_bank_seed": int(collection.collector_seed),
                "label_source": "closed_loop_PhysX_outcomes_only",
                "reference_rmse_used_as_label": False,
                "raw_human_positive": False,
                "policy_method": "fixed_reward_ppo_teacher",
                "checkpoint_update": checkpoint.update,
                "checkpoint_sha256": checkpoint.sha256,
                "snapshot_bank_sha256": sha256_file(snapshot_path),
                "motion_sha256": str(manifest["motion_sha256"]),
                "robot_asset_sha256": str(manifest["robot_asset_sha256"]),
                "candidate_phase_offsets": list(offsets),
                "candidate_time_scales": list(scales),
                "segment_horizon_control_steps": horizon,
                "input_semantics": "current_actor_observation_plus_raw_candidate_reference_segment_samples_and_time_scale",
                "A_mix": "legacy_quarantined",
                "PPO_updates": 0,
            },
        )
        dataset.validate(expected_snapshots=len(bank), expected_segments=len(offsets) * len(scales))
        save_executability_dataset(result_path, dataset)
    finally:
        if trainer is not None:
            close_collection_trainer(trainer)
    return True


__all__ = ["run_closed_loop_executability_real"]
