#!/usr/bin/env python3
"""Collect the frozen Stage-4 same-snapshot branches in the real simulator."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.manifest import (
    PASS,
    DependencyUnavailable,
    ProtocolError,
    canonical_sha256,
    load_spec,
    read_json,
    sha256_file,
    write_json_exclusive,
)
from diagnostics.common.isaac_exit import finish_isaac_entrypoint

_ALLOW_HARD_EXIT = __name__ == "__main__"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--bank-path", type=Path, required=True)
    parser.add_argument("--row-path", type=Path, required=True)
    return parser


class _FrozenPolicy:
    def __init__(self, actor: Any, normalizer: Any) -> None:
        self.actor = copy.deepcopy(actor).eval()
        self.normalizer = copy.deepcopy(normalizer).eval()

    @torch.no_grad()
    def evaluate(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.actor.act_inference(self.normalizer(observation, update=False))
        std = self.actor.std.unsqueeze(0).expand_as(mean)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
            raise ProtocolError("frozen branch policy produced NaN or Inf")
        return mean, std


def _apply_bounds(value: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.minimum(value, high), low)


class _Stage4AdapterBase:
    def __init__(self, *, policy_domain: str, manifest: Mapping[str, Any]) -> None:
        self.policy_domain = policy_domain
        self._manifest = manifest

    def reset_branch(self, *, env: Any, mode: Any, snapshot_ids: Sequence[str]) -> None:
        del env, mode, snapshot_ids

    def provenance(self) -> Mapping[str, Any]:
        return {
            "policy_method": self.policy_domain,
            "policy_source_commit": str(self._manifest["git_commit"]),
            "policy_source_snapshot_sha256": str(self._manifest["source_snapshot_sha256"]),
            "policy_resolved_config_sha256": str(self._manifest["resolved_config_sha256"]),
        }


class _InterpolationAdapter(_Stage4AdapterBase):
    def __init__(self, left: _FrozenPolicy, right: _FrozenPolicy, alpha: float, manifest: Mapping[str, Any]):
        super().__init__(policy_domain="stage4_action_interpolation", manifest=manifest)
        self.left = left
        self.right = right
        self.alpha = float(alpha)

    @torch.no_grad()
    def action_record(
        self,
        *,
        env: Any,
        observation: torch.Tensor,
        imitation: Mapping[str, torch.Tensor],
        common_epsilon: torch.Tensor,
        mode: Any,
        common_sigma: float,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        step: int,
    ) -> Any:
        del env, imitation, mode, common_sigma, step
        from diagnostics.common.rollout_collector import ActionRecord

        left_mean, left_std = self.left.evaluate(observation)
        right_mean, right_std = self.right.evaluate(observation)
        mean = (1.0 - self.alpha) * left_mean + self.alpha * right_mean
        std = (1.0 - self.alpha) * left_std + self.alpha * right_std
        applied = _apply_bounds(mean, action_low, action_high)
        return ActionRecord(
            mean=mean.detach().clone(),
            std=std.detach().clone(),
            common_epsilon=common_epsilon.detach().clone(),
            sampled=mean.detach().clone(),
            applied=applied.detach().clone(),
        )


class _PerturbationAdapter(_Stage4AdapterBase):
    def __init__(self, base: _FrozenPolicy, scale: float, action_iqr: torch.Tensor, manifest: Mapping[str, Any]):
        super().__init__(policy_domain="stage4_action_perturbation", manifest=manifest)
        self.base = base
        self.scale = float(scale)
        self.action_iqr = action_iqr

    @torch.no_grad()
    def action_record(
        self,
        *,
        env: Any,
        observation: torch.Tensor,
        imitation: Mapping[str, torch.Tensor],
        common_epsilon: torch.Tensor,
        mode: Any,
        common_sigma: float,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        step: int,
    ) -> Any:
        del env, imitation, mode, common_sigma, step
        from diagnostics.common.rollout_collector import ActionRecord

        mean, std = self.base.evaluate(observation)
        delta = self.scale * self.action_iqr.to(mean) * common_epsilon
        sampled = mean + delta
        applied = _apply_bounds(sampled, action_low, action_high)
        return ActionRecord(
            mean=mean.detach().clone(),
            std=std.detach().clone(),
            common_epsilon=common_epsilon.detach().clone(),
            sampled=sampled.detach().clone(),
            applied=applied.detach().clone(),
        )


def _status_pass(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise DependencyUnavailable(f"{label} status is absent: {path}")
    payload = read_json(path)
    if payload.get("status") != PASS:
        raise DependencyUnavailable(f"{label} is not PASS")
    return payload


def _load_whole_shard(
    index_path: Path,
    *,
    predicates: Mapping[str, Any],
    snapshot_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from diagnostics.common.canonical_collection import (
        load_rollout_index,
        load_rollout_shard,
        resolve_shard_path,
        validate_canonical_rollout_tree,
    )

    rows = [
        row
        for row in load_rollout_index(index_path)
        if all(row.get(key) == value for key, value in predicates.items())
    ]
    if len(rows) != len(snapshot_ids):
        raise DependencyUnavailable(
            f"canonical branch {dict(predicates)} has {len(rows)} rows, expected {len(snapshot_ids)}"
        )
    rows.sort(key=lambda row: int(row["shard_env_index"]))
    if tuple(str(row["snapshot_id"]) for row in rows) != tuple(snapshot_ids):
        raise ProtocolError("canonical branch snapshot order differs from SnapshotBank")
    paths = {resolve_shard_path(index_path, row) for row in rows}
    if len(paths) != 1:
        raise ProtocolError("one canonical branch spans multiple shards")
    tree = load_rollout_shard(next(iter(paths)))
    validate_canonical_rollout_tree(tree)
    return tree, rows[0]


def _action_iqr(index_path: Path, split_path: Path, *, update: int) -> torch.Tensor:
    from diagnostics.common.canonical_collection import load_rollout_index, load_rollout_trajectory

    split = _status_pass(split_path, "diag_14")["evidence"]["inner_snapshot_split"]["sample_to_split"]
    rows = [
        row
        for row in load_rollout_index(index_path)
        if int(row["checkpoint_update"]) == int(update)
        and str(row["collector_mode"]) == "clean_mean"
        and float(row["common_sigma"]) == 0.0
        and split.get(str(row["sample_id"])) == "train"
    ]
    if not rows:
        raise DependencyUnavailable("u500 clean train trajectories are absent for action-IQR scaling")
    cache: dict[str, dict[str, Any]] = {}
    values = [load_rollout_trajectory(index_path, row, cache=cache)["action"]["mean"] for row in rows]
    bank = torch.cat(values, dim=0).float()
    iqr = torch.quantile(bank, 0.75, dim=0) - torch.quantile(bank, 0.25, dim=0)
    if not bool(torch.isfinite(iqr).all()) or bool((iqr < 0.0).any()) or not bool((iqr > 0.0).any()):
        raise ProtocolError("train-split clean teacher action IQR is invalid")
    return iqr


def main() -> int:
    try:
        from isaaclab.app import AppLauncher
    except (ImportError, ModuleNotFoundError) as exc:
        raise DependencyUnavailable("Isaac Lab is unavailable for real Stage-4 branches") from exc
    parser = _parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    root = args.repo_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    bank_path = args.bank_path.expanduser().resolve()
    row_path = args.row_path.expanduser().resolve()
    spec = load_spec(args.spec)
    from diagnostics.common.canonical_collection import (
        CanonicalCollectionProtocol,
        DenseCheckpointRecord,
        assert_startup_randomization_matches,
        close_collection_trainer,
        collect_canonical_branch,
        make_collection_trainer,
        read_dense_checkpoint_records,
        switch_policy_state,
    )
    from diagnostics.common.imitation_6901 import Commit6901ImitationAdapter
    from diagnostics.common.noise_bank import CollectorMode
    from diagnostics.common.reward_stage4 import (
        BRANCH_BANK_SCHEMA,
        PRIMARY_REWARD_FAMILIES,
        RewardValidityProtocol,
        load_stage3_critics,
        score_flat_bank,
        write_branch_bank,
        write_parquet_exclusive,
    )
    from diagnostics.common.snapshot_bank import SnapshotBank
    from diagnostics.common.stage4_branch_collection import (
        BranchDescriptor,
        extract_real_branch,
        replay_audit,
        replay_max_abs_error,
    )
    protocol = RewardValidityProtocol.from_spec(spec)
    canonical = CanonicalCollectionProtocol.from_spec(spec)
    manifest = _status_pass(output_dir / "manifest.json", "diag_00")
    collection = _status_pass(output_dir / "canonical_rollout_index.status.json", "diag_12")
    _status_pass(output_dir / "split_audit.json", "diag_14")
    snapshot_status = _status_pass(output_dir / "snapshot_bank.status.json", "diag_11")
    index_path = output_dir / "canonical_rollout_index.parquet"
    if sha256_file(index_path) != collection["evidence"].get("canonical_rollout_index_sha256"):
        raise ProtocolError("canonical rollout index changed after diag_12")
    snapshot_path = output_dir / "snapshot_bank.pt"
    if sha256_file(snapshot_path) != snapshot_status["evidence"].get("snapshot_bank_sha256"):
        raise ProtocolError("snapshot bank changed after diag_11")
    snapshot_bank = SnapshotBank.load(snapshot_path)
    checkpoints = read_dense_checkpoint_records(
        output_dir / "checkpoints" / "dense_checkpoint_inventory.csv"
    )
    selected = {
        record.update: record
        for record in checkpoints
        if record.update in protocol.checkpoint_updates
    }
    if tuple(selected) != protocol.checkpoint_updates:
        raise DependencyUnavailable("dense inventory lacks a frozen Stage-4 checkpoint update")
    app_launcher = AppLauncher(args)
    trainer = None
    try:
        short_canonical = CanonicalCollectionProtocol(
            num_envs=canonical.num_envs,
            num_snapshots=canonical.num_snapshots,
            horizon=max(protocol.branch_horizons) + 1,
            phase_strategy=canonical.phase_strategy,
            snapshot_seed=canonical.snapshot_seed,
            collector_seed=canonical.collector_seed,
            common_sigmas=canonical.common_sigmas,
            canonical_checkpoint_updates=canonical.canonical_checkpoint_updates,
        )
        trainer = make_collection_trainer(
            app_launcher.app,
            repo_root=root,
            protocol=short_canonical,
            runtime_dir=output_dir / "runtime" / "diag_42",
        )
        assert_startup_randomization_matches(trainer.env, snapshot_bank)
        adapter = Commit6901ImitationAdapter(trainer.env, repo_root=root)
        start_phases = torch.tensor(
            [snapshot_bank.get(identity).phase for identity in snapshot_bank.snapshot_ids],
            device=trainer.env.device,
            dtype=torch.float32,
        )
        offsets = torch.arange(1 - 10, 1, device=trainer.env.device, dtype=torch.float32)
        phases = start_phases[:, None] + offsets[None, :] * float(trainer.env.motion_frame_delta)
        phases = phases.clamp(
            min=float(trainer.env.motion_start_phase), max=float(trainer.env.motion_end_phase)
        )
        demo_history = adapter.reference_expert_raw_frame(phases.reshape(-1)).reshape(
            len(snapshot_bank), 10, -1
        ).detach().cpu()
        endpoints_parts: list[torch.Tensor] = []
        body_parts: list[torch.Tensor] = []
        root_parts: list[torch.Tensor] = []
        active_parts: list[torch.Tensor] = []
        phase_parts: list[torch.Tensor] = []
        contact_parts: list[torch.Tensor] = []
        quality_rows: list[dict[str, Any]] = []
        descriptors: list[BranchDescriptor] = []

        def append(tree: Mapping[str, Any], descriptor: BranchDescriptor) -> None:
            endpoint, body, root_pos, active, phase, contact, rows = extract_real_branch(
                tree,
                descriptor,
                snapshot_ids=snapshot_bank.snapshot_ids,
                demo_history=demo_history,
                horizons=protocol.branch_horizons,
            )
            descriptors.append(descriptor)
            endpoints_parts.append(endpoint)
            body_parts.append(body)
            root_parts.append(root_pos)
            active_parts.append(active)
            phase_parts.append(phase)
            contact_parts.append(contact)
            branch_index = len(descriptors) - 1
            for row in rows:
                row["endpoint_branch_index"] = branch_index
            quality_rows.extend(rows)

        # Reuse the exact real canonical controlled rollouts for every frozen checkpoint.
        for update, checkpoint in selected.items():
            tree, index_row = _load_whole_shard(
                index_path,
                predicates={
                    "checkpoint_update": update,
                    "collector_mode": "controlled_environment",
                    "common_sigma": 0.0,
                },
                snapshot_ids=snapshot_bank.snapshot_ids,
            )
            append(
                tree,
                BranchDescriptor(
                    branch_id=f"checkpoint_u{update:04d}",
                    category="checkpoint_policy",
                    checkpoint_id=str(index_row["checkpoint_id"]),
                    checkpoint_sha256=str(index_row["checkpoint_sha256"]),
                    checkpoint_update=update,
                    checkpoint_lineage_id=str(index_row["checkpoint_lineage_id"]),
                    policy_domain=str(index_row["policy_domain"]),
                    local_action_candidate=False,
                    details={"source": "canonical_controlled_environment"},
                ),
            )

        action_iqr = _action_iqr(index_path, output_dir / "split_audit.json", update=500).to(
            trainer.env.device
        )
        # Adjacent checkpoint means are evaluated on the same evolving observation.
        for left_update, right_update in zip(protocol.checkpoint_updates[:-1], protocol.checkpoint_updates[1:]):
            left_record = selected[left_update]
            right_record = selected[right_update]
            switch_policy_state(trainer, left_record)
            left_policy = _FrozenPolicy(trainer.algo.actor, trainer.algo.actor_obs_normalizer)
            switch_policy_state(trainer, right_record)
            right_policy = _FrozenPolicy(trainer.algo.actor, trainer.algo.actor_obs_normalizer)
            for alpha in protocol.interpolation_alphas:
                identifier = f"interp_u{left_update:04d}_u{right_update:04d}_a{alpha:.2f}".replace(".", "p")
                synthetic_hash = canonical_sha256(
                    {"left": left_record.sha256, "right": right_record.sha256, "alpha": alpha}
                )
                synthetic = DenseCheckpointRecord(
                    checkpoint_id=identifier,
                    path=left_record.path,
                    sha256=synthetic_hash,
                    update=right_update,
                    lineage_id=left_record.lineage_id,
                    policy_domain="stage4_action_interpolation",
                )
                tree, _, _ = collect_canonical_branch(
                    trainer,
                    snapshot_bank,
                    synthetic,
                    protocol=short_canonical,
                    manifest=manifest,
                    spec=spec,
                    repo_root=root,
                    mode=CollectorMode.CONTROLLED_ENVIRONMENT,
                    common_sigma=0.0,
                    policy_adapter=_InterpolationAdapter(left_policy, right_policy, alpha, manifest),
                )
                append(
                    tree,
                    BranchDescriptor(
                        branch_id=identifier,
                        category="action_mean_interpolation",
                        checkpoint_id=identifier,
                        checkpoint_sha256=synthetic_hash,
                        checkpoint_update=right_update,
                        checkpoint_lineage_id=left_record.lineage_id,
                        policy_domain="stage4_action_interpolation",
                        local_action_candidate=True,
                        details={"left_update": left_update, "right_update": right_update, "alpha": alpha},
                    ),
                )
            del left_policy, right_policy

        base_record = selected[500]
        switch_policy_state(trainer, base_record)
        base_policy = _FrozenPolicy(trainer.algo.actor, trainer.algo.actor_obs_normalizer)
        perturb_trees: list[Mapping[str, Any]] = []
        for scale in protocol.perturbation_scales:
            identifier = f"perturb_u0500_s{scale:.2f}".replace(".", "p")
            synthetic_hash = canonical_sha256(
                {"base": base_record.sha256, "scale": scale, "action_iqr": action_iqr.detach().cpu().tolist()}
            )
            synthetic = DenseCheckpointRecord(
                checkpoint_id=identifier,
                path=base_record.path,
                sha256=synthetic_hash,
                update=500,
                lineage_id=base_record.lineage_id,
                policy_domain="stage4_action_perturbation",
            )
            tree, _, _ = collect_canonical_branch(
                trainer,
                snapshot_bank,
                synthetic,
                protocol=short_canonical,
                manifest=manifest,
                spec=spec,
                repo_root=root,
                mode=CollectorMode.CONTROLLED_ENVIRONMENT,
                common_sigma=0.0,
                policy_adapter=_PerturbationAdapter(base_policy, scale, action_iqr, manifest),
            )
            perturb_trees.append(tree)
            append(
                tree,
                BranchDescriptor(
                    branch_id=identifier,
                    category="action_perturbation",
                    checkpoint_id=identifier,
                    checkpoint_sha256=synthetic_hash,
                    checkpoint_update=500,
                    checkpoint_lineage_id=base_record.lineage_id,
                    policy_domain="stage4_action_perturbation",
                    local_action_candidate=True,
                    details={"base_update": 500, "scale": scale, "scale_basis": "u500_train_clean_action_IQR"},
                ),
            )

        # Repeat one complete perturbation branch from the same bank and NoiseBank.
        replay_scale = protocol.perturbation_scales[0]
        replay_id = "replay_verification"
        replay_record = DenseCheckpointRecord(
            checkpoint_id=replay_id,
            path=base_record.path,
            sha256=canonical_sha256({"replay": base_record.sha256, "scale": replay_scale}),
            update=500,
            lineage_id=base_record.lineage_id,
            policy_domain="stage4_action_perturbation",
        )
        replay_tree, _, _ = collect_canonical_branch(
            trainer,
            snapshot_bank,
            replay_record,
            protocol=short_canonical,
            manifest=manifest,
            spec=spec,
            repo_root=root,
            mode=CollectorMode.CONTROLLED_ENVIRONMENT,
            common_sigma=0.0,
            policy_adapter=_PerturbationAdapter(base_policy, replay_scale, action_iqr, manifest),
        )
        replay_details = replay_audit(perturb_trees[0], replay_tree, atol=1.0e-6)
        replay_error = replay_max_abs_error(perturb_trees[0], replay_tree)
        if not np.isfinite(replay_error) or replay_error > 1.0e-6:
            write_json_exclusive(
                output_dir / "stage4_replay_audit.json",
                {
                    "schema": "largebox_stage4_replay_audit_v1",
                    "status": "INVALID_PROTOCOL",
                    "summary": "same-snapshot replay exceeded the frozen tolerance",
                    "evidence": replay_details,
                },
            )
            raise ProtocolError(f"same-snapshot replay is not numerically stable: {replay_error}")

        # Reuse optional target-domain policies if their formal canonical collection exists.
        for domain, category, filename, status_name in (
            ("A_amp", "pure_amp", "canonical_A_amp_rollout_index.parquet", "canonical_A_amp_rollout_index.status.json"),
            ("B", "reference_free_bc", "canonical_B_rollout_index.parquet", "canonical_B_rollout_index.status.json"),
        ):
            domain_index = output_dir / filename
            domain_status = output_dir / status_name
            if not domain_index.is_file() or not domain_status.is_file() or read_json(domain_status).get("status") != PASS:
                continue
            tree, index_row = _load_whole_shard(
                domain_index,
                predicates={"collector_mode": "controlled_environment", "common_sigma": 0.0},
                snapshot_ids=snapshot_bank.snapshot_ids,
            )
            append(
                tree,
                BranchDescriptor(
                    branch_id=domain,
                    category=category,
                    checkpoint_id=str(index_row["checkpoint_id"]),
                    checkpoint_sha256=str(index_row["checkpoint_sha256"]),
                    checkpoint_update=int(index_row["checkpoint_update"]),
                    checkpoint_lineage_id=str(index_row["checkpoint_lineage_id"]),
                    policy_domain=str(index_row["policy_domain"]),
                    local_action_candidate=False,
                    details={"source": "canonical_controlled_environment"},
                ),
            )

        endpoint_bank = torch.stack(endpoints_parts)
        body_bank = torch.stack(body_parts)
        root_bank = torch.stack(root_parts)
        active_bank = torch.stack(active_parts)
        phase_bank = torch.stack(phase_parts)
        contact_bank = torch.stack(contact_parts)
        critics = load_stage3_critics(output_dir, spec, device="cpu")
        flat = endpoint_bank.reshape(-1, endpoint_bank.shape[-1]).numpy()
        scores = score_flat_bank(critics, flat)
        horizon_count = len(protocol.branch_horizons)
        snapshot_count = len(snapshot_bank)
        for row in quality_rows:
            flat_index = (
                int(row["endpoint_branch_index"]) * horizon_count * snapshot_count
                + int(row["endpoint_horizon_index"]) * snapshot_count
                + int(row["endpoint_env_index"])
            )
            for family in PRIMARY_REWARD_FAMILIES:
                for seed_index, seed in enumerate(critics.seeds):
                    row[f"reward_{family}_seed_{seed}"] = float(scores[family][seed_index, flat_index])
                row[f"reward_{family}_mean"] = float(np.mean(scores[family][:, flat_index]))
        payload = {
            "metadata": {
                "schema": BRANCH_BANK_SCHEMA,
                "real_physx_rollouts": True,
                "same_snapshot_replay_verified": True,
                "same_snapshot_replay_atol": 1.0e-6,
                "same_snapshot_replay_max_abs_error": replay_error,
                "shared_environment_randomness": True,
                "demo_seeded_commit6901_history": True,
                "source_classifier_used_as_reward": False,
                "critic_source_commit": critics.by_positive["K"][0].source_commit,
                "branch_ids": [value.branch_id for value in descriptors],
                "branch_descriptors": [value.to_dict() for value in descriptors],
                "snapshot_ids": list(snapshot_bank.snapshot_ids),
                "horizons": list(protocol.branch_horizons),
                "track_body_names": list(trainer.env.track_body_names),
                "body_position_frame": "instantaneous_root_relative",
                "root_position_frame": "displacement_from_same_snapshot",
                "checkpoint_updates": list(protocol.checkpoint_updates),
                "interpolation_alphas": list(protocol.interpolation_alphas),
                "interpolation_pairs": [list(pair) for pair in zip(protocol.checkpoint_updates[:-1], protocol.checkpoint_updates[1:])],
                "perturbation_scales": list(protocol.perturbation_scales),
                "perturbation_scale_basis": str(
                    spec["analysis_protocols"]["reward_validity"]["action_perturbation_scale_basis"]
                ),
                "action_iqr_source_checkpoint_update": 500,
                "action_iqr": action_iqr.detach().cpu(),
                "A_mix": "legacy_quarantined",
            },
            "endpoint_windows": endpoint_bank,
            "body_pos_local": body_bank,
            "root_pos_local": root_bank,
            "active": active_bank,
            "endpoint_phase": phase_bank,
            "endpoint_contact_mode": contact_bank,
        }
        write_branch_bank(bank_path, payload)
        write_parquet_exclusive(row_path, quality_rows)
        print(
            f"[stage4-real] branches={len(descriptors)} rows={len(quality_rows)} "
            f"bank={bank_path}",
            flush=True,
        )
    finally:
        if trainer is not None:
            close_collection_trainer(trainer)
    # Isaac/Kit teardown is known to hang indefinitely on this platform after
    # successful headless collection.  Artifacts are already fsynced and the
    # collector owns this subprocess, so terminate without synchronous Kit
    # teardown.  Failures still propagate normally with a traceback.
    return finish_isaac_entrypoint(
        0, isaac_launched=True, allow_hard_exit=_ALLOW_HARD_EXIT
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:  # keep the subprocess fail-closed without Kit teardown hangs
        traceback.print_exc()
        finish_isaac_entrypoint(1, isaac_launched=True, allow_hard_exit=True)
