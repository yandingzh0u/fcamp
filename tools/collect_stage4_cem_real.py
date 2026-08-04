#!/usr/bin/env python3
"""Run the frozen H=10 CEM exploit test against real PhysX rollouts."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.manifest import (
    PASS,
    DependencyUnavailable,
    ProtocolError,
    load_spec,
    read_json,
    sha256_file,
    write_json_exclusive,
)
from diagnostics.common.isaac_exit import finish_isaac_entrypoint

_ALLOW_HARD_EXIT = __name__ == "__main__"


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--branch-bank", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    return parser


def _require_pass(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise DependencyUnavailable(f"{name} is absent")
    payload = read_json(path)
    if payload.get("status") != PASS:
        raise DependencyUnavailable(f"{name} is not PASS")
    return payload


class _Policy:
    def __init__(self, algo: Any) -> None:
        self.actor = copy.deepcopy(algo.actor).eval()
        self.normalizer = copy.deepcopy(algo.actor_obs_normalizer).eval()

    @torch.no_grad()
    def mean(self, observation: torch.Tensor) -> torch.Tensor:
        result = self.actor.act_inference(self.normalizer(observation, update=False))
        if not bool(torch.isfinite(result).all()):
            raise ProtocolError("CEM base policy produced NaN or Inf")
        return result


def main() -> int:
    try:
        from isaaclab.app import AppLauncher
    except (ImportError, ModuleNotFoundError) as exc:
        raise DependencyUnavailable("Isaac Lab is unavailable for real CEM") from exc
    parser = _base_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    root = args.repo_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    result_path = args.result_path.expanduser().resolve()
    spec = load_spec(args.spec)
    from components.imitation.motion_features import canonicalize_imitation_window
    from diagnostics.common.canonical_collection import (
        CanonicalCollectionProtocol,
        assert_startup_randomization_matches,
        close_collection_trainer,
        collector_environment,
        make_collection_trainer,
        read_dense_checkpoint_records,
        restore_snapshot_bank,
        switch_policy_state,
    )
    from diagnostics.common.imitation_6901 import Commit6901ImitationAdapter
    from diagnostics.common.noise_bank import CollectorMode, NoiseBank
    from diagnostics.common.reward_stage4 import (
        CEM_BANK_SCHEMA,
        RewardValidityProtocol,
        load_branch_bank,
        load_stage3_critics,
    )
    from diagnostics.common.snapshot_bank import SnapshotBank

    protocol = RewardValidityProtocol.from_spec(spec)
    cem = dict(protocol.cem)
    branch_bank = load_branch_bank(args.branch_bank)
    action_iqr = branch_bank["metadata"].get("action_iqr")
    if not torch.is_tensor(action_iqr) or action_iqr.ndim != 1:
        raise ProtocolError("branch bank lacks the frozen action-IQR scale")
    if branch_bank["metadata"].get("perturbation_scale_basis") != spec["analysis_protocols"]["reward_validity"].get(
        "action_perturbation_scale_basis"
    ):
        raise ProtocolError("CEM action scale differs from Stage-4 branch scaling")
    manifest = _require_pass(output_dir / "manifest.json", "diag_00")
    snapshot_status = _require_pass(output_dir / "snapshot_bank.status.json", "diag_11")
    snapshot_path = output_dir / "snapshot_bank.pt"
    if sha256_file(snapshot_path) != snapshot_status["evidence"].get("snapshot_bank_sha256"):
        raise ProtocolError("snapshot bank changed after diag_11")
    snapshot_bank = SnapshotBank.load(snapshot_path)
    canonical = CanonicalCollectionProtocol.from_spec(spec)
    checkpoints = read_dense_checkpoint_records(
        output_dir / "checkpoints" / "dense_checkpoint_inventory.csv"
    )
    base_record = next((record for record in checkpoints if record.update == 500), None)
    if base_record is None:
        raise DependencyUnavailable("u500 base checkpoint is absent")
    critics = load_stage3_critics(output_dir, spec, device="cpu")
    search_seed = int(cem["seeds"][0])
    audit_seed = int(critics.seeds[-1])
    search_index = critics.seeds.index(search_seed)
    audit_index = critics.seeds.index(audit_seed)
    search_fit = critics.by_positive["T_u500"][search_index]
    audit_fit = critics.by_positive["T_u500"][audit_index]
    search_record = critics.records_by_positive["T_u500"][search_index]
    audit_record = critics.records_by_positive["T_u500"][audit_index]
    if search_seed == audit_seed or search_record["sha256"] == audit_record["sha256"]:
        raise ProtocolError("CEM search and audit critics are not independent")
    app_launcher = AppLauncher(args)
    trainer = None
    try:
        short = CanonicalCollectionProtocol(
            num_envs=canonical.num_envs,
            num_snapshots=canonical.num_snapshots,
            horizon=int(cem["horizon"]),
            phase_strategy=canonical.phase_strategy,
            snapshot_seed=canonical.snapshot_seed,
            collector_seed=canonical.collector_seed,
            common_sigmas=canonical.common_sigmas,
            canonical_checkpoint_updates=canonical.canonical_checkpoint_updates,
        )
        trainer = make_collection_trainer(
            app_launcher.app,
            repo_root=root,
            protocol=short,
            runtime_dir=output_dir / "runtime" / "diag_44",
        )
        assert_startup_randomization_matches(trainer.env, snapshot_bank)
        switch_policy_state(trainer, base_record)
        policy = _Policy(trainer.algo)
        imitation = Commit6901ImitationAdapter(trainer.env, repo_root=root)
        start_phase = torch.tensor(
            [snapshot_bank.get(identity).phase for identity in snapshot_bank.snapshot_ids],
            device=trainer.env.device,
            dtype=torch.float32,
        )
        offsets = torch.arange(-9, 1, device=trainer.env.device, dtype=torch.float32)
        history_phase = (
            start_phase[:, None] + offsets[None, :] * float(trainer.env.motion_frame_delta)
        ).clamp(min=float(trainer.env.motion_start_phase), max=float(trainer.env.motion_end_phase))
        demo_history = imitation.reference_expert_raw_frame(history_phase.reshape(-1)).reshape(
            len(snapshot_bank), 10, -1
        ).detach().cpu()
        noise_bank = NoiseBank(seed=canonical.collector_seed)
        action_iqr_device = action_iqr.to(trainer.env.device, dtype=torch.float32)
        horizon = int(cem["horizon"])

        @torch.no_grad()
        def rollout(delta: torch.Tensor, *, quality: bool) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
            if tuple(delta.shape) != (horizon, int(trainer.env.action_dim)):
                raise ProtocolError("CEM action-offset sequence has an invalid shape")
            restore_snapshot_bank(trainer.env, snapshot_bank, mode=CollectorMode.CONTROLLED_ENVIRONMENT)
            phys_frames: list[torch.Tensor] = []
            done_steps: list[torch.Tensor] = []
            failures: list[torch.Tensor] = []
            completions: list[torch.Tensor] = []
            progresses: list[torch.Tensor] = []
            joint_limits: list[torch.Tensor] = []
            undesired: list[torch.Tensor] = []
            with collector_environment(
                trainer.env,
                noise_bank,
                snapshot_bank.snapshot_ids,
                mode=CollectorMode.CONTROLLED_ENVIRONMENT,
                horizon=horizon,
            ) as randomness:
                randomness.begin_observation(0)
                observation = trainer.env.get_observation()
                for step in range(horizon):
                    action = _apply_bounds(
                        policy.mean(observation) + delta[step],
                        trainer.algo.action_low,
                        trainer.algo.action_high,
                    )
                    randomness.begin_observation(step + 1)
                    observation, _, done, info = trainer.env.step(action)
                    phys_frames.append(imitation.agent_physx_raw_frame().detach().cpu())
                    done_terms = info["done_terms"]
                    failure = done_terms["anchor_pos_bad"] | done_terms["anchor_ori_bad"] | done_terms["ee_body_bad"]
                    done_steps.append(done.detach().cpu().bool())
                    failures.append(failure.detach().cpu().bool())
                    completions.append(done_terms["motion_complete"].detach().cpu().bool())
                    denominator = max(1.0, float(trainer.env.motion_end_phase - trainer.env.motion_start_phase))
                    progresses.append(
                        ((trainer.env.phase_steps - float(trainer.env.motion_start_phase)) / denominator)
                        .detach()
                        .cpu()
                        .float()
                    )
                    joint_limits.append(info["reward_terms"]["joint_limit"].detach().cpu().float())
                    undesired.append(info["reward_terms"]["undesired_contacts"].detach().cpu().float())
            frames = torch.stack(phys_frames)
            done_bank = torch.stack(done_steps)
            windows = []
            used_steps = []
            for env_index in range(len(snapshot_bank)):
                found = torch.where(done_bank[:, env_index])[0]
                used = int(found[0].item() + 1) if found.numel() else horizon
                used_steps.append(used)
                sequence = torch.cat((demo_history[env_index], frames[:used, env_index]), dim=0)
                windows.append(sequence[-10:])
            flat = canonicalize_imitation_window(torch.stack(windows)).reshape(len(windows), -1)
            if not quality:
                return flat.numpy(), None
            failure_bank = torch.stack(failures)
            complete_bank = torch.stack(completions)
            progress_bank = torch.stack(progresses)
            joint_bank = torch.stack(joint_limits)
            contact_bank = torch.stack(undesired)
            panel = {
                "motion_complete": np.empty(len(snapshot_bank)),
                "failure": np.empty(len(snapshot_bank)),
                "reference_progress": np.empty(len(snapshot_bank)),
                "survival": np.asarray(used_steps, dtype=np.float64),
                "joint_limit_incidence": np.empty(len(snapshot_bank)),
                "undesired_contacts": np.empty(len(snapshot_bank)),
            }
            for env_index, used in enumerate(used_steps):
                end = used - 1
                panel["motion_complete"][env_index] = float(complete_bank[end, env_index])
                panel["failure"][env_index] = float(failure_bank[:used, env_index].any())
                panel["reference_progress"][env_index] = float(progress_bank[end, env_index])
                panel["joint_limit_incidence"][env_index] = float(
                    (joint_bank[:used, env_index] > 0.0).float().mean()
                )
                panel["undesired_contacts"][env_index] = float(contact_bank[:used, env_index].mean())
            return flat.numpy(), panel

        def score_many(deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            windows = []
            for candidate in deltas:
                values, _ = rollout(
                    torch.as_tensor(candidate, device=trainer.env.device, dtype=torch.float32),
                    quality=False,
                )
                windows.append(values)
            bank = np.stack(windows)
            scores = search_fit.rewards(bank.reshape(-1, bank.shape[-1])).reshape(
                len(deltas), len(snapshot_bank)
            )
            return scores.mean(axis=1), scores

        zero = np.zeros((horizon, int(trainer.env.action_dim)), dtype=np.float32)
        baseline_windows, baseline_panel = rollout(
            torch.zeros_like(torch.as_tensor(zero, device=trainer.env.device)), quality=True
        )
        assert baseline_panel is not None
        baseline_search = search_fit.rewards(baseline_windows)
        baseline_audit = audit_fit.rewards(baseline_windows)
        records: list[dict[str, Any]] = []
        traces: dict[str, list[float]] = {}
        optimized_window_parts: list[torch.Tensor] = []
        for cem_seed in (int(value) for value in cem["seeds"]):
            rng = np.random.default_rng(cem_seed)
            mean = zero.copy()
            std = float(cem["initial_std_in_action_scale_units"]) * action_iqr.numpy()[None, :]
            baseline_objective = float(baseline_search.mean())
            best_score = baseline_objective
            best_delta = zero.copy()
            trace = [baseline_objective]
            for _ in range(int(cem["iterations"])):
                candidates = rng.normal(
                    loc=mean,
                    scale=std,
                    size=(int(cem["population"]), horizon, int(trainer.env.action_dim)),
                ).astype(np.float32)
                candidates[0] = mean
                objectives, _ = score_many(candidates)
                elite_indices = np.argsort(objectives)[-int(cem["elite_count"]):]
                elite = candidates[elite_indices]
                mean = elite.mean(axis=0)
                std = elite.std(axis=0) * float(cem["std_decay"])
                iteration_best = int(np.argmax(objectives))
                if float(objectives[iteration_best]) > best_score:
                    best_score = float(objectives[iteration_best])
                    best_delta = candidates[iteration_best].copy()
                trace.append(best_score)
            optimized_windows, optimized_panel = rollout(
                torch.as_tensor(best_delta, device=trainer.env.device), quality=True
            )
            assert optimized_panel is not None
            optimized_search = search_fit.rewards(optimized_windows)
            optimized_audit = audit_fit.rewards(optimized_windows)
            optimized_window_parts.append(torch.from_numpy(optimized_windows).float())
            traces[str(cem_seed)] = trace
            for env_index, snapshot_id in enumerate(snapshot_bank.snapshot_ids):
                records.append(
                    {
                        "snapshot_id": snapshot_id,
                        "seed": cem_seed,
                        "baseline_search_reward": float(baseline_search[env_index]),
                        "optimized_search_reward": float(optimized_search[env_index]),
                        "baseline_audit_reward": float(baseline_audit[env_index]),
                        "optimized_audit_reward": float(optimized_audit[env_index]),
                        "best_search_reward_by_iteration": trace,
                        "baseline_outcomes": {
                            name: float(values[env_index]) for name, values in baseline_panel.items()
                        },
                        "optimized_outcomes": {
                            name: float(values[env_index]) for name, values in optimized_panel.items()
                        },
                    }
                )
        window_bank_path = result_path.with_name(f"{result_path.stem}_windows.pt")
        window_payload = {
            "schema": "largebox_cem_endpoint_window_bank_v1",
            "snapshot_ids": list(snapshot_bank.snapshot_ids),
            "seeds": [int(value) for value in cem["seeds"]],
            "baseline_windows": torch.from_numpy(baseline_windows).float(),
            "optimized_windows_by_seed": torch.stack(optimized_window_parts),
            "window_steps": 10,
            "frame_dim": 239,
            "real_physx_rollouts": True,
            "A_mix": "legacy_quarantined",
        }
        window_bank_path.parent.mkdir(parents=True, exist_ok=True)
        with window_bank_path.open("xb") as handle:
            torch.save(window_payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(str(window_bank_path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        result = {
            "schema": CEM_BANK_SCHEMA,
            "real_physx_rollouts": True,
            "same_snapshot_replay_verified": True,
            "audit_critic_independent": True,
            "source_classifier_used_as_reward": False,
            "A_mix": "legacy_quarantined",
            "protocol": cem,
            "search_critic": {
                "path": str(search_record["path"]),
                "sha256": str(search_record["sha256"]),
                "source_negative": "A_amp",
                "destination_positive": "T_u500",
                "seed": search_seed,
            },
            "audit_critic": {
                "path": str(audit_record["path"]),
                "sha256": str(audit_record["sha256"]),
                "source_negative": "A_amp",
                "destination_positive": "T_u500",
                "seed": audit_seed,
            },
            "cem_trace_by_seed": traces,
            "endpoint_window_bank": str(window_bank_path),
            "endpoint_window_bank_sha256": sha256_file(window_bank_path),
            "records": records,
        }
        write_json_exclusive(result_path, result)
        print(f"[stage4-cem-real] records={len(records)} result={result_path}", flush=True)
    finally:
        if trainer is not None:
            close_collection_trainer(trainer)
    return finish_isaac_entrypoint(
        0, isaac_launched=True, allow_hard_exit=_ALLOW_HARD_EXIT
    )


def _apply_bounds(value: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.minimum(value, high), low)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:  # preserve traceback, then avoid a hanging Kit destructor
        traceback.print_exc()
        finish_isaac_entrypoint(1, isaac_launched=True, allow_hard_exit=True)
