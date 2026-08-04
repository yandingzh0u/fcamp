from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from diagnostics.common.domain_data import (
    AMP_FRAME_DIM,
    AMP_SOURCE_COMMIT,
    AMP_WINDOW_STEPS,
    DomainBundle,
    amp_window_feature_groups,
)
from diagnostics.common.family_screeners import (
    EXECUTABILITY_BANK_SCHEMA,
    ExecutabilityDataset,
    deterministic_group_split,
    exact_paired_amp_windows,
    load_executability_dataset,
    same_state_binary_pair_accuracy,
    save_executability_dataset,
    train_denoising_prior,
    train_executability_classifier,
    train_paired_contrastive,
)
from diagnostics.common.imitation_6901 import imitation_contract_metadata
from diagnostics.common.manifest import ProtocolError


def _domain(name: str, family: str, *, execution: bool) -> DomainBundle:
    rng = np.random.default_rng(3 if execution else 2)
    count = 18
    split = np.asarray(["train"] * 6 + ["validation"] * 6 + ["test"] * 6)
    snapshots = np.asarray([f"snapshot-{index:02d}" for index in range(count)])
    endpoints = np.arange(9, 9 + count)
    return DomainBundle(
        name=name,
        family=family,
        features=rng.normal(size=(count, AMP_FRAME_DIM * AMP_WINDOW_STEPS)).astype(np.float32),
        split=split,
        sample_ids=np.asarray([f"{name}:trajectory-{index}:end={endpoint}" for index, endpoint in enumerate(endpoints)]),
        trajectory_ids=np.asarray([f"teacher-u500-{index}" for index in range(count)]),
        snapshot_ids=snapshots,
        checkpoint_lineage_ids=np.asarray(["fixed-reward-lineage"] * count),
        phase=np.linspace(0.05, 0.95, count),
        contact_mode=np.asarray([str(index % 4) for index in range(count)]),
        failure=np.asarray([index % 5 == 0 for index in range(count)]),
        collector_mode=np.asarray(["controlled_environment"] * count),
        feature_groups=amp_window_feature_groups(),
        window_steps=AMP_WINDOW_STEPS,
        coordinate_frame="mimickit_g1_amp_window_v1",
        metadata={
            "amp_source_commit": AMP_SOURCE_COMMIT,
            "amp_representation": "mimickit_g1_chronological_window",
            "amp_frame_dim": AMP_FRAME_DIM,
            "amp_window_steps": AMP_WINDOW_STEPS,
            "amp_root_xy_anchor": "newest_frame",
            "imitation_contract": imitation_contract_metadata(),
        },
    )


def test_diag60_pairing_is_exact_and_never_nearest_neighbor() -> None:
    reference = _domain("K", "K", execution=False)
    execution = _domain("T_u500", "T", execution=True)
    paired = exact_paired_amp_windows(reference, execution)
    assert paired.reference.shape == paired.execution.shape == (18, 2390)
    assert np.array_equal(paired.phase, reference.phase)

    changed_phase = np.asarray(execution.phase).copy()
    changed_phase[0] += 1.0e-9
    with pytest.raises(ProtocolError, match="approximate pairing is forbidden"):
        exact_paired_amp_windows(reference, replace(execution, phase=changed_phase))


def test_diag60_rejects_retired_domain_identity() -> None:
    reference = _domain("K", "K", execution=False)
    retired = replace(_domain("T_u500", "T", execution=True), name="A_mix")
    with pytest.raises(ProtocolError, match="only K<->T_u500"):
        exact_paired_amp_windows(reference, retired)


def _executability_dataset() -> ExecutabilityDataset:
    snapshots = np.asarray([f"s{index}" for index in range(8)])
    segments = np.asarray([f"g{index}" for index in range(7)])
    snapshot_ids = np.repeat(snapshots, len(segments))
    segment_ids = np.tile(segments, len(snapshots))
    success = np.asarray(
        [(snapshot_index + segment_index) % 3 == 0 for snapshot_index in range(8) for segment_index in range(7)],
        dtype=bool,
    )
    complete = success.copy()
    failure = ~success
    zeros = np.zeros_like(success)
    rng = np.random.default_rng(7)
    return ExecutabilityDataset(
        features=rng.normal(size=(success.size, 12)).astype(np.float32),
        success=success,
        segment_complete=complete,
        failure=failure,
        tracking_loss=failure.copy(),
        joint_limit_event=zeros,
        undesired_contact_event=zeros,
        snapshot_ids=snapshot_ids,
        segment_ids=segment_ids,
        phase_bins=np.tile(np.arange(7), 8),
        phase=np.tile(np.linspace(0.1, 0.9, 7), 8),
        requested_phase_offset=np.tile(np.arange(7), 8),
        time_scale=np.ones(success.size),
        metadata={
            "schema": EXECUTABILITY_BANK_SCHEMA,
            "real_physx_outcomes": True,
            "same_snapshot_replay_verified": True,
            "shared_environment_randomness": True,
            "label_source": "closed_loop_PhysX_outcomes_only",
            "reference_rmse_used_as_label": False,
            "raw_human_positive": False,
            "policy_method": "fixed_reward_ppo_teacher",
            "A_mix": "legacy_quarantined",
        },
    )


def test_executability_bank_roundtrip_preserves_only_physx_event_label(tmp_path) -> None:
    dataset = _executability_dataset()
    path = save_executability_dataset(tmp_path / "outcomes.npz", dataset)
    loaded = load_executability_dataset(path)
    assert np.array_equal(loaded.success, dataset.success)
    assert loaded.metadata["reference_rmse_used_as_label"] is False

    invalid = replace(dataset, success=~dataset.success)
    with pytest.raises(ProtocolError, match="event conjunction"):
        invalid.validate()


def test_executability_group_split_never_leaks_groups() -> None:
    groups = np.repeat([f"snapshot-{index}" for index in range(12)], 5)
    split = deterministic_group_split(groups, seed=20260803)
    for group in set(groups):
        assert len(set(split[groups == group])) == 1
    assert set(split) == {"train", "validation", "test"}


def test_same_state_pair_accuracy_uses_only_success_failure_order() -> None:
    scores = np.asarray([0.9, 0.2, 0.8, 0.1])
    labels = np.asarray([True, False, True, False])
    snapshots = np.asarray(["a", "a", "b", "b"])
    result = same_state_binary_pair_accuracy(scores, labels, snapshots)
    assert result["pair_count"] == 2
    assert result["accuracy"] == 1.0


def test_offline_screener_models_execute_without_policy_or_ppo() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=(72, 12)).astype(np.float32)
    prior, metadata = train_denoising_prior(
        values[:48],
        values[48:],
        hidden_dims=[512, 256],
        noise_scales=[0.01, 0.05, 0.10],
        epochs=1,
        batch_size=16,
        learning_rate=0.0003,
        seed=20260803,
    )
    assert prior.score(values[:4]).shape == (4,)
    assert np.isfinite(prior.input_gradient_norm(values[:4]))
    assert metadata["normalizer_source"].endswith("then_frozen")

    reference = {
        "train": values[:40],
        "validation": values[40:56],
        "test": values[56:],
    }
    execution = {
        name: array + 0.01 * rng.normal(size=array.shape).astype(np.float32)
        for name, array in reference.items()
    }
    _, contrastive = train_paired_contrastive(
        reference,
        execution,
        hidden_dims=[256, 256],
        latent_dim=32,
        temperature=0.07,
        batch_size=8,
        epochs_max=1,
        patience=1,
        seed=20260803,
        validation_metadata={
            "snapshot_ids": np.asarray([f"v{index}" for index in range(16)]),
            "phase": np.linspace(0.1, 0.9, 16),
        },
        phase_bins=16,
    )
    assert contrastive["epochs_ran"] == 1

    groups = np.repeat([f"group-{index}" for index in range(12)], 10)
    split = deterministic_group_split(groups, seed=20260803)
    labels = np.asarray(
        [position % 10 < 2 + (position // 10) % 5 for position in range(120)],
        dtype=bool,
    )
    probability, classifier = train_executability_classifier(
        rng.normal(size=(120, 12)).astype(np.float32),
        labels,
        split,
        hidden_dims=[256, 256],
        batch_size=16,
        epochs_max=1,
        patience=1,
        learning_rate=0.0003,
        seed=20260803,
    )
    assert probability.shape == (120,)
    assert 0.0 <= classifier["test_auroc"] <= 1.0
    assert 0.0 <= classifier["test_ece"] <= 1.0
