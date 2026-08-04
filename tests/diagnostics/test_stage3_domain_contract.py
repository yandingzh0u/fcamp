from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from diagnostics.common.domain_data import (
    DomainBundle,
    GapLayerBundle,
    amp_domain_feature_contract,
    build_domain_index,
    load_domain_bundle,
    load_gap_layer_bundle,
    save_domain_bundle,
    save_gap_layer_bundle,
    chronological_amp_windows,
    amp_window_feature_groups,
)
from diagnostics.common.domain_triangle import (
    DomainSplit,
    causal_group_swap_audit,
    pairwise_source_matrix,
    reward_ordering_stability,
)
from diagnostics.common.noise_bank import NoiseProtocolError
from diagnostics.common.imitation_6901 import imitation_contract_metadata
from tools.diag_30_build_domain_triangle import _catalog_from_frozen_spec


def _bundle(name: str, family: str, *, seed: int = 0, mode: str = "clean_mean") -> DomainBundle:
    rng = np.random.default_rng(seed)
    count = 18
    return DomainBundle(
        name=name,
        family=family,
        features=rng.normal(size=(count, 2390)),
        split=np.asarray(["train"] * 10 + ["validation"] * 4 + ["test"] * 4),
        sample_ids=np.asarray([f"{name}-{index}" for index in range(count)]),
        trajectory_ids=np.asarray([f"{name}-trajectory-{index}" for index in range(count)]),
        snapshot_ids=np.asarray([f"snapshot-{index}" for index in range(count)]),
        checkpoint_lineage_ids=np.asarray([f"{name}-lineage"] * count),
        phase=np.linspace(0.0, 1.0, count),
        contact_mode=np.asarray(["01"] * count),
        failure=np.zeros(count, dtype=bool),
        collector_mode=np.asarray([mode] * count),
        feature_groups=amp_window_feature_groups(),
        window_steps=10,
        coordinate_frame="mimickit_g1_amp_window_v1",
        metadata={
            "fixture": True,
            "amp_source_commit": "6901e302499711e2207687e1342348a4078330f8",
            "amp_representation": "mimickit_g1_chronological_window",
            "amp_frame_dim": 239,
            "amp_window_steps": 10,
            "amp_root_xy_anchor": "newest_frame",
            "imitation_contract": imitation_contract_metadata(),
            "derived_from_unified_collector": True,
            "rollout_index_sha256": "a" * 64,
            "split_audit_sha256": "b" * 64,
            "source_shards": [{"path": "fixture.pt", "sha256": "c" * 64}],
        },
    )


def test_domain_triangle_bundle_roundtrip_and_five_families(tmp_path: Path) -> None:
    entries = []
    for index, (name, family) in enumerate(
        (("K", "K"), ("T_u200", "T_early"), ("T_u500", "T"), ("A_amp", "A_amp"), ("B", "B"))
    ):
        path = save_domain_bundle(tmp_path / f"{name}.npz", _bundle(name, family, seed=index))
        assert load_domain_bundle(path).feature_schema_sha256 == _bundle(name, family, seed=index).feature_schema_sha256
        entries.append({"name": name, "bundle_path": path.name})
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"domains": entries}), encoding="utf-8")
    index = build_domain_index(catalog)
    assert index["status"] == "PASS"
    assert set(index["domain_families"]) == {"K", "T_early", "T", "A_amp", "B"}


def test_domain_triangle_exposes_one_exact_offline_amp_feature_contract() -> None:
    first = amp_domain_feature_contract(_bundle("K", "K"))
    second = amp_domain_feature_contract(_bundle("T_u500", "T"))
    assert first == second
    assert first["input_dim"] == 2390
    assert first["imitation_contract"]["imitation_frame_schema_sha256"] == (
        "73c60df570e69bcba2ce324e294f0df492f28ad10916f0f6323a948c68e2cb7e"
    )


def test_native_stochastic_bundle_is_rejected_from_stage3(tmp_path: Path) -> None:
    with pytest.raises(NoiseProtocolError):
        save_domain_bundle(
            tmp_path / "native.npz",
            _bundle("A_amp", "A_amp", mode="native_stochastic"),
        )


def test_gap_bundle_requires_exact_four_same_phase_layers(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    base = rng.normal(size=(8, 239))
    bundle = GapLayerBundle(
        layers={
            "motion_npz": base,
            "reset_readback": base + 0.1,
            "teacher_one_step": base + 0.2,
            "teacher_closed_loop": base + 0.3,
        },
        sample_ids=np.asarray([f"g{index}" for index in range(8)]),
        phase=np.linspace(0, 1, 8),
        scale=np.ones(239),
        feature_groups={"frame": (0, 239)},
        metadata={
            "alignment": "same_phase_conditioned",
            "imitation_contract": imitation_contract_metadata(),
            "real_physx_replay_verified": True,
        },
    )
    path = save_gap_layer_bundle(tmp_path / "gap.npz", bundle)
    loaded = load_gap_layer_bundle(path)
    assert tuple(loaded.layers) == (
        "motion_npz", "reset_readback", "teacher_one_step", "teacher_closed_loop"
    )


def test_source_seeds_share_identical_held_out_bank_and_group_swap_finds_shortcut() -> None:
    rng = np.random.default_rng(7)
    first = DomainSplit(
        train=np.column_stack((rng.normal(-3, 0.2, 200), rng.normal(size=200))),
        validation=np.column_stack((rng.normal(-3, 0.2, 80), rng.normal(size=80))),
        test=np.column_stack((rng.normal(-3, 0.2, 100), rng.normal(size=100))),
    )
    second = DomainSplit(
        train=np.column_stack((rng.normal(3, 0.2, 200), rng.normal(size=200))),
        validation=np.column_stack((rng.normal(3, 0.2, 80), rng.normal(size=80))),
        test=np.column_stack((rng.normal(3, 0.2, 100), rng.normal(size=100))),
    )
    _, details = pairwise_source_matrix({"K": first, "T": second}, seeds=(1, 2, 3))
    labels = [record["labels"] for record in details[("K", "T")]]
    assert all(np.array_equal(labels[0], value) for value in labels[1:])
    causal = causal_group_swap_audit(
        first, second, groups={"shortcut": (0, 1), "noise": (1, 2)}, seed=11
    )
    assert causal["groups"]["shortcut"]["baseline_auc_minus_erasure_auc"] > 0.4
    assert causal["groups"]["shortcut"]["swap_toward_donor_probability_mean"] > 0.3


def test_reward_ordering_stability_reports_decile_agreement() -> None:
    base = np.linspace(-1, 1, 100)
    result = reward_ordering_stability((base, base + 0.01, 2 * base))
    assert result["icc_consistency"] > 0.7
    assert result["top_decile_jaccard_mean"] == 1.0
    assert result["bottom_decile_jaccard_mean"] == 1.0


def test_amp_windows_are_complete_alive_and_anchor_xy_to_newest() -> None:
    frames = np.zeros((12, 239), dtype=np.float32)
    frames[:, 0] = np.arange(12)
    frames[:, 1] = 2 * np.arange(12)
    done = np.zeros(12, dtype=bool)
    done[10] = True
    windows, endpoints = chronological_amp_windows(frames, done=done)
    assert endpoints.tolist() == [9]
    reshaped = windows.reshape(-1, 10, 239)
    assert np.allclose(reshaped[0, -1, :2], 0.0)
    assert np.allclose(reshaped[0, 0, :2], (-9.0, -18.0))


def test_domain_bundle_rejects_snapshot_split_leakage() -> None:
    bundle = _bundle("K", "K")
    splits = bundle.split.copy()
    snapshots = bundle.snapshot_ids.copy()
    snapshots[-1] = snapshots[0]
    leaked = DomainBundle(
        name=bundle.name,
        family=bundle.family,
        features=bundle.features,
        split=splits,
        sample_ids=bundle.sample_ids,
        trajectory_ids=bundle.trajectory_ids,
        snapshot_ids=snapshots,
        checkpoint_lineage_ids=bundle.checkpoint_lineage_ids,
        phase=bundle.phase,
        contact_mode=bundle.contact_mode,
        failure=bundle.failure,
        collector_mode=bundle.collector_mode,
        feature_groups=bundle.feature_groups,
        window_steps=bundle.window_steps,
        coordinate_frame=bundle.coordinate_frame,
        metadata=bundle.metadata,
    )
    with pytest.raises(Exception, match="snapshot_id"):
        leaked.validate()


def test_diag30_generates_catalog_only_from_frozen_spec(tmp_path: Path) -> None:
    bundle = save_domain_bundle(tmp_path / "K.npz", _bundle("K", "K"))
    target = tmp_path / "output" / "domain_catalog.json"
    spec = {
        "suite_id": "fixture",
        "analysis_protocols": {
            "domain_triangle_domains": {
                "protocol_version": "largebox_domain_triangle_v1",
                "primary_collector_mode": "controlled_environment",
                "failure_label": "trajectory_eventual",
                "catalog": [
                    {"name": "K", "family": "K", "frame_role": "reference_expert", "bundle_path": "{repo_root}/K.npz"},
                    {"name": "T_u200", "family": "T_early", "frame_role": "agent_physx", "bundle_path": "{repo_root}/K.npz"},
                    {"name": "T_u500", "family": "T", "frame_role": "agent_physx", "bundle_path": "{repo_root}/K.npz"},
                    {"name": "A_amp", "family": "A_amp", "frame_role": "agent_physx", "bundle_path": "{repo_root}/K.npz"},
                    {"name": "B", "family": "B", "frame_role": "agent_physx", "bundle_path": "{repo_root}/K.npz"},
                ],
            }
        },
    }
    _catalog_from_frozen_spec(
        spec, repo_root=tmp_path, output_dir=target.parent, target=target
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["generated_from_frozen_spec"] is True
    assert payload["domains"][0]["bundle_path"] == str(bundle)
