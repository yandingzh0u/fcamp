from __future__ import annotations

import numpy as np

from diagnostics.common.offline_amp import (
    SOURCE_COMMIT,
    directed_offline_amp_protocol,
    held_out_amp_metrics,
    load_offline_amp_fit,
    save_offline_amp_fit,
    train_offline_amp_critic,
)
from diagnostics.common.manifest import canonical_sha256


def _protocol() -> dict:
    return {
        "source_commit": SOURCE_COMMIT,
        "optimizer": "sgd_momentum",
        "checkpoint_selection": "final_epoch_no_validation_selection",
        "balanced_train_windows_per_domain_max": 64,
        "batch_size_per_class": 32,
        "epochs": 3,
        "hidden_dims": [16, 8],
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "gradient_penalty": 0.1,
        "logit_regularization": 0.001,
        "normalizer_clip": 10.0,
        "reward_scale": 2.0,
        "reward_epsilon": 0.0001,
    }


def test_offline_amp_critic_learns_the_frozen_positive_direction() -> None:
    rng = np.random.default_rng(9)
    negative = rng.normal(-1.5, 0.3, size=(64, 3)).astype(np.float32)
    positive = rng.normal(1.5, 0.3, size=(64, 3)).astype(np.float32)
    protocol = _protocol()
    fit = train_offline_amp_critic(negative, positive, seed=3, protocol=protocol)
    metrics = held_out_amp_metrics(fit, negative, positive)
    assert metrics["auc"] > 0.9
    assert metrics["positive_reward_mean"] > metrics["negative_reward_mean"]


def test_offline_amp_fit_roundtrip_preserves_logits_and_contract(tmp_path) -> None:
    rng = np.random.default_rng(11)
    negative = rng.normal(-1.0, 0.4, size=(64, 4)).astype(np.float32)
    positive = rng.normal(1.0, 0.4, size=(64, 4)).astype(np.float32)
    protocol = _protocol()
    fit = train_offline_amp_critic(negative, positive, seed=17, protocol=protocol)
    feature_contract = {
        "feature_schema_sha256": "a" * 64,
        "input_dim": 4,
        "imitation_frame_schema_sha256": "b" * 64,
    }
    target = save_offline_amp_fit(
        tmp_path / "seed_17.pt",
        fit,
        protocol=protocol,
        feature_contract=feature_contract,
        training_provenance={"negative_domain": "A_amp", "positive_domain": "K"},
    )
    loaded, metadata = load_offline_amp_fit(target)
    assert loaded.seed == 17
    assert metadata["feature_contract"] == feature_contract
    assert np.allclose(loaded.logits(negative), fit.logits(negative), atol=0.0, rtol=0.0)


def test_directed_amp_protocol_rebinds_domain_labels_without_changing_base() -> None:
    base = {**_protocol(), "positive_domains": ["K", "T_u500"], "negative_domain": "A_amp"}
    directed = directed_offline_amp_protocol(
        base, negative_domain="B", positive_domain="T_u200"
    )
    assert base["negative_domain"] == "A_amp"
    assert directed["negative_domain"] == "B"
    assert directed["positive_domains"] == ["T_u200"]
    assert directed["directed_edge_id"] == "B_to_T_u200"
    assert directed["base_protocol_sha256"] == canonical_sha256(base)
