from __future__ import annotations

import numpy as np

from diagnostics.common.policy_class import build_history_windows, knn_action_aliasing


def test_history_windows_never_cross_trajectory_boundary() -> None:
    observations = np.arange(12, dtype=np.float32).reshape(6, 2)
    actions = np.arange(6, dtype=np.float32).reshape(6, 1)
    trajectories = np.asarray([0, 0, 0, 1, 1, 1])
    windows, targets, ends = build_history_windows(
        observations, actions, trajectories, history=2
    )
    assert ends.tolist() == [1, 2, 4, 5]
    assert windows.shape == (4, 2, 2)
    assert targets[:, 0].tolist() == [1.0, 2.0, 4.0, 5.0]


def test_knn_aliasing_detects_same_observation_different_actions() -> None:
    features = np.zeros((20, 2), dtype=np.float64)
    features[:, 0] = np.linspace(0.0, 1.0e-5, 20)
    actions = np.zeros((20, 1), dtype=np.float64)
    actions[::2] = 2.0
    result = knn_action_aliasing(
        features,
        actions,
        np.arange(20),
        np.zeros(20),
        k=3,
        action_delta=0.5,
        phase_period=20.0,
    )
    assert result["action_alias_fraction"] > 0.4
