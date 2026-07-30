from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from envs.imitation_data import (
    G1_IMITATION_FRAME_DIM,
    build_g1_amp_actor_observation,
)


@pytest.fixture()
def observation_mixin(monkeypatch):
    math_module = ModuleType("isaaclab.utils.math")
    for name in (
        "matrix_from_quat",
        "quat_apply",
        "quat_inv",
        "quat_mul",
        "subtract_frame_transforms",
        "yaw_quat",
    ):
        setattr(math_module, name, lambda *args: args[0])
    monkeypatch.setitem(sys.modules, "isaaclab", ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.utils", ModuleType("isaaclab.utils"))
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_module)
    spec_module = ModuleType("envs.spec")
    spec_module.CRITIC_OBS_DIM = G1_IMITATION_FRAME_DIM - 2
    spec_module.OBS_DIM = G1_IMITATION_FRAME_DIM - 2
    monkeypatch.setitem(sys.modules, "envs.spec", spec_module)

    path = Path(__file__).parents[1] / "envs" / "observation.py"
    spec = importlib.util.spec_from_file_location("envs._self_obs_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MimicObservationMixin


def test_actor_observation_is_reference_free_self_state(
    observation_mixin,
) -> None:
    class Fixture(observation_mixin):
        def get_imitation_policy_frame(self, env_ids=None):
            assert env_ids is None
            return self.frame

        def get_tracking_context(self):
            raise AssertionError("actor observation must not query reference state")

    batch = 3
    frame = torch.randn(batch, G1_IMITATION_FRAME_DIM)
    env = Fixture()
    env.frame = frame

    observation = env.build_observation()

    assert observation.shape == (batch, G1_IMITATION_FRAME_DIM - 2)
    torch.testing.assert_close(
        observation,
        build_g1_amp_actor_observation(frame),
    )

    # Global root x/y and reference/phase bookkeeping are not actor inputs.
    env.frame[:, :2] += torch.tensor([100.0, -50.0])
    env.phase_steps = torch.full((batch,), 999.0)
    env.reference = torch.randn(batch, 17)
    torch.testing.assert_close(env.build_observation(), observation)


def test_actor_observation_matches_official_g1_field_order() -> None:
    frame = torch.arange(
        G1_IMITATION_FRAME_DIM, dtype=torch.float32
    ).unsqueeze(0)

    observation = build_g1_amp_actor_observation(frame)

    expected_indices = (
        [2]
        + list(range(3, 9))
        + list(range(204, 207))
        + list(range(207, 210))
        + list(range(9, 189))
        + list(range(210, 239))
        + list(range(189, 204))
    )
    assert observation.shape == (1, 237)
    torch.testing.assert_close(
        observation,
        frame[:, expected_indices],
    )
