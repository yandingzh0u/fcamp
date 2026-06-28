"""End-to-end shape/smoke test for FPO++ collect -> update -> deterministic_actions.

Uses a tiny pure-torch fake env (no IsaacLab) that mimics the slice of the env interface
FPO++ touches. Verifies the full rollout/update pipeline runs and produces sane shapes, that
one optimizer step actually changes the policy, and that zero-sampling is deterministic.

Run:  python tests/test_fpo_pp_rollout.py   (from the repo root)
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_fpo_pp_module():
    algorithms_pkg = types.ModuleType("algorithms")
    algorithms_pkg.__path__ = [str(REPO_ROOT / "algorithms")]
    base_mod = types.ModuleType("algorithms.base")

    class _Algorithm:
        def __init__(self, cfg, env, simulation_app):
            self.cfg = cfg
            self.env = env
            self.simulation_app = simulation_app

    base_mod.Algorithm = _Algorithm
    sys.modules["algorithms"] = algorithms_pkg
    sys.modules["algorithms.base"] = base_mod

    core_pkg = types.ModuleType("core")
    core_pkg.__path__ = [str(REPO_ROOT / "core")]
    logging_mod = types.ModuleType("core.logging")
    logging_mod.log_shared_tracking = lambda *a, **k: None
    logging_mod.log_shared_update_diagnostics = lambda *a, **k: None
    sys.modules["core"] = core_pkg
    sys.modules["core.logging"] = logging_mod

    spec = importlib.util.spec_from_file_location("fpo_pp_standalone", REPO_ROOT / "algorithms" / "fpo_pp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fpo = _load_fpo_pp_module()

DONE_KEYS = ("time_out", "anchor_pos_bad", "anchor_ori_bad", "ee_body_bad")
REWARD_KEYS = ("track", "energy")


class FakeEnv:
    """Minimal deterministic-ish env mimicking the interface FPO++ uses."""

    def __init__(self, num_envs=8, obs_dim=20, critic_dim=24, action_dim=3):
        self.num_envs = num_envs
        self.observation_dim = obs_dim
        self.critic_observation_dim = critic_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu")
        self.task_cfg = types.SimpleNamespace(max_episode_steps=20, motion_file="fake_motion")
        self.episode_steps = torch.zeros(num_envs, dtype=torch.long)
        self._gen = torch.Generator().manual_seed(123)
        self._step = 0

    def _obs(self):
        return torch.randn(self.num_envs, self.observation_dim, generator=self._gen)

    def _critic(self):
        return torch.randn(self.num_envs, self.critic_observation_dim, generator=self._gen)

    def reset(self, **kw):
        self.episode_steps.zero_()
        self._cur_obs = self._obs()
        self._cur_critic = self._critic()
        return self._cur_obs

    def get_observation(self):
        return self._cur_obs

    def get_critic_observation(self):
        return self._cur_critic

    def step(self, action, auto_reset=True, reset_horizon=1):
        self._step += 1
        assert action.shape == (self.num_envs, self.action_dim)
        reward = torch.randn(self.num_envs, generator=self._gen) * 0.1
        # Make a couple of envs terminate periodically to exercise done handling.
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        timeout = torch.zeros(self.num_envs, dtype=torch.bool)
        if self._step % 5 == 0:
            done[0] = True              # failure
            done[1] = True
            timeout[1] = True           # env 1 times out
        done_terms = {k: torch.zeros(self.num_envs, dtype=torch.bool) for k in DONE_KEYS}
        done_terms["time_out"] = timeout
        done_terms["anchor_pos_bad"] = done & ~timeout
        reward_terms = {k: torch.randn(self.num_envs, generator=self._gen) * 0.05 for k in REWARD_KEYS}
        self._cur_obs = self._obs()
        self._cur_critic = self._critic()
        info = {"done_terms": done_terms, "reward_terms": reward_terms}
        if auto_reset and bool(timeout.any()):
            info["final_critic_observation"] = self._critic()
        return self._cur_obs, reward, done, info


def _cfg():
    return types.SimpleNamespace(
        action_dim=3, horizon=4, actor_hidden_dims=(32, 32), activation="elu",
        init_noise_std=1.0, action_squash_scale=5.0, basis_count=4,
        chunk_stitch_frames=0, chunk_stitch_mode="none",
        empirical_normalization=True, policy_lr=3e-4, weight_decay=1e-4,
        fpo_num_mc=6, flow_steps=8, fpo_delta_clip=0.0, fpo_cfm_loss_clamp=0.0,
        init_at_random_ep_len=True,
        rollout_env_steps=8, chunks_per_rollout=2, discount_gamma=0.99,
        num_mini_batches=2, num_learning_epochs=2, clip_range=0.01,
        value_loss_coef=1.0, max_grad_norm=1.0, gae_lambda=0.95,
    )


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  ok: {name}")


def test_end_to_end():
    torch.manual_seed(0)
    env = FakeEnv()
    algo = fpo.FPOPP(cfg=_cfg(), env=env, simulation_app=None)
    algo.build()
    check("chunk_dim == basis_count*action_dim", algo.chunk_dim == 4 * 3)
    check("chunks_per_rollout == rollout_env_steps//horizon", algo._chunks_per_rollout() == 2)

    obs = algo.initial_reset()
    check("initial_reset obs shape", obs.shape == (env.num_envs, env.observation_dim))

    obs = algo.reset_for_update(1)
    rollout = algo.collect(obs)
    T = algo._chunks_per_rollout()
    N = env.num_envs
    M = algo.num_mc
    D = algo.chunk_dim
    check("rollout latent shape", rollout["latent"].shape == (T, N, D))
    check("rollout eps shape", rollout["eps"].shape == (T, N, M, D))
    check("rollout tau shape", rollout["tau"].shape == (T, N, M))
    check("rollout old_cfm shape", rollout["old_cfm"].shape == (T, N, M))
    check("rollout returns shape", rollout["returns"].shape == (T, N, 1))
    check("rollout advantages shape", rollout["advantages"].shape == (T, N, 1))
    check("advantages finite", bool(torch.isfinite(rollout["advantages"]).all()))
    check("old_cfm finite & non-negative",
          bool(torch.isfinite(rollout["old_cfm"]).all()) and bool((rollout["old_cfm"] >= 0).all()))

    # Snapshot actor params; one update must change them.
    before = [p.detach().clone() for p in algo.actor.parameters()]
    metrics = algo.update(rollout, collect_time=0.01)
    after = list(algo.actor.parameters())
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    check("update changes actor params", changed)
    check("actor_loss finite", torch.isfinite(torch.tensor(metrics["fpo/actor_loss"])))
    check("value_loss finite & >=0", metrics["fpo/value_loss"] >= 0)
    check("ratio reported", "fpo/ratio" in metrics and metrics["fpo/ratio"] > 0)
    check("metrics has reward terms", any(k.startswith("reward/") for k in metrics))
    check("metrics has done fracs", any(k.startswith("done/") for k in metrics))

    # log() should not raise (uses stubbed shared logging).
    algo.log(1, 10, metrics)

    # Zero-sampling determinism: same obs -> identical greedy chunk; shape (N, H, A).
    probe = env.get_observation()
    a1 = algo.deterministic_actions(probe)
    a2 = algo.deterministic_actions(probe)
    check("deterministic_actions shape (N, H, A)", a1.shape == (N, algo.horizon, algo.num_act))
    check("zero-sampling is deterministic", torch.allclose(a1, a2, atol=1e-6))
    check("actions within squash bound", bool((a1.abs() <= algo.cfg.action_squash_scale + 1e-4).all()))


if __name__ == "__main__":
    print("[test_end_to_end]")
    test_end_to_end()
    print("\nFPO++ end-to-end rollout/update smoke test passed.")
