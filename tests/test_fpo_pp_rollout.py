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
        self.phase_steps = torch.zeros(num_envs, dtype=torch.long)
        # Executed residual r_{t-1}; the real env stores it in last_action (reset to 0 on done).
        self.last_action = torch.zeros(num_envs, action_dim)
        self.prev_action = torch.zeros(num_envs, action_dim)
        self.motion = types.SimpleNamespace(num_frames=100)
        self._gen = torch.Generator().manual_seed(123)
        self._step = 0
        self.sampler_calls = []

    def update_adaptive_motion_statistics(self, phases, failed, rollout_steps):
        self.sampler_calls.append((phases.clone(), failed.clone(), int(rollout_steps)))

    def _obs(self):
        return torch.randn(self.num_envs, self.observation_dim, generator=self._gen)

    def _critic(self):
        return torch.randn(self.num_envs, self.critic_observation_dim, generator=self._gen)

    def reset(self, **kw):
        self.episode_steps.zero_()
        self.last_action.zero_()
        self.prev_action.zero_()
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
        # Mirror the real env: the applied action is the executed residual r_t, stored as
        # last_action (and zeroed on auto-reset) so the AR(1) state is read back next step.
        self.prev_action = self.last_action.clone()
        self.last_action = action.clone()
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
        # env 0 dies via ee_body_bad (the crawl failure mode); record it as the death cause.
        done_terms["ee_body_bad"] = done & ~timeout
        if auto_reset and bool(done.any()):
            self.last_action[done] = 0.0
            self.prev_action[done] = 0.0
        reward_terms = {k: torch.randn(self.num_envs, generator=self._gen) * 0.05 for k in REWARD_KEYS}
        self.phase_steps = (self.phase_steps + 1) % self.motion.num_frames
        self._cur_obs = self._obs()
        self._cur_critic = self._critic()
        info = {
            "done_terms": done_terms, "reward_terms": reward_terms,
            "termination_phase_steps": self.phase_steps.clone(),
        }
        if auto_reset and bool(timeout.any()):
            info["final_critic_observation"] = self._critic()
        return self._cur_obs, reward, done, info


def _cfg(num_micro_batches=1):
    # Single-step FPO++ (official-aligned): horizon=1, flow acts directly in action space.
    return types.SimpleNamespace(
        action_dim=3, horizon=1, actor_hidden_dims=(32, 32), activation="elu",
        init_noise_std=1.0, action_squash_scale=5.0,
        actor_scale=1.0, mlp_output_scale=1.0, timestep_embed_dim=8,
        cfm_loss_reduction="mean", action_perturb_std=0.1, cfm_loss_t_inverse_cdf_beta=1.0,
        empirical_normalization=True, policy_lr=1e-4, weight_decay=1e-4,
        fpo_num_mc=6, flow_steps=4, fpo_delta_clip=3.0, fpo_cfm_loss_clamp=3.0,
        cfm_loss_clamp_neg_adv=True, cfm_loss_clamp_neg_adv_max=20.0, fpo_adv_clamp=5.0,
        schedule="adaptive", desired_kl=1e-4, trust_region_mode="aspo",
        num_micro_batches=num_micro_batches, storage_action_noise_std=0.0,
        residual_innov_rho=0.9, residual_innov_scale=0.25,
        init_at_random_ep_len=True,
        num_steps_per_env=6, discount_gamma=0.99, terminal_penalty=50.0,
        num_mini_batches=2, num_learning_epochs=2, clip_range=0.01,
        value_loss_coef=1.0, max_grad_norm=1.0, gae_lambda=0.95, value_lr=1.0e-3,
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
    check("chunk_dim == action_dim", algo.chunk_dim == 3)
    check("horizon forced to 1", algo.horizon == 1)
    check("num_steps_per_env", algo.num_steps_per_env == 6)

    obs = algo.initial_reset()
    check("initial_reset obs shape", obs.shape == (env.num_envs, env.observation_dim))

    obs = algo.reset_for_update(1)
    env._step = 0
    rollout = algo.collect(obs)
    T = algo.num_steps_per_env
    N = env.num_envs
    M = algo.num_mc
    D = algo.chunk_dim
    check("env.step called num_steps_per_env times", env._step == T)
    check("rollout actions shape (T, N, action_dim)", rollout["actions"].shape == (T, N, D))
    check("rollout residuals shape (T, N, action_dim)", rollout["residuals"].shape == (T, N, D))
    # Residual-innovation filter: r_t = rho*r_{t-1} + scale*u_t. At t=0 the env's last_action is 0
    # (reset), so r_0 == scale * u_0 (u_t = the stored innovation in "actions").
    check("residual filter: r_0 == scale * u_0 (r_prev=0 after reset)",
          torch.allclose(rollout["residuals"][0], 0.25 * rollout["actions"][0], atol=1e-6))
    # The executed residual is smaller than the raw innovation it was built from (scale<1, rho<1).
    check("executed residual magnitude < innovation magnitude",
          float(rollout["residuals"].abs().mean()) < float(rollout["actions"].abs().mean()))
    check("rollout cfm_eps shape", rollout["cfm_eps"].shape == (T, N, M, D))
    check("rollout cfm_t shape", rollout["cfm_t"].shape == (T, N, M, 1))
    check("rollout old_cfm shape", rollout["old_cfm"].shape == (T, N, M))
    check("rollout x1_pred shape (T, N, M, A)", rollout["x1_pred"].shape == (T, N, M, D))
    check("rollout returns shape", rollout["returns"].shape == (T, N, 1))
    check("rollout advantages shape", rollout["advantages"].shape == (T, N, 1))
    check("advantages finite", bool(torch.isfinite(rollout["advantages"]).all()))
    check("old_cfm finite & non-negative",
          bool(torch.isfinite(rollout["old_cfm"]).all()) and bool((rollout["old_cfm"] >= 0).all()))
    check("x1_pred finite", bool(torch.isfinite(rollout["x1_pred"]).all()))
    check("no chunk-only keys (latent/fail/lost_frames/raw_obs) in rollout",
          not any(k in rollout for k in ("latent", "fail", "lost_frames", "raw_obs")))

    # Survival objective: env 0 dies (non-timeout) every 5th step; its reward must be pushed far
    # below the env reward scale by -terminal_penalty, and the failure must be recorded.
    rewards = rollout["rewards"]
    check("terminal_penalty applied to failure rewards (min << env scale)",
          float(rewards.min().item()) < -10.0)
    check("first_done_step recorded for env 0 (failure)", int(rollout["first_done_step"][0].item()) < T)
    check("env 0 death recorded as ee_body_bad", bool(rollout["first_done_ee_body"][0].item()))
    check("env 0 not a timeout", not bool(rollout["first_done_timeout"][0].item()))
    check("adaptive motion sampler was called", len(env.sampler_calls) == 1)
    sp, sf, rs = env.sampler_calls[0]
    check("sampler shapes/rollout_steps", sp.shape == (N,) and sf.shape == (N,) and rs == T)
    check("sampler flags env 0 as failed", bool(sf[0].item()))

    # Snapshot actor params; one update must change them.
    before = [p.detach().clone() for p in algo.actor.parameters()]
    metrics = algo.update(rollout, collect_time=0.01)
    after = list(algo.actor.parameters())
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    check("update changes actor params", changed)
    check("actor_loss finite", torch.isfinite(torch.tensor(metrics["fpo/actor_loss"])))
    check("value_loss finite & >=0", metrics["fpo/value_loss"] >= 0)
    check("ratio reported", "fpo/ratio" in metrics and metrics["fpo/ratio"] > 0)
    check("kl reported & finite", "fpo/kl" in metrics and torch.isfinite(torch.tensor(metrics["fpo/kl"])))
    # Gradient decoupling: actor and critic clipped separately; critic LR is fixed (value_lr),
    # actor LR is the adaptive one (independent param groups).
    check("separate critic grad reported", "fpo/grad_norm_critic" in metrics)
    check("critic_lr == value_lr (decoupled, fixed)", abs(metrics["fpo/critic_lr"] - 1.0e-3) < 1e-12)
    group_names = {g.get("name") for g in algo.optimizer.param_groups}
    check("optimizer has separate actor/critic groups", {"actor", "critic"} <= group_names)
    critic_group = next(g for g in algo.optimizer.param_groups if g.get("name") == "critic")
    check("critic group LR stayed at value_lr", abs(critic_group["lr"] - 1.0e-3) < 1e-12)
    check("metrics has reward terms", any(k.startswith("reward/") for k in metrics))
    check("metrics has done fracs", any(k.startswith("done/") for k in metrics))
    check("first_failure metrics populated (ee_body_frac > 0)",
          metrics["rollout/first_failure_ee_body_frac"] > 0.0)
    check("failure_frac reported", "rollout/failure_frac" in metrics)

    # log() should not raise (uses stubbed shared logging).
    algo.log(1, 10, metrics)

    # Zero-sampling determinism: same obs -> identical greedy action; shape (N, 1, A).
    probe = env.get_observation()
    a1 = algo.deterministic_actions(probe)
    a2 = algo.deterministic_actions(probe)
    check("deterministic_actions shape (N, 1, A)", a1.shape == (N, 1, algo.num_act))
    check("zero-sampling is deterministic", torch.allclose(a1, a2, atol=1e-6))
    check("actions finite (no clipping in official Flow actor)", bool(torch.isfinite(a1).all()))


def test_microbatch_matches_single_batch():
    """Gradient-accumulation microbatching must produce the same update as one backward."""
    def _run(num_micro):
        torch.manual_seed(7)
        env = FakeEnv()
        algo = fpo.FPOPP(cfg=_cfg(num_micro_batches=num_micro), env=env, simulation_app=None)
        algo.build()
        obs = algo.initial_reset()
        obs = algo.reset_for_update(1)
        env._step = 0
        rollout = algo.collect(obs)
        algo.update(rollout, collect_time=0.0)
        return [p.detach().clone() for p in algo.actor.parameters()]

    single = _run(1)
    micro = _run(3)
    same = all(torch.allclose(a, b, atol=1e-5) for a, b in zip(single, micro))
    check("microbatched update == single-batch update", same)


if __name__ == "__main__":
    print("[test_end_to_end]")
    test_end_to_end()
    print("\n[test_microbatch_matches_single_batch]")
    test_microbatch_matches_single_batch()
    print("\nFPO++ end-to-end rollout/update smoke test passed.")
