from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_fpo_module():
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

    spec = importlib.util.spec_from_file_location(
        "fpo_plus_plus_standalone", REPO_ROOT / "algorithms" / "fpo_plus_plus.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fpo = _load_fpo_module()

DONE_KEYS = ("time_out", "anchor_pos_bad", "anchor_ori_bad", "ee_body_bad")
REWARD_KEYS = ("track", "energy")


class FakeEnv:


    def __init__(self, num_envs=8, obs_dim=20, critic_dim=24, action_dim=3):
        self.num_envs = num_envs
        self.observation_dim = obs_dim
        self.critic_observation_dim = critic_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu")
        self.max_episode_steps = 20
        self.dt = 0.02
        self.config = types.SimpleNamespace(action_rate_weight=0.1)
        self.episode_steps = torch.zeros(num_envs, dtype=torch.long)
        self.phase_steps = torch.zeros(num_envs, dtype=torch.long)
        self.motion = types.SimpleNamespace(num_frames=100)
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

    def adaptive_sampling_stats(self):
        return {}

    def step(self, action, auto_reset=True, reset_horizon=1):
        self._step += 1
        assert action.shape == (self.num_envs, self.action_dim)
        reward = torch.randn(self.num_envs, generator=self._gen) * 0.1

        done = torch.zeros(self.num_envs, dtype=torch.bool)
        timeout = torch.zeros(self.num_envs, dtype=torch.bool)
        if self._step % 5 == 0:
            done[0] = True
            done[1] = True
            timeout[1] = True
        done_terms = {k: torch.zeros(self.num_envs, dtype=torch.bool) for k in DONE_KEYS}
        done_terms["time_out"] = timeout

        done_terms["ee_body_bad"] = done & ~timeout
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

    return types.SimpleNamespace(
        action_dim=3, horizon=1, actor_hidden_dims=(32, 32), critic_hidden_dims=(32, 32), activation="elu",
        actor_scale=1.0, mlp_output_scale=1.0, timestep_embed_dim=8,
        cfm_loss_reduction="mean", action_perturb_std=0.1, cfm_loss_t_inverse_cdf_beta=1.0,
        empirical_normalization=True, policy_lr=1e-4, weight_decay=1e-4,
        fpo_num_mc=6, flow_steps=4, fpo_delta_clip=3.0, fpo_cfm_loss_clamp=3.0,
        cfm_loss_clamp_neg_adv=True, cfm_loss_clamp_neg_adv_max=20.0, fpo_adv_clamp=5.0,
        schedule="adaptive", desired_kl=1e-4,
        num_micro_batches=num_micro_batches,
        init_at_random_ep_len=True,
        num_steps_per_env=6, discount_gamma=0.99,
        num_mini_batches=2, num_learning_epochs=2, clip_range=0.01, value_clip_range=0.2,
        use_clipped_value_loss=False,
        value_loss_coef=1.0, max_grad_norm=1.0, gae_lambda=0.95, value_lr=1.0e-3,
        critic_weight_decay=0.0,
    )


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  ok: {name}")


def test_end_to_end():
    torch.manual_seed(0)
    env = FakeEnv()
    algo = fpo.FPOPlusPlus(cfg=_cfg(), env=env, simulation_app=None)
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


    rewards = rollout["rewards"]
    check("rollout rewards finite", bool(torch.isfinite(rewards).all()))
    check("first_done_step recorded for env 0 (failure)", int(rollout["first_done_step"][0].item()) < T)
    check("env 0 death recorded as ee_body_bad", bool(rollout["first_done_ee_body"][0].item()))
    check("env 0 not a timeout", not bool(rollout["first_done_timeout"][0].item()))


    before = [p.detach().clone() for p in algo.actor.parameters()]
    metrics = algo.update(rollout, collect_time=0.01)
    after = list(algo.actor.parameters())
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    check("update changes actor params", changed)
    check("actor_loss finite", torch.isfinite(torch.tensor(metrics["fpo_pp/actor_loss"])))
    check("value_loss finite & >=0", metrics["fpo_pp/value_loss"] >= 0)
    check("ratio reported", "fpo_pp/ratio" in metrics and metrics["fpo_pp/ratio"] > 0)
    check(
        "kl reported & finite",
        "fpo_pp/kl_x1_mse" in metrics
        and torch.isfinite(torch.tensor(metrics["fpo_pp/kl_x1_mse"])),
    )


    check("separate critic grad reported", "fpo_pp/grad_norm_critic" in metrics)
    check("actor optimizer is the primary (checkpointed) optimizer", algo.optimizer is algo.actor_optimizer)
    check("actor and critic optimizers are independent objects", algo.actor_optimizer is not algo.critic_optimizer)
    critic_lr = algo.critic_optimizer.param_groups[0]["lr"]
    actor_lr = algo.actor_optimizer.param_groups[0]["lr"]
    check("reported critic LR matches optimizer", abs(metrics["fpo_pp/critic_lr"] - critic_lr) < 1e-12)
    check("adaptive scheduler leaves critic LR fixed", abs(critic_lr - 1.0e-3) < 1e-12)
    check("actor LR remains independently scheduled", actor_lr != critic_lr)
    check("FPO++ core diagnostics reported", "fpo_pp/ratio_mc_std" in metrics)
    check("physical transition budget reported", metrics["budget/physical_transitions"] == 48.0)
    check("metrics has reward terms", any(k.startswith("reward/") for k in metrics))
    check("metrics has done fracs", any(k.startswith("done/") for k in metrics))
    check("first_failure metrics populated (ee_body_frac > 0)",
          metrics["rollout/first_failure_ee_body_frac"] > 0.0)
    check("failure_frac reported", "rollout/failure_frac" in metrics)


    algo.log(1, 10, metrics)


    probe = env.get_observation()
    a1 = algo.deterministic_actions(probe)
    a2 = algo.deterministic_actions(probe)
    check("deterministic_actions shape (N, 1, A)", a1.shape == (N, 1, algo.num_act))
    check("zero-sampling is deterministic", torch.allclose(a1, a2, atol=1e-6))
    check("actions finite (no clipping in official Flow actor)", bool(torch.isfinite(a1).all()))


def test_microbatch_matches_single_batch():

    def _run(num_micro):
        torch.manual_seed(7)
        env = FakeEnv()
        algo = fpo.FPOPlusPlus(cfg=_cfg(num_micro_batches=num_micro), env=env, simulation_app=None)
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
