"""A/B/C latent-contract probe for a trained MixGRPO checkpoint.

All three conditions reset every env to the SAME phase-0 state, then roll the policy out
deterministically in the ENVIRONMENT, differing only in how each action chunk's latent is drawn:

  A: zero initial latent + zero SDE noise   (== the eval/play path used by validation)
  B: fixed random initial latent + zero SDE noise
  C: random initial latent + training-time SDE sampling

If B/C survive markedly longer than A, the train/deploy latent contract is inconsistent
(the policy learned random-latent behaviour but is played at the never-trained zero latent).
If all three are ~equal (and low), the bottleneck is phase-0 start coverage, not the latent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--start_phase", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from env.config import DEFAULT_MOTION_FILE, MimicEnvConfig
from env.mimic import G1MimicEnv
from net.mixgrpo.flow_policy import FlowMatchingPolicy
from engine.mixgrpo.sampling import flow_grpo_step


def _build_chunk(policy, obs, steps, sde_eta, initial_noise, sde_noise):
    """Generate one action chunk exactly like the trainer rollout, for given latents."""
    obs_prep = policy._prepare_observation(obs)
    sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=obs.device, dtype=obs.dtype)
    latent = initial_noise * float(policy.init_noise_std)
    for i in range(steps):
        t = torch.full((obs.shape[0],), float(sigma_schedule[i].item()), device=obs.device, dtype=obs.dtype)
        model_output = policy.velocity_field(obs_prep, latent, t)
        step_noise = sde_noise[:, i] if sde_noise is not None else torch.zeros_like(latent)
        latent, _ = flow_grpo_step(
            model_output=model_output, latents=latent, sigmas=sigma_schedule,
            index=i, eta=sde_eta, deterministic=False, sample_noise=step_noise,
        )
    a_dim = policy.action_dim
    actions = policy._action_transform(
        latent,
        start_action=obs[..., -2 * a_dim : -a_dim],
        start_prev_action=obs[..., -a_dim :],
    )
    return actions.view(obs.shape[0], policy.horizon, a_dim)


@torch.no_grad()
def run_condition(env, policy, cfg, *, name, max_steps, start_phase, mode, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    n = env.num_envs
    steps = int(cfg["flow_steps"])
    horizon = int(cfg["horizon"])
    sde_eta = float(cfg.get("sde_eta", 0.7))
    phases = torch.full((n,), max(0, start_phase), dtype=torch.long, device=env.device)
    obs = env.reset(phase_indices=phases)

    # Fixed random initial latent for condition B (same every chunk so it is a deterministic path).
    fixed_noise = torch.randn(n, policy.chunk_dim, device=env.device)

    done = torch.zeros(n, dtype=torch.bool, device=env.device)
    survived = torch.zeros(n, dtype=torch.long, device=env.device)
    done_ori = torch.zeros(n, dtype=torch.bool, device=env.device)
    cached = None
    cidx = horizon
    for _ in range(max_steps):
        if cidx >= horizon:
            if mode == "A":
                init = torch.zeros(n, policy.chunk_dim, device=env.device)
                sde = None
            elif mode == "B":
                init = fixed_noise
                sde = None
            else:  # C: random init + training SDE noise
                init = torch.randn(n, policy.chunk_dim, device=env.device)
                sde = torch.randn(n, steps, policy.chunk_dim, device=env.device)
            cached = _build_chunk(policy, obs, steps, sde_eta, init, sde)
            cidx = 0
        action = cached[:, cidx, :]
        if bool(done.any()):
            action = torch.where(done.unsqueeze(-1), torch.zeros_like(action), action)
        cidx += 1
        obs, _, step_done, info = env.step(action, auto_reset=False)
        active = ~done
        new_done = active & step_done
        if bool(new_done.any()):
            done_ori[new_done] = info["done_terms"]["anchor_ori_bad"][new_done].bool()
        survived += active.to(torch.long)
        done |= step_done
        if bool(done.all()):
            break
    s = survived.float()
    print(
        f"[{name}] steps_mean={s.mean().item():.2f} p50={s.median().item():.0f} "
        f"p95={torch.quantile(s, 0.95).item():.0f} max={s.max().item():.0f} "
        f"done_frac={done.float().mean().item():.3f} anchor_ori_frac={done_ori.float().mean().item():.3f}",
        flush=True,
    )


def main():
    device = torch.device(args_cli.device)
    payload = torch.load(Path(args_cli.checkpoint).expanduser().resolve(), map_location=device)
    cfg = payload["config"]
    env = G1MimicEnv(
        MimicEnvConfig(
            device=str(device),
            num_envs=args_cli.num_envs,
            sim_dt=cfg["sim_dt"],
            render=False,
            startup_randomization=False,
            motion_file=cfg.get("motion_file", str(DEFAULT_MOTION_FILE)),
            max_episode_steps=int(1.0e9),
            reset_noise=False,
            interval_pushes=False,
            observation_noise=False,
            motion_start_phase_ratio=0.0,
        )
    )
    policy = FlowMatchingPolicy(
        obs_dim=cfg.get("policy_obs_dim", 0) or env.observation_dim,
        action_dim=cfg["action_dim"],
        horizon=cfg["horizon"],
        hidden_dims=tuple(cfg.get("actor_hidden_dims", (512, 256, 128))),
        activation=cfg.get("activation", "elu"),
        action_squash_scale=float(cfg.get("action_squash_scale", 5.0)),
        basis_count=int(cfg.get("basis_count", 0)),
    ).to(env.device)
    policy.load_state_dict(payload["policy"])
    policy.eval()

    print(f"[INFO] checkpoint={args_cli.checkpoint} num_envs={env.num_envs} start_phase={args_cli.start_phase}", flush=True)
    run_condition(env, policy, cfg, name="A_zero_latent_zero_sde", max_steps=args_cli.max_steps,
                  start_phase=args_cli.start_phase, mode="A", seed=args_cli.seed)
    run_condition(env, policy, cfg, name="B_fixed_rand_latent_zero_sde", max_steps=args_cli.max_steps,
                  start_phase=args_cli.start_phase, mode="B", seed=args_cli.seed)
    run_condition(env, policy, cfg, name="C_rand_latent_train_sde", max_steps=args_cli.max_steps,
                  start_phase=args_cli.start_phase, mode="C", seed=args_cli.seed)
    simulation_app.close()


if __name__ == "__main__":
    main()
