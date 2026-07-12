from __future__ import annotations

import time

import torch
from torch import nn

from algorithms.fpo_plus_plus import FPOPlusPlus


def original_fpo_ratio(old_cfm: torch.Tensor, new_cfm: torch.Tensor) -> torch.Tensor:
    """Original FPO averages MC losses before exponentiating."""
    if old_cfm.shape != new_cfm.shape:
        raise ValueError("old_cfm and new_cfm must have the same shape")
    return torch.exp(old_cfm.mean(dim=-1, keepdim=True) - new_cfm.mean(dim=-1, keepdim=True))


def ppo_objective(ratio: torch.Tensor, advantage: torch.Tensor, clip: float) -> torch.Tensor:
    return torch.minimum(
        ratio * advantage,
        ratio.clamp(1.0 - clip, 1.0 + clip) * advantage,
    )


class OriginalFPO(FPOPlusPlus):
    """Original FPO: average MC CFM losses before exp, then standard PPO."""

    def update(self, rollout: dict, collect_time: float) -> dict:
        device = self.env.device
        timesteps, num_envs = rollout["actions"].shape[:2]
        batch_size = timesteps * num_envs
        num_mc = self.num_mc
        action_dim = self.num_act
        actor_obs = rollout["actor_obs"].reshape(batch_size, self.actor_obs_dim)
        critic_obs = rollout["critic_obs"].reshape(batch_size, self.critic_obs_dim)
        actions = rollout["actions"].reshape(batch_size, action_dim)
        cfm_t = rollout["cfm_t"].reshape(batch_size, num_mc, 1)
        cfm_eps = rollout["cfm_eps"].reshape(batch_size, num_mc, action_dim)
        old_cfm = rollout["old_cfm"].reshape(batch_size, num_mc)
        old_x1_pred = rollout["x1_pred"].reshape(batch_size, num_mc, action_dim)
        returns = rollout["returns"].reshape(batch_size, 1)
        advantages = rollout["advantages"].reshape(batch_size, 1)
        old_values = rollout["values"].reshape(batch_size, 1)

        num_mini_batches = int(self.cfg.num_mini_batches)
        mini_batch_size = max(1, batch_size // num_mini_batches)
        epochs = int(self.cfg.num_learning_epochs)
        clip = float(self.cfg.clip_range)
        value_clip = float(self.cfg.value_clip_range)
        use_clipped_value_loss = bool(self.cfg.use_clipped_value_loss)
        totals = {
            "actor_loss": 0.0,
            "value_loss": 0.0,
            "ratio": 0.0,
            "ratio_min": float("inf"),
            "ratio_max": 0.0,
            "clip_frac": 0.0,
            "cfm_new": 0.0,
            "cfm_old": 0.0,
            "grad_norm": 0.0,
            "grad_norm_critic": 0.0,
            "kl": 0.0,
            "ratio_mc_std": 0.0,
            "cfm_log_ratio_mean": 0.0,
            "cfm_log_ratio_std": 0.0,
            "positive_adv_frac": 0.0,
            "aspo_positive": 0.0,
            "aspo_negative": 0.0,
            "jensen_gap": 0.0,
            "mean_exp_ratio": 0.0,
        }
        optimizer_steps = 0
        probe_count = min(128, num_envs)
        with torch.no_grad():
            probe_obs = rollout["actor_obs"][0, :probe_count]
            probe_before = self.actor.act_inference(probe_obs, eval_mode="zero")
            parameters_before = [parameter.detach().clone() for parameter in self.actor.parameters()]

        update_start = time.perf_counter()
        for _ in range(epochs):
            permutation = torch.randperm(batch_size, device=device)
            for mini_batch in range(num_mini_batches):
                index = permutation[
                    mini_batch * mini_batch_size : (mini_batch + 1) * mini_batch_size
                ]
                current_size = index.numel()
                if current_size == 0:
                    continue
                advantage_full = advantages[index]
                if self.adv_clamp > 0.0:
                    advantage_full = advantage_full.clamp(-self.adv_clamp, self.adv_clamp)
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                aggregate = {key: 0.0 for key in totals if key not in ("ratio_min", "ratio_max", "grad_norm", "grad_norm_critic")}
                aggregate_ratio_min = float("inf")
                aggregate_ratio_max = 0.0
                micro_batches = torch.chunk(
                    torch.arange(current_size, device=device), self.num_micro_batches
                )
                for micro in micro_batches:
                    if micro.numel() == 0:
                        continue
                    weight = float(micro.numel()) / current_size
                    local_index = index[micro]
                    new_cfm, x1_pred, _ = self.actor.get_cfm_loss(
                        actor_obs[local_index],
                        actions[local_index],
                        cfm_eps[local_index],
                        cfm_t[local_index],
                    )
                    old_cfm_batch = old_cfm[local_index]
                    log_ratio = old_cfm_batch.mean(dim=1, keepdim=True) - new_cfm.mean(
                        dim=1, keepdim=True
                    )
                    ratio = original_fpo_ratio(old_cfm_batch, new_cfm)
                    advantage = advantage_full[micro]
                    surrogate = ppo_objective(ratio, advantage, clip)
                    actor_loss = -surrogate.mean()
                    value = self.critic.evaluate(critic_obs[local_index])
                    if use_clipped_value_loss:
                        value_clipped = old_values[local_index] + (
                            value - old_values[local_index]
                        ).clamp(-value_clip, value_clip)
                        value_loss = torch.maximum(
                            (value - returns[local_index]).pow(2),
                            (value_clipped - returns[local_index]).pow(2),
                        ).mean()
                    else:
                        value_loss = (value - returns[local_index]).pow(2).mean()
                    loss = actor_loss + float(self.cfg.value_loss_coef) * value_loss
                    (loss * weight).backward()

                    with torch.no_grad():
                        per_mc_log_ratio = (old_cfm_batch - new_cfm).clamp(-20.0, 20.0)
                        per_mc_ratio = torch.exp(per_mc_log_ratio)
                        mean_exp_ratio = per_mc_ratio.mean(dim=1, keepdim=True)
                        positive = (advantage >= 0.0).expand_as(surrogate)
                        negative = ~positive
                        aggregate["actor_loss"] += float(actor_loss.item()) * weight
                        aggregate["value_loss"] += float(value_loss.item()) * weight
                        aggregate["ratio"] += float(ratio.mean().item()) * weight
                        aggregate["clip_frac"] += float(
                            ((ratio - 1.0).abs() > clip).float().mean().item()
                        ) * weight
                        aggregate["cfm_new"] += float(new_cfm.mean().item()) * weight
                        aggregate["cfm_old"] += float(old_cfm_batch.mean().item()) * weight
                        aggregate["kl"] += float(
                            (x1_pred.detach() - old_x1_pred[local_index]).pow(2).mean().item()
                        ) * weight
                        aggregate["ratio_mc_std"] += float(
                            per_mc_ratio.std(dim=1, unbiased=False).mean().item()
                        ) * weight
                        aggregate["cfm_log_ratio_mean"] += float(log_ratio.mean().item()) * weight
                        aggregate["cfm_log_ratio_std"] += float(
                            log_ratio.std(unbiased=False).item()
                        ) * weight
                        aggregate["positive_adv_frac"] += float(
                            positive.float().mean().item()
                        ) * weight
                        if bool(positive.any()):
                            aggregate["aspo_positive"] += float(
                                surrogate[positive].mean().item()
                            ) * weight
                        if bool(negative.any()):
                            aggregate["aspo_negative"] += float(
                                surrogate[negative].mean().item()
                            ) * weight
                        aggregate["mean_exp_ratio"] += float(mean_exp_ratio.mean().item()) * weight
                        aggregate["jensen_gap"] += float(
                            (mean_exp_ratio - ratio).mean().item()
                        ) * weight
                        aggregate_ratio_min = min(aggregate_ratio_min, float(ratio.min().item()))
                        aggregate_ratio_max = max(aggregate_ratio_max, float(ratio.max().item()))

                actor_grad = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), float(self.cfg.max_grad_norm)
                )
                critic_grad = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), float(self.cfg.max_grad_norm)
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                for key in aggregate:
                    totals[key] += aggregate[key]
                totals["ratio_min"] = min(totals["ratio_min"], aggregate_ratio_min)
                totals["ratio_max"] = max(totals["ratio_max"], aggregate_ratio_max)
                totals["grad_norm"] += float(actor_grad.item())
                totals["grad_norm_critic"] += float(critic_grad.item())
                optimizer_steps += 1

        update_time = time.perf_counter() - update_start
        denominator = max(optimizer_steps, 1)
        with torch.no_grad():
            probe_after = self.actor.act_inference(probe_obs, eval_mode="zero")
            action_delta = float((probe_after - probe_before).abs().mean().item())
            squared_delta = torch.zeros((), device=device)
            parameter_count = 0
            for parameter, before in zip(
                self.actor.parameters(), parameters_before, strict=True
            ):
                difference = parameter.detach() - before
                squared_delta += difference.pow(2).sum()
                parameter_count += difference.numel()
            parameter_rms_delta = float(
                torch.sqrt(squared_delta / max(parameter_count, 1)).item()
            )
        aggregate_output = {
            key: value / denominator
            for key, value in totals.items()
            if key not in ("ratio_min", "ratio_max")
        }
        aggregate_output.update(
            {
                "ratio_min": totals["ratio_min"]
                if totals["ratio_min"] != float("inf")
                else 0.0,
                "ratio_max": totals["ratio_max"],
                "action_delta": action_delta,
                "param_rms_delta": parameter_rms_delta,
                "mini_batch_size": float(mini_batch_size),
            }
        )
        metrics = super()._build_metrics(rollout, aggregate_output, collect_time, update_time)
        renamed = {
            (key.replace("fpo_pp/", "fpo/") if key.startswith("fpo_pp/") else key): value
            for key, value in metrics.items()
        }
        renamed["fpo/jensen_gap"] = totals["jensen_gap"] / denominator
        renamed["fpo/mean_exp_ratio"] = totals["mean_exp_ratio"] / denominator
        renamed["fpo/ratio_aggregation"] = 1.0
        renamed.pop("fpo/aspo_positive", None)
        renamed.pop("fpo/aspo_negative", None)
        return renamed

    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        print(
            f"[UPDATE] {update_idx}/{max_updates} "
            f"reward_step={metrics['rollout/reward_step_mean']:.5f} "
            f"done_frac={metrics['rollout/done_frac']:.5f} "
            f"mean_reward={metrics.get('train/mean_reward', float('nan')):.5f} "
            f"mean_len={metrics.get('train/mean_episode_length', float('nan')):.2f}",
            flush=True,
        )
        print(
            f"[FPO] policy={metrics['fpo/actor_loss']:.5f} "
            f"value={metrics['fpo/value_loss']:.5f} "
            f"ratio={metrics['fpo/ratio']:.4f} "
            f"[{metrics['fpo/ratio_min']:.3f},{metrics['fpo/ratio_max']:.3f}] "
            f"clip={metrics['fpo/clip_frac']:.4f} "
            f"kl_x1={metrics['fpo/kl_x1_mse']:.6f} "
            f"actor_lr={metrics['fpo/actor_lr']:.6f} "
            f"critic_lr={metrics['fpo/critic_lr']:.6f}",
            flush=True,
        )
        print(
            f"[FPO_CORE] aggregation=mean_loss_before_exp "
            f"log_ratio={metrics['fpo/cfm_log_ratio_mean']:.6f}+/-"
            f"{metrics['fpo/cfm_log_ratio_std']:.6f} "
            f"exp_mean_loss={metrics['fpo/ratio']:.6f} "
            f"mean_exp_loss={metrics['fpo/mean_exp_ratio']:.6f} "
            f"jensen_gap={metrics['fpo/jensen_gap']:.6f} "
            f"mc_ratio_std={metrics['fpo/ratio_mc_std']:.6f}",
            flush=True,
        )
        print(
            f"[BUDGET] physical_transitions={metrics['budget/physical_transitions']:.0f} "
            f"policy_decisions={metrics['budget/policy_decisions']:.0f} "
            f"cfm_mc_terms={metrics['budget/cfm_mc_terms']:.0f}",
            flush=True,
        )
        print(
            f"[TIME] collect={metrics['timing/collect_s']:.3f}s "
            f"update={metrics['timing/update_s']:.3f}s",
            flush=True,
        )

    def log_banner(self) -> None:
        cfg = self.cfg
        env = self.env
        print("[INFO] Starting original FPO training", flush=True)
        print(
            f"[INFO] task={env.task.name} terrain={env.task.terrain} motion_file={env.task.motion_file}",
            flush=True,
        )
        print(
            f"[INFO] algorithm=fpo actor_obs_dim={self.actor_obs_dim} "
            f"critic_obs_dim={self.critic_obs_dim} action_dim={self.num_act} horizon=1 "
            f"num_envs={env.num_envs} num_steps_per_env={self.num_steps_per_env} "
            f"flow_steps={self.flow_steps} num_mc={self.num_mc} "
            f"ratio=exp(mean_old_cfm-mean_new_cfm) objective=ppo "
            f"clip={cfg.clip_range} schedule={self.schedule} "
            f"num_learning_epochs={cfg.num_learning_epochs} "
            f"num_mini_batches={cfg.num_mini_batches} "
            f"actor_lr={cfg.policy_lr} critic_lr={self.critic_learning_rate} "
            f"actor_hidden_dims={list(cfg.actor_hidden_dims)} "
            f"critic_hidden_dims={list(cfg.critic_hidden_dims)}",
            flush=True,
        )
