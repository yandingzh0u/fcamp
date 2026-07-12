# Flow-policy algorithm ports

This repository keeps each imported method as a separate algorithm. Shared
settings are aligned for comparison, while each method retains its defining
objective and policy representation.

## Pinned source repositories

| Algorithm | Local source copy | Upstream | Commit |
| --- | --- | --- | --- |
| FPO++ | `../official_repos/fpo-plus-plus` | `amazon-far/fpo-control` | `b80112be1e8362263c4cd176e7aef21a275ff1c6` |
| FlowRL | `../official_repos/flowrl` | `bytedance/FlowRL` | `e26279b331a7fb08668dc39d925ce2e9c82b2321` |
| ReinFlow | `../official_repos/reinflow` | `ReinFlow/ReinFlow` | `e722e151bed767f3ffef47527cf697f2358af55d` |
| Original FPO | `../official_repos/fpo-original` | `akanazawa/fpo` | `418c2554f7cd22d52e14c07d951280929d73bf2f` |
| FQL | `../official_repos/fql` | `seohongpark/fql` | `e8cd16eb490332924dfa2492097219f181765933` |

The extra `../official_repos/flowrl-taxonomy` directory is not used by any
algorithm. It is a different project with the same name.

## Common comparison budget

| Algorithm | Config | Action horizon | Physical frames/update | Flow steps |
| --- | --- | ---: | ---: | ---: |
| FPO++ | `configs/fpo_plus_plus.yaml` | 1 | 24 | 4 |
| FlowRL | `configs/flowrl.yaml` | 1 | 24 | 4 |
| ReinFlow | `configs/reinflow.yaml` | 4 | 24 | 4 |
| Original FPO | `configs/fpo.yaml` | 1 | 24 | 4 |
| FQL | `configs/fql.yaml` | 1 | 24 | 4 |

The actor and critic widths are `[512, 256, 128]` where the upstream
architecture permits direct alignment. On-policy algorithms use five epochs.
Online replay algorithms perform 24 gradient steps per update; with the default
replay batch of 8192, this consumes 196608 replay transitions, equal to the
196608 newly collected physical transitions.

ReinFlow intentionally has six policy decisions per update because a genuine
`h=4` action chunk executes four open-loop frames. Its physical interaction
budget remains 24 frames, but its on-policy batch contains `6 * num_envs`
chunks. The budget logs expose both physical transitions and policy decisions.

## Preserved algorithm logic

### FPO++

`fpo++` computes one CFM loss ratio for each Monte Carlo sample and applies
ASPO: PPO for positive advantages and SPO for negative advantages. It does not
average CFM losses before exponentiation. Actor and critic learning rates are
independent. Official EMA is evaluation machinery rather than part of the
training objective and is not used here.

Core logs use the `fpo_pp/` prefix and include positive/negative ASPO terms,
per-MC ratio dispersion, CFM log-ratio statistics, parameter counts, and sample
budgets.

### Original FPO

`fpo` uses the original state ratio

`exp(mean(old CFM losses) - mean(new CFM losses))`

followed by the standard clipped PPO objective. It shares the FPO++ network,
rollout, MC count, and optimizer budget so the ratio aggregation and ASPO are
the controlled differences. Logs under `fpo/` include both `exp(mean)` and
`mean(exp)` plus their Jensen gap.

### FlowRL

`flowrl` preserves the official online twin-Q critic, target twin Q, behavior
twin Q, target behavior Q, expectile value function, midpoint flow actor, and
delayed W2-regularized actor update. The target update uses the upstream
convention `target = 0.05 * target + 0.95 * online`. Replay sampling is 80%
uniform and 20% from the latest 2048 transitions.

The upstream code defaults to one integration step; this comparison uses four
as requested. The vectorized port uses 24 random warmup frames instead of the
upstream 5000 scalar-environment steps. All critic, value, actor, replay, and
budget metrics use the `flowrl/` prefix or the shared `budget/` prefix.

### ReinFlow

`reinflow` is a true `h=4` stochastic flow Markov chain. It accounts for the
initial Gaussian density and all four Gaussian transition densities, then
normalizes log likelihood over the five stochastic states and all 116 action
coordinates. PPO acts on one joint chunk ratio. Rewards and terminal masks are
collected per physical frame; discounted four-frame rewards and physical-time
GAE produce the chunk-start advantage.

The official method is designed to fine-tune a pretrained flow policy. An empty
`pretrained_actor_path` deliberately runs from scratch and is printed in the
banner; results from scratch should not be presented as the official pretrained
ReinFlow protocol. Logs under `reinflow/` separate chain likelihood, learned
noise, actor, critic, and budget statistics.

### FQL

`fql` preserves the official three-part update:

1. TD learning for a twin Q critic and soft target critic.
2. Conditional flow matching on replay actions for the behavior flow.
3. A one-step actor trained by four-step flow distillation plus Q maximization.

The one-step actor, not the four-step behavior flow, interacts with the
environment. Distillation targets are stop-gradient. The target critic uses a
stochastic one-step next action. `normalize_q_loss=true` follows the upstream
recommendation for new environments; `alpha=10` keeps the upstream default.

FQL now supports both official-style offline-to-online training and an explicit
online-only fallback. `offline_dataset_path` seeds the replay buffer before any
environment interaction, and `offline_pretrain_gradient_steps` runs the same
joint FQL update on that data before online collection. Online random warmup is
disabled only after offline pretraining actually completes.

The dataset protocol stores normalized FQL actions in `[-1, 1]` and separately
records `environment_action_scale`; online execution applies the same scale.
Mismatched scales and out-of-range actions are rejected. Replay logs report the
offline/online population, sampled offline fraction, seed transition count, and
offline pretraining sample budget. See `docs/fql_offline_to_online.md` for the
collector and training commands.

## Smoke verification

Each port completed a three-update Isaac smoke and saved a checkpoint:

| Algorithm | Run directory |
| --- | --- |
| FPO++ | `runs/fpo_plus_plus_smoke3` |
| FlowRL | `runs/flowrl_smoke3` |
| ReinFlow | `runs/reinflow_h4_smoke3` |
| Original FPO | `runs/fpo_original_smoke3` |
| FQL | `runs/fql_smoke3` |
| FQL offline-to-online protocol | `runs/fql_offline_to_online_smoke3_v2` |

The CPU regression suite covers objective aggregation, stochastic-chain
likelihood, replay updates, Euler/midpoint integration, gradient ownership,
configuration constraints, and fake-environment rollouts.
