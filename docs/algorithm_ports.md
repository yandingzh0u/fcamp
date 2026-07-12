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
| PolicyFlow | `../official_repos/policyflow` | `PolicyFlow2026/PolicyFlow` | `7f304b96e93b2804bdcc943710db885adddd06b8` |
| SAC Flow | `../official_repos/sac-flow` | `Elessar123/SAC-FLOW` | `3df8b9dc33de5160851a82322bc6c9005886f3ce` |
| SEAR | `../official_repos/sear-supplement` | arXiv `2603.01891` source bundle | SHA256 `4697ec577d424cbf17a5d6f8de012ece0dc6ca166a5c498c057187fc3c1137b4` |

The extra `../official_repos/flowrl-taxonomy` directory is not used by any
algorithm. It is a different project with the same name.

SEAR did not have a public upstream Git repository at porting time. Its local
directory is the official arXiv source bundle, including the method,
pseudocode, and hyperparameter supplement. No third-party SEAR implementation
is treated as upstream code.

## No-prior main comparison

The primary from-scratch table includes PPO, Chunk-PPO, Original FPO, FPO++,
FlowRL, PolicyFlow, SAC Flow, SEAR, and SFPO. ReinFlow and FQL remain available
for reproducibility but are excluded from this table: official ReinFlow starts
from a pretrained flow policy, while official FQL is an offline or
offline-to-online method. They belong in a separately labeled prior-data
experiment only when every compared method receives the same prior data.

## Common comparison budget

| Algorithm | Config | Action horizon | Physical frames/update | Flow steps |
| --- | --- | ---: | ---: | ---: |
| FPO++ | `configs/fpo_plus_plus.yaml` | 1 | 24 | 4 |
| FlowRL | `configs/flowrl.yaml` | 1 | 24 | 4 |
| ReinFlow | `configs/reinflow.yaml` | 4 | 24 | 4 |
| Original FPO | `configs/fpo.yaml` | 1 | 24 | 4 |
| FQL | `configs/fql.yaml` | 1 | 24 | 4 |
| PolicyFlow | `configs/policyflow.yaml` | 1 | 24 | 4 |
| SAC Flow | `configs/sac_flow.yaml` | 1 | 24 | 4 |
| SEAR | `configs/sear.yaml` | 4 | 24 | N/A |
| SFPO | `configs/sfpo.yaml` | 4 | 24 | 4 |

The actor and critic widths are `[512, 256, 128]` where the upstream
architecture permits direct alignment. On-policy algorithms use five epochs.
Online replay algorithms perform 24 gradient steps per update; with the default
replay batch of 8192, this consumes 196608 replay transitions, equal to the
196608 newly collected physical transitions.

SEAR samples four-frame replay sequences. It therefore performs six updates
per iteration rather than 24: `6 * 8192 * 4 = 196608` replay frames, exactly
matching the newly collected physical frames. Its logs expose sequence count,
frame count, policy decisions, random-prefix length, and frame-level UTD.

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

### PolicyFlow

`policyflow` preserves the official PPO-style likelihood approximation. A
frozen old flow snapshot and the current flow are evaluated at the same point
on the interpolation path; their velocity difference is the mean of the new
Gaussian perturbation distribution. The old rollout perturbation is evaluated
under that distribution to form one PPO ratio. The Brownian regularizer and
the separate Gaussian entropy term are both retained.

The comparison uses four midpoint ODE steps instead of the upstream default of
ten. The G1 coefficients are retained: Brownian weight `0.006`, Gaussian
entropy weight `0.002`, five epochs, and four mini-batches. Logs under
`policyflow/` expose the ratio, approximate KL, clip fraction, velocity drift,
Brownian loss, learned noise, and independent actor/critic learning rates.

### SAC Flow

`sac-flow` is the upstream from-scratch Flow-G variant. Its gated velocity
network is a GRU-style residual flow. Every one of the four flow transitions
injects state-conditioned Gaussian noise, and the actor uses the exact joint
path likelihood, including the base Gaussian and final tanh Jacobian, in the
SAC objective. The critic follows the upstream CrossQ joint-forward update with
Batch Renormalization, delayed actor updates, and learned entropy temperature.

The upstream from-scratch code explicitly sets target entropy to zero; this is
retained rather than retuned. The common vectorized budget uses 24 gradient
steps with batch 8192, giving frame-level UTD 1. Core logs use the
`sac_flow/` prefix and separate path standard deviation, gate activation,
latent magnitude, critic target, entropy temperature, replay, and budget data.

### SEAR

`sear` is a non-flow action-chunk control baseline. The actor samples a joint
`h=4` squashed-Gaussian chunk. During online collection each environment
executes a uniformly sampled prefix of length 1 through 4 before replanning.
Replay sampling is contiguous within each environment and rejects sequences
that cross an episode boundary.

Twin causal Transformer critics produce all four prefix values in parallel.
Each prefix receives its own discounted multi-horizon TD target, and both the
bootstrap entropy and physical rewards are discounted per frame. The critic is
distributional with 101 bins, two blocks, 16 heads, and source-fraction target
updates of `0.05`. The upstream MetaWorld support `[0, 1000]` is widened to
`[-100, 1000]` because this repository's tracking reward can be negative; the
clipped-target fraction is logged so this adaptation remains auditable.

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
| PolicyFlow | `runs/policyflow_official_smoke3_v2` |
| SAC Flow-G | `runs/sac_flow_g_smoke3_v2` |
| SEAR h=4 | `runs/sear_h4_smoke3` |

The three new smoke runs used 256 environments and retained frame-level UTD 1
for both off-policy algorithms. Their generated checkpoints were successfully
serialized and then removed to avoid adding hundreds of megabytes of temporary
weights to version control; the logs are retained. The CPU regression suite
covers objective aggregation, stochastic-chain likelihood, contiguous replay,
causal-prefix invariance, distributional projection, multi-horizon returns,
Euler/midpoint integration, gradient ownership, configuration constraints, and
fake-environment rollouts.
