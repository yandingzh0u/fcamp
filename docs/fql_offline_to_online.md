# FQL offline-to-online protocol

FQL consumes full transitions, not behavior-cloning pairs. Every dataset entry
contains normalized actor observation, normalized FQL action, reward, next
observation, and the timeout-aware bootstrap mask.

## Collect transitions

The collector can use any checkpoint whose config is registered in this
repository. For an `h=4` teacher it replans every physical frame and executes
the first action, producing the `h=1` transitions expected by FQL.

```bash
/home/y/miniconda3/envs/env_isaaclab/bin/python collect_offline_dataset.py \
  --config configs/sfpo.yaml \
  --checkpoint runs/sfpo_lowrank_cps_500/checkpoints/last.pt \
  --output data/fql/sfpo_teacher_norm5.pt \
  --num_frames 122 \
  --environment_action_scale 5.0 \
  --set environment.num_envs=8192
```

This example collects 999424 transitions. The collector stores
`u = clamp(a_teacher / 5, -1, 1)` and executes `a_env = 5u`, so the replay action
coordinates remain compatible with official FQL while the robot receives its
original action scale. The log reports any teacher elements that still require
clipping.

Using an SFPO checkpoint makes FQL a student of SFPO. Such a run validates the
protocol or studies distillation, but it is not a fair independent comparison.
For a fair offline comparison, use the same external demonstration dataset and
pretraining sample budget for every eligible method.

## Train offline-to-online

```bash
/home/y/miniconda3/envs/env_isaaclab/bin/python train.py \
  --config configs/fql.yaml \
  --run_name fql_offline_to_online \
  --set parameters.environment_action_scale=5.0 \
  --set parameters.offline_dataset_path=/absolute/path/to/data/fql/sfpo_teacher_norm5.pt \
  --set parameters.offline_pretrain_gradient_steps=122
```

With the default replay batch size 8192, 122 offline gradient steps consume
999424 sampled transitions, approximately one replay pass over the example
dataset. More epochs must be reported as additional offline sample budget.

At update 1, FQL performs all offline gradient steps before collecting the first
online frame. Afterward the same replay ring contains both sources. The core
logs expose:

- `budget/offline_seed_transitions`
- `budget/offline_pretrain_samples`
- `fql/replay_offline_transitions`
- `fql/replay_online_transitions`
- `fql/update_offline_sample_fraction`
- `fql_pretrain/*`

When `offline_dataset_path=""` and `offline_pretrain_gradient_steps=0`, FQL is
explicitly in online-replay-only mode and retains the 24-frame random warmup.
