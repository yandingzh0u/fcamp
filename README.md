# G1 Mimic Lab

Research code for G1 whole-body imitation with **FCAMP**: a causal Flow-CPS
chunk actor trained with a temporal discriminator prior, causal frame credit,
and shared dual Flow critics.

## Layout

```text
configs/      FCAMP experiment config.
method/       FCAMP training method.
models/       Neural network modules (Flow-CPS, critics, discriminator).
components/   Shared utilities: imitation features, replay, credit, rollout.
envs/         IsaacLab G1 mimic environment.
engine/       Training loop, checkpointing, metrics, validation.
tests/        Unit tests for config, credit, Flow-CPS, and imitation tools.
runs/         Training outputs. Only the baseline run is tracked in git.
```

## Train

```bash
/home/y/miniconda3/envs/env_isaaclab/bin/python train.py \
  --config configs/fcamp_largebox.yaml \
  --run_name fcamp_largebox_s0
```

## Smoke

```bash
/home/y/miniconda3/envs/env_isaaclab/bin/python train.py \
  --config configs/fcamp_largebox.yaml \
  --run_name smoke_fcamp_u3 \
  --set training.max_updates=3 \
  --set training.validation_every=0 \
  --set training.log_every=1 \
  --set training.save_every=0
```
