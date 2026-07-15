# G1 Mimic Lab

Research code for G1 whole-body imitation. The main method is FCAMP: a causal
Flow-CPS chunk actor trained with a temporal discriminator prior, causal frame
credit, and shared dual Flow critics. AdaMimic is also available as a native
comparison method.

## Layout

```text
configs/      Method experiment configs.
method/       Research methods. Add new algorithms here.
models/       Neural network modules.
components/   Method-neutral utilities: imitation features, replay, credit,
              normalization, optimization, and rollout internals.
envs/         IsaacLab G1 mimic environment.
engine/       Training loop, checkpointing, metrics, validation.
tests/        Fast unit tests for config, credit, Flow-CPS, and imitation tools.
runs/         Local-only training outputs; ignored by git.
```

## Train

Experiment configs carry the default run schedule. Normal runs should not need
manual `--set` overrides for update count, validation cadence, checkpoint
cadence, logging cadence, or environment count.

```bash
/home/y/miniconda3/envs/env_isaaclab/bin/python train.py \
  --config configs/fcamp_largebox.yaml \
  --run_name fcamp_largebox_s0
```

```bash
/home/y/miniconda3/envs/env_isaaclab/bin/python train.py \
  --config configs/adamimic_stage1_largebox.yaml \
  --run_name adamimic_stage1_largebox_s1
```

## Add A Method

Add one method file and, only if needed, one model file:

```text
method/beyondmimic.py
models/beyondmimic_policy.py
configs/beyondmimic_largebox.yaml
```

The method module must expose either `class Beyondmimic`/`class BeyondMimic`
matching the loader convention, or a `build_method(cfg, env, simulation_app)`
factory.

## Smoke

```bash
/home/y/miniconda3/envs/env_isaaclab/bin/python train.py \
  --config configs/fcamp_largebox.yaml \
  --run_name smoke_refactor_u3 \
  --set training.max_updates=3 \
  --set training.validation_every=0 \
  --set training.log_every=1 \
  --set training.save_every=0
```
