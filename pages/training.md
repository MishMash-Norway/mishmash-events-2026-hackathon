# Training

[Back to Repo README](../README.md) | Previous: [Processing](proc.md) | Next: [After Training](after_training.md)

This guide covers practical RAVE training commands and what each run type is for.

## Core Concepts

- Training is step-based (`--max_steps`), not epoch-based.
- Two phases:
  - Phase 1: representation learning (encoder/decoder foundation)
  - Phase 2: adversarial refinement (discriminator enabled)
- Realtime-safe path:
  - train with `--config causal`
  - export with `--streaming`

## Model Size Strategy

- `raspberry` (smallest)
- `v2_small` (`CAPACITY = 48`): faster baseline for smoke tests and controlled comparisons.
- `v2` (`CAPACITY = 96`): heavier model with higher quality potential on capable GPUs.

Recommended flow:

1. Validate the full pipeline with `v2_small`.
2. If quality is still limited, launch a fresh longer `v2` run.

Important:

- Resume only with the same `--model_config` family used by the checkpoint.
- Do not resume a `v2_small` checkpoint as `v2` (or the inverse).

## 1) Quick Smoke Test

Fast sanity run to verify setup:

```bash
python RAVE/train.py \
  --dataset "Emotions" \
  --name "emotions_test" \
  --model_config v2_small \
  --channels 1 \
  --gpu -1 \
  --test \
  --launch_tensorboard
```

## 2) Short 2-Phase Experiment

Useful for pipeline validation and early listening:

```bash
python RAVE/train.py \
  --dataset "Emotions" \
  --name "emotions_2phase_short" \
  --model_config v2_small \
  --channels 1 \
  --gpu 0 \
  --batch 2 \
  --workers 0 \
  --max_steps 12000 \
  --val_every 1000 \
  --override "SAMPLING_RATE = 48000" \
  --override "PHASE_1_DURATION = 3000" \
  --launch_tensorboard
```

## 3) Longer Ubuntu GPU Runs

### A) Controlled extension (`v2_small`, comparable with previous tests)

Use this when you want apples-to-apples comparison and step scaling on the same architecture:

```bash
python RAVE/train.py \
  --dataset "Emotions" \
  --name "emotions_longer_v2small" \
  --model_config v2_small \
  --channels 1 \
  --gpu 0 \
  --batch 2 \
  --workers 4 \
  --max_steps 240000 \
  --val_every 2000 \
  --override "PHASE_1_DURATION = 30000" \
  --override "model.BetaWarmupCallback.warmup_len = 30000" \
  --override "model.BetaWarmupCallback.target_value = 0.003" \
  --launch_tensorboard \
  --tensorboard_port 6007
```

Resume from an existing checkpoint (same config family):

```bash
python RAVE/train.py \
  --dataset "Emotions" \
  --name "emotions_longer_v2small_resume" \
  --model_config v2_small \
  --channels 1 \
  --gpu 0 \
  --batch 2 \
  --workers 4 \
  --max_steps 240000 \
  --val_every 2000 \
  --extra_args --ckpt "/path/to/version_0/checkpoints/last.ckpt"
```

### B) Higher-quality exploratory run (`v2`, new training line)

Use this when quality is the priority and your GPU budget allows a heavier model:

```bash
python RAVE/train.py \
  --dataset "Emotions" \
  --name "emotions_v2_long" \
  --model_config v2 \
  --channels 1 \
  --gpu 0 \
  --batch 1 \
  --workers 4 \
  --max_steps 240000 \
  --val_every 2000 \
  --override "PHASE_1_DURATION = 30000" \
  --override "model.BetaWarmupCallback.warmup_len = 30000" \
  --override "model.BetaWarmupCallback.target_value = 0.003" \
  --launch_tensorboard \
  --tensorboard_port 6007
```

## 4) TensorBoard

- Default launch URL: `http://127.0.0.1:6006`
- Use another port (for example `6007`) if an old process is still running.

Stop stale TensorBoard:

```bash
pkill -f tensorboard.main || true
```

## 5) If CUDA Becomes Unstable

Use a safer command profile:

- lower `--batch` (for example `2`)
- set `--workers 0`
- optionally debug once with:

```bash
CUDA_LAUNCH_BLOCKING=1 python RAVE/train.py ...
```

## 6) Experimental: Continue from Checkpoint on a New Dataset

Use this as an experimental transfer-style option when you want to adapt an existing model to a different dataset.

Important constraints:

- resume from `.ckpt`, not from exported `.ts`
- keep audio channels the same as the source model (`--channels 1` for mono models)
- keep sample-rate assumptions consistent with the source training/deployment pipeline
- prefer `last.ckpt`, otherwise newest `epoch-*.ckpt`, and use `best.ckpt` only as fallback
- set `--max_steps` higher than the checkpoint's current global step

Example:

```bash
python RAVE/train.py \
  --dataset "NewDataset" \
  --name "emotions_ft_newdata" \
  --model_config v2_small \
  --channels 1 \
  --gpu 0 \
  --batch 2 \
  --workers 4 \
  --val_every 2000 \
  --max_steps 320000 \
  --extra_args --ckpt "/path/to/source_run/version_0/checkpoints/last.ckpt"
```

Notes:

- This is continued training, not a guaranteed domain-adaptation recipe.
- If the new dataset is very different, monitor quality drift in TensorBoard and by regular listening tests.

## 7) Optional Advanced: Unconditional Generation with Prior

If you want autonomous latent generation (instead of only manual latent control), continue with:

- [Unconditional Generation with Prior (Advanced)](prior.md)

---

[Back to Repo README](../README.md) | Previous: [Processing](proc.md) | Next: [After Training](after_training.md)
