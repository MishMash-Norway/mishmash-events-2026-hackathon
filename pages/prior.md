# Unconditional Generation with Prior (Advanced)

[Back to Training Menu](main.md) | Previous: [Training](training.md) | Next: [Import into Max/Pd](import_max_pd.md)

## What the Prior Is

RAVE can be split into two model layers:

- a trained VAE (`encoder + decoder`) that maps audio <-> latent space
- an optional prior model that learns to generate latent trajectories over time

At generation time:

1. the prior proposes latent sequences (`z`)
2. the decoder renders those latents as audio

So this is not "decoder-only training." It is a separate latent-generator training procedure that drives the same decoder.

## When to Use It

Use prior training when you want:

- autonomous or semi-autonomous sound generation (without relying on external input audio)
- latent motion that stays closer to dataset statistics
- an additional creative layer beyond manual latent control

For most hackathon teams, this is optional and more complex than the baseline `nn~` control workflow.

## Prerequisites

- a trained RAVE checkpoint (`last.ckpt` preferred)
- a preprocessed dataset path for prior training
- GPU strongly recommended

## 1) Train the Prior

Use the RAVE CLI directly (advanced path):

```bash
rave train_prior \
  --model "training_runs/<vae_run>/version_0/checkpoints/last.ckpt" \
  --db_path "data/Emotions_norm_dataset" \
  --out_path "training_runs" \
  --name "emotions_prior" \
  --batch 8 \
  --gpu 0 \
  --max_steps 200000 \
  --val_every 5000
```

Notes:

- The prior learns from latents produced by the pretrained VAE given by `--model`.
- Keep dataset/channel/sample-rate assumptions compatible with your VAE pipeline.

## 2) Export VAE + Prior Together

For prior-enabled export, use `rave export` with `--prior`:

```bash
rave export \
  --run "training_runs/<vae_run>/version_0/checkpoints/last.ckpt" \
  --prior "training_runs/<prior_run>/version_0/checkpoints/last.ckpt" \
  --streaming \
  --name "emotions_with_prior"
```

This produces a `.ts` model that includes prior-driven generation behavior.

## 3) Practical Caveats

- Prior training is slower and usually needs more tuning than baseline VAE training.
- It is easy to overfit on small datasets.
- Keep regular A/B listening checks against your baseline non-prior export.

---

[Back to Training Menu](main.md) | Previous: [Training](training.md) | Next: [Import into Max/Pd](import_max_pd.md)
