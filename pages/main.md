# Train a new RAVE model

[Back to Repo README](../README.md)

This menu is dedicated to the optional custom-model path.

Prerequisites (recommended first):

1. [Installation and Environment Check](install.md)
2. [BioPoint Sensor Setup](biopoint.md)
3. [Import and Control Pretrained Models](import_max_pd.md)

## Step 1: Data Preprocessing

- [Processing Audio Folders into RAVE Datasets](proc.md)

## Step 2: Training

- [Training Runs: Smoke Tests, 2-Phase Tests, and Long Runs](training.md)
- [EXTRA: Continue from a Checkpoint on a New Dataset](training.md#6-experimental-continue-from-checkpoint-on-a-new-dataset)

## Step 3: Export

- [After Training: Outputs and `.ts` Export](after_training.md)
- [EXTRA: Unconditional Generation with a Prior (Advanced)](prior.md)

## Step 4: Return to Import

- [Import into Max/Pd (`nn~`)](import_max_pd.md)

## NOTES

- Most important knobs to tune first:
  - model size (`CAPACITY`)
  - beta target (`model.BetaWarmupCallback.target_value`)
  - speed augmentation (`--rand_pitch`)
  - gain augmentation / input gain strategy

- Model size determines both quality and runtime cost. A practical approach is to start training, export early, and test on target hardware before committing to long runs.

- Practical model-size workflow:
  - use `v2_small` first for smoke tests and apples-to-apples continuation
  - move to a fresh `v2` run when quality is the priority and GPU budget allows
  - resume only within the same config family (`v2_small` checkpoint -> `v2_small`, `v2` checkpoint -> `v2`)

- Beta controls compression pressure:
  - higher beta -> fewer active latent dimensions, often stronger timbre-transfer behavior
  - lower beta -> often better reconstruction/direct synthesis behavior

- Latent dimensions are mostly organized during phase 1. Exporting with more latents than meaningful dimensions (for example above what very high-fidelity curves imply) can add noisy controls.

- RAVE has 2 phases. In phase 2, GAN objectives are added and validation loss can rise; this is normal. Because of that, `best` can be misleading and `last` is often the better practical checkpoint.

- Do not trust graphs alone. Listen to reconstructions regularly (including early exports) and decide whether to continue or restart.

- Realtime rule: train with `--config causal` and export with `--streaming`. Missing either can produce high-latency or broken `nn~` behavior.

- Sample-rate alignment is critical (`nn~` does not resample). Running a 44.1k model at 48k shifts pitch/speed.

- You can also check:
  - [Victor Shepardson's fork](https://github.com/victor-shepardson/RAVE)
  - [Fran Caspe’s BRAVE/minifusion project](https://minifusion.live/)
  - ["`nn~`" for Supercollider](https://github.com/elgiano/nn.ar)
