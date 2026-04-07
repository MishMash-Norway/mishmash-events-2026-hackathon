# Import into Max/Pd (`nn~`)

[Back to Repo README](../README.md) | Previous: [BioPoint Setup](biopoint.md) | Next: [Processing (Optional)](proc.md)

This guide is the first practical sound stage after setup: load and control pretrained `.ts` models in Max/Pd.

## 1) Start with Pretrained Models

Before training your own RAVE model, use pretrained models to:

- validate your Max/Pd + `nn~` pipeline
- test BioPoint-to-latent mappings quickly
- explore control behavior and sonic range

## 2) Pretrained Models

- [Intelligent Instruments Lab RAVE models (Hugging Face)](https://huggingface.co/Intelligent-Instruments-Lab/rave-models)
- [IRCAM RAVE models download page](https://acids-ircam.github.io/rave_models_download)

## 3) Read Export Naming Quickly

Example:

- `voice_vocalset_b2048_r48000_z16.ts`

Interpretation:

- `b512`: latent block size
- `r48000`: model sample rate
- `z16`: 16 controllable latent channels

## 4) Always Check Sidecar Metadata

Open matching `*.nn_info.json` and confirm:

- `latent_control_channels`
- `sr`
- `latent_block_size`
- `encode_shape` / `decode_shape`

Use `latent_control_channels` as the number of latent controls exposed in `nn~`.

## 5) Fidelity vs Latent Controls

- Higher fidelity usually increases `z` (more controls)
- Lower fidelity usually reduces `z` (simpler control space)

To test:

```bash
python RAVE/export_nn.py \
  --run_path "training_runs/cagri-ML-666/emotions_2phase_safe_fd6661f1c8" \
  --streaming \
  --base_name "emotions_hi_fid" \
  --fidelity 0.999
```

## 6) Sample-Rate Rule

Match host sample rate to model sample rate.

If model is `r48000`, run Max/Pd audio at 48000 when possible.

## 7) Practical Latent Ranges

- default latent value: `0.0`
- test range: `-3.0` to `3.0`
- smooth value transitions for better stability

## 8) Optional Next Step: Train Your Own Model

If you want custom model behavior:

- [Processing Audio Folders into RAVE Datasets](proc.md)
- [Training Runs](training.md)
- [After Training and Export](after_training.md)

Then return to this page and compare pretrained vs custom exports.

---

[Back to Repo README](../README.md) | Previous: [BioPoint Setup](biopoint.md) | Next: [Processing (Optional)](proc.md)
