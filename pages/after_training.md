# After Training

[Back to Repo README](../README.md) | Previous: [Training](training.md) | Next: [Import into Max/Pd](import_max_pd.md)

This guide explains how to inspect a finished run and export a `.ts` file.

## 1) Run Folder Naming

Typical format:

- `training_runs/<run_name>_<gin_hash>`
- optionally: `training_runs/<host>/<run_name>_<gin_hash>`

Example:

- `training_runs/cagri-ML-666/emotions_2phase_safe_fd6661f1c8`

## 2) What You Should Find

Common files:

- `config.gin` (resolved training config snapshot)
- `version_0/hparams.yaml`
- `version_0/events.out.tfevents...` (TensorBoard logs)
- `version_0/checkpoints/*.ckpt` (if checkpointing saved)

## 3) Export `.ts` for `nn~`

Export from a specific run:

```bash
python RAVE/export_nn.py \
  --run_path "training_runs/cagri-ML-666/emotions_2phase_safe_fd6661f1c8" \
  --streaming \
  --base_name "emotions_2phase_safe"
```

Checkpoint choice matters:

- prefer `last.ckpt` when available
- otherwise use the newest `epoch-*.ckpt`
- use `best.ckpt` only as fallback (it can reflect pre-phase-2 behavior)

This writes:

- `*.ts`
- `*.nn_info.json`

## 4) Export Naming

Auto naming format:

- `<base>_b<block>_r<sample_rate>_z<latent_channels>.ts`

Example:

- `emotions_2phase_safe_b512_r48000_z2.ts`

## 5) Quick Fidelity Sweep

To test different latent counts:

```bash
RUN="training_runs/cagri-ML-666/emotions_2phase_safe_fd6661f1c8"
for F in 0.95 0.99 0.995 0.999; do
  python RAVE/export_nn.py \
    --run_path "$RUN" \
    --streaming \
    --base_name "emotions_f${F/./_}" \
    --fidelity "$F"
done
```

Then inspect `latent_control_channels` in each `*.nn_info.json`.

---

[Back to Repo README](../README.md) | Previous: [Training](training.md) | Next: [Import into Max/Pd](import_max_pd.md)
