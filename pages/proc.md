# Processing

[Back to Repo README](../README.md) | Previous: [Import into Max/Pd](import_max_pd.md) | Next: [Training](training.md)

This guide explains how to process one audio folder into a RAVE-ready dataset using terminal commands.

## 1) Activate Environment

```bash
cd /path/to/rave_hackathon
source .venv/bin/activate
which python
```

## 2) Verify System Audio Tools

```bash
which ffmpeg
which ffprobe
```

If `ffprobe` is missing on Ubuntu:

```bash
sudo apt update && sudo apt install -y ffmpeg
```

## 3) Choose Raw Audio Folder

Example raw folder:

- `data/Emotions`

Nested folders/files are supported.

## 4) Preprocess (No Normalization)

```bash
python RAVE/preprocess_audio_folder.py \
  --audio_dir "data/Emotions" \
  --sampling_rate 48000 \
  --channels 1 \
  --lazy
```

Output folder:

- `data/Emotions_dataset`

## 5) Preprocess (With Normalization)

```bash
python RAVE/preprocess_audio_folder.py \
  --audio_dir "data/Emotions" \
  --sampling_rate 48000 \
  --channels 1 \
  --normalize \
  --lazy
```

Output folders:

- `data/Emotions_norm_dataset` (training dataset)
- `data/Emotions_norm_audio` (persistent normalized source used by lazy loading)

Important: do not delete `*_norm_audio` while training a lazy dataset.

## 6) Rebuild Existing Output

```bash
python RAVE/preprocess_audio_folder.py \
  --audio_dir "data/Emotions" \
  --sampling_rate 48000 \
  --channels 1 \
  --normalize \
  --lazy \
  --overwrite
```

## 7) Dataset Selector Convention

For training, pass the original base name:

- `--dataset "Emotions"`

`RAVE/train.py` resolves:

- `Emotions_norm_dataset` first (preferred)
- fallback to `Emotions_dataset`

---

[Back to Repo README](../README.md) | Previous: [Import into Max/Pd](import_max_pd.md) | Next: [Training](training.md)
