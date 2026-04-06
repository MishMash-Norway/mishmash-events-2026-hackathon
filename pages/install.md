# Installation

[Back to Repo README](../README.md) | Previous: [Repo README](../README.md) | Next: [BioPoint Setup](biopoint.md)

This page covers the initial environment setup for the terminal-first workflow.

## 1) System Prerequisites

- Python `3.10+` (3.10/3.11 recommended)
- `ffmpeg` and `ffprobe`

Ubuntu:

```bash
sudo apt update && sudo apt install -y ffmpeg
```

macOS (Homebrew):

```bash
brew install ffmpeg
```

## 2) Create and Activate venv

```bash
cd /path/to/rave_hackathon
python3 -m venv .venv
source .venv/bin/activate
which python
```

## 3) Install PyTorch First

Ubuntu + NVIDIA (CUDA 12.4 wheels):

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

macOS / CPU path:

```bash
pip install torch torchaudio
```

## 4) Install Project Requirements

```bash
pip install -r requirements.txt
pip install --force-reinstall "setuptools<81"
```

## 5) Sanity Checks

```bash
python -c "import torch, torchaudio; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('torchaudio', torchaudio.__version__)"
which ffmpeg
which ffprobe
rave --help
```

## 6) Common Fixes

If `pkg_resources` is missing:

```bash
pip install --force-reinstall "setuptools<81"
```

If CUDA runtime mismatch appears (`libcudart.so` errors), reinstall torch/torchaudio with matching CUDA wheels in this venv.

---

[Back to Repo README](../README.md) | Previous: [Repo README](../README.md) | Next: [BioPoint Setup](biopoint.md)
