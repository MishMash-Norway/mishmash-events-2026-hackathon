# MishMash RAVE Hackathon

The hackathon focuses on creative experimentation with RAVE and biosignal-based interaction. Participants work with pretrained models, build embodied mappings and performance patches, and, where useful, incorporate small custom models trained during the event with limited data and short training runs. These quickly trained models may sound rough, unstable, or only partially usable, but that is part of the exploration. More broadly, the event invites participants to explore a different kind of action-sound relationship: rather than mapping bodily input to clearly defined sound parameters, they will engage the latent space of neural audio models, where the function of individual latent dimensions is not always known in advance, and combinations of values can lead to unexpected sonic results. Unlike aiming for technical optimization or benchmark-quality output, the goal is to explore the sound palette of these systems, including their constraints, instabilities, and surprises.

## Repository Layout

- `README.md`: main entry point
- `pages/`: notebook-like markdown pages (`install`, `proc`, `training`, `after_training`, etc.)
- `RAVE/`: all RAVE scripts and related workflow files
- `BioPoint/`: BioPoint tutorials and assets
- `requirements.txt`, `.gitignore`: baseline repo files

## 1) Core Audio/Patching Environments

### Max

- Download: [Cycling '74 - Max Downloads](https://cycling74.com/downloads)
- Max is paid software.
- It can still be used in full mode without saving, which is enough for workshop/hackathon testing.

### Pure Data (Pd)

- Download: [Pure Data Downloads](https://msp.ucsd.edu/software.html)
- Pure Data is free and open-source.

Quick comparison:

- **Max**: polished commercial visual patching environment with a broad package ecosystem.
- **Pd**: lightweight open-source visual patching environment with similar dataflow concepts.

Historical note:

- Max originated in the 1980s at IRCAM, initiated by Miller Puckette as a visual environment for interactive music systems.
- Over time, Max evolved through commercial development (now maintained by Cycling '74), including the MSP audio layer and a broader production ecosystem.
- Pure Data (Pd), also created by Miller Puckette in the 1990s, continued many of the same patching ideas in a free and open-source form.

Both are valid for this hackathon. Use whichever environment matches your workflow.

## 2) Python + Project Setup (Mac / Ubuntu / Windows)

All commands below assume you are in the repository root.

### A) Check Python

macOS / Ubuntu:

```bash
python3 --version
```

Windows (PowerShell):

```powershell
py --version
```

If Python is not found, install it first:

macOS (Homebrew):

```bash
brew install python@3.11
python3 --version
```

Ubuntu:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
python3 --version
```

Windows (PowerShell, winget):

```powershell
winget install Python.Python.3.11
py --version
```

Windows installer alternative:

- Download from [python.org](https://www.python.org/downloads/windows/)
- During install, enable **Add Python to PATH**

### B) Create and Activate venv

macOS / Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### C) Install PyTorch

macOS (Apple Silicon / CPU path):

```bash
pip install torch torchaudio
```

Ubuntu (NVIDIA CUDA wheels example):

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Windows (NVIDIA CUDA wheels example):

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### D) Install Project Requirements

```bash
pip install -r requirements.txt
pip install --force-reinstall "setuptools<81"
```

Why this pin is included:

- `pkg_resources` is a legacy utility module shipped by `setuptools`.
- Parts of the current RAVE/Lightning dependency stack still import `pkg_resources` at runtime.
- `--force-reinstall` ensures an already-installed incompatible `setuptools` build is replaced with a compatible one in this venv.

### E) Installation Checks

macOS / Ubuntu:

```bash
python -c "import torch, torchaudio; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('torchaudio', torchaudio.__version__)"
which ffmpeg
which ffprobe
rave --help
```

Windows (PowerShell):

```powershell
python -c "import torch, torchaudio; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('torchaudio', torchaudio.__version__)"
where ffmpeg
where ffprobe
rave --help
```

## 3) Onboarding

1. [Installation and Environment Check](pages/install.md)
2. [BioPoint Sensor Setup](pages/biopoint.md)
3. [Import Pretrained Models in Max/Pd](pages/import_max_pd.md)

### Optional: Train your own RAVE

You can also train a custom model after validating the pretrained workflow. For baseline guidance, see the official RAVE resources: [website](https://forum.ircam.fr/projects/detail/rave/), [repository](https://github.com/acids-ircam/RAVE), and [paper](https://arxiv.org/abs/2111.05011).

Practical model-size strategy:

- start with `v2_small` + `causal` for quick validation and controlled comparisons
- if quality remains limited, run a fresh `v2` long training on GPU
- keep sample rate aligned to deployment target (for example 48000 for many Max/Pd setups)
- export in streaming mode for realtime use

Estimated training durations on a single modern NVIDIA GPU (dataset and hardware dependent):

- smoke test (~3000 steps): usually under 1 hour
- short 2-phase test (~12000 steps): around 1-4 hours
- longer 2-phase run (~120000 steps): overnight to multi-day

We also encourage "adventurer" runs: short trainings on modest datasets to explore intentionally rough, unstable, or "stillborn" models and test how far they can be pushed artistically.

1. [Training RAVE: data preprocessing, training, exporting `.ts`, and advanced options](pages/main.md)
2. Optional advanced path: [Unconditional Generation with Prior](pages/prior.md)
3. Return to [Import into Max/Pd](pages/import_max_pd.md) and test the exported model in `nn~`.

## 4) Make

This is the main part of the hackathon. Working either solo or in a team, you will create an interactive music instrument and present it in a short performance at the end of the day.

### What you can use

- A BioPoint armband and one or multiple signal streams from it
  - You may also include additional input devices that you bring with you
- As many RAVE models and `nn~` objects as you wish, as long as your computer can handle them
- Any kind of mapping, signal processing, or control strategy, including filtering, feature extraction, statistics, or additional machine learning methods

### What you cannot use

- Additional audio effects, except for basic live-mixing elements such as reverb and compression
- Audio sources other than RAVE-based synthesis, except when using RAVE as an audio effect (timbre transfer) processing an external audio input
