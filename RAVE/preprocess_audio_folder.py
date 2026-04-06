#!/usr/bin/env python3
"""Preprocess a raw audio folder into a RAVE dataset with optional normalization.

Workflow:
1) Point to one raw audio folder (nested files are supported).
2) Create a sibling dataset folder next to the raw folder:
   - <folder>_dataset
   - <folder>_norm_dataset (when --normalize is enabled)

Note:
- If --normalize and --lazy are both enabled, normalized audio is written to a
  persistent sibling folder (<folder>_norm_audio). This is required because
  lazy datasets reference source file paths at training time.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List


AUDIO_EXTENSIONS = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma"}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def detect_rave_executable(root: Path) -> Path:
    interpreter_rave = Path(sys.executable).resolve().with_name("rave")
    if interpreter_rave.exists():
        return interpreter_rave

    local_rave = root / "rave" / "bin" / "rave"
    if local_rave.exists():
        return local_rave

    system_rave = shutil.which("rave")
    if system_rave:
        return Path(system_rave).resolve()

    raise FileNotFoundError("Could not locate `rave` executable. Activate your environment first.")


def detect_ffmpeg_executable() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("Could not locate `ffmpeg`. Install it before using --normalize.")
    return ffmpeg


def detect_ffprobe_executable() -> Path:
    """Locate ffprobe, required internally by `rave preprocess`."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return Path(ffprobe).resolve()

    # Some installs keep ffprobe next to ffmpeg but outside PATH.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        sibling = Path(ffmpeg).resolve().with_name("ffprobe")
        if sibling.exists():
            return sibling

    raise FileNotFoundError(
        "Could not locate `ffprobe` (required by `rave preprocess`).\n"
        "Install ffmpeg/ffprobe and ensure `ffprobe` is on PATH.\n"
        "Ubuntu: sudo apt update && sudo apt install -y ffmpeg"
    )


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def list_audio_files(audio_dir: Path) -> List[Path]:
    files = [p for p in audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    return sorted(files)


def output_dataset_dir(audio_dir: Path, normalize: bool) -> Path:
    suffix = "_norm_dataset" if normalize else "_dataset"
    return audio_dir.parent / f"{audio_dir.name}{suffix}"


def output_norm_audio_dir(audio_dir: Path) -> Path:
    return audio_dir.parent / f"{audio_dir.name}_norm_audio"


def normalize_folder_to_temp(
    *,
    source_dir: Path,
    files: Iterable[Path],
    ffmpeg: str,
    sample_rate: int,
    channels: int,
    ffmpeg_filter: str,
) -> tempfile.TemporaryDirectory[str]:
    temp_dir = tempfile.TemporaryDirectory(prefix=f"{source_dir.name}_norm_")
    tmp_root = Path(temp_dir.name)

    source_files = list(files)
    total = len(source_files)
    for idx, src in enumerate(source_files, start=1):
        rel = src.relative_to(source_dir)
        dst = (tmp_root / rel).with_suffix(".wav")
        dst.parent.mkdir(parents=True, exist_ok=True)

        cmd = [ffmpeg, "-y", "-i", str(src), "-ar", str(sample_rate), "-ac", str(channels), "-vn"]
        if ffmpeg_filter:
            cmd += ["-af", ffmpeg_filter]
        cmd.append(str(dst))

        print(f"[normalize] {idx}/{total}: {src}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return temp_dir


def normalize_folder_to_dir(
    *,
    source_dir: Path,
    files: Iterable[Path],
    output_root: Path,
    ffmpeg: str,
    sample_rate: int,
    channels: int,
    ffmpeg_filter: str,
    overwrite: bool,
) -> Path:
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            print(f"[normalize] Reusing existing normalized folder: {output_root}")
            return output_root

    output_root.mkdir(parents=True, exist_ok=True)
    source_files = list(files)
    total = len(source_files)
    for idx, src in enumerate(source_files, start=1):
        rel = src.relative_to(source_dir)
        dst = (output_root / rel).with_suffix(".wav")
        dst.parent.mkdir(parents=True, exist_ok=True)

        cmd = [ffmpeg, "-y", "-i", str(src), "-ar", str(sample_rate), "-ac", str(channels), "-vn"]
        if ffmpeg_filter:
            cmd += ["-af", ffmpeg_filter]
        cmd.append(str(dst))

        print(f"[normalize] {idx}/{total}: {src}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_root


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAVE dataset from a raw audio folder.")
    parser.add_argument("--audio_dir", required=True, help="Raw audio folder (nested files allowed).")
    parser.add_argument("--sampling_rate", type=int, default=48000, help="Target sample rate for preprocess.")
    parser.add_argument("--channels", type=int, default=1, help="Channel count for preprocess.")
    parser.add_argument("--normalize", action="store_true", help="Normalize audio with ffmpeg before preprocess.")
    parser.add_argument("--ffmpeg_filter", default="", help="Optional ffmpeg filter chain (example: loudnorm).")
    parser.add_argument("--lazy", dest="lazy", action="store_true", default=True, help="Enable RAVE --lazy (default).")
    parser.add_argument("--no_lazy", dest="lazy", action="store_false", help="Disable RAVE --lazy.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output dataset folder if it exists.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    root = repo_root()
    rave_exe = detect_rave_executable(root)
    ffprobe_exe = detect_ffprobe_executable()

    audio_dir = resolve_path(args.audio_dir, root)
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    audio_files = list_audio_files(audio_dir)
    if not audio_files:
        raise FileNotFoundError(f"No supported audio files found under: {audio_dir}")

    dataset_dir = output_dataset_dir(audio_dir, normalize=args.normalize)
    if dataset_dir.exists():
        if args.overwrite:
            shutil.rmtree(dataset_dir)
        else:
            raise FileExistsError(
                f"Output dataset already exists: {dataset_dir}\n"
                "Use --overwrite to replace it."
            )

    normalize_temp: tempfile.TemporaryDirectory[str] | None = None
    preprocess_input = audio_dir
    if args.normalize:
        ffmpeg = detect_ffmpeg_executable()
        if args.lazy:
            # Lazy mode stores source paths in dataset metadata; use a stable
            # sibling folder instead of temporary files.
            preprocess_input = normalize_folder_to_dir(
                source_dir=audio_dir,
                files=audio_files,
                output_root=output_norm_audio_dir(audio_dir),
                ffmpeg=ffmpeg,
                sample_rate=args.sampling_rate,
                channels=args.channels,
                ffmpeg_filter=args.ffmpeg_filter,
                overwrite=args.overwrite,
            )
        else:
            normalize_temp = normalize_folder_to_temp(
                source_dir=audio_dir,
                files=audio_files,
                ffmpeg=ffmpeg,
                sample_rate=args.sampling_rate,
                channels=args.channels,
                ffmpeg_filter=args.ffmpeg_filter,
            )
            preprocess_input = Path(normalize_temp.name)

    cmd = [
        str(rave_exe),
        "preprocess",
        "--input_path",
        str(preprocess_input),
        "--output_path",
        str(dataset_dir),
        "--channels",
        str(args.channels),
        "--sampling_rate",
        str(args.sampling_rate),
    ]
    if args.lazy:
        cmd.append("--lazy")

    print(f"Found {len(audio_files)} audio files in: {audio_dir}")
    print("Running:", " ".join(shlex.quote(x) for x in cmd))
    runtime_env = os.environ.copy()
    ffprobe_dir = str(ffprobe_exe.parent)
    runtime_env["PATH"] = (
        ffprobe_dir
        if not runtime_env.get("PATH")
        else ffprobe_dir + os.pathsep + runtime_env["PATH"]
    )
    try:
        subprocess.run(
            cmd,
            check=True,
            text=True,
            input="y\n" if args.lazy else None,
            env=runtime_env,
        )
    finally:
        if normalize_temp is not None:
            normalize_temp.cleanup()

    print("\nDone.")
    print(f"Raw audio folder: {audio_dir}")
    if args.normalize and args.lazy:
        print(f"Normalized folder: {preprocess_input}")
    print(f"Dataset folder:   {dataset_dir}")
    print(f"Dataset selector for training: {audio_dir.name}")
    print("\nNext step example:")
    print(
        "python RAVE/train.py "
        f"--dataset {shlex.quote(audio_dir.name)} --name {shlex.quote(audio_dir.name.lower().replace(' ', '_'))}_test "
        "--model_config v2_small --gpu -1 --test --launch_tensorboard"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
