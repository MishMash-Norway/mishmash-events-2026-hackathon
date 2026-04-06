#!/usr/bin/env python3
"""Convenience Python wrapper for `rave train` (+ optional TensorBoard/export)."""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


TEST_DEFAULTS = {
    "batch": 4,
    "max_steps": 3000,
    "val_every": 500,
    "workers": 0,
    "phase_1_duration": 1000,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


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

    raise FileNotFoundError(
        "Could not locate `rave` executable. Activate your environment first "
        "or install acids-rave."
    )


def python_for_rave(rave_exe: Path) -> Path:
    candidate = rave_exe.with_name("python")
    if candidate.exists():
        return candidate
    return Path(sys.executable).resolve()


def normalize_augment(augment: str) -> str:
    """Resolve short augmentation names to explicit gin files when possible."""
    candidate = Path(augment).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    if "/" in augment or "\\" in augment or augment.endswith(".gin"):
        return augment

    try:
        import rave  # type: ignore

        gin_path = Path(rave.BASE_PATH) / "configs" / "augmentations" / f"{augment}.gin"
        if gin_path.exists():
            return str(gin_path.resolve())
    except (ImportError, AttributeError, TypeError):
        pass

    return augment


def maybe_warn_for_empty_dataset(db_path: Path) -> None:
    metadata_file = db_path / "metadata.yaml"
    if not metadata_file.exists():
        return
    text = metadata_file.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^n_seconds:\s*([0-9]+)\s*$", text, re.MULTILINE)
    if match and int(match.group(1)) == 0:
        print(
            "[warning] Dataset metadata has n_seconds: 0. "
            "If training fails with num_samples=0, rebuild preprocess with --lazy.",
            file=sys.stderr,
        )


def read_dataset_metadata_value(db_path: Path, key: str) -> str | None:
    metadata_file = db_path / "metadata.yaml"
    if not metadata_file.exists():
        return None

    text = metadata_file.read_text(encoding="utf-8", errors="ignore")
    match = re.search(rf"^{re.escape(key)}:\s*([^\n#]+)\s*$", text, re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def parse_int_value(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip().strip("'\"")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def maybe_apply_dataset_sampling_rate_override(args: argparse.Namespace, db_path: Path) -> None:
    if not args.auto_sampling_rate:
        return
    if has_override_binding(args.override, "SAMPLING_RATE"):
        return

    sr_value = parse_int_value(read_dataset_metadata_value(db_path, "sr"))
    if sr_value is None or sr_value <= 0:
        return

    args.override.append(f"SAMPLING_RATE = {sr_value}")
    print(
        f"[info] Auto override from dataset metadata: SAMPLING_RATE = {sr_value}",
        file=sys.stderr,
    )


def is_path_like(value: str) -> bool:
    return "/" in value or "\\" in value or value.startswith(".") or value.startswith("~")


def choose_best_dataset(candidates: Sequence[Path]) -> Path:
    # Prefer normalized dataset variant when available.
    norm = [p for p in candidates if p.name.endswith("_norm_dataset")]
    pool = norm if norm else list(candidates)
    return sorted(pool, key=lambda p: p.stat().st_mtime)[-1]


def resolve_dataset_selector(dataset: str, root: Path) -> Path:
    selector = dataset.strip()
    if not selector:
        raise ValueError("Dataset selector cannot be empty.")

    # Path-like selector can be either:
    # - direct LMDB dataset path
    # - raw audio path, in which case sibling *_dataset path is inferred.
    if is_path_like(selector):
        path = resolve_path(selector, root)
        if path.exists() and path.is_dir() and path.name.endswith(("_dataset", "_norm_dataset")):
            return path
        if path.exists() and path.is_dir():
            for suffix in ("_norm_dataset", "_dataset"):
                candidate = path.parent / f"{path.name}{suffix}"
                if candidate.exists() and candidate.is_dir():
                    return candidate.resolve()
        raise FileNotFoundError(
            f"Could not resolve dataset from selector: {selector}\n"
            "If this is a raw audio folder, run RAVE/preprocess_audio_folder.py first."
        )

    expected = [f"{selector}_norm_dataset", f"{selector}_dataset"]
    direct_candidates = [
        root / "data" / expected[0],
        root / "data" / expected[1],
        root / expected[0],
        root / expected[1],
    ]
    found: List[Path] = [p.resolve() for p in direct_candidates if p.exists() and p.is_dir()]
    if not found:
        # Fallback search for non-standard data locations.
        for search_root in [root / "data", root]:
            if not search_root.exists():
                continue
            for pattern in expected:
                for path in search_root.rglob(pattern):
                    if path.is_dir():
                        found.append(path.resolve())

    if not found:
        raise FileNotFoundError(
            f"No preprocessed dataset found for '{selector}'.\n"
            f"Expected folder names include '{selector}_dataset' or '{selector}_norm_dataset'."
        )

    unique = sorted({p for p in found}, key=str)
    chosen = choose_best_dataset(unique)
    if len(unique) > 1:
        print(
            f"[info] Multiple datasets matched '{selector}'. "
            f"Using latest preferred match: {chosen}",
            file=sys.stderr,
        )
    return chosen


def has_override_binding(overrides: Sequence[str], binding_key: str) -> bool:
    for item in overrides:
        lhs = item.split("=", 1)[0].strip()
        if lhs == binding_key:
            return True
    return False


def apply_test_preset(args: argparse.Namespace) -> None:
    if not args.test:
        return
    if args.batch is None:
        args.batch = TEST_DEFAULTS["batch"]
    if args.max_steps is None:
        args.max_steps = TEST_DEFAULTS["max_steps"]
    if args.val_every is None:
        args.val_every = TEST_DEFAULTS["val_every"]
    if args.workers is None:
        args.workers = TEST_DEFAULTS["workers"]

    if not has_override_binding(args.override, "PHASE_1_DURATION"):
        args.override.append(f"PHASE_1_DURATION = {TEST_DEFAULTS['phase_1_duration']}")


def build_train_command(
    args: argparse.Namespace,
    root: Path,
    rave_exe: Path,
) -> Tuple[List[str], Path, Path]:
    db_path = resolve_path(args.db_path, root) if args.db_path else resolve_dataset_selector(args.dataset, root)
    if not db_path.exists() or not db_path.is_dir():
        raise FileNotFoundError(f"Preprocessed dataset directory not found: {db_path}")

    out_path = resolve_path(args.out_path, root)
    out_path.mkdir(parents=True, exist_ok=True)
    maybe_warn_for_empty_dataset(db_path)
    maybe_apply_dataset_sampling_rate_override(args, db_path)

    cmd: List[str] = [
        str(rave_exe),
        "train",
        "--db_path",
        str(db_path),
        "--out_path",
        str(out_path),
        "--name",
        args.name,
    ]

    if args.channels is not None:
        cmd += ["--channels", str(args.channels)]
    if args.model_config:
        cmd += ["--config", args.model_config]

    if not args.no_causal:
        cmd += ["--config", "causal"]

    for augment in args.augment:
        cmd += ["--augment", normalize_augment(augment)]

    if args.batch is not None:
        cmd += ["--batch", str(args.batch)]
    if args.max_steps is not None:
        cmd += ["--max_steps", str(args.max_steps)]
    if args.val_every is not None:
        cmd += ["--val_every", str(args.val_every)]
    if args.workers is not None:
        cmd += ["--workers", str(args.workers)]

    for gpu_id in args.gpu:
        cmd += ["--gpu", str(gpu_id)]

    for override in args.override:
        cmd += ["--override", override]

    if args.extra_args:
        cmd += list(args.extra_args)

    return cmd, db_path, out_path


def choose_latest_run_dir(out_path: Path, run_name: str) -> Path:
    if not out_path.exists():
        return out_path
    candidates = [p for p in out_path.iterdir() if p.is_dir() and p.name.startswith(f"{run_name}_")]
    if not candidates:
        return out_path
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def launch_tensorboard(args: argparse.Namespace, rave_exe: Path, logdir: Path) -> subprocess.Popen:
    tb_python = python_for_rave(rave_exe)
    tb_cmd = [
        str(tb_python),
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--host",
        args.tensorboard_host,
        "--port",
        str(args.tensorboard_port),
    ]
    print("Launching TensorBoard:", " ".join(shlex.quote(x) for x in tb_cmd))
    process = subprocess.Popen(
        tb_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(
        f"TensorBoard started (pid={process.pid}) at "
        f"http://{args.tensorboard_host}:{args.tensorboard_port}"
    )
    return process


def build_export_nn_command(
    args: argparse.Namespace,
    root: Path,
    run_target: Path,
) -> List[str]:
    cmd = [
        str(sys.executable),
        str(root / "RAVE" / "export_nn.py"),
        "--run_path",
        str(run_target),
        "--fidelity",
        str(args.export_fidelity),
    ]
    cmd += ["--streaming"] if args.export_streaming else ["--no_streaming"]

    if args.export_name:
        cmd += ["--name", args.export_name]
    if args.export_output:
        output_path = resolve_path(args.export_output, root)
        output_path.mkdir(parents=True, exist_ok=True)
        cmd += ["--output", str(output_path)]
    if args.export_channels is not None:
        cmd += ["--channels", str(args.export_channels)]
    if args.export_sr is not None:
        cmd += ["--sr", str(args.export_sr)]
    if args.export_ema_weights:
        cmd += ["--ema_weights"]
    if args.export_base_name:
        cmd += ["--base_name", args.export_base_name]
    if args.no_export_auto_name:
        cmd += ["--no_auto_name"]

    return cmd


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAVE training via local Python wrapper.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset",
        help=(
            "Dataset selector (recommended): original raw folder name, for example 'Actor 1'. "
            "Resolves to <name>_norm_dataset or <name>_dataset automatically."
        ),
    )
    source.add_argument("--db_path", help="Path to preprocessed RAVE dataset (legacy explicit mode).")

    parser.add_argument("--out_path", default="training_runs", help="Output directory for runs.")
    parser.add_argument("--name", default="mishmash_demo", help="Run name.")
    parser.add_argument(
        "--channels",
        type=int,
        default=None,
        help="Channel count passed to train. If omitted, RAVE default is used.",
    )
    parser.add_argument(
        "--model_config",
        default=None,
        help="Model config to pass as --config (example: raspberry, v2_small, v2).",
    )
    parser.add_argument("--no_causal", action="store_true", help="Disable default causal config.")
    auto_sr_group = parser.add_mutually_exclusive_group()
    auto_sr_group.add_argument(
        "--auto_sampling_rate",
        dest="auto_sampling_rate",
        action="store_true",
        default=True,
        help="Auto-set SAMPLING_RATE override from dataset metadata (default).",
    )
    auto_sr_group.add_argument(
        "--no_auto_sampling_rate",
        dest="auto_sampling_rate",
        action="store_false",
        help="Disable SAMPLING_RATE auto-override from dataset metadata.",
    )
    parser.add_argument("--test", action="store_true", help="Apply quick local smoke-test presets.")
    parser.add_argument(
        "--augment",
        action="append",
        default=[],
        help="Augmentation gin config(s). Can repeat. Short names auto-map when possible.",
    )
    parser.add_argument("--batch", type=int, default=None, help="Batch size.")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps.")
    parser.add_argument("--val_every", type=int, default=None, help="Validation interval.")
    parser.add_argument("--workers", type=int, default=None, help="Data loader workers.")
    parser.add_argument(
        "--gpu",
        action="append",
        type=int,
        default=[],
        help="GPU id(s). Repeat for multiple GPUs. Use -1 for CPU.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Gin override binding(s). Can repeat.",
    )
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional raw args passed to `rave train`.",
    )

    parser.add_argument(
        "--launch_tensorboard",
        action="store_true",
        help="Launch TensorBoard automatically before training.",
    )
    parser.add_argument(
        "--tensorboard_host",
        default="127.0.0.1",
        help="TensorBoard host (used with --launch_tensorboard).",
    )
    parser.add_argument(
        "--tensorboard_port",
        type=int,
        default=6006,
        help="TensorBoard port (used with --launch_tensorboard).",
    )

    parser.add_argument(
        "--export_after_train",
        action="store_true",
        help="Automatically export + inspect .ts after successful training.",
    )
    parser.add_argument(
        "--export_fidelity",
        type=float,
        default=0.95,
        help="Fidelity used for export (variational models).",
    )
    export_streaming = parser.add_mutually_exclusive_group()
    export_streaming.add_argument(
        "--export_streaming",
        dest="export_streaming",
        action="store_true",
        default=True,
        help="Enable streaming export mode (default).",
    )
    export_streaming.add_argument(
        "--no_export_streaming",
        dest="export_streaming",
        action="store_false",
        help="Disable streaming export mode.",
    )
    parser.add_argument("--export_output", default=None, help="Optional export output directory.")
    parser.add_argument("--export_name", default=None, help="Optional exported model basename.")
    parser.add_argument(
        "--export_base_name",
        default=None,
        help="Base token used by export naming tags (for example birds_dawnchorus).",
    )
    parser.add_argument("--export_channels", type=int, default=None, help="Optional export channels.")
    parser.add_argument("--export_sr", type=int, default=None, help="Optional export sample rate.")
    parser.add_argument(
        "--export_ema_weights",
        action="store_true",
        help="Use EMA weights during export if available.",
    )
    parser.add_argument(
        "--no_export_auto_name",
        action="store_true",
        help="Disable automatic _b*_r*_z* tags in exported filename.",
    )

    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    apply_test_preset(args)

    root = repo_root()
    rave_exe = detect_rave_executable(root)
    train_cmd, db_path, out_path = build_train_command(args, root, rave_exe)

    tb_python = python_for_rave(rave_exe)
    print(f"Resolved dataset path: {db_path}")
    print(
        "TensorBoard command:",
        " ".join(
            shlex.quote(x)
            for x in [
                str(tb_python),
                "-m",
                "tensorboard.main",
                "--logdir",
                str(out_path),
                "--host",
                args.tensorboard_host,
                "--port",
                str(args.tensorboard_port),
            ]
        ),
    )
    if args.launch_tensorboard:
        launch_tensorboard(args, rave_exe, out_path)

    print("Running:", " ".join(shlex.quote(x) for x in train_cmd))
    subprocess.run(train_cmd, check=True)

    if args.export_after_train:
        run_target = choose_latest_run_dir(out_path, args.name)
        export_cmd = build_export_nn_command(args, root, run_target)
        print("Running export:", " ".join(shlex.quote(x) for x in export_cmd))
        subprocess.run(export_cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
