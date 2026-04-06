#!/usr/bin/env python3
"""Export a streaming .ts model for nn~ and inspect latent controls.

Example:
  python RAVE/export_nn.py --run_path training_runs/mishmash_demo_hash
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


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

    raise FileNotFoundError("Could not locate `rave` executable.")


def to_jsonable(value: Any) -> Any:
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None

    if isinstance(value, (str, bool, int, float)):
        return value
    if torch is not None and isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        if value.numel() <= 16:
            return value.detach().cpu().reshape(-1).tolist()
        return {"shape": list(value.shape)}
    return str(value)


def latest_ts_file(search_root: Path) -> Optional[Path]:
    if not search_root.exists():
        return None
    files = sorted(search_root.rglob("*.ts"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if value.isdigit():
            return int(value)
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def inspect_ts(ts_path: Path) -> Dict[str, Any]:
    import torch

    model = torch.jit.load(str(ts_path), map_location="cpu").eval()
    info: Dict[str, Any] = {
        "ts_path": str(ts_path.resolve()),
    }

    for attr in ["latent_size", "full_latent_size", "n_channels", "target_channels", "sr", "block_size", "n_signal"]:
        if hasattr(model, attr):
            info[attr] = to_jsonable(getattr(model, attr))

    if hasattr(model, "fidelity"):
        fidelity = getattr(model, "fidelity")
        if isinstance(fidelity, torch.Tensor):
            info["fidelity_curve_shape"] = list(fidelity.shape)
            info["fidelity_curve_points"] = int(fidelity.numel())

    n_channels = maybe_int(info.get("n_channels")) or 1
    probe_samples = 2**16
    x = torch.zeros(1, n_channels, probe_samples)
    with torch.no_grad():
        z = model.encode(x)
        y = model.decode(z)

    info["probe_samples"] = probe_samples
    info["encode_shape"] = list(z.shape)
    info["decode_shape"] = list(y.shape)
    info["latent_control_channels"] = int(z.shape[1])

    latent_frames = int(z.shape[-1]) if z.ndim >= 3 else None
    if latent_frames and latent_frames > 0:
        info["latent_frames_per_probe"] = latent_frames
        info["latent_block_size"] = int(probe_samples // latent_frames)

    # Use explicit model attr if available, otherwise keep inferred probe value.
    explicit_block = maybe_int(info.get("block_size")) or maybe_int(info.get("n_signal"))
    if explicit_block and explicit_block > 0:
        info["latent_block_size"] = explicit_block

    info["recommended_live_range"] = [-3.0, 3.0]
    info["default_latent_value"] = 0.0
    info["notes"] = [
        "For nn~, latent controls map to decode input channels.",
        "Use latent_control_channels as the number of controllable latent dimensions.",
    ]
    return info


def sanitize_token(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return safe or "model"


def derive_base_name(run_path: Path, ts_path: Path, base_name: Optional[str]) -> str:
    if base_name:
        return sanitize_token(base_name)

    candidate = run_path.name if run_path.name else ts_path.parent.name
    # Remove hash-like suffixes often appended by training framework.
    candidate = re.sub(r"_[0-9a-f]{6,}$", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"_[0-9]{8,}$", "", candidate)
    if candidate in {"runs", "training_runs"}:
        candidate = ts_path.parent.name
    return sanitize_token(candidate)


def build_tagged_name(base_name: str, info: Dict[str, Any], fallback_sr: Optional[int]) -> str:
    block = maybe_int(info.get("latent_block_size"))
    sr = maybe_int(info.get("sr")) or fallback_sr
    z_dim = maybe_int(info.get("latent_control_channels"))

    parts = [sanitize_token(base_name)]
    if block and block > 0:
        parts.append(f"b{block}")
    if sr and sr > 0:
        parts.append(f"r{sr}")
    if z_dim and z_dim > 0:
        parts.append(f"z{z_dim}")
    return "_".join(parts) + ".ts"


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for idx in range(1, 1000):
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find available filename for: {path}")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RAVE .ts for nn~ and inspect latent controls.")
    parser.add_argument("--run_path", required=True, help="Run directory, checkpoint, or training root.")
    parser.add_argument("--fidelity", type=float, default=0.95, help="Fidelity for export.")
    parser.add_argument("--name", default=None, help="Optional exported model basename passed to rave export.")
    parser.add_argument("--base_name", default=None, help="Base token for final auto-tagged filename.")
    parser.add_argument("--output", default=None, help="Optional exported model output directory.")
    parser.add_argument("--channels", type=int, default=None, help="Optional output channels override.")
    parser.add_argument("--sr", type=int, default=None, help="Optional export sample rate override.")
    parser.add_argument("--ema_weights", action="store_true", help="Use EMA weights for export.")
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument("--streaming", dest="streaming", action="store_true", default=True)
    stream_group.add_argument("--no_streaming", dest="streaming", action="store_false")
    auto_name_group = parser.add_mutually_exclusive_group()
    auto_name_group.add_argument("--auto_name", dest="auto_name", action="store_true", default=True)
    auto_name_group.add_argument("--no_auto_name", dest="auto_name", action="store_false")
    parser.add_argument(
        "--skip_export",
        action="store_true",
        help="Skip export command and only inspect latest existing .ts under run/output path.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    root = repo_root()
    rave_exe = detect_rave_executable(root)
    run_path = resolve_path(args.run_path, root)

    if not args.skip_export:
        cmd = [
            str(rave_exe),
            "export",
            "--run",
            str(run_path),
            "--fidelity",
            str(args.fidelity),
        ]
        if args.streaming:
            cmd.append("--streaming")
        if args.name:
            cmd += ["--name", args.name]
        if args.output:
            output = resolve_path(args.output, root)
            output.mkdir(parents=True, exist_ok=True)
            cmd += ["--output", str(output)]
        if args.channels is not None:
            cmd += ["--channels", str(args.channels)]
        if args.sr is not None:
            cmd += ["--sr", str(args.sr)]
        if args.ema_weights:
            cmd.append("--ema_weights")

        print("Running:", " ".join(shlex.quote(x) for x in cmd))
        subprocess.run(cmd, check=True)

    search_root = resolve_path(args.output, root) if args.output else run_path
    ts_path = latest_ts_file(search_root)
    if ts_path is None:
        raise FileNotFoundError(f"No .ts file found under: {search_root}")

    info = inspect_ts(ts_path)

    if args.auto_name:
        base = derive_base_name(run_path, ts_path, args.base_name)
        tagged_name = build_tagged_name(base, info, fallback_sr=args.sr)
        target = ts_path.with_name(tagged_name)
        destination = ts_path if target == ts_path else unique_destination(target)
        if destination != ts_path:
            ts_path.rename(destination)
            ts_path = destination
            info["ts_path"] = str(ts_path.resolve())
            info["auto_named"] = True
            info["auto_name_pattern"] = ts_path.name

    sidecar = ts_path.with_suffix(".nn_info.json")
    sidecar.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"Exported model: {ts_path}")
    print(f"Latent control channels: {info['latent_control_channels']}")
    print(f"Model sample rate: {info.get('sr', 'unknown')}")
    print(f"Metadata written: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
