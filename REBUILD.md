# Captured Mac environment — recreation not tested

The dependency files capture the installed environment inspected on 24 October
2026: **CPython 3.14.2, macOS Apple Silicon (arm64)**. They are a version-preserving
starting point, **not a tested lockfile, wheel archive, or guarantee of a working
rebuild**. The owner explicitly chose to skip recreation and smoke tests before
recoverably removing the old environment. No installation was performed.

`requirements.txt` pins the nine previous direct requirements, matching
torch/torchaudio 2.11.0, and setuptools 80.10.2 (preserving the documented `<81`
constraint). It includes the 159-package installed-version snapshot as constraints;
unused snapshot entries are not automatically installed. Package hashes, original
indexes and wheel artifacts were not retained, and fresh availability/resolution
was not checked. Native wheel metadata includes `cp314`/`arm64` tags; some wheels
name macOS 14. Do not treat this profile as validated on other Python versions,
Intel Macs, Windows or Linux/CUDA.

## Future recreation (not executed or verified)

From the repository root, using an explicitly selected compatible interpreter:

```sh
/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv .venv
.venv/bin/python -m pip install "pip==26.0.1"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

Do not run these over an existing environment. If resolution or imports fail,
preserve the version snapshot and investigate compatibility rather than silently
claiming that upgrading packages recreates the old environment. The older generic
README gives conflicting Python 3.12/3.11 advice; this profile records what was
actually installed, not which Python version the workshop originally intended.

Then, with permission, validate `torch`, `torchaudio`, RAVE CLI and relevant
notebooks/workflows, check external `ffmpeg`/`ffprobe`, and register/select the
replacement kernel. The old environment had 52 launchers pointing at a missing
former `Documents/Skole` path; copying it back is not a portable install procedure.

## Known tutorial gaps

The BioPoint/SiFiBand notebooks import `pythonosc` and `sifi_bridge_py`, neither
present in the captured environment nor declared in the original requirements.
They were not blindly added: confirm the appropriate distribution, supported
version and hardware prerequisites from those tutorials/vendor documentation
before installing them. No notebooks or hardware services were executed.

The old environment had no observed editable/local-source installs or unrecorded
custom resources. Installed RECORD ownership/size checks do not rule out same-size
package edits; installed payload hashes and runtime behavior were not audited.
Keep all project source, Git history, notebooks, models, datasets and training
outputs independently of the disposable `.venv` directory.
