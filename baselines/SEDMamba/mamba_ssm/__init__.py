"""Compatibility shim for newer mamba-ssm package layouts.

This project only uses the original `Mamba` block. Some newer `mamba_ssm`
releases import `Mamba2` at package import time, which pulls in optional
`huggingface_hub` networking dependencies that are irrelevant for training
here and may be missing on cluster environments.

The shim shadows the installed top-level `mamba_ssm` package and forwards only
the symbols this repo actually needs, while delegating submodule lookups to the
real installed package directory.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import site
import sys


def _candidate_package_dirs() -> list[Path]:
    candidates: list[Path] = []
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        candidates.append(Path(user_site) / "mamba_ssm")
    else:
        candidates.extend(Path(path) / "mamba_ssm" for path in user_site)

    for path in site.getsitepackages():
        candidates.append(Path(path) / "mamba_ssm")

    for path in sys.path:
        if path:
            candidates.append(Path(path) / "mamba_ssm")
    return candidates


def _resolve_real_package_dir() -> Path:
    shim_dir = Path(__file__).resolve().parent
    seen: set[Path] = set()
    for candidate in _candidate_package_dirs():
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved == shim_dir:
            continue
        if candidate.is_dir():
            return resolved
    raise ImportError("Could not locate an installed `mamba_ssm` package directory.")


_REAL_PACKAGE_DIR = _resolve_real_package_dir()

# Point submodule imports (for example `mamba_ssm.ops...`) at the real package.
__path__ = [str(_REAL_PACKAGE_DIR)]
__file__ = str(_REAL_PACKAGE_DIR / "__init__.py")

selective_scan_interface = import_module("mamba_ssm.ops.selective_scan_interface")
mamba_simple = import_module("mamba_ssm.modules.mamba_simple")

selective_scan_fn = selective_scan_interface.selective_scan_fn
mamba_inner_fn = selective_scan_interface.mamba_inner_fn
Mamba = mamba_simple.Mamba
__all__ = ["Mamba", "mamba_inner_fn", "selective_scan_fn"]
