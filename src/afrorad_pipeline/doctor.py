from __future__ import annotations

import importlib
import os
import sys
from typing import Sequence

REQUIRED_IMPORTS: Sequence[str] = (
    "accelerate",
    "datasets",
    "evaluate",
    "hydra",
    "omegaconf",
    "pandas",
    "torch",
    "torchaudio",
    "transformers",
)


def _check_python_version() -> list[str]:
    issues: list[str] = []
    if sys.version_info < (3, 10):
        issues.append("Python >= 3.10 is required.")
    return issues


def _check_imports() -> list[str]:
    issues: list[str] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            issues.append(f"Missing import '{module_name}': {exc}")
    return issues


def _check_hf_token() -> list[str]:
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return []
    return ["HF_TOKEN is not set. Required for gated dataset access."]


def _check_torch_runtime() -> list[str]:
    issues: list[str] = []
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import is checked above
        return [f"Unable to inspect torch runtime: {exc}"]

    if torch.cuda.is_available():
        print(f"CUDA available: yes ({torch.cuda.device_count()} GPU)")
    else:
        print("CUDA available: no (CPU fallback)")
    return issues


def main() -> None:
    print(f"Python executable: {sys.executable}")
    print(f"Python version   : {sys.version.split()[0]}")

    issues: list[str] = []
    issues.extend(_check_python_version())
    issues.extend(_check_imports())
    issues.extend(_check_hf_token())
    issues.extend(_check_torch_runtime())

    if issues:
        print("\n[doctor] FAIL")
        for item in issues:
            print(f"- {item}")
        raise SystemExit(1)

    print("\n[doctor] OK")


if __name__ == "__main__":
    main()
