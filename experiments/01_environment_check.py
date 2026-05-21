"""Environment and hardware check.

Run this first. Confirms that the Python environment has the basics
in place and that MPS is available before we attempt anything heavier.
"""

from __future__ import annotations

import importlib
import platform
import sys


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_python() -> None:
    print(f"Python:    {sys.version.split()[0]}")
    print(f"Platform:  {platform.platform()}")
    print(f"Machine:   {platform.machine()}")


def check_torch() -> None:
    try:
        import torch
    except ImportError as e:
        print(f"PyTorch not installed: {e}")
        return

    print(f"PyTorch:       {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built:     {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        try:
            x = torch.randn(256, 256, device="mps")
            y = (x @ x.T).sum().item()
            print(f"MPS smoke OK:  matmul finite={y == y}")  # NaN check
        except Exception as e:
            print(f"MPS smoke failed: {e}")


def check_memory() -> None:
    try:
        import psutil
    except ImportError:
        print("psutil not installed; skipping memory check")
        return
    vm = psutil.virtual_memory()
    print(f"Total RAM:     {vm.total / 1e9:.1f} GB")
    print(f"Available RAM: {vm.available / 1e9:.1f} GB")


def check_libraries() -> None:
    libs = [
        "torch", "transformers", "tokenizers", "numpy", "scipy",
        "sklearn", "matplotlib", "einops", "tqdm", "psutil",
    ]
    for lib in libs:
        try:
            importlib.import_module(lib)
            status = "OK"
        except ImportError:
            status = "MISSING"
        print(f"  {lib:15s} {status}")


if __name__ == "__main__":
    section("Python / platform")
    check_python()

    section("PyTorch / MPS")
    check_torch()

    section("System memory")
    check_memory()

    section("Required libraries")
    check_libraries()
