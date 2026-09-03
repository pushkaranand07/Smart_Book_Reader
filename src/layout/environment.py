"""Environment and package version reporting."""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict


def collect_environment() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except ImportError:
        info["torch"] = "NOT INSTALLED"

    for pkg in ("transformers", "tokenizers", "peft", "accelerate", "torchvision"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[pkg] = "NOT INSTALLED"

    return info


def print_environment() -> Dict[str, Any]:
    info = collect_environment()
    print("ENVIRONMENT")
    print("=" * 40)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 40)
    return info


def environment_text() -> str:
    return "\n".join(f"{k}={v}" for k, v in collect_environment().items())
