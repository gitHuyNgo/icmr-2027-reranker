"""Best-effort metadata; no Git repository/commit is a valid state."""
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import socket
import subprocess
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def collect_run_metadata(repo_root: Path, seed: int, model_name: str, device: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp": utc_timestamp(),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_dirty": None,
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "torch_version": package_version("torch"),
        "torchvision_version": package_version("torchvision"),
        "transformers_version": package_version("transformers"),
        "cuda_available": None,
        "cuda_version": None,
        "gpu_name": None,
        "seed": seed,
        "model_name": model_name,
        "device": device,
    }
    git_status = _git(repo_root, "status", "--porcelain")
    if git_status is not None:
        metadata["git_dirty"] = bool(git_status)
    try:
        import torch
        metadata["cuda_available"] = torch.cuda.is_available()
        metadata["cuda_version"] = torch.version.cuda
        if metadata["cuda_available"]:
            target = device if device.startswith("cuda") else "cuda:0"
            metadata["gpu_name"] = torch.cuda.get_device_name(torch.device(target))
    except (ImportError, OSError, RuntimeError, AssertionError) as exc:
        metadata["environment_probe_error"] = f"{type(exc).__name__}: {exc}"
    return metadata
