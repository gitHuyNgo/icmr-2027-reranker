#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
printf 'Repository: %s\n' "$REPO_ROOT"
"$PYTHON" -c 'import sys; print("Python executable:", sys.executable)'
if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'ERROR: nvidia-smi is unavailable; run on an NVIDIA GPU server.\n' >&2
    exit 1
fi
nvidia-smi
"$PYTHON" - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable. Install CUDA-enabled PyTorch and check GPU/driver access.")
print("GPU:", torch.cuda.get_device_name(0))
PY
"$PYTHON" scripts/run_baseline.py --config configs/baseline/batch1_smoke.yaml
# The runner prints the exact path for this invocation as 'Completed run: ...'.
