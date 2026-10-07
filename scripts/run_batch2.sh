#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
"$PYTHON" -c 'import sys; print("Python executable:", sys.executable)'
COMMAND="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi

case "$COMMAND" in
    prepare)
        "$PYTHON" scripts/prepare_ravenea.py "$@"
        "$PYTHON" scripts/prepare_batch2_models.py
        ;;
    validate)
        "$PYTHON" scripts/prepare_ravenea.py --validate-only "$@"
        ;;
    retrieval|no-rag|top1-rag)
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            printf 'ERROR: nvidia-smi unavailable; use an NVIDIA GPU server.\n' >&2
            exit 1
        fi
        nvidia-smi
        "$PYTHON" - <<'PY'
import torch
import torchvision
print("Torch:", torch.__version__, "Torchvision:", torchvision.__version__)
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable")
print("GPU:", torch.cuda.get_device_name(0))
PY
        case "$COMMAND" in
            retrieval) "$PYTHON" scripts/cache_ravenea_retrieval.py "$@" ;;
            no-rag) "$PYTHON" scripts/run_baseline.py --config configs/baseline/batch2_ravenea_no_rag.yaml "$@" ;;
            top1-rag) "$PYTHON" scripts/run_baseline.py --config configs/baseline/batch2_ravenea_top1_rag.yaml "$@" ;;
        esac
        ;;
    compare) "$PYTHON" scripts/compare_batch2.py "$@" ;;
    *) printf 'Usage: bash scripts/run_batch2.sh {prepare|validate|retrieval|no-rag|top1-rag|compare} [args]\n' >&2; exit 2 ;;
esac
