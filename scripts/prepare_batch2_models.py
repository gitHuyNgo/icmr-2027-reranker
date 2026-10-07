#!/usr/bin/env python3
"""Explicitly cache pinned model snapshots; never called by inference."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from icmr2027.config import load_config, resolve_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline/batch2_ravenea_top1_rag.yaml")
    args = parser.parse_args()
    config = load_config(resolve_path(args.config, ROOT))
    from huggingface_hub import snapshot_download
    targets = [(config.model.name, config.model.revision)]
    if config.retrieval is not None:
        targets.append((config.retrieval.name, config.retrieval.revision))
    for model, revision in targets:
        print(f"Explicit model preparation: {model}@{revision}", flush=True)
        print(snapshot_download(model, revision=revision,
                                allow_patterns=["*.json", "*.safetensors", "*.txt"]), flush=True)
