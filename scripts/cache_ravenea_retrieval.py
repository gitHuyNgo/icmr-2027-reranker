#!/usr/bin/env python3
"""Build or validate the reusable offline RAVENEA-CLIP Top-10 cache."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from icmr2027.config import load_config, resolve_path
from icmr2027.retrieval.build import build_retrieval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline/batch2_ravenea_top1_rag.yaml")
    parser.add_argument("--force", action="store_true", help="Explicitly replace reusable cache artifacts")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    build_retrieval(load_config(resolve_path(args.config, ROOT)), ROOT, args.force, args.validate_only)
