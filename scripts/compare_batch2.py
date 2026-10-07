#!/usr/bin/env python3
"""Compare completed pilot runs, rejecting mismatched samples or settings."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from icmr2027.config import resolve_path
from icmr2027.evaluation.comparison import compare_runs, latest_completed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-rag-run")
    parser.add_argument("--top1-rag-run")
    parser.add_argument("--output-root", default="outputs/batch2/comparison")
    args = parser.parse_args()
    left = resolve_path(args.no_rag_run, ROOT) if args.no_rag_run else latest_completed(ROOT / "outputs/batch2/no_rag")
    right = resolve_path(args.top1_rag_run, ROOT) if args.top1_rag_run else latest_completed(ROOT / "outputs/batch2/top1_rag")
    compare_runs(left, right, resolve_path(args.output_root, ROOT))
