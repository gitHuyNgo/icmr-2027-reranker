#!/usr/bin/env python3
"""Validate and audit the immutable original 50-question Batch 2 runs; no inference."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from icmr2027.config import resolve_path
from icmr2027.evaluation.comparison import latest_completed
from icmr2027.evaluation.format_audit import audit_pair, load_original_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-rag-run")
    parser.add_argument("--top1-rag-run")
    parser.add_argument("--output-root", default="outputs/batch2_1")
    parser.add_argument("--origin-root", help="Original server repository path when auditing byte-identical local copies")
    args = parser.parse_args()
    runs = [resolve_path(value, ROOT) if value else latest_completed(ROOT / "outputs/batch2" / condition)
            for value, condition in ((args.no_rag_run, "no_rag"), (args.top1_rag_run, "top1_rag"))]
    output = resolve_path(args.output_root, ROOT) / "audit"
    result = audit_pair(load_original_pair(ROOT, runs), output, args.origin_root)
    print(json.dumps({key: result[key] for key in ("original_run_paths", "classification_counts", "before")}, indent=2))
    print(f"Audit: {output}")


if __name__ == "__main__":
    main()
