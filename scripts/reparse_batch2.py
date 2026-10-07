#!/usr/bin/env python3
"""Reparse the already audited fixed Batch 2 raw outputs; never loads a model."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from icmr2027.config import resolve_path
from icmr2027.evaluation.format_audit import load_original_pair
from icmr2027.evaluation.reparse import reparse_pair
from icmr2027.utils.io import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/batch2_1")
    args = parser.parse_args()
    output = resolve_path(args.output_root, ROOT)
    audit = json.loads((output / "audit/audit_meta.json").read_text(encoding="utf-8"))
    # Resolve the exact locked runs; never choose newer runs after auditing.
    prefixes = [name.removesuffix("/predictions.jsonl") for name in audit["original_files_sha256"]
                if name.endswith("/predictions.jsonl")]
    paths = [resolve_path(name, ROOT) for name in prefixes]
    by_mode = {read_jsonl(path / "predictions.jsonl")[0]["rag_mode"]: path for path in paths}
    if set(by_mode) != {"none", "top1"} or len(paths) != 2:
        raise ValueError("Audit must lock exactly one original run per condition")
    runs = [by_mode["none"], by_mode["top1"]]
    result = reparse_pair(load_original_pair(ROOT, runs), output)
    print(json.dumps({key: result[key] for key in ("before", "after", "classification_counts")}, indent=2))
    print(f"Before/after: {output / 'comparison/before_after.json'}")


if __name__ == "__main__":
    main()
