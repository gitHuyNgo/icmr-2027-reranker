#!/usr/bin/env python3
"""Explicitly download/validate released RAVENEA; inference never downloads it."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from icmr2027.config import resolve_path
from icmr2027.datasets.prepare import prepare_dataset
from icmr2027.datasets.ravenea import DATASET_REVISION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/ravenea")
    parser.add_argument("--manifest", default="data/ravenea_manifest.json")
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--archive", help="Optional already downloaded official ZIP")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = prepare_dataset(resolve_path(args.data_path, ROOT), resolve_path(args.manifest, ROOT),
                               ROOT / ".cache/ravenea", args.force, args.validate_only,
                               resolve_path(args.archive, ROOT) if args.archive else None, args.revision)
    print("Dataset root:", resolve_path(args.data_path, ROOT))
    print("Manifest:", resolve_path(args.manifest, ROOT))
    print(json.dumps({key: value for key, value in manifest["integrity"].items() if key != "cvqa_images"}, indent=2))


if __name__ == "__main__":
    main()
