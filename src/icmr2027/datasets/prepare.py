"""Safe explicit data preparation and integrity manifests (never invoked by inference)."""
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

from PIL import Image

from icmr2027.datasets.ravenea import (
    DATASET_ID, DATASET_REVISION, OFFICIAL_SOURCE_COMMIT, file_sha256,
    image_path, load_cvqa_records, object_sha256,
)
from icmr2027.utils.io import read_jsonl, write_json


REQUIRED = ("images", "cvqa_downstream.jsonl", "wiki_documents.jsonl")


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            path = PurePosixPath(entry.filename)
            target = destination.joinpath(*path.parts).resolve()
            if (path.is_absolute() or ".." in path.parts or "\\" in entry.filename
                    or ":" in entry.filename or not target.is_relative_to(destination)
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError(f"Unsafe ZIP entry: {entry.filename}")
        zipped.extractall(destination)


def inspect_dataset(root: Path) -> dict:
    for name in REQUIRED:
        if not (root / name).exists():
            raise FileNotFoundError(f"Missing required dataset input: {root / name}")
    records = load_cvqa_records(root)
    documents = read_jsonl(root / "wiki_documents.jsonl")
    document_ids = [document["id"] for document in documents]
    duplicates = len(document_ids) - len(set(document_ids))
    invalid_docs = sum(not isinstance(x.get("id"), str) or not isinstance(x.get("text"), str) for x in documents)
    filenames = sorted({record["file_name"] for record in records})
    missing = []
    inventory = []
    for filename in filenames:
        path = image_path(root, filename)
        if not path.is_file():
            missing.append(filename)
            continue
        with Image.open(path) as image:
            image.verify()
        inventory.append({"file_name": filename, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    missing_doc_ids = sorted({doc for record in records for doc in record["candidate_doc_ids"]} - set(document_ids))
    raw = read_jsonl(root / "cvqa_downstream.jsonl")
    metadata_counts = {}
    for name in ("metadata.jsonl", "metadata_train.jsonl", "metadata_val.jsonl", "metadata_test.jsonl"):
        if (root / name).is_file():
            metadata_counts[name] = len(read_jsonl(root / name))
    if (root / "metadata_test.jsonl").is_file():
        test_names = {row["file_name"] for row in read_jsonl(root / "metadata_test.jsonl")}
        if not set(filenames) <= test_names:
            raise ValueError("Released cVQA images are not a subset of metadata_test")
    report = {"num_cvqa_rows": len(raw), "num_cvqa_examples": len(records),
              "num_unique_cvqa_images": len(filenames), "num_wikipedia_documents": len(documents),
              "num_missing_images": len(missing), "num_duplicate_document_ids": duplicates,
              "num_invalid_documents": invalid_docs, "num_missing_candidate_doc_ids": len(missing_doc_ids),
              "cvqa_keys": list(raw[0]), "document_keys": list(documents[0]),
              "metadata_counts": metadata_counts,
              "cvqa_image_inventory_sha256": object_sha256(inventory), "cvqa_images": inventory}
    if missing or duplicates or invalid_docs or missing_doc_ids:
        raise ValueError(f"Dataset integrity failed: missing_images={len(missing)}, duplicate_docs={duplicates}, "
                         f"invalid_docs={invalid_docs}, missing_candidate_docs={len(missing_doc_ids)}")
    return report


def make_manifest(root: Path, revision: str | None, archive: Path | None = None) -> dict:
    report = inspect_dataset(root)
    critical = {}
    for name in ("cvqa_downstream.jsonl", "wiki_documents.jsonl", "metadata_test.jsonl"):
        path = root / name
        if path.is_file():
            critical[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    manifest = {"dataset": DATASET_ID, "dataset_revision": revision,
                "official_source_commit": OFFICIAL_SOURCE_COMMIT,
                "prepared_at": datetime.now(timezone.utc).isoformat(),
                "files": critical, "integrity": report}
    if archive is not None:
        manifest["archive"] = {"filename": "ravenea.zip", "bytes": archive.stat().st_size,
                               "sha256": file_sha256(archive)}
    return manifest


def verify_manifest(root: Path, manifest_path: Path, verify_images: bool = True) -> dict:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest {manifest_path}; run scripts/prepare_ravenea.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        if file_sha256(root / name) != info["sha256"]:
            raise ValueError(f"Dataset file hash mismatch: {name}; prepare/validate data again")
    if verify_images:
        for info in manifest["integrity"]["cvqa_images"]:
            if file_sha256(image_path(root, info["file_name"])) != info["sha256"]:
                raise ValueError(f"Image hash mismatch: {info['file_name']}")
    return manifest


def prepare_dataset(root: Path, manifest_path: Path, cache_root: Path, force: bool = False,
                    validate_only: bool = False, archive: Path | None = None,
                    revision: str = DATASET_REVISION) -> dict:
    root = root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    complete = all((root / name).exists() for name in REQUIRED)
    if validate_only or (complete and not force):
        previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
        manifest = make_manifest(root, previous.get("dataset_revision") if previous else None)
        if previous and previous.get("files") == manifest["files"]:
            if "archive" in previous:
                manifest["archive"] = previous["archive"]
        elif previous:
            manifest["dataset_revision"] = None
    else:
        if archive is None:
            from huggingface_hub import hf_hub_download
            archive = Path(hf_hub_download(repo_id=DATASET_ID, repo_type="dataset", filename="ravenea.zip",
                                          revision=revision, local_dir=cache_root / "download",
                                          force_download=force))
        with tempfile.TemporaryDirectory(prefix="ravenea-", dir=cache_root) as temporary:
            staging = Path(temporary).resolve()
            safe_extract(archive, staging)
            candidates = [p.parent for p in staging.rglob("cvqa_downstream.jsonl")
                          if all((p.parent / name).exists() for name in REQUIRED)]
            if len(candidates) != 1:
                raise ValueError(f"Archive must contain one valid RAVENEA root; found {len(candidates)}")
            source = candidates[0].resolve()
            if not source.is_relative_to(staging):
                raise ValueError("Dataset root escapes extraction staging directory")
            manifest = make_manifest(source, revision, archive)
            root.parent.mkdir(parents=True, exist_ok=True)
            if root.exists():
                # Preserve old/partial data rather than recursively deleting it.
                backup = root.with_name(root.name + ".previous_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
                if not backup.resolve().is_relative_to(root.parent.resolve()):
                    raise ValueError("Backup path escapes dataset parent")
                root.rename(backup)
                print(f"Preserved previous dataset: {backup}", flush=True)
            shutil.move(str(source), str(root))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    return manifest
