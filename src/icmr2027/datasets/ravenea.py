"""Released RAVENEA cVQA schema and official per-question IDs."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
from typing import Any

from PIL import Image

from icmr2027.datasets.base import MultimodalQADataset
from icmr2027.utils.io import read_jsonl

OFFICIAL_SOURCE_COMMIT = "b7b3c290b3ad2c9caf17f28fefebfd9b912170ec"
DATASET_REVISION = "ff5a212a9bfa1515f82e0930b37b7d64e3e9ee2e"
DATASET_ID = "jaagli/ravenea"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def image_path(root: Path, file_name: str) -> Path:
    # Released paths are './ravenea/images/<name>.jpg'; preserve these in IDs.
    if not isinstance(file_name, str) or "\\" in file_name or ":" in file_name:
        raise ValueError("Invalid RAVENEA image path")
    parts = PurePosixPath(file_name).parts
    if parts and parts[0] == "ravenea":
        parts = parts[1:]
    if not parts or parts[0] != "images" or ".." in parts:
        raise ValueError(f"Expected a local images/ path: {file_name}")
    result = root.joinpath(*parts).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Image path escapes dataset root")
    return result


def load_cvqa_records(root: Path) -> list[dict[str, Any]]:
    examples = []
    seen = set()
    for raw in read_jsonl(root / "cvqa_downstream.jsonl"):
        if raw.get("task_type") != "cVQA":
            raise ValueError("Expected released cVQA records only")
        filename = raw["file_name"]
        image_path(root, filename)
        questions, options, answers = raw["questions"], raw["options"], raw["answers"]
        if not all(isinstance(x, list) for x in (questions, options, answers)) or not (
            len(questions) == len(options) == len(answers) and len(questions) > 0
        ):
            raise ValueError(f"Unaligned cVQA questions/options/answers: {filename}")
        candidates = raw.get("enwiki_ids")
        if not isinstance(candidates, list) or not candidates or not all(isinstance(x, str) for x in candidates):
            raise ValueError(f"Missing released enwiki_ids: {filename}")
        if len(set(candidates)) != len(candidates):
            raise ValueError(f"Duplicate candidate document IDs: {filename}")
        for index, (question, choices, answer) in enumerate(zip(questions, options, answers)):
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Empty cVQA question")
            if not isinstance(choices, list) or len(choices) != 4 or not all(
                isinstance(choice, str) and choice.startswith(f"{label}. ")
                for label, choice in zip("ABCD", choices)
            ):
                raise ValueError(f"Expected four released labelled options: {filename}:{index}")
            if not isinstance(answer, str) or len(answer) != 1 or answer not in "ABCD":
                raise ValueError(f"Expected uppercase answer label A-D: {filename}:{index}")
            identifier = f"{filename}_{index}"
            if identifier in seen:
                raise ValueError(f"Duplicate cVQA question ID: {identifier}")
            seen.add(identifier)
            record = {"id": identifier, "file_name": filename, "question_index": index,
                      "question": question, "answer": answer, "options": choices,
                      "candidate_doc_ids": candidates, "source": "ravenea_cvqa",
                      "raw_metadata": raw}
            for key in ("country", "category"):
                if key in raw:
                    record[key] = raw[key]
            examples.append(record)
    if not examples:
        raise ValueError("No released cVQA questions found")
    return examples


def select_pilot(records: list[dict], seed: int, limit: int, path: Path, dataset_hash: str) -> list[dict]:
    if type(limit) is not int or not 0 < limit <= len(records):
        raise ValueError(f"Pilot limit must be between 1 and {len(records)}")
    indexed = {record["id"]: record for record in records}
    # Sorting makes selection independent of JSONL row order.
    ids = random.Random(seed).sample(sorted(indexed), limit)
    expected = {"sample_seed": seed, "limit": limit, "cvqa_sha256": dataset_hash,
                "sample_ids": ids, "official_source_commit": OFFICIAL_SOURCE_COMMIT}
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Pilot artifact mismatch: {path}; use a new pilot_path for changed data/settings")
    return [indexed[identifier] for identifier in ids]


class RaveneaCVQADataset(MultimodalQADataset):
    def __init__(self, root: Path, seed: int, limit: int, pilot_path: Path):
        self.root = root.resolve()
        self.cvqa_sha256 = file_sha256(self.root / "cvqa_downstream.jsonl")
        self.records = select_pilot(load_cvqa_records(self.root), seed, limit, pilot_path, self.cvqa_sha256)
        self.sample_ids = [record["id"] for record in self.records]
        self.pilot_path = pilot_path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(image_path(self.root, record["file_name"])) as image:
            rgb = image.convert("RGB")
        return {**record, "image": rgb}


def format_cvqa_question(sample: dict) -> str:
    return "Question: " + sample["question"] + "\n\n" + "\n".join(sample["options"])
