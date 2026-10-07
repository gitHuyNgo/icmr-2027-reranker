"""Normalized multimodal QA samples for smoke data and released RAVENEA cVQA."""
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

from PIL import Image

from icmr2027.config import DatasetConfig, resolve_path
from icmr2027.utils.io import read_jsonl


class QASample(TypedDict):
    id: str
    image: Image.Image
    question: str
    answer: str


class MultimodalQADataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> QASample:
        ...

    def __iter__(self) -> Iterator[QASample]:
        for index in range(len(self)):
            yield self[index]


class SmokeDataset(MultimodalQADataset):
    """Infrastructure-only synthetic samples. Metrics have no scientific meaning."""

    def __init__(self, data_path: Path, limit: int | None = None):
        self.data_path = data_path.resolve()
        self.records = read_jsonl(self.data_path / "samples.jsonl")
        seen: set[str] = set()
        for record in self.records:
            for key in ("id", "image", "question", "answer"):
                if not isinstance(record.get(key), str) or not record[key].strip():
                    raise ValueError(f"Smoke record needs a nonempty {key}")
            if record["id"] in seen:
                raise ValueError(f"Duplicate sample ID: {record['id']}")
            seen.add(record["id"])
            image_path = (self.data_path / record["image"]).resolve()
            if not image_path.is_relative_to(self.data_path):
                raise ValueError("Smoke image paths must stay inside the dataset directory")
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
        if limit is not None:
            if type(limit) is not int or limit <= 0:
                raise ValueError("limit must be a positive integer or None")
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError("Smoke dataset contains no samples")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> QASample:
        record = self.records[index]
        with Image.open(self.data_path / record["image"]) as image:
            rgb = image.convert("RGB")
        return QASample(id=record["id"], image=rgb,
                        question=record["question"], answer=record["answer"])


def create_dataset(config: DatasetConfig, repo_root: Path) -> MultimodalQADataset:
    if config.name == "ravenea_cvqa":
        from icmr2027.datasets.ravenea import RaveneaCVQADataset
        return RaveneaCVQADataset(resolve_path(config.data_path, repo_root), config.sample_seed,
                                 config.limit, resolve_path(config.pilot_path, repo_root))
    if config.name != "smoke" or config.split != "smoke":
        raise ValueError("Choose Batch 1 name=smoke, split=smoke or Batch 2 name=ravenea_cvqa, split=test")
    return SmokeDataset(resolve_path(config.data_path, repo_root), config.limit)
