from pathlib import Path

from PIL import Image
import pytest

from icmr2027.config import DatasetConfig
from icmr2027.datasets.base import create_dataset, SmokeDataset
from icmr2027.utils.io import write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_samples_are_loaded_rgb_images():
    dataset = SmokeDataset(ROOT / "data/smoke")
    assert len(dataset) == 3
    for sample in dataset:
        assert set(sample) == {"id", "image", "question", "answer"}
        assert sample["id"].startswith("smoke_")
        assert isinstance(sample["image"], Image.Image)
        assert sample["image"].mode == "RGB"
        assert sample["image"].getpixel((112, 112)) != (255, 255, 255)
    assert len(SmokeDataset(ROOT / "data/smoke", limit=1)) == 1
    assert len(SmokeDataset(ROOT / "data/smoke", limit=10)) == 3


def test_no_real_dataset_is_fabricated():
    with pytest.raises(ValueError, match="Batch 1"):
        create_dataset(DatasetConfig("unknown", "data", "test"), ROOT)


def test_rejects_paths_outside_dataset(tmp_path):
    write_jsonl(tmp_path / "samples.jsonl", [
        {"id": "smoke_001", "image": "../outside.png", "question": "Color?", "answer": "red"}
    ])
    with pytest.raises(ValueError, match="inside"):
        SmokeDataset(tmp_path)


def test_rejects_duplicate_ids(tmp_path):
    Image.new("RGB", (8, 8), "red").save(tmp_path / "image.png")
    record = {"id": "same", "image": "image.png", "question": "Color?", "answer": "red"}
    write_jsonl(tmp_path / "samples.jsonl", [record, record])
    with pytest.raises(ValueError, match="Duplicate"):
        SmokeDataset(tmp_path)
