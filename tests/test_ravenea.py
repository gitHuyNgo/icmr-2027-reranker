import json
from pathlib import Path
import shutil

import pytest

from icmr2027.datasets.ravenea import (
    RaveneaCVQADataset, file_sha256, image_path, load_cvqa_records, select_pilot,
)
from icmr2027.datasets.prepare import inspect_dataset, make_manifest, verify_manifest
from icmr2027.utils.io import write_json

FIXTURE = Path(__file__).parent / "fixtures/ravenea"


def test_released_schema_is_flattened_without_changing_official_ids():
    records = load_cvqa_records(FIXTURE)
    assert len(records) == 3
    assert records[0]["id"] == "./ravenea/images/fixture_001.png_0"
    assert records[1]["id"] == "./ravenea/images/fixture_001.png_1"
    assert records[0]["answer"] == "A"
    assert records[0]["options"][0] == "A. Red"
    assert records[0]["category"] == "Test fixture"
    assert records[0]["raw_metadata"]["generate_caption"].startswith("UNIT TEST")


def test_pilot_is_seeded_persisted_and_independent_of_row_order(tmp_path):
    records = load_cvqa_records(FIXTURE)
    fingerprint = file_sha256(FIXTURE / "cvqa_downstream.jsonl")
    path = tmp_path / "pilot.json"
    first = select_pilot(records, 42, 2, path, fingerprint)
    second = select_pilot(list(reversed(records)), 42, 2, path, fingerprint)
    assert [x["id"] for x in first] == [x["id"] for x in second]
    assert json.loads(path.read_text())["sample_ids"] == [x["id"] for x in first]
    with pytest.raises(ValueError, match="mismatch"):
        select_pilot(records, 42, 1, path, fingerprint)


def test_local_adapter_loads_rgb_without_network(tmp_path):
    dataset = RaveneaCVQADataset(FIXTURE, 42, 3, tmp_path / "pilot.json")
    assert len(dataset) == 3
    assert all(sample["image"].mode == "RGB" for sample in dataset)
    assert dataset.sample_ids == [record["id"] for record in dataset.records]


@pytest.mark.parametrize("filename", ["../outside.png", "./ravenea/images/../../outside.png", "C:/outside.png"])
def test_image_paths_cannot_escape_dataset(filename):
    with pytest.raises(ValueError):
        image_path(FIXTURE, filename)


def test_integrity_and_manifest_hash_mismatch(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(FIXTURE, data)
    report = inspect_dataset(data)
    assert report["num_cvqa_examples"] == 3
    assert report["num_unique_cvqa_images"] == 2
    assert report["num_wikipedia_documents"] == 10
    assert report["num_missing_images"] == 0
    manifest = tmp_path / "manifest.json"
    write_json(manifest, make_manifest(data, "test-only"))
    verify_manifest(data, manifest)
    with (data / "wiki_documents.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_manifest(data, manifest)


def test_unaligned_questions_options_answers_fail(tmp_path):
    row = json.loads((FIXTURE / "cvqa_downstream.jsonl").read_text().splitlines()[0])
    row["answers"] = []
    (tmp_path / "cvqa_downstream.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Unaligned"):
        load_cvqa_records(tmp_path)
