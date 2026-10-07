from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from icmr2027.utils.io import create_run_directory, read_jsonl, write_json, write_jsonl


def test_jsonl_round_trip(tmp_path):
    records = [{"id": "smoke_001", "question": "Màu gì?", "ground_truth": "red",
                "prediction": "red", "correct": True}]
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, records)
    assert read_jsonl(path) == records
    assert "Màu gì?" in path.read_text(encoding="utf-8")
    write_jsonl(path, [])
    assert read_jsonl(path) == []


@pytest.mark.parametrize("body", ["{invalid}\n", "[]\n"])
def test_invalid_jsonl_reports_line(tmp_path, body):
    path = tmp_path / "invalid.jsonl"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=":1:"):
        read_jsonl(path)


def test_unique_run_directories_preserve_previous_artifacts(tmp_path):
    root = tmp_path / "outputs"
    first = create_run_directory(root)
    write_json(first / "metrics.json", {"accuracy": 0.5})
    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(lambda _: create_run_directory(root), range(20)))
    assert len(set(runs + [first])) == 21
    assert all(run.is_dir() for run in runs)
    assert json.loads((first / "metrics.json").read_text()) == {"accuracy": 0.5}
