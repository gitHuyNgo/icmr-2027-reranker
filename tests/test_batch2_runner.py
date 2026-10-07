"""Paired pipeline wiring with explicit test stubs; never real inference results."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import pytest
import yaml

from scripts import run_baseline as runner
from icmr2027.config import load_config
from icmr2027.datasets.base import create_dataset
from icmr2027.datasets.prepare import make_manifest
from icmr2027.datasets.ravenea import OFFICIAL_SOURCE_COMMIT, file_sha256, object_sha256
from icmr2027.evaluation.comparison import compare_runs
from icmr2027.retrieval.cache import rank_candidates
from icmr2027.utils.io import read_jsonl, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures/ravenea"


@pytest.fixture
def paired_setup(tmp_path, monkeypatch):
    data = tmp_path / "data/ravenea"
    shutil.copytree(FIXTURE, data)
    manifest = make_manifest(data, "test-only")
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    config_paths = []
    for mode in ("no_rag", "top1_rag"):
        raw = load_config(ROOT / f"configs/baseline/batch2_ravenea_{mode}.yaml").to_dict()
        raw["dataset"].update(data_path=str(data), manifest_path=str(manifest_path), limit=3,
                              pilot_path=str(tmp_path / "pilot.json"))
        raw["experiment"]["output_root"] = str(tmp_path / mode)
        if mode == "top1_rag": raw["rag"]["retrieval_path"] = str(tmp_path / "cache/top10.jsonl")
        path = tmp_path / f"{mode}.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        config_paths.append(path)
    right_config = load_config(config_paths[1])
    dataset = create_dataset(right_config.dataset, ROOT)
    cache_path = Path(right_config.rag.retrieval_path)
    cache_path.parent.mkdir()
    rows = [{"sample_id": x["id"], "file_name": x["file_name"],
             "retriever": right_config.retrieval.name,
             "candidates": rank_candidates(x["candidate_doc_ids"], list(range(10, 0, -1)))}
            for x in dataset.records]
    write_jsonl(cache_path, rows)
    write_json(cache_path.parent / "retrieval_meta.json", {
        "status": "completed", "cvqa_sha256": dataset.cvqa_sha256,
        "wiki_sha256": manifest["files"]["wiki_documents.jsonl"]["sha256"],
        "sample_ids_sha256": object_sha256(dataset.sample_ids), "top_k": 10,
        "cache_sha256": file_sha256(cache_path), "retrieval_config": asdict(right_config.retrieval),
        "retriever_model_id": right_config.retrieval.name,
        "resolved_model_revision": right_config.retrieval.revision, "candidate_scope": "official_enwiki_ids",
        "query_modality": "image", "official_source_commit": OFFICIAL_SOURCE_COMMIT,
    })
    monkeypatch.setattr(runner, "set_seed", lambda _: None)
    monkeypatch.setattr(runner, "collect_run_metadata", lambda *args: {"timestamp": "test-only"})
    return config_paths, tmp_path


def test_same_samples_shared_runner_and_comparison_without_network(paired_setup, monkeypatch):
    paths, output = paired_setup
    seen = {"none": [], "top1": []}

    class ExplicitTestStub:
        def __init__(self, config):
            self.resolved_revision = config.revision

        def generate(self, image, question, context=None):
            assert image.mode == "RGB" and "A. Red" in question
            mode = "none" if context is None else "top1"
            seen[mode].append(question)
            if context is not None:
                assert "Document 1 is a unit test." in context
            return "Answer: A"

    monkeypatch.setattr(runner, "HuggingFaceVLM", ExplicitTestStub)
    left, right = [runner.run_baseline(path, ROOT) for path in paths]
    rows_left, rows_right = read_jsonl(left / "predictions.jsonl"), read_jsonl(right / "predictions.jsonl")
    assert {x["id"] for x in rows_left} == {x["id"] for x in rows_right}
    assert seen["none"] == seen["top1"]
    assert all(x["retrieved_doc_id"] == "enwiki/1" and x["context_chars"] > 0 for x in rows_right)
    comparison = compare_runs(left, right, output / "comparisons")
    summary = json.loads((comparison / "summary.json").read_text())
    assert summary["num_samples"] == 3
    assert summary["delta_accuracy"] == 0
    # Scientific settings mismatch must fail before a new comparison directory is allocated.
    meta = json.loads((right / "run_meta.json").read_text())
    meta["comparison_identity"]["seed"] = 99
    write_json(right / "run_meta.json", meta)
    with pytest.raises(ValueError, match="settings differ"):
        compare_runs(left, right, output / "rejected_comparison")
    assert not (output / "rejected_comparison").exists()


def test_missing_retrieval_cache_fails_before_loading_vlm(paired_setup, monkeypatch):
    paths, output = paired_setup
    config = load_config(paths[1])
    Path(config.rag.retrieval_path).unlink()
    def forbidden_model_load(config):
        pytest.fail("VLM must not load with missing retrieval prerequisites")
    monkeypatch.setattr(runner, "HuggingFaceVLM", forbidden_model_load)
    with pytest.raises(FileNotFoundError):
        runner.run_baseline(paths[1], ROOT)
    run = next((output / "top1_rag").iterdir())
    assert json.loads((run / "run_meta.json").read_text())["status"] == "failed"
    assert read_jsonl(run / "predictions.jsonl") == []


def test_offline_retrieval_builder_caches_by_image_and_reuses_checked_results(paired_setup, monkeypatch):
    # Explicit synthetic retriever stub; generated artifacts stay in pytest tmp_path.
    from icmr2027.retrieval import build, clip
    paths, output = paired_setup
    config = load_config(paths[1])
    calls = {"load": 0, "retrieve": 0}

    class TestOnlyRetriever:
        def __init__(self, config, corpus):
            calls["load"] += 1
            self.resolved_revision = config.revision

        def prepare_corpus(self, cache_root, wiki_hash, force):
            write_json(cache_root / "corpus_index.json", {"test_only": True})

        def retrieve(self, sample, top_k):
            calls["retrieve"] += 1
            return [{**candidate, "text": "TEST STUB ONLY"}
                    for candidate in rank_candidates(sample["candidate_doc_ids"], list(range(10, 0, -1)))]

    monkeypatch.setattr(clip, "RaveneaCLIPRetriever", TestOnlyRetriever)
    monkeypatch.setattr(build, "set_seed", lambda _: None)
    first = build.build_retrieval(config, ROOT, force=True)
    assert calls == {"load": 1, "retrieve": 2}
    assert len(read_jsonl(first)) == 3
    assert build.build_retrieval(config, ROOT) == first
    assert calls == {"load": 1, "retrieve": 2}
    with first.open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        build.build_retrieval(config, ROOT, validate_only=True)
