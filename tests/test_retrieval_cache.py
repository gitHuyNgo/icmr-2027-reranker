from copy import deepcopy
from pathlib import Path

import pytest

from icmr2027.datasets.ravenea import load_cvqa_records
from icmr2027.retrieval.cache import rank_candidates, top1_context, validate_cache
from icmr2027.retrieval.corpus import WikipediaCorpus
from icmr2027.utils.io import read_jsonl, write_jsonl

FIXTURE = Path(__file__).parent / "fixtures/ravenea"


def cache_rows(samples):
    return [{"sample_id": sample["id"], "file_name": sample["file_name"],
             "candidates": rank_candidates(sample["candidate_doc_ids"], list(range(10, 0, -1)))}
            for sample in samples]


def test_cache_round_trip_joins_and_top1_context(tmp_path):
    samples = load_cvqa_records(FIXTURE)
    corpus = WikipediaCorpus(FIXTURE / "wiki_documents.jsonl")
    rows = cache_rows(samples)
    path = tmp_path / "cache.jsonl"
    write_jsonl(path, rows)
    indexed = validate_cache(read_jsonl(path), samples, corpus)
    context, info = top1_context(samples[0]["id"], indexed, corpus)
    assert info == {"retrieved_doc_id": "enwiki/1", "retrieval_rank": 1, "retrieval_score": 10.0}
    assert context == "Document 1 is a unit test."
    assert all("text" not in candidate for row in rows for candidate in row["candidates"])


def test_tie_order_preserves_official_candidate_order():
    ranked = rank_candidates(["enwiki/2", "enwiki/1"], [0.5, 0.5])
    assert [x["doc_id"] for x in ranked] == ["enwiki/2", "enwiki/1"]


@pytest.mark.parametrize("mutation", ["missing_query", "duplicate_doc", "unknown_doc", "bad_rank", "nonfinite", "wrong_image"])
def test_invalid_cache_is_rejected(mutation):
    samples = load_cvqa_records(FIXTURE)
    corpus = WikipediaCorpus(FIXTURE / "wiki_documents.jsonl")
    rows = deepcopy(cache_rows(samples))
    if mutation == "missing_query": rows.pop()
    if mutation == "duplicate_doc": rows[0]["candidates"][1]["doc_id"] = "enwiki/1"
    if mutation == "unknown_doc": rows[0]["candidates"][0]["doc_id"] = "enwiki/missing"
    if mutation == "bad_rank": rows[0]["candidates"][0]["rank"] = 0
    if mutation == "nonfinite": rows[0]["candidates"][0]["score"] = float("nan")
    if mutation == "wrong_image": rows[0]["file_name"] = "different-image"
    with pytest.raises(ValueError):
        validate_cache(rows, samples, corpus)
