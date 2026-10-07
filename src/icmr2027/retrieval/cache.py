"""Validate cached ranks and dataset joins before any VLM generation."""
import json
import math
from pathlib import Path

from icmr2027.datasets.ravenea import OFFICIAL_SOURCE_COMMIT, file_sha256, object_sha256
from icmr2027.retrieval.corpus import WikipediaCorpus
from icmr2027.utils.io import read_jsonl


def rank_candidates(doc_ids: list[str], scores: list[float]) -> list[dict]:
    if len(doc_ids) != len(scores) or len(set(doc_ids)) != len(doc_ids):
        raise ValueError("Candidate IDs/scores must align and be unique")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Nonfinite retrieval score")
    # Stable sorting preserves official annotation order for exact score ties.
    ordered = sorted(zip(doc_ids, scores), key=lambda pair: pair[1], reverse=True)
    return [{"rank": rank, "doc_id": doc_id, "score": float(score)}
            for rank, (doc_id, score) in enumerate(ordered, 1)]


def validate_cache(rows: list[dict], samples: list[dict], corpus: WikipediaCorpus, top_k: int = 10) -> dict:
    indexed = {}
    samples_by_id = {sample["id"]: sample for sample in samples}
    for row in rows:
        identifier = row["sample_id"]
        if identifier in indexed or identifier not in samples_by_id:
            raise ValueError("Duplicate or unknown sample ID in retrieval cache")
        sample = samples_by_id[identifier]
        if row.get("file_name") != sample["file_name"]:
            raise ValueError("Retrieval query image does not match dataset sample")
        candidates = row["candidates"]
        if len(candidates) != top_k:
            raise ValueError(f"Expected exactly {top_k} cached candidates")
        seen = set()
        previous = math.inf
        for rank, candidate in enumerate(candidates, 1):
            doc = candidate["doc_id"]
            score = candidate["score"]
            if candidate["rank"] != rank or doc in seen or doc not in corpus.documents:
                raise ValueError("Invalid ranks, duplicate or unknown document in retrieval cache")
            if doc not in sample["candidate_doc_ids"]:
                raise ValueError("Retrieved document is outside the official image candidate pool")
            if type(score) not in {int, float} or not math.isfinite(score) or score > previous:
                raise ValueError("Retrieval scores must be finite and descending")
            seen.add(doc)
            previous = score
        indexed[identifier] = row
    if set(indexed) != set(samples_by_id):
        raise ValueError("Retrieval sample IDs must equal the persisted pilot ID set")
    return indexed


def load_cache(path: Path, samples: list[dict], corpus: WikipediaCorpus,
               cvqa_hash: str, wiki_hash: str, retriever_config=None) -> tuple[dict, dict]:
    meta_path = path.parent / "retrieval_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "completed":
        raise ValueError("Retrieval cache is not completed")
    if (meta.get("top_k") != 10 or meta.get("query_modality") != "image"
            or meta.get("candidate_scope") != "official_enwiki_ids"
            or meta.get("official_source_commit") != OFFICIAL_SOURCE_COMMIT):
        raise ValueError("Retrieval cache protocol/source metadata mismatch")
    if meta.get("cvqa_sha256") != cvqa_hash or meta.get("wiki_sha256") != wiki_hash:
        raise ValueError("Retrieval cache dataset fingerprint mismatch")
    if meta.get("sample_ids_sha256") != object_sha256([x["id"] for x in samples]):
        raise ValueError("Retrieval cache pilot fingerprint mismatch")
    if meta.get("cache_sha256") != file_sha256(path):
        raise ValueError("Retrieval cache file checksum mismatch")
    if retriever_config is not None:
        from dataclasses import asdict
        if meta.get("retrieval_config") != asdict(retriever_config):
            raise ValueError("Retrieval cache configuration mismatch")
    rows = read_jsonl(path)
    if any(row.get("retriever") != meta.get("retriever_model_id") for row in rows):
        raise ValueError("Retrieval cache row/checkpoint metadata mismatch")
    return validate_cache(rows, samples, corpus, meta["top_k"]), meta


def top1_context(sample_id: str, cache: dict, corpus: WikipediaCorpus) -> tuple[str, dict]:
    candidate = cache[sample_id]["candidates"][0]
    return corpus.context(candidate["doc_id"]), {
        "retrieved_doc_id": candidate["doc_id"], "retrieval_rank": candidate["rank"],
        "retrieval_score": candidate["score"],
    }
