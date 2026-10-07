"""Offline cache generation, independent of VLM inference."""
from dataclasses import asdict
from pathlib import Path

from icmr2027.config import resolve_path
from icmr2027.datasets.base import create_dataset
from icmr2027.datasets.prepare import verify_manifest
from icmr2027.datasets.ravenea import OFFICIAL_SOURCE_COMMIT, file_sha256, object_sha256
from icmr2027.retrieval.cache import load_cache, validate_cache
from icmr2027.retrieval.corpus import WikipediaCorpus
from icmr2027.utils.io import write_json, write_jsonl
from icmr2027.utils.reproducibility import set_seed
from icmr2027.utils.run_metadata import collect_run_metadata, utc_timestamp


def build_retrieval(config, root: Path, force: bool = False, validate_only: bool = False) -> Path:
    if config.retrieval is None or config.rag.retrieval_path is None:
        raise ValueError("Use the Batch 2 Top-1 config containing retrieval settings")
    data_root = resolve_path(config.dataset.data_path, root)
    manifest_path = resolve_path(config.dataset.manifest_path, root)
    manifest = verify_manifest(data_root, manifest_path)
    dataset = create_dataset(config.dataset, root)
    corpus = WikipediaCorpus(data_root / "wiki_documents.jsonl")
    wiki_hash = manifest["files"]["wiki_documents.jsonl"]["sha256"]
    cache_path = resolve_path(config.rag.retrieval_path, root)
    if validate_only or (cache_path.exists() and not force):
        load_cache(cache_path, dataset.records, corpus, dataset.cvqa_sha256, wiki_hash, config.retrieval)
        print(f"Validated existing retrieval cache: {cache_path}; {len(dataset)} queries", flush=True)
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache_path.parent / "retrieval_meta.json"
    meta = collect_run_metadata(root, config.experiment.seed, config.retrieval.name, config.retrieval.device)
    meta.update({"status": "running", "retriever_model_id": config.retrieval.name,
                 "retrieval_config": asdict(config.retrieval), "cvqa_sha256": dataset.cvqa_sha256,
                 "wiki_sha256": wiki_hash, "dataset_manifest_sha256": file_sha256(manifest_path),
                 "sample_ids_sha256": object_sha256(dataset.sample_ids), "sample_ids": dataset.sample_ids,
                 "top_k": config.retrieval.top_k, "query_modality": "image",
                 "candidate_scope": "official_enwiki_ids", "official_source_commit": OFFICIAL_SOURCE_COMMIT})
    write_json(meta_path, meta)
    try:
        set_seed(config.experiment.seed)
        # Import only for actual retrieval; cache validation stays CPU/model-free.
        from icmr2027.retrieval.clip import RaveneaCLIPRetriever
        retriever = RaveneaCLIPRetriever(config.retrieval, corpus)
        retriever.prepare_corpus(cache_path.parent, wiki_hash, force)
        meta["resolved_model_revision"] = retriever.resolved_revision
        if retriever.resolved_revision not in {None, config.retrieval.revision}:
            raise ValueError("Loaded retriever revision differs from pinned config")
        meta["corpus_index_sha256"] = file_sha256(cache_path.parent / "corpus_index.json")
        rows = []
        by_image = {}
        for sample in dataset:
            filename = sample["file_name"]
            if filename not in by_image:
                # No question text, answer, culture labels or generated caption enters retrieval.
                ranked = retriever.retrieve(sample, config.retrieval.top_k)
                by_image[filename] = [{key: candidate[key] for key in ("rank", "doc_id", "score")}
                                      for candidate in ranked]
            rows.append({"sample_id": sample["id"], "file_name": filename,
                         "retriever": config.retrieval.name, "candidates": by_image[filename]})
        validate_cache(rows, dataset.records, corpus, config.retrieval.top_k)
        temporary = cache_path.with_suffix(".jsonl.tmp")
        write_jsonl(temporary, rows)
        temporary.replace(cache_path)
        meta.update(status="completed", cache_sha256=file_sha256(cache_path), num_queries=len(rows),
                    num_unique_query_images=len(by_image))
        print(f"Completed retrieval cache: {cache_path}; {len(rows)} queries", flush=True)
    except BaseException as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        meta["finished_at"] = utc_timestamp()
        write_json(meta_path, meta)
    return cache_path
