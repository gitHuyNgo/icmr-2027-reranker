"""Small cVQA hooks used by the existing experiment runner."""
from dataclasses import asdict
from pathlib import Path

from icmr2027.config import resolve_path
from icmr2027.datasets.prepare import verify_manifest
from icmr2027.datasets.ravenea import OFFICIAL_SOURCE_COMMIT, file_sha256, object_sha256
from icmr2027.evaluation.ravenea import cvqa_correct, parse_cvqa_answer
from icmr2027.retrieval.cache import load_cache, top1_context
from icmr2027.retrieval.corpus import WikipediaCorpus
from icmr2027.utils.run_metadata import package_version


def prepare_cvqa_run(config, dataset, root: Path) -> tuple[dict, object, dict]:
    data_root = resolve_path(config.dataset.data_path, root)
    manifest_path = resolve_path(config.dataset.manifest_path, root)
    manifest = verify_manifest(data_root, manifest_path)
    wiki_hash = manifest["files"]["wiki_documents.jsonl"]["sha256"]
    identity = {"model": asdict(config.model), "evaluation": asdict(config.evaluation),
                "seed": config.experiment.seed, "sample_seed": config.dataset.sample_seed,
                "cvqa_sha256": dataset.cvqa_sha256, "wiki_sha256": wiki_hash,
                "cvqa_image_inventory_sha256": manifest["integrity"]["cvqa_image_inventory_sha256"],
                "sample_ids_sha256": object_sha256(dataset.sample_ids),
                "official_source_commit": OFFICIAL_SOURCE_COMMIT,
                "inference_backend": "transformers",
                "software_versions": {name: package_version(name)
                                      for name in ("torch", "torchvision", "transformers")}}
    metadata = {"comparison_identity": identity, "sample_ids": dataset.sample_ids,
                "dataset_revision": manifest["dataset_revision"], "pilot_only": True,
                "dataset_manifest_sha256": file_sha256(manifest_path),
                "official_source_commit": OFFICIAL_SOURCE_COMMIT}
    corpus, cache = None, {}
    if config.rag.mode == "top1":
        corpus = WikipediaCorpus(data_root / "wiki_documents.jsonl")
        cache_path = resolve_path(config.rag.retrieval_path, root)
        cache, retrieval_meta = load_cache(cache_path, dataset.records, corpus,
                                          dataset.cvqa_sha256, wiki_hash, config.retrieval)
        metadata.update(retrieval_cache_sha256=file_sha256(cache_path),
                        retrieval_meta_sha256=file_sha256(cache_path.parent / "retrieval_meta.json"),
                        retriever_model_id=retrieval_meta["retriever_model_id"],
                        retriever_revision=retrieval_meta["resolved_model_revision"],
                        candidate_scope=retrieval_meta["candidate_scope"])
    return metadata, corpus, cache


def cvqa_context(sample_id: str, mode: str, corpus, cache: dict) -> tuple[str | None, dict]:
    if mode == "none":
        return None, {}
    context, candidate = top1_context(sample_id, cache, corpus)
    return context, {**candidate, "context_chars": len(context)}


def cvqa_prediction(sample: dict, raw: str, mode: str, evidence: dict) -> dict:
    record = {"id": sample["id"], "file_name": sample["file_name"],
              "question_index": sample["question_index"], "question": sample["question"],
              "options": sample["options"], "ground_truth": sample["answer"],
              "raw_prediction": raw, "parsed_prediction": parse_cvqa_answer(raw),
              "correct": cvqa_correct(raw, sample["answer"]), "rag_mode": mode,
              "source": "ravenea_cvqa", "pilot_only": True, **evidence}
    for key in ("country", "category"):
        if key in sample:
            record[key] = sample[key]
    return record
