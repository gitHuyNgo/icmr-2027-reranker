"""Immutable Batch 2 input validation and deterministic malformed-output audit."""
from collections import Counter
from dataclasses import asdict
import csv
import io
import json
from pathlib import Path
import re

from icmr2027.config import load_config, resolve_path
from icmr2027.datasets.prepare import verify_manifest
from icmr2027.datasets.ravenea import OFFICIAL_SOURCE_COMMIT, file_sha256, load_cvqa_records, object_sha256
from icmr2027.evaluation.comparison import pair_predictions
from icmr2027.evaluation.ravenea import build_cvqa_prompt, evaluate_cvqa
from icmr2027.retrieval.cache import load_cache, top1_context
from icmr2027.retrieval.corpus import WikipediaCorpus
from icmr2027.utils.io import read_jsonl


def save_artifacts(output: Path, artifacts: dict[str, str]) -> None:
    # Preflight every destination before writing. Identical reruns are idempotent.
    for name, content in artifacts.items():
        path = output / name
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite different audit results: {path}")
    for name, content in artifacts.items():
        path = output / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="") as handle:
                handle.write(content)


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def table_artifacts(stem: str, rows: list[dict], fields: list[str]) -> dict[str, str]:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False)
                         if isinstance(row.get(key), (list, dict)) else row.get(key) for key in fields})
    return {stem + ".csv": stream.getvalue(), stem + ".jsonl": "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)}


def load_original_pair(root: Path, runs: list[Path], expected_samples: int = 50) -> dict:
    configs = [load_config(run / "config.yaml") for run in runs]
    metas = [json.loads((run / "run_meta.json").read_text(encoding="utf-8")) for run in runs]
    rows = [read_jsonl(run / "predictions.jsonl") for run in runs]
    left, right = configs
    if (left.rag.mode, right.rag.mode) != ("none", "top1"):
        raise ValueError("Expected original No-RAG / Top-1 RAG saved configs")
    if (asdict(left.model), asdict(left.dataset), asdict(left.evaluation), left.experiment.seed) != (
            asdict(right.model), asdict(right.dataset), asdict(right.evaluation), right.experiment.seed):
        raise ValueError("Original model/data/prompt/generation/seed settings differ")
    if left.dataset.name != "ravenea_cvqa" or left.dataset.limit != expected_samples:
        raise ValueError("Expected the fixed original 50-question cVQA pilot")
    identity = metas[0].get("comparison_identity")
    if not identity or identity != metas[1].get("comparison_identity"):
        raise ValueError("Original comparison identities differ")
    if identity.get("model") != asdict(left.model) or identity.get("evaluation") != asdict(left.evaluation):
        raise ValueError("Saved configs do not match run identity")
    if identity.get("seed") != left.experiment.seed or identity.get("sample_seed") != left.dataset.sample_seed:
        raise ValueError("Saved seed does not match run identity")
    for meta in metas:
        if meta.get("status") != "completed" or meta.get("resolved_model_revision") != left.model.revision:
            raise ValueError("Original runs must be completed at the pinned VLM revision")
        if meta.get("official_source_commit") != OFFICIAL_SOURCE_COMMIT:
            raise ValueError("Official source revision mismatch")
    pilot_path = resolve_path(left.dataset.pilot_path, root)
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))  # Never create or resample.
    ids = pilot["sample_ids"]
    if (len(ids) != expected_samples or len(set(ids)) != expected_samples
            or pilot.get("sample_seed") != left.dataset.sample_seed
            or pilot.get("official_source_commit") != OFFICIAL_SOURCE_COMMIT):
        raise ValueError("Persisted pilot identity mismatch")
    if any(meta.get("sample_ids") != ids for meta in metas):
        raise ValueError("Run IDs/order differ from the original persisted pilot")
    if identity.get("sample_ids_sha256") != object_sha256(ids):
        raise ValueError("Pilot checksum differs from run identity")
    data_root = resolve_path(left.dataset.data_path, root)
    manifest_path = resolve_path(left.dataset.manifest_path, root)
    manifest = verify_manifest(data_root, manifest_path)
    cvqa_hash = manifest["files"]["cvqa_downstream.jsonl"]["sha256"]
    wiki_hash = manifest["files"]["wiki_documents.jsonl"]["sha256"]
    if pilot.get("cvqa_sha256") != cvqa_hash or identity.get("cvqa_sha256") != cvqa_hash:
        raise ValueError("cVQA checksum differs from original run")
    if (identity.get("wiki_sha256") != wiki_hash or identity.get("cvqa_image_inventory_sha256")
            != manifest["integrity"]["cvqa_image_inventory_sha256"]):
        raise ValueError("Original corpus or images changed")
    if any(meta.get("dataset_manifest_sha256") != file_sha256(manifest_path) for meta in metas):
        raise ValueError("Original dataset manifest changed")
    all_samples = {row["id"]: row for row in load_cvqa_records(data_root)}
    samples = [all_samples[identifier] for identifier in ids]
    for condition_rows in rows:
        if [row["id"] for row in condition_rows] != ids:
            raise ValueError("Predictions must retain every original sample in pilot order")
        for row in condition_rows:
            sample = all_samples[row["id"]]
            for key in ("question", "options", "file_name"):
                if row[key] != sample[key]:
                    raise ValueError("Prediction content differs from original dataset")
            if row["ground_truth"] != sample["answer"]:
                raise ValueError("Stored ground truth differs from dataset")
    before, _ = pair_predictions(*rows)
    for run, condition_rows in zip(runs, rows):
        saved = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        recomputed = evaluate_cvqa(condition_rows)
        if saved.get("status") != "completed" or any(saved.get(key) != recomputed[key]
                for key in ("num_samples", "accuracy", "num_malformed")):
            raise ValueError("Historical metrics disagree with raw predictions")
    corpus = WikipediaCorpus(data_root / "wiki_documents.jsonl")
    cache_path = resolve_path(right.rag.retrieval_path, root)
    cache, _ = load_cache(cache_path, samples, corpus, cvqa_hash, wiki_hash, right.retrieval)
    for key, path in (("retrieval_cache_sha256", cache_path),
                      ("retrieval_meta_sha256", cache_path.parent / "retrieval_meta.json")):
        if metas[1].get(key) != file_sha256(path):
            raise ValueError("Retrieval artifacts differ from original run")
    contexts = {}
    for row in rows[1]:
        context, candidate = top1_context(row["id"], cache, corpus)
        if any(row.get(key) != value for key, value in candidate.items()):
            raise ValueError("Top-1 document/rank/score changed since original generation")
        if row.get("context_chars") != len(context):
            raise ValueError("Effective evidence length changed since original generation")
        contexts[row["id"]] = context
    tracked = [pilot_path, manifest_path, data_root / "cvqa_downstream.jsonl",
               data_root / "wiki_documents.jsonl", cache_path, cache_path.parent / "retrieval_meta.json"]
    tracked += [run / name for run in runs for name in
                ("config.yaml", "run_meta.json", "predictions.jsonl", "metrics.json")]
    def relative(path):
        path = path.resolve()
        return path.relative_to(root.resolve()).as_posix() if path.is_relative_to(root.resolve()) else str(path)
    hashes = {relative(path): file_sha256(path) for path in tracked}
    return dict(rows=rows, runs=runs, ids=ids, before=before, contexts=contexts,
                hashes=hashes, root=root, configs=configs)


def diagnose(raw: str) -> tuple[str, str]:
    # Diagnostic rules never see options or ground truth and never score an answer.
    labels = set(re.findall(r"(?<![\w])[A-D](?![\w])", raw))
    if len(labels) > 1:
        return "ambiguous_multiple_options", "Multiple different standalone option labels; conservative review."
    if re.search(r"\b(?:context|document|prompt|truncat\w*|instruction)\b", raw, re.I):
        return "needs_manual_review", "Mentions prompt/evidence; causality cannot be inferred automatically."
    if re.search(r"\b(?:maybe|perhaps|could|might|cannot|unable|uncertain)\b", raw, re.I):
        return "no_option_identifiable", "Uncertainty without a safe single-option commitment."
    concise = r"(?:[A-D]\.?|\([A-D]\)|\[[A-D]\]|(?:Answer:\s*|(?:The )?(?:correct )?answer is\s+|Option\s+)\(?[A-D]\)?\.?)"
    if re.fullmatch(concise, raw.strip(), re.I) or re.fullmatch(r"[A-D]\.\s+[^\n]+", raw.strip()) and len(labels) == 1:
        return "parser_failure", "Explicit single label in a concise format missed by the official parser."
    if len(labels) == 1 and re.search(r"\b(?:answer\s*(?::|is)|option)\s+([A-D])(?!\w)", raw, re.I):
        return "verbose_but_identifiable", "Explanation with an explicit single-option commitment; inspect manually."
    if not labels:
        return "no_option_identifiable", "No explicit reliably identifiable option label."
    return "needs_manual_review", "A standalone label alone does not establish answer commitment."


def audit_pair(bundle: dict, output: Path, origin_root: str | None = None) -> dict:
    rows, context_rows = [], []
    for condition, records in zip(("no_rag", "top1_rag"), bundle["rows"]):
        for row in records:
            context = bundle["contexts"].get(row["id"]) if condition == "top1_rag" else None
            if context is not None:
                question = "Question: " + row["question"] + "\n\n" + "\n".join(row["options"])
                prompt = build_cvqa_prompt(question, context)
                context_rows.append({"sample_id": row["id"], "context_chars": len(context),
                    "context_sha256": object_sha256(context), "retrieved_doc_id": row["retrieved_doc_id"],
                    "retrieval_score": row["retrieval_score"], "prompt_chars": len(prompt),
                    "prompt_sha256": object_sha256(prompt), "question_options_intact": prompt.endswith(question),
                    "evidence_before_question": True, "format_instruction_after_evidence": False,
                    "metadata_inserted": False, "context_policy": "official heading removal and 256-word sentence rule",
                    "token_policy": "budget failure; no silent truncation",
                    "context_option_like_lines": bool(re.search(r"(?m)^\s*[A-D][.)]\s", context))})
            if row["parsed_prediction"] is not None:
                continue
            category, reason = diagnose(row["raw_prediction"])
            rows.append({"sample_id": row["id"], "condition": condition,
                "question": row["question"], "options": row["options"], "ground_truth": row["ground_truth"],
                "raw_prediction": row["raw_prediction"], "parsed_prediction": None,
                "parse_status": "malformed_official", "classification": category, "classification_reason": reason,
                "retrieved_doc_id": row.get("retrieved_doc_id") if context is not None else None,
                "retrieval_score": row.get("retrieval_score") if context is not None else None,
                "context_chars": len(context) if context is not None else 0,
                "retrieved_document_preview": context[:400] if context is not None else None})
    original_paths = {}
    for condition, run in zip(("no_rag", "top1_rag"), bundle["runs"]):
        original_paths[condition] = (origin_root.rstrip("/") + "/" + run.resolve().relative_to(
            bundle["root"].resolve()).as_posix()) if origin_root else str(run.resolve())
    report = {"status": "audited", "original_run_paths": original_paths,
              "sample_ids": bundle["ids"], "sample_ids_sha256": object_sha256(bundle["ids"]),
              "original_files_sha256": bundle["hashes"], "official_source_commit": OFFICIAL_SOURCE_COMMIT,
              "before": bundle["before"], "classification_counts": dict(Counter(row["classification"] for row in rows)),
              "classification_by_condition": {condition: dict(Counter(row["classification"] for row in rows
                  if row["condition"] == condition)) for condition in ("no_rag", "top1_rag")},
              "prompt_changed": False, "inference_rerun": False,
              "prompt_findings": "Official instruction precedes evidence; no final repetition. All question/options intact. "
                  "No metadata injection or silent truncation. Placement alone does not establish malformed causality.",
              "pilot_only": True}
    fields = list(rows[0]) if rows else ["sample_id", "condition", "classification", "raw_prediction"]
    artifacts = table_artifacts("malformed_cases", rows, fields)
    artifacts.update(table_artifacts("context_checks", context_rows, list(context_rows[0])))
    existing_path = output / "audit_meta.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if (existing["original_files_sha256"] != report["original_files_sha256"]
                or existing["sample_ids"] != report["sample_ids"]
                or existing["classification_counts"] != report["classification_counts"]):
            raise ValueError("Original audit inputs or classifications changed")
        if origin_root and existing["original_run_paths"] != report["original_run_paths"]:
            raise ValueError("Original run path provenance differs from the existing audit")
        # Preserve the pre-change metadata even when later metric schemas add fields.
        report = existing
    artifacts["audit_meta.json"] = json_text(report)
    save_artifacts(output, artifacts)
    return report
