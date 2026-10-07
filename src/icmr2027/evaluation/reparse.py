"""Re-evaluate historical raw generations only after a locked malformed-case audit."""
from copy import deepcopy
import json
from pathlib import Path

from icmr2027.datasets.ravenea import file_sha256
from icmr2027.evaluation.comparison import pair_predictions
from icmr2027.evaluation.format_audit import json_text, save_artifacts, table_artifacts
from icmr2027.evaluation.option_parser import CONSERVATIVE, parse_conservative_answer
from icmr2027.evaluation.ravenea import cvqa_correct, evaluate_cvqa
from icmr2027.utils.io import read_jsonl


def reparse_pair(bundle: dict, output: Path) -> dict:
    audit_path = output / "audit/audit_meta.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["original_files_sha256"] != bundle["hashes"] or audit["sample_ids"] != bundle["ids"]:
        raise ValueError("Original files or pilot changed after the required pre-change audit")
    cases = read_jsonl(output / "audit/malformed_cases.jsonl")
    malformed = {(condition, row["id"]): row["raw_prediction"]
                 for condition, rows in zip(("no_rag", "top1_rag"), bundle["rows"])
                 for row in rows if row["parsed_prediction"] is None}
    if len(cases) != len(malformed) or {(row["condition"], row["sample_id"]): row["raw_prediction"]
                                      for row in cases} != malformed:
        raise ValueError("Audit must contain every original malformed raw prediction exactly once")
    counts = audit["classification_counts"]
    if not counts.get("parser_failure", 0) and not counts.get("verbose_but_identifiable", 0):
        raise ValueError("Audit does not establish a parser-only correction")
    if any(row["classification"] not in {"parser_failure", "verbose_but_identifiable",
                                         "ambiguous_multiple_options", "no_option_identifiable"} for row in cases):
        raise ValueError("Manual prompt/context review required before parser-only re-evaluation")
    new_rows = deepcopy(bundle["rows"])
    for rows in new_rows:
        for row in rows:
            row["parsed_prediction"] = parse_conservative_answer(row["raw_prediction"])
            row["correct"] = cvqa_correct(row["raw_prediction"], row["ground_truth"], CONSERVATIVE)
            row["answer_parser_version"] = CONSERVATIVE
    before, _ = pair_predictions(*bundle["rows"])
    after, paired = pair_predictions(*new_rows, parser_version=CONSERVATIVE)
    def with_aliases(summary):
        return {**summary, "no_rag_malformed": summary["no_rag_num_malformed"],
                "top1_rag_malformed": summary["top1_rag_num_malformed"]}
    report = {"num_samples": len(bundle["ids"]), "before": with_aliases(before), "after": with_aliases(after),
              "original_run_paths": audit["original_run_paths"], "classification_counts": counts,
              "sample_ids_sha256": audit["sample_ids_sha256"], "same_sample_ids": True,
              "original_files_sha256": bundle["hashes"], "retrieval_cache_unchanged": True,
              "retrieved_document_ids_scores_unchanged": True, "raw_predictions_unchanged": True,
              "prompt_changed": False, "inference_rerun": False, "after_parser": CONSERVATIVE,
              "official_parser_preserved": True, "benchmark_note": "After scores use explicit format normalization; "
                  "they differ from the official regex implementation. Before retains exact official scoring.",
              "audit_meta_sha256": file_sha256(audit_path), "pilot_only": True,
              "notice": "Evaluation-validity pilot only; not paper results or hypothesis interpretation."}
    transitions = []
    for old_left, old_right, new_left, new_right in zip(*bundle["rows"], *new_rows):
        row = {"sample_id": old_left["id"], "ground_truth": old_left["ground_truth"],
               "retrieved_doc_id": old_right["retrieved_doc_id"], "retrieval_score": old_right["retrieval_score"],
               "context_chars": old_right["context_chars"]}
        for prefix, prediction in (("old_no_rag", old_left), ("old_rag", old_right),
                                   ("new_no_rag", new_left), ("new_rag", new_right)):
            for suffix, key in (("raw", "raw_prediction"), ("parsed", "parsed_prediction"), ("correct", "correct")):
                row[prefix + "_" + suffix] = prediction[key]
        transitions.append(row)
    artifacts = {"comparison/before_after.json": json_text(report), "reparse/comparison.json": json_text(after)}
    artifacts.update(table_artifacts("comparison/per_sample_transitions", transitions, list(transitions[0])))
    artifacts.update(table_artifacts("reparse/paired_transitions", paired, list(paired[0])))
    for condition, rows, original_run in zip(("no_rag", "top1_rag"), new_rows, bundle["runs"]):
        metrics = {**evaluate_cvqa(rows, CONSERVATIVE), "status": "completed", "evaluation_only": True}
        artifacts[f"reparse/{condition}_metrics.json"] = json_text(metrics)
        artifacts[f"reparse/{condition}/metrics.json"] = json_text(metrics)
        artifacts.update(table_artifacts(f"reparse/{condition}/predictions", rows, list(rows[0])))
        meta = json.loads((original_run / "run_meta.json").read_text(encoding="utf-8"))
        meta.update(answer_parser_version=CONSERVATIVE, evaluation_only=True, inference_rerun=False,
                    original_run=audit["original_run_paths"][condition],
                    original_predictions_sha256=file_sha256(original_run / "predictions.jsonl"))
        meta["comparison_identity"]["answer_parser_version"] = CONSERVATIVE
        artifacts[f"reparse/{condition}/run_meta.json"] = json_text(meta)
        artifacts[f"reparse/{condition}/config.yaml"] = (original_run / "config.yaml").read_text(encoding="utf-8")
    # Verify source integrity again before writing any evaluation products.
    for name, digest in bundle["hashes"].items():
        if file_sha256(bundle["root"] / name) != digest:
            raise ValueError("Original inputs changed during re-evaluation")
    save_artifacts(output, artifacts)
    return report
