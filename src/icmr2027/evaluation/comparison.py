"""Paired pilot transitions with strict sample and experiment compatibility checks."""
import json
from pathlib import Path

from icmr2027.evaluation.ravenea import cvqa_correct
from icmr2027.evaluation.option_parser import answer_parser
from icmr2027.utils.io import create_run_directory, read_jsonl, write_json, write_jsonl


def pair_predictions(no_rag: list[dict], top1: list[dict], parser_version: str = "official") -> tuple[dict, list[dict]]:
    parse = answer_parser(parser_version)
    def index(records):
        result = {record["id"]: record for record in records}
        if len(result) != len(records) or not records:
            raise ValueError("Predictions must be nonempty with unique sample IDs")
        return result
    left, right = index(no_rag), index(top1)
    if set(left) != set(right):
        raise ValueError("No-RAG sample ID set != Top-1 RAG sample ID set")
    counts = {name: 0 for name in ("wrong_to_correct", "correct_to_wrong", "correct_to_correct", "wrong_to_wrong")}
    comparisons = []
    left_malformed = right_malformed = 0
    for identifier in sorted(left):
        a, b = left[identifier], right[identifier]
        for key in ("question", "options", "ground_truth", "file_name"):
            if a.get(key) != b.get(key):
                raise ValueError(f"Paired sample content differs: {identifier}:{key}")
        if a.get("rag_mode") != "none" or b.get("rag_mode") != "top1":
            raise ValueError("Comparison requires the expected none/top1 conditions")
        if not b.get("retrieved_doc_id") or b.get("retrieval_rank") != 1:
            raise ValueError("Top-1 prediction is missing retrieved document identity")
        for record in (a, b):
            if record.get("answer_parser_version", "official") != parser_version:
                raise ValueError("Prediction parser version differs from comparison scorer")
            if record.get("parsed_prediction") != parse(record["raw_prediction"]):
                raise ValueError("Stored parsed prediction does not match raw model output")
        a_ok = cvqa_correct(a["raw_prediction"], a["ground_truth"], parser_version)
        b_ok = cvqa_correct(b["raw_prediction"], b["ground_truth"], parser_version)
        if a.get("correct") != a_ok or b.get("correct") != b_ok:
            raise ValueError("Stored correctness does not match versioned scoring")
        transition = ("correct" if a_ok else "wrong") + "_to_" + ("correct" if b_ok else "wrong")
        counts[transition] += 1
        left_malformed += parse(a["raw_prediction"]) is None
        right_malformed += parse(b["raw_prediction"]) is None
        comparisons.append({"sample_id": identifier, "no_rag_correct": a_ok,
                            "top1_rag_correct": b_ok, "transition": transition,
                            "retrieved_doc_id": b["retrieved_doc_id"]})
    total = len(comparisons)
    left_accuracy = (counts["correct_to_wrong"] + counts["correct_to_correct"]) / total
    right_accuracy = (counts["wrong_to_correct"] + counts["correct_to_correct"]) / total
    summary = {"num_samples": total, "no_rag_accuracy": left_accuracy,
               "top1_rag_accuracy": right_accuracy, "delta_accuracy": right_accuracy - left_accuracy,
               "wrong_to_correct": counts["wrong_to_correct"], "correct_to_wrong": counts["correct_to_wrong"],
               "correct_to_correct": counts["correct_to_correct"], "wrong_to_wrong": counts["wrong_to_wrong"],
               "correction_rate": counts["wrong_to_correct"] / (counts["wrong_to_correct"] + counts["wrong_to_wrong"])
                   if counts["wrong_to_correct"] + counts["wrong_to_wrong"] else 0.0,
               "corruption_rate": counts["correct_to_wrong"] / (counts["correct_to_wrong"] + counts["correct_to_correct"])
                   if counts["correct_to_wrong"] + counts["correct_to_correct"] else 0.0,
               "answer_parser_version": parser_version,
               "no_rag_malformed_rate": left_malformed / total, "top1_rag_malformed_rate": right_malformed / total,
               "unchanged_correct": counts["correct_to_correct"], "unchanged_wrong": counts["wrong_to_wrong"],
               "no_rag_num_malformed": left_malformed, "top1_rag_num_malformed": right_malformed,
               "pilot_only": True, "notice": "Batch 2 pilot results are not final paper results."}
    return summary, comparisons


def compare_runs(no_rag_run: Path, top1_run: Path, output_root: Path) -> Path:
    metas = [json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
             for run in (no_rag_run, top1_run)]
    if any(meta.get("status") != "completed" for meta in metas):
        raise ValueError("Both inference runs must be completed before comparison")
    if not metas[0].get("comparison_identity") or metas[0]["comparison_identity"] != metas[1].get("comparison_identity"):
        raise ValueError("Paired model/data/prompt/generation/seed settings differ")
    if metas[0].get("resolved_model_revision") != metas[1].get("resolved_model_revision"):
        raise ValueError("Paired VLM resolved revisions differ")
    records = [read_jsonl(run / "predictions.jsonl") for run in (no_rag_run, top1_run)]
    for rows, meta in zip(records, metas):
        if set(row["id"] for row in rows) != set(meta["sample_ids"]):
            raise ValueError("Prediction IDs do not match the persisted pilot IDs")
    versions = [meta.get("answer_parser_version", "official") for meta in metas]
    if versions[0] != versions[1]:
        raise ValueError("Paired answer parser versions differ")
    summary, comparisons = pair_predictions(*records, parser_version=versions[0])
    output = create_run_directory(output_root)
    write_json(output / "summary.json", summary)
    write_jsonl(output / "comparisons.jsonl", comparisons)
    write_json(output / "pair_meta.json", {"no_rag_run": str(no_rag_run), "top1_rag_run": str(top1_run),
                                           "comparison_identity": metas[0]["comparison_identity"]})
    print(json.dumps(summary, indent=2))
    print(f"Comparison directory: {output}")
    return output


def latest_completed(root: Path) -> Path:
    if root.is_dir():
        for run in sorted(root.iterdir(), reverse=True):
            meta = run / "run_meta.json"
            if meta.is_file() and json.loads(meta.read_text(encoding="utf-8")).get("status") == "completed":
                return run
    raise FileNotFoundError(f"No completed runs found in {root}")
