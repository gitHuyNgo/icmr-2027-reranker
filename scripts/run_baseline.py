#!/usr/bin/env python3
"""Shared Batch 1/2 runner: no-RAG or cached Top-1 evidence, one sample at a time."""
import argparse
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from icmr2027.config import load_config, resolve_path
from icmr2027.datasets.base import create_dataset
from icmr2027.evaluation.metrics import evaluate_predictions, exact_match
from icmr2027.evaluation.ravenea import evaluate_cvqa
from icmr2027.datasets.ravenea import format_cvqa_question
from icmr2027.experiments import prepare_cvqa_run, cvqa_context, cvqa_prediction
from icmr2027.models.vlm import HuggingFaceVLM
from icmr2027.utils.io import create_run_directory, read_jsonl, write_json, write_jsonl, write_yaml
from icmr2027.utils.reproducibility import set_seed
from icmr2027.utils.run_metadata import collect_run_metadata, package_version, utc_timestamp


SMOKE_NOTICE = "Infrastructure-only synthetic smoke data. Never use these numbers in the paper."


def run_baseline(config_path: Path, repo_root: Path = REPO_ROOT) -> Path:
    config = load_config(config_path)
    run_dir = create_run_directory(resolve_path(config.experiment.output_root, repo_root))
    print(f"Run directory: {run_dir}", flush=True)
    write_yaml(run_dir / "config.yaml", config.to_dict())
    metadata = collect_run_metadata(repo_root, config.experiment.seed, config.model.name, config.model.device)
    metadata.update({
        "experiment_name": config.experiment.name,
        "dataset_name": config.dataset.name,
        "dataset_split": config.dataset.split,
        "infrastructure_only": config.dataset.name == "smoke",
        "context": None if config.rag.mode == "none" else "cached_top1",
        "rag_mode": config.rag.mode,
        "status": "running",
        "requested_model_revision": config.model.revision,
        "package_versions": {name: package_version(name)
                             for name in ("accelerate", "pillow", "pyyaml", "numpy", "huggingface_hub")},
    })
    write_json(run_dir / "run_meta.json", metadata)
    write_jsonl(run_dir / "predictions.jsonl", [])
    write_json(run_dir / "metrics.json", {"status": "running", "num_samples": 0, "accuracy": None})
    completed = 0
    started = time.perf_counter()
    try:
        set_seed(config.experiment.seed)
        dataset = create_dataset(config.dataset, repo_root)
        if config.dataset.name == "smoke":
            print(SMOKE_NOTICE, flush=True)
        corpus, cache = None, {}
        if config.dataset.name == "ravenea_cvqa":
            details, corpus, cache = prepare_cvqa_run(config, dataset, repo_root)
            metadata.update(details)
            write_json(run_dir / "run_meta.json", metadata)
        model = HuggingFaceVLM(config.model)
        metadata["resolved_model_revision"] = getattr(model, "resolved_revision", None)
        if config.dataset.name == "ravenea_cvqa" and metadata["resolved_model_revision"] not in {None, config.model.revision}:
            raise ValueError("Loaded VLM revision differs from the pinned Batch 2 config")
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for sample in dataset:
                sample_started = time.perf_counter()
                context, evidence = None, {}
                question = sample["question"]
                if config.dataset.name == "ravenea_cvqa":
                    question = format_cvqa_question(sample)
                    context, evidence = cvqa_context(sample["id"], config.rag.mode, corpus, cache)
                prediction = model.generate(sample["image"], question, context=context)
                record = {
                    "id": sample["id"],
                    "question": sample["question"],
                    "ground_truth": sample["answer"],
                    "prediction": prediction,
                    "correct": exact_match(prediction, sample["answer"], config.evaluation.strip_punctuation),
                    "inference_seconds": time.perf_counter() - sample_started,
                    "infrastructure_only": config.dataset.name == "smoke",
                }
                if config.dataset.name == "ravenea_cvqa":
                    record = cvqa_prediction(sample, prediction, config.rag.mode, evidence)
                    record["inference_seconds"] = time.perf_counter() - sample_started
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                completed += 1
        # Re-read the saved artifacts to score exactly what was serialized.
        records = read_jsonl(run_dir / "predictions.jsonl")
        metrics = (evaluate_cvqa(records) if config.dataset.name == "ravenea_cvqa" else
                   evaluate_predictions(records, config.evaluation.strip_punctuation))
        metrics.update({"status": "completed", "metric": config.evaluation.metric,
                        "infrastructure_only": config.dataset.name == "smoke"})
        if config.dataset.name == "smoke":
            metrics["notice"] = SMOKE_NOTICE
        write_json(run_dir / "metrics.json", metrics)
        metadata["status"] = "completed"
        print(f"Samples: {completed}; accuracy: {metrics['accuracy']:.3f}", flush=True)
    except BaseException as exc:
        metadata.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(run_dir / "metrics.json", {
            "status": "failed", "num_samples": completed, "accuracy": None,
            "infrastructure_only": config.dataset.name == "smoke",
        })
        raise
    finally:
        metadata.update({"finished_at": utc_timestamp(), "num_predictions": completed,
                         "elapsed_seconds": time.perf_counter() - started})
        write_json(run_dir / "run_meta.json", metadata)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline/batch1_smoke.yaml")
    args = parser.parse_args()
    try:
        run_dir = run_baseline(resolve_path(args.config, REPO_ROOT))
    except Exception as exc:
        print(f"Baseline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Completed run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
