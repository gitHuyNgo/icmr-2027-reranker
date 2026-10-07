"""Normalized exact match; replace/extend with official scoring in later batches."""
from collections.abc import Iterable
import string
from typing import Any


def normalize_answer(answer: str, strip_punctuation: bool = True) -> str:
    normalized = " ".join(answer.strip().lower().split())
    if strip_punctuation:
        # Only strip boundary ASCII punctuation; preserve decimals and word interiors.
        normalized = normalized.strip(string.punctuation + " ")
    return normalized


def exact_match(prediction: str, answer: str, strip_punctuation: bool = True) -> bool:
    return normalize_answer(prediction, strip_punctuation) == normalize_answer(answer, strip_punctuation)


def evaluate_predictions(
    predictions: Iterable[dict[str, Any]], strip_punctuation: bool = True
) -> dict[str, int | float]:
    count = correct = 0
    for record in predictions:
        count += 1
        correct += exact_match(record["prediction"], record["ground_truth"], strip_punctuation)
    return {"num_samples": count, "accuracy": correct / count if count else 0.0}
