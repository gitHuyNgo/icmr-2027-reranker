"""cVQA parsing equivalent to official helper.is_match for released A-D truths."""
import re
from collections import defaultdict


CVQA_INSTRUCTION = (
    "Answer the following multiple choice question for the image. The last line must be "
    "of the following format: 'Answer: $LETTER' (without quotes) where LETTER must be "
    "one of A, B, C, or D."
)


def build_cvqa_prompt(formatted_question: str, context: str | None) -> str:
    evidence = ""
    if context is not None:
        evidence = (
            " Use the shared culture between the image and the following document to answer "
            "the question.\n\nDocument:\n" + context
        )
    return CVQA_INSTRUCTION + evidence + "\n\n" + formatted_question


def parse_cvqa_answer(raw: str) -> str | None:
    # Official semantics: case-sensitive, first Answer match anywhere; no final-line check.
    match = re.search(r"Answer:\s*([A-D])", raw)
    if match:
        return match.group(1)
    match = re.search(r"^([A-D])\s*\)\s*(.*)", raw)
    if match:
        return match.group(1)
    candidate = raw.strip()
    return candidate if len(candidate) == 1 and candidate in "ABCD" else None


def cvqa_correct(raw: str, ground_truth: str, parser_version: str = "official") -> bool:
    # Released answers are A-D. Preserve the official ground-truth fallback semantics.
    match = re.search(r"^([A-D])(?:\)|\s|$)", ground_truth)
    label = match.group(1) if match else ground_truth.strip()
    from icmr2027.evaluation.option_parser import answer_parser
    return answer_parser(parser_version)(raw) == label


def evaluate_cvqa(records: list[dict], parser_version: str = "official") -> dict:
    from icmr2027.evaluation.option_parser import answer_parser
    parse = answer_parser(parser_version)
    correct = malformed = 0
    countries = defaultdict(lambda: {"correct": 0, "num_samples": 0})
    for record in records:
        raw = record["raw_prediction"]
        ok = cvqa_correct(raw, record["ground_truth"], parser_version)
        correct += ok
        malformed += parse(raw) is None
        if "country" in record:
            countries[record["country"]]["correct"] += ok
            countries[record["country"]]["num_samples"] += 1
    for stats in countries.values():
        stats["accuracy"] = stats["correct"] / stats["num_samples"]
    return {"num_samples": len(records), "accuracy": correct / len(records) if records else 0.0,
            "num_malformed": malformed, "malformed_rate": malformed / len(records) if records else 0.0,
            "answer_parser_version": parser_version, "dataset": "ravenea_cvqa", "metric": "ravenea_cvqa_accuracy",
            "by_country": dict(countries), "pilot_only": True,
            "notice": "Batch 2 pilot results are not final paper results."}
