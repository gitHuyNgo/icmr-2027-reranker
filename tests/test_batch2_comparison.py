from copy import deepcopy

import pytest

from icmr2027.evaluation.comparison import pair_predictions
from icmr2027.evaluation.ravenea import cvqa_correct, parse_cvqa_answer


def record(identifier, raw, mode):
    result = {"id": identifier, "question": "Fixture?", "options": ["A", "B", "C", "D"],
              "file_name": identifier + ".png", "ground_truth": "A", "raw_prediction": raw,
              "parsed_prediction": parse_cvqa_answer(raw), "correct": cvqa_correct(raw, "A"),
              "rag_mode": mode}
    if mode == "top1": result.update(retrieved_doc_id="enwiki/1", retrieval_rank=1)
    return result


def test_all_pairwise_transitions_and_malformed_counts():
    left = [record(str(i), raw, "none") for i, raw in enumerate(["B", "A", "A", "invalid"])]
    right = [record(str(i), raw, "top1") for i, raw in enumerate(["A", "B", "A", "B"])]
    summary, rows = pair_predictions(left, list(reversed(right)))
    assert summary["num_samples"] == 4
    assert summary["no_rag_accuracy"] == summary["top1_rag_accuracy"] == 0.5
    for key in ("wrong_to_correct", "correct_to_wrong", "unchanged_correct", "unchanged_wrong"):
        assert summary[key] == 1
    assert summary["no_rag_num_malformed"] == 1
    assert {x["transition"] for x in rows} == {"wrong_to_correct", "correct_to_wrong", "correct_to_correct", "wrong_to_wrong"}


def test_mismatched_sample_ids_fail():
    with pytest.raises(ValueError, match="sample ID set"):
        pair_predictions([record("one", "A", "none")], [record("two", "A", "top1")])


def test_changed_question_or_tampered_correctness_fail():
    left, right = [record("one", "A", "none")], [record("one", "A", "top1")]
    right[0]["correct"] = False
    with pytest.raises(ValueError, match="correctness"):
        pair_predictions(left, right)
    right = [record("one", "A", "top1")]
    right[0]["question"] = "Changed question"
    with pytest.raises(ValueError, match="content differs"):
        pair_predictions(left, right)
