import pytest

from icmr2027.evaluation.option_parser import CONSERVATIVE, answer_parser, parse_conservative_answer
from icmr2027.evaluation.ravenea import evaluate_cvqa, parse_cvqa_answer
from icmr2027.evaluation.format_audit import diagnose


@pytest.mark.parametrize("raw", ["A", "a", "(A)", "[A]", "Answer: A", "Answer: (A)",
    "The answer is A.", "Option A", "A.", "A. labelled option text", "A) labelled option text",
    "Answer: [a]", "The correct answer is A.", "Reasoning ends here.\nAnswer: A",
    "A. A labelled response", "Answer: A because a custom explains it."])
def test_unambiguous_forms(raw):
    assert parse_conservative_answer(raw) == "A"


@pytest.mark.parametrize("label", "ABCDabcd")
def test_all_labels_without_ground_truth_or_option_text(label):
    for raw in (label, f"({label})", f"[{label}]", f"Answer: {label}", f"Option {label}",
                f"{label}. text", f"The answer is {label}."):
        assert parse_conservative_answer(raw) == label.upper()


@pytest.mark.parametrize("raw", ["A or B", "Maybe A, perhaps B", "I cannot determine",
    "Both B and C could be correct", "", "APPLE", "CAB", "ABCD", "Answer: APPLE",
    "Option BEAR", "The answer is CLOUD.", "Answer: A or b", "Answer: A\nAnswer: B",
    "Answer: A. (B) is another candidate", "not A", "Perhaps Answer: A",
    "Answer: A, but not A", "The capital is Baghdad.", "A building in a city", "I choose B",
    "Answer: A? Maybe.", "Answer: (A) or [b]", "Answer: [a] or [b]", "Option A2"])
def test_ambiguous_uncertain_and_word_fragments_stay_malformed(raw):
    assert parse_conservative_answer(raw) is None


def test_official_scoring_is_preserved_and_versioned_reparse_can_reduce_scores():
    assert parse_cvqa_answer("Answer: APPLE") == "A"
    assert parse_cvqa_answer("Answer: A then Answer: B") == "A"
    rows = [{"raw_prediction": "Answer: A then Answer: B", "ground_truth": "A"}]
    assert evaluate_cvqa(rows)["accuracy"] == 1
    strict = evaluate_cvqa(rows, CONSERVATIVE)
    assert strict["accuracy"] == 0 and strict["num_malformed"] == 1
    assert strict["malformed_rate"] == 1
    assert strict["answer_parser_version"] == CONSERVATIVE
    assert evaluate_cvqa([])["malformed_rate"] == 0
    with pytest.raises(ValueError, match="Unknown"):
        answer_parser("unregistered")


@pytest.mark.parametrize("raw,category", [("B. explicit option", "parser_failure"),
    ("The answer is B.", "parser_failure"),
    ("This explains the image. The answer is B.", "verbose_but_identifiable"),
    ("A or B", "ambiguous_multiple_options"), ("cannot determine", "no_option_identifiable"),
    ("The document was truncated", "needs_manual_review"), ("I choose B", "needs_manual_review")])
def test_diagnostics_do_not_see_ground_truth(raw, category):
    assert diagnose(raw)[0] == category
