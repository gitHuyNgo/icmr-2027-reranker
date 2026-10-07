import pytest

from icmr2027.evaluation.metrics import normalize_answer, exact_match, evaluate_predictions


def test_normalization():
    assert normalize_answer("  RED \n   square.  ") == "red square"
    assert exact_match(" Blue! ", "blue")
    assert not exact_match("blue!", "blue", strip_punctuation=False)
    assert normalize_answer("3.14") == "3.14"
    assert not exact_match("blue", "red")


def test_accuracy_recomputes_correctness():
    result = evaluate_predictions([
        {"prediction": "Red.", "ground_truth": "red", "correct": False},
        {"prediction": "green", "ground_truth": "blue"},
        {"prediction": " BLUE  ", "ground_truth": "blue"},
    ])
    assert result["num_samples"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_empty_predictions():
    assert evaluate_predictions([]) == {"num_samples": 0, "accuracy": 0.0}
