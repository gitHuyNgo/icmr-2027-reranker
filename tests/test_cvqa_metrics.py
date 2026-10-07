import pytest

from icmr2027.evaluation.ravenea import build_cvqa_prompt, cvqa_correct, evaluate_cvqa, parse_cvqa_answer
from icmr2027.retrieval.corpus import evidence_text, retrieval_text


@pytest.mark.parametrize("raw,parsed", [
    ("Answer: A", "A"), ("Reasoning.\nAnswer: D)", "D"), ("B) text", "B"),
    (" C \n", "C"), ("Answer:A then Answer: B", "A"),
    ("Answer: APPLE", "A"),  # Preserve the official permissive prefix regex.
    ("A. text", None), ("answer: A", None), ("a", None),
    (" B) text", None), ("Answer: E", None), ("", None), ("?", None),
])
def test_official_case_sensitive_first_match_parsing(raw, parsed):
    assert parse_cvqa_answer(raw) == parsed


def test_correct_wrong_and_malformed_are_scored_on_all_samples():
    rows = [{"raw_prediction": "Answer: A", "ground_truth": "A", "country": "China"},
            {"raw_prediction": "Answer: B", "ground_truth": "A", "country": "China"},
            {"raw_prediction": "not an answer", "ground_truth": "D", "country": "Nigeria"}]
    result = evaluate_cvqa(rows)
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["num_malformed"] == 1
    assert result["num_samples"] == 3
    assert result["by_country"]["China"]["accuracy"] == 0.5
    assert cvqa_correct("A", "A) answer text")
    assert not cvqa_correct("B", "A")


def test_official_prompt_changes_only_evidence_insertion():
    question = "Question: Fixture?\n\nA. a\nB. b\nC. c\nD. d"
    no_rag = build_cvqa_prompt(question, None)
    rag = build_cvqa_prompt(question, "fixture evidence.")
    assert no_rag.endswith(question) and rag.endswith(question)
    assert "Answer: $LETTER" in no_rag
    assert "Document:\nfixture evidence." in rag
    assert "Document:" not in no_rag


def test_official_context_policy_uses_256_words_and_sentence_boundary():
    text = "# Title\n\n## Heading\nFirst sentence. Second sentence. " + " ".join(["word"] * 300)
    assert "Title" not in retrieval_text(text)
    assert evidence_text(text) == "First sentence. Second sentence."
    assert evidence_text("No sentence boundary") == "No sentence boundary."
